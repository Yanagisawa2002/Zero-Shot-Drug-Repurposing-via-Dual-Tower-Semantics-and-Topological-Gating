from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path
from typing import Dict, Mapping

import pandas as pd
import torch
from torch import Tensor
from torch_geometric.data import HeteroData

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.train_split_probe import derive_ho_triplets_for_pair_splits, load_pair_splits
from src.evaluation_utils import (
    _build_known_positive_paths,
    _compute_pairwise_ranking_metrics,
    _extract_x_dict,
    _infer_model_device,
    _move_batch_to_device,
)
from src.feature_utils import inject_features_to_graph
from src.graph_surgery import (
    build_split_isolation_targets,
    collect_holdout_pairs_from_pair_splits,
    remove_direct_leakage_edges,
    remove_leakage_edges,
)
from src.pair_path_bpr_sampler import build_pair_path_bpr_dataloader
from src.primekg_data_processor import PrimeKGDataProcessor
from src.repurposing_rgcn import EmbeddingDict, EdgeType, RepurposingRGCN
from src.training_utils import train_epoch


ResultDict = Dict[str, Dict[str, float]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Run isolated no-leakage quad-mode split probe with optional OT external evaluation.'
    )
    parser.add_argument('--processed-path', type=Path, required=True)
    parser.add_argument('--output-json', type=Path, required=True)
    parser.add_argument('--nodes-csv', type=Path, default=Path('data/PrimeKG/nodes.csv'))
    parser.add_argument('--edges-csv', type=Path, default=Path('data/PrimeKG/edges.csv'))
    parser.add_argument('--feature-dir', type=Path, default=Path('outputs/pubmedbert_hybrid_features'))
    parser.add_argument('--ot-novel-csv', type=Path, default=Path('outputs/ot_random_external_profile/novel_ood_triplets.csv'))
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--hidden-channels', type=int, default=32)
    parser.add_argument('--out-dim', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=1e-5)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument(
        '--graph-surgery-mode',
        type=str,
        default='strict',
        choices=['strict', 'direct_only'],
        help='strict: old full mechanism-removal surgery; direct_only: remove only held-out drug-disease shortcut edges.',
    )
    return parser.parse_args()


def evaluate_model_quad(
    model: RepurposingRGCN,
    data: HeteroData,
    valid_ho_paths: Tensor,
    batch_size: int = 32,
    ho_attr_name: str = 'ho_pos_paths',
) -> ResultDict:
    if valid_ho_paths.dim() != 2 or valid_ho_paths.size(1) != 3:
        raise ValueError('`valid_ho_paths` ??? `(N, 3)` ???????')

    valid_ho_paths = valid_ho_paths.detach().cpu().to(torch.long).contiguous()
    known_positive_paths = _build_known_positive_paths(
        data=data,
        valid_ho_paths=valid_ho_paths,
        ho_attr_name=ho_attr_name,
    )

    dataloaders = {}
    for negative_strategy in ('random', 'cross_drug', 'cross_disease'):
        dataloaders[negative_strategy] = build_pair_path_bpr_dataloader(
            data=data,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            ho_attr_name=ho_attr_name,
            positive_paths=valid_ho_paths,
            known_positive_pairs=known_positive_paths,
            negative_strategy=negative_strategy,
            use_pathway_quads=True,
        )

    was_training = model.training
    model.eval()
    device = _infer_model_device(model)
    graph_data = copy.deepcopy(data).to(device)

    try:
        with torch.no_grad():
            x_dict = _extract_x_dict(full_graph_data=graph_data)
            edge_index_dict: Mapping[EdgeType, Tensor] = graph_data.edge_index_dict
            node_embs_dict: EmbeddingDict = model.encode(
                x_dict=x_dict,
                edge_index_dict=edge_index_dict,
            )

            results: ResultDict = {}
            for setting_name, dataloader in dataloaders.items():
                pos_score_chunks = []
                neg_score_chunks = []
                for batch in dataloader:
                    tensor_batch = _move_batch_to_device(batch=batch, device=device)
                    pos_scores = model.score_batch(
                        node_embs_dict=node_embs_dict,
                        pair_ids=tensor_batch['pos_pair_ids'],
                        paths=tensor_batch['pos_paths'],
                        attention_mask=tensor_batch['pos_attention_mask'],
                    )
                    neg_scores = model.score_batch(
                        node_embs_dict=node_embs_dict,
                        pair_ids=tensor_batch['neg_pair_ids'],
                        paths=tensor_batch['neg_paths'],
                        attention_mask=tensor_batch['neg_attention_mask'],
                    )
                    pos_score_chunks.append(pos_scores.detach().cpu())
                    neg_score_chunks.append(neg_scores.detach().cpu())
                results[setting_name] = _compute_pairwise_ranking_metrics(
                    pos_scores=torch.cat(pos_score_chunks, dim=0),
                    neg_scores=torch.cat(neg_score_chunks, dim=0),
                )
    finally:
        if was_training:
            model.train()

    return results


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device:', device)

    split_mode, pair_splits = load_pair_splits(args.processed_path)
    processor = PrimeKGDataProcessor(node_csv_path=args.nodes_csv, edge_csv_path=args.edges_csv)
    processor.build_entity_mappings()
    split_triplets = derive_ho_triplets_for_pair_splits(
        processor=processor,
        edge_csv_path=args.edges_csv,
        pair_splits=pair_splits,
    )

    ot_df = pd.read_csv(args.ot_novel_csv)
    global_entity2id = processor.global_entity2id
    ot_triplets = []
    for row in ot_df.itertuples(index=False):
        drug_raw = getattr(row, 'primekg_drug_id')
        gene_raw = getattr(row, 'primekg_target_id')
        disease_raw = getattr(row, 'primekg_disease_id')
        if drug_raw not in global_entity2id or gene_raw not in global_entity2id or disease_raw not in global_entity2id:
            continue
        ot_triplets.append((global_entity2id[drug_raw], global_entity2id[gene_raw], global_entity2id[disease_raw]))
    ot_triplets = list(dict.fromkeys(ot_triplets))

    data = processor.build_heterodata(ho_id_paths=split_triplets['train'], add_inverse_edges=False)
    inject_features_to_graph(data=data, feature_dir=args.feature_dir)

    heldout_triplets = split_triplets['valid'] + split_triplets['test'] + ot_triplets
    if args.graph_surgery_mode == 'direct_only':
        holdout_pairs = collect_holdout_pairs_from_pair_splits(pair_splits=pair_splits)
        clean_data = remove_direct_leakage_edges(
            data=data,
            holdout_pairs=holdout_pairs,
        )
    elif args.graph_surgery_mode == 'strict':
        isolate_nodes_by_type = build_split_isolation_targets(split_mode=split_mode, pair_splits=pair_splits)
        clean_edge_index_dict = remove_leakage_edges(
            data=data,
            target_paths=torch.tensor(heldout_triplets, dtype=torch.long),
            isolate_nodes_by_type=isolate_nodes_by_type,
        )
        clean_data = copy.deepcopy(data)
        for edge_type, clean_edge_index in clean_edge_index_dict.items():
            clean_data[edge_type].edge_index = clean_edge_index
    else:
        raise ValueError(f'Unsupported graph_surgery_mode: {args.graph_surgery_mode}')

    total_removed_edges = 0
    for edge_type in data.edge_index_dict.keys():
        original_edge_index = data[edge_type].edge_index
        clean_edge_index = clean_data[edge_type].edge_index
        total_removed_edges += int(original_edge_index.size(1) - clean_edge_index.size(1))

    train_loader = build_pair_path_bpr_dataloader(
        data=clean_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        negative_strategy='mixed',
        use_pathway_quads=True,
    )

    model = RepurposingRGCN(
        data=clean_data,
        in_channels=768,
        hidden_channels=args.hidden_channels,
        out_dim=args.out_dim,
        scorer_hidden_dim=args.out_dim,
        dropout=args.dropout,
        use_pathway_quads=True,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    history = []
    start_time = time.time()
    for epoch in range(1, args.epochs + 1):
        train_metrics = train_epoch(
            model=model,
            full_graph_data=clean_data,
            bpr_dataloader=train_loader,
            optimizer=optimizer,
        )
        item = {'epoch': epoch, 'train': train_metrics}

        if epoch in {1, 10, 20, 30, 40}:
            valid_tensor = torch.tensor(split_triplets['valid'], dtype=torch.long)
            test_tensor = torch.tensor(split_triplets['test'], dtype=torch.long)
            ot_tensor = torch.tensor(ot_triplets, dtype=torch.long)
            valid_eval = evaluate_model_quad(model=model, data=clean_data, valid_ho_paths=valid_tensor, batch_size=args.batch_size)
            test_eval = evaluate_model_quad(model=model, data=clean_data, valid_ho_paths=test_tensor, batch_size=args.batch_size)
            ot_eval = evaluate_model_quad(model=model, data=clean_data, valid_ho_paths=ot_tensor, batch_size=args.batch_size)
            item['valid_eval'] = valid_eval
            item['test_eval'] = test_eval
            item['ot_eval'] = ot_eval
            print(
                f"epoch={epoch:02d} "
                f"valid_random_auc={valid_eval['random']['auroc']:.4f} "
                f"test_random_auc={test_eval['random']['auroc']:.4f} "
                f"test_cross_drug_auc={test_eval['cross_drug']['auroc']:.4f} "
                f"test_cross_disease_auc={test_eval['cross_disease']['auroc']:.4f} "
                f"ot_random_auc={ot_eval['random']['auroc']:.4f} "
                f"ot_cross_drug_auc={ot_eval['cross_drug']['auroc']:.4f} "
                f"ot_cross_disease_auc={ot_eval['cross_disease']['auroc']:.4f}"
            )
        history.append(item)

    payload = {
        'config': {
            'split_mode': split_mode,
            'processed_path': str(args.processed_path),
            'feature_dir': str(args.feature_dir),
            'ot_novel_csv': str(args.ot_novel_csv),
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'hidden_channels': args.hidden_channels,
            'out_dim': args.out_dim,
            'dropout': args.dropout,
            'quad_mode': True,
            'use_pathway_quads': True,
            'graph_surgery_mode': args.graph_surgery_mode,
        },
        'triplet_summary': {
            split_name: {
                'num_triplets': len(paths),
                'num_pairs': len({(d, c) for d, _, c in paths}),
            }
            for split_name, paths in split_triplets.items()
        },
        'ot_novel_triplets': len(ot_triplets),
        'ot_novel_pairs': len({(d, c) for d, _, c in ot_triplets}),
        'total_removed_leakage_edges': total_removed_edges,
        'total_time_sec': time.time() - start_time,
        'history': history,
    }
    args.output_json.write_text(json.dumps(payload, indent=2), encoding='utf-8')
    print('Saved metrics to:', args.output_json)


if __name__ == '__main__':
    main()
