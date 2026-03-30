from __future__ import annotations

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

from src.feature_utils import inject_features_to_graph
from src.graph_surgery import remove_leakage_edges
from src.pair_path_bpr_sampler import build_pair_path_bpr_dataloader
from src.primekg_data_processor import PrimeKGDataProcessor
from src.repurposing_rgcn import EmbeddingDict, EdgeType, RepurposingRGCN
from src.training_utils import train_epoch
from scripts.train_split_probe import load_pair_splits, derive_ho_triplets_for_pair_splits
from src.evaluation_utils import _build_known_positive_paths, _compute_pairwise_ranking_metrics, _extract_x_dict, _infer_model_device, _move_batch_to_device


ResultDict = Dict[str, Dict[str, float]]


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
    processed_path = Path('data/PrimeKG/processed/primekg_indication_mvp.pt')
    nodes_csv = Path('data/PrimeKG/nodes.csv')
    edges_csv = Path('data/PrimeKG/edges.csv')
    feature_dir = Path('outputs/pubmedbert_hybrid_features')
    ot_novel_csv = Path('outputs/ot_random_external_profile/novel_ood_triplets.csv')
    output_json = Path('outputs/random_split_probe_quad_pathway_noleak_pubmedbert_hybrid_40epoch.json')

    torch.manual_seed(0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device:', device)

    split_mode, pair_splits = load_pair_splits(processed_path)
    processor = PrimeKGDataProcessor(node_csv_path=nodes_csv, edge_csv_path=edges_csv)
    processor.build_entity_mappings()
    split_triplets = derive_ho_triplets_for_pair_splits(
        processor=processor,
        edge_csv_path=edges_csv,
        pair_splits=pair_splits,
    )

    ot_df = pd.read_csv(ot_novel_csv)
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
    inject_features_to_graph(data=data, feature_dir=feature_dir)

    heldout_triplets = split_triplets['valid'] + split_triplets['test'] + ot_triplets
    clean_edge_index_dict = remove_leakage_edges(
        data=data,
        target_paths=torch.tensor(heldout_triplets, dtype=torch.long),
    )
    clean_data = copy.deepcopy(data)
    total_removed_edges = 0
    for edge_type, clean_edge_index in clean_edge_index_dict.items():
        original_edge_index = data[edge_type].edge_index
        clean_data[edge_type].edge_index = clean_edge_index
        total_removed_edges += int(original_edge_index.size(1) - clean_edge_index.size(1))

    train_loader = build_pair_path_bpr_dataloader(
        data=clean_data,
        batch_size=512,
        shuffle=True,
        num_workers=0,
        negative_strategy='mixed',
        use_pathway_quads=True,
    )

    model = RepurposingRGCN(
        data=clean_data,
        in_channels=768,
        hidden_channels=32,
        out_dim=32,
        scorer_hidden_dim=32,
        dropout=0.1,
        use_pathway_quads=True,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

    history = []
    start_time = time.time()
    for epoch in range(1, 41):
        train_metrics = train_epoch(
            model=model,
            full_graph_data=clean_data,
            bpr_dataloader=train_loader,
            optimizer=optimizer,
        )
        item = {
            'epoch': epoch,
            'train': train_metrics,
        }

        if epoch in {1, 10, 20, 30, 40}:
            test_tensor = torch.tensor(split_triplets['test'], dtype=torch.long)
            ot_tensor = torch.tensor(ot_triplets, dtype=torch.long)
            test_eval = evaluate_model_quad(
                model=model,
                data=clean_data,
                valid_ho_paths=test_tensor,
                batch_size=512,
            )
            ot_eval = evaluate_model_quad(
                model=model,
                data=clean_data,
                valid_ho_paths=ot_tensor,
                batch_size=512,
            )
            item['test_eval'] = test_eval
            item['ot_eval'] = ot_eval
            print(
                f"epoch={epoch:02d} "
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
            'processed_path': str(processed_path),
            'feature_dir': str(feature_dir),
            'ot_novel_csv': str(ot_novel_csv),
            'epochs': 40,
            'batch_size': 512,
            'hidden_channels': 32,
            'out_dim': 32,
            'dropout': 0.1,
            'quad_mode': True,
            'use_pathway_quads': True,
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
    output_json.write_text(json.dumps(payload, indent=2), encoding='utf-8')
    print('Saved metrics to:', output_json)


if __name__ == '__main__':
    main()
