from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path
from typing import Dict, Mapping, Optional

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


def _format_entity_probe(entity_lookup: Optional[Mapping[int, object]], global_id: int) -> str:
    if entity_lookup is None:
        return f'id={global_id}'
    record = entity_lookup.get(int(global_id))
    if record is None:
        return f'id={global_id}'
    name = getattr(record, 'name', None) or 'UNKNOWN'
    raw_id = getattr(record, 'raw_id', None) or 'UNKNOWN'
    node_type = getattr(record, 'node_type', None) or 'UNKNOWN'
    return f'name={name} | raw_id={raw_id} | node_type={node_type}'


def _emit_negative_tail_name_probe(
    dataset,
    setting_name: str,
    entity_lookup: Optional[Mapping[int, object]],
    num_preview: int = 3,
) -> None:
    if len(dataset) == 0:
        return
    valid_tail_types = {'disease', 'phenotype', 'disease/phenotype'}
    print(f"--- [PROBE] First {min(num_preview, len(dataset))} negative tails | Setting: {setting_name} ---")
    for sample_index in range(min(num_preview, len(dataset))):
        sample = dataset[sample_index]
        neg_tail_id = int(sample['neg_pair_ids'][1].item())
        print(f"--- [PROBE] neg[{sample_index}] tail_global_id={neg_tail_id} | {_format_entity_probe(entity_lookup, neg_tail_id)} ---")
        if entity_lookup is not None:
            record = entity_lookup.get(neg_tail_id)
            if record is not None:
                node_type = str(getattr(record, 'node_type', '')).strip().lower()
                if node_type not in valid_tail_types:
                    raise AssertionError(
                        f'FATAL: negative tail is not a disease-like node: global_id={neg_tail_id}, node_type={node_type}'
                    )


def _emit_empty_path_ratio_probe(
    *,
    pos_attention_mask: Tensor,
    neg_attention_mask: Tensor,
    setting_name: str,
    batch_index: int,
) -> None:
    pos_has_any_path = pos_attention_mask.to(dtype=torch.bool).any(dim=1)
    neg_has_any_path = neg_attention_mask.to(dtype=torch.bool).any(dim=1)

    pos_total = int(pos_has_any_path.numel())
    neg_total = int(neg_has_any_path.numel())
    pos_empty = int((~pos_has_any_path).sum().item())
    neg_empty = int((~neg_has_any_path).sum().item())

    pos_ratio = (100.0 * pos_empty / pos_total) if pos_total > 0 else 0.0
    neg_ratio = (100.0 * neg_empty / neg_total) if neg_total > 0 else 0.0
    print(
        f"[PROBE] Setting={setting_name} Batch={batch_index:03d} | "
        f"POS Empty Path Ratio: {pos_empty}/{pos_total} ({pos_ratio:.1f}%) | "
        f"NEG Empty Path Ratio: {neg_empty}/{neg_total} ({neg_ratio:.1f}%)"
    )


def _emit_eval_disease_degree_probe(
    *,
    valid_ho_paths: Tensor,
    edge_index_dict: Mapping[EdgeType, Tensor],
    disease_global_ids: Tensor,
    probe_name: str,
) -> None:
    eval_disease_global_ids = sorted({int(path[-1]) for path in valid_ho_paths.detach().cpu().tolist()})
    if not eval_disease_global_ids:
        print(f"[PROBE] {probe_name} Unseen Diseases Avg Degree in GNN: 0.0 edges per disease | Non-drug avg degree: 0.0 | num_diseases=0")
        return

    global_to_local = {
        int(global_id): local_idx
        for local_idx, global_id in enumerate(disease_global_ids.detach().cpu().tolist())
    }
    eval_disease_local_ids = [
        global_to_local[global_id]
        for global_id in eval_disease_global_ids
        if global_id in global_to_local
    ]
    if not eval_disease_local_ids:
        print(f"[PROBE] {probe_name} Unseen Diseases Avg Degree in GNN: 0.0 edges per disease | Non-drug avg degree: 0.0 | num_diseases=0")
        return

    eval_disease_local_tensor = torch.tensor(eval_disease_local_ids, dtype=torch.long)
    total_degree = 0
    non_drug_degree = 0

    for edge_type, edge_index in edge_index_dict.items():
        count = 0
        cpu_edge_index = edge_index.detach().cpu()
        if edge_type[2] == 'disease':
            count += int(torch.isin(cpu_edge_index[1], eval_disease_local_tensor).sum().item())
        if edge_type[0] == 'disease':
            count += int(torch.isin(cpu_edge_index[0], eval_disease_local_tensor).sum().item())
        total_degree += count
        if count > 0 and 'drug' not in edge_type:
            non_drug_degree += count

    num_diseases = max(len(eval_disease_local_ids), 1)
    avg_degree = total_degree / num_diseases
    avg_non_drug_degree = non_drug_degree / num_diseases
    print(
        f"[PROBE] {probe_name} Unseen Diseases Avg Degree in GNN: {avg_degree:.1f} edges per disease | "
        f"Non-drug avg degree: {avg_non_drug_degree:.1f} | num_diseases={len(eval_disease_local_ids)}"
    )

def _compute_avg_non_disease_drug_degree(
    *,
    pair_paths: Tensor,
    edge_index_dict: Mapping[EdgeType, Tensor],
    drug_global_ids: Tensor,
) -> float:
    eval_drug_global_ids = sorted({int(path[0]) for path in pair_paths.detach().cpu().tolist()})
    if not eval_drug_global_ids:
        return 0.0

    global_to_local = {
        int(global_id): local_idx
        for local_idx, global_id in enumerate(drug_global_ids.detach().cpu().tolist())
    }
    eval_drug_local_ids = [
        global_to_local[global_id]
        for global_id in eval_drug_global_ids
        if global_id in global_to_local
    ]
    if not eval_drug_local_ids:
        return 0.0

    eval_drug_local_tensor = torch.tensor(eval_drug_local_ids, dtype=torch.long)
    total_degree = 0
    for edge_type, edge_index in edge_index_dict.items():
        if 'disease' in edge_type[0] or 'disease' in edge_type[2]:
            continue
        cpu_edge_index = edge_index.detach().cpu()
        if edge_type[2] == 'drug':
            total_degree += int(torch.isin(cpu_edge_index[1], eval_drug_local_tensor).sum().item())
        if edge_type[0] == 'drug':
            total_degree += int(torch.isin(cpu_edge_index[0], eval_drug_local_tensor).sum().item())

    return total_degree / max(len(eval_drug_local_ids), 1)


def _emit_eval_drug_degree_probe(
    *,
    pair_reference_paths: Tensor,
    ot_paths: Tensor,
    edge_index_dict: Mapping[EdgeType, Tensor],
    drug_global_ids: Tensor,
    probe_name: str,
) -> None:
    pair_avg_degree = _compute_avg_non_disease_drug_degree(
        pair_paths=pair_reference_paths,
        edge_index_dict=edge_index_dict,
        drug_global_ids=drug_global_ids,
    )
    ot_avg_degree = _compute_avg_non_disease_drug_degree(
        pair_paths=ot_paths,
        edge_index_dict=edge_index_dict,
        drug_global_ids=drug_global_ids,
    )
    print(f"\n[PROBE - DRUG DEGREE LEAKAGE AUDIT] {probe_name}")
    print(f"Pair(valid+test) Drugs Avg Non-Disease Degree: {pair_avg_degree:.1f}")
    print(f"OT Drugs Avg Non-Disease Degree: {ot_avg_degree:.1f}")


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
        help='strict/direct_only: targeted pair-level clean that removes only held-out drug-disease prediction edges while preserving auxiliary mechanism edges.',
    )
    return parser.parse_args()


def evaluate_model_quad(
    model: RepurposingRGCN,
    data: HeteroData,
    valid_ho_paths: Tensor,
    batch_size: int = 32,
    ho_attr_name: str = 'ho_pos_paths',
    restrict_negative_disease_pool: bool = False,
    entity_lookup: Optional[Mapping[int, object]] = None,
    probe_name: str = 'EVAL',
    comparison_pair_paths: Optional[Tensor] = None,
) -> ResultDict:
    if valid_ho_paths.dim() != 2 or valid_ho_paths.size(1) != 3:
        raise ValueError('`valid_ho_paths` ??? `(N, 3)` ???????')

    valid_ho_paths = valid_ho_paths.detach().cpu().to(torch.long).contiguous()
    known_positive_paths = _build_known_positive_paths(
        data=data,
        valid_ho_paths=valid_ho_paths,
        ho_attr_name=ho_attr_name,
    )

    negative_disease_pool = None
    if restrict_negative_disease_pool:
        negative_disease_pool = sorted({int(path[-1]) for path in valid_ho_paths.tolist()})

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
            negative_disease_pool=negative_disease_pool,
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
            _emit_eval_disease_degree_probe(
                valid_ho_paths=valid_ho_paths,
                edge_index_dict=edge_index_dict,
                disease_global_ids=graph_data['disease'].global_id,
                probe_name=probe_name,
            )
            if comparison_pair_paths is not None:
                _emit_eval_drug_degree_probe(
                    pair_reference_paths=comparison_pair_paths,
                    ot_paths=valid_ho_paths,
                    edge_index_dict=edge_index_dict,
                    drug_global_ids=graph_data['drug'].global_id,
                    probe_name=probe_name,
                )

            results: ResultDict = {}
            for setting_name, dataloader in dataloaders.items():
                if hasattr(model, '_eval_path_probe_emitted'):
                    model._eval_path_probe_emitted = False
                dataset = dataloader.dataset
                print(
                    f"--- [PROBE] Evaluating on dataset: {dataset.__class__.__name__} | "
                    f"Setting: {setting_name} | Size: {len(dataset)} | PositivePaths: {int(valid_ho_paths.size(0))} ---"
                )
                if len(dataset) > 0:
                    first_sample = dataset[0]
                    first_pos_path = first_sample['pos_paths'][0].tolist() if first_sample['pos_paths'].size(0) > 0 else []
                    print(
                        f"--- [PROBE] First sample pos_pair={first_sample['pos_pair_ids'].tolist()} "
                        f"neg_pair={first_sample['neg_pair_ids'].tolist()} "
                        f"first_pos_path={first_pos_path} ---"
                    )
                    _emit_negative_tail_name_probe(
                        dataset=dataset,
                        setting_name=setting_name,
                        entity_lookup=entity_lookup,
                    )
                pos_score_chunks = []
                neg_score_chunks = []
                for batch_index, batch in enumerate(dataloader, start=1):
                    tensor_batch = _move_batch_to_device(batch=batch, device=device)
                    if batch_index <= 3:
                        _emit_empty_path_ratio_probe(
                            pos_attention_mask=tensor_batch['pos_attention_mask'],
                            neg_attention_mask=tensor_batch['neg_attention_mask'],
                            setting_name=setting_name,
                            batch_index=batch_index,
                        )
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
    holdout_pairs = collect_holdout_pairs_from_pair_splits(pair_splits=pair_splits)
    if args.graph_surgery_mode not in {'direct_only', 'strict'}:
        raise ValueError(f'Unsupported graph_surgery_mode: {args.graph_surgery_mode}')

    # Keep auxiliary mechanism edges for held-out nodes. We only remove held-out
    # drug-disease prediction edges so internal valid/test and OT see symmetric
    # non-prediction topology during message passing.
    clean_data = remove_direct_leakage_edges(
        data=data,
        holdout_pairs=holdout_pairs,
    )

    total_removed_edges = 0
    for edge_type in data.edge_index_dict.keys():
        original_edge_index = data[edge_type].edge_index
        clean_edge_index = clean_data[edge_type].edge_index
        total_removed_edges += int(original_edge_index.size(1) - clean_edge_index.size(1))

    train_negative_sampling_pools = build_train_negative_sampling_pools(
        split_mode=split_mode,
        pair_splits=pair_splits,
    )
    print(
        'train_negative_sampling_pools:',
        f"negative_drug_pool_size={0 if train_negative_sampling_pools['negative_drug_pool'] is None else len(train_negative_sampling_pools['negative_drug_pool'])}",
        f"negative_disease_pool_size={0 if train_negative_sampling_pools['negative_disease_pool'] is None else len(train_negative_sampling_pools['negative_disease_pool'])}",
    )
    train_loader = build_pair_path_bpr_dataloader(
        data=clean_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        negative_strategy='mixed',
        use_pathway_quads=True,
        negative_drug_pool=train_negative_sampling_pools['negative_drug_pool'],
        negative_disease_pool=train_negative_sampling_pools['negative_disease_pool'],
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
            valid_eval = evaluate_model_quad(
                model=model,
                data=clean_data,
                valid_ho_paths=valid_tensor,
                batch_size=args.batch_size,
                restrict_negative_disease_pool=True,
                entity_lookup=processor.id2entity,
            )
            test_eval = evaluate_model_quad(
                model=model,
                data=clean_data,
                valid_ho_paths=test_tensor,
                batch_size=args.batch_size,
                restrict_negative_disease_pool=True,
                entity_lookup=processor.id2entity,
            )
            ot_eval = evaluate_model_quad(
                model=model,
                data=clean_data,
                valid_ho_paths=ot_tensor,
                batch_size=args.batch_size,
                restrict_negative_disease_pool=True,
                entity_lookup=processor.id2entity,
            )
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
