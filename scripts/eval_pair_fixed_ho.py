from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import pandas as pd
import torch
from sklearn.metrics import average_precision_score
from torch import Tensor

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.evaluate_ho_mechanisms import (  # noqa: E402
    build_full_node_corruption_benchmark,
    build_full_real_mechanism_quads,
    build_gene_to_pathways_from_graph,
    collect_global_node_id_pools,
    expand_paths_to_quads,
)
from scripts.train_quad_split_ho_probe import (  # noqa: E402
    build_clean_graph_without_leakage,
    load_ot_triplets,
)
from scripts.train_split_probe import derive_ho_triplets_for_pair_splits, load_pair_splits  # noqa: E402
from src.evaluation_utils import _extract_x_dict, _infer_model_device  # noqa: E402
from src.feature_utils import inject_features_to_graph  # noqa: E402
from src.primekg_data_processor import PrimeKGDataProcessor  # noqa: E402
from src.repurposing_rgcn import EmbeddingDict, EdgeType, RepurposingRGCN  # noqa: E402

IdTriplet = Tuple[int, int, int]
PathQuad = Tuple[int, int, int, int]
MetricDict = Dict[str, float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Evaluate pair-fixed HO metrics by keeping only pos/neg_gene/neg_pathway candidates.'
    )
    parser.add_argument('--checkpoint-path', type=Path, required=True)
    parser.add_argument('--processed-path', type=Path, required=True)
    parser.add_argument('--nodes-csv', type=Path, default=Path('data/PrimeKG/nodes.csv'))
    parser.add_argument('--edges-csv', type=Path, default=Path('data/PrimeKG/edges.csv'))
    parser.add_argument('--feature-dir', type=Path, default=Path('outputs/pubmedbert_hybrid_features_clean'))
    parser.add_argument('--ot-novel-csv', type=Path, default=Path('outputs/ot_random_external_profile/novel_ood_triplets.csv'))
    parser.add_argument('--graph-surgery-mode', type=str, default='direct_only', choices=['strict', 'direct_only'])
    parser.add_argument('--max-sampling-attempts', type=int, default=256)
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num-trials', type=int, default=10)
    parser.add_argument('--tie-noise-std', type=float, default=1e-6)
    parser.add_argument('--ablate-gnn', action='store_true', help='Force ablate_gnn=True regardless of checkpoint config.')
    parser.add_argument('--output-json', type=Path, default=None)
    return parser.parse_args()


def instantiate_model_from_checkpoint(
    checkpoint_path: Path,
    clean_data,
    nodes_csv: Path,
    ablate_gnn_override: bool,
) -> RepurposingRGCN:
    payload = torch.load(checkpoint_path, map_location='cpu')
    model_config: Dict[str, object] = payload['model_config']
    effective_ablate = bool(model_config.get('ablate_gnn', False)) or bool(ablate_gnn_override)

    model = RepurposingRGCN(
        data=clean_data,
        in_channels=768,
        hidden_channels=int(model_config['hidden_channels']),
        out_dim=int(model_config['out_dim']),
        scorer_hidden_dim=int(model_config['scorer_hidden_dim']),
        dropout=float(model_config['dropout']),
        initial_residual_alpha=float(model_config.get('initial_residual_alpha', 0.2)),
        encoder_type=str(model_config['encoder_type']),
        agg_type=str(model_config['agg_type']),
        use_pathway_quads=True,
        triplet_text_embeddings_path=model_config.get('triplet_text_embeddings_path'),
        drug_morgan_fingerprints_path=model_config.get('drug_morgan_fingerprints_path'),
        drug_text_embeddings_path=model_config.get('drug_text_embeddings_path'),
        disease_text_embeddings_path=model_config.get('disease_text_embeddings_path'),
        nodes_csv_path=nodes_csv,
        text_distill_alpha=float(model_config.get('text_distill_alpha', 0.0)),
        use_early_external_fusion=bool(model_config.get('use_early_external_fusion', False)),
        dropedge_p=float(model_config.get('dropedge_p', 0.15)),
        ablate_gnn=effective_ablate,
    )
    model.load_state_dict(payload['model_state_dict'], strict=True)
    return model


def build_clean_eval_graph(
    processed_path: Path,
    nodes_csv: Path,
    edges_csv: Path,
    feature_dir: Path,
    ot_novel_csv: Path,
    graph_surgery_mode: str,
):
    processor = PrimeKGDataProcessor(node_csv_path=nodes_csv, edge_csv_path=edges_csv)
    processor.build_entity_mappings()
    split_mode, pair_splits = load_pair_splits(processed_path)
    split_triplets = derive_ho_triplets_for_pair_splits(
        processor=processor,
        edge_csv_path=edges_csv,
        pair_splits=pair_splits,
    )
    ot_triplets = load_ot_triplets(ot_novel_csv, processor.global_entity2id)

    full_data = processor.build_heterodata(
        ho_id_paths=split_triplets['train'],
        add_inverse_edges=False,
    )
    inject_features_to_graph(data=full_data, feature_dir=feature_dir)

    heldout_triplets = split_triplets['valid'] + split_triplets['test'] + ot_triplets
    clean_data, total_removed_edges, leakage_edge_summary = build_clean_graph_without_leakage(
        data=full_data,
        heldout_triplets=heldout_triplets,
        split_mode=split_mode,
        pair_splits=pair_splits,
        graph_surgery_mode=graph_surgery_mode,
    )
    return processor, split_mode, split_triplets, clean_data, total_removed_edges, leakage_edge_summary


def score_grouped_paths(
    model: RepurposingRGCN,
    data,
    grouped_paths: Tensor,
    batch_size: int,
) -> Tensor:
    if grouped_paths.dim() != 3 or grouped_paths.size(1) != 3 or grouped_paths.size(2) != 4:
        raise ValueError(f'Expected grouped_paths to have shape (N, 3, 4), got {tuple(grouped_paths.shape)}.')

    was_training = model.training
    model.eval()
    device = _infer_model_device(model)
    graph_data = copy.deepcopy(data).to(device)

    try:
        with torch.no_grad():
            x_dict = _extract_x_dict(full_graph_data=graph_data)
            edge_index_dict: Mapping[EdgeType, Tensor] = graph_data.edge_index_dict
            node_embs_dict: EmbeddingDict = model.encode(x_dict=x_dict, edge_index_dict=edge_index_dict)

            chunks: List[Tensor] = []
            grouped_paths_cpu = grouped_paths.detach().cpu().to(torch.long)
            num_groups = grouped_paths_cpu.size(0)
            for start in range(0, num_groups, batch_size):
                end = min(start + batch_size, num_groups)
                batch_group = grouped_paths_cpu[start:end].to(device)
                flat_paths = batch_group.view(-1, 1, 4)
                pair_ids = torch.stack([flat_paths[:, 0, 0], flat_paths[:, 0, 3]], dim=1)
                attention_mask = torch.ones((flat_paths.size(0), 1), dtype=torch.bool, device=device)
                logits = model.score_batch(
                    node_embs_dict=node_embs_dict,
                    pair_ids=pair_ids,
                    paths=flat_paths,
                    attention_mask=attention_mask,
                )
                chunks.append(logits.detach().cpu().view(-1, 3))
    finally:
        if was_training:
            model.train()

    if not chunks:
        raise ValueError('No score chunks were produced.')
    return torch.cat(chunks, dim=0)


def compute_pair_fixed_metrics(
    group_scores: Tensor,
    num_trials: int = 10,
    tie_noise_std: float = 1e-6,
) -> MetricDict:
    if group_scores.dim() != 2 or group_scores.size(1) != 3:
        raise ValueError(f'Expected group_scores to have shape (N, 3), got {tuple(group_scores.shape)}.')
    if num_trials <= 0:
        raise ValueError('`num_trials` must be positive.')
    if tie_noise_std <= 0.0:
        raise ValueError('`tie_noise_std` must be positive.')

    grouped_labels = torch.zeros((group_scores.size(0), 3), dtype=torch.long)
    grouped_labels[:, 0] = 1

    y_score = group_scores.reshape(-1).detach().cpu().numpy()
    y_true = grouped_labels.reshape(-1).detach().cpu().numpy()
    auprc = float(average_precision_score(y_true, y_score))

    hit_values = []
    mrr_values = []
    for _ in range(num_trials):
        noise = torch.randn_like(group_scores) * tie_noise_std
        logits_for_ranking = group_scores + noise
        sorted_indices = torch.argsort(logits_for_ranking, dim=1, descending=True)
        positive_rank_positions = (sorted_indices == 0).nonzero(as_tuple=False)[:, 1]
        hit_values.append(float((positive_rank_positions == 0).float().mean().item()))
        mrr_values.append(float((1.0 / (positive_rank_positions.to(torch.float32) + 1.0)).mean().item()))

    hit_at_1 = float(sum(hit_values) / len(hit_values))
    mrr = float(sum(mrr_values) / len(mrr_values))

    return {
        'auprc': auprc,
        'hit_at_1': hit_at_1,
        'mrr': mrr,
        'num_groups': float(group_scores.size(0)),
        'num_candidates': float(group_scores.numel()),
        'num_trials': float(num_trials),
        'tie_noise_std': float(tie_noise_std),
    }


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    processor, split_mode, split_triplets, clean_data, total_removed_edges, leakage_edge_summary = build_clean_eval_graph(
        processed_path=args.processed_path,
        nodes_csv=args.nodes_csv,
        edges_csv=args.edges_csv,
        feature_dir=args.feature_dir,
        ot_novel_csv=args.ot_novel_csv,
        graph_surgery_mode=args.graph_surgery_mode,
    )
    model = instantiate_model_from_checkpoint(
        checkpoint_path=args.checkpoint_path,
        clean_data=clean_data,
        nodes_csv=args.nodes_csv,
        ablate_gnn_override=args.ablate_gnn,
    )

    gene_to_pathways = build_gene_to_pathways_from_graph(clean_data)
    positive_triplets = torch.tensor(split_triplets['test'], dtype=torch.long)
    positive_quads = expand_paths_to_quads(
        path_tensor=positive_triplets,
        gene_to_pathways=gene_to_pathways,
        pathway_dummy_global_id=0,
    )
    real_quad_set = build_full_real_mechanism_quads(
        data=clean_data,
        gene_to_pathways=gene_to_pathways,
        pathway_dummy_global_id=0,
    )
    node_id_pools = collect_global_node_id_pools(clean_data)
    grouped_paths_5, _ = build_full_node_corruption_benchmark(
        positive_quads=positive_quads,
        node_id_pools=node_id_pools,
        real_quad_set=real_quad_set,
        max_sampling_attempts=args.max_sampling_attempts,
        generator=torch.Generator().manual_seed(args.seed),
    )

    # Keep only: positive, neg_gene, neg_pathway.
    grouped_paths_3 = grouped_paths_5[:, [0, 2, 3], :].contiguous()
    group_scores = score_grouped_paths(
        model=model,
        data=clean_data,
        grouped_paths=grouped_paths_3,
        batch_size=args.batch_size,
    )
    metrics = compute_pair_fixed_metrics(
        group_scores,
        num_trials=args.num_trials,
        tie_noise_std=args.tie_noise_std,
    )

    payload = {
        'checkpoint_path': str(args.checkpoint_path),
        'processed_path': str(args.processed_path),
        'split_mode': split_mode,
        'graph_surgery_mode': args.graph_surgery_mode,
        'ablate_gnn_effective': bool(model.ablate_gnn),
        'total_removed_leakage_edges': int(total_removed_edges),
        'leakage_edge_summary': leakage_edge_summary,
        'pair_fixed_metrics': metrics,
        'num_positive_triplets': int(positive_triplets.size(0)),
        'num_positive_quads': int(positive_quads.size(0)),
        'num_pair_fixed_groups': int(grouped_paths_3.size(0)),
    }

    print('# Pair-Fixed HO Evaluation')
    print(f'checkpoint: {args.checkpoint_path}')
    print(f'processed_path: {args.processed_path}')
    print(f'split_mode: {split_mode}')
    print(f'ablate_gnn_effective: {model.ablate_gnn}')
    print(f'num_pair_fixed_groups: {grouped_paths_3.size(0)}')
    print()
    print('| Metric | Value |')
    print('|---|---:|')
    print(f"| Pair-Fixed AUPRC | {metrics['auprc']:.4f} |")
    print(f"| Pair-Fixed Hit@1 | {metrics['hit_at_1']:.4f} |")
    print(f"| Pair-Fixed MRR | {metrics['mrr']:.4f} |")

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(payload, indent=2), encoding='utf-8')
        print()
        print(f'Saved report to: {args.output_json}')


if __name__ == '__main__':
    main()
