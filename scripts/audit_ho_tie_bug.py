from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List

import torch
from torch import Tensor

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.evaluate_ho_mechanisms import (
    build_full_node_corruption_benchmark,
    build_full_real_mechanism_quads,
    build_gene_to_pathways_from_graph,
    collect_global_node_id_pools,
    expand_paths_to_quads,
)
from scripts.train_quad_split_ho_probe import (
    build_clean_graph_without_leakage,
    load_ot_triplets,
)
from scripts.train_split_probe import derive_ho_triplets_for_pair_splits, load_pair_splits
from src.evaluation_utils import _extract_x_dict, _infer_model_device
from src.feature_utils import inject_features_to_graph
from src.primekg_data_processor import PrimeKGDataProcessor
from src.repurposing_rgcn import RepurposingRGCN


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Audit whether ablate_gnn HO scores collapse into tie scores.'
    )
    parser.add_argument(
        '--checkpoint-path',
        type=Path,
        default=Path(
            'outputs/asym_route_dropedge_pairclean_sapbert_pathgate_dualtower_ablategnn_20260329/'
            'cross_disease_split_quad_asym_route_dropedge_pairclean_sapbert_pathgate_dualtower_ablategnn_60epoch.pt'
        ),
    )
    parser.add_argument(
        '--processed-path',
        type=Path,
        default=Path('data/PrimeKG/processed/primekg_indication_cross_disease.pt'),
    )
    parser.add_argument('--nodes-csv', type=Path, default=Path('data/PrimeKG/nodes.csv'))
    parser.add_argument('--edges-csv', type=Path, default=Path('data/PrimeKG/edges.csv'))
    parser.add_argument(
        '--feature-dir',
        type=Path,
        default=Path('outputs/pubmedbert_hybrid_features_clean'),
    )
    parser.add_argument(
        '--ot-novel-csv',
        type=Path,
        default=Path('outputs/ot_random_external_profile/novel_ood_triplets.csv'),
    )
    parser.add_argument('--num-groups', type=int, default=3)
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()


def instantiate_model_from_checkpoint(
    checkpoint_path: Path,
    clean_data,
    nodes_csv: Path,
) -> RepurposingRGCN:
    payload = torch.load(checkpoint_path, map_location='cpu')
    model_config: Dict[str, object] = payload['model_config']

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
        ablate_gnn=bool(model_config.get('ablate_gnn', False)),
    )
    model.load_state_dict(payload['model_state_dict'], strict=True)
    return model


def build_clean_eval_graph(args: argparse.Namespace):
    processor = PrimeKGDataProcessor(node_csv_path=args.nodes_csv, edge_csv_path=args.edges_csv)
    processor.build_entity_mappings()
    split_mode, pair_splits = load_pair_splits(args.processed_path)
    split_triplets = derive_ho_triplets_for_pair_splits(
        processor=processor,
        edge_csv_path=args.edges_csv,
        pair_splits=pair_splits,
    )
    ot_triplets = load_ot_triplets(args.ot_novel_csv, processor.global_entity2id)

    full_data = processor.build_heterodata(
        ho_id_paths=split_triplets['train'],
        add_inverse_edges=False,
    )
    inject_features_to_graph(data=full_data, feature_dir=args.feature_dir)

    heldout_triplets = split_triplets['valid'] + split_triplets['test'] + ot_triplets
    clean_data, _, _ = build_clean_graph_without_leakage(
        data=full_data,
        heldout_triplets=heldout_triplets,
        split_mode=split_mode,
        pair_splits=pair_splits,
        graph_surgery_mode='direct_only',
    )
    return clean_data, split_triplets


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    clean_data, split_triplets = build_clean_eval_graph(args)
    model = instantiate_model_from_checkpoint(
        checkpoint_path=args.checkpoint_path,
        clean_data=clean_data,
        nodes_csv=args.nodes_csv,
    )

    device = _infer_model_device(model)
    model = model.to(device)
    model.eval()

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
    grouped_paths, _ = build_full_node_corruption_benchmark(
        positive_quads=positive_quads,
        node_id_pools=node_id_pools,
        real_quad_set=real_quad_set,
        max_sampling_attempts=256,
        generator=torch.Generator().manual_seed(args.seed),
    )

    num_groups = min(int(args.num_groups), int(grouped_paths.size(0)))
    selected_groups = grouped_paths[:num_groups].clone()

    with torch.no_grad():
        graph_data = clean_data.to(device)
        x_dict = _extract_x_dict(full_graph_data=graph_data)
        node_embs_dict = model.encode(
            x_dict=x_dict,
            edge_index_dict=graph_data.edge_index_dict,
        )

        flat_paths = selected_groups.to(device).view(-1, 1, 4)
        pair_ids = torch.stack([flat_paths[:, 0, 0], flat_paths[:, 0, 3]], dim=1)
        attention_mask = torch.ones((flat_paths.size(0), 1), dtype=torch.bool, device=device)
        logits = model.score_batch(
            node_embs_dict=node_embs_dict,
            pair_ids=pair_ids,
            paths=flat_paths,
            attention_mask=attention_mask,
        ).detach().cpu().view(num_groups, 5)

    candidate_names: List[str] = [
        'positive',
        'neg_drug',
        'neg_gene',
        'neg_pathway',
        'neg_disease',
    ]

    print('# HO Tie Audit')
    print(f'checkpoint: {args.checkpoint_path}')
    print(f'processed_path: {args.processed_path}')
    print(f'ablate_gnn: {model.ablate_gnn}')
    print(f'groups_inspected: {num_groups}')
    print()

    for group_index in range(num_groups):
        group_paths = selected_groups[group_index]
        group_scores = logits[group_index]
        same_pair_scores = group_scores[[0, 2, 3]]
        same_pair_exact = bool(torch.equal(same_pair_scores, same_pair_scores[0].expand_as(same_pair_scores)))
        same_pair_allclose = bool(
            torch.allclose(
                same_pair_scores,
                same_pair_scores[0].expand_as(same_pair_scores),
                atol=1e-8,
                rtol=1e-7,
            )
        )
        full_group_allclose = bool(
            torch.allclose(
                group_scores,
                group_scores[0].expand_as(group_scores),
                atol=1e-8,
                rtol=1e-7,
            )
        )

        print(f'## Group {group_index + 1}')
        for name, path_row, score in zip(candidate_names, group_paths.tolist(), group_scores.tolist()):
            print(f'{name:12s} path={tuple(int(x) for x in path_row)} score={score:.10f}')
        print('same_pair(pos/neg_gene/neg_pathway):', [float(x) for x in same_pair_scores.tolist()])
        print(
            'same_pair_exact_equal=', same_pair_exact,
            ' same_pair_allclose=', same_pair_allclose,
            ' full_group_allclose=', full_group_allclose,
            ' max_minus_min=', float(group_scores.max().item() - group_scores.min().item()),
        )
        print()


if __name__ == '__main__':
    main()
