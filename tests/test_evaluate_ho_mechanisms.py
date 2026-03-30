from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch_geometric.data import HeteroData

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.evaluate_ho_mechanisms import (
    build_full_node_corruption_benchmark,
    build_full_real_mechanism_quads,
    build_gene_to_pathways_from_graph,
    collect_global_node_id_pools,
    compute_imbalanced_metrics,
    evaluate_ho_mechanisms,
)
from src.repurposing_rgcn import RepurposingRGCN


def _build_toy_heterodata_with_pathways() -> HeteroData:
    data = HeteroData()

    data['drug'].num_nodes = 3
    data['drug'].global_id = torch.tensor([10, 11, 12], dtype=torch.long)

    data['gene/protein'].num_nodes = 4
    data['gene/protein'].global_id = torch.tensor([20, 21, 22, 23], dtype=torch.long)

    data['pathway'].num_nodes = 3
    data['pathway'].global_id = torch.tensor([40, 41, 42], dtype=torch.long)

    data['disease'].num_nodes = 3
    data['disease'].global_id = torch.tensor([30, 31, 32], dtype=torch.long)

    data[('drug', 'targets', 'gene/protein')].edge_index = torch.tensor(
        [[0, 0, 1, 2], [0, 1, 2, 3]], dtype=torch.long
    )
    data[('gene/protein', 'targets__reverse__', 'drug')].edge_index = torch.tensor(
        [[0, 1, 2, 3], [0, 0, 1, 2]], dtype=torch.long
    )
    data[('gene/protein', 'associates', 'disease')].edge_index = torch.tensor(
        [[0, 1, 2, 3], [0, 0, 1, 2]], dtype=torch.long
    )
    data[('disease', 'associated_with', 'gene/protein')].edge_index = torch.tensor(
        [[0, 0, 1, 2], [0, 1, 2, 3]], dtype=torch.long
    )
    data[('gene/protein', 'participates_in', 'pathway')].edge_index = torch.tensor(
        [[0, 0, 2], [0, 1, 1]], dtype=torch.long
    )
    data[('pathway', 'participates_in__reverse__', 'gene/protein')].edge_index = torch.tensor(
        [[0, 1, 1], [0, 0, 2]], dtype=torch.long
    )

    data.ho_pos_paths = torch.tensor(
        [
            [10, 20, 40, 30],
            [10, 20, 41, 30],
            [10, 21, 0, 30],
            [11, 22, 41, 31],
        ],
        dtype=torch.long,
    )
    data.ho_path_node_types = ('drug', 'gene/protein', 'pathway', 'disease')
    return data


def test_full_node_corruption_sampler_respects_real_quad_set() -> None:
    data = _build_toy_heterodata_with_pathways()
    gene_to_pathways = build_gene_to_pathways_from_graph(data)
    real_quad_set = build_full_real_mechanism_quads(data=data, gene_to_pathways=gene_to_pathways)
    node_id_pools = collect_global_node_id_pools(data=data)

    positive_quads = data.ho_pos_paths[:2]
    grouped_paths, grouped_labels = build_full_node_corruption_benchmark(
        positive_quads=positive_quads,
        node_id_pools=node_id_pools,
        real_quad_set=real_quad_set,
        max_sampling_attempts=128,
        generator=torch.Generator().manual_seed(0),
    )

    assert grouped_paths.shape == (2, 5, 4)
    assert grouped_labels.shape == (2, 5)
    assert torch.equal(grouped_labels[:, 0], torch.ones(2, dtype=torch.long))
    assert torch.equal(grouped_labels[:, 1:], torch.zeros((2, 4), dtype=torch.long))

    for group in grouped_paths.tolist():
        pos, neg_d, neg_g, neg_p, neg_c = [tuple(int(x) for x in row) for row in group]
        assert pos in real_quad_set
        assert neg_d not in real_quad_set
        assert neg_g not in real_quad_set
        assert neg_p not in real_quad_set
        assert neg_c not in real_quad_set
        assert neg_d[1:] == pos[1:]
        assert (neg_g[0], neg_g[2], neg_g[3]) == (pos[0], pos[2], pos[3])
        assert (neg_p[0], neg_p[1], neg_p[3]) == (pos[0], pos[1], pos[3])
        assert neg_c[:3] == pos[:3]


def test_evaluate_ho_mechanisms_returns_finite_metrics() -> None:
    torch.manual_seed(0)

    data = _build_toy_heterodata_with_pathways()
    model = RepurposingRGCN(
        data=data,
        hidden_dim=8,
        out_dim=8,
        scorer_hidden_dim=4,
        dropout=0.0,
        use_pathway_quads=True,
    )

    positive_quads = data.ho_pos_paths
    results = evaluate_ho_mechanisms(
        model=model,
        eval_data=data,
        positive_quads=positive_quads,
        reference_data=data,
        batch_size=2,
        max_sampling_attempts=128,
        seed=0,
    )

    metrics = results['metrics']
    grouped_paths = results['grouped_paths']
    grouped_labels = results['grouped_labels']
    group_scores = results['group_scores']

    assert grouped_paths.shape == (positive_quads.size(0), 5, 4)
    assert grouped_labels.shape == (positive_quads.size(0), 5)
    assert group_scores.shape == (positive_quads.size(0), 5)
    assert torch.isfinite(group_scores).all()
    assert 0.0 <= metrics['auprc'] <= 1.0
    assert 0.0 <= metrics['hit_at_1'] <= 1.0
    assert 0.0 < metrics['mrr'] <= 1.0
    assert int(metrics['num_groups']) == positive_quads.size(0)
    assert int(metrics['num_candidates']) == positive_quads.size(0) * 5

    recomputed = compute_imbalanced_metrics(group_scores=group_scores, grouped_labels=grouped_labels)
    assert abs(metrics['auprc'] - recomputed['auprc']) < 1e-8
    assert abs(metrics['hit_at_1'] - recomputed['hit_at_1']) < 1e-8
    assert abs(metrics['mrr'] - recomputed['mrr']) < 1e-8
