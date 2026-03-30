from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch_geometric.data import HeteroData


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.graph_surgery import (
    build_split_isolation_targets,
    collect_holdout_pairs_from_pair_splits,
    remove_direct_leakage_edges,
    remove_leakage_edges,
)


def _build_toy_heterodata(include_pathway: bool = False) -> HeteroData:
    data = HeteroData()

    data['drug'].num_nodes = 3
    data['drug'].global_id = torch.tensor([10, 11, 12], dtype=torch.long)

    data['gene/protein'].num_nodes = 3
    data['gene/protein'].global_id = torch.tensor([20, 21, 22], dtype=torch.long)

    data['disease'].num_nodes = 3
    data['disease'].global_id = torch.tensor([30, 31, 32], dtype=torch.long)

    data[('drug', 'targets', 'gene/protein')].edge_index = torch.tensor(
        [[0, 1, 2], [0, 1, 2]], dtype=torch.long
    )
    data[('gene/protein', 'targeted_by', 'drug')].edge_index = torch.tensor(
        [[0, 1, 2], [0, 1, 2]], dtype=torch.long
    )
    data[('gene/protein', 'associates', 'disease')].edge_index = torch.tensor(
        [[0, 1, 2], [0, 1, 2]], dtype=torch.long
    )
    data[('disease', 'associated_with', 'gene/protein')].edge_index = torch.tensor(
        [[0, 1, 2], [0, 1, 2]], dtype=torch.long
    )
    data[('drug', 'treats', 'disease')].edge_index = torch.tensor(
        [[0, 1, 2], [0, 1, 2]], dtype=torch.long
    )

    if include_pathway:
        data['pathway'].num_nodes = 2
        data['pathway'].global_id = torch.tensor([40, 41], dtype=torch.long)
        data[('gene/protein', 'participates_in', 'pathway')].edge_index = torch.tensor(
            [[0, 0, 1], [0, 1, 1]], dtype=torch.long
        )
        data[('pathway', 'participates_in__reverse__', 'gene/protein')].edge_index = torch.tensor(
            [[0, 1, 1], [0, 0, 1]], dtype=torch.long
        )

    data.ho_path_node_types = ('drug', 'gene/protein', 'disease')
    return data


def _build_toy_direct_leakage_graph() -> HeteroData:
    data = _build_toy_heterodata(include_pathway=True)
    data[('drug', 'indication', 'disease')].edge_index = torch.tensor(
        [[0, 1, 2], [0, 1, 2]], dtype=torch.long
    )
    data[('drug', 'off-label use', 'disease')].edge_index = torch.tensor(
        [[0, 1], [0, 2]], dtype=torch.long
    )
    data[('drug', 'contraindication', 'disease')].edge_index = torch.tensor(
        [[0, 2], [1, 2]], dtype=torch.long
    )
    data[('disease', 'contraindication', 'drug')].edge_index = torch.tensor(
        [[1, 2], [0, 2]], dtype=torch.long
    )
    data[('disease', 'off-label use', 'drug')].edge_index = torch.tensor(
        [[0, 2], [0, 1]], dtype=torch.long
    )
    data[('disease', 'rev_indication', 'drug')].edge_index = torch.tensor(
        [[0, 1, 2], [0, 1, 2]], dtype=torch.long
    )
    return data


def test_collect_holdout_pairs_from_pair_splits_uses_valid_and_test_only() -> None:
    pair_splits = {
        'train': {(10, 30), (11, 31)},
        'valid': {(12, 32)},
        'test': {(13, 33), (14, 34)},
    }

    holdout_pairs = collect_holdout_pairs_from_pair_splits(pair_splits)

    assert torch.equal(
        holdout_pairs,
        torch.tensor([[12, 32], [13, 33], [14, 34]], dtype=torch.long),
    )


def test_remove_direct_leakage_edges_only_removes_holdout_shortcuts() -> None:
    data = _build_toy_direct_leakage_graph()

    original_drug_gene = data[('drug', 'targets', 'gene/protein')].edge_index.clone()
    original_gene_disease = data[('disease', 'associated_with', 'gene/protein')].edge_index.clone()
    original_gene_pathway = data[('gene/protein', 'participates_in', 'pathway')].edge_index.clone()

    clean_data = remove_direct_leakage_edges(
        data=data,
        holdout_pairs=torch.tensor([[10, 30], [12, 32]], dtype=torch.long),
    )

    assert torch.equal(
        clean_data[('drug', 'indication', 'disease')].edge_index,
        torch.tensor([[1], [1]], dtype=torch.long),
    )
    assert torch.equal(
        clean_data[('drug', 'off-label use', 'disease')].edge_index,
        torch.tensor([[1], [2]], dtype=torch.long),
    )
    assert torch.equal(
        clean_data[('drug', 'contraindication', 'disease')].edge_index,
        torch.tensor([[0], [1]], dtype=torch.long),
    )
    assert torch.equal(
        clean_data[('disease', 'contraindication', 'drug')].edge_index,
        torch.tensor([[1], [0]], dtype=torch.long),
    )
    assert torch.equal(
        clean_data[('disease', 'off-label use', 'drug')].edge_index,
        torch.tensor([[2], [1]], dtype=torch.long),
    )
    assert torch.equal(
        clean_data[('disease', 'rev_indication', 'drug')].edge_index,
        torch.tensor([[1], [1]], dtype=torch.long),
    )

    # ????????????????
    assert torch.equal(clean_data[('drug', 'targets', 'gene/protein')].edge_index, original_drug_gene)
    assert torch.equal(clean_data[('disease', 'associated_with', 'gene/protein')].edge_index, original_gene_disease)
    assert torch.equal(clean_data[('gene/protein', 'participates_in', 'pathway')].edge_index, original_gene_pathway)

    # ???????????
    assert torch.equal(
        data[('drug', 'indication', 'disease')].edge_index,
        torch.tensor([[0, 1, 2], [0, 1, 2]], dtype=torch.long),
    )


def test_remove_leakage_edges_removes_forward_reverse_and_pair_shortcuts() -> None:
    data = _build_toy_heterodata()
    original_direct_edges = data[('drug', 'treats', 'disease')].edge_index.clone()

    clean_edge_index_dict = remove_leakage_edges(
        data=data,
        target_paths=torch.tensor([[10, 20, 30], [12, 22, 32]], dtype=torch.long),
    )

    assert torch.equal(
        clean_edge_index_dict[('drug', 'targets', 'gene/protein')],
        torch.tensor([[1], [1]], dtype=torch.long),
    )
    assert torch.equal(
        clean_edge_index_dict[('gene/protein', 'targeted_by', 'drug')],
        torch.tensor([[1], [1]], dtype=torch.long),
    )
    assert torch.equal(
        clean_edge_index_dict[('gene/protein', 'associates', 'disease')],
        torch.tensor([[1], [1]], dtype=torch.long),
    )
    assert torch.equal(
        clean_edge_index_dict[('disease', 'associated_with', 'gene/protein')],
        torch.tensor([[1], [1]], dtype=torch.long),
    )
    assert torch.equal(
        clean_edge_index_dict[('drug', 'treats', 'disease')],
        torch.tensor([[1], [1]], dtype=torch.long),
    )
    assert torch.equal(data[('drug', 'treats', 'disease')].edge_index, original_direct_edges)



def test_triplet_mode_removes_all_gene_pathway_edges_for_heldout_gene() -> None:
    data = _build_toy_heterodata(include_pathway=True)

    clean_edge_index_dict = remove_leakage_edges(
        data=data,
        target_paths=torch.tensor([[10, 20, 30]], dtype=torch.long),
    )

    assert torch.equal(
        clean_edge_index_dict[('gene/protein', 'participates_in', 'pathway')],
        torch.tensor([[1], [1]], dtype=torch.long),
    )
    assert torch.equal(
        clean_edge_index_dict[('pathway', 'participates_in__reverse__', 'gene/protein')],
        torch.tensor([[1], [1]], dtype=torch.long),
    )



def test_quad_mode_removes_only_matching_gene_pathway_pair() -> None:
    data = _build_toy_heterodata(include_pathway=True)

    clean_edge_index_dict = remove_leakage_edges(
        data=data,
        target_paths=torch.tensor([[10, 20, 40, 30]], dtype=torch.long),
    )

    assert torch.equal(
        clean_edge_index_dict[('gene/protein', 'participates_in', 'pathway')],
        torch.tensor([[0, 1], [1, 1]], dtype=torch.long),
    )
    assert torch.equal(
        clean_edge_index_dict[('pathway', 'participates_in__reverse__', 'gene/protein')],
        torch.tensor([[1, 1], [0, 1]], dtype=torch.long),
    )



def test_remove_leakage_edges_supports_entity_isolation() -> None:
    data = _build_toy_heterodata()

    clean_edge_index_dict = remove_leakage_edges(
        data=data,
        target_paths=torch.tensor([[10, 20, 30]], dtype=torch.long),
        isolate_nodes_by_type={'disease': [31]},
    )

    assert torch.equal(
        clean_edge_index_dict[('drug', 'treats', 'disease')],
        torch.tensor([[2], [2]], dtype=torch.long),
    )
    assert torch.equal(
        clean_edge_index_dict[('gene/protein', 'associates', 'disease')],
        torch.tensor([[2], [2]], dtype=torch.long),
    )
    assert torch.equal(
        clean_edge_index_dict[('disease', 'associated_with', 'gene/protein')],
        torch.tensor([[2], [2]], dtype=torch.long),
    )



def test_build_split_isolation_targets_returns_expected_nodes() -> None:
    pair_splits = {
        'train': {(10, 30)},
        'valid': {(11, 31)},
        'test': {(12, 32)},
    }

    drug_targets = build_split_isolation_targets(split_mode='cross_drug', pair_splits=pair_splits)
    disease_targets = build_split_isolation_targets(split_mode='cross_disease', pair_splits=pair_splits)

    assert torch.equal(drug_targets['drug'], torch.tensor([11, 12], dtype=torch.long))
    assert torch.equal(disease_targets['disease'], torch.tensor([31, 32], dtype=torch.long))
