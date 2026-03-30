from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
from torch_geometric.data import HeteroData

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.audit_data_leakage import (
    audit_graph_surgery_residuals,
    audit_ot_pair_level_leakage,
    audit_strict_cross_split_purity,
    audit_text_feature_contamination,
)


def _build_toy_no_leak_graph() -> HeteroData:
    data = HeteroData()
    data['drug'].num_nodes = 2
    data['drug'].global_id = torch.tensor([10, 11], dtype=torch.long)

    data['gene/protein'].num_nodes = 2
    data['gene/protein'].global_id = torch.tensor([20, 21], dtype=torch.long)

    data['pathway'].num_nodes = 2
    data['pathway'].global_id = torch.tensor([40, 41], dtype=torch.long)

    data['disease'].num_nodes = 2
    data['disease'].global_id = torch.tensor([30, 31], dtype=torch.long)

    data[('drug', 'targets', 'gene/protein')].edge_index = torch.tensor([[0], [0]], dtype=torch.long)
    data[('gene/protein', 'associates', 'disease')].edge_index = torch.tensor([[0], [0]], dtype=torch.long)
    data[('gene/protein', 'participates_in', 'pathway')].edge_index = torch.tensor([[0], [0]], dtype=torch.long)
    return data


def test_text_feature_audit_passes_when_no_name_leak() -> None:
    id_to_text = {
        10: 'Aspirin interacts with PTGS1.',
        30: 'Headache is linked to inflammation.',
    }
    id_to_name = {10: 'Aspirin', 30: 'Headache'}
    result = audit_text_feature_contamination(
        id_to_text=id_to_text,
        id_to_name=id_to_name,
        valid_pairs=[(10, 30)],
        test_pairs=[],
    )
    assert result['num_leaks'] == 0


def test_text_feature_audit_raises_on_string_leak() -> None:
    id_to_text = {
        10: 'Aspirin may treat Headache directly.',
        30: 'Headache symptoms can be reduced by aspirin.',
    }
    id_to_name = {10: 'Aspirin', 30: 'Headache'}
    with pytest.raises(AssertionError):
        audit_text_feature_contamination(
            id_to_text=id_to_text,
            id_to_name=id_to_name,
            valid_pairs=[(10, 30)],
            test_pairs=[],
        )


def test_graph_surgery_audit_raises_on_residual_shortcut_or_reverse_edge() -> None:
    data = _build_toy_no_leak_graph()
    data[('gene/protein', 'targets__reverse__', 'drug')].edge_index = torch.tensor(
        [[0], [0]], dtype=torch.long
    )
    data[('drug', 'indication', 'disease')].edge_index = torch.tensor([[0], [0]], dtype=torch.long)

    test_quads = torch.tensor([[10, 20, 40, 30]], dtype=torch.long)
    with pytest.raises(AssertionError):
        audit_graph_surgery_residuals(no_leakage_data=data, test_quads=test_quads)


def test_cross_split_purity_audit_passes_on_clean_cross_disease_case() -> None:
    data = _build_toy_no_leak_graph()
    pair_splits = {
        'train': {(10, 30)},
        'valid': {(10, 31)},
        'test': {(11, 31)},
    }
    result = audit_strict_cross_split_purity(
        split_mode='cross_disease',
        pair_splits=pair_splits,
        no_leakage_data=data,
    )
    assert result['num_entity_overlap'] == 0


def test_cross_split_purity_audit_raises_when_test_entity_still_in_graph() -> None:
    data = _build_toy_no_leak_graph()
    data[('gene/protein', 'associates_extra', 'disease')].edge_index = torch.tensor(
        [[1], [1]], dtype=torch.long
    )
    pair_splits = {
        'train': {(10, 30)},
        'valid': {(10, 31)},
        'test': {(11, 31)},
    }
    with pytest.raises(AssertionError):
        audit_strict_cross_split_purity(
            split_mode='cross_disease',
            pair_splits=pair_splits,
            no_leakage_data=data,
        )


def test_ot_pair_overlap_audit_raises_on_pair_collision() -> None:
    with pytest.raises(AssertionError):
        audit_ot_pair_level_leakage(
            train_pairs=[(10, 30), (11, 31)],
            ot_pairs=[(11, 31), (12, 32)],
        )
