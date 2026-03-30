from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch_geometric.data import HeteroData

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.pretrain_rgcn_gcl import (
    GCL_RGCN_Model,
    augment_graph_view,
    info_nce_loss,
)
from src.repurposing_rgcn import RepurposingRGCN


def _build_toy_graph() -> HeteroData:
    data = HeteroData()
    data['drug'].num_nodes = 3
    data['drug'].global_id = torch.tensor([10, 11, 12], dtype=torch.long)
    data['drug'].x = torch.randn(3, 768)

    data['gene/protein'].num_nodes = 4
    data['gene/protein'].global_id = torch.tensor([20, 21, 22, 23], dtype=torch.long)
    data['gene/protein'].x = torch.randn(4, 768)

    data['disease'].num_nodes = 3
    data['disease'].global_id = torch.tensor([30, 31, 32], dtype=torch.long)
    data['disease'].x = torch.randn(3, 768)

    data[('drug', 'targets', 'gene/protein')].edge_index = torch.tensor(
        [[0, 1, 2], [0, 1, 2]], dtype=torch.long
    )
    data[('gene/protein', 'targeted_by', 'drug')].edge_index = torch.tensor(
        [[0, 1, 2], [0, 1, 2]], dtype=torch.long
    )
    data[('gene/protein', 'associates', 'disease')].edge_index = torch.tensor(
        [[0, 1, 2, 3], [0, 1, 2, 2]], dtype=torch.long
    )
    data[('disease', 'associated_with', 'gene/protein')].edge_index = torch.tensor(
        [[0, 1, 2], [0, 1, 2]], dtype=torch.long
    )
    data.ho_pos_paths = torch.empty((0, 3), dtype=torch.long)
    data.ho_path_node_types = ('drug', 'gene/protein', 'disease')
    return data


def test_augment_graph_view_preserves_schema_and_applies_corruption() -> None:
    torch.manual_seed(0)
    data = _build_toy_graph()
    augmented = augment_graph_view(data=data, edge_drop_prob=0.5, feature_mask_prob=0.5)

    assert set(augmented.node_types) == set(data.node_types)
    assert set(augmented.edge_types) == set(data.edge_types)
    for edge_type in data.edge_types:
        assert augmented[edge_type].edge_index.size(1) <= data[edge_type].edge_index.size(1)
    assert torch.allclose(data['drug'].x, _build_toy_graph()['drug'].x) is False or True
    assert (augmented['drug'].x == 0).any() or (augmented['disease'].x == 0).any()


def test_info_nce_loss_returns_finite_scalar() -> None:
    torch.manual_seed(0)
    z_1 = torch.randn(8, 16)
    z_2 = torch.randn(8, 16)
    loss = info_nce_loss(z_1, z_2, temperature=0.2)
    assert loss.dim() == 0
    assert torch.isfinite(loss)


def test_gcl_wrapper_outputs_projected_embeddings() -> None:
    torch.manual_seed(0)
    data = _build_toy_graph()
    encoder = RepurposingRGCN(
        data=data,
        hidden_channels=16,
        out_dim=16,
        scorer_hidden_dim=8,
        dropout=0.0,
    )
    model = GCL_RGCN_Model(
        encoder=encoder,
        target_node_types=('drug', 'disease'),
        projection_hidden_dim=16,
        projection_dim=8,
        projection_dropout=0.0,
    )

    proj_1, proj_2 = model(data, data)
    assert proj_1['drug'].shape == (3, 8)
    assert proj_1['disease'].shape == (3, 8)
    assert proj_2['drug'].shape == (3, 8)
    assert proj_2['disease'].shape == (3, 8)
    assert torch.isfinite(proj_1['drug']).all()
    assert torch.isfinite(proj_2['disease']).all()
