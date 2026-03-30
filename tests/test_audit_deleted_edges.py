from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torch_geometric.data import HeteroData

from scripts.audit_deleted_edges import (
    DIRECT_LEAKAGE_LABEL,
    MECHANISTIC_LABEL,
    build_deletion_dataframe,
    count_edges_by_type,
)
from src.graph_surgery import remove_leakage_edges


def build_toy_graph() -> HeteroData:
    data = HeteroData()
    data['drug'].global_id = torch.tensor([100, 101], dtype=torch.long)
    data['drug'].num_nodes = 2
    data['gene/protein'].global_id = torch.tensor([200, 201], dtype=torch.long)
    data['gene/protein'].num_nodes = 2
    data['pathway'].global_id = torch.tensor([300, 301], dtype=torch.long)
    data['pathway'].num_nodes = 2
    data['disease'].global_id = torch.tensor([400, 401], dtype=torch.long)
    data['disease'].num_nodes = 2

    data[('drug', 'targets', 'gene/protein')].edge_index = torch.tensor(
        [[0, 1], [0, 1]], dtype=torch.long
    )
    data[('disease', 'disease_protein', 'gene/protein')].edge_index = torch.tensor(
        [[0, 1], [0, 1]], dtype=torch.long
    )
    data[('drug', 'indication', 'disease')].edge_index = torch.tensor(
        [[0, 1], [0, 1]], dtype=torch.long
    )
    data[('gene/protein', 'in_pathway', 'pathway')].edge_index = torch.tensor(
        [[0, 1], [0, 1]], dtype=torch.long
    )
    data[('drug', 'drug_drug', 'drug')].edge_index = torch.tensor(
        [[0], [1]], dtype=torch.long
    )
    data.quad_path_node_types = ('drug', 'gene/protein', 'pathway', 'disease')
    return data


def test_build_deletion_dataframe_separates_direct_and_mechanistic_edges() -> None:
    data = build_toy_graph()
    target_paths = torch.tensor([[100, 200, 300, 400]], dtype=torch.long)

    clean_edge_index_dict = remove_leakage_edges(data=data, target_paths=target_paths)
    raw_counts = count_edges_by_type(data.edge_index_dict)
    clean_counts = count_edges_by_type(clean_edge_index_dict)
    diff = build_deletion_dataframe(raw_counts=raw_counts, clean_counts=clean_counts)

    assert not diff.empty
    assert set(diff['edge_type']) == {
        "('disease', 'disease_protein', 'gene/protein')",
        "('drug', 'indication', 'disease')",
        "('drug', 'targets', 'gene/protein')",
        "('gene/protein', 'in_pathway', 'pathway')",
    }

    direct_rows = diff[diff['category'] == DIRECT_LEAKAGE_LABEL]
    mechanistic_rows = diff[diff['category'] == MECHANISTIC_LABEL]

    assert len(direct_rows) == 1
    assert int(direct_rows.iloc[0]['deleted_count']) == 1
    assert direct_rows.iloc[0]['edge_type'] == "('drug', 'indication', 'disease')"

    assert len(mechanistic_rows) == 3
    assert int(mechanistic_rows['deleted_count'].sum()) == 3
