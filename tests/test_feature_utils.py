from __future__ import annotations

import shutil
import sys
import uuid
import warnings
from pathlib import Path

import torch
from torch_geometric.data import HeteroData


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.feature_utils import inject_features_to_graph


def _build_toy_graph() -> HeteroData:
    data = HeteroData()
    data['drug'].num_nodes = 2
    data['drug'].global_id = torch.tensor([10, 11], dtype=torch.long)
    data['gene/protein'].num_nodes = 3
    data['gene/protein'].global_id = torch.tensor([20, 21, 22], dtype=torch.long)
    data['disease'].num_nodes = 2
    data['disease'].global_id = torch.tensor([30, 31], dtype=torch.long)
    return data


def _make_workspace_temp_dir() -> Path:
    temp_dir = PROJECT_ROOT / f'feature_utils_{uuid.uuid4().hex}'
    temp_dir.mkdir(parents=True, exist_ok=False)
    return temp_dir


def test_inject_features_to_graph_loads_existing_feature_files() -> None:
    data = _build_toy_graph()
    temp_dir = _make_workspace_temp_dir()
    try:
        torch.save(torch.randn(2, 768), temp_dir / 'drug_features.pt')
        torch.save(torch.randn(3, 768), temp_dir / 'gene_protein_features.pt')
        torch.save(torch.randn(2, 768), temp_dir / 'disease_features.pt')

        inject_features_to_graph(data=data, feature_dir=temp_dir)

        assert data['drug'].x.shape == (2, 768)
        assert data['gene/protein'].x.shape == (3, 768)
        assert data['disease'].x.shape == (2, 768)
        assert data['drug'].x.dtype == torch.float32
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_inject_features_to_graph_warns_when_feature_file_is_missing() -> None:
    data = _build_toy_graph()
    temp_dir = _make_workspace_temp_dir()
    try:
        torch.save(torch.randn(2, 768), temp_dir / 'drug_features.pt')
        torch.save(torch.randn(2, 768), temp_dir / 'disease_features.pt')

        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter('always')
            inject_features_to_graph(data=data, feature_dir=temp_dir)

        warning_messages = [str(item.message) for item in caught_warnings]
        assert any('gene/protein' in message for message in warning_messages)
        assert 'x' not in data['gene/protein']
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
