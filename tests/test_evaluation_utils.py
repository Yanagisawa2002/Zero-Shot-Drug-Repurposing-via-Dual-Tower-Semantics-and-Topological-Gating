from __future__ import annotations

import io
import sys
from contextlib import redirect_stdout
from pathlib import Path

import torch
from torch_geometric.data import HeteroData


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation_utils import evaluate_model
from src.repurposing_rgcn import RepurposingRGCN


def _build_toy_heterodata() -> HeteroData:
    data = HeteroData()

    data['drug'].num_nodes = 4
    data['drug'].global_id = torch.tensor([10, 11, 12, 13], dtype=torch.long)

    data['gene/protein'].num_nodes = 4
    data['gene/protein'].global_id = torch.tensor([20, 21, 22, 23], dtype=torch.long)

    data['disease'].num_nodes = 4
    data['disease'].global_id = torch.tensor([30, 31, 32, 33], dtype=torch.long)

    data[('drug', 'targets', 'gene/protein')].edge_index = torch.tensor(
        [[0, 1, 2, 3], [0, 1, 2, 3]], dtype=torch.long
    )
    data[('gene/protein', 'targeted_by', 'drug')].edge_index = torch.tensor(
        [[0, 1, 2, 3], [0, 1, 2, 3]], dtype=torch.long
    )
    data[('gene/protein', 'associates', 'disease')].edge_index = torch.tensor(
        [[0, 1, 2, 3], [0, 1, 2, 3]], dtype=torch.long
    )
    data[('disease', 'associated_with', 'gene/protein')].edge_index = torch.tensor(
        [[0, 1, 2, 3], [0, 1, 2, 3]], dtype=torch.long
    )

    data.ho_pos_paths = torch.tensor(
        [
            [10, 20, 30],
            [11, 21, 31],
            [12, 22, 32],
        ],
        dtype=torch.long,
    )
    data.ho_path_node_types = ('drug', 'gene/protein', 'disease')
    return data


def test_evaluate_model_returns_all_pair_level_metrics() -> None:
    torch.manual_seed(0)

    data = _build_toy_heterodata()
    valid_ho_paths = torch.tensor(
        [
            [13, 23, 33],
            [13, 20, 33],
            [10, 21, 32],
        ],
        dtype=torch.long,
    )
    model = RepurposingRGCN(
        data=data,
        hidden_dim=8,
        out_dim=8,
        scorer_hidden_dim=4,
        dropout=0.0,
    )

    stdout_buffer = io.StringIO()
    with redirect_stdout(stdout_buffer):
        results = evaluate_model(
            model=model,
            data=data,
            valid_ho_paths=valid_ho_paths,
            batch_size=1,
            verbose=True,
        )
    printed_text = stdout_buffer.getvalue()

    assert 'Pair-Level Multi-Path Evaluation Results' in printed_text
    assert set(results.keys()) == {'random', 'cross_drug', 'cross_disease'}

    for setting_name, metrics in results.items():
        assert set(metrics.keys()) == {'pairwise_accuracy', 'auroc', 'auprc'}
        assert setting_name in printed_text
        for metric_value in metrics.values():
            assert 0.0 <= metric_value <= 1.0


def test_evaluate_model_restores_training_mode_after_eval() -> None:
    torch.manual_seed(0)

    data = _build_toy_heterodata()
    valid_ho_paths = torch.tensor(
        [
            [13, 23, 33],
            [10, 21, 32],
        ],
        dtype=torch.long,
    )
    model = RepurposingRGCN(
        data=data,
        hidden_dim=8,
        out_dim=8,
        scorer_hidden_dim=4,
        dropout=0.1,
    )
    model.train()

    results = evaluate_model(
        model=model,
        data=data,
        valid_ho_paths=valid_ho_paths,
        batch_size=2,
        verbose=False,
    )

    assert model.training is True
    assert set(results.keys()) == {'random', 'cross_drug', 'cross_disease'}
