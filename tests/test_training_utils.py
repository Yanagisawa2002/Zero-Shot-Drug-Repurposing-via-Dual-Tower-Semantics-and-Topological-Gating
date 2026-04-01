from __future__ import annotations

import pickle
import sys
import uuid
from pathlib import Path
from unittest.mock import patch

import torch
from torch.optim import Adam
from torch_geometric.data import HeteroData


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pair_path_bpr_sampler import build_pair_path_bpr_dataloader
from src.path_bpr_sampler import build_path_bpr_dataloader
from src.repurposing_rgcn import RepurposingRGCN
from src.training_utils import (
    _mask_direct_target_edges_for_batch,
    compute_bce_loss,
    compute_bpr_loss,
    train_epoch,
)


def _build_toy_heterodata() -> HeteroData:
    data = HeteroData()

    data['drug'].num_nodes = 3
    data['drug'].global_id = torch.tensor([10, 11, 12], dtype=torch.long)

    data['gene/protein'].num_nodes = 4
    data['gene/protein'].global_id = torch.tensor([20, 21, 22, 23], dtype=torch.long)

    data['disease'].num_nodes = 3
    data['disease'].global_id = torch.tensor([30, 31, 32], dtype=torch.long)

    data[('drug', 'targets', 'gene/protein')].edge_index = torch.tensor(
        [[0, 0, 1, 2], [0, 1, 2, 3]], dtype=torch.long
    )
    data[('gene/protein', 'targeted_by', 'drug')].edge_index = torch.tensor(
        [[0, 1, 2, 3], [0, 0, 1, 2]], dtype=torch.long
    )
    data[('gene/protein', 'associates', 'disease')].edge_index = torch.tensor(
        [[0, 1, 2, 3], [0, 1, 1, 2]], dtype=torch.long
    )
    data[('disease', 'associated_with', 'gene/protein')].edge_index = torch.tensor(
        [[0, 1, 2], [0, 2, 3]], dtype=torch.long
    )

    data.ho_pos_paths = torch.tensor(
        [
            [10, 20, 30],
            [10, 21, 30],
            [10, 21, 31],
            [11, 22, 31],
            [12, 23, 32],
        ],
        dtype=torch.long,
    )
    data.ho_path_node_types = ('drug', 'gene/protein', 'disease')
    return data


def _build_toy_heterodata_with_direct_prediction_edges() -> HeteroData:
    data = HeteroData()

    data['drug'].num_nodes = 2
    data['drug'].global_id = torch.tensor([10, 11], dtype=torch.long)

    data['gene/protein'].num_nodes = 2
    data['gene/protein'].global_id = torch.tensor([20, 21], dtype=torch.long)

    data['disease'].num_nodes = 2
    data['disease'].global_id = torch.tensor([30, 31], dtype=torch.long)

    data[('drug', 'targets', 'gene/protein')].edge_index = torch.tensor(
        [[0, 1], [0, 1]], dtype=torch.long
    )
    data[('gene/protein', 'targeted_by', 'drug')].edge_index = torch.tensor(
        [[0, 1], [0, 1]], dtype=torch.long
    )
    data[('disease', 'disease_protein', 'gene/protein')].edge_index = torch.tensor(
        [[0, 1], [0, 1]], dtype=torch.long
    )
    data[('gene/protein', 'disease_protein__reverse__', 'disease')].edge_index = torch.tensor(
        [[0, 1], [0, 1]], dtype=torch.long
    )
    data[('drug', 'indication', 'disease')].edge_index = torch.tensor(
        [[0, 1], [0, 1]], dtype=torch.long
    )
    data[('disease', 'rev_indication', 'drug')].edge_index = torch.tensor(
        [[0, 1], [0, 1]], dtype=torch.long
    )

    data.ho_pos_paths = torch.tensor(
        [
            [10, 20, 30],
            [11, 21, 31],
        ],
        dtype=torch.long,
    )
    data.ho_path_node_types = ('drug', 'gene/protein', 'disease')
    return data


def _build_sequential_toy_heterodata() -> HeteroData:
    data = HeteroData()

    data['drug'].num_nodes = 2
    data['drug'].global_id = torch.tensor([0, 1], dtype=torch.long)

    data['gene/protein'].num_nodes = 2
    data['gene/protein'].global_id = torch.tensor([2, 3], dtype=torch.long)

    data['disease'].num_nodes = 2
    data['disease'].global_id = torch.tensor([4, 5], dtype=torch.long)

    data[('drug', 'targets', 'gene/protein')].edge_index = torch.tensor(
        [[0, 1], [0, 1]], dtype=torch.long
    )
    data[('gene/protein', 'targeted_by', 'drug')].edge_index = torch.tensor(
        [[0, 1], [0, 1]], dtype=torch.long
    )
    data[('gene/protein', 'associates', 'disease')].edge_index = torch.tensor(
        [[0, 1], [0, 1]], dtype=torch.long
    )
    data[('disease', 'associated_with', 'gene/protein')].edge_index = torch.tensor(
        [[0, 1], [0, 1]], dtype=torch.long
    )

    data.ho_pos_paths = torch.tensor(
        [
            [0, 2, 4],
            [1, 3, 5],
        ],
        dtype=torch.long,
    )
    data.ho_path_node_types = ('drug', 'gene/protein', 'disease')
    return data


def test_compute_bpr_loss_matches_manual_formula() -> None:
    pos_scores = torch.tensor([2.0, 1.5, 0.2], dtype=torch.float32)
    neg_scores = torch.tensor([1.0, -0.5, 0.1], dtype=torch.float32)

    loss = compute_bpr_loss(pos_scores=pos_scores, neg_scores=neg_scores)
    expected = -(torch.nn.functional.logsigmoid(pos_scores - neg_scores)).mean()
    assert torch.allclose(loss, expected)


def test_compute_bce_loss_matches_manual_formula() -> None:
    pos_scores = torch.tensor([2.0, 1.5, 0.2], dtype=torch.float32)
    neg_scores = torch.tensor([1.0, -0.5, 0.1], dtype=torch.float32)

    loss = compute_bce_loss(pos_scores=pos_scores, neg_scores=neg_scores)
    logits = torch.cat([pos_scores, neg_scores], dim=0)
    labels = torch.cat([torch.ones_like(pos_scores), torch.zeros_like(neg_scores)], dim=0)
    expected = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels)
    assert torch.allclose(loss, expected)



def test_train_epoch_runs_and_updates_parameters_with_legacy_path_batches() -> None:
    torch.manual_seed(7)

    data = _build_toy_heterodata()
    dataloader = build_path_bpr_dataloader(
        data=data,
        batch_size=2,
        shuffle=False,
        num_workers=0,
    )
    model = RepurposingRGCN(
        data=data,
        hidden_dim=8,
        out_dim=8,
        scorer_hidden_dim=4,
        dropout=0.0,
    )
    optimizer = Adam(model.parameters(), lr=1e-2)

    before_weight = model.scorer.output_mlp[0].weight.detach().clone()
    metrics = train_epoch(
        model=model,
        full_graph_data=data,
        bpr_dataloader=dataloader,
        optimizer=optimizer,
    )
    after_weight = model.scorer.output_mlp[0].weight.detach().clone()

    assert metrics['num_examples'] == 5.0
    assert metrics['loss'] > 0.0
    assert metrics['bpr_loss'] > 0.0
    assert metrics['distill_loss'] == 0.0
    assert metrics['path_loss'] >= 0.0
    assert torch.isfinite(torch.tensor(metrics['loss']))
    assert torch.isfinite(torch.tensor(metrics['avg_pos_score']))
    assert torch.isfinite(torch.tensor(metrics['avg_neg_score']))
    assert not torch.allclose(before_weight, after_weight)



def test_train_epoch_runs_and_updates_parameters_with_pair_level_batches() -> None:
    torch.manual_seed(7)

    data = _build_toy_heterodata()
    dataloader = build_pair_path_bpr_dataloader(
        data=data,
        batch_size=2,
        shuffle=False,
        num_workers=0,
        negative_strategy='mixed',
    )
    model = RepurposingRGCN(
        data=data,
        hidden_dim=8,
        out_dim=8,
        scorer_hidden_dim=4,
        dropout=0.0,
    )
    optimizer = Adam(model.parameters(), lr=1e-2)

    before_weight = model.scorer.output_mlp[0].weight.detach().clone()
    metrics = train_epoch(
        model=model,
        full_graph_data=data,
        bpr_dataloader=dataloader,
        optimizer=optimizer,
    )
    after_weight = model.scorer.output_mlp[0].weight.detach().clone()

    assert metrics['num_examples'] == 4.0
    assert metrics['loss'] > 0.0
    assert metrics['bpr_loss'] > 0.0
    assert metrics['distill_loss'] == 0.0
    assert metrics['path_loss'] >= 0.0
    assert torch.isfinite(torch.tensor(metrics['loss']))
    assert torch.isfinite(torch.tensor(metrics['avg_pos_score']))
    assert torch.isfinite(torch.tensor(metrics['avg_neg_score']))
    assert not torch.allclose(before_weight, after_weight)



def test_train_epoch_reports_positive_distill_loss_when_teacher_is_available() -> None:
    torch.manual_seed(3)

    data = _build_sequential_toy_heterodata()
    dataloader = build_path_bpr_dataloader(
        data=data,
        batch_size=2,
        shuffle=False,
        num_workers=0,
    )

    tmp_dir = PROJECT_ROOT / 'tmp_test_artifacts' / f'training_utils_{uuid.uuid4().hex}'
    tmp_dir.mkdir(parents=True, exist_ok=True)

    nodes_csv = tmp_dir / 'nodes.csv'
    nodes_csv.write_text(
        'id\ndrug::d0\ndrug::d1\ngene/protein::g0\ngene/protein::g1\ndisease::c0\ndisease::c1\n',
        encoding='utf-8',
    )

    teacher_path = tmp_dir / 'triplet_text_embeddings.pkl'
    with teacher_path.open('wb') as file:
        pickle.dump(
            {
                ('drug::d0', 'disease::c0', 'gene/protein::g0'): torch.randn(768),
                ('drug::d1', 'disease::c1', 'gene/protein::g1'): torch.randn(768),
            },
            file,
        )

    model = RepurposingRGCN(
        data=data,
        hidden_dim=8,
        out_dim=8,
        scorer_hidden_dim=4,
        dropout=0.0,
        triplet_text_embeddings_path=teacher_path,
        nodes_csv_path=nodes_csv,
        text_distill_alpha=0.5,
    )
    optimizer = Adam(model.parameters(), lr=1e-2)

    metrics = train_epoch(
        model=model,
        full_graph_data=data,
        bpr_dataloader=dataloader,
        optimizer=optimizer,
    )

    assert metrics['num_examples'] == 2.0
    assert metrics['bpr_loss'] > 0.0
    assert metrics['distill_loss'] > 0.0
    assert metrics['loss'] > metrics['bpr_loss']


def test_train_epoch_runs_with_bce_primary_loss() -> None:
    torch.manual_seed(11)

    data = _build_toy_heterodata()
    dataloader = build_pair_path_bpr_dataloader(
        data=data,
        batch_size=2,
        shuffle=False,
        num_workers=0,
        negative_strategy='mixed',
    )
    model = RepurposingRGCN(
        data=data,
        hidden_dim=8,
        out_dim=8,
        scorer_hidden_dim=4,
        dropout=0.0,
    )
    optimizer = Adam(model.parameters(), lr=1e-2)

    metrics = train_epoch(
        model=model,
        full_graph_data=data,
        bpr_dataloader=dataloader,
        optimizer=optimizer,
        primary_loss_type='bce',
    )

    assert metrics['num_examples'] == 4.0
    assert metrics['loss'] > 0.0
    assert metrics['pair_loss'] > 0.0
    assert metrics['bce_loss'] > 0.0
    assert metrics['bpr_loss'] == 0.0
    assert metrics['distill_loss'] == 0.0
    assert metrics['path_loss'] >= 0.0


def test_train_epoch_reports_positive_path_loss_when_enabled() -> None:
    torch.manual_seed(13)

    data = _build_toy_heterodata()
    dataloader = build_pair_path_bpr_dataloader(
        data=data,
        batch_size=2,
        shuffle=False,
        num_workers=0,
        negative_strategy='mixed',
    )
    model = RepurposingRGCN(
        data=data,
        hidden_dim=8,
        out_dim=8,
        scorer_hidden_dim=4,
        dropout=0.0,
    )
    optimizer = Adam(model.parameters(), lr=1e-2)

    metrics = train_epoch(
        model=model,
        full_graph_data=data,
        bpr_dataloader=dataloader,
        optimizer=optimizer,
        path_loss_weight=0.1,
    )

    assert metrics['num_examples'] == 4.0
    assert metrics['loss'] > 0.0
    assert metrics['path_loss'] >= 0.0
    assert torch.isfinite(torch.tensor(metrics['path_loss']))


def test_mask_direct_target_edges_for_batch_removes_only_current_positive_pairs() -> None:
    data = _build_toy_heterodata_with_direct_prediction_edges()
    positive_pairs = torch.tensor([[10, 30]], dtype=torch.long)

    masked_edge_index_dict = _mask_direct_target_edges_for_batch(
        full_graph_data=data,
        edge_index_dict=data.edge_index_dict,
        positive_pairs=positive_pairs,
    )

    forward_edges = masked_edge_index_dict[('drug', 'indication', 'disease')]
    reverse_edges = masked_edge_index_dict[('disease', 'rev_indication', 'drug')]

    assert forward_edges.shape[1] == 1
    assert reverse_edges.shape[1] == 1
    assert torch.equal(forward_edges, torch.tensor([[1], [1]], dtype=torch.long))
    assert torch.equal(reverse_edges, torch.tensor([[1], [1]], dtype=torch.long))
    assert torch.equal(
        masked_edge_index_dict[('drug', 'targets', 'gene/protein')],
        data[('drug', 'targets', 'gene/protein')].edge_index,
    )


def test_train_epoch_passes_batch_masked_direct_edges_into_encode() -> None:
    torch.manual_seed(5)

    data = _build_toy_heterodata_with_direct_prediction_edges()
    dataloader = build_pair_path_bpr_dataloader(
        data=data,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        negative_strategy='mixed',
    )
    model = RepurposingRGCN(
        data=data,
        hidden_dim=8,
        out_dim=8,
        scorer_hidden_dim=4,
        dropout=0.0,
        dropedge_p=0.0,
    )
    optimizer = Adam(model.parameters(), lr=1e-2)

    with patch.object(model, 'encode', wraps=model.encode) as encode_mock:
        train_epoch(
            model=model,
            full_graph_data=data,
            bpr_dataloader=dataloader,
            optimizer=optimizer,
        )

    first_edge_index_dict = encode_mock.call_args_list[0].kwargs['edge_index_dict']
    forward_edges = first_edge_index_dict[('drug', 'indication', 'disease')]
    reverse_edges = first_edge_index_dict[('disease', 'rev_indication', 'drug')]

    assert forward_edges.shape[1] == 1
    assert reverse_edges.shape[1] == 1
    assert torch.equal(forward_edges, torch.tensor([[1], [1]], dtype=torch.long))
    assert torch.equal(reverse_edges, torch.tensor([[1], [1]], dtype=torch.long))
