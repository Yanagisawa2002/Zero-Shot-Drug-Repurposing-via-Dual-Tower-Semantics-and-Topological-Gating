from __future__ import annotations

import pickle
import sys
from pathlib import Path
from unittest.mock import patch

import torch
from torch import nn
from torch_geometric.data import HeteroData


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.repurposing_rgcn import RepurposingRGCN


def _build_toy_heterodata() -> HeteroData:
    data = HeteroData()

    data['drug'].num_nodes = 2
    data['drug'].global_id = torch.tensor([10, 11], dtype=torch.long)

    data['gene/protein'].num_nodes = 3
    data['gene/protein'].global_id = torch.tensor([20, 21, 22], dtype=torch.long)

    data['disease'].num_nodes = 2
    data['disease'].global_id = torch.tensor([30, 31], dtype=torch.long)

    data[('drug', 'targets', 'gene/protein')].edge_index = torch.tensor(
        [[0, 1, 1], [0, 1, 2]], dtype=torch.long
    )
    data[('gene/protein', 'targeted_by', 'drug')].edge_index = torch.tensor(
        [[0, 1, 2], [0, 1, 1]], dtype=torch.long
    )
    data[('gene/protein', 'associates', 'disease')].edge_index = torch.tensor(
        [[0, 1, 2], [0, 1, 1]], dtype=torch.long
    )
    data[('disease', 'associated_with', 'gene/protein')].edge_index = torch.tensor(
        [[0, 1], [0, 1]], dtype=torch.long
    )

    data.ho_pos_paths = torch.tensor(
        [[10, 20, 30], [10, 21, 30], [11, 22, 31]],
        dtype=torch.long,
    )
    data.ho_path_node_types = ('drug', 'gene/protein', 'disease')
    return data


def _build_toy_heterodata_with_pathways() -> HeteroData:
    data = _build_toy_heterodata()
    data['pathway'].num_nodes = 2
    data['pathway'].global_id = torch.tensor([40, 41], dtype=torch.long)

    data[('gene/protein', 'participates_in', 'pathway')].edge_index = torch.tensor(
        [[0, 2], [0, 1]], dtype=torch.long
    )
    data[('pathway', 'participates_in__reverse__', 'gene/protein')].edge_index = torch.tensor(
        [[0, 1], [0, 2]], dtype=torch.long
    )
    return data


def test_repurposing_rgcn_encode_and_legacy_forward() -> None:
    data = _build_toy_heterodata()
    model = RepurposingRGCN(data=data, hidden_dim=8, out_dim=8, scorer_hidden_dim=4, dropout=0.0)

    node_embs_dict, path_logits = model(
        x_dict=None,
        edge_index_dict=data.edge_index_dict,
        path_tensor=data.ho_pos_paths[:2],
        return_node_embs=True,
    )

    assert set(node_embs_dict.keys()) == {'drug', 'gene/protein', 'disease'}
    assert node_embs_dict['drug'].shape == (2, 8)
    assert node_embs_dict['gene/protein'].shape == (3, 8)
    assert node_embs_dict['disease'].shape == (2, 8)
    assert path_logits.shape == (2,)
    assert torch.is_floating_point(path_logits)


def test_repurposing_rgcn_encode_with_external_pubmedbert_features() -> None:
    data = _build_toy_heterodata()
    data['drug'].x = torch.randn(2, 768)
    data['gene/protein'].x = torch.randn(3, 768)
    data['disease'].x = torch.randn(2, 768)

    model = RepurposingRGCN(
        data=data,
        hidden_channels=8,
        in_channels=768,
        out_dim=8,
        scorer_hidden_dim=4,
        dropout=0.0,
    )
    node_embs_dict = model.encode(x_dict=data.x_dict, edge_index_dict=data.edge_index_dict)

    assert model.feature_projections['drug'].in_features == 768
    assert model.feature_projections['drug'].out_features == 8
    assert node_embs_dict['drug'].shape == (2, 8)
    assert node_embs_dict['gene/protein'].shape == (3, 8)
    assert node_embs_dict['disease'].shape == (2, 8)
    assert torch.isfinite(node_embs_dict['drug']).all()


def test_repurposing_rgcn_score_batch_with_dummy_paths() -> None:
    data = _build_toy_heterodata()
    model = RepurposingRGCN(data=data, hidden_dim=8, out_dim=6, scorer_hidden_dim=5, dropout=0.0)
    node_embs_dict = model.encode(x_dict=None, edge_index_dict=data.edge_index_dict)

    pair_ids = torch.tensor([[10, 30], [11, 31]], dtype=torch.long)
    paths = torch.tensor(
        [
            [[10, 20, 30], [10, 21, 30]],
            [[0, 0, 0], [0, 0, 0]],
        ],
        dtype=torch.long,
    )
    attention_mask = torch.tensor([[True, True], [False, False]], dtype=torch.bool)

    logits, attention_weights = model.score_batch(
        node_embs_dict=node_embs_dict,
        pair_ids=pair_ids,
        paths=paths,
        attention_mask=attention_mask,
        return_attention=True,
    )

    assert logits.shape == (2,)
    assert attention_weights.shape == (2, 2)
    assert torch.isfinite(logits).all()
    assert torch.isfinite(attention_weights).all()
    assert torch.allclose(attention_weights[0].sum(), torch.tensor(1.0), atol=1e-6)
    assert torch.allclose(attention_weights[1], torch.zeros(2), atol=1e-6)


def test_repurposing_rgcn_pair_level_forward_returns_pos_and_neg_scores() -> None:
    data = _build_toy_heterodata()
    model = RepurposingRGCN(data=data, hidden_dim=8, out_dim=8, scorer_hidden_dim=4, dropout=0.0)

    pos_pair_ids = torch.tensor([[10, 30], [11, 31]], dtype=torch.long)
    pos_paths = torch.tensor(
        [
            [[10, 20, 30], [10, 21, 30]],
            [[11, 22, 31], [0, 0, 0]],
        ],
        dtype=torch.long,
    )
    pos_attention_mask = torch.tensor([[True, True], [True, False]], dtype=torch.bool)

    neg_pair_ids = torch.tensor([[10, 31], [11, 30]], dtype=torch.long)
    neg_paths = torch.tensor(
        [
            [[0, 0, 0], [0, 0, 0]],
            [[0, 0, 0], [0, 0, 0]],
        ],
        dtype=torch.long,
    )
    neg_attention_mask = torch.tensor([[False, False], [False, False]], dtype=torch.bool)

    pos_scores, neg_scores = model(
        x_dict=None,
        edge_index_dict=data.edge_index_dict,
        pos_pair_ids=pos_pair_ids,
        pos_paths=pos_paths,
        pos_attention_mask=pos_attention_mask,
        neg_pair_ids=neg_pair_ids,
        neg_paths=neg_paths,
        neg_attention_mask=neg_attention_mask,
    )

    assert pos_scores.shape == (2,)
    assert neg_scores.shape == (2,)
    assert torch.is_floating_point(pos_scores)
    assert torch.is_floating_point(neg_scores)
    assert torch.isfinite(pos_scores).all()
    assert torch.isfinite(neg_scores).all()


def test_score_paths_rejects_invalid_global_ids() -> None:
    data = _build_toy_heterodata()
    model = RepurposingRGCN(data=data, hidden_dim=8, out_dim=8, scorer_hidden_dim=4, dropout=0.0)
    node_embs_dict = model.encode(x_dict=None, edge_index_dict=data.edge_index_dict)

    invalid_paths = torch.tensor([[999, 20, 30]], dtype=torch.long)
    try:
        model.score_paths(node_embs_dict=node_embs_dict, path_tensor=invalid_paths)
        raise AssertionError('?????? IndexError ? KeyError?')
    except (IndexError, KeyError):
        pass


def test_repurposing_rgcn_quad_score_batch_with_dummy_pathways() -> None:
    data = _build_toy_heterodata_with_pathways()
    model = RepurposingRGCN(
        data=data,
        hidden_dim=8,
        out_dim=6,
        scorer_hidden_dim=5,
        dropout=0.0,
        use_pathway_quads=True,
    )
    node_embs_dict = model.encode(x_dict=None, edge_index_dict=data.edge_index_dict)

    assert model.scorer.path_value_proj.in_features == 4 * model.out_dim

    pair_ids = torch.tensor([[10, 30], [11, 31]], dtype=torch.long)
    paths = torch.tensor(
        [
            [[10, 20, 40, 30], [10, 21, 0, 30]],
            [[11, 22, 41, 31], [0, 0, 0, 0]],
        ],
        dtype=torch.long,
    )
    attention_mask = torch.tensor([[True, True], [True, False]], dtype=torch.bool)

    logits, attention_weights = model.score_batch(
        node_embs_dict=node_embs_dict,
        pair_ids=pair_ids,
        paths=paths,
        attention_mask=attention_mask,
        return_attention=True,
    )

    assert logits.shape == (2,)
    assert attention_weights.shape == (2, 2)
    assert torch.isfinite(logits).all()
    assert torch.isfinite(attention_weights).all()
    assert torch.allclose(attention_weights[0].sum(), torch.tensor(1.0), atol=1e-6)


def test_repurposing_rgcn_quad_forward_returns_pos_and_neg_scores() -> None:
    data = _build_toy_heterodata_with_pathways()
    model = RepurposingRGCN(
        data=data,
        hidden_dim=8,
        out_dim=8,
        scorer_hidden_dim=4,
        dropout=0.0,
        use_pathway_quads=True,
    )

    pos_pair_ids = torch.tensor([[10, 30], [11, 31]], dtype=torch.long)
    pos_paths = torch.tensor(
        [
            [[10, 20, 40, 30], [10, 21, 0, 30]],
            [[11, 22, 41, 31], [0, 0, 0, 0]],
        ],
        dtype=torch.long,
    )
    pos_attention_mask = torch.tensor([[True, True], [True, False]], dtype=torch.bool)

    neg_pair_ids = torch.tensor([[10, 31], [11, 30]], dtype=torch.long)
    neg_paths = torch.tensor(
        [
            [[0, 0, 0, 0], [0, 0, 0, 0]],
            [[0, 0, 0, 0], [0, 0, 0, 0]],
        ],
        dtype=torch.long,
    )
    neg_attention_mask = torch.tensor([[False, False], [False, False]], dtype=torch.bool)

    pos_scores, neg_scores = model(
        x_dict=None,
        edge_index_dict=data.edge_index_dict,
        pos_pair_ids=pos_pair_ids,
        pos_paths=pos_paths,
        pos_attention_mask=pos_attention_mask,
        neg_pair_ids=neg_pair_ids,
        neg_paths=neg_paths,
        neg_attention_mask=neg_attention_mask,
    )

    assert pos_scores.shape == (2,)
    assert neg_scores.shape == (2,)
    assert torch.isfinite(pos_scores).all()
    assert torch.isfinite(neg_scores).all()


def test_repurposing_rgcn_mlp_encoder_ignores_graph_edges() -> None:
    torch.manual_seed(0)

    data = _build_toy_heterodata()
    data['drug'].x = torch.randn(2, 768)
    data['gene/protein'].x = torch.randn(3, 768)
    data['disease'].x = torch.randn(2, 768)

    model = RepurposingRGCN(
        data=data,
        hidden_channels=8,
        in_channels=768,
        out_dim=6,
        scorer_hidden_dim=4,
        dropout=0.0,
        encoder_type='mlp',
    )

    empty_edge_index_dict = {
        edge_type: torch.empty((2, 0), dtype=torch.long)
        for edge_type in data.edge_index_dict
    }

    node_embs_with_edges = model.encode(x_dict=data.x_dict, edge_index_dict=data.edge_index_dict)
    node_embs_without_edges = model.encode(x_dict=data.x_dict, edge_index_dict=empty_edge_index_dict)

    assert model.conv1 is None
    assert model.conv2 is None
    assert node_embs_with_edges['drug'].shape == (2, 6)
    assert node_embs_with_edges['gene/protein'].shape == (3, 6)
    assert node_embs_with_edges['disease'].shape == (2, 6)
    assert torch.allclose(node_embs_with_edges['drug'], node_embs_without_edges['drug'])
    assert torch.allclose(node_embs_with_edges['gene/protein'], node_embs_without_edges['gene/protein'])
    assert torch.allclose(node_embs_with_edges['disease'], node_embs_without_edges['disease'])


def test_repurposing_rgcn_supports_mean_aggregation_ablation() -> None:
    data = _build_toy_heterodata()
    model = RepurposingRGCN(
        data=data,
        hidden_dim=8,
        out_dim=8,
        scorer_hidden_dim=4,
        dropout=0.0,
        agg_type='mean',
    )

    pos_pair_ids = torch.tensor([[10, 30], [11, 31]], dtype=torch.long)
    pos_paths = torch.tensor(
        [
            [[10, 20, 30], [10, 21, 30]],
            [[11, 22, 31], [0, 0, 0]],
        ],
        dtype=torch.long,
    )
    pos_attention_mask = torch.tensor([[True, True], [True, False]], dtype=torch.bool)

    logits = model(
        x_dict=None,
        edge_index_dict=data.edge_index_dict,
        pos_pair_ids=pos_pair_ids,
        pos_paths=pos_paths,
        pos_attention_mask=pos_attention_mask,
    )

    assert model.scorer.agg_type == 'mean'
    assert logits.shape == (2,)
    assert torch.isfinite(logits).all()


def test_repurposing_rgcn_rejects_invalid_encoder_type() -> None:
    data = _build_toy_heterodata()
    try:
        RepurposingRGCN(
            data=data,
            hidden_dim=8,
            out_dim=8,
            scorer_hidden_dim=4,
            dropout=0.0,
            encoder_type='cnn',
        )
        raise AssertionError('??????? encoder_type ???')
    except ValueError:
        pass



def test_repurposing_rgcn_initial_residual_alpha_starts_at_point_two() -> None:
    data = _build_toy_heterodata()
    model = RepurposingRGCN(
        data=data,
        hidden_dim=8,
        out_dim=6,
        scorer_hidden_dim=4,
        dropout=0.0,
        initial_residual_alpha=0.2,
    )

    assert torch.allclose(torch.sigmoid(model.conv1_alpha_logit), torch.tensor(0.2), atol=1e-6)
    assert torch.allclose(torch.sigmoid(model.conv2_alpha_logit), torch.tensor(0.2), atol=1e-6)


def test_repurposing_rgcn_initial_residual_projection_matches_target_dims() -> None:
    data = _build_toy_heterodata()
    model = RepurposingRGCN(
        data=data,
        hidden_dim=8,
        out_dim=6,
        scorer_hidden_dim=4,
        dropout=0.0,
    )

    assert isinstance(model.initial_residual_projections_conv1['drug'], nn.Identity)
    assert isinstance(model.initial_residual_projections_conv2['drug'], nn.Linear)
    assert model.initial_residual_projections_conv2['drug'].in_features == 8
    assert model.initial_residual_projections_conv2['drug'].out_features == 6



def test_repurposing_rgcn_early_external_fusion_overwrites_disease_inputs_only() -> None:
    data = HeteroData()
    data['drug'].num_nodes = 2
    data['drug'].global_id = torch.tensor([0, 1], dtype=torch.long)
    data['gene/protein'].num_nodes = 2
    data['gene/protein'].global_id = torch.tensor([2, 3], dtype=torch.long)
    data['disease'].num_nodes = 2
    data['disease'].global_id = torch.tensor([4, 5], dtype=torch.long)

    data[('drug', 'targets', 'gene/protein')].edge_index = torch.tensor([[0, 1], [0, 1]], dtype=torch.long)
    data[('gene/protein', 'targeted_by', 'drug')].edge_index = torch.tensor([[0, 1], [0, 1]], dtype=torch.long)
    data[('gene/protein', 'associates', 'disease')].edge_index = torch.tensor([[0, 1], [0, 1]], dtype=torch.long)
    data[('disease', 'associated_with', 'gene/protein')].edge_index = torch.tensor([[0, 1], [0, 1]], dtype=torch.long)
    data.ho_pos_paths = torch.tensor([[0, 2, 4], [1, 3, 5]], dtype=torch.long)
    data.ho_path_node_types = ('drug', 'gene/protein', 'disease')

    tmp_dir = PROJECT_ROOT / 'tmp_test_artifacts' / f'rgcn_early_fusion_{Path(__file__).stem}'
    tmp_dir.mkdir(parents=True, exist_ok=True)
    nodes_csv = tmp_dir / 'nodes.csv'
    nodes_csv.write_text(
        'id\n'
        'drug::d0\n'
        'drug::d1\n'
        'gene/protein::g0\n'
        'gene/protein::g1\n'
        'disease::c0\n'
        'disease::c1\n',
        encoding='utf-8',
    )

    fp_path = tmp_dir / 'drug.pkl'
    disease_path = tmp_dir / 'disease.pkl'
    with fp_path.open('wb') as file:
        pickle.dump({'drug::d0': torch.ones(1024)}, file)
    with disease_path.open('wb') as file:
        pickle.dump({'disease::c0': torch.full((768,), 2.0)}, file)

    model = RepurposingRGCN(
        data=data,
        hidden_dim=8,
        out_dim=8,
        scorer_hidden_dim=4,
        dropout=0.0,
        use_early_external_fusion=True,
        triplet_text_embeddings_path=None,
        drug_morgan_fingerprints_path=fp_path,
        disease_text_embeddings_path=disease_path,
        nodes_csv_path=nodes_csv,
    )

    with torch.no_grad():
        model.node_embeddings['drug'].weight.zero_()
        model.node_embeddings['gene/protein'].weight.zero_()
        model.node_embeddings['disease'].weight.zero_()
        model.disease_proj.weight.zero_()
        model.disease_proj.bias.fill_(4.0)
        model.disease_external_norm = nn.Identity()

    model.eval()
    prepared = model._prepare_input_features(x_dict=None)

    assert not hasattr(model, 'drug_proj')
    assert not hasattr(model, 'gene_proj')
    assert model.scorer.use_external_late_fusion is True
    assert torch.allclose(prepared['drug'][0], torch.zeros(8))
    assert torch.allclose(prepared['drug'][1], torch.zeros(8))
    assert torch.allclose(prepared['gene/protein'][0], torch.zeros(8))
    assert torch.allclose(prepared['gene/protein'][1], torch.zeros(8))
    assert torch.allclose(prepared['disease'][0], torch.full((8,), 4.0))
    assert torch.allclose(prepared['disease'][1], torch.zeros(8))


def test_repurposing_rgcn_applies_dropedge_only_during_training() -> None:
    data = _build_toy_heterodata()
    model = RepurposingRGCN(data=data, hidden_dim=8, out_dim=8, scorer_hidden_dim=4, dropout=0.0)

    call_counter = {'count': 0}

    def _fake_dropout_edge(edge_index, p, training):
        call_counter['count'] += 1
        assert p == 0.15
        assert training is True
        return edge_index, torch.ones(edge_index.size(1), dtype=torch.bool, device=edge_index.device)

    model.train()
    with patch('src.repurposing_rgcn.dropout_edge', side_effect=_fake_dropout_edge):
        node_embs = model.encode(x_dict=None, edge_index_dict=data.edge_index_dict)

    assert call_counter['count'] == len(data.edge_types)
    assert set(node_embs.keys()) == {'drug', 'gene/protein', 'disease'}

    call_counter['count'] = 0
    model.eval()
    with patch('src.repurposing_rgcn.dropout_edge', side_effect=_fake_dropout_edge):
        model.encode(x_dict=None, edge_index_dict=data.edge_index_dict)

    assert call_counter['count'] == 0
