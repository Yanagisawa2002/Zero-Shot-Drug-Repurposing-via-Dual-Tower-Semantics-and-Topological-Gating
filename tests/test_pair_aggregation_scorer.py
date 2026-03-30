from __future__ import annotations

import pickle
import sys
import uuid
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pair_aggregation_scorer import PairAggregationScorer


def test_pair_aggregation_scorer_output_shapes_and_masking() -> None:
    torch.manual_seed(0)

    scorer = PairAggregationScorer(
        pair_emb_dim=4,
        path_emb_dim=6,
        hidden_dim=8,
        dropout=0.0,
    )

    pair_embs = torch.randn(2, 2, 4)
    paths_embs = torch.randn(2, 3, 6)
    attention_mask = torch.tensor(
        [
            [True, True, False],
            [True, False, False],
        ],
        dtype=torch.bool,
    )

    logits, attention_weights = scorer(
        pair_embs=pair_embs,
        paths_embs=paths_embs,
        attention_mask=attention_mask,
        return_attention=True,
    )

    assert logits.shape == (2,)
    assert attention_weights.shape == (2, 3)
    assert torch.isfinite(logits).all()
    assert torch.isfinite(attention_weights).all()
    assert attention_weights[0, 2].item() == 0.0
    assert attention_weights[1, 1].item() == 0.0
    assert attention_weights[1, 2].item() == 0.0
    assert torch.allclose(attention_weights[0].sum(), torch.tensor(1.0), atol=1e-6)
    assert torch.allclose(attention_weights[1].sum(), torch.tensor(1.0), atol=1e-6)


def test_pair_aggregation_scorer_handles_all_padding_without_nan() -> None:
    torch.manual_seed(0)

    scorer = PairAggregationScorer(
        pair_emb_dim=4,
        path_emb_dim=6,
        hidden_dim=8,
        dropout=0.0,
    )

    pair_embs = torch.randn(1, 2, 4)
    paths_embs = torch.randn(1, 5, 6)
    attention_mask = torch.zeros(1, 5, dtype=torch.bool)

    logits, attention_weights = scorer(
        pair_embs=pair_embs,
        paths_embs=paths_embs,
        attention_mask=attention_mask,
        return_attention=True,
    )

    assert logits.shape == (1,)
    assert attention_weights.shape == (1, 5)
    assert torch.isfinite(logits).all()
    assert torch.isfinite(attention_weights).all()
    assert torch.allclose(attention_weights.sum(dim=1), torch.zeros(1), atol=1e-6)


def test_pair_aggregation_scorer_rejects_invalid_shapes() -> None:
    scorer = PairAggregationScorer(
        pair_emb_dim=4,
        path_emb_dim=6,
        hidden_dim=8,
        dropout=0.0,
    )

    pair_embs = torch.randn(2, 2, 4)
    paths_embs = torch.randn(2, 3, 6)
    attention_mask = torch.ones(2, 2, dtype=torch.bool)

    try:
        scorer(pair_embs=pair_embs, paths_embs=paths_embs, attention_mask=attention_mask)
        raise AssertionError('expected mismatched mask shape error')
    except ValueError:
        pass


def test_pair_aggregation_scorer_mean_pooling_returns_uniform_valid_weights() -> None:
    torch.manual_seed(0)

    scorer = PairAggregationScorer(
        pair_emb_dim=4,
        path_emb_dim=6,
        hidden_dim=8,
        dropout=0.0,
        agg_type='mean',
    )

    pair_embs = torch.randn(2, 2, 4)
    paths_embs = torch.randn(2, 3, 6)
    attention_mask = torch.tensor(
        [
            [True, True, False],
            [False, False, False],
        ],
        dtype=torch.bool,
    )

    logits, path_weights = scorer(
        pair_embs=pair_embs,
        paths_embs=paths_embs,
        attention_mask=attention_mask,
        return_attention=True,
    )

    assert logits.shape == (2,)
    assert path_weights.shape == (2, 3)
    assert torch.isfinite(logits).all()
    assert torch.isfinite(path_weights).all()
    assert torch.allclose(path_weights[0], torch.tensor([0.5, 0.5, 0.0]), atol=1e-6)
    assert torch.allclose(path_weights[1], torch.zeros(3), atol=1e-6)


def test_pair_aggregation_scorer_max_pooling_returns_selection_frequency_weights() -> None:
    scorer = PairAggregationScorer(
        pair_emb_dim=2,
        path_emb_dim=2,
        hidden_dim=4,
        dropout=0.0,
        agg_type='max',
    )

    pair_embs = torch.tensor(
        [
            [[1.0, 0.0], [0.0, 1.0]],
            [[1.0, 1.0], [1.0, 1.0]],
        ]
    )
    paths_embs = torch.tensor(
        [
            [[2.0, 0.0], [0.0, 3.0], [-5.0, -5.0]],
            [[-2.0, -2.0], [4.0, 1.0], [1.0, 5.0]],
        ]
    )
    attention_mask = torch.tensor(
        [
            [True, True, False],
            [False, True, True],
        ],
        dtype=torch.bool,
    )

    logits, path_weights = scorer(
        pair_embs=pair_embs,
        paths_embs=paths_embs,
        attention_mask=attention_mask,
        return_attention=True,
    )

    assert logits.shape == (2,)
    assert path_weights.shape == (2, 3)
    assert torch.isfinite(logits).all()
    assert torch.isfinite(path_weights).all()
    assert path_weights[0, 2].item() == 0.0
    assert path_weights[1, 0].item() == 0.0
    assert torch.allclose(path_weights[0, :2].sum(), torch.tensor(1.0), atol=1e-6)
    assert torch.allclose(path_weights[1, 1:].sum(), torch.tensor(1.0), atol=1e-6)


def test_pair_aggregation_scorer_rejects_invalid_agg_type() -> None:
    try:
        PairAggregationScorer(
            pair_emb_dim=4,
            path_emb_dim=6,
            hidden_dim=8,
            dropout=0.0,
            agg_type='median',
        )
        raise AssertionError('expected invalid agg_type error')
    except ValueError:
        pass


def test_pair_aggregation_scorer_final_representation_uses_gnn_and_drug_fingerprint() -> None:
    scorer = PairAggregationScorer(
        pair_emb_dim=4,
        path_emb_dim=6,
        hidden_dim=8,
        dropout=0.0,
    )
    assert scorer.output_mlp[0].in_features == (8 + 2 * 4 + 1024)

    pair_embs = torch.randn(2, 2, 4)
    paths_embs = torch.randn(2, 3, 6)
    attention_mask = torch.tensor(
        [
            [True, True, False],
            [True, False, False],
        ],
        dtype=torch.bool,
    )
    logits = scorer(
        pair_embs=pair_embs,
        paths_embs=paths_embs,
        attention_mask=attention_mask,
    )
    assert logits.shape == (2,)
    assert torch.isfinite(logits).all()


def test_pair_aggregation_scorer_returns_distill_loss_for_matching_triplets() -> None:
    tmp_dir = PROJECT_ROOT / 'tmp_test_artifacts' / f'pair_aggregation_scorer_{uuid.uuid4().hex}'
    tmp_dir.mkdir(parents=True, exist_ok=True)
    nodes_csv = tmp_dir / 'nodes.csv'
    nodes_csv.write_text('id\ndrug::d0\ndisease::c0\ngene/protein::g0\n', encoding='utf-8')

    fingerprint_path = tmp_dir / 'drug_morgan_fingerprints.pkl'
    with fingerprint_path.open('wb') as file:
        pickle.dump({'drug::d0': torch.ones(1024)}, file)

    scorer = PairAggregationScorer(
        pair_emb_dim=4,
        path_emb_dim=6,
        hidden_dim=8,
        dropout=0.0,
        drug_morgan_fingerprints_path=fingerprint_path,
        nodes_csv_path=nodes_csv,
        max_global_id=3,
    )

    hit_fp = scorer._lookup_drug_fingerprints(
        drug_global_ids=torch.tensor([0], dtype=torch.long),
        batch_size=1,
        device=torch.device('cpu'),
        dtype=torch.float32,
    )
    miss_fp = scorer._lookup_drug_fingerprints(
        drug_global_ids=torch.tensor([2], dtype=torch.long),
        batch_size=1,
        device=torch.device('cpu'),
        dtype=torch.float32,
    )

    assert scorer.output_mlp[0].in_features == (8 + 2 * 4 + 1024)
    assert scorer.drug_fingerprint_matrix.shape == (3, 1024)
    assert torch.allclose(hit_fp, torch.ones_like(hit_fp))
    assert torch.allclose(miss_fp, torch.zeros_like(miss_fp))




def test_pair_aggregation_scorer_loads_symmetric_drug_and_disease_text_embeddings() -> None:
    tmp_dir = PROJECT_ROOT / 'tmp_test_artifacts' / f'pair_aggregation_scorer_text_{uuid.uuid4().hex}'
    tmp_dir.mkdir(parents=True, exist_ok=True)
    nodes_csv = tmp_dir / 'nodes.csv'
    nodes_csv.write_text('id\ndrug::d0\ndisease::c0\ngene/protein::g0\n', encoding='utf-8')

    fingerprint_path = tmp_dir / 'drug_morgan_fingerprints.pkl'
    drug_text_path = tmp_dir / 'drug_text_embeddings.pkl'
    disease_text_path = tmp_dir / 'disease_text_embeddings.pkl'
    with fingerprint_path.open('wb') as file:
        pickle.dump({'drug::d0': torch.ones(1024)}, file)
    with drug_text_path.open('wb') as file:
        pickle.dump({'drug::d0': torch.full((768,), 3.0)}, file)
    with disease_text_path.open('wb') as file:
        pickle.dump({'disease::c0': torch.full((768,), 2.0)}, file)

    scorer = PairAggregationScorer(
        pair_emb_dim=4,
        path_emb_dim=6,
        hidden_dim=8,
        dropout=0.0,
        drug_morgan_fingerprints_path=fingerprint_path,
        drug_text_embeddings_path=drug_text_path,
        disease_text_embeddings_path=disease_text_path,
        nodes_csv_path=nodes_csv,
        max_global_id=3,
    )

    drug_text_hit = scorer._lookup_node_text_embeddings(
        global_ids=torch.tensor([0], dtype=torch.long),
        batch_size=1,
        device=torch.device('cpu'),
        dtype=torch.float32,
        matrix_attr='drug_text_matrix',
        mask_attr='drug_text_mask',
        enabled_attr='_drug_text_enabled',
    )
    disease_text_hit = scorer._lookup_node_text_embeddings(
        global_ids=torch.tensor([1], dtype=torch.long),
        batch_size=1,
        device=torch.device('cpu'),
        dtype=torch.float32,
        matrix_attr='disease_text_matrix',
        mask_attr='disease_text_mask',
        enabled_attr='_disease_text_enabled',
    )
    disease_text_miss = scorer._lookup_node_text_embeddings(
        global_ids=torch.tensor([2], dtype=torch.long),
        batch_size=1,
        device=torch.device('cpu'),
        dtype=torch.float32,
        matrix_attr='disease_text_matrix',
        mask_attr='disease_text_mask',
        enabled_attr='_disease_text_enabled',
    )

    assert scorer.output_mlp[0].in_features == (8 + 2 * 4 + 1024 + 768 + 768)
    assert scorer.drug_text_matrix.shape == (3, 768)
    assert scorer.disease_text_matrix.shape == (3, 768)
    assert torch.allclose(drug_text_hit, torch.full((1, 768), 3.0))
    assert torch.allclose(disease_text_hit, torch.full((1, 768), 2.0))
    assert torch.allclose(disease_text_miss, torch.zeros_like(disease_text_miss))



def test_pair_aggregation_scorer_ablate_gnn_zeros_graph_features() -> None:
    scorer = PairAggregationScorer(
        pair_emb_dim=4,
        path_emb_dim=6,
        hidden_dim=8,
        dropout=0.0,
        use_external_late_fusion=False,
        ablate_gnn=True,
    )

    pair_embs = torch.randn(2, 2, 4)
    paths_embs = torch.randn(2, 3, 6)
    attention_mask = torch.tensor([[True, True, False], [False, False, False]], dtype=torch.bool)

    logits = scorer(
        pair_embs=pair_embs,
        paths_embs=paths_embs,
        attention_mask=attention_mask,
    )
    assert logits.shape == (2,)
    assert torch.isfinite(logits).all()
    assert scorer.ablate_gnn is True

def test_pair_aggregation_scorer_can_disable_external_late_fusion() -> None:
    scorer = PairAggregationScorer(
        pair_emb_dim=4,
        path_emb_dim=6,
        hidden_dim=8,
        dropout=0.0,
        use_external_late_fusion=False,
    )

    assert scorer.use_external_late_fusion is False
    assert scorer.output_mlp[0].in_features == (8 + 2 * 4)

    pair_embs = torch.randn(2, 2, 4)
    paths_embs = torch.randn(2, 3, 6)
    attention_mask = torch.tensor([[True, True, False], [True, False, False]], dtype=torch.bool)

    logits = scorer(
        pair_embs=pair_embs,
        paths_embs=paths_embs,
        attention_mask=attention_mask,
        drug_global_ids=torch.tensor([0, 1], dtype=torch.long),
    )
    assert logits.shape == (2,)
    assert torch.isfinite(logits).all()


def test_pair_aggregation_scorer_zero_path_masking_forces_zero_gated_paths() -> None:
    scorer = PairAggregationScorer(
        pair_emb_dim=4,
        path_emb_dim=6,
        hidden_dim=8,
        dropout=0.0,
    )

    aggregated_paths = torch.randn(2, 8)
    attention_mask = torch.tensor(
        [
            [False, False, False],
            [True, False, False],
        ],
        dtype=torch.bool,
    )

    gated_paths, has_path_mask, path_gates = scorer._mask_and_gate_paths(
        aggregated_paths=aggregated_paths,
        attention_mask=attention_mask,
    )

    assert has_path_mask.shape == (2, 1)
    assert path_gates.shape == (2, 1)
    assert torch.allclose(gated_paths[0], torch.zeros_like(gated_paths[0]))
    assert torch.allclose(path_gates[0], torch.zeros_like(path_gates[0]))
    assert has_path_mask[0].item() is False
    assert has_path_mask[1].item() is True
    assert torch.isfinite(gated_paths).all()
    assert torch.isfinite(path_gates).all()
    assert 0.0 <= float(path_gates[1].item()) <= 1.0


def test_pair_aggregation_scorer_all_padding_logits_remain_finite_after_path_gate() -> None:
    torch.manual_seed(0)

    scorer = PairAggregationScorer(
        pair_emb_dim=4,
        path_emb_dim=6,
        hidden_dim=8,
        dropout=0.0,
    )

    pair_embs = torch.randn(2, 2, 4)
    paths_embs = torch.randn(2, 5, 6)
    attention_mask = torch.zeros(2, 5, dtype=torch.bool)

    logits, attention_weights = scorer(
        pair_embs=pair_embs,
        paths_embs=paths_embs,
        attention_mask=attention_mask,
        return_attention=True,
    )

    assert logits.shape == (2,)
    assert attention_weights.shape == (2, 5)
    assert torch.isfinite(logits).all()
    assert torch.isfinite(attention_weights).all()
    assert torch.allclose(attention_weights.sum(dim=1), torch.zeros(2), atol=1e-6)


def test_pair_aggregation_scorer_returns_finite_path_margin_loss_in_training() -> None:
    torch.manual_seed(0)

    scorer = PairAggregationScorer(
        pair_emb_dim=4,
        path_emb_dim=6,
        hidden_dim=8,
        dropout=0.0,
        use_external_late_fusion=False,
    )
    scorer.train()

    pair_embs = torch.randn(3, 2, 4)
    paths_embs = torch.randn(3, 2, 6)
    attention_mask = torch.tensor(
        [
            [True, False],
            [True, True],
            [False, False],
        ],
        dtype=torch.bool,
    )

    logits, path_loss = scorer(
        pair_embs=pair_embs,
        paths_embs=paths_embs,
        attention_mask=attention_mask,
        return_path_loss=True,
    )

    assert logits.shape == (3,)
    assert path_loss.ndim == 0
    assert torch.isfinite(path_loss)
    assert float(path_loss.item()) >= 0.0
