from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch_geometric.data import HeteroData


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pair_path_bpr_sampler import PairPathBPRDataset, build_pair_path_bpr_dataloader


def _build_toy_data() -> HeteroData:
    data = HeteroData()
    data['drug'].num_nodes = 4
    data['drug'].global_id = torch.tensor([10, 11, 12, 13], dtype=torch.long)

    data['gene/protein'].num_nodes = 5
    data['gene/protein'].global_id = torch.tensor([20, 21, 22, 23, 24], dtype=torch.long)

    data['disease'].num_nodes = 4
    data['disease'].global_id = torch.tensor([30, 31, 32, 33], dtype=torch.long)

    data[('drug', 'targets', 'gene/protein')].edge_index = torch.tensor(
        [[0, 0, 1, 2, 3], [0, 1, 2, 3, 4]],
        dtype=torch.long,
    )
    data[('gene/protein', 'targets__reverse__', 'drug')].edge_index = torch.tensor(
        [[0, 1, 2, 3, 4], [0, 0, 1, 2, 3]],
        dtype=torch.long,
    )

    data[('disease', 'disease_protein', 'gene/protein')].edge_index = torch.tensor(
        [[0, 0, 0, 0, 1, 2, 2], [0, 1, 3, 4, 2, 1, 4]],
        dtype=torch.long,
    )
    data[('gene/protein', 'disease_protein__reverse__', 'disease')].edge_index = torch.tensor(
        [[0, 1, 3, 4, 2, 1, 4], [0, 0, 0, 0, 1, 2, 2]],
        dtype=torch.long,
    )

    data.ho_pos_paths = torch.tensor(
        [[10, 20, 30], [10, 21, 30], [11, 22, 31]],
        dtype=torch.long,
    )
    data.ho_path_node_types = ('drug', 'gene/protein', 'disease')
    return data


def _build_toy_pathway_data() -> HeteroData:
    data = _build_toy_data()
    data['pathway'].num_nodes = 2
    data['pathway'].global_id = torch.tensor([40, 41], dtype=torch.long)

    data[('gene/protein', 'participates_in', 'pathway')].edge_index = torch.tensor(
        [[0, 0, 2, 3], [0, 1, 1, 0]],
        dtype=torch.long,
    )
    data[('pathway', 'participates_in__reverse__', 'gene/protein')].edge_index = torch.tensor(
        [[0, 1, 1, 0], [0, 0, 2, 3]],
        dtype=torch.long,
    )
    return data


def test_pair_path_dataset_groups_positive_paths_by_pair() -> None:
    data = _build_toy_data()
    dataset = PairPathBPRDataset(data=data, negative_strategy='cross_drug')

    assert len(dataset) == 2
    assert set(dataset.positive_pairs) == {(10, 30), (11, 31)}

    pos_paths = dataset.path_bank[(10, 30)]
    assert pos_paths.shape == (2, 3)
    assert set(tuple(int(x) for x in row) for row in pos_paths.tolist()) == {
        (10, 20, 30),
        (10, 21, 30),
    }


def test_topology_aware_cross_drug_negative_returns_real_paths() -> None:
    torch.manual_seed(0)

    data = _build_toy_data()
    dataset = PairPathBPRDataset(data=data, negative_strategy='cross_drug')

    assert dataset.disease_to_connected_drugs[30] == (10, 12, 13)

    sample = dataset[0]
    neg_pair = tuple(int(x) for x in sample['neg_pair_ids'].tolist())
    neg_paths = sample['neg_paths']

    assert neg_pair in {(12, 30), (13, 30)}
    assert neg_paths.shape[0] >= 1
    assert neg_paths.shape[1] == 3
    assert all(int(row[0]) == neg_pair[0] and int(row[2]) == neg_pair[1] for row in neg_paths.tolist())


def test_topology_aware_cross_disease_negative_returns_real_paths() -> None:
    torch.manual_seed(0)

    data = _build_toy_data()
    dataset = PairPathBPRDataset(data=data, negative_strategy='cross_disease')

    assert dataset.drug_to_connected_diseases[10] == (30, 32)

    sample = dataset[0]
    neg_pair = tuple(int(x) for x in sample['neg_pair_ids'].tolist())
    neg_paths = sample['neg_paths']

    assert neg_pair == (10, 32)
    assert neg_paths.shape[0] >= 1
    assert neg_paths.shape[1] == 3
    assert all(int(row[0]) == 10 and int(row[2]) == 32 for row in neg_paths.tolist())


def test_dataset_supports_external_positive_paths_and_known_positive_pairs() -> None:
    torch.manual_seed(0)

    data = _build_toy_data()
    valid_ho_paths = torch.tensor([[12, 23, 30], [10, 21, 32]], dtype=torch.long)
    known_positive_paths = torch.cat([data.ho_pos_paths, valid_ho_paths], dim=0)

    dataset = PairPathBPRDataset(
        data=data,
        positive_paths=valid_ho_paths,
        known_positive_pairs=known_positive_paths,
        negative_strategy='cross_drug',
    )

    assert len(dataset) == 2
    assert set(dataset.positive_pairs) == {(12, 30), (10, 32)}
    assert dataset.path_bank[(12, 30)].shape == (1, 3)

    known_pairs = {(10, 30), (11, 31), (12, 30), (10, 32)}
    sample = dataset[0]
    neg_pair = tuple(int(x) for x in sample['neg_pair_ids'].tolist())
    assert neg_pair not in known_pairs
    assert sample['neg_paths'].shape[0] >= 1


def test_collate_fn_still_supports_dummy_path_for_disconnected_random_negative() -> None:
    torch.manual_seed(0)

    data = _build_toy_data()
    dataset = PairPathBPRDataset(data=data, negative_strategy='random')
    dataset.known_positive_pair_set.update(dataset.connected_pair_set)

    sample = dataset[0]
    neg_pair = tuple(int(x) for x in sample['neg_pair_ids'].tolist())
    assert neg_pair not in dataset.connected_pair_set
    assert sample['neg_paths'].shape == (0, 3)

    batch = dataset.collate_fn([sample])
    assert batch['pos_pair_ids'].shape == (1, 2)
    assert batch['neg_pair_ids'].shape == (1, 2)
    assert batch['pos_paths'].shape == (1, 2, 3)
    assert batch['neg_paths'].shape == (1, 1, 3)
    assert torch.equal(batch['neg_paths'][0, 0], torch.zeros(3, dtype=torch.long))
    assert batch['neg_attention_mask'][0, 0].item() is False


def test_dataloader_returns_padded_pair_level_batch_with_real_negative_paths() -> None:
    torch.manual_seed(0)

    data = _build_toy_data()
    dataloader = build_pair_path_bpr_dataloader(
        data=data,
        batch_size=2,
        shuffle=False,
        num_workers=0,
        negative_strategy='cross_drug',
    )

    batch = next(iter(dataloader))
    assert batch['pos_pair_ids'].shape == (2, 2)
    assert batch['neg_pair_ids'].shape == (2, 2)
    assert batch['pos_paths'].shape[0] == 2
    assert batch['neg_paths'].shape[0] == 2
    assert batch['pos_paths'].shape[2] == 3
    assert batch['neg_paths'].shape[2] == 3
    assert batch['pos_attention_mask'].dtype == torch.bool
    assert batch['neg_attention_mask'].dtype == torch.bool
    assert batch['pos_attention_mask'].any().item() is True
    assert batch['neg_attention_mask'].any(dim=1).all().item() is True


def test_quad_dataset_expands_gene_to_pathway_paths_when_enabled() -> None:
    data = _build_toy_pathway_data()
    dataset = PairPathBPRDataset(
        data=data,
        negative_strategy='cross_drug',
        use_pathway_quads=True,
    )

    pos_paths = dataset.path_bank[(10, 30)]
    assert pos_paths.shape == (3, 4)
    assert set(tuple(int(x) for x in row) for row in pos_paths.tolist()) == {
        (10, 20, 40, 30),
        (10, 20, 41, 30),
        (10, 21, 0, 30),
    }


def test_quad_dataset_uses_dummy_pathway_when_gene_has_no_pathway() -> None:
    data = _build_toy_pathway_data()
    dataset = PairPathBPRDataset(
        data=data,
        negative_strategy='cross_drug',
        use_pathway_quads=True,
    )

    pos_paths = dataset.path_bank[(10, 30)]
    assert any(tuple(int(x) for x in row) == (10, 21, 0, 30) for row in pos_paths.tolist())


def test_quad_negative_paths_include_pathway_column() -> None:
    torch.manual_seed(0)

    data = _build_toy_pathway_data()
    dataset = PairPathBPRDataset(
        data=data,
        negative_strategy='cross_drug',
        use_pathway_quads=True,
    )

    sample = dataset[0]
    neg_pair = tuple(int(x) for x in sample['neg_pair_ids'].tolist())
    neg_paths = sample['neg_paths']

    assert neg_pair in {(12, 30), (13, 30)}
    assert neg_paths.shape[1] == 4
    assert all(int(row[0]) == neg_pair[0] and int(row[3]) == neg_pair[1] for row in neg_paths.tolist())


def test_quad_collate_fn_still_supports_disconnected_random_negative() -> None:
    torch.manual_seed(0)

    data = _build_toy_pathway_data()
    dataset = PairPathBPRDataset(
        data=data,
        negative_strategy='random',
        use_pathway_quads=True,
    )
    dataset.known_positive_pair_set.update(dataset.connected_pair_set)

    sample = dataset[0]
    assert sample['neg_paths'].shape == (0, 4)

    batch = dataset.collate_fn([sample])
    assert batch['neg_paths'].shape == (1, 1, 4)
    assert torch.equal(batch['neg_paths'][0, 0], torch.zeros(4, dtype=torch.long))
    assert batch['neg_attention_mask'][0, 0].item() is False
