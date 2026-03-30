from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.path_bpr_sampler import PathBPRSampler, build_path_bpr_dataloader
from src.primekg_data_processor import PrimeKGDataProcessor


def _build_test_data():
    processor = PrimeKGDataProcessor(
        node_csv_path=PROJECT_ROOT / 'data' / 'PrimeKG' / 'nodes.csv',
        edge_csv_path=PROJECT_ROOT / 'data' / 'PrimeKG' / 'edges.csv',
    )
    processor.build_entity_mappings()

    ho_triplets = [
        ('Copper', 'PHYHIP', 'hypertensive disorder'),
        ('Flunisolide', 'KIF15', 'Parkinson disease'),
        ('Oxygen', 'GPANK1', 'restless legs syndrome'),
    ]
    ho_id_paths = processor.convert_ho_triplets_to_ids(ho_triplets)
    data = processor.build_heterodata(ho_id_paths)
    return processor, data


def test_path_bpr_sampler_getitem_and_negative_validity() -> None:
    processor, data = _build_test_data()
    dataset = PathBPRSampler(data)

    assert len(dataset) == 3

    positive_path, negative_path = dataset[0]
    assert positive_path.shape == (3,)
    assert negative_path.shape == (3,)
    assert positive_path.dtype == torch.long
    assert negative_path.dtype == torch.long

    # Safe negative: 固定 drug 和 disease，仅替换 gene。
    assert int(positive_path[0]) == int(negative_path[0])
    assert int(positive_path[2]) == int(negative_path[2])
    assert int(positive_path[1]) != int(negative_path[1])

    negative_triplet = tuple(int(x) for x in negative_path.tolist())
    assert negative_triplet not in dataset.positive_path_set

    for global_id, expected_type in zip(negative_path.tolist(), data.ho_path_node_types):
        assert processor.id2entity[int(global_id)].node_type == expected_type


def test_path_bpr_dataloader_batch_shapes() -> None:
    _, data = _build_test_data()

    dataloader = build_path_bpr_dataloader(
        data=data,
        batch_size=2,
        shuffle=False,
        num_workers=0,
    )

    positive_batch, negative_batch = next(iter(dataloader))
    assert positive_batch.shape == (2, 3)
    assert negative_batch.shape == (2, 3)
    assert positive_batch.dtype == torch.long
    assert negative_batch.dtype == torch.long

    dataset = dataloader.dataset
    assert isinstance(dataset, PathBPRSampler)
    for negative_triplet in negative_batch.tolist():
        assert tuple(int(x) for x in negative_triplet) not in dataset.positive_path_set


def test_manual_dataloader_instantiation_example() -> None:
    _, data = _build_test_data()
    dataset = PathBPRSampler(data)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0)

    positive_batch, negative_batch = next(iter(dataloader))
    assert positive_batch.shape[1] == 3
    assert negative_batch.shape[1] == 3
