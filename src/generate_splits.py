from __future__ import annotations

import argparse
import csv
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import torch
from torch import Tensor

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.primekg_data_processor import PrimeKGDataProcessor


TARGET_RELATIONS = frozenset({'indication', 'off-label use', 'contraindication'})
SPLIT_NAMES = ('train', 'valid', 'test', 'ot')
DEFAULT_FRACTIONS = {
    'train': 0.7,
    'valid': 0.1,
    'test': 0.1,
    'ot': 0.1,
}


@dataclass(frozen=True)
class TargetPairRecord:
    drug_id: int
    disease_id: int
    relation_names: Tuple[str, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            'Generate fresh PrimeKG repurposing split assets from raw nodes.csv and '
            'edges.csv using physically correct pair-level target masking.'
        )
    )
    parser.add_argument(
        '--nodes-csv',
        type=Path,
        default=Path('data/PrimeKG/nodes.csv'),
        help='Path to PrimeKG nodes.csv.',
    )
    parser.add_argument(
        '--edges-csv',
        type=Path,
        default=Path('data/PrimeKG/edges.csv'),
        help='Path to PrimeKG edges.csv.',
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('data/PrimeKG/processed'),
        help='Directory where the new .pt assets will be written.',
    )
    parser.add_argument('--seed', type=int, default=42, help='Random seed for all split generation.')
    parser.add_argument('--train-frac', type=float, default=DEFAULT_FRACTIONS['train'])
    parser.add_argument('--valid-frac', type=float, default=DEFAULT_FRACTIONS['valid'])
    parser.add_argument('--test-frac', type=float, default=DEFAULT_FRACTIONS['test'])
    parser.add_argument('--ot-frac', type=float, default=DEFAULT_FRACTIONS['ot'])
    parser.add_argument(
        '--filename-prefix',
        type=str,
        default='primekg_indication',
        help='Prefix for generated files, e.g. primekg_indication.',
    )
    parser.add_argument(
        '--suffix',
        type=str,
        default='v2',
        help='Suffix appended before .pt, e.g. v2 -> *_v2.pt.',
    )
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Overwrite existing output files if they already exist.',
    )
    return parser.parse_args()


def _normalize_direct_relation_name(relation: str) -> str:
    normalized = relation.strip()
    if normalized.endswith('__reverse__'):
        normalized = normalized[: -len('__reverse__')]
    if normalized.startswith('rev_'):
        normalized = normalized[len('rev_') :]
    if normalized.startswith('reverse_'):
        normalized = normalized[len('reverse_') :]
    return normalized


def _validate_fractions(fractions: Mapping[str, float]) -> None:
    total = sum(float(fractions[name]) for name in SPLIT_NAMES)
    if abs(total - 1.0) > 1e-8:
        raise ValueError(f'Split fractions must sum to 1.0, got {total:.8f}')
    for split_name, value in fractions.items():
        if value < 0.0:
            raise ValueError(f'Split fraction must be non-negative: {split_name}={value}')


def _compute_split_sizes(total: int, fractions: Mapping[str, float]) -> Dict[str, int]:
    if total < 0:
        raise ValueError(f'Total size must be >= 0, got {total}')
    raw_counts = {name: total * float(fractions[name]) for name in SPLIT_NAMES}
    sizes = {name: int(raw_counts[name]) for name in SPLIT_NAMES}
    remainder = total - sum(sizes.values())
    if remainder > 0:
        # Assign the leftover items to the largest fractional remainders.
        ranked = sorted(
            SPLIT_NAMES,
            key=lambda name: (raw_counts[name] - sizes[name], fractions[name]),
            reverse=True,
        )
        for index in range(remainder):
            sizes[ranked[index % len(ranked)]] += 1
    return sizes


def _chunk_by_sizes(items: Sequence[int], sizes: Mapping[str, int]) -> Dict[str, List[int]]:
    chunks: Dict[str, List[int]] = {}
    cursor = 0
    for split_name in SPLIT_NAMES:
        split_size = int(sizes[split_name])
        chunks[split_name] = list(items[cursor: cursor + split_size])
        cursor += split_size
    if cursor != len(items):
        raise RuntimeError('Chunking failed: leftover items remained after applying split sizes.')
    return chunks


def _pair_tensor(pairs: Sequence[Tuple[int, int]]) -> Tensor:
    if not pairs:
        return torch.empty((0, 2), dtype=torch.long)
    return torch.tensor(list(pairs), dtype=torch.long)


def _id_tensor(values: Iterable[int]) -> Tensor:
    unique_values = sorted({int(value) for value in values})
    if not unique_values:
        return torch.empty((0,), dtype=torch.long)
    return torch.tensor(unique_values, dtype=torch.long)


def _collect_target_pairs(
    *,
    nodes_csv: Path,
    edges_csv: Path,
    processor: PrimeKGDataProcessor,
) -> Tuple[List[TargetPairRecord], List[int], List[int]]:
    processor.build_entity_mappings()
    global_entity2id = processor.global_entity2id
    relation_bucket: Dict[Tuple[int, int], set[str]] = {}

    with edges_csv.open('r', encoding='utf-8-sig', newline='') as edge_file:
        reader = csv.DictReader(edge_file)
        required_columns = {'src_id', 'src_type', 'rel', 'dst_id', 'dst_type'}
        missing_columns = required_columns.difference(reader.fieldnames or [])
        if missing_columns:
            raise KeyError(f'edges.csv is missing required columns: {sorted(missing_columns)}')

        for row in reader:
            src_type = row['src_type'].strip()
            dst_type = row['dst_type'].strip()
            normalized_relation = _normalize_direct_relation_name(row['rel'])
            if normalized_relation not in TARGET_RELATIONS:
                continue
            if {src_type, dst_type} != {'drug', 'disease'}:
                continue

            src_raw_id = row['src_id'].strip()
            dst_raw_id = row['dst_id'].strip()
            if src_raw_id not in global_entity2id or dst_raw_id not in global_entity2id:
                continue

            if src_type == 'drug':
                drug_gid = int(global_entity2id[src_raw_id])
                disease_gid = int(global_entity2id[dst_raw_id])
            else:
                drug_gid = int(global_entity2id[dst_raw_id])
                disease_gid = int(global_entity2id[src_raw_id])

            relation_bucket.setdefault((drug_gid, disease_gid), set()).add(normalized_relation)

    target_records = [
        TargetPairRecord(
            drug_id=drug_gid,
            disease_id=disease_gid,
            relation_names=tuple(sorted(relations)),
        )
        for (drug_gid, disease_gid), relations in sorted(relation_bucket.items())
    ]

    unique_drugs = sorted({record.drug_id for record in target_records})
    unique_diseases = sorted({record.disease_id for record in target_records})
    return target_records, unique_drugs, unique_diseases


def _generate_random_split(
    *,
    target_records: Sequence[TargetPairRecord],
    fractions: Mapping[str, float],
    seed: int,
) -> Tuple[Dict[str, Tensor], Dict[str, object]]:
    rng = random.Random(seed)
    shuffled_records = list(target_records)
    rng.shuffle(shuffled_records)
    split_sizes = _compute_split_sizes(len(shuffled_records), fractions)

    target_pairs: Dict[str, Tensor] = {}
    pair_lists: Dict[str, List[Tuple[int, int]]] = {}
    cursor = 0
    for split_name in SPLIT_NAMES:
        next_cursor = cursor + split_sizes[split_name]
        split_records = shuffled_records[cursor:next_cursor]
        cursor = next_cursor
        pair_list = [(record.drug_id, record.disease_id) for record in split_records]
        pair_lists[split_name] = pair_list
        target_pairs[split_name] = _pair_tensor(pair_list)

    metadata = {
        'entity_split_type': 'edge',
        'split_sizes': {name: int(target_pairs[name].size(0)) for name in SPLIT_NAMES},
        'split_entity_counts': None,
        'pair_lists': pair_lists,
    }
    return target_pairs, metadata


def _generate_entity_level_split(
    *,
    split_mode: str,
    target_records: Sequence[TargetPairRecord],
    unique_entity_ids: Sequence[int],
    fractions: Mapping[str, float],
    seed: int,
) -> Tuple[Dict[str, Tensor], Dict[str, object]]:
    rng = random.Random(seed)
    shuffled_entities = list(unique_entity_ids)
    rng.shuffle(shuffled_entities)
    entity_split_sizes = _compute_split_sizes(len(shuffled_entities), fractions)
    entity_splits = _chunk_by_sizes(shuffled_entities, entity_split_sizes)

    entity_to_split = {
        int(entity_id): split_name
        for split_name, entity_ids in entity_splits.items()
        for entity_id in entity_ids
    }

    pair_lists: Dict[str, List[Tuple[int, int]]] = {name: [] for name in SPLIT_NAMES}
    for record in target_records:
        if split_mode == 'cross_drug':
            split_name = entity_to_split[int(record.drug_id)]
        elif split_mode == 'cross_disease':
            split_name = entity_to_split[int(record.disease_id)]
        else:
            raise ValueError(f'Unsupported entity-level split mode: {split_mode}')
        pair_lists[split_name].append((record.drug_id, record.disease_id))

    for split_name in SPLIT_NAMES:
        rng.shuffle(pair_lists[split_name])

    target_pairs = {split_name: _pair_tensor(pair_list) for split_name, pair_list in pair_lists.items()}
    metadata = {
        'entity_split_type': 'node',
        'split_sizes': {name: int(target_pairs[name].size(0)) for name in SPLIT_NAMES},
        'split_entity_counts': {name: len(entity_splits[name]) for name in SPLIT_NAMES},
        'entity_splits': {split_name: _id_tensor(entity_ids) for split_name, entity_ids in entity_splits.items()},
        'pair_lists': pair_lists,
    }
    return target_pairs, metadata


def _assert_pair_disjointness(target_pairs: Mapping[str, Tensor]) -> None:
    pair_sets = {
        split_name: {tuple(map(int, row)) for row in target_pairs[split_name].tolist()}
        for split_name in SPLIT_NAMES
    }
    for left_index, left_name in enumerate(SPLIT_NAMES):
        for right_name in SPLIT_NAMES[left_index + 1 :]:
            overlap = pair_sets[left_name].intersection(pair_sets[right_name])
            assert not overlap, f'Pair overlap detected between {left_name} and {right_name}: {len(overlap)} pairs'


def _assert_entity_disjointness(entity_splits: Mapping[str, Tensor], label: str) -> None:
    entity_sets = {
        split_name: {int(value) for value in tensor.tolist()}
        for split_name, tensor in entity_splits.items()
    }
    for left_index, left_name in enumerate(SPLIT_NAMES):
        for right_name in SPLIT_NAMES[left_index + 1 :]:
            overlap = entity_sets[left_name].intersection(entity_sets[right_name])
            assert not overlap, (
                f'{label} overlap detected between {left_name} and {right_name}: '
                f'{len(overlap)} entities'
            )


def _build_lightweight_asset(
    *,
    split_mode: str,
    target_pairs: Mapping[str, Tensor],
    processor: PrimeKGDataProcessor,
    nodes_csv: Path,
    edges_csv: Path,
    seed: int,
    fractions: Mapping[str, float],
    target_records: Sequence[TargetPairRecord],
    unique_drugs: Sequence[int],
    unique_diseases: Sequence[int],
    metadata: Mapping[str, object],
) -> Dict[str, object]:
    idx_to_node_id = [record.raw_id for _, record in sorted(processor.id2entity.items())]
    idx_to_node_name = [record.name for _, record in sorted(processor.id2entity.items())]
    idx_to_node_type = [record.node_type for _, record in sorted(processor.id2entity.items())]

    payload: Dict[str, object] = {
        'split_mode': split_mode,
        'split_version': 'v2_rebuilt_from_raw_primekg',
        'nodes_csv': str(nodes_csv),
        'edges_csv': str(edges_csv),
        'seed': int(seed),
        'fractions': {name: float(fractions[name]) for name in SPLIT_NAMES},
        'target_relation_names': sorted(TARGET_RELATIONS),
        'target_source_type': 'drug',
        'target_destination_type': 'disease',
        'target_pairs': {name: tensor.clone() for name, tensor in target_pairs.items()},
        'splits': {name: tensor.clone() for name, tensor in target_pairs.items()},
        'all_target_pairs': _pair_tensor([(record.drug_id, record.disease_id) for record in target_records]),
        'node_id_to_idx': dict(processor.global_entity2id),
        'idx_to_node_id': idx_to_node_id,
        'idx_to_node_name': idx_to_node_name,
        'idx_to_node_type': idx_to_node_type,
        'num_nodes': len(idx_to_node_id),
        'drug_node_indices': _id_tensor(unique_drugs),
        'disease_node_indices': _id_tensor(unique_diseases),
        'stats': {
            'num_target_pairs': len(target_records),
            'num_unique_target_drugs': len(unique_drugs),
            'num_unique_target_diseases': len(unique_diseases),
            'split_pair_counts': {name: int(target_pairs[name].size(0)) for name in SPLIT_NAMES},
            'split_entity_counts': metadata.get('split_entity_counts'),
        },
    }

    entity_splits = metadata.get('entity_splits')
    if isinstance(entity_splits, dict):
        payload['entity_splits'] = {name: tensor.clone() for name, tensor in entity_splits.items()}

    return payload


def _save_asset(payload: Mapping[str, object], output_path: Path, overwrite: bool) -> None:
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f'Output already exists: {output_path}. Pass --overwrite to replace it.'
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(payload), output_path)


def main() -> None:
    args = parse_args()
    fractions = {
        'train': float(args.train_frac),
        'valid': float(args.valid_frac),
        'test': float(args.test_frac),
        'ot': float(args.ot_frac),
    }
    _validate_fractions(fractions)

    processor = PrimeKGDataProcessor(node_csv_path=args.nodes_csv, edge_csv_path=args.edges_csv)
    processor.build_entity_mappings()
    target_records, unique_drugs, unique_diseases = _collect_target_pairs(
        nodes_csv=args.nodes_csv,
        edges_csv=args.edges_csv,
        processor=processor,
    )
    if not target_records:
        raise RuntimeError('No target drug-disease pairs were found for the requested clinical relations.')

    random_pairs, random_meta = _generate_random_split(
        target_records=target_records,
        fractions=fractions,
        seed=int(args.seed),
    )
    cross_drug_pairs, cross_drug_meta = _generate_entity_level_split(
        split_mode='cross_drug',
        target_records=target_records,
        unique_entity_ids=unique_drugs,
        fractions=fractions,
        seed=int(args.seed),
    )
    cross_disease_pairs, cross_disease_meta = _generate_entity_level_split(
        split_mode='cross_disease',
        target_records=target_records,
        unique_entity_ids=unique_diseases,
        fractions=fractions,
        seed=int(args.seed),
    )

    _assert_pair_disjointness(random_pairs)
    _assert_pair_disjointness(cross_drug_pairs)
    _assert_pair_disjointness(cross_disease_pairs)

    cross_drug_entity_splits = cross_drug_meta['entity_splits']
    assert isinstance(cross_drug_entity_splits, dict)
    _assert_entity_disjointness(cross_drug_entity_splits, label='Drug split')

    cross_disease_entity_splits = cross_disease_meta['entity_splits']
    assert isinstance(cross_disease_entity_splits, dict)
    _assert_entity_disjointness(cross_disease_entity_splits, label='Disease split')

    random_asset = _build_lightweight_asset(
        split_mode='random',
        target_pairs=random_pairs,
        processor=processor,
        nodes_csv=args.nodes_csv,
        edges_csv=args.edges_csv,
        seed=int(args.seed),
        fractions=fractions,
        target_records=target_records,
        unique_drugs=unique_drugs,
        unique_diseases=unique_diseases,
        metadata=random_meta,
    )
    cross_drug_asset = _build_lightweight_asset(
        split_mode='cross_drug',
        target_pairs=cross_drug_pairs,
        processor=processor,
        nodes_csv=args.nodes_csv,
        edges_csv=args.edges_csv,
        seed=int(args.seed),
        fractions=fractions,
        target_records=target_records,
        unique_drugs=unique_drugs,
        unique_diseases=unique_diseases,
        metadata=cross_drug_meta,
    )
    cross_disease_asset = _build_lightweight_asset(
        split_mode='cross_disease',
        target_pairs=cross_disease_pairs,
        processor=processor,
        nodes_csv=args.nodes_csv,
        edges_csv=args.edges_csv,
        seed=int(args.seed),
        fractions=fractions,
        target_records=target_records,
        unique_drugs=unique_drugs,
        unique_diseases=unique_diseases,
        metadata=cross_disease_meta,
    )

    random_path = args.output_dir / f'{args.filename_prefix}_random_{args.suffix}.pt'
    cross_drug_path = args.output_dir / f'{args.filename_prefix}_cross_drug_{args.suffix}.pt'
    cross_disease_path = args.output_dir / f'{args.filename_prefix}_cross_disease_{args.suffix}.pt'

    _save_asset(random_asset, random_path, overwrite=args.overwrite)
    _save_asset(cross_drug_asset, cross_drug_path, overwrite=args.overwrite)
    _save_asset(cross_disease_asset, cross_disease_path, overwrite=args.overwrite)

    print('Generated split assets:')
    for split_mode, asset_path, asset in (
        ('random', random_path, random_asset),
        ('cross_drug', cross_drug_path, cross_drug_asset),
        ('cross_disease', cross_disease_path, cross_disease_asset),
    ):
        pair_counts = asset['stats']['split_pair_counts']
        print(
            f'- {split_mode}: {asset_path} | '
            f"train={pair_counts['train']} valid={pair_counts['valid']} "
            f"test={pair_counts['test']} ot={pair_counts['ot']}"
        )

    print('Sanity assertions passed:')
    print('- cross_drug train/valid/test/ot drug sets are pairwise disjoint')
    print('- cross_disease train/valid/test/ot disease sets are pairwise disjoint')


if __name__ == '__main__':
    main()
