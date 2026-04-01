from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.train_split_probe import derive_ho_triplets_for_pair_splits, load_pair_splits
from src.primekg_data_processor import PrimeKGDataProcessor


IdPair = Tuple[int, int]
IdTriplet = Tuple[int, int, int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Audit HO/micro-path coverage for valid/test/ot splits.'
    )
    parser.add_argument(
        '--processed-path',
        type=Path,
        default=Path('data/PrimeKG/processed/primekg_indication_cross_disease_v2.pt'),
        help='Processed split asset to audit.',
    )
    parser.add_argument(
        '--nodes-csv',
        type=Path,
        default=Path('data/PrimeKG/nodes.csv'),
        help='PrimeKG nodes.csv path.',
    )
    parser.add_argument(
        '--edges-csv',
        type=Path,
        default=Path('data/PrimeKG/edges.csv'),
        help='PrimeKG edges.csv path.',
    )
    parser.add_argument(
        '--output-json',
        type=Path,
        default=None,
        help='Optional path to save the audit report as JSON.',
    )
    return parser.parse_args()


def _compute_pair_coverage(
    pair_splits: Dict[str, set[IdPair]],
    split_triplets: Dict[str, List[IdTriplet]],
) -> Dict[str, Dict[str, object]]:
    report: Dict[str, Dict[str, object]] = {}
    for split_name in ('valid', 'test', 'ot'):
        pair_set = set(pair_splits.get(split_name, set()))
        covered_pairs = {(int(drug_id), int(disease_id)) for drug_id, _, disease_id in split_triplets.get(split_name, [])}
        uncovered_pairs = sorted(pair_set.difference(covered_pairs))
        total_pairs = len(pair_set)
        covered_count = len(covered_pairs)
        coverage_rate = (covered_count / total_pairs) if total_pairs > 0 else 0.0
        report[split_name] = {
            'total_pairs': total_pairs,
            'covered_pairs': covered_count,
            'coverage_rate': coverage_rate,
            'num_triplets': len(split_triplets.get(split_name, [])),
            'uncovered_pairs_preview': uncovered_pairs[:10],
        }
    return report


def main() -> None:
    args = parse_args()
    processor = PrimeKGDataProcessor(node_csv_path=args.nodes_csv, edge_csv_path=args.edges_csv)
    processor.build_entity_mappings()

    split_mode, pair_splits = load_pair_splits(args.processed_path)
    split_triplets = derive_ho_triplets_for_pair_splits(
        processor=processor,
        edge_csv_path=args.edges_csv,
        pair_splits=pair_splits,
    )
    coverage_report = _compute_pair_coverage(pair_splits=pair_splits, split_triplets=split_triplets)

    payload = {
        'processed_path': str(args.processed_path),
        'split_mode': split_mode,
        'coverage': coverage_report,
    }

    print(f'processed_path: {args.processed_path}')
    print(f'split_mode: {split_mode}')
    for split_name in ('valid', 'test', 'ot'):
        stats = coverage_report[split_name]
        print(
            f"{split_name}: total_pairs={stats['total_pairs']} covered_pairs={stats['covered_pairs']} "
            f"coverage_rate={stats['coverage_rate'] * 100:.2f}% num_triplets={stats['num_triplets']}"
        )
        if stats['uncovered_pairs_preview']:
            print(f"  uncovered_preview={stats['uncovered_pairs_preview']}")

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(payload, indent=2), encoding='utf-8')
        print(f'saved_json: {args.output_json}')


if __name__ == '__main__':
    main()
