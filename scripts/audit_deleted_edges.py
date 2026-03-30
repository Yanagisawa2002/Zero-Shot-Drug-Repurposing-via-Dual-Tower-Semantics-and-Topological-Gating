from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import pandas as pd
import torch
from torch import Tensor
from torch_geometric.data import HeteroData

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.train_split_probe import derive_ho_triplets_for_pair_splits, load_pair_splits
from src.graph_surgery import build_split_isolation_targets, remove_leakage_edges
from src.pair_path_bpr_sampler import PairPathBPRDataset
from src.primekg_data_processor import PrimeKGDataProcessor


EdgeType = Tuple[str, str, str]
IdTriplet = Tuple[int, int, int]

DIRECT_LEAKAGE_LABEL = 'direct_leakage'
MECHANISTIC_LABEL = 'mechanistic'
OTHER_LABEL = 'other'


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            'Audit edge deletions introduced by remove_leakage_edges by comparing the raw '
            'PrimeKG graph against the cleaned graph under a chosen split.'
        )
    )
    parser.add_argument('--processed-path', type=Path, required=True)
    parser.add_argument('--nodes-csv', type=Path, default=Path('data/PrimeKG/nodes.csv'))
    parser.add_argument('--edges-csv', type=Path, default=Path('data/PrimeKG/edges.csv'))
    parser.add_argument(
        '--target-subset',
        type=str,
        choices=('valid', 'test', 'heldout'),
        default='test',
    )
    parser.add_argument(
        '--path-format',
        type=str,
        choices=('auto', 'triplet', 'quad'),
        default='auto',
    )
    parser.add_argument('--disable-split-isolation', action='store_true')
    parser.add_argument('--report-json', type=Path, default=None)
    return parser.parse_args()


def build_raw_graph(
    nodes_csv: Path,
    edges_csv: Path,
    train_triplets: Sequence[IdTriplet],
) -> Tuple[PrimeKGDataProcessor, HeteroData]:
    processor = PrimeKGDataProcessor(node_csv_path=nodes_csv, edge_csv_path=edges_csv)
    processor.build_entity_mappings()
    raw_data = processor.build_heterodata(train_triplets)
    return processor, raw_data


def resolve_target_triplets(
    split_triplets: Mapping[str, List[IdTriplet]],
    target_subset: str,
) -> List[IdTriplet]:
    if target_subset == 'valid':
        return list(split_triplets['valid'])
    if target_subset == 'test':
        return list(split_triplets['test'])
    if target_subset == 'heldout':
        return list(split_triplets['valid']) + list(split_triplets['test'])
    raise ValueError(f'Unsupported target_subset: {target_subset}')


def expand_triplets_to_quads(data: HeteroData, target_triplets: Sequence[IdTriplet]) -> Tensor:
    if not target_triplets:
        return torch.empty((0, 4), dtype=torch.long)

    holder = copy.deepcopy(data)
    holder.ho_pos_paths = torch.tensor(target_triplets, dtype=torch.long)
    dataset = PairPathBPRDataset(
        data=holder,
        positive_paths=holder.ho_pos_paths,
        known_positive_pairs=holder.ho_pos_paths,
        negative_strategy='random',
        use_pathway_quads=True,
    )

    quad_rows: List[Tuple[int, int, int, int]] = []
    for path_tensor in dataset.path_bank.values():
        quad_rows.extend(tuple(int(x) for x in row) for row in path_tensor.tolist())

    unique_quads = list(dict.fromkeys(quad_rows))
    if not unique_quads:
        return torch.empty((0, 4), dtype=torch.long)
    return torch.tensor(unique_quads, dtype=torch.long)


def resolve_target_paths(
    data: HeteroData,
    target_triplets: Sequence[IdTriplet],
    path_format: str,
) -> Tensor:
    if path_format == 'triplet':
        return torch.tensor(target_triplets, dtype=torch.long)

    if path_format == 'quad':
        return expand_triplets_to_quads(data=data, target_triplets=target_triplets)

    if 'pathway' in data.node_types:
        quad_tensor = expand_triplets_to_quads(data=data, target_triplets=target_triplets)
        if quad_tensor.numel() > 0:
            return quad_tensor
    return torch.tensor(target_triplets, dtype=torch.long)


def count_edges_by_type(edge_index_dict: Mapping[EdgeType, Tensor]) -> Dict[EdgeType, int]:
    return {
        edge_type: int(edge_index.size(1))
        for edge_type, edge_index in edge_index_dict.items()
    }


def classify_edge_type(edge_type: EdgeType) -> str:
    src_type, _, dst_type = edge_type
    node_types = {src_type, dst_type}

    if node_types == {'drug', 'disease'}:
        return DIRECT_LEAKAGE_LABEL

    if node_types.intersection({'gene/protein', 'pathway'}):
        return MECHANISTIC_LABEL

    return OTHER_LABEL


def build_deletion_dataframe(
    raw_counts: Mapping[EdgeType, int],
    clean_counts: Mapping[EdgeType, int],
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    all_edge_types = sorted(set(raw_counts.keys()).union(clean_counts.keys()))
    for edge_type in all_edge_types:
        raw_count = int(raw_counts.get(edge_type, 0))
        clean_count = int(clean_counts.get(edge_type, 0))
        deleted_count = raw_count - clean_count
        if deleted_count <= 0:
            continue

        rows.append(
            {
                'edge_type': str(edge_type),
                'src_type': edge_type[0],
                'relation': edge_type[1],
                'dst_type': edge_type[2],
                'category': classify_edge_type(edge_type),
                'raw_count': raw_count,
                'clean_count': clean_count,
                'deleted_count': deleted_count,
                'deleted_ratio': deleted_count / raw_count if raw_count > 0 else 0.0,
            }
        )

    if not rows:
        return pd.DataFrame(
            columns=[
                'edge_type',
                'src_type',
                'relation',
                'dst_type',
                'category',
                'raw_count',
                'clean_count',
                'deleted_count',
                'deleted_ratio',
            ]
        )

    frame = pd.DataFrame(rows)
    return frame.sort_values(
        by=['category', 'deleted_count', 'edge_type'],
        ascending=[True, False, True],
        ignore_index=True,
    )


def format_markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return '_None_'

    display = frame.copy()
    display['deleted_ratio'] = display['deleted_ratio'].map(lambda x: f'{x:.2%}')
    display = display[
        ['edge_type', 'raw_count', 'clean_count', 'deleted_count', 'deleted_ratio']
    ]
    return display.to_markdown(index=False)


def render_markdown_report(
    *,
    processed_path: Path,
    split_mode: str,
    target_subset: str,
    target_paths: Tensor,
    include_split_isolation: bool,
    deletion_frame: pd.DataFrame,
) -> str:
    total_deleted = int(deletion_frame['deleted_count'].sum()) if not deletion_frame.empty else 0
    direct_frame = deletion_frame[deletion_frame['category'] == DIRECT_LEAKAGE_LABEL]
    mechanistic_frame = deletion_frame[deletion_frame['category'] == MECHANISTIC_LABEL]
    other_frame = deletion_frame[deletion_frame['category'] == OTHER_LABEL]

    direct_deleted = int(direct_frame['deleted_count'].sum()) if not direct_frame.empty else 0
    mechanistic_deleted = (
        int(mechanistic_frame['deleted_count'].sum()) if not mechanistic_frame.empty else 0
    )
    other_deleted = int(other_frame['deleted_count'].sum()) if not other_frame.empty else 0

    lines = [
        '# Deleted Edge Audit Report',
        '',
        '## Config',
        '',
        f'- Processed split: `{processed_path}`',
        f'- Split mode: `{split_mode}`',
        f'- Target subset: `{target_subset}`',
        f'- Target path shape: `{tuple(target_paths.shape)}`',
        f'- Strict split isolation enabled: `{include_split_isolation}`',
        '',
        '## Summary',
        '',
        f'- Total deleted edges: `{total_deleted}`',
        f'- Direct leakage deletions: `{direct_deleted}`',
        f'- Mechanistic deletions: `{mechanistic_deleted}`',
        f'- Other deletions: `{other_deleted}`',
        '',
        '## Direct Leakage',
        '',
        format_markdown_table(direct_frame),
        '',
        '## Mechanistic Edges',
        '',
        format_markdown_table(mechanistic_frame),
        '',
        '## Other Deleted Edges',
        '',
        format_markdown_table(other_frame),
        '',
        '## All Deleted Edge Types',
        '',
        format_markdown_table(deletion_frame),
    ]
    return '\n'.join(lines)


def run_audit(args: argparse.Namespace) -> Dict[str, object]:
    split_mode, pair_splits = load_pair_splits(args.processed_path)

    processor, _ = build_raw_graph(
        nodes_csv=args.nodes_csv,
        edges_csv=args.edges_csv,
        train_triplets=[],
    )
    split_triplets = derive_ho_triplets_for_pair_splits(
        processor=processor,
        edge_csv_path=args.edges_csv,
        pair_splits=pair_splits,
    )

    raw_data = processor.build_heterodata(split_triplets['train'])
    target_triplets = resolve_target_triplets(
        split_triplets=split_triplets,
        target_subset=args.target_subset,
    )
    target_paths = resolve_target_paths(
        data=raw_data,
        target_triplets=target_triplets,
        path_format=args.path_format,
    )

    isolate_nodes_by_type: Mapping[str, Tensor] | None = None
    if not args.disable_split_isolation:
        isolate_nodes_by_type = build_split_isolation_targets(
            split_mode=split_mode,
            pair_splits=pair_splits,
        )

    clean_edge_index_dict = remove_leakage_edges(
        data=raw_data,
        target_paths=target_paths,
        isolate_nodes_by_type=isolate_nodes_by_type,
    )

    clean_data = copy.deepcopy(raw_data)
    for edge_type, edge_index in clean_edge_index_dict.items():
        clean_data[edge_type].edge_index = edge_index

    raw_counts = count_edges_by_type(raw_data.edge_index_dict)
    clean_counts = count_edges_by_type(clean_data.edge_index_dict)
    deletion_frame = build_deletion_dataframe(raw_counts=raw_counts, clean_counts=clean_counts)
    report_md = render_markdown_report(
        processed_path=args.processed_path,
        split_mode=split_mode,
        target_subset=args.target_subset,
        target_paths=target_paths,
        include_split_isolation=not args.disable_split_isolation,
        deletion_frame=deletion_frame,
    )

    return {
        'config': {
            'processed_path': str(args.processed_path),
            'nodes_csv': str(args.nodes_csv),
            'edges_csv': str(args.edges_csv),
            'split_mode': split_mode,
            'target_subset': args.target_subset,
            'path_format': args.path_format,
            'split_isolation_enabled': not args.disable_split_isolation,
        },
        'target_triplet_count': len(target_triplets),
        'target_path_shape': list(target_paths.shape),
        'deletion_rows': deletion_frame.to_dict(orient='records'),
        'report_markdown': report_md,
    }


def main() -> None:
    args = parse_args()
    payload = run_audit(args)
    print(payload['report_markdown'])

    if args.report_json is not None:
        args.report_json.parent.mkdir(parents=True, exist_ok=True)
        args.report_json.write_text(json.dumps(payload, indent=2), encoding='utf-8')
        print(f'\nSaved structured report to: {args.report_json}')


if __name__ == '__main__':
    main()
