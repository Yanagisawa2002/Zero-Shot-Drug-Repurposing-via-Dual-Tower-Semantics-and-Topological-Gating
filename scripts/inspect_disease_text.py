from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.diagnose_cross_disease_failure import build_type_specific_degree_maps
from scripts.extract_thick_disease_embs import (
    build_text_to_encode,
    build_thick_disease_table,
    load_base_diseases,
    load_definitions,
    load_ot_context,
    safe_text,
)
from scripts.train_quad_split_ho_probe import build_clean_graph_without_leakage
from scripts.train_split_probe import derive_ho_triplets_for_pair_splits, load_pair_splits
from src.primekg_data_processor import PrimeKGDataProcessor


DEFAULT_OT_TABLES = [
    Path(r'open target/Open_target_Phase_0.xlsx'),
    Path(r'open target/Open_target_Phase_1.xlsx'),
    Path(r'open target/Open_target_Phase_2.xlsx'),
    Path(r'open target/Open_target_Phase_3.xlsx'),
    Path(r'open target/Open_target_Phase_4.xlsx'),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Inspect raw thick disease texts for hard cross_disease test diseases.'
    )
    parser.add_argument('--processed-path', type=Path, default=Path('data/PrimeKG/processed/primekg_indication_cross_disease.pt'))
    parser.add_argument('--nodes-csv', type=Path, default=Path('data/PrimeKG/nodes.csv'))
    parser.add_argument('--edges-csv', type=Path, default=Path('data/PrimeKG/edges.csv'))
    parser.add_argument('--ot-table', type=Path, action='append', default=None)
    parser.add_argument('--definition-table', type=Path, default=None)
    parser.add_argument('--top-k', type=int, default=5)
    return parser.parse_args()


def load_cross_disease_artifacts(processed_path: Path, nodes_csv: Path, edges_csv: Path):
    split_mode, pair_splits = load_pair_splits(processed_path)
    if split_mode != 'cross_disease':
        raise ValueError(f'Expected cross_disease split, got {split_mode}')

    processor = PrimeKGDataProcessor(node_csv_path=nodes_csv, edge_csv_path=edges_csv)
    processor.build_entity_mappings()
    split_triplets = derive_ho_triplets_for_pair_splits(
        processor=processor,
        edge_csv_path=edges_csv,
        pair_splits=pair_splits,
    )
    full_data = processor.build_heterodata(ho_id_paths=split_triplets['train'], add_inverse_edges=False)
    heldout_triplets = split_triplets['valid'] + split_triplets['test']
    clean_data, _, _ = build_clean_graph_without_leakage(
        data=full_data,
        heldout_triplets=heldout_triplets,
        split_mode=split_mode,
        pair_splits=pair_splits,
        graph_surgery_mode='direct_only',
    )
    return split_mode, pair_splits, processor, clean_data


def build_disease_text_lookup(
    nodes_csv: Path,
    ot_tables: List[Path],
    definition_table: Path | None,
) -> Dict[str, Tuple[str, str]]:
    base_diseases = load_base_diseases(
        disease_table=nodes_csv,
        disease_id_column='id',
        disease_name_column='name',
        disease_type_column='type',
        disease_type_value='disease',
    )
    ot_context = load_ot_context(
        ot_tables=ot_tables,
        ot_id_column='MONDO_ID',
        ot_label_column='label',
        ot_ancestors_column='ancestors',
    )
    definition_context = load_definitions(
        definition_table=definition_table,
        definition_id_column='disease_id',
        definition_text_column='definition',
    )
    thick_table = build_thick_disease_table(
        base_diseases=base_diseases,
        ot_context=ot_context,
        definition_context=definition_context,
    )

    lookup: Dict[str, Tuple[str, str]] = {}
    for row in thick_table.itertuples(index=False):
        disease_id = str(row.disease_id)
        disease_name = safe_text(row.name)
        text = build_text_to_encode(
            name=safe_text(row.name),
            ancestors=safe_text(row.ancestors),
            definition=safe_text(row.definition),
        )
        lookup[disease_id] = (disease_name, text)
    return lookup


def select_hard_test_diseases(pair_splits, degree_maps, top_k: int) -> List[Tuple[int, int]]:
    gene_degree_map = degree_maps.get('gene/protein', {})
    phenotype_keys = [key for key in degree_maps.keys() if 'phenotype' in key]
    phenotype_degree_map: Dict[int, int] = {}
    for key in phenotype_keys:
        for disease_id, count in degree_maps[key].items():
            phenotype_degree_map[int(disease_id)] = phenotype_degree_map.get(int(disease_id), 0) + int(count)

    test_disease_ids = sorted({int(disease_id) for _, disease_id in pair_splits['test']})
    ranked = []
    for disease_id in test_disease_ids:
        effective_degree = int(gene_degree_map.get(disease_id, 0) + phenotype_degree_map.get(disease_id, 0))
        ranked.append((effective_degree, disease_id))
    ranked.sort(key=lambda item: (item[0], item[1]))
    return ranked[:top_k]


def main() -> None:
    args = parse_args()
    ot_tables = args.ot_table if args.ot_table is not None else DEFAULT_OT_TABLES

    split_mode, pair_splits, processor, clean_data = load_cross_disease_artifacts(
        processed_path=args.processed_path,
        nodes_csv=args.nodes_csv,
        edges_csv=args.edges_csv,
    )
    degree_maps = build_type_specific_degree_maps(clean_data)
    chosen = select_hard_test_diseases(pair_splits=pair_splits, degree_maps=degree_maps, top_k=args.top_k)
    text_lookup = build_disease_text_lookup(
        nodes_csv=args.nodes_csv,
        ot_tables=ot_tables,
        definition_table=args.definition_table,
    )

    print('# Thick Disease Text Inspection')
    print()
    print('LLM purification used for thick disease text: NO')
    print('Source fields used: PrimeKG disease name + OT label/ancestors + optional definition')
    print(f'Split mode: {split_mode}')
    print(f'Chosen hard test diseases: {len(chosen)}')
    print()

    for index, (effective_degree, disease_gid) in enumerate(chosen, start=1):
        raw_id = processor.id2entity[disease_gid].raw_id
        fallback_name = processor.id2entity[disease_gid].name
        disease_name, thick_text = text_lookup.get(raw_id, (fallback_name, ''))
        print(f'## Disease {index}')
        print(f'Disease Global ID: {disease_gid}')
        print(f'Disease Raw ID: {raw_id}')
        print(f'Disease Name: {disease_name}')
        print(f'Effective Degree (gene/protein + phenotype): {effective_degree}')
        print('Thick Text Content:')
        print(thick_text)
        print()


if __name__ == '__main__':
    main()
