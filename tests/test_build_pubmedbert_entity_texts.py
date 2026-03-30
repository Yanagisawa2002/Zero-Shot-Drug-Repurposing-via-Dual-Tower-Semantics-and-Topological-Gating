from __future__ import annotations

import csv
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.build_pubmedbert_entity_texts import (
    PrimeKGNodeRecord,
    build_graph_context_by_raw_id,
    build_name_phrase_index,
    sanitize_auxiliary_text,
)


def test_graph_context_builder_excludes_label_relations() -> None:
    node_records = [
        PrimeKGNodeRecord(0, 0, 'drug::DB1', 'drug', 'Aspirin', 'DrugBank'),
        PrimeKGNodeRecord(1, 0, 'gene/protein::101', 'gene/protein', 'PTGS1', 'NCBI'),
        PrimeKGNodeRecord(2, 0, 'disease::1', 'disease', 'Headache', 'MONDO'),
    ]
    workdir = PROJECT_ROOT / 'tests' / '_artifacts'
    workdir.mkdir(parents=True, exist_ok=True)
    edges_csv = workdir / 'build_pubmedbert_edges.csv'
    with edges_csv.open('w', encoding='utf-8', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=['src_id', 'src_type', 'dst_id', 'dst_type', 'rel'])
        writer.writeheader()
        writer.writerow({'src_id': 'drug::DB1', 'src_type': 'drug', 'dst_id': 'gene/protein::101', 'dst_type': 'gene/protein', 'rel': 'targets'})
        writer.writerow({'src_id': 'drug::DB1', 'src_type': 'drug', 'dst_id': 'disease::1', 'dst_type': 'disease', 'rel': 'indication'})
        writer.writerow({'src_id': 'drug::DB1', 'src_type': 'drug', 'dst_id': 'disease::1', 'dst_type': 'disease', 'rel': 'off-label use'})
        writer.writerow({'src_id': 'drug::DB1', 'src_type': 'drug', 'dst_id': 'disease::1', 'dst_type': 'disease', 'rel': 'contraindication'})

    graph_text = build_graph_context_by_raw_id(node_records=node_records, edges_csv=edges_csv, max_neighbors=8)

    assert graph_text['drug::DB1'] == 'targets: PTGS1.'
    assert 'Headache' not in graph_text['drug::DB1']
    assert 'indication' not in graph_text['drug::DB1'].casefold()
    assert 'off-label' not in graph_text['drug::DB1'].casefold()


def test_sanitize_auxiliary_text_drops_segments_with_blocklisted_entity_names() -> None:
    node_records = [
        PrimeKGNodeRecord(0, 0, 'drug::DB1', 'drug', 'Aspirin', 'DrugBank'),
        PrimeKGNodeRecord(1, 0, 'gene/protein::101', 'gene/protein', 'PTGS1', 'NCBI'),
        PrimeKGNodeRecord(2, 0, 'disease::1', 'disease', 'Headache', 'MONDO'),
    ]
    drug_phrase_index = build_name_phrase_index(node_records=node_records, node_type='drug')
    disease_phrase_index = build_name_phrase_index(node_records=node_records, node_type='disease')

    cleaned_drug_text = sanitize_auxiliary_text(
        node_type='drug',
        text='mechanism of action: used in Headache. drug type: small molecule.',
        drug_phrase_index=drug_phrase_index,
        disease_phrase_index=disease_phrase_index,
    )
    cleaned_gene_text = sanitize_auxiliary_text(
        node_type='gene/protein',
        text='associated diseases: Headache. target classes: enzyme.',
        drug_phrase_index=drug_phrase_index,
        disease_phrase_index=disease_phrase_index,
    )
    cleaned_disease_text = sanitize_auxiliary_text(
        node_type='disease',
        text='indicated drugs: Aspirin. ancestors: neurologic disease.',
        drug_phrase_index=drug_phrase_index,
        disease_phrase_index=disease_phrase_index,
    )

    assert cleaned_drug_text == 'drug type: small molecule.'
    assert cleaned_gene_text == 'target classes: enzyme.'
    assert cleaned_disease_text == 'ancestors: neurologic disease.'
