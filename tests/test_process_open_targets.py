from __future__ import annotations

import csv
import sys
from pathlib import Path

import pandas as pd
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.process_open_targets import filter_novel_ood_triplets, load_primekg_train_pairs


def test_filter_novel_ood_triplets_uses_pair_level_overlap() -> None:
    aligned_df = pd.DataFrame(
        {
            'primekg_drug_id': ['drug::A', 'drug::A', 'drug::B'],
            'primekg_target_id': ['gene::1', 'gene::2', 'gene::3'],
            'primekg_disease_id': ['disease::X', 'disease::X', 'disease::Y'],
        }
    )
    train_pair_df = pd.DataFrame(
        {
            'primekg_drug_id': ['drug::A'],
            'primekg_disease_id': ['disease::X'],
        }
    )

    overlap_df, novel_df, novelty_report = filter_novel_ood_triplets(
        aligned_triplets_df=aligned_df,
        primekg_train_pairs_df=train_pair_df,
    )

    assert len(overlap_df) == 2
    assert len(novel_df) == 1
    assert novelty_report['filter_level'] == 'pair'
    assert novelty_report['overlap_pairs'] == 1
    assert novelty_report['novel_pairs'] == 1


def test_load_primekg_train_pairs_reads_target_pairs_from_processed_pt() -> None:
    workdir = PROJECT_ROOT / 'tests' / '_artifacts'
    workdir.mkdir(parents=True, exist_ok=True)
    nodes_csv = workdir / 'process_ot_nodes.csv'
    with nodes_csv.open('w', encoding='utf-8', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=['id', 'type', 'name', 'source'])
        writer.writeheader()
        writer.writerow({'id': 'drug::A', 'type': 'drug', 'name': 'Drug A', 'source': 'DrugBank'})
        writer.writerow({'id': 'gene::1', 'type': 'gene/protein', 'name': 'GENE1', 'source': 'NCBI'})
        writer.writerow({'id': 'disease::X', 'type': 'disease', 'name': 'Disease X', 'source': 'MONDO'})

    processed_pt = workdir / 'process_ot_processed.pt'
    torch.save({'target_pairs': {'train': torch.tensor([[0, 2]], dtype=torch.long)}}, processed_pt)

    train_pairs_df = load_primekg_train_pairs(path=processed_pt, primekg_nodes_csv=nodes_csv)

    assert train_pairs_df.to_dict(orient='records') == [
        {'primekg_drug_id': 'drug::A', 'primekg_disease_id': 'disease::X'}
    ]
