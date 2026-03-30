from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd
import pubchempy as pcp
from tqdm import tqdm


VALID_TERMINAL_STATUSES = {'NOT_FOUND', 'ERROR'}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Patch missing SMILES in a static-merged drug table using PubChem fallback.',
    )
    parser.add_argument(
        '--input-csv',
        type=Path,
        default=Path('outputs/static_smiles/primekg_drugs_with_smiles.csv'),
        help='Input CSV produced by merge_smiles.py.',
    )
    parser.add_argument(
        '--output-csv',
        type=Path,
        default=Path('outputs/static_smiles/primekg_drugs_with_smiles_final.csv'),
        help='Output CSV with patched SMILES.',
    )
    parser.add_argument(
        '--drug-column',
        type=str,
        default='name',
        help='Column containing the real drug names to query in PubChem.',
    )
    parser.add_argument(
        '--flush-every',
        type=int,
        default=50,
        help='Persist merged output every N processed drugs.',
    )
    parser.add_argument(
        '--sleep-seconds',
        type=float,
        default=0.3,
        help='Sleep after each PubChem call to respect rate limits.',
    )
    parser.add_argument(
        '--retry-sleep-seconds',
        type=float,
        default=2.0,
        help='Sleep before the retry after a failed request.',
    )
    parser.add_argument(
        '--max-drugs',
        type=int,
        default=None,
        help='Only patch the first N missing drugs, useful for debugging.',
    )
    return parser.parse_args()


def fetch_smiles_with_retry(drug_name: str, retry_sleep_seconds: float) -> str:
    smiles = 'NOT_FOUND'
    try:
        compounds = pcp.get_compounds(drug_name, 'name')
        if compounds:
            smiles = compounds[0].canonical_smiles or compounds[0].isomeric_smiles or 'NOT_FOUND'
    except Exception:
        time.sleep(retry_sleep_seconds)
        try:
            compounds = pcp.get_compounds(drug_name, 'name')
            if compounds:
                smiles = compounds[0].canonical_smiles or compounds[0].isomeric_smiles or 'NOT_FOUND'
        except Exception:
            smiles = 'ERROR'
    return smiles if smiles else 'NOT_FOUND'


def patch_missing_smiles(
    input_csv: Path,
    output_csv: Path,
    drug_column: str = 'name',
    flush_every: int = 50,
    sleep_seconds: float = 0.3,
    retry_sleep_seconds: float = 2.0,
    max_drugs: int | None = None,
) -> None:
    print(f'Loading merged dataset from {input_csv}...')
    df = pd.read_csv(input_csv)

    if 'smiles' not in df.columns:
        raise ValueError('Input CSV must contain a `smiles` column.')
    if drug_column not in df.columns:
        raise ValueError(f'Drug name column `{drug_column}` not found. Available columns: {list(df.columns)}')

    missing_mask = df['smiles'].isin(['NOT_FOUND', 'ERROR']) | df['smiles'].isna()
    missing_drugs_df = df[missing_mask].copy()
    unique_missing_drugs = missing_drugs_df[drug_column].dropna().astype(str).str.strip().unique().tolist()

    if max_drugs is not None:
        unique_missing_drugs = unique_missing_drugs[:max_drugs]

    print(f'Found {len(unique_missing_drugs)} unique drugs missing SMILES. Starting PubChem fetch...')

    patched_dict: dict[str, str] = {}
    if output_csv.exists():
        temp_df = pd.read_csv(output_csv)
        if drug_column not in temp_df.columns or 'smiles' not in temp_df.columns:
            raise ValueError(
                f'Existing output CSV must contain `{drug_column}` and `smiles` columns: {output_csv}'
            )
        valid_patched = temp_df[~temp_df['smiles'].isin(['NOT_FOUND', 'ERROR'])].copy()
        patched_dict = dict(
            zip(
                valid_patched[drug_column].astype(str).str.strip(),
                valid_patched['smiles'].astype(str).str.strip(),
            )
        )
        print(f'Resumed from checkpoint. {len(patched_dict)} already patched.')

    results_count = 0
    progress_bar = tqdm(unique_missing_drugs, desc='Patching Missing SMILES', unit='drug')
    for drug in progress_bar:
        if drug in patched_dict:
            continue

        smiles = fetch_smiles_with_retry(
            drug_name=drug,
            retry_sleep_seconds=retry_sleep_seconds,
        )
        patched_dict[drug] = smiles
        results_count += 1
        time.sleep(sleep_seconds)

        resolved_count = sum(1 for value in patched_dict.values() if value not in VALID_TERMINAL_STATUSES)
        progress_bar.set_postfix(
            resolved=resolved_count,
            unresolved=len(patched_dict) - resolved_count,
        )

        if results_count % flush_every == 0:
            df['patched_smiles'] = df[drug_column].astype(str).str.strip().map(patched_dict)
            df['smiles'] = df.apply(
                lambda row: row['patched_smiles'] if pd.notna(row['patched_smiles']) else row['smiles'],
                axis=1,
            )
            df.drop(columns=['patched_smiles'], inplace=True, errors='ignore')
            output_csv.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(output_csv, index=False)

    print()
    print('Merge complete. Saving final dataset...')
    df['patched_smiles'] = df[drug_column].astype(str).str.strip().map(patched_dict)
    df['smiles'] = df.apply(
        lambda row: row['patched_smiles'] if pd.notna(row['patched_smiles']) else row['smiles'],
        axis=1,
    )
    df.drop(columns=['patched_smiles'], inplace=True, errors='ignore')

    final_missing = int(df['smiles'].isin(['NOT_FOUND', 'ERROR']).sum())
    total = int(len(df))
    final_matched = total - final_missing
    print(f'Total Drugs: {total}')
    print(f'Final Matched: {final_matched} ({(final_matched / total) * 100:.2f}%)')
    print(f'Still Missing: {final_missing} (Biologics/Complex mixtures/No data)')

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f'Saved to {output_csv}!')


if __name__ == '__main__':
    args = parse_args()
    patch_missing_smiles(
        input_csv=args.input_csv,
        output_csv=args.output_csv,
        drug_column=args.drug_column,
        flush_every=args.flush_every,
        sleep_seconds=args.sleep_seconds,
        retry_sleep_seconds=args.retry_sleep_seconds,
        max_drugs=args.max_drugs,
    )
