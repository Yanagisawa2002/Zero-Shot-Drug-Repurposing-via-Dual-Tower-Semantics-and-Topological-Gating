from __future__ import annotations

import argparse
import csv
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import pandas as pd
from tqdm import tqdm

try:
    import pubchempy as pcp
except ImportError as exc:  # pragma: no cover - ???????
    pcp = None
    PUBCHEMPY_IMPORT_ERROR = exc
else:  # pragma: no cover - ?????????
    PUBCHEMPY_IMPORT_ERROR = None


DEFAULT_NAME_CANDIDATES: Sequence[str] = ('drug_name', 'name', 'prefName', 'preferred_name')
DEFAULT_TYPE_CANDIDATES: Sequence[str] = ('type', 'node_type', 'category')
RESUME_SKIP_STATUSES = {'FOUND', 'NOT_FOUND'}


@dataclass
class SmilesFetchResult:
    """????? SMILES ?????"""

    drug_name: str
    queried_name: str
    smiles: str
    status: str
    cid: Optional[int]
    error_message: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Fetch canonical/isomeric SMILES from PubChem for unique drugs in a CSV file.',
    )
    parser.add_argument(
        '--input-csv',
        type=Path,
        default=Path('data/PrimeKG/nodes.csv'),
        help='?? CSV????? PrimeKG nodes.csv?',
    )
    parser.add_argument(
        '--output-csv',
        type=Path,
        default=Path('drug_smiles_mapping.csv'),
        help='??????????????????? drug_smiles_mapping.csv?',
    )
    parser.add_argument(
        '--drug-name-column',
        type=str,
        default=None,
        help='????????????????????????',
    )
    parser.add_argument(
        '--type-column',
        type=str,
        default=None,
        help='???????????????? --drug-type-value ???',
    )
    parser.add_argument(
        '--drug-type-value',
        type=str,
        default='drug',
        help='???????????? drug?',
    )
    parser.add_argument(
        '--flush-every',
        type=int,
        default=50,
        help='?????????????????? 50?',
    )
    parser.add_argument(
        '--sleep-seconds',
        type=float,
        default=0.3,
        help='?? PubChem API ????????????? 0.3 ??',
    )
    parser.add_argument(
        '--retry-sleep-seconds',
        type=float,
        default=2.0,
        help='??????????????? 2 ??',
    )
    parser.add_argument(
        '--max-drugs',
        type=int,
        default=None,
        help='???? N ????????????????',
    )
    return parser.parse_args()


def normalize_drug_name(raw_name: object) -> str:
    """???????????????????????"""
    return ' '.join(str(raw_name).strip().split())


def resolve_column_name(
    available_columns: Iterable[str],
    explicit_name: Optional[str],
    fallback_candidates: Sequence[str],
    column_role: str,
) -> Optional[str]:
    columns = list(available_columns)
    lower_to_original = {column.lower(): column for column in columns}

    if explicit_name is not None:
        if explicit_name in columns:
            return explicit_name
        explicit_lower = explicit_name.lower()
        if explicit_lower in lower_to_original:
            return lower_to_original[explicit_lower]
        raise ValueError(
            f'?????? {column_role} ? `{explicit_name}`????: {columns}'
        )

    for candidate in fallback_candidates:
        candidate_lower = candidate.lower()
        if candidate_lower in lower_to_original:
            return lower_to_original[candidate_lower]
    return None


def extract_unique_drugs(
    input_csv: Path,
    drug_name_column: Optional[str] = None,
    type_column: Optional[str] = None,
    drug_type_value: str = 'drug',
) -> List[str]:
    """
    ??? CSV ??????????

    ???
    - ???? type ????? type == drug_type_value ???
    - ?????????????????????
    """
    if not input_csv.exists():
        raise FileNotFoundError(f'???????: {input_csv}')

    frame = pd.read_csv(input_csv)
    if frame.empty:
        return []

    resolved_name_column = resolve_column_name(
        available_columns=frame.columns,
        explicit_name=drug_name_column,
        fallback_candidates=DEFAULT_NAME_CANDIDATES,
        column_role='????',
    )
    if resolved_name_column is None:
        raise ValueError(
            '??????????????? --drug-name-column ?????'
        )

    resolved_type_column = resolve_column_name(
        available_columns=frame.columns,
        explicit_name=type_column,
        fallback_candidates=DEFAULT_TYPE_CANDIDATES,
        column_role='????',
    )

    filtered_frame = frame
    if resolved_type_column is not None:
        type_series = filtered_frame[resolved_type_column].astype(str).str.strip().str.lower()
        filtered_frame = filtered_frame[type_series == drug_type_value.strip().lower()].copy()

    if filtered_frame.empty:
        return []

    drug_names = (
        filtered_frame[resolved_name_column]
        .dropna()
        .map(normalize_drug_name)
        .replace('', pd.NA)
        .dropna()
        .drop_duplicates()
        .tolist()
    )
    return sorted(drug_names)


def load_existing_results(output_csv: Path) -> Dict[str, str]:
    """
    ????????????????

    ???????????????
    - FOUND
    - NOT_FOUND
    ERROR ????????
    """
    if not output_csv.exists():
        return {}

    existing_frame = pd.read_csv(output_csv)
    if existing_frame.empty or 'drug_name' not in existing_frame.columns:
        return {}

    status_by_drug: Dict[str, str] = {}
    status_column = 'status' if 'status' in existing_frame.columns else None
    smiles_column = 'smiles' if 'smiles' in existing_frame.columns else None

    for row in existing_frame.itertuples(index=False):
        drug_name = normalize_drug_name(getattr(row, 'drug_name'))
        if not drug_name:
            continue

        if status_column is not None:
            status = str(getattr(row, status_column)).strip().upper()
        elif smiles_column is not None:
            smiles_value = str(getattr(row, smiles_column)).strip()
            status = 'FOUND' if smiles_value and smiles_value not in {'ERROR', 'NOT_FOUND'} else smiles_value
        else:
            status = 'FOUND'
        status_by_drug[drug_name] = status
    return status_by_drug


def query_pubchem_smiles(
    drug_name: str,
    sleep_seconds: float = 0.3,
    retry_sleep_seconds: float = 2.0,
) -> SmilesFetchResult:
    """
    ??????? PubChem ???

    ???
    - ??????? 2 ??????
    - ?? API ????? sleep????????
    - ???????? NOT_FOUND??????? ERROR?
    """
    if pcp is None:
        raise ImportError(
            '?? pubchempy ??????? `python -m pip install pubchempy`?'
        ) from PUBCHEMPY_IMPORT_ERROR

    last_error_message = ''
    max_attempts = 2

    for attempt in range(max_attempts):
        try:
            compounds = pcp.get_compounds(drug_name, 'name')
            time.sleep(sleep_seconds)

            if not compounds:
                return SmilesFetchResult(
                    drug_name=drug_name,
                    queried_name=drug_name,
                    smiles='NOT_FOUND',
                    status='NOT_FOUND',
                    cid=None,
                    error_message='',
                )

            compound = compounds[0]
            smiles = getattr(compound, 'canonical_smiles', None) or getattr(compound, 'isomeric_smiles', None)
            if not smiles:
                return SmilesFetchResult(
                    drug_name=drug_name,
                    queried_name=drug_name,
                    smiles='NOT_FOUND',
                    status='NOT_FOUND',
                    cid=getattr(compound, 'cid', None),
                    error_message='PubChem returned a compound but no SMILES field was available.',
                )

            return SmilesFetchResult(
                drug_name=drug_name,
                queried_name=drug_name,
                smiles=str(smiles),
                status='FOUND',
                cid=getattr(compound, 'cid', None),
                error_message='',
            )
        except Exception as exc:  # pragma: no cover - ??????/API ??
            last_error_message = str(exc)
            if attempt + 1 < max_attempts:
                time.sleep(retry_sleep_seconds)
            else:
                return SmilesFetchResult(
                    drug_name=drug_name,
                    queried_name=drug_name,
                    smiles='ERROR',
                    status='ERROR',
                    cid=None,
                    error_message=last_error_message,
                )

    return SmilesFetchResult(
        drug_name=drug_name,
        queried_name=drug_name,
        smiles='ERROR',
        status='ERROR',
        cid=None,
        error_message=last_error_message,
    )


def append_results(output_csv: Path, rows: Sequence[SmilesFetchResult]) -> None:
    """????????? CSV?"""
    if not rows:
        return

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    file_exists = output_csv.exists()
    fieldnames = list(asdict(rows[0]).keys())

    with output_csv.open('a', encoding='utf-8', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def main() -> None:
    args = parse_args()

    unique_drugs = extract_unique_drugs(
        input_csv=args.input_csv,
        drug_name_column=args.drug_name_column,
        type_column=args.type_column,
        drug_type_value=args.drug_type_value,
    )
    if args.max_drugs is not None:
        unique_drugs = unique_drugs[: args.max_drugs]

    existing_status = load_existing_results(args.output_csv)
    pending_drugs = [
        drug_name
        for drug_name in unique_drugs
        if existing_status.get(drug_name, '').upper() not in RESUME_SKIP_STATUSES
    ]

    print(f'Input CSV: {args.input_csv}')
    print(f'Unique drugs found: {len(unique_drugs)}')
    print(f'Already processed and skipped: {len(unique_drugs) - len(pending_drugs)}')
    print(f'Pending drugs to query: {len(pending_drugs)}')
    print(f'Output CSV: {args.output_csv}')

    buffer: List[SmilesFetchResult] = []
    found_count = 0
    not_found_count = 0
    error_count = 0

    progress_bar = tqdm(pending_drugs, desc='Fetching SMILES', unit='drug')
    for index, drug_name in enumerate(progress_bar, start=1):
        result = query_pubchem_smiles(
            drug_name=drug_name,
            sleep_seconds=args.sleep_seconds,
            retry_sleep_seconds=args.retry_sleep_seconds,
        )
        buffer.append(result)

        if result.status == 'FOUND':
            found_count += 1
        elif result.status == 'NOT_FOUND':
            not_found_count += 1
        else:
            error_count += 1

        processed_count = found_count + not_found_count + error_count
        success_rate = found_count / processed_count if processed_count > 0 else 0.0
        progress_bar.set_postfix(
            found=found_count,
            not_found=not_found_count,
            error=error_count,
            success_rate=f'{success_rate:.2%}',
        )

        if index % args.flush_every == 0:
            append_results(args.output_csv, buffer)
            buffer.clear()

    append_results(args.output_csv, buffer)

    total_processed = found_count + not_found_count + error_count
    final_success_rate = found_count / total_processed if total_processed > 0 else 0.0
    print('Finished fetching SMILES.')
    print(f'Processed this run: {total_processed}')
    print(f'FOUND: {found_count}')
    print(f'NOT_FOUND: {not_found_count}')
    print(f'ERROR: {error_count}')
    print(f'Success rate (FOUND / processed_this_run): {final_success_rate:.2%}')


if __name__ == '__main__':
    main()
