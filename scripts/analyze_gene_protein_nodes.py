from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd


def _pick_column(df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
    normalized = {column.lower(): column for column in df.columns}
    for candidate in candidates:
        if candidate.lower() in normalized:
            return normalized[candidate.lower()]
    return None


def _infer_id_pattern(id_series: pd.Series) -> str:
    cleaned = id_series.dropna().astype(str).str.strip()
    if cleaned.empty:
        return 'No non-null IDs found.'

    suffixes = cleaned.str.split('::').str[-1]
    pure_numeric_ratio = suffixes.str.fullmatch(r'\d+').mean()
    alpha_numeric_ratio = suffixes.str.fullmatch(r'[A-Za-z0-9._-]+').mean()

    examples = cleaned.head(5).tolist()
    if pure_numeric_ratio > 0.95:
        return (
            'IDs are mostly prefix+numeric suffix (for example gene/protein::9796), '
            'which strongly suggests NCBI Entrez-style identifiers.'
            f' Example IDs: {examples}'
        )
    if alpha_numeric_ratio > 0.95:
        return (
            'IDs are mostly alphanumeric suffixes, which is more consistent with '
            'UniProt/HGNC-like identifier formats.'
            f' Example IDs: {examples}'
        )
    return f'Mixed ID pattern detected. Example IDs: {examples}'


def main() -> None:
    nodes_path = Path('data/PrimeKG/nodes.csv')
    df = pd.read_csv(nodes_path)

    id_col = _pick_column(df, ['node_id', 'id'])
    type_col = _pick_column(df, ['node_type', 'type'])
    name_col = _pick_column(df, ['node_name', 'name'])

    if id_col is None or type_col is None or name_col is None:
        raise ValueError(
            f'Could not detect required columns. Found columns: {df.columns.tolist()}'
        )

    type_series = df[type_col].astype(str).str.lower()
    mask = (
        type_series.str.contains('gene/protein', regex=False)
        | type_series.str.contains('gene', regex=False)
        | type_series.str.contains('protein', regex=False)
    )
    gene_df = df.loc[mask].copy()

    print('=== Gene/Protein Node Audit ===')
    print(f'Input file: {nodes_path}')
    print(f'Detected columns -> id: {id_col}, type: {type_col}, name: {name_col}')
    print(f'Total gene/protein-like nodes: {len(gene_df)}')
    print()

    missing_id = int(gene_df[id_col].isna().sum())
    missing_name = int(gene_df[name_col].isna().sum())
    empty_name = int(gene_df[name_col].astype(str).str.strip().eq('').sum())
    print('=== Missingness ===')
    print(f'Missing {id_col}: {missing_id}')
    print(f'Missing {name_col}: {missing_name}')
    print(f'Empty-string {name_col}: {empty_name}')
    print()

    print('=== ID Pattern Heuristic ===')
    print(_infer_id_pattern(gene_df[id_col]))
    print()

    print('=== First 5 Rows ===')
    print(gene_df.head(5).to_string(index=False))
    print()

    print('=== Last 5 Rows ===')
    print(gene_df.tail(5).to_string(index=False))


if __name__ == '__main__':
    main()
