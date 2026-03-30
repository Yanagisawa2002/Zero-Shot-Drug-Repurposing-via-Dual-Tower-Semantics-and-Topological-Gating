from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem import AllChem
from tqdm import tqdm


DEFAULT_INPUT_CSV = Path('outputs/static_smiles/primekg_drugs_with_smiles_final.csv')
DEFAULT_OUTPUT_PKL = Path('drug_morgan_fingerprints.pkl')
INVALID_SMILES_VALUES = {'NOT_FOUND', 'ERROR', ''}


def parse_args() -> argparse.Namespace:
    """????????"""

    parser = argparse.ArgumentParser(
        description='Extract 1024-d Morgan fingerprints from a SMILES table.'
    )
    parser.add_argument(
        '--input-csv',
        type=Path,
        default=DEFAULT_INPUT_CSV,
        help='?????? ID ? SMILES ? CSV ?????',
    )
    parser.add_argument(
        '--output-pkl',
        type=Path,
        default=DEFAULT_OUTPUT_PKL,
        help='??? Morgan ???? pkl ?????',
    )
    parser.add_argument(
        '--node-id-column',
        type=str,
        default='node_id',
        help='???? ID ???',
    )
    parser.add_argument(
        '--smiles-column',
        type=str,
        default='smiles',
        help='SMILES ???',
    )
    parser.add_argument(
        '--radius',
        type=int,
        default=2,
        help='Morgan ?????',
    )
    parser.add_argument(
        '--n-bits',
        type=int,
        default=1024,
        help='Morgan ?????',
    )
    return parser.parse_args()


def is_missing_smiles(smiles_value: object) -> bool:
    """???? SMILES ?????????"""

    if smiles_value is None:
        return True
    if pd.isna(smiles_value):
        return True

    smiles_text = str(smiles_value).strip()
    return smiles_text in INVALID_SMILES_VALUES


def smiles_to_morgan_tensor(smiles: str, radius: int, n_bits: int) -> torch.Tensor | None:
    """
    ??? SMILES ????? Morgan ?????

    ? SMILES ??? RDKit ???????? None?
    """

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    bit_vector = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=n_bits)
    fingerprint = torch.tensor(
        list(map(int, bit_vector.ToBitString())),
        dtype=torch.float32,
    )
    return fingerprint


def build_fingerprints_dict(
    dataframe: pd.DataFrame,
    node_id_column: str,
    smiles_column: str,
    radius: int,
    n_bits: int,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, int]]:
    """
    ??????? `node_id -> Morgan fingerprint tensor` ???

    ???
    - fingerprints_dict: ????
    - stats: ??????
    """

    if node_id_column not in dataframe.columns:
        raise KeyError(f'CSV ????? ID ??{node_id_column}')
    if smiles_column not in dataframe.columns:
        raise KeyError(f'CSV ??? SMILES ??{smiles_column}')

    zero_tensor = torch.zeros(n_bits, dtype=torch.float32)
    fingerprints_dict: Dict[str, torch.Tensor] = {}
    success_count = 0
    zero_fill_count = 0
    invalid_smiles_count = 0
    missing_smiles_count = 0

    iterator = tqdm(
        dataframe.itertuples(index=False),
        total=len(dataframe),
        desc='Extracting Morgan fingerprints',
    )
    for row in iterator:
        node_id = str(getattr(row, node_id_column))
        smiles_value = getattr(row, smiles_column)

        if is_missing_smiles(smiles_value):
            fingerprints_dict[node_id] = zero_tensor.clone()
            zero_fill_count += 1
            missing_smiles_count += 1
            continue

        smiles = str(smiles_value).strip()
        fingerprint = smiles_to_morgan_tensor(smiles=smiles, radius=radius, n_bits=n_bits)
        if fingerprint is None:
            fingerprints_dict[node_id] = zero_tensor.clone()
            zero_fill_count += 1
            invalid_smiles_count += 1
            continue

        fingerprints_dict[node_id] = fingerprint
        success_count += 1

    stats = {
        'total_rows': int(len(dataframe)),
        'success_count': int(success_count),
        'zero_fill_count': int(zero_fill_count),
        'missing_smiles_count': int(missing_smiles_count),
        'invalid_smiles_count': int(invalid_smiles_count),
    }
    return fingerprints_dict, stats


def save_fingerprints_dict(output_pkl: Path, fingerprints_dict: Dict[str, torch.Tensor]) -> None:
    """????????? pickle ???"""

    output_pkl.parent.mkdir(parents=True, exist_ok=True)
    with output_pkl.open('wb') as file:
        pickle.dump(fingerprints_dict, file)


def main() -> None:
    args = parse_args()

    print(f'Loading CSV from: {args.input_csv}')
    dataframe = pd.read_csv(args.input_csv)
    print(f'Total rows: {len(dataframe)}')

    fingerprints_dict, stats = build_fingerprints_dict(
        dataframe=dataframe,
        node_id_column=args.node_id_column,
        smiles_column=args.smiles_column,
        radius=args.radius,
        n_bits=args.n_bits,
    )
    save_fingerprints_dict(output_pkl=args.output_pkl, fingerprints_dict=fingerprints_dict)

    print('\nExtraction complete.')
    print(f"Total rows: {stats['total_rows']}")
    print(f"Successfully converted: {stats['success_count']}")
    print(f"Zero-filled: {stats['zero_fill_count']}")
    print(f"  Missing SMILES: {stats['missing_smiles_count']}")
    print(f"  Invalid SMILES parse: {stats['invalid_smiles_count']}")
    print(f'Saved to: {args.output_pkl}')


if __name__ == '__main__':
    main()
