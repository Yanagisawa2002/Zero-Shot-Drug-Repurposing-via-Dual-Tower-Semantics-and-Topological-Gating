from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


STATIC_SMILES_URL = (
    "https://raw.githubusercontent.com/choderalab/nano-drugbank/master/df_drugbank_smiles.csv"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge a static DrugBank-to-SMILES table with PrimeKG drugs.",
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=Path("data/PrimeKG/nodes.csv"),
        help="Input CSV containing PrimeKG drugs or a pre-extracted drug table.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("outputs/static_smiles/primekg_drugs_with_smiles.csv"),
        help="Output CSV path.",
    )
    return parser.parse_args()


def load_primekg_drugs(primekg_drugs_csv: Path) -> pd.DataFrame:
    """
    Load a drug table and normalize it to contain a `node_id` column.

    Supported inputs:
    - PrimeKG `nodes.csv` with columns: `id`, `type`, `name`, ...
    - A pre-built CSV with a `node_id` column already present.
    """
    if not primekg_drugs_csv.exists():
        raise FileNotFoundError(f"Input CSV does not exist: {primekg_drugs_csv}")

    my_drugs = pd.read_csv(primekg_drugs_csv)
    if my_drugs.empty:
        raise ValueError("Input CSV is empty.")

    if "node_id" in my_drugs.columns:
        normalized = my_drugs.copy()
    elif {"id", "type"}.issubset(my_drugs.columns):
        normalized = my_drugs.copy()
        normalized = normalized[
            normalized["type"].astype(str).str.strip().str.lower() == "drug"
        ].copy()
        normalized = normalized.rename(columns={"id": "node_id"})
    else:
        raise ValueError(
            "Input CSV must contain either `node_id`, or PrimeKG-style `id` and `type` columns."
        )

    normalized["node_id"] = normalized["node_id"].astype(str).str.strip()
    normalized = normalized[normalized["node_id"].str.startswith("drug::")].copy()
    normalized = normalized.drop_duplicates(subset=["node_id"])
    return normalized


def map_smiles_from_static_repo(primekg_drugs_csv: Path, output_csv: Path) -> None:
    print("1. Downloading static DrugBank SMILES mapping from open-source repo...")

    db_df = pd.read_csv(STATIC_SMILES_URL, usecols=["drugbank_id", "smiles"])
    db_df["drugbank_id"] = db_df["drugbank_id"].astype(str).str.strip()
    db_df["node_id"] = "drug::" + db_df["drugbank_id"]
    db_df = db_df.drop_duplicates(subset=["node_id"])
    print(f"Loaded {len(db_df)} unique SMILES from static database.")

    print()
    print("2. Loading your PrimeKG drugs...")
    my_drugs = load_primekg_drugs(primekg_drugs_csv)
    original_count = len(my_drugs)

    print()
    print("3. Merging...")
    merged_df = pd.merge(my_drugs, db_df[["node_id", "smiles"]], on="node_id", how="left")

    matched_count = int(merged_df["smiles"].notna().sum())
    missing_count = int(original_count - matched_count)

    print("Merge Complete!")
    print(f"Total Drugs: {original_count}")
    print(f"Matched SMILES: {matched_count} ({(matched_count / original_count) * 100:.2f}%)")
    print(f"Missing SMILES: {missing_count} ({(missing_count / original_count) * 100:.2f}%)")

    merged_df["smiles"] = merged_df["smiles"].fillna("NOT_FOUND")
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    merged_df.to_csv(output_csv, index=False)
    print()
    print(f"Final mapping saved to {output_csv}")


if __name__ == "__main__":
    args = parse_args()
    map_smiles_from_static_repo(
        primekg_drugs_csv=args.input_csv,
        output_csv=args.output_csv,
    )
