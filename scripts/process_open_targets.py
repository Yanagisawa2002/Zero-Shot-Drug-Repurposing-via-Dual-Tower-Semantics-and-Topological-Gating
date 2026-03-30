from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import pandas as pd


TripletColumns = Tuple[str, str, str]


@dataclass(frozen=True)
class PrimeKGAliasIndex:
    """PrimeKG 的轻量别名索引，用于把外部 OT 实体解析回 PrimeKG 实体。"""

    lookup_by_type: Dict[str, Dict[str, str]]
    raw_id_to_name: Dict[str, str]
    global_id_to_raw_id: List[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='将 Open Targets 数据处理为 PrimeKG 可理解的 (drug, gene/protein, disease) 高阶机制路径。'
    )
    parser.add_argument('--ot-drug-target-path', type=Path, required=True, help='OT 药物-靶点关联表。')
    parser.add_argument('--ot-target-disease-path', type=Path, required=True, help='OT 靶点-疾病关联表。')
    parser.add_argument('--primekg-nodes-csv', type=Path, required=True, help='PrimeKG 的 nodes.csv。')
    parser.add_argument(
        '--primekg-train-paths',
        type=Path,
        required=True,
        help='PrimeKG 训练集 HO 路径，可为 .pt/.csv/.tsv/.parquet/.json/.jsonl。',
    )
    parser.add_argument('--drug-mapping-path', type=Path, default=None, help='OT Drug -> PrimeKG Drug 的外部映射。')
    parser.add_argument('--target-mapping-path', type=Path, default=None, help='OT Target -> PrimeKG Gene 的外部映射。')
    parser.add_argument('--disease-mapping-path', type=Path, default=None, help='OT Disease -> PrimeKG Disease 的外部映射。')
    parser.add_argument('--output-dir', type=Path, required=True, help='输出目录。')
    parser.add_argument('--save-intermediate', action='store_true', help='是否保存中间结果表。')
    return parser.parse_args()


def load_ot_drug_target_associations(path: Path) -> pd.DataFrame:
    """
    加载 OT 的药物-靶点关联数据，并统一字段命名。

    目标输出字段：
    - `ot_drug_id`: OT 侧药物标识，通常为 ChEMBL ID。
    - `ot_target_id`: OT 侧靶点标识，通常为 Ensembl ID。
    - `ot_drug_name`: 可选，药物名称。
    - `ot_target_name`: 可选，基因/蛋白名称。
    """

    table = _load_table(path)
    table = _standardize_dataframe_columns(table)

    drug_id_col = _pick_first_existing_column(
        table,
        candidates=[
            'drug_id',
            'drugid',
            'chembl_id',
            'molecule_chembl_id',
            'drug_chembl_id',
            'molecule_id',
        ],
        table_name='ot_drug_target',
    )
    target_id_col = _pick_first_existing_column(
        table,
        candidates=[
            'target_id',
            'targetid',
            'target_from_source_id',
            'ensembl_id',
            'target_ensembl_id',
            'gene_id',
        ],
        table_name='ot_drug_target',
    )

    selected = pd.DataFrame(
        {
            'ot_drug_id': table[drug_id_col],
            'ot_target_id': table[target_id_col],
        }
    )

    drug_name_col = _pick_optional_column(
        table,
        ['drug_name', 'drug', 'molecule_name', 'drug_from_source', 'pref_name'],
    )
    target_name_col = _pick_optional_column(
        table,
        ['target_name', 'approved_symbol', 'symbol', 'gene_symbol', 'target_symbol'],
    )
    if drug_name_col is not None:
        selected['ot_drug_name'] = table[drug_name_col]
    if target_name_col is not None:
        selected['ot_target_name'] = table[target_name_col]

    selected = _drop_null_and_deduplicate(
        selected,
        required_columns=['ot_drug_id', 'ot_target_id'],
        dedup_columns=['ot_drug_id', 'ot_target_id'],
    )
    return selected


def load_ot_target_disease_associations(path: Path) -> pd.DataFrame:
    """
    加载 OT 的靶点-疾病关联数据，并统一字段命名。

    目标输出字段：
    - `ot_target_id`: OT 侧靶点标识，通常为 Ensembl ID。
    - `ot_disease_id`: OT 侧疾病标识，通常为 EFO/OT Disease ID。
    - `ot_target_name`: 可选，靶点名称。
    - `ot_disease_name`: 可选，疾病名称。
    """

    table = _load_table(path)
    table = _standardize_dataframe_columns(table)

    target_id_col = _pick_first_existing_column(
        table,
        candidates=[
            'target_id',
            'targetid',
            'target_from_source_id',
            'ensembl_id',
            'target_ensembl_id',
            'gene_id',
        ],
        table_name='ot_target_disease',
    )
    disease_id_col = _pick_first_existing_column(
        table,
        candidates=[
            'disease_id',
            'diseaseid',
            'disease_from_source_mapped_id',
            'efo_id',
            'disease_efo_id',
        ],
        table_name='ot_target_disease',
    )

    selected = pd.DataFrame(
        {
            'ot_target_id': table[target_id_col],
            'ot_disease_id': table[disease_id_col],
        }
    )

    target_name_col = _pick_optional_column(
        table,
        ['target_name', 'approved_symbol', 'symbol', 'gene_symbol', 'target_symbol'],
    )
    disease_name_col = _pick_optional_column(
        table,
        ['disease_name', 'disease', 'disease_label', 'disease_from_source'],
    )
    if target_name_col is not None:
        selected['ot_target_name'] = table[target_name_col]
    if disease_name_col is not None:
        selected['ot_disease_name'] = table[disease_name_col]

    selected = _drop_null_and_deduplicate(
        selected,
        required_columns=['ot_target_id', 'ot_disease_id'],
        dedup_columns=['ot_target_id', 'ot_disease_id'],
    )
    return selected


def construct_candidate_triplets(
    drug_target_df: pd.DataFrame,
    target_disease_df: pd.DataFrame,
) -> pd.DataFrame:
    """以 `target` 为桥梁，Inner Join 构造 OT 候选机制路径全集。"""

    joined = drug_target_df.merge(
        target_disease_df,
        on='ot_target_id',
        how='inner',
        suffixes=('_drug_side', '_disease_side'),
    )

    triplets = pd.DataFrame(
        {
            'ot_drug_id': joined['ot_drug_id'],
            'ot_target_id': joined['ot_target_id'],
            'ot_disease_id': joined['ot_disease_id'],
        }
    )

    if 'ot_drug_name' in joined.columns:
        source_col = 'ot_drug_name'
        if source_col in joined.columns:
            triplets['ot_drug_name'] = joined[source_col]

    target_name_candidates = [
        column_name
        for column_name in ['ot_target_name_drug_side', 'ot_target_name_disease_side', 'ot_target_name']
        if column_name in joined.columns
    ]
    if target_name_candidates:
        triplets['ot_target_name'] = joined[target_name_candidates].bfill(axis=1).iloc[:, 0]

    disease_name_candidates = [
        column_name
        for column_name in ['ot_disease_name', 'ot_disease_name_disease_side']
        if column_name in joined.columns
    ]
    if disease_name_candidates:
        triplets['ot_disease_name'] = joined[disease_name_candidates].bfill(axis=1).iloc[:, 0]

    triplets = _drop_null_and_deduplicate(
        triplets,
        required_columns=['ot_drug_id', 'ot_target_id', 'ot_disease_id'],
        dedup_columns=['ot_drug_id', 'ot_target_id', 'ot_disease_id'],
    )
    return triplets


def align_entities_to_primekg(
    triplets_df: pd.DataFrame,
    primekg_nodes_csv: Path,
    drug_mapping_path: Optional[Path] = None,
    target_mapping_path: Optional[Path] = None,
    disease_mapping_path: Optional[Path] = None,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """
    将 OT 三元组对齐到 PrimeKG 实体体系。

    对齐优先级：
    1. 先走外部 mapping 文件；
    2. 再走 OT 原始 ID 本身；
    3. 最后走名称字段（若存在）。

    说明：
    - 外部 mapping 的 value 可以是 PrimeKG raw id、PrimeKG 名称、或 PrimeKG 本地后缀 ID；
    - 例如：DrugBank `DB00180`、Gene Entrez `9796`、MONDO `MONDO:0005044` 都能被 fallback alias 解析。
    """

    alias_index = build_primekg_alias_index(primekg_nodes_csv)
    drug_mapping = load_mapping_dict(drug_mapping_path)
    target_mapping = load_mapping_dict(target_mapping_path)
    disease_mapping = load_mapping_dict(disease_mapping_path)

    aligned = triplets_df.copy()
    aligned['primekg_drug_id'] = _resolve_primekg_entity_ids(
        id_series=aligned['ot_drug_id'],
        name_series=aligned.get('ot_drug_name'),
        external_mapping=drug_mapping,
        primekg_lookup=alias_index.lookup_by_type['drug'],
    )
    aligned['primekg_target_id'] = _resolve_primekg_entity_ids(
        id_series=aligned['ot_target_id'],
        name_series=aligned.get('ot_target_name'),
        external_mapping=target_mapping,
        primekg_lookup=alias_index.lookup_by_type['gene/protein'],
    )
    aligned['primekg_disease_id'] = _resolve_primekg_entity_ids(
        id_series=aligned['ot_disease_id'],
        name_series=aligned.get('ot_disease_name'),
        external_mapping=disease_mapping,
        primekg_lookup=alias_index.lookup_by_type['disease'],
    )

    aligned['drug_mapped'] = aligned['primekg_drug_id'].notna()
    aligned['target_mapped'] = aligned['primekg_target_id'].notna()
    aligned['disease_mapped'] = aligned['primekg_disease_id'].notna()
    aligned['is_fully_mapped'] = (
        aligned['drug_mapped'] & aligned['target_mapped'] & aligned['disease_mapped']
    )

    aligned['primekg_drug_name'] = aligned['primekg_drug_id'].map(alias_index.raw_id_to_name)
    aligned['primekg_target_name'] = aligned['primekg_target_id'].map(alias_index.raw_id_to_name)
    aligned['primekg_disease_name'] = aligned['primekg_disease_id'].map(alias_index.raw_id_to_name)

    fully_mapped = aligned.loc[aligned['is_fully_mapped']].copy()
    fully_mapped = fully_mapped.drop_duplicates(
        subset=['primekg_drug_id', 'primekg_target_id', 'primekg_disease_id']
    ).reset_index(drop=True)

    mapping_report: Dict[str, object] = {
        'triplet_level_success_rate': {
            'drug': _safe_rate(aligned['drug_mapped'].sum(), len(aligned)),
            'target': _safe_rate(aligned['target_mapped'].sum(), len(aligned)),
            'disease': _safe_rate(aligned['disease_mapped'].sum(), len(aligned)),
            'fully_mapped_triplet_survival': _safe_rate(aligned['is_fully_mapped'].sum(), len(aligned)),
        },
        'triplet_level_counts': {
            'input_triplets': int(len(aligned)),
            'drug_mapped_triplets': int(aligned['drug_mapped'].sum()),
            'target_mapped_triplets': int(aligned['target_mapped'].sum()),
            'disease_mapped_triplets': int(aligned['disease_mapped'].sum()),
            'fully_mapped_triplets_before_dedup': int(aligned['is_fully_mapped'].sum()),
            'fully_mapped_triplets_after_dedup': int(len(fully_mapped)),
        },
        'unique_entity_alignment': {
            'drug': _compute_unique_mapping_summary(aligned, 'ot_drug_id', 'primekg_drug_id'),
            'target': _compute_unique_mapping_summary(aligned, 'ot_target_id', 'primekg_target_id'),
            'disease': _compute_unique_mapping_summary(aligned, 'ot_disease_id', 'primekg_disease_id'),
        },
    }
    return fully_mapped, mapping_report


def load_primekg_train_paths(
    path: Path,
    primekg_nodes_csv: Path,
) -> pd.DataFrame:
    """
    读取 PrimeKG 训练集高阶路径，并统一为 PrimeKG raw id 三元组。

    支持：
    - `.pt`: Tensor 或包含 `ho_pos_paths` / `train_paths` / `train_ho_paths` 的字典；
    - `.csv` / `.tsv` / `.parquet` / `.json` / `.jsonl`。

    若是整数 Tensor，默认按 PrimeKG 全局节点 ID 解释，并利用 `nodes.csv` 恢复 raw id。
    """

    global_id_to_raw_id = build_primekg_alias_index(primekg_nodes_csv).global_id_to_raw_id
    suffix = path.suffix.casefold()

    if suffix == '.pt':
        import torch

        obj = torch.load(path, map_location='cpu', weights_only=False)
        tensor = _extract_train_path_tensor(obj)
        if tensor.ndim != 2 or tensor.shape[1] != 3:
            raise ValueError(
                '`primekg_train_paths` Tensor 形状必须为 `(N, 3)`，'
                f'实际得到 {tuple(tensor.shape)}。'
            )
        rows = tensor.detach().cpu().tolist()
        records = [
            {
                'primekg_drug_id': global_id_to_raw_id[int(row[0])],
                'primekg_target_id': global_id_to_raw_id[int(row[1])],
                'primekg_disease_id': global_id_to_raw_id[int(row[2])],
            }
            for row in rows
        ]
        return pd.DataFrame.from_records(records).drop_duplicates(ignore_index=True)

    table = _load_table(path)
    table = _standardize_dataframe_columns(table)
    drug_col, target_col, disease_col = _infer_triplet_columns(table)
    train_df = pd.DataFrame(
        {
            'primekg_drug_id': table[drug_col],
            'primekg_target_id': table[target_col],
            'primekg_disease_id': table[disease_col],
        }
    )

    if pd.api.types.is_integer_dtype(train_df['primekg_drug_id']):
        train_df['primekg_drug_id'] = train_df['primekg_drug_id'].map(
            lambda x: global_id_to_raw_id[int(x)]
        )
    if pd.api.types.is_integer_dtype(train_df['primekg_target_id']):
        train_df['primekg_target_id'] = train_df['primekg_target_id'].map(
            lambda x: global_id_to_raw_id[int(x)]
        )
    if pd.api.types.is_integer_dtype(train_df['primekg_disease_id']):
        train_df['primekg_disease_id'] = train_df['primekg_disease_id'].map(
            lambda x: global_id_to_raw_id[int(x)]
        )

    train_df = _drop_null_and_deduplicate(
        train_df,
        required_columns=['primekg_drug_id', 'primekg_target_id', 'primekg_disease_id'],
        dedup_columns=['primekg_drug_id', 'primekg_target_id', 'primekg_disease_id'],
    )
    return train_df


def load_primekg_train_pairs(
    path: Path,
    primekg_nodes_csv: Path,
) -> pd.DataFrame:
    """
    ?? PrimeKG ????? `(drug, disease)` pairs?

    ???
    - processed split `.pt`????? `target_pairs['train']`?
    - HO ?? `.pt/.csv/.tsv/.parquet/.json/.jsonl`?? `(drug, gene, disease)` ?? pair?
    - ? pair ?????? `(drug, disease)` ???
    """

    global_id_to_raw_id = build_primekg_alias_index(primekg_nodes_csv).global_id_to_raw_id
    suffix = path.suffix.casefold()

    if suffix == '.pt':
        import torch

        obj = torch.load(path, map_location='cpu', weights_only=False)
        pair_tensor = _extract_train_pair_tensor(obj)
        if pair_tensor is not None:
            if pair_tensor.ndim != 2 or pair_tensor.shape[1] != 2:
                raise ValueError(
                    '`target_pairs[train]` ?????? `(N, 2)`?'
                    f'???? {tuple(pair_tensor.shape)}?'
                )
            records = [
                {
                    'primekg_drug_id': global_id_to_raw_id[int(row[0])],
                    'primekg_disease_id': global_id_to_raw_id[int(row[1])],
                }
                for row in pair_tensor.detach().cpu().tolist()
            ]
            return pd.DataFrame.from_records(records).drop_duplicates(ignore_index=True)

        triplet_tensor = _extract_train_path_tensor(obj)
        if triplet_tensor.ndim != 2 or triplet_tensor.shape[1] != 3:
            raise ValueError(
                '`primekg_train_paths` Tensor ????? `(N, 3)`?'
                f'???? {tuple(triplet_tensor.shape)}?'
            )
        records = [
            {
                'primekg_drug_id': global_id_to_raw_id[int(row[0])],
                'primekg_disease_id': global_id_to_raw_id[int(row[2])],
            }
            for row in triplet_tensor.detach().cpu().tolist()
        ]
        return pd.DataFrame.from_records(records).drop_duplicates(ignore_index=True)

    table = _load_table(path)
    table = _standardize_dataframe_columns(table)

    pair_columns = _infer_pair_columns(table)
    if pair_columns is not None:
        drug_col, disease_col = pair_columns
        train_pair_df = pd.DataFrame(
            {
                'primekg_drug_id': table[drug_col],
                'primekg_disease_id': table[disease_col],
            }
        )
    else:
        drug_col, _, disease_col = _infer_triplet_columns(table)
        train_pair_df = pd.DataFrame(
            {
                'primekg_drug_id': table[drug_col],
                'primekg_disease_id': table[disease_col],
            }
        )

    if pd.api.types.is_integer_dtype(train_pair_df['primekg_drug_id']):
        train_pair_df['primekg_drug_id'] = train_pair_df['primekg_drug_id'].map(
            lambda x: global_id_to_raw_id[int(x)]
        )
    if pd.api.types.is_integer_dtype(train_pair_df['primekg_disease_id']):
        train_pair_df['primekg_disease_id'] = train_pair_df['primekg_disease_id'].map(
            lambda x: global_id_to_raw_id[int(x)]
        )

    return _drop_null_and_deduplicate(
        train_pair_df,
        required_columns=['primekg_drug_id', 'primekg_disease_id'],
        dedup_columns=['primekg_drug_id', 'primekg_disease_id'],
    )



def filter_novel_ood_triplets(
    aligned_triplets_df: pd.DataFrame,
    primekg_train_pairs_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, object]]:
    """? pair-level Novelty ?? OOD ???????? `(drug, disease)`?"""

    aligned = aligned_triplets_df.copy()
    aligned['pair_key'] = _make_triplet_key(
        aligned,
        columns=('primekg_drug_id', 'primekg_disease_id'),
    )
    train_pair_keys = set(
        _make_triplet_key(
            primekg_train_pairs_df,
            columns=('primekg_drug_id', 'primekg_disease_id'),
        )
    )
    aligned['is_overlap_with_primekg_train_pair'] = aligned['pair_key'].isin(train_pair_keys)
    aligned['is_overlap_with_primekg_train'] = aligned['is_overlap_with_primekg_train_pair']

    overlap_df = aligned.loc[aligned['is_overlap_with_primekg_train_pair']].copy()
    novel_df = aligned.loc[~aligned['is_overlap_with_primekg_train_pair']].copy()

    overlap_pair_count = int(overlap_df[['primekg_drug_id', 'primekg_disease_id']].drop_duplicates().shape[0])
    novel_pair_count = int(novel_df[['primekg_drug_id', 'primekg_disease_id']].drop_duplicates().shape[0])

    overlap_df = overlap_df.drop(columns=['pair_key']).reset_index(drop=True)
    novel_df = novel_df.drop(columns=['pair_key']).reset_index(drop=True)

    novelty_report: Dict[str, object] = {
        'filter_level': 'pair',
        'aligned_triplets': int(len(aligned_triplets_df)),
        'overlap_triplets': int(len(overlap_df)),
        'novel_triplets': int(len(novel_df)),
        'overlap_ratio': _safe_rate(len(overlap_df), len(aligned_triplets_df)),
        'novel_ratio': _safe_rate(len(novel_df), len(aligned_triplets_df)),
        'overlap_pairs': overlap_pair_count,
        'novel_pairs': novel_pair_count,
    }
    return overlap_df, novel_df, novelty_report



def generate_report(
    drug_target_df: pd.DataFrame,
    target_disease_df: pd.DataFrame,
    candidate_triplets_df: pd.DataFrame,
    aligned_triplets_df: pd.DataFrame,
    overlap_df: pd.DataFrame,
    novel_ood_df: pd.DataFrame,
    mapping_report: Mapping[str, object],
    novelty_report: Mapping[str, object],
    output_path: Path,
) -> Dict[str, object]:
    """?????????????"""

    report: Dict[str, object] = {
        'raw_ot_counts': {
            'drug_target_associations': int(len(drug_target_df)),
            'target_disease_associations': int(len(target_disease_df)),
        },
        'triplet_construction': {
            'candidate_triplets_after_inner_join': int(len(candidate_triplets_df)),
        },
        'mapping': mapping_report,
        'novelty': novelty_report,
        'ood_unique_entities': {
            'unique_drugs': int(novel_ood_df['primekg_drug_id'].nunique()),
            'unique_targets': int(novel_ood_df['primekg_target_id'].nunique()),
            'unique_diseases': int(novel_ood_df['primekg_disease_id'].nunique()),
            'unique_pairs': int(novel_ood_df[['primekg_drug_id', 'primekg_disease_id']].drop_duplicates().shape[0]),
        },
    }

    print('=== Open Targets -> PrimeKG OOD Triplet Report ===')
    print('[???] OT ??????')
    print(f"  drug-target ???: {report['raw_ot_counts']['drug_target_associations']}")
    print(f"  target-disease ???: {report['raw_ot_counts']['target_disease_associations']}")
    print(f"  Join ???????: {report['triplet_construction']['candidate_triplets_after_inner_join']}")

    print('[???] ??????')
    mapping_success = report['mapping']['triplet_level_success_rate']
    print(f"  Drug ?????: {mapping_success['drug']:.4f}")
    print(f"  Target ?????: {mapping_success['target']:.4f}")
    print(f"  Disease ?????: {mapping_success['disease']:.4f}")
    print(f"  ????????: {mapping_success['fully_mapped_triplet_survival']:.4f}")
    print(f"  ?????????????: {len(aligned_triplets_df)}")

    print('[???] Pair-level Novelty / OOD ??')
    print(f"  ????: {report['novelty']['filter_level']}-level")
    print(f"  ? PrimeKG ??? pair ???????: {len(overlap_df)}")
    print(f"  ?? Novel / OOD ?????: {len(novel_ood_df)}")
    print(f"  Overlap ??: {report['novelty']['overlap_ratio']:.4f}")
    print(f"  Novel ??: {report['novelty']['novel_ratio']:.4f}")
    print(f"  Overlap pairs: {report['novelty']['overlap_pairs']}")
    print(f"  Novel pairs: {report['novelty']['novel_pairs']}")

    print('[???] OOD ????')
    print(f"  OOD unique drugs: {report['ood_unique_entities']['unique_drugs']}")
    print(f"  OOD unique targets: {report['ood_unique_entities']['unique_targets']}")
    print(f"  OOD unique diseases: {report['ood_unique_entities']['unique_diseases']}")
    print(f"  OOD unique pairs: {report['ood_unique_entities']['unique_pairs']}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('w', encoding='utf-8') as file:
        json.dump(report, file, ensure_ascii=False, indent=2)
    return report


def load_mapping_dict(path: Optional[Path]) -> Dict[str, str]:
    """
    ?????????? `normalized_source -> raw_target_text` ???

    ???
    - `.json`: ?? dict?? record list?
    - `.csv/.tsv/.parquet/.jsonl`: ???? source/target ??
    """

    if path is None:
        return {}

    suffix = path.suffix.casefold()
    if suffix == '.json':
        with path.open('r', encoding='utf-8') as file:
            payload = json.load(file)
        if isinstance(payload, dict):
            return {
                _normalize_text(key): _normalize_text(value)
                for key, value in payload.items()
                if _normalize_text(key) and _normalize_text(value)
            }
        if isinstance(payload, list):
            table = pd.DataFrame(payload)
        else:
            raise ValueError(f'???? JSON ????: {type(payload)}')
    else:
        table = _load_table(path)

    table = _standardize_dataframe_columns(table)
    source_col = _pick_first_existing_column(
        table,
        candidates=['source_id', 'from_id', 'query_id', 'input_id', 'ot_id', 'source'],
        table_name=f'mapping:{path.name}',
    )
    target_col = _pick_first_existing_column(
        table,
        candidates=['target_id', 'to_id', 'mapped_id', 'output_id', 'primekg_id', 'target'],
        table_name=f'mapping:{path.name}',
    )

    mapping_df = pd.DataFrame(
        {
            'source': table[source_col],
            'target': table[target_col],
        }
    )
    mapping_df = _drop_null_and_deduplicate(
        mapping_df,
        required_columns=['source', 'target'],
        dedup_columns=['source'],
    )
    return {
        _normalize_text(source): _normalize_text(target)
        for source, target in zip(mapping_df['source'].tolist(), mapping_df['target'].tolist())
        if _normalize_text(source) and _normalize_text(target)
    }


def run_pipeline(args: argparse.Namespace) -> None:
    print('[1/4] ?? OT ????...')
    drug_target_df = load_ot_drug_target_associations(args.ot_drug_target_path)
    target_disease_df = load_ot_target_disease_associations(args.ot_target_disease_path)

    print('[2/4] ?? OT ?????...')
    candidate_triplets_df = construct_candidate_triplets(
        drug_target_df=drug_target_df,
        target_disease_df=target_disease_df,
    )

    print('[3/4] OT -> PrimeKG ????...')
    aligned_triplets_df, mapping_report = align_entities_to_primekg(
        triplets_df=candidate_triplets_df,
        primekg_nodes_csv=args.primekg_nodes_csv,
        drug_mapping_path=args.drug_mapping_path,
        target_mapping_path=args.target_mapping_path,
        disease_mapping_path=args.disease_mapping_path,
    )

    print('[4/4] Pair-level Novelty ????? OOD ??...')
    primekg_train_pairs_df = load_primekg_train_pairs(
        path=args.primekg_train_paths,
        primekg_nodes_csv=args.primekg_nodes_csv,
    )
    overlap_df, novel_ood_df, novelty_report = filter_novel_ood_triplets(
        aligned_triplets_df=aligned_triplets_df,
        primekg_train_pairs_df=primekg_train_pairs_df,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = generate_report(
        drug_target_df=drug_target_df,
        target_disease_df=target_disease_df,
        candidate_triplets_df=candidate_triplets_df,
        aligned_triplets_df=aligned_triplets_df,
        overlap_df=overlap_df,
        novel_ood_df=novel_ood_df,
        mapping_report=mapping_report,
        novelty_report=novelty_report,
        output_path=args.output_dir / 'report.json',
    )

    aligned_triplets_df.to_csv(args.output_dir / 'aligned_triplets.csv', index=False)
    overlap_df.to_csv(args.output_dir / 'overlap_triplets.csv', index=False)
    novel_ood_df.to_csv(args.output_dir / 'novel_ood_triplets.csv', index=False)
    if args.save_intermediate:
        drug_target_df.to_csv(args.output_dir / 'ot_drug_target.csv', index=False)
        target_disease_df.to_csv(args.output_dir / 'ot_target_disease.csv', index=False)
        candidate_triplets_df.to_csv(args.output_dir / 'candidate_triplets.csv', index=False)

    print(f"????????: {args.output_dir / 'aligned_triplets.csv'}")
    print(f"??? OOD ???: {args.output_dir / 'novel_ood_triplets.csv'}")
    print(f"???????: {args.output_dir / 'report.json'}")
    print('????:', json.dumps(report['novelty'], ensure_ascii=False))


def _load_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.casefold()
    if suffix == '.csv':
        return pd.read_csv(path)
    if suffix in {'.tsv', '.txt'}:
        return pd.read_csv(path, sep='\t')
    if suffix in {'.parquet', '.pq'}:
        return pd.read_parquet(path)
    if suffix == '.json':
        return pd.read_json(path)
    if suffix in {'.jsonl', '.ndjson'}:
        return pd.read_json(path, lines=True)
    raise ValueError(f'不支持的表格格式: {path}')


def _standardize_dataframe_columns(df: pd.DataFrame) -> pd.DataFrame:
    renamed = df.copy()
    renamed.columns = [
        re.sub(r'[^0-9a-zA-Z]+', '_', str(column).strip()).strip('_').casefold()
        for column in renamed.columns
    ]
    return renamed


def _pick_first_existing_column(
    df: pd.DataFrame,
    candidates: Sequence[str],
    table_name: str,
) -> str:
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
    raise KeyError(
        f'在 `{table_name}` 中找不到任何候选列。'
        f'候选列={list(candidates)}, 实际列={list(df.columns)}'
    )


def _pick_optional_column(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
    return None


def _drop_null_and_deduplicate(
    df: pd.DataFrame,
    required_columns: Sequence[str],
    dedup_columns: Sequence[str],
) -> pd.DataFrame:
    out = df.copy()
    for column_name in required_columns:
        out[column_name] = out[column_name].map(lambda value: None if _is_null_like(value) else str(value).strip())
    out = out.dropna(subset=list(required_columns))
    out = out.loc[(out[list(required_columns)] != '').all(axis=1)]
    out = out.drop_duplicates(subset=list(dedup_columns)).reset_index(drop=True)
    return out


def _resolve_primekg_entity_ids(
    id_series: pd.Series,
    name_series: Optional[pd.Series],
    external_mapping: Mapping[str, str],
    primekg_lookup: Mapping[str, str],
) -> pd.Series:
    """按“外部 mapping -> 原始 ID -> 名称”顺序尝试解析。"""

    normalized_id = id_series.map(_normalize_text)
    resolved = pd.Series([pd.NA] * len(id_series), index=id_series.index, dtype='object')

    if external_mapping:
        mapped_candidate = normalized_id.map(external_mapping)
        resolved = resolved.fillna(mapped_candidate.map(primekg_lookup))

    resolved = resolved.fillna(normalized_id.map(primekg_lookup))

    if name_series is not None:
        normalized_name = name_series.map(_normalize_text)
        resolved = resolved.fillna(normalized_name.map(primekg_lookup))

    return resolved


def _compute_unique_mapping_summary(
    df: pd.DataFrame,
    source_column: str,
    mapped_column: str,
) -> Dict[str, float]:
    unique_source = df[source_column].dropna().astype(str).str.strip()
    unique_source = unique_source[unique_source != '']
    if len(unique_source) == 0:
        return {
            'input_unique_entities': 0,
            'mapped_unique_entities': 0,
            'success_rate': 0.0,
        }

    unique_table = df[[source_column, mapped_column]].drop_duplicates(subset=[source_column]).copy()
    unique_table[source_column] = unique_table[source_column].astype(str).str.strip()
    unique_table = unique_table.loc[unique_table[source_column] != '']
    mapped_unique_entities = int(unique_table[mapped_column].notna().sum())
    input_unique_entities = int(len(unique_table))
    return {
        'input_unique_entities': input_unique_entities,
        'mapped_unique_entities': mapped_unique_entities,
        'success_rate': _safe_rate(mapped_unique_entities, input_unique_entities),
    }


def build_primekg_alias_index(nodes_csv: Path) -> PrimeKGAliasIndex:
    """
    ? PrimeKG ?????????

    ??????????
    - ?? raw id ???
    - ?? `type::suffix` ? suffix?
    - ???? name?
    - ???? source-specific heuristic alias??? DrugBank / Entrez / MONDO?
    """

    lookup_by_type: Dict[str, Dict[str, str]] = {
        'drug': {},
        'gene/protein': {},
        'disease': {},
    }
    raw_id_to_name: Dict[str, str] = {}
    global_id_to_raw_id: List[str] = []

    with nodes_csv.open('r', encoding='utf-8-sig', newline='') as file:
        reader = csv.DictReader(file)
        for row in reader:
            raw_id = str(row['id']).strip()
            node_type = str(row['type']).strip()
            name = str(row['name']).strip()
            source = str(row['source']).strip()

            raw_id_to_name[raw_id] = name
            global_id_to_raw_id.append(raw_id)

            if node_type not in lookup_by_type:
                continue

            for alias in _iter_primekg_aliases(raw_id=raw_id, name=name, node_type=node_type, source=source):
                lookup_by_type[node_type].setdefault(alias, raw_id)

    return PrimeKGAliasIndex(
        lookup_by_type=lookup_by_type,
        raw_id_to_name=raw_id_to_name,
        global_id_to_raw_id=global_id_to_raw_id,
    )


def _make_triplet_key(df: pd.DataFrame, columns: TripletColumns) -> pd.Series:
    return (
        df[list(columns)]
        .astype(str)
        .agg('||'.join, axis=1)
    )


def _safe_rate(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return float(numerator) / float(denominator)


def _normalize_text(value: object) -> str:
    if _is_null_like(value):
        return ''
    text = str(value).strip()
    if not text:
        return ''
    text = re.sub(r'\s+', ' ', text)
    return text.casefold()


def _is_null_like(value: object) -> bool:
    return value is None or (isinstance(value, float) and pd.isna(value)) or pd.isna(value)


def _iter_primekg_aliases(
    raw_id: str,
    name: str,
    node_type: str,
    source: str,
) -> Iterable[str]:
    """为单个 PrimeKG 节点生成尽量稳健的可匹配别名。"""

    suffix = raw_id.split('::', maxsplit=1)[1] if '::' in raw_id else raw_id

    basic_aliases = {
        raw_id,
        suffix,
        name,
    }
    for alias in basic_aliases:
        normalized = _normalize_text(alias)
        if normalized:
            yield normalized

    if node_type == 'drug':
        normalized_suffix = _normalize_text(suffix)
        if normalized_suffix.startswith('db'):
            yield normalized_suffix
            yield _normalize_text(f'drugbank:{suffix}')

    if node_type == 'gene/protein':
        normalized_suffix = _normalize_text(suffix)
        if normalized_suffix.isdigit():
            yield normalized_suffix
            yield _normalize_text(f'entrez:{suffix}')
            yield _normalize_text(f'ncbi:{suffix}')

    if node_type == 'disease':
        normalized_suffix = _normalize_text(suffix)
        if source.startswith('MONDO') and normalized_suffix.isdigit():
            mondo_numeric = int(normalized_suffix)
            yield _normalize_text(f'mondo:{mondo_numeric}')
            yield _normalize_text(f'mondo:{mondo_numeric:07d}')
            yield _normalize_text(f'mondo_{mondo_numeric:07d}')
            yield _normalize_text(f'mondo:{normalized_suffix}')


def _extract_train_path_tensor(obj):
    import torch

    if isinstance(obj, torch.Tensor):
        return obj
    if isinstance(obj, dict):
        for candidate_key in ('ho_pos_paths', 'train_ho_paths', 'train_paths', 'paths'):
            if candidate_key in obj:
                tensor = obj[candidate_key]
                if not isinstance(tensor, torch.Tensor):
                    raise TypeError(f'`{candidate_key}` 不是 Tensor，实际为 {type(tensor)}')
                return tensor
    raise ValueError('无法从 `primekg_train_paths` 中提取训练路径 Tensor。')


def _extract_train_pair_tensor(obj):
    import torch

    if not isinstance(obj, dict):
        return None
    target_pairs = obj.get('target_pairs')
    if not isinstance(target_pairs, dict):
        return None
    train_pairs = target_pairs.get('train')
    if train_pairs is None:
        return None
    if not isinstance(train_pairs, torch.Tensor):
        raise TypeError(f'`target_pairs["train"]` ?? Tensor???? {type(train_pairs)}')
    return train_pairs



def _infer_pair_columns(df: pd.DataFrame) -> Optional[Tuple[str, str]]:
    drug_col = _pick_optional_column(
        df,
        ['primekg_drug_id', 'drug_id', 'drug', 'src_drug_id'],
    )
    disease_col = _pick_optional_column(
        df,
        ['primekg_disease_id', 'disease_id', 'disease', 'dst_disease_id'],
    )
    if drug_col is None or disease_col is None:
        return None
    if drug_col == disease_col:
        return None
    return drug_col, disease_col


def _infer_triplet_columns(df: pd.DataFrame) -> TripletColumns:
    drug_col = _pick_first_existing_column(
        df,
        candidates=['primekg_drug_id', 'drug_id', 'drug', 'src_drug_id'],
        table_name='primekg_train_paths',
    )
    target_col = _pick_first_existing_column(
        df,
        candidates=['primekg_target_id', 'target_id', 'gene_id', 'target', 'gene'],
        table_name='primekg_train_paths',
    )
    disease_col = _pick_first_existing_column(
        df,
        candidates=['primekg_disease_id', 'disease_id', 'disease', 'dst_disease_id'],
        table_name='primekg_train_paths',
    )
    return drug_col, target_col, disease_col


def main() -> None:
    args = parse_args()
    run_pipeline(args)


if __name__ == '__main__':
    main()
