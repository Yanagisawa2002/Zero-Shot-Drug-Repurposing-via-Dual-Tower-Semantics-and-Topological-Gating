from __future__ import annotations

import argparse
import math
import pickle
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd
import torch
from torch import Tensor
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


HF_MODEL_NAME = 'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext'
LOCAL_MODEL_DIR = Path('models/pubmedbert_biomedbert_base_uncased_abstract_fulltext')


@dataclass(frozen=True)
class DiseaseSourceSpec:
    """??????????"""

    csv_path: Path
    disease_id_column: str
    disease_name_column: str
    source_name: str
    type_column: Optional[str] = None
    type_value: Optional[str] = None


@dataclass
class SourceLoadStats:
    """????????????"""

    source_name: str
    total_rows: int
    rows_after_filter: int
    unique_ids_in_source: int


@dataclass
class MergeStats:
    """???????????"""

    total_records_before_dedup: int
    unique_disease_ids: int
    valid_name_count: int
    invalid_name_count: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Extract unified PubMedBERT CLS embeddings for diseases from multiple CSV sources.',
    )
    parser.add_argument(
        '--output-pkl',
        type=Path,
        default=Path('disease_text_embeddings.pkl'),
        help='??? disease embedding pickle ?????',
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=64,
        help='PubMedBERT ????',
    )
    parser.add_argument(
        '--max-length',
        type=int,
        default=512,
        help='tokenize ???????',
    )
    parser.add_argument(
        '--model-name-or-path',
        type=str,
        default=HF_MODEL_NAME,
        help='Transformers ??????????',
    )
    parser.add_argument(
        '--prefer-local-model',
        action='store_true',
        help='????????? PubMedBERT ???',
    )

    parser.add_argument(
        '--skip-primekg',
        action='store_true',
        help='?? PrimeKG ??????',
    )
    parser.add_argument(
        '--primekg-csv',
        type=Path,
        default=Path('data/PrimeKG/nodes.csv'),
        help='PrimeKG ??????',
    )
    parser.add_argument('--primekg-id-column', type=str, default='id')
    parser.add_argument('--primekg-name-column', type=str, default='name')
    parser.add_argument('--primekg-type-column', type=str, default='type')
    parser.add_argument('--primekg-type-value', type=str, default='disease')

    parser.add_argument(
        '--skip-holdout',
        action='store_true',
        help='?? Holdout ????????',
    )
    parser.add_argument(
        '--holdout-csv',
        type=Path,
        default=Path('triple_level_pubmedbert_llm_cleaned.csv'),
        help='Holdout ??????',
    )
    parser.add_argument('--holdout-id-column', type=str, default='disease_id')
    parser.add_argument('--holdout-name-column', type=str, default='disease_name')

    parser.add_argument(
        '--ot-csv',
        type=Path,
        default=None,
        help='Open Targets CSV ?????????? OT ???',
    )
    parser.add_argument('--ot-id-column', type=str, default='disease_id')
    parser.add_argument('--ot-name-column', type=str, default='disease_name')
    parser.add_argument('--ot-type-column', type=str, default='')
    parser.add_argument('--ot-type-value', type=str, default='')

    parser.add_argument(
        '--extra-source-spec',
        action='append',
        default=[],
        help=(
            '????????: path|id_col|name_col|source_name|type_col|type_value?'
            '???????????: extra.csv|disease_id|disease_name|extra||'
        ),
    )
    return parser.parse_args()


def resolve_model_source(model_name_or_path: str, prefer_local_model: bool) -> str:
    """??????????????? HuggingFace ???"""
    if prefer_local_model and LOCAL_MODEL_DIR.exists():
        return str(LOCAL_MODEL_DIR)

    explicit_path = Path(model_name_or_path)
    if explicit_path.exists():
        return str(explicit_path)

    if LOCAL_MODEL_DIR.exists() and model_name_or_path == HF_MODEL_NAME:
        return str(LOCAL_MODEL_DIR)

    return model_name_or_path


def detect_device() -> torch.device:
    """???? CUDA?????? CPU?"""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def is_valid_disease_name(name_value: object) -> bool:
    """??????????????"""
    if name_value is None or pd.isna(name_value):
        return False
    name_text = str(name_value).strip()
    if not name_text:
        return False
    if name_text.lower() in {'nan', 'none', 'null'}:
        return False
    return True


def normalize_optional_text(text: str) -> Optional[str]:
    stripped = str(text).strip()
    return stripped if stripped else None


def parse_extra_source_spec(spec_text: str) -> DiseaseSourceSpec:
    """?????????"""
    parts = spec_text.split('|')
    if len(parts) < 4 or len(parts) > 6:
        raise ValueError(
            'Invalid --extra-source-spec. Expected format: '
            'path|id_col|name_col|source_name|type_col|type_value'
        )
    while len(parts) < 6:
        parts.append('')
    csv_path, disease_id_column, disease_name_column, source_name, type_column, type_value = parts
    return DiseaseSourceSpec(
        csv_path=Path(csv_path),
        disease_id_column=disease_id_column,
        disease_name_column=disease_name_column,
        source_name=source_name,
        type_column=normalize_optional_text(type_column),
        type_value=normalize_optional_text(type_value),
    )


def build_source_specs(args: argparse.Namespace) -> List[DiseaseSourceSpec]:
    """????????????"""
    source_specs: List[DiseaseSourceSpec] = []

    if not args.skip_primekg:
        source_specs.append(
            DiseaseSourceSpec(
                csv_path=args.primekg_csv,
                disease_id_column=args.primekg_id_column,
                disease_name_column=args.primekg_name_column,
                source_name='primekg',
                type_column=args.primekg_type_column,
                type_value=args.primekg_type_value,
            )
        )

    if not args.skip_holdout:
        source_specs.append(
            DiseaseSourceSpec(
                csv_path=args.holdout_csv,
                disease_id_column=args.holdout_id_column,
                disease_name_column=args.holdout_name_column,
                source_name='holdout',
            )
        )

    if args.ot_csv is not None:
        source_specs.append(
            DiseaseSourceSpec(
                csv_path=args.ot_csv,
                disease_id_column=args.ot_id_column,
                disease_name_column=args.ot_name_column,
                source_name='ot',
                type_column=normalize_optional_text(args.ot_type_column),
                type_value=normalize_optional_text(args.ot_type_value),
            )
        )

    for spec_text in args.extra_source_spec:
        source_specs.append(parse_extra_source_spec(spec_text))

    if not source_specs:
        raise ValueError('No disease source is enabled. Please keep at least one source.')
    return source_specs


def load_source_dataframe(csv_path: Path) -> pd.DataFrame:
    """???? CSV ???"""
    if not csv_path.exists():
        raise FileNotFoundError(f'Source CSV does not exist: {csv_path}')
    frame = pd.read_csv(csv_path)
    if frame.empty:
        raise ValueError(f'Source CSV is empty: {csv_path}')
    return frame


def validate_columns(frame: pd.DataFrame, required_columns: Sequence[str]) -> None:
    """??????????"""
    missing_columns = [column for column in required_columns if column and column not in frame.columns]
    if missing_columns:
        raise ValueError(
            f'Missing required columns: {missing_columns}. Available columns: {list(frame.columns)}'
        )


def extract_disease_records_from_source(
    spec: DiseaseSourceSpec,
) -> Tuple[List[Tuple[str, Optional[str]]], SourceLoadStats]:
    """??????? `(disease_id, disease_name)` ???"""
    frame = load_source_dataframe(spec.csv_path)
    required_columns = [spec.disease_id_column, spec.disease_name_column]
    if spec.type_column is not None:
        required_columns.append(spec.type_column)
    validate_columns(frame, required_columns)

    total_rows = int(len(frame))
    if spec.type_column is not None and spec.type_value is not None:
        filtered_frame = frame[
            frame[spec.type_column].astype(str) == str(spec.type_value)
        ].copy()
    else:
        filtered_frame = frame.copy()

    records: List[Tuple[str, Optional[str]]] = []
    for row in filtered_frame[[spec.disease_id_column, spec.disease_name_column]].itertuples(index=False, name=None):
        disease_id, disease_name = row
        if disease_id is None or pd.isna(disease_id):
            continue
        disease_id_str = str(disease_id).strip()
        if not disease_id_str:
            continue

        if is_valid_disease_name(disease_name):
            disease_name_text: Optional[str] = str(disease_name).strip()
        else:
            disease_name_text = None
        records.append((disease_id_str, disease_name_text))

    stats = SourceLoadStats(
        source_name=spec.source_name,
        total_rows=total_rows,
        rows_after_filter=int(len(filtered_frame)),
        unique_ids_in_source=len({record[0] for record in records}),
    )
    return records, stats


def merge_disease_records(
    source_records: Sequence[Tuple[str, Optional[str]]],
) -> Tuple[List[Tuple[str, Optional[str]]], MergeStats]:
    """? disease_id ????????????"""
    merged: OrderedDict[str, Optional[str]] = OrderedDict()
    total_before_dedup = 0
    for disease_id, disease_name in source_records:
        total_before_dedup += 1
        if disease_id not in merged:
            merged[disease_id] = disease_name
            continue
        if merged[disease_id] is None and disease_name is not None:
            merged[disease_id] = disease_name

    merged_records = list(merged.items())
    valid_name_count = sum(1 for _, name in merged_records if name is not None)
    invalid_name_count = len(merged_records) - valid_name_count
    stats = MergeStats(
        total_records_before_dedup=total_before_dedup,
        unique_disease_ids=len(merged_records),
        valid_name_count=valid_name_count,
        invalid_name_count=invalid_name_count,
    )
    return merged_records, stats


def iter_batches(items: Sequence[Tuple[str, Optional[str]]], batch_size: int) -> Iterable[List[Tuple[str, Optional[str]]]]:
    """? batch ???????"""
    for start_index in range(0, len(items), batch_size):
        yield list(items[start_index : start_index + batch_size])


def encode_text_batch(
    texts: Sequence[str],
    tokenizer: AutoTokenizer,
    model: AutoModel,
    device: torch.device,
    max_length: int,
) -> Tensor:
    """??? batch ??????? [CLS] ???"""
    encoded_inputs = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors='pt',
    )
    encoded_inputs = {key: value.to(device) for key, value in encoded_inputs.items()}

    with torch.no_grad():
        outputs = model(**encoded_inputs)
        cls_embeddings = outputs.last_hidden_state[:, 0, :].detach().cpu()
    return cls_embeddings


def build_disease_embedding_dict(
    merged_records: Sequence[Tuple[str, Optional[str]]],
    tokenizer: AutoTokenizer,
    model: AutoModel,
    device: torch.device,
    batch_size: int,
    max_length: int,
) -> Tuple[Dict[str, Tensor], Dict[str, int]]:
    """???????????? `disease_id -> embedding` ???"""
    zero_embedding = torch.zeros(768, dtype=torch.float32)
    embedding_dict: Dict[str, Tensor] = {}

    valid_records: List[Tuple[str, str]] = []
    zero_fill_count = 0
    success_count = 0
    encode_error_count = 0

    for disease_id, disease_name in merged_records:
        if disease_name is None:
            embedding_dict[disease_id] = zero_embedding.clone()
            zero_fill_count += 1
            continue
        valid_records.append((disease_id, disease_name))

    progress_bar = tqdm(
        iter_batches(valid_records, batch_size),
        total=math.ceil(len(valid_records) / batch_size) if valid_records else 0,
        desc='Extracting unified disease CLS embeddings',
        unit='batch',
    )

    for batch in progress_bar:
        batch_ids = [item[0] for item in batch]
        batch_texts = [item[1] for item in batch]
        try:
            batch_embeddings = encode_text_batch(
                texts=batch_texts,
                tokenizer=tokenizer,
                model=model,
                device=device,
                max_length=max_length,
            )
            for disease_id, embedding in zip(batch_ids, batch_embeddings):
                embedding_dict[disease_id] = embedding.clone()
                success_count += 1
        except Exception as batch_exception:
            print(
                'Warning: batch encoding failed; falling back to single-item encoding. '
                f'Batch size={len(batch_texts)}. Error: {batch_exception}'
            )
            for disease_id, disease_text in zip(batch_ids, batch_texts):
                try:
                    single_embedding = encode_text_batch(
                        texts=[disease_text],
                        tokenizer=tokenizer,
                        model=model,
                        device=device,
                        max_length=max_length,
                    )[0]
                    embedding_dict[disease_id] = single_embedding.clone()
                    success_count += 1
                except Exception as single_exception:
                    print(
                        f'Warning: single disease encoding failed for {disease_id!r}; '
                        f'fallback to zero. Error: {single_exception}'
                    )
                    embedding_dict[disease_id] = zero_embedding.clone()
                    zero_fill_count += 1
                    encode_error_count += 1

    stats = {
        'total_unique_diseases': int(len(merged_records)),
        'valid_name_rows': int(len(valid_records)),
        'success_count': int(success_count),
        'zero_fill_count': int(zero_fill_count),
        'encode_error_count': int(encode_error_count),
    }
    return embedding_dict, stats


def save_embedding_dict(output_pkl: Path, embedding_dict: Dict[str, Tensor]) -> None:
    """?????? pickle?"""
    output_pkl.parent.mkdir(parents=True, exist_ok=True)
    with output_pkl.open('wb') as file:
        pickle.dump(embedding_dict, file, protocol=pickle.HIGHEST_PROTOCOL)


def main() -> None:
    args = parse_args()
    device = detect_device()
    model_source = resolve_model_source(
        model_name_or_path=args.model_name_or_path,
        prefer_local_model=args.prefer_local_model,
    )
    source_specs = build_source_specs(args)

    print(f'Output PKL: {args.output_pkl}')
    print(f'Device: {device}')
    print(f'Model source: {model_source}')
    print('Enabled sources:')
    for spec in source_specs:
        print(f'  - {spec.source_name}: {spec.csv_path}')

    all_records: List[Tuple[str, Optional[str]]] = []
    source_stats: List[SourceLoadStats] = []
    for spec in source_specs:
        records, stats = extract_disease_records_from_source(spec)
        all_records.extend(records)
        source_stats.append(stats)
        print(
            f'[source={stats.source_name}] total_rows={stats.total_rows} '
            f'rows_after_filter={stats.rows_after_filter} '
            f'unique_ids_in_source={stats.unique_ids_in_source}'
        )

    merged_records, merge_stats = merge_disease_records(all_records)
    print(
        'After merge: '
        f'total_records_before_dedup={merge_stats.total_records_before_dedup} '
        f'unique_disease_ids={merge_stats.unique_disease_ids} '
        f'valid_names={merge_stats.valid_name_count} '
        f'invalid_names={merge_stats.invalid_name_count}'
    )

    tokenizer = AutoTokenizer.from_pretrained(model_source)
    model = AutoModel.from_pretrained(model_source)
    model = model.to(device)
    model.eval()

    embedding_dict, embedding_stats = build_disease_embedding_dict(
        merged_records=merged_records,
        tokenizer=tokenizer,
        model=model,
        device=device,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )
    save_embedding_dict(args.output_pkl, embedding_dict)

    print('Embedding extraction complete!')
    print(f"Total covered disease IDs: {len(embedding_dict)}")
    print(f"Successful encodes: {embedding_stats['success_count']}")
    print(f"Zero-filled diseases: {embedding_stats['zero_fill_count']}")
    print(f"Encoding fallback failures: {embedding_stats['encode_error_count']}")
    print(f'Saved to: {args.output_pkl}')


if __name__ == '__main__':
    main()
