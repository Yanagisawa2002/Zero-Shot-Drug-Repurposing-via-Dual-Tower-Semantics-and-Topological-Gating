from __future__ import annotations

import argparse
import ast
import math
import pickle
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd
import torch
from torch import Tensor
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


HF_MODEL_NAME = 'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext'
LOCAL_MODEL_DIR = Path('models/pubmedbert_biomedbert_base_uncased_abstract_fulltext')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Extract thick disease PubMedBERT embeddings from disease intrinsic attributes only.',
    )
    parser.add_argument(
        '--disease-table',
        type=Path,
        default=Path('data/PrimeKG/nodes.csv'),
        help='?????????? PrimeKG nodes.csv?',
    )
    parser.add_argument('--disease-id-column', type=str, default='id')
    parser.add_argument('--disease-name-column', type=str, default='name')
    parser.add_argument('--disease-type-column', type=str, default='type')
    parser.add_argument('--disease-type-value', type=str, default='disease')

    parser.add_argument(
        '--ot-table',
        type=Path,
        action='append',
        default=[],
        help='Open Targets ??????????? CSV/XLSX?',
    )
    parser.add_argument('--ot-id-column', type=str, default='MONDO_ID')
    parser.add_argument('--ot-label-column', type=str, default='label')
    parser.add_argument('--ot-ancestors-column', type=str, default='ancestors')

    parser.add_argument(
        '--definition-table',
        type=Path,
        default=None,
        help='????????????? disease_id ? definition ??',
    )
    parser.add_argument('--definition-id-column', type=str, default='disease_id')
    parser.add_argument('--definition-text-column', type=str, default='definition')

    parser.add_argument(
        '--output-pkl',
        type=Path,
        default=Path('thick_disease_text_embeddings.pkl'),
        help='?? pickle ?????',
    )
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--max-length', type=int, default=256)
    parser.add_argument('--model-name-or-path', type=str, default=HF_MODEL_NAME)
    parser.add_argument('--prefer-local-model', action='store_true')
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
    """?????? GPU?"""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_table(table_path: Path) -> pd.DataFrame:
    """????????? CSV/TSV/XLSX?"""
    if not table_path.exists():
        raise FileNotFoundError(f'Table does not exist: {table_path}')

    suffix = table_path.suffix.lower()
    if suffix == '.csv':
        frame = pd.read_csv(table_path)
    elif suffix == '.tsv':
        frame = pd.read_csv(table_path, sep='\t')
    elif suffix in {'.xlsx', '.xls'}:
        frame = pd.read_excel(table_path)
    else:
        raise ValueError(f'Unsupported table format: {table_path}')

    if frame.empty:
        raise ValueError(f'Table is empty: {table_path}')
    return frame


def validate_columns(frame: pd.DataFrame, required_columns: Sequence[str], table_name: str) -> None:
    """?????????????"""
    missing_columns = [column for column in required_columns if column not in frame.columns]
    if missing_columns:
        raise ValueError(
            f'Missing required columns in {table_name}: {missing_columns}. '
            f'Available columns: {list(frame.columns)}'
        )


def is_missing_text(value: object) -> bool:
    """???????????"""
    if value is None or pd.isna(value):
        return True
    text = str(value).strip()
    if not text:
        return True
    return text.lower() in {'nan', 'none', 'null'}


def safe_text(value: object) -> str:
    """???????????????? 'nan'?"""
    if is_missing_text(value):
        return ''
    return str(value).strip()


def parse_ancestors_to_text(ancestors_value: object) -> str:
    """? OT ancestors ???????????"""
    raw_text = safe_text(ancestors_value)
    if not raw_text:
        return ''

    if isinstance(ancestors_value, (list, tuple, set)):
        items = [safe_text(item) for item in ancestors_value]
        return ', '.join([item for item in items if item])

    parsed_items: List[str] = []

    try:
        parsed_literal = ast.literal_eval(raw_text)
        if isinstance(parsed_literal, (list, tuple, set)):
            parsed_items = [safe_text(item) for item in parsed_literal]
    except Exception:
        parsed_literal = None

    if not parsed_items:
        cleaned = raw_text.replace('[', ' ').replace(']', ' ')
        cleaned = cleaned.replace('\n', ' ')
        candidate_items = re.split(r"[,'\"]+|\s{2,}", cleaned)
        parsed_items = [safe_text(item) for item in candidate_items]

    dedup_items: List[str] = []
    seen = set()
    for item in parsed_items:
        if not item:
            continue
        if item in seen:
            continue
        seen.add(item)
        dedup_items.append(item)
    return ', '.join(dedup_items)


def first_non_empty(series: pd.Series) -> str:
    """???????????????"""
    for value in series:
        text = safe_text(value)
        if text:
            return text
    return ''


def merge_non_empty_texts(series: pd.Series, parser=None) -> str:
    """????????????"""
    pieces: List[str] = []
    seen = set()
    for value in series:
        text = parser(value) if parser is not None else safe_text(value)
        if not text:
            continue
        for sub_piece in [item.strip() for item in text.split(',')]:
            if not sub_piece or sub_piece in seen:
                continue
            seen.add(sub_piece)
            pieces.append(sub_piece)
    return ', '.join(pieces)


def load_base_diseases(
    disease_table: Path,
    disease_id_column: str,
    disease_name_column: str,
    disease_type_column: str,
    disease_type_value: str,
) -> pd.DataFrame:
    """???????????????????"""
    frame = load_table(disease_table)
    validate_columns(
        frame,
        [disease_id_column, disease_name_column, disease_type_column],
        table_name=str(disease_table),
    )

    filtered = frame[frame[disease_type_column].astype(str) == str(disease_type_value)].copy()
    if filtered.empty:
        raise ValueError(
            f'No rows matched {disease_type_column} == {disease_type_value!r} in {disease_table}'
        )

    filtered = filtered[[disease_id_column, disease_name_column]].rename(
        columns={
            disease_id_column: 'disease_id',
            disease_name_column: 'name',
        }
    )
    filtered['disease_id'] = filtered['disease_id'].astype(str).str.strip()
    filtered['name'] = filtered['name'].map(safe_text)
    filtered = filtered[filtered['disease_id'] != ''].copy()
    filtered = filtered.drop_duplicates(subset=['disease_id'], keep='first')
    return filtered


def load_ot_context(
    ot_tables: Sequence[Path],
    ot_id_column: str,
    ot_label_column: str,
    ot_ancestors_column: str,
) -> pd.DataFrame:
    """????? OT ??? label/ancestors ??????????"""
    if not ot_tables:
        return pd.DataFrame(columns=['disease_id', 'ot_label', 'ancestors'])

    frames: List[pd.DataFrame] = []
    for ot_table in ot_tables:
        frame = load_table(ot_table)
        validate_columns(
            frame,
            [ot_id_column, ot_label_column, ot_ancestors_column],
            table_name=str(ot_table),
        )
        subset = frame[[ot_id_column, ot_label_column, ot_ancestors_column]].rename(
            columns={
                ot_id_column: 'disease_id',
                ot_label_column: 'ot_label',
                ot_ancestors_column: 'ancestors',
            }
        )
        subset['disease_id'] = subset['disease_id'].astype(str).str.strip()
        subset = subset[subset['disease_id'] != ''].copy()
        frames.append(subset)

    merged = pd.concat(frames, axis=0, ignore_index=True)
    aggregated = (
        merged.groupby('disease_id', as_index=False)
        .agg(
            {
                'ot_label': first_non_empty,
                'ancestors': lambda series: merge_non_empty_texts(series, parser=parse_ancestors_to_text),
            }
        )
        .copy()
    )
    return aggregated


def load_definitions(
    definition_table: Optional[Path],
    definition_id_column: str,
    definition_text_column: str,
) -> pd.DataFrame:
    """???????????"""
    if definition_table is None:
        return pd.DataFrame(columns=['disease_id', 'definition'])

    frame = load_table(definition_table)
    validate_columns(
        frame,
        [definition_id_column, definition_text_column],
        table_name=str(definition_table),
    )
    subset = frame[[definition_id_column, definition_text_column]].rename(
        columns={
            definition_id_column: 'disease_id',
            definition_text_column: 'definition',
        }
    )
    subset['disease_id'] = subset['disease_id'].astype(str).str.strip()
    subset = subset[subset['disease_id'] != ''].copy()
    aggregated = subset.groupby('disease_id', as_index=False).agg({'definition': first_non_empty}).copy()
    return aggregated


def build_thick_disease_table(
    base_diseases: pd.DataFrame,
    ot_context: pd.DataFrame,
    definition_context: pd.DataFrame,
) -> pd.DataFrame:
    """? disease_id ???????????"""
    merged = base_diseases.merge(ot_context, on='disease_id', how='left')
    merged = merged.merge(definition_context, on='disease_id', how='left')

    merged['name'] = merged['name'].map(safe_text)
    merged['ot_label'] = merged['ot_label'].map(safe_text) if 'ot_label' in merged.columns else ''
    merged['ancestors'] = merged['ancestors'].map(safe_text) if 'ancestors' in merged.columns else ''
    merged['definition'] = merged['definition'].map(safe_text) if 'definition' in merged.columns else ''

    # ?? PrimeKG ?????????? OT label ???????
    merged.loc[merged['name'] == '', 'name'] = merged.loc[merged['name'] == '', 'ot_label']

    return merged[['disease_id', 'name', 'ancestors', 'definition']].copy()


def build_text_to_encode(name: str, ancestors: str, definition: str) -> str:
    """????????????????????????"""
    segments = [f'Disease: {name}.']
    if ancestors:
        segments.append(f'Classification/Ancestors: {ancestors}.')
    if definition:
        segments.append(f'Definition: {definition}.')
    return ' '.join(segments)


def iter_batches(items: Sequence[Tuple[str, str]], batch_size: int) -> Iterable[List[Tuple[str, str]]]:
    """? batch ?? `(disease_id, text)` ???"""
    for start_index in range(0, len(items), batch_size):
        yield list(items[start_index : start_index + batch_size])


def encode_text_batch(
    texts: Sequence[str],
    tokenizer: AutoTokenizer,
    model: AutoModel,
    device: torch.device,
    max_length: int,
) -> Tensor:
    """??????? [CLS] ???"""
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


def build_embedding_dict(
    disease_table: pd.DataFrame,
    tokenizer: AutoTokenizer,
    model: AutoModel,
    device: torch.device,
    batch_size: int,
    max_length: int,
) -> Tuple[Dict[str, Tensor], Dict[str, int]]:
    """????????? `disease_id -> embedding` ???"""
    zero_embedding = torch.zeros(768, dtype=torch.float32)
    embedding_dict: Dict[str, Tensor] = {}

    valid_records: List[Tuple[str, str]] = []
    zero_fill_count = 0
    success_count = 0
    encode_error_count = 0

    for row in disease_table.itertuples(index=False):
        disease_id = str(row.disease_id)
        text_to_encode = build_text_to_encode(
            name=safe_text(row.name),
            ancestors=safe_text(row.ancestors),
            definition=safe_text(row.definition),
        )
        if not text_to_encode.strip() or text_to_encode.strip() == 'Disease: .':
            embedding_dict[disease_id] = zero_embedding.clone()
            zero_fill_count += 1
            continue
        valid_records.append((disease_id, text_to_encode))

    progress_bar = tqdm(
        iter_batches(valid_records, batch_size),
        total=math.ceil(len(valid_records) / batch_size) if valid_records else 0,
        desc='Extracting thick disease CLS embeddings',
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
            for disease_id, text_to_encode in zip(batch_ids, batch_texts):
                try:
                    single_embedding = encode_text_batch(
                        texts=[text_to_encode],
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
        'total_diseases': int(len(disease_table)),
        'valid_text_rows': int(len(valid_records)),
        'success_count': int(success_count),
        'zero_fill_count': int(zero_fill_count),
        'encode_error_count': int(encode_error_count),
    }
    return embedding_dict, stats


def save_embedding_dict(output_pkl: Path, embedding_dict: Dict[str, Tensor]) -> None:
    """???????"""
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

    print(f'Disease table: {args.disease_table}')
    print(f'OT tables: {[str(path) for path in args.ot_table]}')
    print(f'Definition table: {args.definition_table}')
    print(f'Output PKL: {args.output_pkl}')
    print(f'Device: {device}')
    print(f'Model source: {model_source}')

    base_diseases = load_base_diseases(
        disease_table=args.disease_table,
        disease_id_column=args.disease_id_column,
        disease_name_column=args.disease_name_column,
        disease_type_column=args.disease_type_column,
        disease_type_value=args.disease_type_value,
    )
    ot_context = load_ot_context(
        ot_tables=args.ot_table,
        ot_id_column=args.ot_id_column,
        ot_label_column=args.ot_label_column,
        ot_ancestors_column=args.ot_ancestors_column,
    )
    definition_context = load_definitions(
        definition_table=args.definition_table,
        definition_id_column=args.definition_id_column,
        definition_text_column=args.definition_text_column,
    )
    thick_disease_table = build_thick_disease_table(
        base_diseases=base_diseases,
        ot_context=ot_context,
        definition_context=definition_context,
    )

    print(f'Base disease nodes: {len(base_diseases)}')
    print(f'OT context rows after aggregation: {len(ot_context)}')
    print(f'Definition rows after aggregation: {len(definition_context)}')
    print(f'Unified thick disease rows: {len(thick_disease_table)}')

    tokenizer = AutoTokenizer.from_pretrained(model_source)
    model = AutoModel.from_pretrained(model_source)
    model = model.to(device)
    model.eval()

    embedding_dict, stats = build_embedding_dict(
        disease_table=thick_disease_table,
        tokenizer=tokenizer,
        model=model,
        device=device,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )
    save_embedding_dict(args.output_pkl, embedding_dict)

    print('Extraction complete!')
    print(f"Total covered disease IDs: {len(embedding_dict)}")
    print(f"Successful encodes: {stats['success_count']}")
    print(f"Zero-filled diseases: {stats['zero_fill_count']}")
    print(f"Encoding fallback failures: {stats['encode_error_count']}")
    print(f'Saved to: {args.output_pkl}')


if __name__ == '__main__':
    main()
