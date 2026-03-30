from __future__ import annotations

import argparse
import math
import pickle
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import pandas as pd
import torch
from torch import Tensor
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


HF_MODEL_NAME = 'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext'
LOCAL_MODEL_DIR = Path('models/pubmedbert_biomedbert_base_uncased_abstract_fulltext')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Extract PubMedBERT CLS embeddings for disease names.',
    )
    parser.add_argument(
        '--input-csv',
        type=Path,
        default=Path('data/PrimeKG/nodes.csv'),
        help='??????? CSV ?????',
    )
    parser.add_argument(
        '--output-pkl',
        type=Path,
        default=Path('disease_text_embeddings.pkl'),
        help='??? pickle ?????',
    )
    parser.add_argument(
        '--node-id-column',
        type=str,
        default='id',
        help='?? ID ???',
    )
    parser.add_argument(
        '--disease-name-column',
        type=str,
        default='name',
        help='???????',
    )
    parser.add_argument(
        '--filter-type-column',
        type=str,
        default='type',
        help='?????????????????????????',
    )
    parser.add_argument(
        '--filter-type-value',
        type=str,
        default='disease',
        help='??????????',
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=64,
        help='??????',
    )
    parser.add_argument(
        '--max-length',
        type=int,
        default=512,
        help='?? PubMedBERT ??? token ???',
    )
    parser.add_argument(
        '--model-name-or-path',
        type=str,
        default=HF_MODEL_NAME,
        help='Transformers ???????????',
    )
    parser.add_argument(
        '--prefer-local-model',
        action='store_true',
        help='????????????????????',
    )
    return parser.parse_args()


def resolve_model_source(model_name_or_path: str, prefer_local_model: bool) -> str:
    """????????????"""
    if prefer_local_model and LOCAL_MODEL_DIR.exists():
        return str(LOCAL_MODEL_DIR)

    explicit_path = Path(model_name_or_path)
    if explicit_path.exists():
        return str(explicit_path)

    if LOCAL_MODEL_DIR.exists() and model_name_or_path == HF_MODEL_NAME:
        return str(LOCAL_MODEL_DIR)

    return model_name_or_path


def detect_device() -> torch.device:
    """???? GPU???????? CPU?"""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_input_dataframe(input_csv: Path) -> pd.DataFrame:
    """???? CSV?"""
    if not input_csv.exists():
        raise FileNotFoundError(f'Input CSV does not exist: {input_csv}')

    frame = pd.read_csv(input_csv)
    if frame.empty:
        raise ValueError(f'Input CSV is empty: {input_csv}')
    return frame


def validate_required_columns(frame: pd.DataFrame, required_columns: Sequence[str]) -> None:
    missing_columns = [column for column in required_columns if column and column not in frame.columns]
    if missing_columns:
        raise ValueError(
            f'Missing required columns: {missing_columns}. Available columns: {list(frame.columns)}'
        )


def filter_disease_rows(
    frame: pd.DataFrame,
    filter_type_column: str,
    filter_type_value: str,
) -> pd.DataFrame:
    """
    ????????

    ? `filter_type_column` ???????????
    """
    if not filter_type_column:
        return frame.copy()

    if filter_type_column not in frame.columns:
        raise ValueError(
            f'Filter type column {filter_type_column!r} does not exist in input CSV.'
        )

    filtered = frame[frame[filter_type_column].astype(str) == str(filter_type_value)].copy()
    if filtered.empty:
        raise ValueError(
            f'No rows matched {filter_type_column} == {filter_type_value!r}. '
            'Please check your filtering settings.'
        )
    return filtered


def is_valid_disease_name(name_value: object) -> bool:
    """??????????????"""
    if name_value is None:
        return False
    if pd.isna(name_value):
        return False

    name_text = str(name_value).strip()
    if not name_text:
        return False
    if name_text.lower() in {'nan', 'none', 'null'}:
        return False
    return True


def iter_batches(items: Sequence[Tuple[str, str]], batch_size: int) -> Iterable[List[Tuple[str, str]]]:
    """? batch ?? `(node_id, disease_name)` ???"""
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


def build_disease_embedding_dict(
    frame: pd.DataFrame,
    node_id_column: str,
    disease_name_column: str,
    tokenizer: AutoTokenizer,
    model: AutoModel,
    device: torch.device,
    batch_size: int,
    max_length: int,
) -> Tuple[Dict[str, Tensor], Dict[str, int]]:
    """
    ?? `node_id -> disease embedding` ???

    ?????
    - ????????? batch ???
    - ??????????
    - batch ?????????????
    - ?????????
    """
    zero_embedding = torch.zeros(768, dtype=torch.float32)
    embedding_dict: Dict[str, Tensor] = {}

    valid_records: List[Tuple[str, str]] = []
    zero_fill_count = 0
    invalid_name_count = 0
    encode_error_count = 0
    success_count = 0

    for row in frame[[node_id_column, disease_name_column]].itertuples(index=False, name=None):
        node_id, disease_name = row
        node_id = str(node_id)
        if not is_valid_disease_name(disease_name):
            embedding_dict[node_id] = zero_embedding.clone()
            zero_fill_count += 1
            invalid_name_count += 1
            continue
        valid_records.append((node_id, str(disease_name).strip()))

    progress_bar = tqdm(
        iter_batches(valid_records, batch_size),
        total=math.ceil(len(valid_records) / batch_size) if valid_records else 0,
        desc='Extracting disease CLS embeddings',
        unit='batch',
    )

    for batch in progress_bar:
        batch_node_ids = [item[0] for item in batch]
        batch_texts = [item[1] for item in batch]

        try:
            batch_embeddings = encode_text_batch(
                texts=batch_texts,
                tokenizer=tokenizer,
                model=model,
                device=device,
                max_length=max_length,
            )
            for node_id, embedding in zip(batch_node_ids, batch_embeddings):
                embedding_dict[node_id] = embedding.clone()
                success_count += 1
        except Exception as batch_exception:
            print(
                'Warning: batch encoding failed; falling back to single-item encoding. '
                f'Batch size={len(batch_texts)}. Error: {batch_exception}'
            )
            for node_id, text in zip(batch_node_ids, batch_texts):
                try:
                    single_embedding = encode_text_batch(
                        texts=[text],
                        tokenizer=tokenizer,
                        model=model,
                        device=device,
                        max_length=max_length,
                    )[0]
                    embedding_dict[node_id] = single_embedding.clone()
                    success_count += 1
                except Exception as single_exception:
                    print(
                        f'Warning: single disease encoding failed for {node_id!r}; '
                        f'fallback to zero. Error: {single_exception}'
                    )
                    embedding_dict[node_id] = zero_embedding.clone()
                    zero_fill_count += 1
                    encode_error_count += 1

    stats = {
        'total_rows': int(len(frame)),
        'valid_name_rows': int(len(valid_records)),
        'success_count': int(success_count),
        'zero_fill_count': int(zero_fill_count),
        'invalid_name_count': int(invalid_name_count),
        'encode_error_count': int(encode_error_count),
    }
    return embedding_dict, stats


def save_embedding_dict(output_pkl: Path, embedding_dict: Dict[str, Tensor]) -> None:
    """???? embedding ???"""
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

    print(f'Input CSV: {args.input_csv}')
    print(f'Output PKL: {args.output_pkl}')
    print(f'Device: {device}')
    print(f'Model source: {model_source}')

    frame = load_input_dataframe(args.input_csv)
    validate_required_columns(
        frame,
        required_columns=[
            args.node_id_column,
            args.disease_name_column,
            args.filter_type_column,
        ] if args.filter_type_column else [args.node_id_column, args.disease_name_column],
    )
    disease_frame = filter_disease_rows(
        frame=frame,
        filter_type_column=args.filter_type_column,
        filter_type_value=args.filter_type_value,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_source)
    model = AutoModel.from_pretrained(model_source)
    model = model.to(device)
    model.eval()

    embedding_dict, stats = build_disease_embedding_dict(
        frame=disease_frame,
        node_id_column=args.node_id_column,
        disease_name_column=args.disease_name_column,
        tokenizer=tokenizer,
        model=model,
        device=device,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )
    save_embedding_dict(output_pkl=args.output_pkl, embedding_dict=embedding_dict)

    print('\nExtraction complete.')
    print(f"Disease rows processed: {stats['total_rows']}")
    print(f"Valid disease names: {stats['valid_name_rows']}")
    print(f"Successfully encoded: {stats['success_count']}")
    print(f"Zero-filled: {stats['zero_fill_count']}")
    print(f"  Invalid/missing names: {stats['invalid_name_count']}")
    print(f"  Encoding failures: {stats['encode_error_count']}")
    print(f'Embeddings saved: {len(embedding_dict)}')
    print(f'Saved to: {args.output_pkl}')


if __name__ == '__main__':
    main()
