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


TripletKey = Tuple[str, str, str]

HF_MODEL_NAME = 'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext'
LOCAL_MODEL_DIR = Path('models/pubmedbert_biomedbert_base_uncased_abstract_fulltext')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Extract PubMedBERT embeddings for cleaned triplet-level mechanism texts.',
    )
    parser.add_argument(
        '--input-csv',
        type=Path,
        default=Path('triple_level_pubmedbert_llm_cleaned.csv'),
        help='Input CSV containing triplet-level mechanism texts.',
    )
    parser.add_argument(
        '--output-pkl',
        type=Path,
        default=Path('triplet_text_embeddings.pkl'),
        help='Output pickle file that stores {(drug_id, disease_id, protein_id): embedding}.',
    )
    parser.add_argument(
        '--text-column',
        type=str,
        default='triple_summary_llm_pubmedbert',
        help='Text column used to extract PubMedBERT features.',
    )
    parser.add_argument(
        '--drug-id-column',
        type=str,
        default='drug_id',
        help='Drug ID column.',
    )
    parser.add_argument(
        '--disease-id-column',
        type=str,
        default='disease_id',
        help='Disease ID column.',
    )
    parser.add_argument(
        '--protein-id-column',
        type=str,
        default='protein_id',
        help='Protein/Gene ID column.',
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Batch size for transformer inference.',
    )
    parser.add_argument(
        '--max-length',
        type=int,
        default=512,
        help='Maximum token length passed to PubMedBERT.',
    )
    parser.add_argument(
        '--model-name-or-path',
        type=str,
        default=HF_MODEL_NAME,
        help='Transformers model name or local model path.',
    )
    parser.add_argument(
        '--prefer-local-model',
        action='store_true',
        help='Prefer the local cached PubMedBERT directory when available.',
    )
    return parser.parse_args()


def resolve_model_source(model_name_or_path: str, prefer_local_model: bool) -> str:
    """????????????????????????"""
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


def load_triplet_dataframe(input_csv: Path) -> pd.DataFrame:
    """????????????????????"""
    if not input_csv.exists():
        raise FileNotFoundError(f'Input CSV does not exist: {input_csv}')

    frame = pd.read_csv(input_csv)
    if frame.empty:
        raise ValueError(f'Input CSV is empty: {input_csv}')
    return frame


def validate_required_columns(frame: pd.DataFrame, required_columns: Sequence[str]) -> None:
    missing_columns = [column for column in required_columns if column not in frame.columns]
    if missing_columns:
        raise ValueError(
            f'Missing required columns: {missing_columns}. Available columns: {list(frame.columns)}'
        )


def normalize_texts(text_series: pd.Series) -> List[str]:
    """????????????????? NaN ?? tokenizer?"""
    return text_series.fillna('').astype(str).tolist()


def build_triplet_keys(
    frame: pd.DataFrame,
    drug_id_column: str,
    disease_id_column: str,
    protein_id_column: str,
) -> List[TripletKey]:
    """?????????????? Tuple??? pickle ???????"""
    keys: List[TripletKey] = []
    for row in frame[[drug_id_column, disease_id_column, protein_id_column]].itertuples(index=False, name=None):
        drug_id, disease_id, protein_id = row
        keys.append((str(drug_id), str(disease_id), str(protein_id)))
    return keys


def iter_batches(items: Sequence[str], batch_size: int) -> Iterable[Tuple[int, List[str]]]:
    """? batch ????????????"""
    for start_index in range(0, len(items), batch_size):
        yield start_index, list(items[start_index : start_index + batch_size])


def extract_cls_embeddings(
    texts: Sequence[str],
    tokenizer: AutoTokenizer,
    model: AutoModel,
    device: torch.device,
    batch_size: int,
    max_length: int,
) -> Tensor:
    """
    ?? [CLS] token ??????????????

    ?????: (num_samples, hidden_dim)
    """
    all_embeddings: List[Tensor] = []
    num_batches = math.ceil(len(texts) / batch_size)

    progress_bar = tqdm(
        iter_batches(texts, batch_size),
        total=num_batches,
        desc='Extracting PubMedBERT CLS embeddings',
        unit='batch',
    )

    with torch.no_grad():
        for _, batch_texts in progress_bar:
            encoded_inputs = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors='pt',
            )
            encoded_inputs = {
                key: value.to(device)
                for key, value in encoded_inputs.items()
            }

            outputs = model(**encoded_inputs)
            cls_embeddings = outputs.last_hidden_state[:, 0, :].detach().cpu()
            all_embeddings.append(cls_embeddings)

    if not all_embeddings:
        raise ValueError('No embeddings were extracted. Please check the input texts.')
    return torch.cat(all_embeddings, dim=0)


def build_embedding_dict(keys: Sequence[TripletKey], embeddings: Tensor) -> Dict[TripletKey, Tensor]:
    """?????? CPU embedding ????? Python ???"""
    if len(keys) != int(embeddings.size(0)):
        raise ValueError(
            f'Key count and embedding count mismatch: {len(keys)} vs {int(embeddings.size(0))}'
        )

    embedding_dict: Dict[TripletKey, Tensor] = {}
    duplicate_key_count = 0
    for key, embedding in zip(keys, embeddings):
        if key in embedding_dict:
            duplicate_key_count += 1
        embedding_dict[key] = embedding.clone()

    if duplicate_key_count > 0:
        print(f'Warning: detected {duplicate_key_count} duplicate triplet keys; later rows overwrote earlier ones.')
    return embedding_dict


def save_embedding_dict(output_pkl: Path, embedding_dict: Dict[TripletKey, Tensor]) -> None:
    """?? pickle ????? embedding ???"""
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

    frame = load_triplet_dataframe(args.input_csv)
    validate_required_columns(
        frame,
        required_columns=[
            args.drug_id_column,
            args.disease_id_column,
            args.protein_id_column,
            args.text_column,
        ],
    )

    tokenizer = AutoTokenizer.from_pretrained(model_source)
    model = AutoModel.from_pretrained(model_source)
    model = model.to(device)
    model.eval()

    triplet_keys = build_triplet_keys(
        frame=frame,
        drug_id_column=args.drug_id_column,
        disease_id_column=args.disease_id_column,
        protein_id_column=args.protein_id_column,
    )
    texts = normalize_texts(frame[args.text_column])

    embeddings = extract_cls_embeddings(
        texts=texts,
        tokenizer=tokenizer,
        model=model,
        device=device,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )
    embedding_dict = build_embedding_dict(keys=triplet_keys, embeddings=embeddings)
    save_embedding_dict(output_pkl=args.output_pkl, embedding_dict=embedding_dict)

    print(f'Rows processed: {len(texts)}')
    print(f'Unique triplet keys saved: {len(embedding_dict)}')
    print(f'Embedding shape per triplet: {tuple(embeddings[0].shape)}')
    print('Finished extracting and saving triplet embeddings.')


if __name__ == '__main__':
    main()
