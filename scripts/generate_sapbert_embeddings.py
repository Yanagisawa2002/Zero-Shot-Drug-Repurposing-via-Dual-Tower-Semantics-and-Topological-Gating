from __future__ import annotations

import argparse
import json
import math
import pickle
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import torch
from torch import Tensor
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


HF_MODEL_NAME = 'cambridgeltl/SapBERT-from-PubMedBERT-fulltext'
DEFAULT_INPUT_JSON = Path('data/disease_mechanism_texts.json')
DEFAULT_OUTPUT_PKL = Path('thick_disease_text_embeddings_sapbert.pkl')
LOCAL_MODEL_CANDIDATES = (
    Path('models/sapbert_from_pubmedbert_fulltext'),
    Path('models/SapBERT-from-PubMedBERT-fulltext'),
)
EMBED_DIM = 768


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Generate SapBERT disease embeddings from disease mechanism texts.',
    )
    parser.add_argument('--input-json', type=Path, default=DEFAULT_INPUT_JSON)
    parser.add_argument('--output-pkl', type=Path, default=DEFAULT_OUTPUT_PKL)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--max-length', type=int, default=512)
    parser.add_argument('--model-name-or-path', type=str, default=HF_MODEL_NAME)
    parser.add_argument('--prefer-local-model', action='store_true')
    return parser.parse_args()


def resolve_model_source(model_name_or_path: str, prefer_local_model: bool) -> str:
    explicit_path = Path(model_name_or_path)
    if explicit_path.exists():
        return str(explicit_path)

    if prefer_local_model:
        for candidate in LOCAL_MODEL_CANDIDATES:
            if candidate.exists():
                return str(candidate)

    return model_name_or_path


def detect_device() -> torch.device:
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def safe_text(value: object) -> str:
    if value is None:
        return ''
    text = str(value).strip()
    if not text:
        return ''
    if text.lower() in {'nan', 'none', 'null'}:
        return ''
    return text


def load_mechanism_json(input_json: Path) -> Dict[str, Dict[str, str]]:
    if not input_json.exists():
        raise FileNotFoundError(f'Input JSON does not exist: {input_json}')

    with input_json.open('r', encoding='utf-8') as file_pointer:
        payload = json.load(file_pointer)

    if not isinstance(payload, dict):
        raise TypeError(f'Expected top-level dict in {input_json}, got {type(payload)}')

    normalized: Dict[str, Dict[str, str]] = {}
    for disease_id, value in payload.items():
        if not isinstance(value, dict):
            raise TypeError(f'Expected dict value for disease_id={disease_id}, got {type(value)}')
        normalized[str(disease_id)] = {
            'name': safe_text(value.get('name')),
            'mechanism_text': safe_text(value.get('mechanism_text')),
        }

    if not normalized:
        raise ValueError(f'No disease entries found in {input_json}')
    return normalized


def build_text_to_encode(name: str, mechanism_text: str) -> str:
    if mechanism_text:
        return mechanism_text
    fallback_name = name if name else 'This disease'
    return f'{fallback_name} is a medical condition.'


def iter_batches(items: Sequence[Tuple[str, str, str]], batch_size: int) -> Iterable[List[Tuple[str, str, str]]]:
    for start_index in range(0, len(items), batch_size):
        yield list(items[start_index : start_index + batch_size])


def encode_text_batch(
    texts: Sequence[str],
    tokenizer: AutoTokenizer,
    model: AutoModel,
    device: torch.device,
    max_length: int,
) -> Tensor:
    encoded_inputs = tokenizer(
        list(texts),
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors='pt',
    )
    encoded_inputs = {key: value.to(device) for key, value in encoded_inputs.items()}

    with torch.no_grad():
        outputs = model(**encoded_inputs)
        cls_embeddings = outputs.last_hidden_state[:, 0, :].detach().cpu()

    del outputs
    del encoded_inputs
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return cls_embeddings


def build_embedding_dict(
    records: Sequence[Tuple[str, str, str]],
    tokenizer: AutoTokenizer,
    model: AutoModel,
    device: torch.device,
    batch_size: int,
    max_length: int,
) -> Tuple[Dict[str, Tensor], Dict[str, int]]:
    zero_embedding = torch.zeros(EMBED_DIM, dtype=torch.float32)
    embedding_dict: Dict[str, Tensor] = {}
    success_count = 0
    fallback_text_count = 0
    encode_error_count = 0

    progress_bar = tqdm(
        iter_batches(records, batch_size),
        total=math.ceil(len(records) / batch_size) if records else 0,
        desc='Encoding disease texts with SapBERT',
        unit='batch',
    )

    for batch in progress_bar:
        batch_ids = [item[0] for item in batch]
        batch_names = [item[1] for item in batch]
        batch_mechanisms = [item[2] for item in batch]
        batch_texts = []
        for name, mechanism_text in zip(batch_names, batch_mechanisms):
            if not mechanism_text:
                fallback_text_count += 1
            batch_texts.append(build_text_to_encode(name=name, mechanism_text=mechanism_text))

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
                'Warning: SapBERT batch encoding failed; falling back to single-item encoding. '
                f'Batch size={len(batch_texts)}. Error: {batch_exception}'
            )
            for disease_id, name, mechanism_text in zip(batch_ids, batch_names, batch_mechanisms):
                try:
                    single_text = build_text_to_encode(name=name, mechanism_text=mechanism_text)
                    single_embedding = encode_text_batch(
                        texts=[single_text],
                        tokenizer=tokenizer,
                        model=model,
                        device=device,
                        max_length=max_length,
                    )[0]
                    embedding_dict[disease_id] = single_embedding.clone()
                    success_count += 1
                except Exception as single_exception:
                    print(
                        f'Warning: SapBERT single-item encoding failed for {disease_id!r}; '
                        f'fallback to zero. Error: {single_exception}'
                    )
                    embedding_dict[disease_id] = zero_embedding.clone()
                    encode_error_count += 1

    stats = {
        'total_records': int(len(records)),
        'success_count': int(success_count),
        'fallback_text_count': int(fallback_text_count),
        'encode_error_count': int(encode_error_count),
    }
    return embedding_dict, stats


def save_embedding_dict(output_pkl: Path, embedding_dict: Dict[str, Tensor]) -> None:
    output_pkl.parent.mkdir(parents=True, exist_ok=True)
    with output_pkl.open('wb') as file_pointer:
        pickle.dump(embedding_dict, file_pointer, protocol=pickle.HIGHEST_PROTOCOL)


def main() -> None:
    args = parse_args()
    device = detect_device()
    model_source = resolve_model_source(
        model_name_or_path=args.model_name_or_path,
        prefer_local_model=args.prefer_local_model,
    )

    print(f'Input JSON: {args.input_json}')
    print(f'Output PKL: {args.output_pkl}')
    print(f'Device: {device}')
    print(f'Model source: {model_source}')

    mechanism_payload = load_mechanism_json(args.input_json)
    records = [
        (disease_id, item['name'], item['mechanism_text'])
        for disease_id, item in mechanism_payload.items()
    ]
    print(f'Total diseases to encode: {len(records)}')

    tokenizer = AutoTokenizer.from_pretrained(model_source)
    model = AutoModel.from_pretrained(model_source)
    model = model.to(device)
    model.eval()

    embedding_dict, stats = build_embedding_dict(
        records=records,
        tokenizer=tokenizer,
        model=model,
        device=device,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )
    save_embedding_dict(output_pkl=args.output_pkl, embedding_dict=embedding_dict)

    print('\nSapBERT extraction complete.')
    print(f"Total diseases: {stats['total_records']}")
    print(f"Successful encodes: {stats['success_count']}")
    print(f"Fallback texts used: {stats['fallback_text_count']}")
    print(f"Encoding failures: {stats['encode_error_count']}")
    print(f'Saved to: {args.output_pkl}')


if __name__ == '__main__':
    main()

