# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import math
import pickle
import re
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import pandas as pd
import torch
from torch import Tensor
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

HF_MODEL_NAME = 'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext'
LOCAL_MODEL_DIR = Path('models/pubmedbert_biomedbert_base_uncased_abstract_fulltext')
GENE_NODE_TYPE = 'gene/protein'
EMBED_DIM = 768


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Extract mean-pooled PubMedBERT embeddings for PrimeKG gene/protein nodes using MyGene annotations.',
    )
    parser.add_argument('--nodes-csv', type=Path, default=Path('data/PrimeKG/nodes.csv'), help='PrimeKG nodes.csv path.')
    parser.add_argument('--output-pkl', type=Path, default=Path('gene_text_embeddings_mean.pkl'), help='Output pickle path.')
    parser.add_argument('--node-id-column', type=str, default='id', help='Node ID column name.')
    parser.add_argument('--node-type-column', type=str, default='type', help='Node type column name.')
    parser.add_argument('--node-name-column', type=str, default='name', help='Node display name column name.')
    parser.add_argument('--gene-type-value', type=str, default=GENE_NODE_TYPE, help='Value that marks gene/protein nodes.')
    parser.add_argument('--mygene-batch-size', type=int, default=1000, help='Batch size for MyGene API calls.')
    parser.add_argument('--bert-batch-size', type=int, default=64, help='Batch size for PubMedBERT encoding.')
    parser.add_argument('--max-length', type=int, default=256, help='Maximum token length for PubMedBERT.')
    parser.add_argument('--model-name-or-path', type=str, default=HF_MODEL_NAME, help='HF model name or local path.')
    parser.add_argument('--prefer-local-model', action='store_true', help='Prefer the local cached PubMedBERT model.')
    parser.add_argument('--fields', type=str, default='symbol,name,summary,go', help='Fields requested from MyGene.')
    return parser.parse_args()


def resolve_model_source(model_name_or_path: str, prefer_local_model: bool) -> str:
    if prefer_local_model and LOCAL_MODEL_DIR.exists():
        return str(LOCAL_MODEL_DIR)

    explicit_path = Path(model_name_or_path)
    if explicit_path.exists():
        return str(explicit_path)

    if LOCAL_MODEL_DIR.exists() and model_name_or_path == HF_MODEL_NAME:
        return str(LOCAL_MODEL_DIR)

    return model_name_or_path


def detect_device() -> torch.device:
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_gene_nodes(
    nodes_csv: Path,
    node_id_column: str,
    node_type_column: str,
    node_name_column: str,
    gene_type_value: str,
) -> pd.DataFrame:
    """?? PrimeKG ?????? gene/protein ???"""
    if not nodes_csv.exists():
        raise FileNotFoundError(f'Nodes CSV does not exist: {nodes_csv}')

    frame = pd.read_csv(nodes_csv)
    required_columns = [node_id_column, node_type_column, node_name_column]
    missing_columns = [column for column in required_columns if column not in frame.columns]
    if missing_columns:
        raise ValueError(
            f'Missing required columns: {missing_columns}. Available columns: {list(frame.columns)}'
        )

    filtered = frame[frame[node_type_column].astype(str) == str(gene_type_value)].copy()
    if filtered.empty:
        raise ValueError(
            f'No rows matched {node_type_column} == {gene_type_value!r} in {nodes_csv}.'
        )

    filtered = filtered[[node_id_column, node_name_column]].rename(
        columns={node_id_column: 'node_id', node_name_column: 'node_name'}
    )
    filtered['node_id'] = filtered['node_id'].astype(str).str.strip()
    filtered['node_name'] = filtered['node_name'].fillna('').astype(str).str.strip()
    return filtered


def extract_entrez_id(node_id: str) -> Optional[str]:
    """? gene/protein::9796 ???? ID ??? Entrez ID?"""
    match = re.search(r'(\d+)$', str(node_id).strip())
    if match is None:
        return None
    return match.group(1)


def attach_entrez_ids(frame: pd.DataFrame) -> pd.DataFrame:
    enriched = frame.copy()
    enriched['entrez_id'] = enriched['node_id'].map(extract_entrez_id)
    return enriched


def iter_batches(items: Sequence[str], batch_size: int) -> Iterable[List[str]]:
    for start_index in range(0, len(items), batch_size):
        yield list(items[start_index : start_index + batch_size])


def import_mygene_module():
    try:
        import mygene  # type: ignore
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError(
            'The `mygene` package is not installed. Please run `pip install mygene` before executing this script.'
        ) from error
    return mygene


def query_mygene_records(
    entrez_ids: Sequence[str],
    fields: str,
    mygene_batch_size: int,
) -> Dict[str, dict]:
    """???? MyGene???? Entrez ID ??????????"""
    mygene = import_mygene_module()
    mg = mygene.MyGeneInfo()

    record_by_entrez_id: Dict[str, dict] = {}
    progress_bar = tqdm(
        iter_batches(list(entrez_ids), mygene_batch_size),
        total=math.ceil(len(entrez_ids) / mygene_batch_size) if entrez_ids else 0,
        desc='Fetching MyGene records',
        unit='batch',
    )

    for batch_ids in progress_bar:
        try:
            records = mg.getgenes(batch_ids, fields=fields, as_dataframe=False)
        except Exception as batch_error:
            print(
                'Warning: MyGene batch query failed. Falling back to per-ID fetch for this batch. '
                f'Error: {batch_error}'
            )
            records = []
            for entrez_id in batch_ids:
                try:
                    single_record = mg.getgene(entrez_id, fields=fields)
                except Exception as single_error:
                    print(f'Warning: MyGene lookup failed for Entrez {entrez_id}: {single_error}')
                    single_record = {'query': entrez_id, 'notfound': True}
                records.append(single_record)

        for record in records:
            if not isinstance(record, dict):
                continue
            query_value = record.get('query')
            if query_value is None:
                continue
            record_by_entrez_id[str(query_value)] = record

    return record_by_entrez_id


def is_missing_text(value: object) -> bool:
    if value is None or pd.isna(value):
        return True
    text = str(value).strip()
    if not text:
        return True
    return text.lower() in {'nan', 'none', 'null'}


def safe_text(value: object, default: str = '') -> str:
    if is_missing_text(value):
        return default
    return str(value).strip()


def _extract_go_terms(go_entry: object, branch_name: str) -> List[str]:
    if not isinstance(go_entry, Mapping):
        return []
    branch_value = go_entry.get(branch_name)
    if branch_value is None:
        return []

    if isinstance(branch_value, Mapping):
        branch_items = [branch_value]
    elif isinstance(branch_value, list):
        branch_items = branch_value
    else:
        return []

    terms: List[str] = []
    seen = set()
    for item in branch_items:
        if not isinstance(item, Mapping):
            continue
        term = safe_text(item.get('term'))
        if not term or term in seen:
            continue
        seen.add(term)
        terms.append(term)
    return terms


def _join_non_empty(parts: Sequence[str]) -> str:
    cleaned = [part.strip() for part in parts if part and part.strip()]
    return ' '.join(cleaned)


def build_gene_text(record: Optional[dict], fallback_name: str) -> str:
    """??????????????????"""
    if record is None or record.get('notfound'):
        return safe_text(fallback_name, default='Unknown')

    symbol = safe_text(record.get('symbol'))
    full_name = safe_text(record.get('name'))
    summary = safe_text(record.get('summary'))
    go_entry = record.get('go')
    go_mf_terms = ', '.join(_extract_go_terms(go_entry, 'MF'))
    go_bp_terms = ', '.join(_extract_go_terms(go_entry, 'BP'))

    primary_name = _join_non_empty([symbol, full_name])
    if not primary_name:
        primary_name = safe_text(fallback_name, default='Unknown')

    content_parts = [primary_name]
    if summary:
        content_parts.append(summary)
    if go_mf_terms:
        content_parts.append(go_mf_terms)
    if go_bp_terms:
        content_parts.append(go_bp_terms)

    text = '. '.join([part for part in content_parts if part])
    return re.sub(r'\s+', ' ', text).strip()


def load_pubmedbert(model_name_or_path: str, device: torch.device) -> Tuple[AutoTokenizer, AutoModel]:
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModel.from_pretrained(model_name_or_path)
    model.eval()
    model.to(device)
    return tokenizer, model


def mean_pool_last_hidden(last_hidden_state: Tensor, attention_mask: Tensor) -> Tensor:
    """?? attention_mask ???? mean pooling??? CLS ?????"""
    mask = attention_mask.unsqueeze(-1).to(dtype=last_hidden_state.dtype)
    masked_hidden = last_hidden_state * mask
    pooled_sum = masked_hidden.sum(dim=1)
    valid_counts = mask.sum(dim=1).clamp(min=1e-9)
    return pooled_sum / valid_counts


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
        pooled_embeddings = mean_pool_last_hidden(
            last_hidden_state=outputs.last_hidden_state,
            attention_mask=encoded_inputs['attention_mask'],
        ).detach().cpu()
    return pooled_embeddings


def build_gene_embedding_dict(
    gene_frame: pd.DataFrame,
    text_by_node_id: Dict[str, str],
    tokenizer: AutoTokenizer,
    model: AutoModel,
    device: torch.device,
    bert_batch_size: int,
    max_length: int,
) -> Tuple[Dict[str, Tensor], Dict[str, int]]:
    """?????? gene ????????????????"""
    embedding_dict: Dict[str, Tensor] = {}
    zero_embedding = torch.zeros(EMBED_DIM, dtype=torch.float32)
    encode_error_count = 0
    success_count = 0

    node_records = list(text_by_node_id.items())
    progress_bar = tqdm(
        range(0, len(node_records), bert_batch_size),
        total=math.ceil(len(node_records) / bert_batch_size) if node_records else 0,
        desc='Encoding gene texts with PubMedBERT',
        unit='batch',
    )

    for start_index in progress_bar:
        batch_records = node_records[start_index : start_index + bert_batch_size]
        batch_node_ids = [item[0] for item in batch_records]
        batch_texts = [item[1] for item in batch_records]

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
        except Exception as batch_error:
            print(
                'Warning: PubMedBERT batch encoding failed; falling back to single-item encoding. '
                f'Error: {batch_error}'
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
                except Exception as single_error:
                    print(f'Warning: failed to encode node {node_id}: {single_error}')
                    embedding_dict[node_id] = zero_embedding.clone()
                    encode_error_count += 1

    for node_id in gene_frame['node_id'].astype(str):
        if node_id not in embedding_dict:
            embedding_dict[node_id] = zero_embedding.clone()
            encode_error_count += 1

    stats = {
        'total_nodes': int(len(gene_frame)),
        'success_count': int(success_count),
        'zero_fill_count': int(max(len(gene_frame) - success_count, 0)),
        'encode_error_count': int(encode_error_count),
    }
    return embedding_dict, stats


def save_embeddings(output_pkl: Path, embedding_dict: Dict[str, Tensor]) -> None:
    output_pkl.parent.mkdir(parents=True, exist_ok=True)
    with output_pkl.open('wb') as file_pointer:
        pickle.dump(embedding_dict, file_pointer)


def main() -> None:
    args = parse_args()
    device = detect_device()
    model_source = resolve_model_source(args.model_name_or_path, args.prefer_local_model)

    print('Loading PrimeKG gene/protein nodes...')
    gene_frame = load_gene_nodes(
        nodes_csv=args.nodes_csv,
        node_id_column=args.node_id_column,
        node_type_column=args.node_type_column,
        node_name_column=args.node_name_column,
        gene_type_value=args.gene_type_value,
    )
    gene_frame = attach_entrez_ids(gene_frame)

    valid_entrez_frame = gene_frame[gene_frame['entrez_id'].notna()].copy()
    entrez_ids = valid_entrez_frame['entrez_id'].astype(str).drop_duplicates().tolist()
    invalid_entrez_count = int(gene_frame['entrez_id'].isna().sum())

    print(f'Total gene/protein nodes: {len(gene_frame)}')
    print(f'Valid Entrez IDs: {len(entrez_ids)}')
    print(f'Invalid Entrez IDs: {invalid_entrez_count}')

    print('Fetching MyGene annotations...')
    record_by_entrez_id = query_mygene_records(
        entrez_ids=entrez_ids,
        fields=args.fields,
        mygene_batch_size=args.mygene_batch_size,
    )

    print('Building biological text prompts...')
    text_by_node_id: Dict[str, str] = {}
    mygene_hit_count = 0
    for row in gene_frame.itertuples(index=False):
        node_id = str(row.node_id)
        node_name = safe_text(row.node_name, default='Unknown')
        entrez_id = None if pd.isna(row.entrez_id) else str(row.entrez_id)
        record = None if entrez_id is None else record_by_entrez_id.get(entrez_id)
        if record is not None and not record.get('notfound'):
            mygene_hit_count += 1
        text_by_node_id[node_id] = build_gene_text(record=record, fallback_name=node_name)

    print(f'MyGene matched nodes: {mygene_hit_count}')
    print('Loading PubMedBERT...')
    tokenizer, model = load_pubmedbert(model_source, device)
    print(f'Encoding on device: {device}')

    embedding_dict, stats = build_gene_embedding_dict(
        gene_frame=gene_frame,
        text_by_node_id=text_by_node_id,
        tokenizer=tokenizer,
        model=model,
        device=device,
        bert_batch_size=args.bert_batch_size,
        max_length=args.max_length,
    )

    save_embeddings(args.output_pkl, embedding_dict)

    print('Saved gene embeddings to:', args.output_pkl)
    print('Summary:')
    print(f"  Total nodes: {stats['total_nodes']}")
    print(f"  Successful embeddings: {stats['success_count']}")
    print(f"  Zero-filled embeddings: {stats['zero_fill_count']}")
    print(f"  Encoding errors: {stats['encode_error_count']}")


if __name__ == '__main__':
    main()

