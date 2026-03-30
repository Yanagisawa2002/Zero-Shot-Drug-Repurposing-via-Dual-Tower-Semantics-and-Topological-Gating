from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Dict, Iterable, Tuple

import torch
from torch import Tensor


TripletKey = Tuple[str, str, str]
EmbeddingDict = Dict[TripletKey, Tensor]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Quick sanity check for triplet_text_embeddings.pkl.',
    )
    parser.add_argument(
        '--pkl-path',
        type=Path,
        default=Path('triplet_text_embeddings.pkl'),
        help='Path to the serialized triplet embedding pickle file.',
    )
    parser.add_argument(
        '--num-samples',
        type=int,
        default=5,
        help='How many sample triplet keys to print.',
    )
    parser.add_argument(
        '--drug-id',
        type=str,
        default=None,
        help='Optional drug_id for an explicit lookup.',
    )
    parser.add_argument(
        '--disease-id',
        type=str,
        default=None,
        help='Optional disease_id for an explicit lookup.',
    )
    parser.add_argument(
        '--protein-id',
        type=str,
        default=None,
        help='Optional protein_id for an explicit lookup.',
    )
    return parser.parse_args()


def load_embedding_dict(pkl_path: Path) -> EmbeddingDict:
    """????? embedding ??????????????"""
    if not pkl_path.exists():
        raise FileNotFoundError(f'Pickle file does not exist: {pkl_path}')

    with pkl_path.open('rb') as file:
        obj = pickle.load(file)

    if not isinstance(obj, dict):
        raise TypeError(f'Expected a dict, but got: {type(obj).__name__}')
    return obj


def summarize_embedding_dict(embedding_dict: EmbeddingDict) -> None:
    """??????????? embedding ???????"""
    if not embedding_dict:
        raise ValueError('Embedding dictionary is empty.')

    first_key = next(iter(embedding_dict.keys()))
    first_embedding = embedding_dict[first_key]
    if not isinstance(first_embedding, torch.Tensor):
        raise TypeError('Embedding values must be torch.Tensor.')

    norms = torch.stack([
        embedding.float().norm(p=2)
        for embedding in embedding_dict.values()
    ])

    print('Embedding dictionary summary:')
    print(f'  num_triplets = {len(embedding_dict)}')
    print(f'  embedding_dim = {tuple(first_embedding.shape)}')
    print(f'  dtype = {first_embedding.dtype}')
    print(f'  device = {first_embedding.device}')
    print(f'  l2_norm_mean = {norms.mean().item():.6f}')
    print(f'  l2_norm_std = {norms.std(unbiased=False).item():.6f}')
    print(f'  l2_norm_min = {norms.min().item():.6f}')
    print(f'  l2_norm_max = {norms.max().item():.6f}')


def iter_sample_items(embedding_dict: EmbeddingDict, num_samples: int) -> Iterable[Tuple[TripletKey, Tensor]]:
    """??????????????"""
    for index, item in enumerate(embedding_dict.items()):
        if index >= num_samples:
            break
        yield item


def print_sample_items(embedding_dict: EmbeddingDict, num_samples: int) -> None:
    """????? triplet key ?? embedding ?????"""
    print()
    print(f'Showing {min(num_samples, len(embedding_dict))} sample triplets:')
    for key, embedding in iter_sample_items(embedding_dict, num_samples):
        head_values = embedding[:8].tolist()
        print(f'  key = {key}')
        print(f'    shape = {tuple(embedding.shape)}, dtype = {embedding.dtype}, device = {embedding.device}')
        print(f'    l2_norm = {embedding.float().norm(p=2).item():.6f}')
        print(f'    first_8_values = {[round(float(value), 6) for value in head_values]}')


def maybe_lookup_triplet(
    embedding_dict: EmbeddingDict,
    drug_id: str | None,
    disease_id: str | None,
    protein_id: str | None,
) -> None:
    """????????? triplet key?????????????"""
    if not all([drug_id, disease_id, protein_id]):
        return

    key: TripletKey = (str(drug_id), str(disease_id), str(protein_id))
    print()
    print(f'Explicit lookup: {key}')
    if key not in embedding_dict:
        print('  status = NOT_FOUND')
        return

    embedding = embedding_dict[key]
    print('  status = FOUND')
    print(f'  shape = {tuple(embedding.shape)}')
    print(f'  dtype = {embedding.dtype}')
    print(f'  device = {embedding.device}')
    print(f'  l2_norm = {embedding.float().norm(p=2).item():.6f}')
    print(f'  first_8_values = {[round(float(value), 6) for value in embedding[:8].tolist()]}')


def main() -> None:
    args = parse_args()
    embedding_dict = load_embedding_dict(args.pkl_path)
    summarize_embedding_dict(embedding_dict)
    print_sample_items(embedding_dict, num_samples=args.num_samples)
    maybe_lookup_triplet(
        embedding_dict=embedding_dict,
        drug_id=args.drug_id,
        disease_id=args.disease_id,
        protein_id=args.protein_id,
    )


if __name__ == '__main__':
    main()
