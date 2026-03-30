from __future__ import annotations

import argparse
import math
import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from tqdm import tqdm


DEFAULT_INPUT = Path('thick_disease_text_embeddings.pkl')
DEFAULT_OUTPUT = Path('thick_disease_text_embeddings_whitened.pkl')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Whiten disease text embeddings with PCA whitening + L2 normalization.'
    )
    parser.add_argument('--input-pkl', type=Path, default=DEFAULT_INPUT)
    parser.add_argument('--output-pkl', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--n-components', type=int, default=768)
    parser.add_argument('--similarity-batch-size', type=int, default=1024)
    parser.add_argument('--similarity-device', type=str, default='auto', choices=['auto', 'cpu', 'cuda'])
    parser.add_argument('--random-state', type=int, default=42)
    return parser.parse_args()


def load_embedding_dict(input_pkl: Path) -> Dict[str, torch.Tensor]:
    if not input_pkl.exists():
        raise FileNotFoundError(f'Input pickle does not exist: {input_pkl}')
    with input_pkl.open('rb') as f:
        obj = pickle.load(f)
    if not isinstance(obj, dict):
        raise TypeError(f'Expected dict-like pickle, got {type(obj)}')

    out: Dict[str, torch.Tensor] = {}
    expected_dim = None
    for key, value in obj.items():
        tensor = torch.as_tensor(value, dtype=torch.float32).view(-1)
        if expected_dim is None:
            expected_dim = int(tensor.numel())
        if tensor.numel() != expected_dim:
            raise ValueError(
                f'Embedding dimension mismatch for key={key}: expected {expected_dim}, got {tensor.numel()}'
            )
        out[str(key)] = tensor.clone()
    if not out:
        raise ValueError('Input embedding dictionary is empty.')
    return out


def stack_embeddings(embedding_dict: Dict[str, torch.Tensor]) -> Tuple[List[str], np.ndarray]:
    keys = list(embedding_dict.keys())
    matrix = torch.stack([embedding_dict[key] for key in keys], dim=0).cpu().numpy().astype(np.float32, copy=False)
    return keys, matrix


def resolve_similarity_device(mode: str) -> torch.device:
    if mode == 'cpu':
        return torch.device('cpu')
    if mode == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA requested for similarity stats, but no GPU is available.')
        return torch.device('cuda')
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def compute_rowwise_max_cosine_stats(
    matrix: np.ndarray,
    batch_size: int,
    device: torch.device,
    description: str,
) -> Dict[str, float]:
    if matrix.ndim != 2:
        raise ValueError(f'Expected 2D matrix, got shape={matrix.shape}')
    num_rows = matrix.shape[0]
    if num_rows < 2:
        raise ValueError('Need at least two embeddings to compute non-self cosine similarity.')

    tensor = torch.from_numpy(matrix).to(device=device, dtype=torch.float32)
    tensor = torch.nn.functional.normalize(tensor, p=2, dim=1)
    max_values: List[torch.Tensor] = []

    num_batches = math.ceil(num_rows / batch_size)
    progress = tqdm(range(0, num_rows, batch_size), total=num_batches, desc=description, unit='batch')
    for start in progress:
        end = min(start + batch_size, num_rows)
        batch = tensor[start:end]
        sims = batch @ tensor.T
        row_indices = torch.arange(end - start, device=device)
        sims[row_indices, start + row_indices] = -1.0
        batch_max = sims.max(dim=1).values.detach().cpu()
        max_values.append(batch_max)

    max_tensor = torch.cat(max_values, dim=0).to(torch.float32)
    return {
        'mean': float(max_tensor.mean().item()),
        'variance': float(max_tensor.var(unbiased=False).item()),
        'std': float(max_tensor.std(unbiased=False).item()),
        'min': float(max_tensor.min().item()),
        'max': float(max_tensor.max().item()),
    }


def whiten_and_normalize(matrix: np.ndarray, n_components: int, random_state: int) -> Tuple[np.ndarray, PCA]:
    if matrix.ndim != 2:
        raise ValueError(f'Expected 2D matrix, got shape={matrix.shape}')
    num_samples, num_features = matrix.shape
    resolved_n_components = min(int(n_components), num_samples, num_features)
    if resolved_n_components <= 0:
        raise ValueError(f'Invalid n_components={resolved_n_components}')

    print(f'Running PCA whitening: n_samples={num_samples}, n_features={num_features}, n_components={resolved_n_components}')
    pca = PCA(
        n_components=resolved_n_components,
        whiten=True,
        svd_solver='auto',
        random_state=random_state,
    )
    whitened = pca.fit_transform(matrix)
    whitened = normalize(whitened, norm='l2', axis=1)
    whitened = whitened.astype(np.float32, copy=False)
    return whitened, pca


def rebuild_embedding_dict(keys: List[str], matrix: np.ndarray) -> Dict[str, torch.Tensor]:
    if len(keys) != matrix.shape[0]:
        raise ValueError('Key count does not match matrix rows.')
    return {
        key: torch.from_numpy(matrix[idx]).clone()
        for idx, key in enumerate(keys)
    }


def main() -> None:
    args = parse_args()
    print(f'Loading disease embeddings from: {args.input_pkl}')
    embedding_dict = load_embedding_dict(args.input_pkl)
    keys, matrix = stack_embeddings(embedding_dict)
    print(f'Loaded {len(keys)} disease embeddings with dimension {matrix.shape[1]}.')

    similarity_device = resolve_similarity_device(args.similarity_device)
    print(f'Using similarity device: {similarity_device}')

    print('\n[1/3] Computing original nearest-neighbor cosine statistics...')
    original_stats = compute_rowwise_max_cosine_stats(
        matrix=matrix,
        batch_size=args.similarity_batch_size,
        device=similarity_device,
        description='Original max cosine',
    )
    print(
        'Original max cosine stats: '
        f"mean={original_stats['mean']:.4f}, var={original_stats['variance']:.6f}, "
        f"std={original_stats['std']:.4f}, min={original_stats['min']:.4f}, max={original_stats['max']:.4f}"
    )

    print('\n[2/3] Whitening + L2 normalization...')
    whitened_matrix, pca = whiten_and_normalize(
        matrix=matrix,
        n_components=args.n_components,
        random_state=args.random_state,
    )
    explained = float(np.sum(pca.explained_variance_ratio_))
    print(f'Whitening finished. Total explained variance ratio kept: {explained:.6f}')

    print('\n[3/3] Computing whitened nearest-neighbor cosine statistics...')
    whitened_stats = compute_rowwise_max_cosine_stats(
        matrix=whitened_matrix,
        batch_size=args.similarity_batch_size,
        device=similarity_device,
        description='Whitened max cosine',
    )
    print(
        'Whitened max cosine stats: '
        f"mean={whitened_stats['mean']:.4f}, var={whitened_stats['variance']:.6f}, "
        f"std={whitened_stats['std']:.4f}, min={whitened_stats['min']:.4f}, max={whitened_stats['max']:.4f}"
    )

    whitened_dict = rebuild_embedding_dict(keys, whitened_matrix)
    args.output_pkl.parent.mkdir(parents=True, exist_ok=True)
    with args.output_pkl.open('wb') as f:
        pickle.dump(whitened_dict, f)
    print(f'\nSaved whitened disease embeddings to: {args.output_pkl}')


if __name__ == '__main__':
    main()


