from __future__ import annotations

import argparse
import math
import pickle
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from tqdm import tqdm


DEFAULT_INPUT = Path('thick_disease_text_embeddings.pkl')
DEFAULT_OUTPUT_DIR = Path('.')
DEFAULT_K_VALUES = (1, 2, 3, 5, 10)
HEALTHY_MIN = 0.4
HEALTHY_MAX = 0.6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Apply All-but-the-top (ABTT) post-processing to disease text embeddings.'
    )
    parser.add_argument('--input-pkl', type=Path, default=DEFAULT_INPUT)
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument('--k-values', type=int, nargs='+', default=list(DEFAULT_K_VALUES))
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


def format_stats(stats: Dict[str, float]) -> str:
    return (
        f"mean={stats['mean']:.4f}, std={stats['std']:.4f}, "
        f"min={stats['min']:.4f}, max={stats['max']:.4f}"
    )


def fit_pca_on_centered_matrix(centered_matrix: np.ndarray, random_state: int) -> PCA:
    print(
        'Running PCA on mean-centered matrix: '
        f'n_samples={centered_matrix.shape[0]}, n_features={centered_matrix.shape[1]}, '
        f'n_components={centered_matrix.shape[1]}'
    )
    pca = PCA(
        n_components=centered_matrix.shape[1],
        whiten=False,
        svd_solver='auto',
        random_state=random_state,
    )
    pca.fit(centered_matrix)
    return pca


def apply_abtt(centered_matrix: np.ndarray, pca: PCA, top_k: int) -> np.ndarray:
    transformed = pca.transform(centered_matrix)
    transformed[:, :top_k] = 0.0
    reconstructed = pca.inverse_transform(transformed)
    reconstructed = normalize(reconstructed, norm='l2', axis=1)
    return reconstructed.astype(np.float32, copy=False)


def rebuild_embedding_dict(keys: List[str], matrix: np.ndarray) -> Dict[str, torch.Tensor]:
    if len(keys) != matrix.shape[0]:
        raise ValueError('Key count does not match matrix rows.')
    return {
        key: torch.from_numpy(matrix[idx]).clone()
        for idx, key in enumerate(keys)
    }


def choose_best_k(
    results: Iterable[Tuple[int, Dict[str, float]]],
    healthy_min: float,
    healthy_max: float,
) -> int:
    result_list = list(results)
    in_range = [
        (k, stats) for k, stats in result_list
        if healthy_min <= stats['mean'] <= healthy_max
    ]
    if in_range:
        in_range.sort(key=lambda item: (abs(item[1]['mean'] - 0.5), item[0]))
        return int(in_range[0][0])
    result_list.sort(key=lambda item: (abs(item[1]['mean'] - 0.5), item[0]))
    return int(result_list[0][0])


def main() -> None:
    args = parse_args()
    print(f'Loading disease embeddings from: {args.input_pkl}')
    embedding_dict = load_embedding_dict(args.input_pkl)
    keys, matrix = stack_embeddings(embedding_dict)
    print(f'Loaded {len(keys)} disease embeddings with dimension {matrix.shape[1]}.')

    similarity_device = resolve_similarity_device(args.similarity_device)
    print(f'Using similarity device: {similarity_device}')

    print('\n[1/4] Computing original nearest-neighbor cosine statistics...')
    original_stats = compute_rowwise_max_cosine_stats(
        matrix=matrix,
        batch_size=args.similarity_batch_size,
        device=similarity_device,
        description='Original max cosine',
    )
    print(f'Original max cosine stats: {format_stats(original_stats)}')

    print('\n[2/4] Mean-centering embeddings...')
    mean_vector = matrix.mean(axis=0, keepdims=True)
    centered_matrix = matrix - mean_vector
    print(f'Mean-centering finished. Mean vector L2 norm: {float(np.linalg.norm(mean_vector)):.4f}')

    print('\n[3/4] Fitting PCA on centered features...')
    pca = fit_pca_on_centered_matrix(centered_matrix=centered_matrix, random_state=args.random_state)
    explained = float(np.sum(pca.explained_variance_ratio_))
    print(f'PCA fit finished. Total explained variance ratio kept: {explained:.6f}')

    print('\n[4/4] Evaluating ABTT candidates...')
    evaluated_results: List[Tuple[int, Dict[str, float]]] = []
    transformed_cache: Dict[int, np.ndarray] = {}
    for top_k in args.k_values:
        if top_k <= 0 or top_k >= matrix.shape[1]:
            raise ValueError(f'Invalid K={top_k}; expected 1 <= K < {matrix.shape[1]}')
        print(f'\nEvaluating ABTT with top-K removal: K={top_k}')
        abtt_matrix = apply_abtt(centered_matrix=centered_matrix, pca=pca, top_k=top_k)
        stats = compute_rowwise_max_cosine_stats(
            matrix=abtt_matrix,
            batch_size=args.similarity_batch_size,
            device=similarity_device,
            description=f'ABTT K={top_k}',
        )
        print(f'ABTT K={top_k} max cosine stats: {format_stats(stats)}')
        evaluated_results.append((top_k, stats))
        transformed_cache[top_k] = abtt_matrix

    best_k = choose_best_k(
        results=evaluated_results,
        healthy_min=HEALTHY_MIN,
        healthy_max=HEALTHY_MAX,
    )
    best_stats = dict(evaluated_results)[best_k]
    output_pkl = args.output_dir / f'thick_disease_text_embeddings_abtt_K{best_k}.pkl'

    print('\nSelection summary:')
    if HEALTHY_MIN <= best_stats['mean'] <= HEALTHY_MAX:
        print(
            f'Chosen K={best_k} because its nearest-neighbor mean cosine lies in the healthy range '
            f'[{HEALTHY_MIN:.1f}, {HEALTHY_MAX:.1f}].'
        )
    else:
        print(
            f'No candidate reached the healthy range [{HEALTHY_MIN:.1f}, {HEALTHY_MAX:.1f}]. '
            f'Chosen K={best_k} as the closest-to-0.5 fallback.'
        )
    print(f'Chosen ABTT stats: {format_stats(best_stats)}')

    output_dict = rebuild_embedding_dict(keys, transformed_cache[best_k])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with output_pkl.open('wb') as f:
        pickle.dump(output_dict, f)
    print(f'\nSaved ABTT-processed disease embeddings to: {output_pkl}')


if __name__ == '__main__':
    main()


