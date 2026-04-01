from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.train_quad_split_ho_probe import build_clean_graph_without_leakage, load_ot_triplets
from scripts.train_split_probe import derive_ho_triplets_for_pair_splits, load_pair_splits
from src.feature_utils import inject_features_to_graph
from src.pair_path_bpr_sampler import PairPathBPRDataset
from src.primekg_data_processor import PrimeKGDataProcessor


Pair = Tuple[int, int]
PathTuple = Tuple[int, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Audit OT positive/negative pair difficulty and dummy-pathway usage.'
    )
    parser.add_argument('--processed-path', type=Path, default=Path('data/PrimeKG/processed/primekg_indication_cross_disease.pt'))
    parser.add_argument('--nodes-csv', type=Path, default=Path('data/PrimeKG/nodes.csv'))
    parser.add_argument('--edges-csv', type=Path, default=Path('data/PrimeKG/edges.csv'))
    parser.add_argument('--feature-dir', type=Path, default=Path('outputs/pubmedbert_hybrid_features'))
    parser.add_argument('--ot-novel-csv', type=Path, default=Path('outputs/ot_random_external_profile/novel_ood_triplets.csv'))
    parser.add_argument('--graph-surgery-mode', type=str, default='strict', choices=['strict', 'direct_only'])
    parser.add_argument('--negative-strategy', type=str, default='random', choices=['random', 'cross_drug', 'cross_disease', 'mixed'])
    parser.add_argument('--sample-count', type=int, default=10)
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()


def build_id_to_name_map(processor: PrimeKGDataProcessor) -> Dict[int, str]:
    return {int(global_id): record.name for global_id, record in processor.id2entity.items()}


def get_pair_display(pair: Pair, id_to_name: Dict[int, str]) -> str:
    drug_name = id_to_name.get(int(pair[0]), f'<unknown:{pair[0]}>')
    disease_name = id_to_name.get(int(pair[1]), f'<unknown:{pair[1]}>')
    return f'Drug: [{drug_name}] | Disease: [{disease_name}]'


def sample_indices(total: int, sample_count: int, seed: int) -> List[int]:
    if total <= 0:
        return []
    rng = random.Random(seed)
    all_indices = list(range(total))
    rng.shuffle(all_indices)
    return all_indices[: min(sample_count, total)]


def iter_all_negative_paths(dataset: PairPathBPRDataset) -> Iterable[torch.Tensor]:
    for pos_pair in dataset.positive_pairs:
        neg_pair = dataset._sample_negative_pair(pos_pair=pos_pair)
        yield dataset.topology_path_bank.get(
            neg_pair,
            torch.empty((0, dataset.path_len), dtype=torch.long),
        )


def compute_dummy_ratio(path_tensors: Iterable[torch.Tensor], dummy_id: int, pathway_index: int) -> Tuple[int, int, float]:
    total_paths = 0
    dummy_paths = 0
    for path_tensor in path_tensors:
        if path_tensor.numel() == 0:
            continue
        total_paths += int(path_tensor.size(0))
        dummy_paths += int((path_tensor[:, pathway_index] == int(dummy_id)).sum().item())
    ratio = float(dummy_paths / total_paths) if total_paths > 0 else 0.0
    return dummy_paths, total_paths, ratio


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    split_mode, pair_splits = load_pair_splits(args.processed_path)
    if split_mode != 'cross_disease':
        print(f'[WARN] Expected cross_disease split, got: {split_mode}')

    processor = PrimeKGDataProcessor(node_csv_path=args.nodes_csv, edge_csv_path=args.edges_csv)
    processor.build_entity_mappings()
    split_triplets = derive_ho_triplets_for_pair_splits(
        processor=processor,
        edge_csv_path=args.edges_csv,
        pair_splits=pair_splits,
    )
    ot_triplets = load_ot_triplets(args.ot_novel_csv, processor.global_entity2id)

    full_data = processor.build_heterodata(ho_id_paths=split_triplets['train'], add_inverse_edges=False)
    inject_features_to_graph(data=full_data, feature_dir=args.feature_dir)

    heldout_triplets = split_triplets['valid'] + split_triplets['test'] + ot_triplets
    clean_data, total_removed_edges, leakage_edge_summary = build_clean_graph_without_leakage(
        data=full_data,
        heldout_triplets=heldout_triplets,
        split_mode=split_mode,
        pair_splits=pair_splits,
        graph_surgery_mode=args.graph_surgery_mode,
    )

    ot_tensor = torch.tensor(ot_triplets, dtype=torch.long)
    negative_disease_pool = sorted({int(path[-1]) for path in ot_tensor.tolist()})
    dataset = PairPathBPRDataset(
        data=clean_data,
        positive_paths=ot_tensor,
        negative_strategy=args.negative_strategy,
        use_pathway_quads=True,
        negative_disease_pool=negative_disease_pool,
    )

    id_to_name = build_id_to_name_map(processor)
    sampled_indices = sample_indices(total=len(dataset), sample_count=args.sample_count, seed=args.seed)

    print('=== OT Negative Audit ===')
    print(f'split_mode: {split_mode}')
    print(f'graph_surgery_mode: {args.graph_surgery_mode}')
    print(f'negative_strategy: {args.negative_strategy}')
    print(f'ot_positive_paths: {int(ot_tensor.size(0))}')
    print(f'ot_positive_pairs(dataset size): {len(dataset)}')
    print(f'negative_disease_pool_size: {len(negative_disease_pool)}')
    print(f'total_removed_edges: {total_removed_edges}')
    if leakage_edge_summary:
        print(f'leakage_edge_types: {sorted(leakage_edge_summary.keys())}')
    print()

    print('--- OT Positive Pairs (Label=1) ---')
    for rank, dataset_index in enumerate(sampled_indices, start=1):
        sample = dataset[dataset_index]
        pos_pair = tuple(int(x) for x in sample['pos_pair_ids'].tolist())
        print(f'{rank}. {get_pair_display(pos_pair, id_to_name)}')
    print()

    print(f'--- OT Negative Pairs (Label=0) | Strategy={args.negative_strategy} ---')
    for rank, dataset_index in enumerate(sampled_indices, start=1):
        sample = dataset[dataset_index]
        neg_pair = tuple(int(x) for x in sample['neg_pair_ids'].tolist())
        print(f'{rank}. {get_pair_display(neg_pair, id_to_name)}')
    print()

    positive_dummy_paths, positive_total_paths, positive_dummy_ratio = compute_dummy_ratio(
        path_tensors=dataset.positive_pair_to_paths.values(),
        dummy_id=dataset.pathway_dummy_global_id,
        pathway_index=2,
    )
    negative_dummy_paths, negative_total_paths, negative_dummy_ratio = compute_dummy_ratio(
        path_tensors=iter_all_negative_paths(dataset),
        dummy_id=dataset.pathway_dummy_global_id,
        pathway_index=2,
    )

    print('--- Dummy Pathway Ratio Audit ---')
    print(
        f'OT positives: dummy_paths={positive_dummy_paths} / total_paths={positive_total_paths} '
        f'({positive_dummy_ratio:.2%})'
    )
    print(
        f'OT negatives: dummy_paths={negative_dummy_paths} / total_paths={negative_total_paths} '
        f'({negative_dummy_ratio:.2%})'
    )


if __name__ == '__main__':
    main()
