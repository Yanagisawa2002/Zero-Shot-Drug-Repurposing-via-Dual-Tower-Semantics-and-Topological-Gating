from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation_utils import evaluate_model
from src.feature_utils import inject_features_to_graph
from src.graph_surgery import build_split_isolation_targets, remove_leakage_edges
from src.pair_path_bpr_sampler import build_pair_path_bpr_dataloader
from src.primekg_data_processor import PrimeKGDataProcessor
from src.repurposing_rgcn import RepurposingRGCN
from src.training_utils import train_epoch


IdPair = Tuple[int, int]
IdTriplet = Tuple[int, int, int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Run PrimeKG pair-level probe under a specified split with no-leakage message passing.'
    )
    parser.add_argument(
        '--processed-path',
        type=Path,
        required=True,
        help='Processed split asset (.pt), e.g. random / cross_drug / cross_disease split.',
    )
    parser.add_argument(
        '--nodes-csv',
        type=Path,
        default=Path('data/PrimeKG/nodes.csv'),
        help='PrimeKG nodes.csv path.',
    )
    parser.add_argument(
        '--edges-csv',
        type=Path,
        default=Path('data/PrimeKG/edges.csv'),
        help='PrimeKG edges.csv path.',
    )
    parser.add_argument(
        '--feature-dir',
        type=Path,
        default=None,
        help='Optional feature directory containing per-node-type .pt features.',
    )
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--hidden-channels', type=int, default=32)
    parser.add_argument('--out-dim', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=1e-5)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--eval-every', type=int, default=10)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--output-json', type=Path, required=True)
    return parser.parse_args()


def load_pair_splits(processed_path: Path) -> Tuple[str, Dict[str, set[IdPair]]]:
    obj = torch.load(processed_path, map_location='cpu', weights_only=False)
    if not isinstance(obj, dict):
        raise TypeError(f'Processed split file must be a dict payload, got {type(obj)}')
    split_mode = obj.get('split_mode')
    if split_mode not in {'random', 'cross_drug', 'cross_disease'}:
        raise ValueError(
            f'Unsupported split_mode in {processed_path}: {split_mode}'
        )
    if 'target_pairs' not in obj:
        raise KeyError(f'Processed split file missing `target_pairs`: {processed_path}')

    pair_splits: Dict[str, set[IdPair]] = {}
    for split_name, pair_tensor in obj['target_pairs'].items():
        pair_splits[split_name] = {tuple(map(int, row)) for row in pair_tensor.tolist()}
    return split_mode, pair_splits


def derive_ho_triplets_for_pair_splits(
    processor: PrimeKGDataProcessor,
    edge_csv_path: Path,
    pair_splits: Dict[str, set[IdPair]],
) -> Dict[str, List[IdTriplet]]:
    pair_to_split: Dict[IdPair, str] = {}
    for split_name, pair_set in pair_splits.items():
        for pair in pair_set:
            pair_to_split[pair] = split_name

    gene_to_drugs: Dict[int, List[int]] = defaultdict(list)
    gene_to_diseases: Dict[int, List[int]] = defaultdict(list)
    global_entity2id = processor.global_entity2id

    import csv
    with edge_csv_path.open('r', encoding='utf-8-sig', newline='') as edge_file:
        reader = csv.DictReader(edge_file)
        for row in reader:
            src_type = row['src_type'].strip()
            dst_type = row['dst_type'].strip()
            relation = row['rel'].strip()
            src_id = row['src_id'].strip()
            dst_id = row['dst_id'].strip()

            if src_id not in global_entity2id or dst_id not in global_entity2id:
                continue

            src_gid = global_entity2id[src_id]
            dst_gid = global_entity2id[dst_id]
            if src_type == 'drug' and dst_type == 'gene/protein' and relation == 'targets':
                gene_to_drugs[dst_gid].append(src_gid)
            elif src_type == 'disease' and dst_type == 'gene/protein' and relation == 'disease_protein':
                gene_to_diseases[dst_gid].append(src_gid)

    split_triplets: Dict[str, List[IdTriplet]] = {'train': [], 'valid': [], 'test': []}
    for gene_gid, drug_ids in gene_to_drugs.items():
        disease_ids = gene_to_diseases.get(gene_gid)
        if not disease_ids:
            continue

        unique_drugs = list(dict.fromkeys(drug_ids))
        unique_diseases = list(dict.fromkeys(disease_ids))
        for drug_gid in unique_drugs:
            for disease_gid in unique_diseases:
                pair = (drug_gid, disease_gid)
                split_name = pair_to_split.get(pair)
                if split_name is None:
                    continue
                split_triplets[split_name].append((drug_gid, gene_gid, disease_gid))

    for split_name, triplets in split_triplets.items():
        split_triplets[split_name] = list(dict.fromkeys(triplets))
    return split_triplets


def summarize_triplets(split_triplets: Dict[str, List[IdTriplet]]) -> Dict[str, Dict[str, int]]:
    summary: Dict[str, Dict[str, int]] = {}
    for split_name, triplets in split_triplets.items():
        summary[split_name] = {
            'num_triplets': len(triplets),
            'num_pairs': len({(drug_id, disease_id) for drug_id, _, disease_id in triplets}),
        }
    return summary


def maybe_inject_features(data, feature_dir: Path | None) -> None:
    if feature_dir is None:
        print('Feature mode: disabled (using learned node embeddings).')
        return
    print(f'Feature mode: loading from {feature_dir}')
    inject_features_to_graph(data=data, feature_dir=feature_dir)


def build_clean_graph_without_leakage(
    data,
    heldout_triplets: List[IdTriplet],
    split_mode: str,
    pair_splits: Dict[str, set[IdPair]],
):
    heldout_tensor = torch.tensor(heldout_triplets, dtype=torch.long)
    isolate_nodes_by_type = build_split_isolation_targets(split_mode=split_mode, pair_splits=pair_splits)
    clean_edge_index_dict = remove_leakage_edges(
        data=data,
        target_paths=heldout_tensor,
        isolate_nodes_by_type=isolate_nodes_by_type,
    )

    clean_data = copy.deepcopy(data)
    leakage_summary: Dict[str, Dict[str, int]] = {}
    total_removed_edges = 0
    for edge_type, clean_edge_index in clean_edge_index_dict.items():
        original_edge_index = data[edge_type].edge_index
        clean_data[edge_type].edge_index = clean_edge_index

        removed_edges = int(original_edge_index.size(1) - clean_edge_index.size(1))
        total_removed_edges += removed_edges
        if removed_edges > 0:
            leakage_summary['|'.join(edge_type)] = {
                'before': int(original_edge_index.size(1)),
                'after': int(clean_edge_index.size(1)),
                'removed': removed_edges,
            }

    return clean_data, leakage_summary, total_removed_edges


def evaluate_split_bundle(
    model: RepurposingRGCN,
    data,
    split_triplets: Dict[str, List[IdTriplet]],
    batch_size: int,
) -> Dict[str, Dict[str, Dict[str, float]]]:
    results: Dict[str, Dict[str, Dict[str, float]]] = {}
    for split_name in ('valid', 'test'):
        split_tensor = torch.tensor(split_triplets[split_name], dtype=torch.long)
        results[split_name] = evaluate_model(
            model=model,
            data=data,
            valid_ho_paths=split_tensor,
            batch_size=batch_size,
            verbose=False,
        )
    return results


def print_eval_snapshot(epoch: int, eval_results: Dict[str, Dict[str, Dict[str, float]]]) -> None:
    print(f'[Eval @ epoch {epoch:02d}]')
    for split_name, split_metrics in eval_results.items():
        random_metrics = split_metrics['random']
        cross_drug_metrics = split_metrics['cross_drug']
        cross_disease_metrics = split_metrics['cross_disease']
        print(
            f'  {split_name}: '
            f"random_auc={random_metrics['auroc']:.4f}, "
            f"cross_drug_auc={cross_drug_metrics['auroc']:.4f}, "
            f"cross_disease_auc={cross_disease_metrics['auroc']:.4f}, "
            f"random_acc={random_metrics['pairwise_accuracy']:.4f}"
        )


def run_probe(args: argparse.Namespace) -> None:
    if args.epochs <= 0:
        raise ValueError('`--epochs` must be positive.')
    if args.batch_size <= 0:
        raise ValueError('`--batch-size` must be positive.')

    torch.manual_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device:', device)
    if device.type == 'cuda':
        print('CUDA Device:', torch.cuda.get_device_name(device))
        print('CUDA Memory Allocated (before):', torch.cuda.memory_allocated(device))

    print('\n[Prepare] Loading processed split target pairs...')
    split_mode, pair_splits = load_pair_splits(args.processed_path)
    print('  split_mode:', split_mode)
    for split_name, pair_set in pair_splits.items():
        print(f'  {split_name}: {len(pair_set)} target pairs')

    print('\n[Prepare] Building entity mappings and deriving HO triplets...')
    processor = PrimeKGDataProcessor(
        node_csv_path=args.nodes_csv,
        edge_csv_path=args.edges_csv,
    )
    processor.build_entity_mappings()
    split_triplets = derive_ho_triplets_for_pair_splits(
        processor=processor,
        edge_csv_path=args.edges_csv,
        pair_splits=pair_splits,
    )
    triplet_summary = summarize_triplets(split_triplets)
    for split_name, stats in triplet_summary.items():
        print(
            f"  {split_name}: {stats['num_triplets']} HO triplets / {stats['num_pairs']} unique pairs"
        )

    if triplet_summary['train']['num_triplets'] == 0:
        raise RuntimeError('train split has no HO triplets.')
    if triplet_summary['valid']['num_triplets'] == 0 or triplet_summary['test']['num_triplets'] == 0:
        raise RuntimeError('valid/test split has no HO triplets.')

    print('\n[Prepare] Building HeteroData...')
    data = processor.build_heterodata(split_triplets['train'])
    maybe_inject_features(data=data, feature_dir=args.feature_dir)

    heldout_triplets = split_triplets['valid'] + split_triplets['test']
    clean_data, leakage_summary, total_removed_edges = build_clean_graph_without_leakage(
        data=data,
        heldout_triplets=heldout_triplets,
        split_mode=split_mode,
        pair_splits=pair_splits,
    )
    print('  heldout HO triplets removed from message passing graph:', len(heldout_triplets))
    print('  total removed leakage edges:', total_removed_edges)
    for edge_name, stats in leakage_summary.items():
        print(
            f"    {edge_name}: before={stats['before']}, after={stats['after']}, removed={stats['removed']}"
        )

    train_loader = build_pair_path_bpr_dataloader(
        data=clean_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        negative_strategy='mixed',
    )
    print('  train ho_pos_paths shape:', tuple(clean_data.ho_pos_paths.shape))
    print('  train loader batches:', len(train_loader))

    model = RepurposingRGCN(
        data=clean_data,
        hidden_channels=args.hidden_channels,
        out_dim=args.out_dim,
        scorer_hidden_dim=args.out_dim,
        scorer_output_hidden_dim=args.out_dim,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    history: List[Dict[str, object]] = []
    total_start = time.time()
    print('\n[Train] Starting training...')
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        train_metrics = train_epoch(
            model=model,
            full_graph_data=clean_data,
            bpr_dataloader=train_loader,
            optimizer=optimizer,
        )
        epoch_record: Dict[str, object] = {
            'epoch': epoch,
            'train': train_metrics,
            'epoch_time_sec': time.time() - epoch_start,
        }

        print(
            f"epoch={epoch:02d} "
            f"loss={train_metrics['loss']:.6f} "
            f"pos={train_metrics['avg_pos_score']:.6f} "
            f"neg={train_metrics['avg_neg_score']:.6f} "
            f"time={epoch_record['epoch_time_sec']:.2f}s"
        )

        if epoch == 1 or epoch % args.eval_every == 0 or epoch == args.epochs:
            eval_results = evaluate_split_bundle(
                model=model,
                data=clean_data,
                split_triplets=split_triplets,
                batch_size=args.batch_size,
            )
            epoch_record['eval'] = eval_results
            print_eval_snapshot(epoch=epoch, eval_results=eval_results)

        history.append(epoch_record)

    total_time = time.time() - total_start
    print('\n[Done]')
    print('Total training time (sec):', round(total_time, 2))
    if device.type == 'cuda':
        print('Peak GPU memory (MB):', round(torch.cuda.max_memory_allocated(device) / (1024 ** 2), 2))

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    serialized_config = {
        key: (str(value) if isinstance(value, Path) else value)
        for key, value in vars(args).items()
    }
    output_payload = {
        'split_mode': split_mode,
        'config': serialized_config,
        'triplet_summary': triplet_summary,
        'total_removed_leakage_edges': total_removed_edges,
        'leakage_edge_summary': leakage_summary,
        'total_time_sec': total_time,
        'history': history,
    }
    with args.output_json.open('w', encoding='utf-8') as file:
        json.dump(output_payload, file, ensure_ascii=False, indent=2)
    print('Saved metrics to:', args.output_json)


def main() -> None:
    args = parse_args()
    run_probe(args)


if __name__ == '__main__':
    main()
