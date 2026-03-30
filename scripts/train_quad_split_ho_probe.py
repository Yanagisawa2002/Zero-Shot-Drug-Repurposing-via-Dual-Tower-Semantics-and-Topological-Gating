from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import torch
from torch import Tensor

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.evaluate_ho_mechanisms import evaluate_ho_mechanisms
from scripts.train_quad_split_ot_probe import evaluate_model_quad
from scripts.train_split_probe import derive_ho_triplets_for_pair_splits, load_pair_splits
from src.feature_utils import inject_features_to_graph
from src.graph_surgery import (
    build_split_isolation_targets,
    collect_holdout_pairs_from_pair_splits,
    remove_direct_leakage_edges,
    remove_leakage_edges,
)
from src.pair_path_bpr_sampler import build_pair_path_bpr_dataloader
from src.primekg_data_processor import PrimeKGDataProcessor
from src.repurposing_rgcn import RepurposingRGCN
from src.training_utils import train_epoch


IdTriplet = Tuple[int, int, int]
MetricDict = Dict[str, float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Train isolated quad-pathway model and evaluate HO mechanism ranking on a split.'
    )
    parser.add_argument('--processed-path', type=Path, required=True)
    parser.add_argument('--output-json', type=Path, required=True)
    parser.add_argument('--checkpoint-path', type=Path, required=True)
    parser.add_argument('--nodes-csv', type=Path, default=Path('data/PrimeKG/nodes.csv'))
    parser.add_argument('--edges-csv', type=Path, default=Path('data/PrimeKG/edges.csv'))
    parser.add_argument('--feature-dir', type=Path, default=Path('outputs/pubmedbert_hybrid_features'))
    parser.add_argument('--triplet-text-embeddings-path', type=Path, default=Path('triplet_text_embeddings.pkl'))
    parser.add_argument('--drug-morgan-fingerprints-path', type=Path, default=Path('drug_morgan_fingerprints.pkl'))
    parser.add_argument('--drug-text-embeddings-path', type=Path, default=None)
    parser.add_argument('--disease-text-embeddings-path', type=Path, default=Path('thick_disease_text_embeddings.pkl'))
    parser.add_argument('--text-distill-alpha', type=float, default=0.2)
    parser.add_argument('--primary-loss-type', type=str, default='bce', choices=['bpr', 'bce'])
    parser.add_argument('--disable-triplet-distill', action='store_true')
    parser.add_argument('--disable-morgan', action='store_true')
    parser.add_argument('--disable-disease-semantic', action='store_true')
    parser.add_argument('--use-early-external-fusion', action='store_true')
    parser.add_argument('--dropedge-p', type=float, default=0.15)
    parser.add_argument('--pretrained-encoder-path', type=Path, default=None)
    parser.add_argument('--ot-novel-csv', type=Path, default=Path('outputs/ot_random_external_profile/novel_ood_triplets.csv'))
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--hidden-channels', type=int, default=128)
    parser.add_argument('--out-dim', type=int, default=128)
    parser.add_argument('--scorer-hidden-dim', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=1e-5)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--initial-residual-alpha', type=float, default=0.2)
    parser.add_argument('--encoder-type', type=str, default='rgcn', choices=['rgcn', 'mlp'])
    parser.add_argument('--agg-type', type=str, default='attention', choices=['attention', 'mean', 'max'])
    parser.add_argument('--max-ho-sampling-attempts', type=int, default=256)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--ablate-gnn', action='store_true')
    parser.add_argument('--path-loss-weight', type=float, default=0.1)
    parser.add_argument(
        '--graph-surgery-mode',
        type=str,
        default='strict',
        choices=['strict', 'direct_only'],
        help='strict: old full mechanism-removal surgery; direct_only: remove only held-out drug-disease shortcut edges.',
    )
    return parser.parse_args()


def load_ot_triplets(
    ot_novel_csv: Path,
    global_entity2id: Dict[str, int],
) -> List[IdTriplet]:
    if not ot_novel_csv.exists():
        return []

    ot_df = pd.read_csv(ot_novel_csv)
    ot_triplets: List[IdTriplet] = []
    for row in ot_df.itertuples(index=False):
        drug_raw = getattr(row, 'primekg_drug_id')
        gene_raw = getattr(row, 'primekg_target_id')
        disease_raw = getattr(row, 'primekg_disease_id')
        if (
            drug_raw not in global_entity2id
            or gene_raw not in global_entity2id
            or disease_raw not in global_entity2id
        ):
            continue
        ot_triplets.append(
            (
                int(global_entity2id[drug_raw]),
                int(global_entity2id[gene_raw]),
                int(global_entity2id[disease_raw]),
            )
        )
    return list(dict.fromkeys(ot_triplets))


def build_clean_graph_without_leakage(
    data,
    heldout_triplets: List[IdTriplet],
    split_mode: str,
    pair_splits: Dict[str, set[Tuple[int, int]]],
    graph_surgery_mode: str,
):
    if graph_surgery_mode == 'direct_only':
        holdout_pairs = collect_holdout_pairs_from_pair_splits(pair_splits=pair_splits)
        clean_data = remove_direct_leakage_edges(
            data=data,
            holdout_pairs=holdout_pairs,
        )
    elif graph_surgery_mode == 'strict':
        heldout_tensor = torch.tensor(heldout_triplets, dtype=torch.long)
        isolate_nodes_by_type = build_split_isolation_targets(split_mode=split_mode, pair_splits=pair_splits)
        clean_edge_index_dict = remove_leakage_edges(
            data=data,
            target_paths=heldout_tensor,
            isolate_nodes_by_type=isolate_nodes_by_type,
        )
        clean_data = copy.deepcopy(data)
        for edge_type, clean_edge_index in clean_edge_index_dict.items():
            clean_data[edge_type].edge_index = clean_edge_index
    else:
        raise ValueError(f'Unsupported graph_surgery_mode: {graph_surgery_mode}')

    total_removed_edges = 0
    leakage_edge_summary: Dict[str, Dict[str, int]] = {}
    for edge_type in data.edge_index_dict.keys():
        original_edge_index = data[edge_type].edge_index
        clean_edge_index = clean_data[edge_type].edge_index
        removed_edges = int(original_edge_index.size(1) - clean_edge_index.size(1))
        total_removed_edges += removed_edges
        if removed_edges > 0:
            leakage_edge_summary['|'.join(edge_type)] = {
                'before': int(original_edge_index.size(1)),
                'after': int(clean_edge_index.size(1)),
                'removed': removed_edges,
            }

    return clean_data, total_removed_edges, leakage_edge_summary


def select_primary_setting(split_mode: str) -> str:
    if split_mode not in {'random', 'cross_drug', 'cross_disease'}:
        raise ValueError(f'Unsupported split_mode: {split_mode}')
    return split_mode


def save_checkpoint(
    checkpoint_path: Path,
    model: RepurposingRGCN,
    epoch: int,
    split_mode: str,
    valid_eval: Dict[str, Dict[str, float]],
    test_eval: Dict[str, Dict[str, float]],
    args: argparse.Namespace,
    effective_triplet_text_embeddings_path: Path | None,
    effective_drug_morgan_fingerprints_path: Path | None,
    effective_drug_text_embeddings_path: Path | None,
    effective_disease_text_embeddings_path: Path | None,
    effective_text_distill_alpha: float,
    path_loss_weight: float,
) -> None:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        'epoch': int(epoch),
        'split_mode': split_mode,
        'model_config': {
            'in_channels': 768,
            'hidden_channels': int(args.hidden_channels),
            'out_dim': int(args.out_dim),
            'scorer_hidden_dim': int(args.scorer_hidden_dim),
            'dropout': float(args.dropout),
            'encoder_type': str(args.encoder_type),
            'agg_type': str(args.agg_type),
            'use_pathway_quads': True,
            'graph_surgery_mode': str(args.graph_surgery_mode),
            'triplet_text_embeddings_path': None if effective_triplet_text_embeddings_path is None else str(effective_triplet_text_embeddings_path),
            'drug_morgan_fingerprints_path': None if effective_drug_morgan_fingerprints_path is None else str(effective_drug_morgan_fingerprints_path),
            'drug_text_embeddings_path': None if effective_drug_text_embeddings_path is None else str(effective_drug_text_embeddings_path),
            'disease_text_embeddings_path': None if effective_disease_text_embeddings_path is None else str(effective_disease_text_embeddings_path),
            'text_distill_alpha': float(effective_text_distill_alpha),
            'path_loss_weight': float(path_loss_weight),
            'primary_loss_type': str(args.primary_loss_type),
            'initial_residual_alpha': float(args.initial_residual_alpha),
            'use_early_external_fusion': bool(args.use_early_external_fusion),
            'dropedge_p': float(args.dropedge_p),
            'ablate_gnn': bool(args.ablate_gnn),
            'pretrained_encoder_path': None if args.pretrained_encoder_path is None else str(args.pretrained_encoder_path),
            'dropedge_p': float(args.dropedge_p),
        },
        'model_state_dict': {k: v.detach().cpu() for k, v in model.state_dict().items()},
        'valid_eval': valid_eval,
        'test_eval': test_eval,
    }
    torch.save(payload, checkpoint_path)


def _strip_module_prefix_from_state_dict(state_dict: Dict[str, Tensor]) -> Dict[str, Tensor]:
    cleaned_state_dict: Dict[str, Tensor] = {}
    for key, value in state_dict.items():
        normalized_key = key[7:] if key.startswith('module.') else key
        cleaned_state_dict[normalized_key] = value
    return cleaned_state_dict


def load_pretrained_encoder_weights(
    model: RepurposingRGCN,
    pretrained_encoder_path: Path,
) -> Dict[str, object]:
    payload = torch.load(pretrained_encoder_path, map_location='cpu')
    if isinstance(payload, dict) and 'model_state_dict' in payload and isinstance(payload['model_state_dict'], dict):
        raw_state_dict = payload['model_state_dict']
    elif isinstance(payload, dict):
        raw_state_dict = payload
    else:
        raise TypeError('Unsupported pretrained checkpoint format.')

    cleaned_state_dict = _strip_module_prefix_from_state_dict(raw_state_dict)
    target_state_dict = model.state_dict()

    compatible_state_dict: Dict[str, Tensor] = {}
    skipped_keys: List[str] = []
    for key, value in cleaned_state_dict.items():
        if key.startswith('scorer.'):
            skipped_keys.append(key)
            continue
        if key not in target_state_dict:
            skipped_keys.append(key)
            continue
        if target_state_dict[key].shape != value.shape:
            skipped_keys.append(key)
            continue
        compatible_state_dict[key] = value

    incompatible = model.load_state_dict(compatible_state_dict, strict=False)
    return {
        'loaded_keys': sorted(compatible_state_dict.keys()),
        'num_loaded_keys': len(compatible_state_dict),
        'skipped_keys': sorted(skipped_keys),
        'missing_after_load': sorted(incompatible.missing_keys),
        'unexpected_after_load': sorted(incompatible.unexpected_keys),
    }


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device:', device)

    split_mode, pair_splits = load_pair_splits(args.processed_path)
    print('split_mode:', split_mode)

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

    train_loader = build_pair_path_bpr_dataloader(
        data=clean_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        negative_strategy='mixed',
        use_pathway_quads=True,
    )

    effective_text_distill_alpha = (
        0.0 if args.disable_triplet_distill or float(args.text_distill_alpha) <= 0.0 else float(args.text_distill_alpha)
    )
    effective_triplet_text_embeddings_path = (
        None if effective_text_distill_alpha <= 0.0 else args.triplet_text_embeddings_path
    )
    effective_drug_morgan_fingerprints_path = None if args.disable_morgan else args.drug_morgan_fingerprints_path
    effective_drug_text_embeddings_path = args.drug_text_embeddings_path
    effective_disease_text_embeddings_path = None if args.disable_disease_semantic else args.disease_text_embeddings_path

    model = RepurposingRGCN(
        data=clean_data,
        in_channels=768,
        hidden_channels=args.hidden_channels,
        out_dim=args.out_dim,
        scorer_hidden_dim=args.scorer_hidden_dim,
        dropout=args.dropout,
        initial_residual_alpha=args.initial_residual_alpha,
        encoder_type=args.encoder_type,
        agg_type=args.agg_type,
        use_pathway_quads=True,
        triplet_text_embeddings_path=effective_triplet_text_embeddings_path,
        drug_morgan_fingerprints_path=effective_drug_morgan_fingerprints_path,
        drug_text_embeddings_path=effective_drug_text_embeddings_path,
        disease_text_embeddings_path=effective_disease_text_embeddings_path,
        nodes_csv_path=args.nodes_csv,
        text_distill_alpha=effective_text_distill_alpha,
        use_early_external_fusion=args.use_early_external_fusion,
        dropedge_p=args.dropedge_p,
        ablate_gnn=args.ablate_gnn,
    ).to(device)
    pretrained_load_summary: Dict[str, object] | None = None
    if args.pretrained_encoder_path is not None:
        pretrained_load_summary = load_pretrained_encoder_weights(
            model=model,
            pretrained_encoder_path=args.pretrained_encoder_path,
        )
        print(
            'Loaded pretrained encoder:',
            f"path={args.pretrained_encoder_path}",
            f"num_loaded_keys={pretrained_load_summary['num_loaded_keys']}",
            f"num_skipped_keys={len(pretrained_load_summary['skipped_keys'])}",
        )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    primary_setting = select_primary_setting(split_mode)
    eval_epochs = {1, 10, 20, 30, 40, 50, 60}
    best_valid_auroc = float('-inf')
    best_epoch = -1
    best_valid_eval: Dict[str, Dict[str, float]] | None = None
    best_test_eval: Dict[str, Dict[str, float]] | None = None
    history: List[Dict[str, object]] = []
    start_time = time.time()

    valid_tensor = torch.tensor(split_triplets['valid'], dtype=torch.long)
    test_tensor = torch.tensor(split_triplets['test'], dtype=torch.long)

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_epoch(
            model=model,
            full_graph_data=clean_data,
            bpr_dataloader=train_loader,
            optimizer=optimizer,
            primary_loss_type=args.primary_loss_type,
            path_loss_weight=args.path_loss_weight,
        )
        record: Dict[str, object] = {'epoch': epoch, 'train': train_metrics}

        if epoch in eval_epochs:
            valid_eval = evaluate_model_quad(
                model=model,
                data=clean_data,
                valid_ho_paths=valid_tensor,
                batch_size=args.batch_size,
            )
            test_eval = evaluate_model_quad(
                model=model,
                data=clean_data,
                valid_ho_paths=test_tensor,
                batch_size=args.batch_size,
            )
            record['valid_eval'] = valid_eval
            record['test_eval'] = test_eval

            target_valid_auroc = float(valid_eval[primary_setting]['auroc'])
            print(
                f"epoch={epoch:02d} "
                f"train_pair_loss={train_metrics['pair_loss']:.4f} "
                f"train_path_loss={train_metrics.get('path_loss', 0.0):.4f} "
                f"valid_{primary_setting}_auc={target_valid_auroc:.4f} "
                f"test_random_auc={test_eval['random']['auroc']:.4f} "
                f"test_cross_drug_auc={test_eval['cross_drug']['auroc']:.4f} "
                f"test_cross_disease_auc={test_eval['cross_disease']['auroc']:.4f}"
            )

            if target_valid_auroc > best_valid_auroc:
                best_valid_auroc = target_valid_auroc
                best_epoch = epoch
                best_valid_eval = valid_eval
                best_test_eval = test_eval
                save_checkpoint(
                    checkpoint_path=args.checkpoint_path,
                    model=model,
                    epoch=epoch,
                    split_mode=split_mode,
                    valid_eval=valid_eval,
                    test_eval=test_eval,
                    args=args,
                    effective_triplet_text_embeddings_path=effective_triplet_text_embeddings_path,
                    effective_drug_morgan_fingerprints_path=effective_drug_morgan_fingerprints_path,
                    effective_drug_text_embeddings_path=effective_drug_text_embeddings_path,
                    effective_disease_text_embeddings_path=effective_disease_text_embeddings_path,
                    effective_text_distill_alpha=effective_text_distill_alpha,
                    path_loss_weight=float(args.path_loss_weight),
                )
        history.append(record)

    if best_epoch < 0:
        raise RuntimeError('??????????????????? HO ???')

    checkpoint = torch.load(args.checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    ho_results = evaluate_ho_mechanisms(
        model=model,
        eval_data=clean_data,
        positive_quads=test_tensor,
        reference_data=full_data,
        batch_size=args.batch_size,
        max_sampling_attempts=args.max_ho_sampling_attempts,
        seed=args.seed,
    )
    ot_tensor = torch.tensor(ot_triplets, dtype=torch.long)
    ot_eval = evaluate_model_quad(
        model=model,
        data=clean_data,
        valid_ho_paths=ot_tensor,
        batch_size=args.batch_size,
    ) if ot_tensor.numel() > 0 else None

    payload = {
        'config': {
            'processed_path': str(args.processed_path),
            'checkpoint_path': str(args.checkpoint_path),
            'feature_dir': str(args.feature_dir),
            'ot_novel_csv': str(args.ot_novel_csv),
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'hidden_channels': args.hidden_channels,
            'out_dim': args.out_dim,
            'scorer_hidden_dim': args.scorer_hidden_dim,
            'dropout': args.dropout,
            'encoder_type': args.encoder_type,
            'agg_type': args.agg_type,
            'split_mode': split_mode,
            'quad_mode': True,
            'use_pathway_quads': True,
            'graph_surgery_mode': args.graph_surgery_mode,
            'initial_residual_alpha': args.initial_residual_alpha,
            'triplet_text_embeddings_path': None if effective_triplet_text_embeddings_path is None else str(effective_triplet_text_embeddings_path),
            'drug_morgan_fingerprints_path': None if effective_drug_morgan_fingerprints_path is None else str(effective_drug_morgan_fingerprints_path),
            'drug_text_embeddings_path': None if effective_drug_text_embeddings_path is None else str(effective_drug_text_embeddings_path),
            'disease_text_embeddings_path': None if effective_disease_text_embeddings_path is None else str(effective_disease_text_embeddings_path),
            'text_distill_alpha': float(effective_text_distill_alpha),
            'path_loss_weight': float(args.path_loss_weight),
            'primary_loss_type': str(args.primary_loss_type),
            'disable_triplet_distill': bool(args.disable_triplet_distill),
            'disable_morgan': bool(args.disable_morgan),
            'disable_disease_semantic': bool(args.disable_disease_semantic),
            'pretrained_encoder_path': None if args.pretrained_encoder_path is None else str(args.pretrained_encoder_path),
            'dropedge_p': float(args.dropedge_p),
        },
        'triplet_summary': {
            split_name: {
                'num_triplets': len(paths),
                'num_pairs': len({(d, c) for d, _, c in paths}),
            }
            for split_name, paths in split_triplets.items()
        },
        'ot_novel_triplets': len(ot_triplets),
        'ot_novel_pairs': len({(d, c) for d, _, c in ot_triplets}),
        'total_removed_leakage_edges': total_removed_edges,
        'leakage_edge_summary': leakage_edge_summary,
        'pretrained_load_summary': pretrained_load_summary,
        'best_epoch': best_epoch,
        'best_valid_auroc': best_valid_auroc,
        'best_valid_eval': best_valid_eval,
        'best_test_eval': best_test_eval,
        'ot_eval': ot_eval,
        'ho_eval': {
            'metrics': ho_results['metrics'],
            'num_real_reference_quads': ho_results['num_real_reference_quads'],
            'num_test_quads': int(ho_results['grouped_paths'].size(0)),
        },
        'total_time_sec': time.time() - start_time,
        'history': history,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2), encoding='utf-8')
    print('Saved results to:', args.output_json)
    if ot_eval is not None:
        print(
            'OT metrics:',
            f"random_auc={ot_eval['random']['auroc']:.4f}",
            f"cross_drug_auc={ot_eval['cross_drug']['auroc']:.4f}",
            f"cross_disease_auc={ot_eval['cross_disease']['auroc']:.4f}",
        )
    print(
        'HO metrics:',
        f"AUPRC={payload['ho_eval']['metrics']['auprc']:.4f}",
        f"Hit@1={payload['ho_eval']['metrics']['hit_at_1']:.4f}",
        f"MRR={payload['ho_eval']['metrics']['mrr']:.4f}",
    )


if __name__ == '__main__':
    main()
