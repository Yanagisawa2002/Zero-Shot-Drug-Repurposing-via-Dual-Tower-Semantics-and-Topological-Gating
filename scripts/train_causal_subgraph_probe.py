from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Mapping, Tuple

import pandas as pd
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader
from torch_geometric.data import HeteroData

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.train_split_probe import derive_ho_triplets_for_pair_splits, load_pair_splits
from src.causal_subgraph_model import CausalRepurposingNet, calc_sparsity_loss
from src.evaluation_utils import _compute_pairwise_ranking_metrics, _move_batch_to_device
from src.graph_surgery import collect_holdout_pairs_from_pair_splits, remove_direct_leakage_edges
from src.pair_path_bpr_sampler import build_pair_path_bpr_dataloader
from src.primekg_data_processor import PrimeKGDataProcessor


ResultDict = Dict[str, Dict[str, float]]
IdTriplet = Tuple[int, int, int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Train an independent causal-subgraph drug repositioning model on PrimeKG splits.'
    )
    parser.add_argument('--processed-path', type=Path, required=True)
    parser.add_argument('--output-json', type=Path, required=True)
    parser.add_argument('--checkpoint-path', type=Path, required=True)
    parser.add_argument('--nodes-csv', type=Path, default=Path('data/PrimeKG/nodes.csv'))
    parser.add_argument('--edges-csv', type=Path, default=Path('data/PrimeKG/edges.csv'))
    parser.add_argument('--drug-morgan-fingerprints-path', type=Path, default=Path('drug_morgan_fingerprints.pkl'))
    parser.add_argument('--disease-text-embeddings-path', type=Path, default=Path('thick_disease_text_embeddings.pkl'))
    parser.add_argument('--ot-novel-csv', type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--hidden-dim', type=int, default=128)
    parser.add_argument('--out-dim', type=int, default=128)
    parser.add_argument('--predictor-num-layers', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--generator-temperature', type=float, default=1.0)
    parser.add_argument('--generator-relation-emb-dim', type=int, default=64)
    parser.add_argument('--generator-hidden-dim', type=int, default=128)
    parser.add_argument('--sparsity-beta', type=float, default=0.01)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=1e-5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--graph-surgery-mode', type=str, default='direct_only', choices=['direct_only'])
    parser.add_argument('--eval-epochs', type=str, default='1,10,20,30,40,50,60')
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


def build_clean_graph(
    data: HeteroData,
    pair_splits: Mapping[str, set[Tuple[int, int]]],
) -> Tuple[HeteroData, int, Dict[str, Dict[str, int]]]:
    holdout_pairs = collect_holdout_pairs_from_pair_splits(pair_splits=pair_splits)
    clean_data = remove_direct_leakage_edges(
        data=data,
        holdout_pairs=holdout_pairs,
    )

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


def infer_model_device(model: torch.nn.Module) -> torch.device:
    return next(model.parameters()).device


def train_epoch_causal(
    model: CausalRepurposingNet,
    full_graph_data: HeteroData,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    sparsity_beta: float,
) -> Dict[str, float]:
    model.train()
    device = infer_model_device(model)
    graph_data = copy.deepcopy(full_graph_data).to(device)
    edge_index_dict = graph_data.edge_index_dict

    total_loss = 0.0
    total_pair_loss = 0.0
    total_sparsity_loss = 0.0
    total_pos_score = 0.0
    total_neg_score = 0.0
    total_examples = 0

    for batch in train_loader:
        optimizer.zero_grad(set_to_none=True)
        tensor_batch = _move_batch_to_device(batch=batch, device=device)

        node_embs_dict, edge_mask_dict = model.encode_graph(edge_index_dict=edge_index_dict, device=device)
        pos_scores = model.score_pairs(node_embs_dict=node_embs_dict, pair_ids=tensor_batch['pos_pair_ids'])
        neg_scores = model.score_pairs(node_embs_dict=node_embs_dict, pair_ids=tensor_batch['neg_pair_ids'])

        logits = torch.cat([pos_scores, neg_scores], dim=0)
        labels = torch.cat([
            torch.ones_like(pos_scores),
            torch.zeros_like(neg_scores),
        ], dim=0)
        pair_loss = F.binary_cross_entropy_with_logits(logits, labels)
        sparse_loss = calc_sparsity_loss(edge_mask_dict).to(device=logits.device, dtype=logits.dtype)
        loss = pair_loss + float(sparsity_beta) * sparse_loss
        loss.backward()
        optimizer.step()

        batch_size = int(tensor_batch['pos_pair_ids'].size(0))
        total_loss += float(loss.detach().item()) * batch_size
        total_pair_loss += float(pair_loss.detach().item()) * batch_size
        total_sparsity_loss += float(sparse_loss.detach().item()) * batch_size
        total_pos_score += float(pos_scores.detach().sum().item())
        total_neg_score += float(neg_scores.detach().sum().item())
        total_examples += batch_size

    if total_examples == 0:
        raise ValueError('Training dataloader produced zero examples.')

    return {
        'loss': total_loss / total_examples,
        'pair_loss': total_pair_loss / total_examples,
        'sparsity_loss': total_sparsity_loss / total_examples,
        'avg_pos_score': total_pos_score / total_examples,
        'avg_neg_score': total_neg_score / total_examples,
        'num_examples': float(total_examples),
    }


def evaluate_causal_model(
    model: CausalRepurposingNet,
    data: HeteroData,
    eval_triplets: Tensor,
    batch_size: int,
    use_pathway_quads: bool = True,
) -> ResultDict:
    if eval_triplets.dim() != 2 or eval_triplets.size(1) != 3:
        raise ValueError('`eval_triplets` must have shape [N, 3].')

    dataloaders = {}
    for negative_strategy in ('random', 'cross_drug', 'cross_disease'):
        dataloaders[negative_strategy] = build_pair_path_bpr_dataloader(
            data=data,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            positive_paths=eval_triplets.detach().cpu().to(torch.long),
            known_positive_pairs=eval_triplets.detach().cpu().to(torch.long),
            negative_strategy=negative_strategy,
            use_pathway_quads=False,
        )

    was_training = model.training
    model.eval()
    device = infer_model_device(model)
    graph_data = copy.deepcopy(data).to(device)
    try:
        with torch.no_grad():
            node_embs_dict, edge_mask_dict = model.encode_graph(
                edge_index_dict=graph_data.edge_index_dict,
                device=device,
            )
            mask_values = [mask.reshape(-1) for mask in edge_mask_dict.values() if mask.numel() > 0]
            avg_edge_mask = float(torch.cat(mask_values, dim=0).mean().item()) if mask_values else 0.0

            results: ResultDict = {}
            for setting_name, dataloader in dataloaders.items():
                pos_chunks = []
                neg_chunks = []
                for batch in dataloader:
                    tensor_batch = _move_batch_to_device(batch=batch, device=device)
                    pos_scores = model.score_pairs(node_embs_dict=node_embs_dict, pair_ids=tensor_batch['pos_pair_ids'])
                    neg_scores = model.score_pairs(node_embs_dict=node_embs_dict, pair_ids=tensor_batch['neg_pair_ids'])
                    pos_chunks.append(pos_scores.detach().cpu())
                    neg_chunks.append(neg_scores.detach().cpu())

                metrics = _compute_pairwise_ranking_metrics(
                    pos_scores=torch.cat(pos_chunks, dim=0),
                    neg_scores=torch.cat(neg_chunks, dim=0),
                )
                metrics['avg_edge_mask'] = avg_edge_mask
                results[setting_name] = metrics
    finally:
        if was_training:
            model.train()

    return results


def save_checkpoint(
    checkpoint_path: Path,
    model: CausalRepurposingNet,
    epoch: int,
    split_mode: str,
    valid_eval: ResultDict,
    test_eval: ResultDict,
    args: argparse.Namespace,
) -> None:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        'epoch': int(epoch),
        'split_mode': split_mode,
        'model_config': {
            'hidden_dim': int(args.hidden_dim),
            'out_dim': int(args.out_dim),
            'predictor_num_layers': int(args.predictor_num_layers),
            'dropout': float(args.dropout),
            'generator_temperature': float(args.generator_temperature),
            'generator_relation_emb_dim': int(args.generator_relation_emb_dim),
            'generator_hidden_dim': int(args.generator_hidden_dim),
            'drug_morgan_fingerprints_path': str(args.drug_morgan_fingerprints_path),
            'disease_text_embeddings_path': str(args.disease_text_embeddings_path),
            'nodes_csv_path': str(args.nodes_csv),
            'sparsity_beta': float(args.sparsity_beta),
        },
        'model_state_dict': {k: v.detach().cpu() for k, v in model.state_dict().items()},
        'valid_eval': valid_eval,
        'test_eval': test_eval,
    }
    torch.save(payload, checkpoint_path)


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device:', device)

    eval_epochs = {
        int(item.strip())
        for item in str(args.eval_epochs).split(',')
        if item.strip()
    }
    eval_epochs.add(int(args.epochs))

    split_mode, pair_splits = load_pair_splits(args.processed_path)
    processor = PrimeKGDataProcessor(node_csv_path=args.nodes_csv, edge_csv_path=args.edges_csv)
    processor.build_entity_mappings()
    split_triplets = derive_ho_triplets_for_pair_splits(
        processor=processor,
        edge_csv_path=args.edges_csv,
        pair_splits=pair_splits,
    )
    ot_triplets = load_ot_triplets(
        ot_novel_csv=args.ot_novel_csv,
        global_entity2id=processor.global_entity2id,
    )

    data = processor.build_heterodata(ho_id_paths=split_triplets['train'], add_inverse_edges=False)
    clean_data, total_removed_edges, leakage_edge_summary = build_clean_graph(
        data=data,
        pair_splits=pair_splits,
    )

    train_loader = build_pair_path_bpr_dataloader(
        data=clean_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        negative_strategy='mixed',
        use_pathway_quads=False,
    )

    model = CausalRepurposingNet(
        data=clean_data,
        hidden_dim=args.hidden_dim,
        out_dim=args.out_dim,
        predictor_num_layers=args.predictor_num_layers,
        dropout=args.dropout,
        generator_temperature=args.generator_temperature,
        generator_relation_emb_dim=args.generator_relation_emb_dim,
        generator_hidden_dim=args.generator_hidden_dim,
        disease_text_embeddings=args.disease_text_embeddings_path,
        drug_morgan_fingerprints=args.drug_morgan_fingerprints_path,
        nodes_csv_path=args.nodes_csv,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    valid_tensor = torch.tensor(split_triplets['valid'], dtype=torch.long)
    test_tensor = torch.tensor(split_triplets['test'], dtype=torch.long)
    ot_tensor = torch.tensor(ot_triplets, dtype=torch.long)

    history = []
    best_epoch = 0
    best_valid_auroc = float('-inf')
    best_valid_eval = None
    best_test_eval = None
    best_ot_eval = None
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_epoch_causal(
            model=model,
            full_graph_data=clean_data,
            train_loader=train_loader,
            optimizer=optimizer,
            sparsity_beta=args.sparsity_beta,
        )
        item = {'epoch': epoch, 'train': train_metrics}

        if epoch in eval_epochs:
            valid_eval = evaluate_causal_model(
                model=model,
                data=clean_data,
                eval_triplets=valid_tensor,
                batch_size=args.batch_size,
                use_pathway_quads=False,
            )
            test_eval = evaluate_causal_model(
                model=model,
                data=clean_data,
                eval_triplets=test_tensor,
                batch_size=args.batch_size,
                use_pathway_quads=False,
            )
            ot_eval = None
            if ot_tensor.numel() > 0:
                ot_eval = evaluate_causal_model(
                    model=model,
                    data=clean_data,
                    eval_triplets=ot_tensor,
                    batch_size=args.batch_size,
                    use_pathway_quads=False,
                )

            item['valid_eval'] = valid_eval
            item['test_eval'] = test_eval
            if ot_eval is not None:
                item['ot_eval'] = ot_eval

            current_valid_auroc = float(valid_eval[split_mode]['auroc'])
            if current_valid_auroc > best_valid_auroc:
                best_valid_auroc = current_valid_auroc
                best_epoch = epoch
                best_valid_eval = valid_eval
                best_test_eval = test_eval
                best_ot_eval = ot_eval
                save_checkpoint(
                    checkpoint_path=args.checkpoint_path,
                    model=model,
                    epoch=epoch,
                    split_mode=split_mode,
                    valid_eval=valid_eval,
                    test_eval=test_eval,
                    args=args,
                )

            print(
                f"epoch={epoch:02d} "
                f"valid_{split_mode}_auc={valid_eval[split_mode]['auroc']:.4f} "
                f"test_random_auc={test_eval['random']['auroc']:.4f} "
                f"test_cross_drug_auc={test_eval['cross_drug']['auroc']:.4f} "
                f"test_cross_disease_auc={test_eval['cross_disease']['auroc']:.4f} "
                f"avg_edge_mask={test_eval[split_mode]['avg_edge_mask']:.4f}"
            )
            if ot_eval is not None:
                print(
                    f"OT metrics: random_auc={ot_eval['random']['auroc']:.4f} "
                    f"cross_drug_auc={ot_eval['cross_drug']['auroc']:.4f} "
                    f"cross_disease_auc={ot_eval['cross_disease']['auroc']:.4f}"
                )
        history.append(item)

    payload = {
        'config': {
            'processed_path': str(args.processed_path),
            'checkpoint_path': str(args.checkpoint_path),
            'nodes_csv': str(args.nodes_csv),
            'edges_csv': str(args.edges_csv),
            'drug_morgan_fingerprints_path': str(args.drug_morgan_fingerprints_path),
            'disease_text_embeddings_path': str(args.disease_text_embeddings_path),
            'ot_novel_csv': str(args.ot_novel_csv),
            'epochs': int(args.epochs),
            'batch_size': int(args.batch_size),
            'hidden_dim': int(args.hidden_dim),
            'out_dim': int(args.out_dim),
            'predictor_num_layers': int(args.predictor_num_layers),
            'dropout': float(args.dropout),
            'generator_temperature': float(args.generator_temperature),
            'generator_relation_emb_dim': int(args.generator_relation_emb_dim),
            'generator_hidden_dim': int(args.generator_hidden_dim),
            'sparsity_beta': float(args.sparsity_beta),
            'split_mode': split_mode,
            'graph_surgery_mode': str(args.graph_surgery_mode),
            'seed': int(args.seed),
        },
        'best_epoch': int(best_epoch),
        'best_valid_auroc': float(best_valid_auroc),
        'best_valid_eval': best_valid_eval,
        'best_test_eval': best_test_eval,
        'ot_eval': best_ot_eval,
        'ho_eval': None,
        'total_removed_leakage_edges': int(total_removed_edges),
        'leakage_edge_summary': leakage_edge_summary,
        'total_time_sec': float(time.time() - start_time),
        'history': history,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2), encoding='utf-8')
    print(f'Saved results to: {args.output_json}')


if __name__ == '__main__':
    main()
