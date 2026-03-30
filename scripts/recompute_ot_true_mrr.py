from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.train_quad_split_ho_probe import build_clean_graph_without_leakage, load_ot_triplets
from scripts.train_split_probe import derive_ho_triplets_for_pair_splits, load_pair_splits
from src.feature_utils import inject_features_to_graph
from src.pair_path_bpr_sampler import PairPathBPRDataset
from src.primekg_data_processor import PrimeKGDataProcessor
from src.repurposing_rgcn import RepurposingRGCN

ROOT = Path(__file__).resolve().parents[1]
MULTISEED_DIR = ROOT / 'outputs/asym_route_dropedge_multiseed_20260327'
OUTPUT_JSON = MULTISEED_DIR / 'ot_metrics_true_mrr.json'
OUTPUT_CSV = MULTISEED_DIR / 'ot_metrics_true_mrr.csv'

SPLITS = ['random', 'cross_drug', 'cross_disease']
SEEDS = [42, 43, 44]

IdPair = Tuple[int, int]
IdTriplet = Tuple[int, int, int]


def extract_x_dict(data):
    x_dict = {}
    for node_type in data.node_types:
        if 'x' in data[node_type]:
            x_dict[node_type] = data[node_type].x
    return x_dict


def infer_device(model: torch.nn.Module) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device('cpu')


def move_batch(batch, device):
    out = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            out[key] = value.to(device)
        else:
            raise TypeError(f'Unexpected batch value type for {key}: {type(value)}')
    return out


def pad_paths(path_tensors: Sequence[torch.Tensor], path_len: int) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size = len(path_tensors)
    max_num_paths = max(max(int(t.size(0)), 1) for t in path_tensors)
    padded = torch.zeros((batch_size, max_num_paths, path_len), dtype=torch.long)
    mask = torch.zeros((batch_size, max_num_paths), dtype=torch.bool)
    for idx, tensor in enumerate(path_tensors):
        if tensor.numel() == 0:
            continue
        n = int(tensor.size(0))
        padded[idx, :n] = tensor
        mask[idx, :n] = True
    return padded, mask


def build_model_and_data(split: str, seed: int):
    result_json_path = MULTISEED_DIR / f'{split}_seed{seed}.json'
    checkpoint_path = MULTISEED_DIR / f'{split}_seed{seed}.pt'
    payload = json.loads(result_json_path.read_text(encoding='utf-8'))
    cfg = payload['config']

    processed_path = ROOT / cfg['processed_path']
    nodes_csv = ROOT / 'data/PrimeKG/nodes.csv'
    edges_csv = ROOT / 'data/PrimeKG/edges.csv'
    feature_dir = ROOT / cfg['feature_dir']
    ot_csv = ROOT / cfg['ot_novel_csv']

    split_mode, pair_splits = load_pair_splits(processed_path)
    processor = PrimeKGDataProcessor(node_csv_path=nodes_csv, edge_csv_path=edges_csv)
    processor.build_entity_mappings()
    split_triplets = derive_ho_triplets_for_pair_splits(
        processor=processor,
        edge_csv_path=edges_csv,
        pair_splits=pair_splits,
    )
    ot_triplets = load_ot_triplets(ot_csv, processor.global_entity2id)
    full_data = processor.build_heterodata(ho_id_paths=split_triplets['train'], add_inverse_edges=False)
    inject_features_to_graph(data=full_data, feature_dir=feature_dir)
    heldout_triplets = split_triplets['valid'] + split_triplets['test'] + ot_triplets
    clean_data, _, _ = build_clean_graph_without_leakage(
        data=full_data,
        heldout_triplets=heldout_triplets,
        split_mode=split_mode,
        pair_splits=pair_splits,
        graph_surgery_mode=cfg.get('graph_surgery_mode', 'direct_only'),
    )

    triplet_path = cfg.get('triplet_text_embeddings_path')
    drug_morgan_path = cfg.get('drug_morgan_fingerprints_path')
    disease_path = cfg.get('disease_text_embeddings_path')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = RepurposingRGCN(
        data=clean_data,
        in_channels=768,
        hidden_channels=int(cfg['hidden_channels']),
        out_dim=int(cfg['out_dim']),
        scorer_hidden_dim=int(cfg['scorer_hidden_dim']),
        dropout=float(cfg['dropout']),
        initial_residual_alpha=float(cfg.get('initial_residual_alpha', 0.2)),
        encoder_type=str(cfg.get('encoder_type', 'rgcn')),
        agg_type=str(cfg.get('agg_type', 'attention')),
        use_pathway_quads=True,
        triplet_text_embeddings_path=None if triplet_path is None else ROOT / triplet_path,
        drug_morgan_fingerprints_path=None if drug_morgan_path is None else ROOT / drug_morgan_path,
        disease_text_embeddings_path=None if disease_path is None else ROOT / disease_path,
        nodes_csv_path=nodes_csv,
        text_distill_alpha=float(cfg.get('text_distill_alpha', 0.0)),
        use_early_external_fusion=True,
        dropedge_p=float(cfg.get('dropedge_p', 0.15)),
    ).to(device)
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, clean_data, split_mode, pair_splits, split_triplets, ot_triplets


def collect_scores(model, node_embs_dict, data, positive_paths, known_positive_pairs, negative_strategy, batch_size=256):
    dataset = PairPathBPRDataset(
        data=data,
        positive_paths=positive_paths,
        known_positive_pairs=known_positive_pairs,
        negative_strategy=negative_strategy,
        use_pathway_quads=True,
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        drop_last=False,
        collate_fn=dataset.collate_fn,
    )
    device = infer_device(model)
    pos_chunks = []
    neg_chunks = []
    with torch.no_grad():
        for batch in loader:
            batch = move_batch(batch, device)
            pos_scores = model.score_batch(
                node_embs_dict=node_embs_dict,
                pair_ids=batch['pos_pair_ids'],
                paths=batch['pos_paths'],
                attention_mask=batch['pos_attention_mask'],
            )
            neg_scores = model.score_batch(
                node_embs_dict=node_embs_dict,
                pair_ids=batch['neg_pair_ids'],
                paths=batch['neg_paths'],
                attention_mask=batch['neg_attention_mask'],
            )
            if isinstance(pos_scores, tuple):
                pos_scores = pos_scores[0]
            if isinstance(neg_scores, tuple):
                neg_scores = neg_scores[0]
            pos_chunks.append(pos_scores.detach().cpu())
            neg_chunks.append(neg_scores.detach().cpu())
    return torch.cat(pos_chunks, dim=0), torch.cat(neg_chunks, dim=0)


def best_f1_threshold(valid_pos: torch.Tensor, valid_neg: torch.Tensor) -> tuple[float, float]:
    y_true = np.concatenate([
        np.ones(valid_pos.numel(), dtype=np.int64),
        np.zeros(valid_neg.numel(), dtype=np.int64),
    ])
    y_prob = np.concatenate([
        torch.sigmoid(valid_pos).numpy(),
        torch.sigmoid(valid_neg).numpy(),
    ])
    candidate_thresholds = np.unique(y_prob)
    if candidate_thresholds.size == 0:
        return 0.5, 0.0
    best_threshold = float(candidate_thresholds[0])
    best_f1 = -1.0
    for threshold in candidate_thresholds:
        y_pred = (y_prob > threshold).astype(np.int64)
        score = f1_score(y_true, y_pred, zero_division=0)
        if score > best_f1:
            best_f1 = float(score)
            best_threshold = float(threshold)
    return best_threshold, best_f1


def compute_metrics(pos_scores: torch.Tensor, neg_scores: torch.Tensor, threshold: float) -> Dict[str, float]:
    y_true = np.concatenate([
        np.ones(pos_scores.numel(), dtype=np.int64),
        np.zeros(neg_scores.numel(), dtype=np.int64),
    ])
    y_score = np.concatenate([pos_scores.numpy(), neg_scores.numpy()])
    y_prob = np.concatenate([
        torch.sigmoid(pos_scores).numpy(),
        torch.sigmoid(neg_scores).numpy(),
    ])
    y_pred = (y_prob > threshold).astype(np.int64)
    return {
        'auroc': float(roc_auc_score(y_true, y_score)),
        'auprc': float(average_precision_score(y_true, y_score)),
        'f1': float(f1_score(y_true, y_pred, zero_division=0)),
        'threshold': float(threshold),
    }


def score_pairs_with_bank(model, node_embs_dict, pair_list: Sequence[IdPair], path_bank: Mapping[IdPair, torch.Tensor], path_len: int, batch_size: int = 1024) -> torch.Tensor:
    device = infer_device(model)
    chunks = []
    with torch.no_grad():
        for start in range(0, len(pair_list), batch_size):
            chunk = pair_list[start:start + batch_size]
            pair_ids = torch.tensor(chunk, dtype=torch.long)
            path_tensors = [path_bank.get(pair, torch.empty((0, path_len), dtype=torch.long)) for pair in chunk]
            paths, attention_mask = pad_paths(path_tensors, path_len)
            pair_ids = pair_ids.to(device)
            paths = paths.to(device)
            attention_mask = attention_mask.to(device)
            scores = model.score_batch(
                node_embs_dict=node_embs_dict,
                pair_ids=pair_ids,
                paths=paths,
                attention_mask=attention_mask,
            )
            if isinstance(scores, tuple):
                scores = scores[0]
            chunks.append(scores.detach().cpu())
    return torch.cat(chunks, dim=0)


def compute_ot_filtered_mrr(model, clean_data, node_embs_dict, ot_triplets: Sequence[IdTriplet], all_known_pairs: Sequence[IdPair], batch_size: int = 1024) -> float:
    ot_tensor = torch.tensor(ot_triplets, dtype=torch.long)
    dataset = PairPathBPRDataset(
        data=clean_data,
        positive_paths=ot_tensor,
        known_positive_pairs=all_known_pairs,
        negative_strategy='random',
        use_pathway_quads=True,
    )
    path_bank = dataset.path_bank
    topology_path_bank = dataset.topology_path_bank
    path_len = dataset.path_len
    all_drugs = [int(x) for x in clean_data[dataset.drug_node_type].global_id.detach().cpu().tolist()]

    disease_to_known_drugs: Dict[int, set[int]] = {}
    for drug_id, disease_id in all_known_pairs:
        disease_to_known_drugs.setdefault(int(disease_id), set()).add(int(drug_id))

    disease_to_ot_drugs: Dict[int, List[int]] = {}
    for drug_id, _, disease_id in ot_triplets:
        disease_to_ot_drugs.setdefault(int(disease_id), []).append(int(drug_id))
    disease_to_ot_drugs = {k: list(dict.fromkeys(v)) for k, v in disease_to_ot_drugs.items()}

    reciprocal_ranks: List[float] = []
    for disease_id, positive_drugs in tqdm(disease_to_ot_drugs.items(), desc='OT filtered 1-vs-all MRR', leave=False):
        candidate_pairs = [(int(drug_id), int(disease_id)) for drug_id in all_drugs]
        candidate_scores = score_pairs_with_bank(
            model=model,
            node_embs_dict=node_embs_dict,
            pair_list=candidate_pairs,
            path_bank=topology_path_bank,
            path_len=path_len,
            batch_size=batch_size,
        )
        score_by_drug = {drug_id: float(score) for drug_id, score in zip(all_drugs, candidate_scores.tolist())}

        positive_pairs = [(int(drug_id), int(disease_id)) for drug_id in positive_drugs]
        positive_scores = score_pairs_with_bank(
            model=model,
            node_embs_dict=node_embs_dict,
            pair_list=positive_pairs,
            path_bank=path_bank,
            path_len=path_len,
            batch_size=min(batch_size, max(len(positive_pairs), 1)),
        )
        target_score_by_drug = {drug_id: float(score) for drug_id, score in zip(positive_drugs, positive_scores.tolist())}

        filtered_other_true_drugs = disease_to_known_drugs.get(int(disease_id), set())
        for target_drug in positive_drugs:
            target_score = target_score_by_drug[int(target_drug)]
            greater = 0
            ties = 0
            for candidate_drug in all_drugs:
                if candidate_drug == int(target_drug):
                    continue
                if candidate_drug in filtered_other_true_drugs:
                    continue
                candidate_score = score_by_drug[int(candidate_drug)]
                if candidate_score > target_score:
                    greater += 1
                elif candidate_score == target_score:
                    ties += 1
            rank = 1.0 + float(greater) + 0.5 * float(ties)
            reciprocal_ranks.append(1.0 / rank)

    return float(np.mean(reciprocal_ranks))


def main():
    rows = []
    for split in SPLITS:
        for seed in SEEDS:
            print(f'=== Recomputing OT metrics for {split} seed={seed} ===')
            model, clean_data, split_mode, pair_splits, split_triplets, ot_triplets = build_model_and_data(split, seed)
            primary_setting = split_mode
            all_known_pairs = sorted(set().union(*pair_splits.values()) | {(int(d), int(c)) for d, _, c in ot_triplets})
            valid_tensor = torch.tensor(split_triplets['valid'], dtype=torch.long)
            ot_tensor = torch.tensor(ot_triplets, dtype=torch.long)

            device = infer_device(model)
            graph_data = clean_data.to(device)
            with torch.no_grad():
                node_embs_dict = model.encode(
                    x_dict=extract_x_dict(graph_data),
                    edge_index_dict=graph_data.edge_index_dict,
                )

            valid_pos, valid_neg = collect_scores(
                model=model,
                node_embs_dict=node_embs_dict,
                data=clean_data,
                positive_paths=valid_tensor,
                known_positive_pairs=sorted(set().union(*pair_splits.values())),
                negative_strategy=primary_setting,
                batch_size=256,
            )
            threshold, valid_best_f1 = best_f1_threshold(valid_pos, valid_neg)

            ot_pos, ot_neg = collect_scores(
                model=model,
                node_embs_dict=node_embs_dict,
                data=clean_data,
                positive_paths=ot_tensor,
                known_positive_pairs=all_known_pairs,
                negative_strategy=primary_setting,
                batch_size=256,
            )
            ot_metrics = compute_metrics(ot_pos, ot_neg, threshold)
            ot_true_mrr = compute_ot_filtered_mrr(
                model=model,
                clean_data=clean_data,
                node_embs_dict=node_embs_dict,
                ot_triplets=ot_triplets,
                all_known_pairs=all_known_pairs,
                batch_size=1024,
            )
            row = {
                'split': split,
                'seed': seed,
                'auroc': ot_metrics['auroc'],
                'auprc': ot_metrics['auprc'],
                'f1': ot_metrics['f1'],
                'threshold': ot_metrics['threshold'],
                'valid_best_f1': valid_best_f1,
                'mrr_filtered_1vAll': ot_true_mrr,
                'num_ot_pairs': len({(int(d), int(c)) for d, _, c in ot_triplets}),
            }
            rows.append(row)
            print(row)

    df = pd.DataFrame(rows).sort_values(['split', 'seed']).reset_index(drop=True)
    summary_rows = []
    for split, group in df.groupby('split', sort=False):
        summary_rows.append({
            'split': split,
            'auroc_mean': group['auroc'].mean(),
            'auroc_std': group['auroc'].std(ddof=1),
            'auprc_mean': group['auprc'].mean(),
            'auprc_std': group['auprc'].std(ddof=1),
            'f1_mean': group['f1'].mean(),
            'f1_std': group['f1'].std(ddof=1),
            'mrr_mean': group['mrr_filtered_1vAll'].mean(),
            'mrr_std': group['mrr_filtered_1vAll'].std(ddof=1),
        })
    summary_df = pd.DataFrame(summary_rows)
    payload = {
        'per_seed': df.to_dict(orient='records'),
        'summary': summary_df.to_dict(orient='records'),
        'notes': {
            'mrr_definition': 'Filtered 1-vs-all drug ranking MRR on OT positives: for each OT disease, rank each target drug against the full drug pool while filtering all other known true drugs from internal splits and OT.',
            'f1_definition': 'Max F1 threshold selected on internal validation probabilities after sigmoid, then applied once on OT probabilities.',
        },
    }
    OUTPUT_JSON.write_text(json.dumps(payload, indent=2), encoding='utf-8')
    df.to_csv(OUTPUT_CSV, index=False)
    print('\nSaved:')
    print(OUTPUT_JSON)
    print(OUTPUT_CSV)
    print('\nSummary:')
    print(summary_df.to_string(index=False))


if __name__ == '__main__':
    main()

