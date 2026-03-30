from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.evaluate_ho_mechanisms import evaluate_ho_mechanisms
from scripts.train_quad_split_ho_probe import build_clean_graph_without_leakage, load_ot_triplets
from scripts.train_split_probe import derive_ho_triplets_for_pair_splits, load_pair_splits
from src.feature_utils import inject_features_to_graph
from src.primekg_data_processor import PrimeKGDataProcessor
from src.repurposing_rgcn import RepurposingRGCN

ROOT = Path(__file__).resolve().parents[1]
MULTISEED_DIR = ROOT / 'outputs/asym_route_dropedge_multiseed_20260327'
OUTPUT_JSON = MULTISEED_DIR / 'ho_metrics_auroc_auprc.json'
OUTPUT_CSV = MULTISEED_DIR / 'ho_metrics_auroc_auprc.csv'

SPLITS = ['random', 'cross_drug', 'cross_disease']
SEEDS = [42, 43, 44]


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
    return model, clean_data, full_data, split_triplets, int(cfg.get('batch_size', 32)), int(cfg.get('max_ho_sampling_attempts', 256)) if 'max_ho_sampling_attempts' in cfg else 256


def compute_ho_metrics(results: Dict[str, object]) -> Dict[str, float]:
    grouped_labels = results['grouped_labels']
    group_scores = results['group_scores']
    assert isinstance(grouped_labels, torch.Tensor)
    assert isinstance(group_scores, torch.Tensor)
    y_true = grouped_labels.reshape(-1).detach().cpu().numpy()
    y_score = group_scores.reshape(-1).detach().cpu().numpy()
    return {
        'auroc': float(roc_auc_score(y_true, y_score)),
        'auprc': float(average_precision_score(y_true, y_score)),
        'num_groups': float(grouped_labels.size(0)),
        'num_candidates': float(grouped_labels.numel()),
    }


def main() -> None:
    rows: List[Dict[str, float | int | str]] = []
    for split in SPLITS:
        for seed in SEEDS:
            print(f'=== Recomputing HO AUROC/AUPRC for {split} seed={seed} ===')
            model, clean_data, full_data, split_triplets, batch_size, max_sampling_attempts = build_model_and_data(split, seed)
            test_tensor = torch.tensor(split_triplets['test'], dtype=torch.long)
            results = evaluate_ho_mechanisms(
                model=model,
                eval_data=clean_data,
                positive_quads=test_tensor,
                reference_data=full_data,
                batch_size=batch_size,
                max_sampling_attempts=max_sampling_attempts,
                seed=seed,
            )
            metrics = compute_ho_metrics(results)
            row = {
                'split': split,
                'seed': seed,
                'auroc': metrics['auroc'],
                'auprc': metrics['auprc'],
                'num_groups': int(metrics['num_groups']),
                'num_candidates': int(metrics['num_candidates']),
            }
            rows.append(row)
            print(row)

    df = pd.DataFrame(rows).sort_values(['split', 'seed']).reset_index(drop=True)
    summary_rows = []
    for split, group in df.groupby('split', sort=False):
        summary_rows.append({
            'split': split,
            'auroc_mean': float(group['auroc'].mean()),
            'auroc_std': float(group['auroc'].std(ddof=1)),
            'auprc_mean': float(group['auprc'].mean()),
            'auprc_std': float(group['auprc'].std(ddof=1)),
        })
    summary_df = pd.DataFrame(summary_rows)

    payload = {
        'per_seed': df.to_dict(orient='records'),
        'summary': summary_df.to_dict(orient='records'),
        'notes': {
            'auroc_definition': 'HO 1:4 full-node corruption AUROC computed by flattening grouped candidate scores and labels over all positive/negative path candidates.',
            'auprc_definition': 'HO 1:4 full-node corruption AUPRC computed by flattening grouped candidate scores and labels over all positive/negative path candidates.',
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

