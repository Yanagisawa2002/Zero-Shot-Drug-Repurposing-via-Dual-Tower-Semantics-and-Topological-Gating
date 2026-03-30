from __future__ import annotations

import copy
import csv
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation_utils import evaluate_model
from src.pair_path_bpr_sampler import build_pair_path_bpr_dataloader
from src.primekg_data_processor import PrimeKGDataProcessor
from src.repurposing_rgcn import RepurposingRGCN
from src.training_utils import compute_bpr_loss


RawTriplet = Tuple[str, str, str]
IdTriplet = Tuple[int, int, int]


def derive_real_ho_triplets(
    edge_csv_path: str | Path,
    max_pairs: int = 256,
    max_paths_per_pair: int = 4,
    max_drugs_per_gene: int = 12,
    max_diseases_per_gene: int = 12,
) -> List[RawTriplet]:
    """从真实 PrimeKG 边中抽取一批 drug-gene-disease 高阶路径。"""

    gene_to_drugs: Dict[str, List[str]] = defaultdict(list)
    gene_to_diseases: Dict[str, List[str]] = defaultdict(list)

    with Path(edge_csv_path).open('r', encoding='utf-8-sig', newline='') as edge_file:
        reader = csv.DictReader(edge_file)
        for row in reader:
            src_type = row['src_type'].strip()
            dst_type = row['dst_type'].strip()
            relation = row['rel'].strip()
            src_id = row['src_id'].strip()
            dst_id = row['dst_id'].strip()

            if src_type == 'drug' and dst_type == 'gene/protein' and relation == 'targets':
                gene_to_drugs[dst_id].append(src_id)
            elif src_type == 'disease' and dst_type == 'gene/protein' and relation == 'disease_protein':
                gene_to_diseases[dst_id].append(src_id)

    pair_to_genes: Dict[Tuple[str, str], List[str]] = {}
    for gene_id, drugs in gene_to_drugs.items():
        diseases = gene_to_diseases.get(gene_id)
        if not diseases:
            continue

        unique_drugs = list(dict.fromkeys(drugs))[:max_drugs_per_gene]
        unique_diseases = list(dict.fromkeys(diseases))[:max_diseases_per_gene]

        for drug_id in unique_drugs:
            for disease_id in unique_diseases:
                pair = (drug_id, disease_id)
                if len(pair_to_genes) >= max_pairs and pair not in pair_to_genes:
                    continue
                if pair not in pair_to_genes:
                    pair_to_genes[pair] = []
                if len(pair_to_genes[pair]) < max_paths_per_pair:
                    pair_to_genes[pair].append(gene_id)

    ho_triplets: List[RawTriplet] = []
    for (drug_id, disease_id), gene_ids in pair_to_genes.items():
        for gene_id in gene_ids:
            ho_triplets.append((drug_id, gene_id, disease_id))

    return ho_triplets


def split_triplets(
    ho_id_paths: Sequence[IdTriplet],
    train_ratio: float = 0.7,
    valid_ratio: float = 0.15,
    seed: int = 0,
) -> Dict[str, List[IdTriplet]]:
    """将 HO 正路径按 triplet 维度拆成 train/valid/test 三个 split。"""

    if not 0.0 < train_ratio < 1.0:
        raise ValueError('`train_ratio` 必须位于 (0, 1) 区间。')
    if not 0.0 < valid_ratio < 1.0:
        raise ValueError('`valid_ratio` 必须位于 (0, 1) 区间。')
    if train_ratio + valid_ratio >= 1.0:
        raise ValueError('`train_ratio + valid_ratio` 必须小于 1。')

    shuffled_paths = list(ho_id_paths)
    rng = random.Random(seed)
    rng.shuffle(shuffled_paths)

    num_paths = len(shuffled_paths)
    train_end = int(num_paths * train_ratio)
    valid_end = train_end + int(num_paths * valid_ratio)

    return {
        'train': shuffled_paths[:train_end],
        'valid': shuffled_paths[train_end:valid_end],
        'test': shuffled_paths[valid_end:],
    }


def move_batch_to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    """将 pair-level batch 中的所有张量移动到指定设备。"""

    return {key: value.to(device) for key, value in batch.items()}


def run_sanity_checks() -> None:
    seed = 0
    random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device:', device)
    if device.type == 'cuda':
        print('CUDA Device:', torch.cuda.get_device_name(device))

    print('\n[Prepare] Deriving real HO triplets from PrimeKG edges...')
    ho_triplets = derive_real_ho_triplets(
        edge_csv_path='data/PrimeKG/edges.csv',
        max_pairs=256,
        max_paths_per_pair=4,
        max_drugs_per_gene=12,
        max_diseases_per_gene=12,
    )
    print('Derived raw HO triplets:', len(ho_triplets))

    processor = PrimeKGDataProcessor(
        node_csv_path='data/PrimeKG/nodes.csv',
        edge_csv_path='data/PrimeKG/edges.csv',
    )
    processor.build_entity_mappings()
    ho_id_paths = processor.convert_ho_triplets_to_ids(ho_triplets)
    print('Mapped HO triplets:', len(ho_id_paths))

    split_dict = split_triplets(ho_id_paths, train_ratio=0.7, valid_ratio=0.15, seed=seed)
    for split_name, split_paths in split_dict.items():
        print(f'  {split_name}: {len(split_paths)} triplets')

    print('\n[Prepare] Building full-graph HeteroData with train HO positives...')
    build_start = time.time()
    data_cpu = processor.build_heterodata(split_dict['train'])
    print('HeteroData build time:', round(time.time() - build_start, 2), 'sec')
    print('Train ho_pos_paths shape:', tuple(data_cpu.ho_pos_paths.shape))

    train_loader = build_pair_path_bpr_dataloader(
        data=data_cpu,
        batch_size=32,
        shuffle=False,
        num_workers=0,
        negative_strategy='mixed',
    )
    real_batch_cpu = next(iter(train_loader))
    data_gpu = copy.deepcopy(data_cpu).to(device)
    real_batch_gpu = move_batch_to_device(real_batch_cpu, device)

    print('\n[Stage 1] Single real batch end-to-end sanity check')
    torch.cuda.reset_peak_memory_stats(device) if device.type == 'cuda' else None
    model_stage1 = RepurposingRGCN(
        data=data_cpu,
        hidden_dim=32,
        out_dim=32,
        scorer_hidden_dim=32,
        scorer_output_hidden_dim=32,
        dropout=0.0,
    ).to(device)
    optimizer_stage1 = torch.optim.Adam(model_stage1.parameters(), lr=1e-3)

    stage1_start = time.time()
    optimizer_stage1.zero_grad(set_to_none=True)
    pos_scores, neg_scores = model_stage1(
        x_dict=None,
        edge_index_dict=data_gpu.edge_index_dict,
        pos_pair_ids=real_batch_gpu['pos_pair_ids'],
        pos_paths=real_batch_gpu['pos_paths'],
        pos_attention_mask=real_batch_gpu['pos_attention_mask'],
        neg_pair_ids=real_batch_gpu['neg_pair_ids'],
        neg_paths=real_batch_gpu['neg_paths'],
        neg_attention_mask=real_batch_gpu['neg_attention_mask'],
    )
    loss = compute_bpr_loss(pos_scores=pos_scores, neg_scores=neg_scores)
    loss.backward()
    optimizer_stage1.step()
    stage1_elapsed = time.time() - stage1_start

    print('  pos_pair_ids shape:', tuple(real_batch_cpu['pos_pair_ids'].shape))
    print('  pos_paths shape:', tuple(real_batch_cpu['pos_paths'].shape))
    print('  pos_attention_mask true count:', int(real_batch_cpu['pos_attention_mask'].sum().item()))
    print('  neg_paths shape:', tuple(real_batch_cpu['neg_paths'].shape))
    print('  neg_attention_mask true count:', int(real_batch_cpu['neg_attention_mask'].sum().item()))
    print('  loss:', round(float(loss.item()), 6))
    print('  pos score mean:', round(float(pos_scores.mean().item()), 6))
    print('  neg score mean:', round(float(neg_scores.mean().item()), 6))
    print('  step time:', round(stage1_elapsed, 3), 'sec')
    if device.type == 'cuda':
        print('  peak GPU memory:', round(torch.cuda.max_memory_allocated(device) / (1024**2), 2), 'MB')

    print('\n[Stage 2] Overfitting a fixed micro-batch')
    torch.cuda.reset_peak_memory_stats(device) if device.type == 'cuda' else None
    model_stage2 = RepurposingRGCN(
        data=data_cpu,
        hidden_dim=32,
        out_dim=32,
        scorer_hidden_dim=32,
        scorer_output_hidden_dim=32,
        dropout=0.0,
    ).to(device)
    optimizer_stage2 = torch.optim.Adam(model_stage2.parameters(), lr=1e-2)

    overfit_history = []
    overfit_start = time.time()
    for epoch in range(1, 61):
        model_stage2.train()
        optimizer_stage2.zero_grad(set_to_none=True)
        pos_scores, neg_scores = model_stage2(
            x_dict=None,
            edge_index_dict=data_gpu.edge_index_dict,
            pos_pair_ids=real_batch_gpu['pos_pair_ids'],
            pos_paths=real_batch_gpu['pos_paths'],
            pos_attention_mask=real_batch_gpu['pos_attention_mask'],
            neg_pair_ids=real_batch_gpu['neg_pair_ids'],
            neg_paths=real_batch_gpu['neg_paths'],
            neg_attention_mask=real_batch_gpu['neg_attention_mask'],
        )
        loss = compute_bpr_loss(pos_scores=pos_scores, neg_scores=neg_scores)
        loss.backward()
        optimizer_stage2.step()

        avg_margin = float((pos_scores - neg_scores).mean().item())
        overfit_history.append(
            (epoch, float(loss.item()), float(pos_scores.mean().item()), float(neg_scores.mean().item()), avg_margin)
        )
        if epoch in {1, 2, 3, 5, 10, 20, 40, 60}:
            print(
                f'  epoch={epoch:02d} '
                f'loss={loss.item():.6f} '
                f'pos={pos_scores.mean().item():.6f} '
                f'neg={neg_scores.mean().item():.6f} '
                f'margin={avg_margin:.6f}'
            )

    print('  total overfit time:', round(time.time() - overfit_start, 3), 'sec')
    print('  final loss:', round(overfit_history[-1][1], 8))
    print('  final margin:', round(overfit_history[-1][4], 6))
    if device.type == 'cuda':
        print('  peak GPU memory:', round(torch.cuda.max_memory_allocated(device) / (1024**2), 2), 'MB')

    print('\n[Stage 3] Dry-run evaluation on train/valid/test splits')
    model_stage3 = RepurposingRGCN(
        data=data_cpu,
        hidden_dim=32,
        out_dim=32,
        scorer_hidden_dim=32,
        scorer_output_hidden_dim=32,
        dropout=0.0,
    ).to(device)

    for split_name in ['train', 'valid', 'test']:
        split_tensor = torch.tensor(split_dict[split_name], dtype=torch.long)
        print(f'\n  Split: {split_name}  num_paths={split_tensor.size(0)}')
        evaluate_model(
            model=model_stage3,
            data=copy.deepcopy(data_cpu),
            valid_ho_paths=split_tensor,
            batch_size=256,
            verbose=True,
        )


if __name__ == '__main__':
    run_sanity_checks()
