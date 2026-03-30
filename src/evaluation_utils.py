from __future__ import annotations

import copy
from typing import Dict, Mapping, Optional

import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import Tensor, nn
from torch.utils.data import DataLoader
from torch_geometric.data import HeteroData

from src.pair_path_bpr_sampler import build_pair_path_bpr_dataloader
from src.repurposing_rgcn import EmbeddingDict, EdgeType, RepurposingRGCN


MetricDict = Dict[str, float]
ResultDict = Dict[str, MetricDict]


def evaluate_model(
    model: RepurposingRGCN,
    data: HeteroData,
    valid_ho_paths: Tensor,
    batch_size: int = 32,
    ho_attr_name: str = 'ho_pos_paths',
    verbose: bool = True,
) -> ResultDict:
    """
    ?? pair-level multi-path DataLoader ????? 1:1 ??????????

    ?????
    1. ?? `valid_ho_paths` ?????? DataLoader??????
       - Random
       - Cross-Drug
       - Cross-Disease
    2. ? `torch.no_grad()` ?????????????????? embedding?
    3. ???? DataLoader???? batch ??/? pair ???? `model.score_batch()`?
    4. ???????????
       - Pairwise Accuracy
       - AUROC
       - AUPRC

    ???
    - `model`: ????? `RepurposingRGCN`?
    - `data`: ?? `HeteroData`?
    - `valid_ho_paths`: ??/??????? HO ?????? `(num_paths, 3)`?
    - `batch_size`: ?? DataLoader ? batch ???
    - `ho_attr_name`: `data` ????????????? `ho_pos_paths`?
    - `verbose`: ???? 9 ????

    ???
    - ????????? `random` / `cross_drug` / `cross_disease` ??????
      `pairwise_accuracy`?`auroc`?`auprc`?
    """

    if valid_ho_paths.dim() != 2 or valid_ho_paths.size(1) != 3:
        raise ValueError(
            '`valid_ho_paths` ?????? `(num_paths, 3)`?'
            f'???? {tuple(valid_ho_paths.shape)}?'
        )
    if valid_ho_paths.size(0) == 0:
        raise ValueError('`valid_ho_paths` ??????????')
    if batch_size <= 0:
        raise ValueError('`batch_size` ???????')

    valid_ho_paths = valid_ho_paths.detach().cpu().to(torch.long).contiguous()
    known_positive_paths = _build_known_positive_paths(
        data=data,
        valid_ho_paths=valid_ho_paths,
        ho_attr_name=ho_attr_name,
    )
    dataloaders = _build_eval_dataloaders(
        data=data,
        valid_ho_paths=valid_ho_paths,
        known_positive_paths=known_positive_paths,
        batch_size=batch_size,
        ho_attr_name=ho_attr_name,
    )

    was_training = model.training
    model.eval()
    device = _infer_model_device(model)
    graph_data = copy.deepcopy(data).to(device)

    try:
        with torch.no_grad():
            x_dict = _extract_x_dict(full_graph_data=graph_data)
            edge_index_dict: Mapping[EdgeType, Tensor] = graph_data.edge_index_dict
            node_embs_dict: EmbeddingDict = model.encode(
                x_dict=x_dict,
                edge_index_dict=edge_index_dict,
            )

            results: ResultDict = {}
            for setting_name, dataloader in dataloaders.items():
                pos_scores, neg_scores = _collect_scores_from_dataloader(
                    model=model,
                    node_embs_dict=node_embs_dict,
                    dataloader=dataloader,
                    device=device,
                )
                results[setting_name] = _compute_pairwise_ranking_metrics(
                    pos_scores=pos_scores,
                    neg_scores=neg_scores,
                )
    finally:
        if was_training:
            model.train()

    if verbose:
        print('Pair-Level Multi-Path Evaluation Results:')
        for setting_name, metrics in results.items():
            print(
                f"  [{setting_name}] "
                f"Accuracy={metrics['pairwise_accuracy']:.4f}, "
                f"AUROC={metrics['auroc']:.4f}, "
                f"AUPRC={metrics['auprc']:.4f}"
            )

    return results


def _build_eval_dataloaders(
    data: HeteroData,
    valid_ho_paths: Tensor,
    known_positive_paths: Tensor,
    batch_size: int,
    ho_attr_name: str,
) -> Dict[str, DataLoader[Dict[str, Tensor]]]:
    """??? 1:1 ??????????? DataLoader?"""

    dataloaders: Dict[str, DataLoader[Dict[str, Tensor]]] = {}
    for negative_strategy in ('random', 'cross_drug', 'cross_disease'):
        dataloaders[negative_strategy] = build_pair_path_bpr_dataloader(
            data=data,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
            drop_last=False,
            ho_attr_name=ho_attr_name,
            positive_paths=valid_ho_paths,
            known_positive_pairs=known_positive_paths,
            negative_strategy=negative_strategy,
        )
    return dataloaders


def _collect_scores_from_dataloader(
    model: RepurposingRGCN,
    node_embs_dict: EmbeddingDict,
    dataloader: DataLoader[Dict[str, Tensor]],
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    """
    ???? pair-level DataLoader?????????????

    ???
    - ????????? batch ?????
    - ???????????? DataLoader ? pair ????
    - ?? `shuffle=False`???????? pair ????????
    """

    pos_score_chunks: list[Tensor] = []
    neg_score_chunks: list[Tensor] = []

    for batch in dataloader:
        tensor_batch = _move_batch_to_device(batch=batch, device=device)

        pos_scores = model.score_batch(
            node_embs_dict=node_embs_dict,
            pair_ids=tensor_batch['pos_pair_ids'],
            paths=tensor_batch['pos_paths'],
            attention_mask=tensor_batch['pos_attention_mask'],
        )
        neg_scores = model.score_batch(
            node_embs_dict=node_embs_dict,
            pair_ids=tensor_batch['neg_pair_ids'],
            paths=tensor_batch['neg_paths'],
            attention_mask=tensor_batch['neg_attention_mask'],
        )

        pos_score_chunks.append(pos_scores.detach().cpu())
        neg_score_chunks.append(neg_scores.detach().cpu())

    if not pos_score_chunks:
        raise ValueError('?? DataLoader ??????????')

    return torch.cat(pos_score_chunks, dim=0), torch.cat(neg_score_chunks, dim=0)


def _move_batch_to_device(batch: Dict[str, Tensor], device: torch.device) -> Dict[str, Tensor]:
    """? pair-level batch ??????????????"""

    required_keys = {
        'pos_pair_ids',
        'pos_paths',
        'pos_attention_mask',
        'neg_pair_ids',
        'neg_paths',
        'neg_attention_mask',
    }
    missing_keys = required_keys.difference(batch.keys())
    if missing_keys:
        raise KeyError(f'?? batch ???????{sorted(missing_keys)}')

    moved_batch: Dict[str, Tensor] = {}
    for key, value in batch.items():
        if not isinstance(value, Tensor):
            raise TypeError(f'batch[{key!r}] ?? Tensor?????? {type(value)}')

        if value.dtype == torch.bool:
            moved_batch[key] = value.to(device=device)
        else:
            moved_batch[key] = value.to(device=device, dtype=value.dtype)
    return moved_batch


def _compute_pairwise_ranking_metrics(pos_scores: Tensor, neg_scores: Tensor) -> MetricDict:
    """???? 1:1 ?????? Accuracy / AUROC / AUPRC?"""

    if pos_scores.shape != neg_scores.shape:
        raise ValueError(
            '`pos_scores` ? `neg_scores` ????????'
            f'???? {tuple(pos_scores.shape)} vs {tuple(neg_scores.shape)}?'
        )

    pairwise_accuracy = (pos_scores > neg_scores).float().mean().item()
    y_score = torch.cat([pos_scores, neg_scores], dim=0).detach().cpu().numpy()
    y_true = torch.cat(
        [
            torch.ones_like(pos_scores, dtype=torch.long),
            torch.zeros_like(neg_scores, dtype=torch.long),
        ],
        dim=0,
    ).detach().cpu().numpy()

    return {
        'pairwise_accuracy': float(pairwise_accuracy),
        'auroc': float(roc_auc_score(y_true, y_score)),
        'auprc': float(average_precision_score(y_true, y_score)),
    }


def _build_known_positive_paths(
    data: HeteroData,
    valid_ho_paths: Tensor,
    ho_attr_name: str,
) -> Tensor:
    """
    ???????????????

    ???
    - ?????????????????????? pair?
    - ?????? `(drug, gene, disease)` ???????? DataLoader
      ??????? pair ????????
    """

    path_tensors = [valid_ho_paths.detach().cpu().to(torch.long).contiguous()]

    if hasattr(data, ho_attr_name):
        train_ho_paths = getattr(data, ho_attr_name)
        if not isinstance(train_ho_paths, Tensor):
            raise TypeError(f'`data.{ho_attr_name}` ??? torch.Tensor?')
        if train_ho_paths.dim() != 2 or train_ho_paths.size(1) != 3:
            raise ValueError(
                f'`data.{ho_attr_name}` ?????? `(num_paths, 3)`?'
                f'???? {tuple(train_ho_paths.shape)}?'
            )
        path_tensors.insert(0, train_ho_paths.detach().cpu().to(torch.long).contiguous())

    return torch.cat(path_tensors, dim=0)


def _extract_x_dict(full_graph_data: HeteroData) -> Optional[Dict[str, Tensor]]:
    """? HeteroData ??? `x_dict`???????? `None`?"""

    x_dict: Dict[str, Tensor] = {}
    for node_type in full_graph_data.node_types:
        if 'x' in full_graph_data[node_type]:
            x_dict[node_type] = full_graph_data[node_type].x
    if not x_dict:
        return None
    return x_dict


def _infer_model_device(model: nn.Module) -> torch.device:
    """???????????"""

    try:
        return next(model.parameters()).device
    except StopIteration as exc:
        raise ValueError('????????????????????') from exc
