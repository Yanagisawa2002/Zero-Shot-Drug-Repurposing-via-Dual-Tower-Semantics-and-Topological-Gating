from __future__ import annotations

from collections.abc import Mapping as MappingABC
from typing import Dict, Mapping, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch_geometric.data import HeteroData

from src.repurposing_rgcn import EmbeddingDict, EdgeType, RepurposingRGCN


def compute_bpr_loss(pos_scores: Tensor, neg_scores: Tensor) -> Tensor:
    """
    ????????? BPR Loss?

    ?????
        L_BPR = - mean(log(sigmoid(pos_scores - neg_scores)))

    ???? `torch.nn.functional.logsigmoid`???????
    `log(sigmoid(x))` ??????????
    """

    if pos_scores.shape != neg_scores.shape:
        raise ValueError(
            "`pos_scores` ? `neg_scores` ??????????"
            f"???? {tuple(pos_scores.shape)} vs {tuple(neg_scores.shape)}?"
        )

    pairwise_margin = pos_scores - neg_scores
    loss = -F.logsigmoid(pairwise_margin).mean()
    return loss


def compute_bce_loss(pos_scores: Tensor, neg_scores: Tensor) -> Tensor:
    """
    ????? BCEWithLogits ????

    ????????? 1????????? 0?
    ???? batch ??????????????
    """

    if pos_scores.shape != neg_scores.shape:
        raise ValueError(
            "`pos_scores` and `neg_scores` must share the same shape for BCE loss: "
            f"{tuple(pos_scores.shape)} vs {tuple(neg_scores.shape)}."
        )

    logits = torch.cat([pos_scores, neg_scores], dim=0)
    labels = torch.cat([torch.ones_like(pos_scores), torch.zeros_like(neg_scores)], dim=0)
    return F.binary_cross_entropy_with_logits(logits, labels)


def train_epoch(
    model: RepurposingRGCN,
    full_graph_data: HeteroData,
    bpr_dataloader: DataLoader,
    optimizer: Optimizer,
    primary_loss_type: str = 'bpr',
    path_loss_weight: float = 0.0,
) -> Dict[str, float]:
    """
    ???? epoch?

    ????? BPR ? BCEWithLogits ????

        total_loss = primary_loss + model.text_distill_alpha * distill_loss

    ?? `distill_loss` ????????????
    """

    model.train()
    device = _infer_model_device(model)
    full_graph_data = full_graph_data.to(device)

    x_dict = _extract_x_dict(full_graph_data=full_graph_data)
    edge_index_dict: Mapping[EdgeType, Tensor] = full_graph_data.edge_index_dict

    if primary_loss_type not in {'bpr', 'bce'}:
        raise ValueError("`primary_loss_type` must be either 'bpr' or 'bce'.")

    total_loss = 0.0
    total_primary_loss = 0.0
    total_distill_loss = 0.0
    total_path_loss = 0.0
    total_pos_score = 0.0
    total_neg_score = 0.0
    total_examples = 0

    for batch in bpr_dataloader:
        optimizer.zero_grad(set_to_none=True)

        node_embs_dict = model.encode(
            x_dict=x_dict,
            edge_index_dict=edge_index_dict,
        )
        pos_scores, neg_scores, distill_loss, path_loss, batch_size = _score_training_batch(
            model=model,
            node_embs_dict=node_embs_dict,
            batch=batch,
            device=device,
        )

        if primary_loss_type == 'bpr':
            primary_loss = compute_bpr_loss(pos_scores=pos_scores, neg_scores=neg_scores)
        else:
            primary_loss = compute_bce_loss(pos_scores=pos_scores, neg_scores=neg_scores)
        total_batch_loss = primary_loss + model.text_distill_alpha * distill_loss + float(path_loss_weight) * path_loss
        total_batch_loss.backward()
        optimizer.step()

        total_loss += float(total_batch_loss.detach().item()) * batch_size
        total_primary_loss += float(primary_loss.detach().item()) * batch_size
        total_distill_loss += float(distill_loss.detach().item()) * batch_size
        total_path_loss += float(path_loss.detach().item()) * batch_size
        total_pos_score += float(pos_scores.detach().sum().item()) * 1.0
        total_neg_score += float(neg_scores.detach().sum().item()) * 1.0
        total_examples += batch_size

    if total_examples == 0:
        raise ValueError("`bpr_dataloader` ????? epoch ?????????")

    metrics = {
        "loss": total_loss / total_examples,
        "pair_loss": total_primary_loss / total_examples,
        "distill_loss": total_distill_loss / total_examples,
        "path_loss": total_path_loss / total_examples,
        "avg_pos_score": total_pos_score / total_examples,
        "avg_neg_score": total_neg_score / total_examples,
        "num_examples": float(total_examples),
        "primary_loss_type": float(0.0 if primary_loss_type == 'bpr' else 1.0),
    }
    if primary_loss_type == 'bpr':
        metrics["bpr_loss"] = metrics["pair_loss"]
        metrics["bce_loss"] = 0.0
    else:
        metrics["bce_loss"] = metrics["pair_loss"]
        metrics["bpr_loss"] = 0.0
    return metrics


def _score_training_batch(
    model: RepurposingRGCN,
    node_embs_dict: EmbeddingDict,
    batch: object,
    device: torch.device,
) -> Tuple[Tensor, Tensor, Tensor, Tensor, int]:
    """
    ?? batch ???????????
        (pos_scores, neg_scores, distill_loss, path_loss, batch_size)
    """

    if isinstance(batch, (tuple, list)) and len(batch) == 2:
        pos_paths, neg_paths = batch
        pos_paths = pos_paths.to(device=device, dtype=torch.long)
        neg_paths = neg_paths.to(device=device, dtype=torch.long)

        pos_scores, pos_distill_loss, pos_path_loss = model.score_paths(
            node_embs_dict=node_embs_dict,
            path_tensor=pos_paths,
            return_distill_loss=True,
            return_path_loss=True,
        )
        neg_scores, neg_distill_loss = model.score_paths(
            node_embs_dict=node_embs_dict,
            path_tensor=neg_paths,
            return_distill_loss=True,
        )
        distill_loss = 0.5 * (pos_distill_loss + neg_distill_loss)
        return pos_scores, neg_scores, distill_loss, pos_path_loss, int(pos_paths.size(0))

    if isinstance(batch, MappingABC):
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
            raise KeyError(f'pair-level batch ???????{sorted(missing_keys)}')

        tensor_batch: Dict[str, Tensor] = {}
        for key, value in batch.items():
            if not isinstance(value, Tensor):
                raise TypeError(f'batch[{key!r}] ?? Tensor?????? {type(value)}')
            if value.dtype == torch.bool:
                tensor_batch[key] = value.to(device=device)
            else:
                tensor_batch[key] = value.to(device=device, dtype=value.dtype)

        pos_scores, pos_distill_loss, pos_path_loss = model.score_batch(
            node_embs_dict=node_embs_dict,
            pair_ids=tensor_batch['pos_pair_ids'],
            paths=tensor_batch['pos_paths'],
            attention_mask=tensor_batch['pos_attention_mask'],
            return_distill_loss=True,
            return_path_loss=True,
        )
        neg_scores, neg_distill_loss = model.score_batch(
            node_embs_dict=node_embs_dict,
            pair_ids=tensor_batch['neg_pair_ids'],
            paths=tensor_batch['neg_paths'],
            attention_mask=tensor_batch['neg_attention_mask'],
            return_distill_loss=True,
        )
        distill_loss = 0.5 * (pos_distill_loss + neg_distill_loss)
        return pos_scores, neg_scores, distill_loss, pos_path_loss, int(tensor_batch['pos_pair_ids'].size(0))

    raise TypeError(
        '???? batch ???`train_epoch` ??? `(pos_paths, neg_paths)` '
        '? pair-level `dict` batch?'
    )


def _extract_x_dict(full_graph_data: HeteroData) -> Optional[Dict[str, Tensor]]:
    """
    ? HeteroData ??????????

    ?????????? `x`???? `None`???????????? embedding?
    """

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
