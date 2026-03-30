from __future__ import annotations

import copy
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple

import torch
from torch import Tensor
from torch_geometric.data import HeteroData


EdgeType = Tuple[str, str, str]
DEFAULT_DIRECT_LEAKAGE_RELATIONS = frozenset({'indication', 'off-label use', 'contraindication'})
DEFAULT_HOLDOUT_PAIR_SPLIT_NAMES = ('valid', 'test')


def remove_direct_leakage_edges(
    data: HeteroData,
    holdout_pairs: Tensor | Sequence[Sequence[int]],
    target_edge_types: Optional[Sequence[EdgeType]] = None,
) -> HeteroData:
    """
    ??? held-out `(drug, disease)` pair ??????????

    ?????
    - ??????????`drug -> disease` ? indication / off-label / contraindication
      ????????????????????
    - ????????????????? message passing ??????????
    - ?????`drug-gene`?`disease-gene`?`gene-pathway` ???????????
      ????????????????????????????????

    ????????????????????????
    - ??? `drug-disease` ????
    - ????????? gene / protein / pathway ?????
    - ??????????????? hold-out ??????

    ??
    - data: ?? PrimeKG ????
    - holdout_pairs: `(N, 2)` ? `(drug_id, disease_id)` ?? ID ??
    - target_edge_types: ??????????????????
      ?? None???????? `(drug, relation, disease)` / `(disease, relation, drug)`
      ? `relation` ?? `DEFAULT_DIRECT_LEAKAGE_RELATIONS` ???

    ??
    - ???? `HeteroData` ??????
    """

    pair_tensor = _normalize_holdout_pairs(holdout_pairs=holdout_pairs)
    clean_data = copy.deepcopy(data)
    if pair_tensor.numel() == 0:
        return clean_data

    _require_global_id(data=data, node_type='drug')
    _require_global_id(data=data, node_type='disease')

    explicit_edge_types = set(tuple(edge_type) for edge_type in target_edge_types) if target_edge_types is not None else None
    hash_base = _infer_hash_base_for_pairs(data=data, pair_tensor=pair_tensor)
    holdout_pair_codes = torch.unique(_encode_pairs(pair_tensor[:, 0], pair_tensor[:, 1], hash_base))

    for edge_type, edge_index in data.edge_index_dict.items():
        if not _is_direct_leakage_edge_type(edge_type=edge_type, explicit_edge_types=explicit_edge_types):
            continue

        if edge_index.numel() == 0:
            clean_data[edge_type].edge_index = edge_index.clone()
            continue

        src_type, _, dst_type = edge_type
        edge_index_cpu = edge_index.detach().cpu().to(torch.long)
        src_global_ids = data[src_type].global_id.detach().cpu().to(torch.long)[edge_index_cpu[0]]
        dst_global_ids = data[dst_type].global_id.detach().cpu().to(torch.long)[edge_index_cpu[1]]

        # ?????????????? `(drug_id, disease_id)` ???????
        if src_type == 'drug' and dst_type == 'disease':
            edge_pair_codes = _encode_pairs(src_global_ids, dst_global_ids, hash_base)
        elif src_type == 'disease' and dst_type == 'drug':
            edge_pair_codes = _encode_pairs(dst_global_ids, src_global_ids, hash_base)
        else:
            # ???????????? _is_direct_leakage_edge_type ?????
            clean_data[edge_type].edge_index = edge_index.clone()
            continue

        remove_mask = torch.isin(edge_pair_codes, holdout_pair_codes)
        keep_mask = ~remove_mask
        clean_data[edge_type].edge_index = edge_index[:, keep_mask.to(edge_index.device)]

    return clean_data


def collect_holdout_pairs_from_pair_splits(
    pair_splits: Mapping[str, Iterable[Tuple[int, int]]],
    split_names: Sequence[str] = DEFAULT_HOLDOUT_PAIR_SPLIT_NAMES,
) -> Tensor:
    """Collect the full pair-level held-out `(drug, disease)` set from split assets.

    This helper intentionally works at the pair level instead of the HO-covered
    triplet/quad level, so graph surgery can remove every direct shortcut edge
    touching validation/test target pairs.
    """

    holdout_pairs = sorted(
        {
            (int(drug_id), int(disease_id))
            for split_name in split_names
            for drug_id, disease_id in pair_splits.get(split_name, [])
        }
    )
    if not holdout_pairs:
        return torch.empty((0, 2), dtype=torch.long)
    return torch.tensor(holdout_pairs, dtype=torch.long)


def remove_leakage_edges(
    data: HeteroData,
    target_paths: Tensor | Sequence[Sequence[int]],
    isolate_nodes_by_type: Optional[Mapping[str, Tensor | Sequence[int]]] = None,
) -> Dict[EdgeType, Tensor]:
    """
    ?? held-out ????? message passing ???????

    ???????
    - `(drug_id, gene_id, disease_id)` ?????
    - `(drug_id, gene_id, pathway_id, disease_id)` ?????

    ????????
    1. `drug-gene` ? `gene-disease` ????
    2. held-out `(drug, disease)` ????? `drug-disease` ????
    3. ??? pathway?????? `gene-pathway` ??
    4. ????? `isolate_nodes_by_type`??????????????

    ?????? `edge_index_dict`??????? `data`?
    """

    path_tensor = _normalize_target_paths(target_paths=target_paths)
    if path_tensor.numel() == 0:
        return {
            edge_type: edge_index.clone()
            for edge_type, edge_index in data.edge_index_dict.items()
        }

    path_node_types = _infer_path_node_types(data=data, path_tensor=path_tensor)
    for node_type in path_node_types:
        if node_type not in data.node_types:
            raise KeyError(f'HeteroData ???????? `{node_type}`?')
        _require_global_id(data=data, node_type=node_type)

    isolate_node_tensors = _normalize_isolation_targets(isolate_nodes_by_type)
    hash_base = _infer_hash_base(
        data=data,
        path_tensor=path_tensor,
        isolate_node_tensors=isolate_node_tensors,
    )

    drug_node_type = path_node_types[0]
    gene_node_type = path_node_types[1]
    disease_node_type = path_node_types[-1]
    pathway_node_type = (
        path_node_types[2]
        if len(path_node_types) == 4
        else ('pathway' if 'pathway' in data.node_types else None)
    )

    drug_gene_codes = torch.unique(_encode_pairs(path_tensor[:, 0], path_tensor[:, 1], hash_base))
    gene_disease_codes = torch.unique(_encode_pairs(path_tensor[:, 1], path_tensor[:, -1], hash_base))
    drug_disease_codes = torch.unique(_encode_pairs(path_tensor[:, 0], path_tensor[:, -1], hash_base))

    heldout_gene_ids = torch.unique(path_tensor[:, 1])
    gene_pathway_codes: Optional[Tensor]
    if path_tensor.size(1) == 4:
        gene_pathway_codes = torch.unique(_encode_pairs(path_tensor[:, 1], path_tensor[:, 2], hash_base))
    else:
        gene_pathway_codes = None

    clean_edge_index_dict: Dict[EdgeType, Tensor] = {}
    for edge_type, edge_index in data.edge_index_dict.items():
        src_type, _, dst_type = edge_type
        if edge_index.numel() == 0:
            clean_edge_index_dict[edge_type] = edge_index.clone()
            continue

        edge_index_cpu = edge_index.detach().cpu().to(torch.long)
        src_global_ids = data[src_type].global_id.detach().cpu().to(torch.long)[edge_index_cpu[0]]
        dst_global_ids = data[dst_type].global_id.detach().cpu().to(torch.long)[edge_index_cpu[1]]

        remove_mask = torch.zeros(edge_index_cpu.size(1), dtype=torch.bool)

        if _matches_unordered_types(src_type, dst_type, drug_node_type, gene_node_type):
            if src_type == drug_node_type:
                edge_codes = _encode_pairs(src_global_ids, dst_global_ids, hash_base)
            else:
                edge_codes = _encode_pairs(dst_global_ids, src_global_ids, hash_base)
            remove_mask |= torch.isin(edge_codes, drug_gene_codes)

        if _matches_unordered_types(src_type, dst_type, gene_node_type, disease_node_type):
            if src_type == gene_node_type:
                edge_codes = _encode_pairs(src_global_ids, dst_global_ids, hash_base)
            else:
                edge_codes = _encode_pairs(dst_global_ids, src_global_ids, hash_base)
            remove_mask |= torch.isin(edge_codes, gene_disease_codes)

        if _matches_unordered_types(src_type, dst_type, drug_node_type, disease_node_type):
            if src_type == drug_node_type:
                edge_codes = _encode_pairs(src_global_ids, dst_global_ids, hash_base)
            else:
                edge_codes = _encode_pairs(dst_global_ids, src_global_ids, hash_base)
            remove_mask |= torch.isin(edge_codes, drug_disease_codes)

        if (
            pathway_node_type is not None
            and pathway_node_type in data.node_types
            and _matches_unordered_types(src_type, dst_type, gene_node_type, pathway_node_type)
        ):
            if src_type == gene_node_type:
                gene_global_ids = src_global_ids
                pathway_global_ids = dst_global_ids
            else:
                gene_global_ids = dst_global_ids
                pathway_global_ids = src_global_ids

            if gene_pathway_codes is not None:
                edge_codes = _encode_pairs(gene_global_ids, pathway_global_ids, hash_base)
                remove_mask |= torch.isin(edge_codes, gene_pathway_codes)
            else:
                remove_mask |= torch.isin(gene_global_ids, heldout_gene_ids)

        if src_type in isolate_node_tensors:
            remove_mask |= torch.isin(src_global_ids, isolate_node_tensors[src_type])
        if dst_type in isolate_node_tensors:
            remove_mask |= torch.isin(dst_global_ids, isolate_node_tensors[dst_type])

        keep_mask = ~remove_mask
        clean_edge_index_dict[edge_type] = edge_index[:, keep_mask.to(edge_index.device)]

    return clean_edge_index_dict



def build_split_isolation_targets(
    split_mode: str,
    pair_splits: Mapping[str, Iterable[Tuple[int, int]]],
) -> Dict[str, Tensor]:
    """? cross_drug / cross_disease ?? held-out ???????"""

    if split_mode == 'cross_drug':
        heldout_drug_ids = sorted(
            {
                int(drug_id)
                for split_name in ('valid', 'test')
                for drug_id, _ in pair_splits.get(split_name, [])
            }
        )
        if heldout_drug_ids:
            return {'drug': torch.tensor(heldout_drug_ids, dtype=torch.long)}
        return {}

    if split_mode == 'cross_disease':
        heldout_disease_ids = sorted(
            {
                int(disease_id)
                for split_name in ('valid', 'test')
                for _, disease_id in pair_splits.get(split_name, [])
            }
        )
        if heldout_disease_ids:
            return {'disease': torch.tensor(heldout_disease_ids, dtype=torch.long)}
        return {}

    return {}



def _normalize_holdout_pairs(holdout_pairs: Tensor | Sequence[Sequence[int]]) -> Tensor:
    if isinstance(holdout_pairs, Tensor):
        pair_tensor = holdout_pairs.detach().cpu().to(torch.long).contiguous()
    else:
        pair_tensor = torch.tensor(holdout_pairs, dtype=torch.long)

    if pair_tensor.dim() != 2 or pair_tensor.size(1) != 2:
        raise ValueError(
            '`holdout_pairs` ??? `(N, 2)` ? `(drug_id, disease_id)` ???'
            f' ???? {tuple(pair_tensor.shape)}?'
        )
    return pair_tensor



def _normalize_target_paths(target_paths: Tensor | Sequence[Sequence[int]]) -> Tensor:
    if isinstance(target_paths, Tensor):
        path_tensor = target_paths.detach().cpu().to(torch.long).contiguous()
    else:
        path_tensor = torch.tensor(target_paths, dtype=torch.long)

    if path_tensor.dim() != 2 or path_tensor.size(1) not in {3, 4}:
        raise ValueError(
            '`target_paths` ??? `(N, 3)` ? `(N, 4)`?'
            f' ???? {tuple(path_tensor.shape)}?'
        )
    return path_tensor



def _infer_path_node_types(data: HeteroData, path_tensor: Tensor) -> Tuple[str, ...]:
    path_len = int(path_tensor.size(1))
    if path_len == 3:
        path_node_types = tuple(getattr(data, 'ho_path_node_types', ('drug', 'gene/protein', 'disease')))
        if len(path_node_types) != 3:
            raise ValueError(
                '??? `data.ho_path_node_types` ????? 3 ??????'
                f'{path_node_types}?'
            )
        return path_node_types

    quad_node_types = tuple(getattr(data, 'quad_path_node_types', ()))
    if len(quad_node_types) == 4:
        return quad_node_types

    ho_path_node_types = tuple(getattr(data, 'ho_path_node_types', ()))
    if len(ho_path_node_types) == 4:
        return ho_path_node_types

    return ('drug', 'gene/protein', 'pathway', 'disease')



def _normalize_isolation_targets(
    isolate_nodes_by_type: Optional[Mapping[str, Tensor | Sequence[int]]],
) -> Dict[str, Tensor]:
    if not isolate_nodes_by_type:
        return {}

    normalized: Dict[str, Tensor] = {}
    for node_type, node_ids in isolate_nodes_by_type.items():
        if isinstance(node_ids, Tensor):
            tensor = node_ids.detach().cpu().to(torch.long).contiguous()
        else:
            tensor = torch.tensor(list(node_ids), dtype=torch.long)
        if tensor.numel() == 0:
            continue
        normalized[node_type] = torch.unique(tensor)
    return normalized



def _require_global_id(data: HeteroData, node_type: str) -> None:
    if node_type not in data.node_types:
        raise KeyError(f'HeteroData ???????? `{node_type}`?')
    if 'global_id' not in data[node_type]:
        raise KeyError(f'`data[{node_type!r}]` ?? `global_id` ???')



def _infer_hash_base_for_pairs(data: HeteroData, pair_tensor: Tensor) -> int:
    max_global_id = int(pair_tensor.max().item()) if pair_tensor.numel() > 0 else 0
    for node_type in ('drug', 'disease'):
        if node_type not in data.node_types or 'global_id' not in data[node_type]:
            continue
        node_global_ids = data[node_type].global_id.detach().cpu()
        if node_global_ids.numel() == 0:
            continue
        max_global_id = max(max_global_id, int(node_global_ids.max().item()))
    return max_global_id + 1



def _infer_hash_base(
    data: HeteroData,
    path_tensor: Tensor,
    isolate_node_tensors: Mapping[str, Tensor],
) -> int:
    max_global_id = int(path_tensor.max().item()) if path_tensor.numel() > 0 else 0
    for node_type in data.node_types:
        if 'global_id' not in data[node_type]:
            continue
        node_global_ids = data[node_type].global_id.detach().cpu()
        if node_global_ids.numel() == 0:
            continue
        max_global_id = max(max_global_id, int(node_global_ids.max().item()))
    for tensor in isolate_node_tensors.values():
        if tensor.numel() > 0:
            max_global_id = max(max_global_id, int(tensor.max().item()))
    return max_global_id + 1



def _is_direct_leakage_edge_type(
    edge_type: EdgeType,
    explicit_edge_types: Optional[set[EdgeType]],
) -> bool:
    if explicit_edge_types is not None:
        return edge_type in explicit_edge_types

    src_type, relation, dst_type = edge_type
    normalized_relation = _normalize_direct_relation_name(relation)
    if normalized_relation not in DEFAULT_DIRECT_LEAKAGE_RELATIONS:
        return False
    return {src_type, dst_type} == {'drug', 'disease'}



def _normalize_direct_relation_name(relation: str) -> str:
    normalized = relation.strip()
    if normalized.endswith('__reverse__'):
        normalized = normalized[: -len('__reverse__')]
    if normalized.startswith('rev_'):
        normalized = normalized[len('rev_') :]
    if normalized.startswith('reverse_'):
        normalized = normalized[len('reverse_') :]
    return normalized



def _matches_unordered_types(src_type: str, dst_type: str, left_type: str, right_type: str) -> bool:
    return (
        (src_type == left_type and dst_type == right_type)
        or (src_type == right_type and dst_type == left_type)
    )



def _encode_pairs(src_ids: Tensor, dst_ids: Tensor, hash_base: int) -> Tensor:
    return src_ids.to(torch.long) * int(hash_base) + dst_ids.to(torch.long)
