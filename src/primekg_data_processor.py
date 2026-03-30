from __future__ import annotations

import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import DefaultDict, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from torch import Tensor
from torch_geometric.data import HeteroData


EdgeType = Tuple[str, str, str]
HOPath = Tuple[int, ...]


@dataclass(frozen=True)
class NodeRecord:
    """单个 PrimeKG 节点的结构化记录。"""

    global_id: int
    local_id: int
    raw_id: str
    node_type: str
    name: str
    source: str


@dataclass(frozen=True)
class MissingEntityRecord:
    """记录 HO 路径中无法可靠映射的字符串实体。"""

    path_index: int
    position: int
    expected_type: str
    raw_value: str
    reason: str
    candidate_global_ids: Tuple[int, ...] = ()


class PrimeKGDataProcessor:
    """
    PrimeKG 数据预处理器。

    主要职责：
    1. 读取 PrimeKG 的节点与边；
    2. 为每种节点类型建立“字符串别名 -> 全局唯一 ID”的映射；
    3. 将 HO 字符串三元组转换为 ID 三元组；
    4. 构建 PyG 的 HeteroData，并挂载 HO 正样本路径。
    """

    def __init__(
        self,
        node_csv_path: str | Path,
        edge_csv_path: str | Path,
        default_ho_type_order: Sequence[str] = ("drug", "gene", "disease"),
        lowercase_lookup: bool = True,
    ) -> None:
        self.node_csv_path = Path(node_csv_path)
        self.edge_csv_path = Path(edge_csv_path)
        self.lowercase_lookup = lowercase_lookup

        # 将用户友好的节点类型别名统一到 PrimeKG 的真实类型名。
        self.type_aliases: Dict[str, str] = {
            "drug": "drug",
            "compound": "drug",
            "gene": "gene/protein",
            "protein": "gene/protein",
            "gene/protein": "gene/protein",
            "gene_protein": "gene/protein",
            "disease": "disease",
            "phenotype": "disease",
            "pathway": "pathway",
            "effect/phenotype": "effect/phenotype",
            "molecular_function": "molecular_function",
            "biological_process": "biological_process",
            "cellular_component": "cellular_component",
            "anatomy": "anatomy",
            "exposure": "exposure",
        }

        self.default_ho_type_order: Tuple[str, ...] = tuple(
            self._normalize_node_type(node_type) for node_type in default_ho_type_order
        )

        # 这些属性会在 build_entity_mappings() 中填充。
        self.global_entity2id: Dict[str, int] = {}
        self.entity2id_by_type: Dict[str, Dict[str, int]] = {}
        self.ambiguous_aliases_by_type: Dict[str, Dict[str, Tuple[int, ...]]] = {}
        self.id2entity: Dict[int, NodeRecord] = {}
        self.global2local_by_type: Dict[str, Dict[int, int]] = {}
        self.local2global_by_type: Dict[str, List[int]] = {}
        self.node_records_by_type: Dict[str, List[NodeRecord]] = {}

        # HO 路径转换过程中的日志式记录。
        self.missing_ho_entities: List[MissingEntityRecord] = []
        self.skipped_ho_paths: List[int] = []
        self.last_ho_type_order: Tuple[str, ...] = self.default_ho_type_order

    def build_entity_mappings(self, force_rebuild: bool = False) -> Dict[str, Dict[str, int]]:
        """
        读取 PrimeKG 节点文件，并为每种节点类型建立字符串到全局唯一 ID 的映射。

        说明：
        - `global_entity2id` 使用 PrimeKG 原始 `id` 字段作为键，映射到全局唯一整数 ID；
        - `entity2id_by_type` 使用节点类型分组，支持通过名字、原始 ID 等别名查找；
        - 若同一类型下某个别名对应多个节点，会被记为歧义别名，不进入 `entity2id_by_type`。
        """

        if self.entity2id_by_type and not force_rebuild:
            return self.entity2id_by_type

        self.global_entity2id = {}
        self.entity2id_by_type = {}
        self.ambiguous_aliases_by_type = {}
        self.id2entity = {}
        self.global2local_by_type = defaultdict(dict)
        self.local2global_by_type = defaultdict(list)
        self.node_records_by_type = defaultdict(list)

        alias_bucket_by_type: DefaultDict[str, DefaultDict[str, set[int]]] = defaultdict(
            lambda: defaultdict(set)
        )

        with self.node_csv_path.open("r", encoding="utf-8-sig", newline="") as node_file:
            reader = csv.DictReader(node_file)
            for row in reader:
                raw_id = row["id"].strip()
                node_type = self._normalize_node_type(row["type"])
                name = row["name"].strip()
                source = row["source"].strip()

                if raw_id in self.global_entity2id:
                    raise ValueError(f"PrimeKG 节点 ID 重复，无法建立唯一映射：{raw_id}")

                global_id = len(self.global_entity2id)
                local_id = len(self.local2global_by_type[node_type])

                self.global_entity2id[raw_id] = global_id
                self.global2local_by_type[node_type][global_id] = local_id
                self.local2global_by_type[node_type].append(global_id)

                record = NodeRecord(
                    global_id=global_id,
                    local_id=local_id,
                    raw_id=raw_id,
                    node_type=node_type,
                    name=name,
                    source=source,
                )
                self.id2entity[global_id] = record
                self.node_records_by_type[node_type].append(record)

                for alias in self._iter_entity_aliases(raw_id=raw_id, name=name):
                    alias_bucket_by_type[node_type][alias].add(global_id)

        self.entity2id_by_type = {}
        self.ambiguous_aliases_by_type = {}
        for node_type, alias_bucket in alias_bucket_by_type.items():
            unique_aliases: Dict[str, int] = {}
            ambiguous_aliases: Dict[str, Tuple[int, ...]] = {}
            for alias, candidate_ids in alias_bucket.items():
                sorted_candidate_ids = tuple(sorted(candidate_ids))
                if len(sorted_candidate_ids) == 1:
                    unique_aliases[alias] = sorted_candidate_ids[0]
                else:
                    ambiguous_aliases[alias] = sorted_candidate_ids

            self.entity2id_by_type[node_type] = unique_aliases
            self.ambiguous_aliases_by_type[node_type] = ambiguous_aliases

        return self.entity2id_by_type

    def convert_ho_triplets_to_ids(
        self,
        ho_triplets: Sequence[Sequence[str]],
        ho_type_order: Optional[Sequence[str]] = None,
    ) -> List[HOPath]:
        """
        将 HO 字符串路径转换为全局唯一 ID 路径。

        参数：
        - `ho_triplets`: 形如 `[("drugA", "geneX", "disease1"), ...]` 的字符串路径；
        - `ho_type_order`: 每个位置对应的节点类型，默认使用初始化时的设置。

        返回：
        - 仅包含成功映射路径的 ID 列表；
        - 任何缺失或歧义实体都会被记录到 `self.missing_ho_entities` 中，并跳过整条路径。
        """

        if not self.entity2id_by_type:
            self.build_entity_mappings()

        normalized_type_order = tuple(
            self._normalize_node_type(node_type)
            for node_type in (ho_type_order or self.default_ho_type_order)
        )
        self.last_ho_type_order = normalized_type_order

        self.missing_ho_entities = []
        self.skipped_ho_paths = []

        ho_id_paths: List[HOPath] = []

        for path_index, path in enumerate(ho_triplets):
            if len(path) != len(normalized_type_order):
                raise ValueError(
                    "HO 路径长度与 ho_type_order 不一致："
                    f"path_index={path_index}, path_len={len(path)}, "
                    f"type_order_len={len(normalized_type_order)}"
                )

            current_path_ids: List[int] = []
            path_has_error = False

            for position, (raw_value, expected_type) in enumerate(zip(path, normalized_type_order)):
                lookup_status, global_id, candidate_ids = self._lookup_entity(
                    node_type=expected_type,
                    raw_value=raw_value,
                )

                if global_id is None:
                    self.missing_ho_entities.append(
                        MissingEntityRecord(
                            path_index=path_index,
                            position=position,
                            expected_type=expected_type,
                            raw_value=raw_value,
                            reason=lookup_status,
                            candidate_global_ids=candidate_ids,
                        )
                    )
                    path_has_error = True
                    continue

                current_path_ids.append(global_id)

            if path_has_error:
                self.skipped_ho_paths.append(path_index)
                continue

            ho_id_paths.append(tuple(current_path_ids))

        return ho_id_paths

    def build_heterodata(
        self,
        ho_id_paths: Sequence[Sequence[int]],
        ho_type_order: Optional[Sequence[str]] = None,
        add_inverse_edges: bool = False,
        inverse_suffix: str = "__reverse__",
        ho_attr_name: str = "ho_pos_paths",
    ) -> HeteroData:
        """
        读取 PrimeKG 边文件，构建 PyG 的 HeteroData。

        设计说明：
        - HeteroData 中的 `edge_index` 必须使用“分类型局部索引”；
        - 但用户需要“全局唯一实体 ID”，因此每种节点类型下都额外挂载 `global_id`；
        - HO 路径默认保存为全局 ID 张量，挂在 `data.ho_pos_paths`。
        """

        if not self.entity2id_by_type:
            self.build_entity_mappings()

        normalized_type_order = tuple(
            self._normalize_node_type(node_type)
            for node_type in (ho_type_order or self.last_ho_type_order)
        )

        edge_buffer: DefaultDict[EdgeType, List[List[int]]] = defaultdict(lambda: [[], []])

        with self.edge_csv_path.open("r", encoding="utf-8-sig", newline="") as edge_file:
            reader = csv.DictReader(edge_file)
            for row in reader:
                src_raw_id = row["src_id"].strip()
                dst_raw_id = row["dst_id"].strip()
                src_type = self._normalize_node_type(row["src_type"])
                dst_type = self._normalize_node_type(row["dst_type"])
                relation = row["rel"].strip()

                if src_raw_id not in self.global_entity2id:
                    raise KeyError(f"边中的源节点未在 nodes.csv 中找到：{src_raw_id}")
                if dst_raw_id not in self.global_entity2id:
                    raise KeyError(f"边中的目标节点未在 nodes.csv 中找到：{dst_raw_id}")

                src_global_id = self.global_entity2id[src_raw_id]
                dst_global_id = self.global_entity2id[dst_raw_id]
                src_local_id = self.global2local_by_type[src_type][src_global_id]
                dst_local_id = self.global2local_by_type[dst_type][dst_global_id]

                edge_type = (src_type, relation, dst_type)
                edge_buffer[edge_type][0].append(src_local_id)
                edge_buffer[edge_type][1].append(dst_local_id)

                if add_inverse_edges:
                    inverse_edge_type = (dst_type, f"{relation}{inverse_suffix}", src_type)
                    edge_buffer[inverse_edge_type][0].append(dst_local_id)
                    edge_buffer[inverse_edge_type][1].append(src_local_id)

        data = HeteroData()

        for node_type, global_ids in self.local2global_by_type.items():
            data[node_type].num_nodes = len(global_ids)
            data[node_type].node_id = torch.arange(len(global_ids), dtype=torch.long)
            data[node_type].global_id = torch.tensor(global_ids, dtype=torch.long)

        for edge_type, (src_indices, dst_indices) in edge_buffer.items():
            if src_indices:
                edge_index = torch.tensor([src_indices, dst_indices], dtype=torch.long)
            else:
                edge_index = torch.empty((2, 0), dtype=torch.long)
            data[edge_type].edge_index = edge_index

        ho_tensor = self._build_ho_tensor(ho_id_paths=ho_id_paths, num_columns=len(normalized_type_order))
        setattr(data, ho_attr_name, ho_tensor)

        # 保存 HO 路径每一列对应的节点类型，方便后续训练时解释每一列的语义。
        data.ho_path_node_types = normalized_type_order

        return data

    def _lookup_entity(
        self,
        node_type: str,
        raw_value: str,
    ) -> Tuple[str, Optional[int], Tuple[int, ...]]:
        """
        在指定节点类型下查找字符串实体。

        返回：
        - `status`: "ok" / "missing" / "ambiguous"
        - `global_id`: 成功时返回全局 ID，否则为 None
        - `candidate_ids`: 歧义时返回候选 ID 列表
        """

        normalized_value = self._normalize_text(raw_value)
        if not normalized_value:
            return "missing", None, ()

        typed_mapping = self.entity2id_by_type.get(node_type, {})
        if normalized_value in typed_mapping:
            return "ok", typed_mapping[normalized_value], ()

        ambiguous_mapping = self.ambiguous_aliases_by_type.get(node_type, {})
        if normalized_value in ambiguous_mapping:
            return "ambiguous", None, ambiguous_mapping[normalized_value]

        return "missing", None, ()

    def _iter_entity_aliases(self, raw_id: str, name: str) -> Iterable[str]:
        """
        为单个节点生成一组可用于匹配的字符串别名。

        当前默认包含：
        - PrimeKG 原始 `id`；
        - 去掉类型前缀后的尾部 ID，例如 `drug::DB00180 -> DB00180`；
        - 节点 `name`。
        """

        alias_candidates = {raw_id, name}
        if "::" in raw_id:
            alias_candidates.add(raw_id.split("::", maxsplit=1)[1])

        for alias in alias_candidates:
            normalized_alias = self._normalize_text(alias)
            if normalized_alias:
                yield normalized_alias

    def _normalize_node_type(self, node_type: str) -> str:
        """统一节点类型名称。"""

        normalized_key = node_type.strip().casefold().replace("_", "/")
        if normalized_key in self.type_aliases:
            return self.type_aliases[normalized_key]
        return node_type.strip()

    def _normalize_text(self, value: str) -> str:
        """统一字符串实体的匹配格式。"""

        normalized_value = value.strip()
        if self.lowercase_lookup:
            normalized_value = normalized_value.casefold()
        return normalized_value

    def _build_ho_tensor(self, ho_id_paths: Sequence[Sequence[int]], num_columns: int) -> Tensor:
        """将 HO ID 列表转换为 PyTorch LongTensor。"""

        if not ho_id_paths:
            return torch.empty((0, num_columns), dtype=torch.long)
        return torch.tensor(list(ho_id_paths), dtype=torch.long)
