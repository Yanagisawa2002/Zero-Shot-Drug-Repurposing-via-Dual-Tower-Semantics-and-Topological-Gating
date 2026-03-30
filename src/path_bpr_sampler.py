from __future__ import annotations

from collections import defaultdict
from typing import DefaultDict, List, Sequence, Set, Tuple

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torch_geometric.data import HeteroData


Triplet = Tuple[int, int, int]


class PathBPRSampler(Dataset[Tuple[Tensor, Tensor]]):
    """
    面向 BPR Loss 的路径级采样数据集。

    命名上叫 Sampler，但实现上继承自 `torch.utils.data.Dataset`，
    便于直接交给 `DataLoader` 做 batch 化。

    正样本：
    - 来自 `data.ho_pos_paths`，格式为 `(drug_id, gene_id, disease_id)`。

    负样本：
    - 固定正样本的 `(drug_id, disease_id)` 两端点；
    - 随机替换中间的 `gene_id`；
    - 并确保新三元组不在已知 HO 正样本集合中。

    这种负采样方式比完全随机三元组更安全：
    - 保持药物与疾病上下文不变；
    - 仅扰动潜在机制桥接节点 gene；
    - 更适合 BPR / PU Learning 场景下的排序学习。
    """

    def __init__(
        self,
        data: HeteroData,
        ho_attr_name: str = "ho_pos_paths",
        gene_node_type: str = "gene/protein",
        expected_ho_node_types: Sequence[str] = ("drug", "gene/protein", "disease"),
        max_sampling_attempts: int = 32,
    ) -> None:
        super().__init__()

        if not hasattr(data, ho_attr_name):
            raise AttributeError(f"HeteroData 中不存在属性 `{ho_attr_name}`。")

        ho_pos_paths = getattr(data, ho_attr_name)
        if not isinstance(ho_pos_paths, Tensor):
            raise TypeError(f"`data.{ho_attr_name}` 必须是 torch.Tensor。")
        if ho_pos_paths.dim() != 2 or ho_pos_paths.size(1) != 3:
            raise ValueError(
                f"`data.{ho_attr_name}` 的形状必须为 [num_paths, 3]，"
                f"实际得到 {tuple(ho_pos_paths.shape)}。"
            )

        if gene_node_type not in data.node_types:
            raise KeyError(f"HeteroData 中不存在节点类型 `{gene_node_type}`。")
        if "global_id" not in data[gene_node_type]:
            raise KeyError(
                f"`data['{gene_node_type}']` 中缺少 `global_id`，"
                "无法进行基于全局实体 ID 的安全负采样。"
            )

        if hasattr(data, "ho_path_node_types"):
            actual_ho_node_types = tuple(getattr(data, "ho_path_node_types"))
            expected_types = tuple(expected_ho_node_types)
            if actual_ho_node_types != expected_types:
                raise ValueError(
                    "HO 路径列语义与采样器预期不一致："
                    f"expected={expected_types}, actual={actual_ho_node_types}"
                )

        self.data = data
        self.ho_attr_name = ho_attr_name
        self.gene_node_type = gene_node_type
        self.max_sampling_attempts = max_sampling_attempts

        # 统一转到 CPU/long，避免 DataLoader worker 中出现设备不一致问题。
        self.ho_pos_paths: Tensor = ho_pos_paths.detach().cpu().to(torch.long).contiguous()
        self.all_gene_ids: Tensor = (
            data[gene_node_type].global_id.detach().cpu().to(torch.long).contiguous()
        )

        if self.ho_pos_paths.size(0) == 0:
            raise ValueError("`data.ho_pos_paths` 为空，无法构造 BPR 训练样本。")
        if self.all_gene_ids.numel() == 0:
            raise ValueError(f"`data['{gene_node_type}'].global_id` 为空，无法进行负采样。")

        self.all_gene_ids_list: List[int] = [int(gene_id) for gene_id in self.all_gene_ids.tolist()]
        self.positive_path_list: List[Triplet] = [
            (int(path[0]), int(path[1]), int(path[2])) for path in self.ho_pos_paths.tolist()
        ]
        self.positive_path_set: Set[Triplet] = set(self.positive_path_list)

        # 对每个 `(drug, disease)` 端点对，记录所有已知正例 gene。
        # 后续只要采样到不在该集合中的 gene，即可保证是安全负样本。
        self.positive_genes_by_pair: DefaultDict[Tuple[int, int], Set[int]] = defaultdict(set)
        for drug_id, gene_id, disease_id in self.positive_path_list:
            self.positive_genes_by_pair[(drug_id, disease_id)].add(gene_id)

    def __len__(self) -> int:
        """数据集长度等于 HO 正样本路径数量。"""

        return len(self.positive_path_list)

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        """
        返回一对 `(positive_path, negative_path)`。

        返回张量形状均为 `[3]`，默认 DataLoader 会自动拼成 `[batch_size, 3]`。
        """

        positive_path = self.ho_pos_paths[index]
        negative_path = self._sample_negative_path(positive_path=positive_path)
        return positive_path.clone(), negative_path

    def _sample_negative_path(self, positive_path: Tensor) -> Tensor:
        """
        为单条正样本路径生成一条安全负样本路径。

        采样策略：
        1. 固定 `(drug_id, disease_id)`；
        2. 从全图所有 gene 中随机采样新的 `gene_id`；
        3. 如果 `(drug_id, sampled_gene_id, disease_id)` 已在 HO 正例集合中，则重采样；
        4. 若多次随机失败，则回退为线性扫描，保证能找到合法负样本或明确报错。
        """

        drug_id = int(positive_path[0].item())
        pos_gene_id = int(positive_path[1].item())
        disease_id = int(positive_path[2].item())

        forbidden_gene_ids = self.positive_genes_by_pair[(drug_id, disease_id)]
        if len(forbidden_gene_ids) >= len(self.all_gene_ids_list):
            raise RuntimeError(
                "当前 `(drug, disease)` 对应的所有 gene 都已被正例占满，无法构造负样本："
                f"(drug_id={drug_id}, disease_id={disease_id})"
            )

        # 优先使用随机重采样，通常 forbidden 集合远小于全体 gene，效率较高。
        for _ in range(self.max_sampling_attempts):
            sampled_index = torch.randint(
                low=0,
                high=len(self.all_gene_ids_list),
                size=(1,),
            ).item()
            neg_gene_id = self.all_gene_ids_list[sampled_index]
            if neg_gene_id not in forbidden_gene_ids:
                negative_triplet = (drug_id, neg_gene_id, disease_id)
                if negative_triplet not in self.positive_path_set:
                    return torch.tensor(negative_triplet, dtype=torch.long)

        # 若随机多次失败，则改为线性扫描，保证采样稳定性。
        for neg_gene_id in self.all_gene_ids_list:
            if neg_gene_id == pos_gene_id:
                continue
            if neg_gene_id in forbidden_gene_ids:
                continue

            negative_triplet = (drug_id, neg_gene_id, disease_id)
            if negative_triplet not in self.positive_path_set:
                return torch.tensor(negative_triplet, dtype=torch.long)

        raise RuntimeError(
            "负采样失败：遍历所有 gene 后仍未找到合法负样本。"
        )


def build_path_bpr_dataloader(
    data: HeteroData,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = False,
    drop_last: bool = False,
    ho_attr_name: str = "ho_pos_paths",
    gene_node_type: str = "gene/protein",
    expected_ho_node_types: Sequence[str] = ("drug", "gene/protein", "disease"),
    max_sampling_attempts: int = 32,
) -> DataLoader[Tuple[Tensor, Tensor]]:
    """
    便捷函数：从 `HeteroData` 直接构造 PathBPRSampler 对应的 DataLoader。
    """

    dataset = PathBPRSampler(
        data=data,
        ho_attr_name=ho_attr_name,
        gene_node_type=gene_node_type,
        expected_ho_node_types=expected_ho_node_types,
        max_sampling_attempts=max_sampling_attempts,
    )

    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )
