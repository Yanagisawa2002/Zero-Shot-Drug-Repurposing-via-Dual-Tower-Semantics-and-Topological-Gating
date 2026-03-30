from __future__ import annotations

import csv
import sys
from collections import Counter
from pathlib import Path
from typing import Dict

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.primekg_data_processor import PrimeKGDataProcessor


def _count_nodes_by_type(node_csv_path: Path) -> Dict[str, int]:
    """统计原始 PrimeKG 节点文件中每种节点类型的数量。"""

    counter: Counter[str] = Counter()
    with node_csv_path.open("r", encoding="utf-8-sig", newline="") as node_file:
        reader = csv.DictReader(node_file)
        for row in reader:
            counter[row["type"].strip()] += 1
    return dict(counter)


def test_build_heterodata_shapes() -> None:
    """
    使用几条真实可匹配的 HO 样本做最小化集成测试。

    重点验证：
    1. 各节点类型的 `num_nodes` 与原始 CSV 统计一致；
    2. 每种边类型的 `edge_index` 都是 `[2, num_edges]` 形状；
    3. `ho_pos_paths` 的维度与有效 HO 路径数量一致；
    4. 被跳过的 HO 路径会进入缺失记录。
    """

    node_csv_path = PROJECT_ROOT / "data" / "PrimeKG" / "nodes.csv"
    edge_csv_path = PROJECT_ROOT / "data" / "PrimeKG" / "edges.csv"

    processor = PrimeKGDataProcessor(
        node_csv_path=node_csv_path,
        edge_csv_path=edge_csv_path,
    )
    processor.build_entity_mappings()

    ho_triplets = [
        ("Copper", "PHYHIP", "hypertensive disorder"),
        ("Flunisolide", "KIF15", "Parkinson disease"),
        ("not_exist_drug", "PHYHIP", "hypertensive disorder"),
    ]
    ho_id_paths = processor.convert_ho_triplets_to_ids(ho_triplets)
    data = processor.build_heterodata(ho_id_paths)

    expected_num_nodes = _count_nodes_by_type(node_csv_path)
    assert set(data.node_types) == set(expected_num_nodes.keys())

    for node_type, node_count in expected_num_nodes.items():
        assert data[node_type].num_nodes == node_count
        assert data[node_type].node_id.shape == (node_count,)
        assert data[node_type].global_id.shape == (node_count,)
        assert data[node_type].node_id.dtype == torch.long
        assert data[node_type].global_id.dtype == torch.long

    assert len(data.edge_types) > 0
    for src_type, _, dst_type in data.edge_types:
        edge_index = data[(src_type, _, dst_type)].edge_index
        assert edge_index.dim() == 2
        assert edge_index.shape[0] == 2
        assert edge_index.dtype == torch.long

        if edge_index.shape[1] > 0:
            assert int(edge_index[0].min()) >= 0
            assert int(edge_index[1].min()) >= 0
            assert int(edge_index[0].max()) < data[src_type].num_nodes
            assert int(edge_index[1].max()) < data[dst_type].num_nodes

    assert data.ho_pos_paths.shape == (2, 3)
    assert data.ho_pos_paths.dtype == torch.long
    assert data.ho_path_node_types == ("drug", "gene/protein", "disease")

    # 两条路径成功映射，一条因缺失 drug 被跳过。
    assert len(ho_id_paths) == 2
    assert processor.skipped_ho_paths == [2]
    assert len(processor.missing_ho_entities) == 1
    assert processor.missing_ho_entities[0].expected_type == "drug"
    assert processor.missing_ho_entities[0].raw_value == "not_exist_drug"

    # 额外检查 HO 路径中每一列对应的全局 ID 的节点类型是否正确。
    for path in data.ho_pos_paths.tolist():
        for global_id, expected_type in zip(path, data.ho_path_node_types):
            assert processor.id2entity[global_id].node_type == expected_type
