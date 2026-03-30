from __future__ import annotations

import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.extract_thick_disease_embs import (
    build_text_to_encode,
    build_thick_disease_table,
    load_base_diseases,
    load_definitions,
    load_ot_context,
)
from scripts.train_quad_split_ho_probe import build_clean_graph_without_leakage, load_ot_triplets
from scripts.train_split_probe import derive_ho_triplets_for_pair_splits, load_pair_splits
from src.graph_surgery import DEFAULT_DIRECT_LEAKAGE_RELATIONS
from src.pair_path_bpr_sampler import PairPathBPRDataset
from src.primekg_data_processor import PrimeKGDataProcessor

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs"
RESULTS_DIR = ROOT / "outputs/asym_route_dropedge_multiseed_20260327"
REPORT_JSON = OUT_DIR / "sota_defensive_audit_20260327.json"

SPLITS = ("random", "cross_drug", "cross_disease")
NEGATIVE_STRATEGIES = ("random", "cross_drug", "cross_disease")
AUDIT_SEEDS = (42, 43, 44)
OT_TABLES = [
    ROOT / "open target/Open_target_Phase_0.xlsx",
    ROOT / "open target/Open_target_Phase_1.xlsx",
    ROOT / "open target/Open_target_Phase_2.xlsx",
    ROOT / "open target/Open_target_Phase_3.xlsx",
    ROOT / "open target/Open_target_Phase_4.xlsx",
]

Pair = Tuple[int, int]


def load_config_for_split(split: str) -> Dict[str, Any]:
    path = RESULTS_DIR / f"{split}_seed42.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload["config"]


def build_processor() -> PrimeKGDataProcessor:
    processor = PrimeKGDataProcessor(
        node_csv_path=ROOT / "data/PrimeKG/nodes.csv",
        edge_csv_path=ROOT / "data/PrimeKG/edges.csv",
    )
    processor.build_entity_mappings()
    return processor


def build_split_artifacts(split: str):
    cfg = load_config_for_split(split)
    processor = build_processor()
    processed_path = ROOT / cfg["processed_path"]
    split_mode, pair_splits = load_pair_splits(processed_path)
    split_triplets = derive_ho_triplets_for_pair_splits(
        processor=processor,
        edge_csv_path=ROOT / "data/PrimeKG/edges.csv",
        pair_splits=pair_splits,
    )
    ot_triplets = load_ot_triplets(ROOT / cfg["ot_novel_csv"], processor.global_entity2id)
    full_data = processor.build_heterodata(ho_id_paths=split_triplets["train"], add_inverse_edges=False)
    heldout_triplets = split_triplets["valid"] + split_triplets["test"] + ot_triplets
    clean_data, total_removed_edges, leakage_edge_summary = build_clean_graph_without_leakage(
        data=full_data,
        heldout_triplets=heldout_triplets,
        split_mode=split_mode,
        pair_splits=pair_splits,
        graph_surgery_mode="direct_only",
    )
    return {
        "processor": processor,
        "split_mode": split_mode,
        "pair_splits": pair_splits,
        "split_triplets": split_triplets,
        "full_data": full_data,
        "clean_data": clean_data,
        "total_removed_edges": total_removed_edges,
        "leakage_edge_summary": leakage_edge_summary,
    }


def edge_pairs_for_type(data, edge_type: Tuple[str, str, str]) -> set[Pair]:
    if edge_type not in data.edge_index_dict:
        return set()
    edge_index = data[edge_type].edge_index.detach().cpu()
    if edge_index.numel() == 0:
        return set()
    src_type, _, dst_type = edge_type
    src_global = data[src_type].global_id.detach().cpu()[edge_index[0]].tolist()
    dst_global = data[dst_type].global_id.detach().cpu()[edge_index[1]].tolist()
    if src_type == "drug" and dst_type == "disease":
        return {(int(d), int(c)) for d, c in zip(src_global, dst_global)}
    if src_type == "disease" and dst_type == "drug":
        return {(int(c), int(d)) for c, d in zip(src_global, dst_global)}
    return set()


def all_direct_dd_edge_pairs_by_type(data) -> Dict[str, set[Pair]]:
    out: Dict[str, set[Pair]] = {}
    for edge_type in data.edge_index_dict.keys():
        src_type, _, dst_type = edge_type
        if {src_type, dst_type} != {"drug", "disease"}:
            continue
        out["|".join(edge_type)] = edge_pairs_for_type(data, edge_type)
    return out


def summarize_edge_hits(heldout_pairs: set[Pair], direct_pair_sets: Mapping[str, set[Pair]]) -> Tuple[Dict[str, int], Dict[str, int]]:
    direct_hits: Dict[str, int] = {}
    shadow_hits: Dict[str, int] = {}
    for edge_type_key, pair_set in direct_pair_sets.items():
        rel = edge_type_key.split("|")[1]
        overlap = heldout_pairs.intersection(pair_set)
        if not overlap:
            continue
        if rel in DEFAULT_DIRECT_LEAKAGE_RELATIONS:
            direct_hits[edge_type_key] = len(overlap)
        else:
            shadow_hits[edge_type_key] = len(overlap)
    return direct_hits, shadow_hits


def graph_surgery_audit(split: str) -> Dict[str, Any]:
    artifacts = build_split_artifacts(split)
    clean_data = artifacts["clean_data"]
    pair_level_holdout_pairs = {
        (int(drug_id), int(disease_id))
        for split_name in ("valid", "test")
        for drug_id, disease_id in artifacts["pair_splits"][split_name]
    }
    ho_covered_holdout_pairs = {
        (int(drug_id), int(disease_id))
        for split_name in ("valid", "test")
        for drug_id, _, disease_id in artifacts["split_triplets"][split_name]
    }
    direct_pair_sets = all_direct_dd_edge_pairs_by_type(clean_data)
    pair_direct_hits, pair_shadow_hits = summarize_edge_hits(pair_level_holdout_pairs, direct_pair_sets)
    ho_direct_hits, ho_shadow_hits = summarize_edge_hits(ho_covered_holdout_pairs, direct_pair_sets)
    return {
        "split": split,
        "num_pair_level_holdout_pairs": len(pair_level_holdout_pairs),
        "num_ho_covered_holdout_pairs": len(ho_covered_holdout_pairs),
        "pair_level_pass": len(pair_direct_hits) == 0,
        "ho_covered_pass": len(ho_direct_hits) == 0,
        "pair_level_direct_hits": pair_direct_hits,
        "ho_covered_direct_hits": ho_direct_hits,
        "pair_level_shadow_hits": pair_shadow_hits,
        "ho_covered_shadow_hits": ho_shadow_hits,
        "total_removed_edges": artifacts["total_removed_edges"],
        "removed_edge_summary": artifacts["leakage_edge_summary"],
    }


def load_global_indication_pairs(processor: PrimeKGDataProcessor) -> set[Pair]:
    edges_df = pd.read_csv(ROOT / "data/PrimeKG/edges.csv")
    filt = (
        (edges_df["src_type"].astype(str) == "drug")
        & (edges_df["dst_type"].astype(str) == "disease")
        & (edges_df["rel"].astype(str) == "indication")
    )
    subset = edges_df.loc[filt, ["src_id", "dst_id"]].drop_duplicates()
    pairs = set()
    for src_id, dst_id in subset.itertuples(index=False):
        if src_id in processor.global_entity2id and dst_id in processor.global_entity2id:
            pairs.add((int(processor.global_entity2id[src_id]), int(processor.global_entity2id[dst_id])))
    return pairs


def load_all_direct_relation_pairs(processor: PrimeKGDataProcessor) -> Dict[str, set[Pair]]:
    edges_df = pd.read_csv(ROOT / "data/PrimeKG/edges.csv")
    filt = (
        (edges_df["src_type"].astype(str) == "drug")
        & (edges_df["dst_type"].astype(str) == "disease")
    )
    out: Dict[str, set[Pair]] = defaultdict(set)
    for src_id, rel, dst_id in edges_df.loc[filt, ["src_id", "rel", "dst_id"]].itertuples(index=False):
        if src_id in processor.global_entity2id and dst_id in processor.global_entity2id:
            out[str(rel)].add((int(processor.global_entity2id[src_id]), int(processor.global_entity2id[dst_id])))
    return dict(out)


def negative_sampling_audit() -> Dict[str, Any]:
    processor = build_processor()
    global_indication_pairs = load_global_indication_pairs(processor)
    all_direct_relation_pairs = load_all_direct_relation_pairs(processor)
    split_reports: List[Dict[str, Any]] = []
    hard_failures: List[Dict[str, Any]] = []
    shadow_relation_overlaps: Dict[str, int] = defaultdict(int)

    for split in SPLITS:
        artifacts = build_split_artifacts(split)
        clean_data = artifacts["clean_data"]
        pair_splits = artifacts["pair_splits"]
        split_triplets = artifacts["split_triplets"]
        all_known_pairs = sorted(set().union(*pair_splits.values()))
        known_positive_pairs = torch.tensor(all_known_pairs, dtype=torch.long)
        split_report: Dict[str, Any] = {"split": split, "checks": []}

        for split_name in ("valid", "test"):
            positive_tensor = torch.tensor(split_triplets[split_name], dtype=torch.long)
            for neg_strategy in NEGATIVE_STRATEGIES:
                for seed in AUDIT_SEEDS:
                    torch.manual_seed(seed)
                    dataset = PairPathBPRDataset(
                        data=clean_data,
                        positive_paths=positive_tensor,
                        known_positive_pairs=known_positive_pairs,
                        negative_strategy=neg_strategy,
                        use_pathway_quads=True,
                    )
                    neg_pairs = [tuple(map(int, dataset[idx]["neg_pair_ids"].tolist())) for idx in range(len(dataset))]
                    neg_pair_set = set(neg_pairs)
                    overlap = neg_pair_set.intersection(global_indication_pairs)
                    if overlap:
                        hard_failures.append(
                            {
                                "split": split,
                                "subset": split_name,
                                "negative_strategy": neg_strategy,
                                "seed": seed,
                                "num_overlaps": len(overlap),
                                "examples": [list(pair) for pair in list(sorted(overlap))[:10]],
                            }
                        )
                    for rel, rel_pairs in all_direct_relation_pairs.items():
                        if rel == "indication":
                            continue
                        shadow_relation_overlaps[rel] += len(neg_pair_set.intersection(rel_pairs))
                    split_report["checks"].append(
                        {
                            "subset": split_name,
                            "negative_strategy": neg_strategy,
                            "seed": seed,
                            "num_negatives": len(neg_pairs),
                            "num_overlap_with_global_indication": len(overlap),
                        }
                    )
        split_reports.append(split_report)

    return {
        "pass": len(hard_failures) == 0,
        "global_indication_pairs": len(global_indication_pairs),
        "split_reports": split_reports,
        "hard_failures": hard_failures,
        "shadow_relation_overlaps": dict(sorted(shadow_relation_overlaps.items())),
    }


def load_node_name_maps() -> Dict[int, str]:
    nodes_df = pd.read_csv(ROOT / "data/PrimeKG/nodes.csv")
    processor = build_processor()
    id_to_name: Dict[int, str] = {}
    for node_id, name in nodes_df[["id", "name"]].itertuples(index=False):
        if node_id in processor.global_entity2id:
            id_to_name[int(processor.global_entity2id[node_id])] = str(name)
    return id_to_name


def build_disease_text_lookup() -> Dict[str, str]:
    base = load_base_diseases(
        disease_table=ROOT / "data/PrimeKG/nodes.csv",
        disease_id_column="id",
        disease_name_column="name",
        disease_type_column="type",
        disease_type_value="disease",
    )
    ot_context = load_ot_context(
        ot_tables=OT_TABLES,
        ot_id_column="MONDO_ID",
        ot_label_column="label",
        ot_ancestors_column="ancestors",
    )
    definition_context = load_definitions(None, "disease_id", "definition")
    thick = build_thick_disease_table(base, ot_context, definition_context)
    text_lookup: Dict[str, str] = {}
    for disease_id, name, ancestors, definition in thick.itertuples(index=False):
        text_lookup[str(disease_id)] = build_text_to_encode(str(name), str(ancestors), str(definition))
    return text_lookup


def textual_leakage_audit() -> Dict[str, Any]:
    processor = build_processor()
    processed_path = ROOT / load_config_for_split("cross_drug")["processed_path"]
    split_mode, pair_splits = load_pair_splits(processed_path)
    assert split_mode == "cross_drug"
    id_to_name = load_node_name_maps()
    disease_text_lookup = build_disease_text_lookup()

    train_drugs = {int(drug_id) for drug_id, _ in pair_splits["train"]}
    heldout_drugs = sorted({int(drug_id) for split_name in ("valid", "test") for drug_id, _ in pair_splits[split_name]})
    exclusive_drugs = [drug_id for drug_id in heldout_drugs if drug_id not in train_drugs]
    rng = random.Random(42)
    sampled_drugs = exclusive_drugs if len(exclusive_drugs) <= 100 else rng.sample(exclusive_drugs, 100)
    sampled_set = set(sampled_drugs)
    candidate_pairs = [pair for split_name in ("valid", "test") for pair in pair_splits[split_name] if int(pair[0]) in sampled_set]

    hits: List[Dict[str, Any]] = []
    checked_pairs = 0
    for drug_id, disease_id in candidate_pairs:
        checked_pairs += 1
        drug_name = (id_to_name.get(int(drug_id)) or "").strip()
        disease_node_id = processor.id2entity[int(disease_id)].raw_id
        disease_text = disease_text_lookup.get(disease_node_id, "")
        if drug_name and drug_name.lower() in disease_text.lower():
            hits.append(
                {
                    "drug_global_id": int(drug_id),
                    "drug_name": drug_name,
                    "disease_global_id": int(disease_id),
                    "disease_node_id": disease_node_id,
                    "matched_text_preview": disease_text[:300],
                }
            )

    return {
        "pass": len(hits) == 0,
        "sampled_exclusive_cross_drug_drugs": len(sampled_drugs),
        "checked_valid_test_pairs": checked_pairs,
        "num_hits": len(hits),
        "hit_examples": hits[:10],
    }


def print_report(report: Dict[str, Any]) -> None:
    print("# SOTA Defensive Audit")
    print()
    print("## Graph Surgery Audit")
    print("| Split | Pair-level Pass | HO-covered Pass | Pair-level Direct Hits | HO-covered Direct Hits | Shadow Edge Types |")
    print("|---|---:|---:|---:|---:|---:|")
    for item in report["graph_surgery"]:
        print(
            f"| `{item['split']}` | {str(item['pair_level_pass'])} | {str(item['ho_covered_pass'])} | "
            f"{sum(item['pair_level_direct_hits'].values())} | {sum(item['ho_covered_direct_hits'].values())} | "
            f"{len(item['pair_level_shadow_hits'])} |"
        )
    print()
    print("## Negative Sampling Audit")
    ns = report["negative_sampling"]
    print(f"Pass: {ns['pass']}")
    print(f"Global indication positives: {ns['global_indication_pairs']}")
    print(f"Hard failures: {len(ns['hard_failures'])}")
    if ns['shadow_relation_overlaps']:
        print("Shadow direct-relation overlaps (non-indication):")
        for rel, count in ns['shadow_relation_overlaps'].items():
            print(f"- `{rel}`: {count}")
    else:
        print("Shadow direct-relation overlaps (non-indication): none")
    print()
    print("## Textual Leakage Audit")
    txt = report['textual_leakage']
    print(f"Pass: {txt['pass']}")
    print(
        f"Sampled exclusive cross_drug drugs: {txt['sampled_exclusive_cross_drug_drugs']}; "
        f"checked pairs: {txt['checked_valid_test_pairs']}; hits: {txt['num_hits']}"
    )
    if txt['hit_examples']:
        print("Hit examples:")
        for hit in txt['hit_examples']:
            print(f"- {hit['drug_name']} -> {hit['disease_node_id']}")


def main() -> None:
    graph_reports = [graph_surgery_audit(split) for split in SPLITS]
    negative_report = negative_sampling_audit()
    text_report = textual_leakage_audit()
    report = {
        "graph_surgery": graph_reports,
        "negative_sampling": negative_report,
        "textual_leakage": text_report,
    }
    REPORT_JSON.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print_report(report)
    print(f"\nSaved JSON: {REPORT_JSON}")


if __name__ == "__main__":
    main()

