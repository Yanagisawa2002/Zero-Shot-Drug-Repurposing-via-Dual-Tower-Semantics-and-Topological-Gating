from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import pandas as pd
import torch
from torch import Tensor
from torch_geometric.data import HeteroData

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.train_split_probe import derive_ho_triplets_for_pair_splits, load_pair_splits
from src.graph_surgery import build_split_isolation_targets, remove_leakage_edges
from src.pair_path_bpr_sampler import PairPathBPRDataset
from src.primekg_data_processor import PrimeKGDataProcessor


Pair = Tuple[int, int]
Triplet = Tuple[int, int, int]
Quad = Tuple[int, int, int, int]
EdgeType = Tuple[str, str, str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Audit possible data leakage for pairwise, split, OT, and HO evaluations.'
    )
    parser.add_argument('--processed-path', type=Path, required=True)
    parser.add_argument(
        '--text-json',
        type=Path,
        default=Path('outputs/pubmedbert_hybrid_entity_texts.json'),
        help='Entity text JSON used to generate PubMedBERT features.',
    )
    parser.add_argument('--nodes-csv', type=Path, default=Path('data/PrimeKG/nodes.csv'))
    parser.add_argument('--edges-csv', type=Path, default=Path('data/PrimeKG/edges.csv'))
    parser.add_argument(
        '--ot-novel-csv',
        type=Path,
        default=Path('outputs/ot_random_external_profile/novel_ood_triplets.csv'),
    )
    parser.add_argument('--feature-strategy-name', type=str, default='pubmedbert_hybrid')
    parser.add_argument('--skip-ot-audit', action='store_true')
    parser.add_argument('--report-json', type=Path, default=None)
    parser.add_argument('--max-examples', type=int, default=20)
    return parser.parse_args()


def load_entity_text_records(text_json_path: Path) -> List[Dict[str, object]]:
    payload = json.loads(text_json_path.read_text(encoding='utf-8'))
    if not isinstance(payload, list):
        raise TypeError('`entity_texts.json` ????????')
    return payload


def build_text_maps(
    records: Sequence[Mapping[str, object]],
) -> Tuple[Dict[int, str], Dict[int, str], Dict[int, str]]:
    id_to_text: Dict[int, str] = {}
    id_to_name: Dict[int, str] = {}
    id_to_raw_id: Dict[int, str] = {}
    for record in records:
        global_id = int(record['global_id'])
        id_to_text[global_id] = str(record.get('text', '') or '')
        id_to_name[global_id] = str(record.get('name', '') or '')
        id_to_raw_id[global_id] = str(record.get('raw_id', '') or '')
    return id_to_text, id_to_name, id_to_raw_id


def build_primekg_data(
    node_csv_path: Path,
    edge_csv_path: Path,
    train_triplets: Sequence[Triplet],
) -> Tuple[PrimeKGDataProcessor, HeteroData]:
    processor = PrimeKGDataProcessor(node_csv_path=node_csv_path, edge_csv_path=edge_csv_path)
    processor.build_entity_mappings()
    data = processor.build_heterodata(
        ho_id_paths=train_triplets,
        ho_type_order=('drug', 'gene/protein', 'disease'),
        add_inverse_edges=False,
    )
    return processor, data


def derive_split_triplets(
    processor: PrimeKGDataProcessor,
    processed_path: Path,
    edge_csv_path: Path,
) -> Tuple[str, Dict[str, set[Pair]], Dict[str, List[Triplet]]]:
    split_mode, pair_splits = load_pair_splits(processed_path)
    split_triplets = derive_ho_triplets_for_pair_splits(
        processor=processor,
        edge_csv_path=edge_csv_path,
        pair_splits=pair_splits,
    )
    return split_mode, pair_splits, split_triplets


def build_no_leakage_graph(
    data: HeteroData,
    heldout_triplets: Sequence[Triplet],
    split_mode: str,
    pair_splits: Mapping[str, Set[Pair]],
) -> HeteroData:
    isolate_nodes_by_type = build_split_isolation_targets(split_mode=split_mode, pair_splits=pair_splits)
    clean_edge_index_dict = remove_leakage_edges(
        data=data,
        target_paths=torch.tensor(heldout_triplets, dtype=torch.long),
        isolate_nodes_by_type=isolate_nodes_by_type,
    )
    clean_data = copy.deepcopy(data)
    for edge_type, edge_index in clean_edge_index_dict.items():
        clean_data[edge_type].edge_index = edge_index
    return clean_data


def load_ot_triplets_and_pairs(
    ot_novel_csv: Path,
    global_entity2id: Mapping[str, int],
) -> Tuple[List[Triplet], Set[Pair]]:
    if not ot_novel_csv.exists():
        return [], set()

    frame = pd.read_csv(ot_novel_csv)
    required_columns = {'primekg_drug_id', 'primekg_target_id', 'primekg_disease_id'}
    missing = required_columns.difference(frame.columns)
    if missing:
        raise KeyError(f'OT novel CSV ?????: {sorted(missing)}')

    triplets: List[Triplet] = []
    pair_set: Set[Pair] = set()
    for row in frame.itertuples(index=False):
        drug_raw = getattr(row, 'primekg_drug_id')
        target_raw = getattr(row, 'primekg_target_id')
        disease_raw = getattr(row, 'primekg_disease_id')
        if (
            drug_raw not in global_entity2id
            or target_raw not in global_entity2id
            or disease_raw not in global_entity2id
        ):
            continue
        drug_id = int(global_entity2id[drug_raw])
        gene_id = int(global_entity2id[target_raw])
        disease_id = int(global_entity2id[disease_raw])
        triplets.append((drug_id, gene_id, disease_id))
        pair_set.add((drug_id, disease_id))

    return list(dict.fromkeys(triplets)), pair_set


def audit_text_feature_contamination(
    id_to_text: Mapping[int, str],
    id_to_name: Mapping[int, str],
    valid_pairs: Iterable[Pair],
    test_pairs: Iterable[Pair],
    max_examples: int = 20,
) -> Dict[str, object]:
    inspected_pairs = list(valid_pairs) + list(test_pairs)
    leaks: List[Dict[str, object]] = []

    for drug_id, disease_id in inspected_pairs:
        drug_text = str(id_to_text.get(drug_id, ''))
        disease_text = str(id_to_text.get(disease_id, ''))
        drug_name = str(id_to_name.get(drug_id, '')).strip()
        disease_name = str(id_to_name.get(disease_id, '')).strip()

        if not drug_name or not disease_name:
            continue

        drug_text_lower = drug_text.casefold()
        disease_text_lower = disease_text.casefold()
        drug_name_lower = drug_name.casefold()
        disease_name_lower = disease_name.casefold()

        if disease_name_lower and disease_name_lower in drug_text_lower:
            leaks.append(
                {
                    'direction': 'drug_text_contains_disease_name',
                    'drug_id': int(drug_id),
                    'drug_name': drug_name,
                    'disease_id': int(disease_id),
                    'disease_name': disease_name,
                    'matched_substring': disease_name,
                    'text_preview': drug_text[:300],
                }
            )

        if drug_name_lower and drug_name_lower in disease_text_lower:
            leaks.append(
                {
                    'direction': 'disease_text_contains_drug_name',
                    'drug_id': int(drug_id),
                    'drug_name': drug_name,
                    'disease_id': int(disease_id),
                    'disease_name': disease_name,
                    'matched_substring': drug_name,
                    'text_preview': disease_text[:300],
                }
            )

    leak_pair_keys = {
        (int(item['drug_id']), int(item['disease_id']), str(item['direction']))
        for item in leaks
    }
    total_checks = len(inspected_pairs) * 2 if inspected_pairs else 0
    contamination_ratio = len(leak_pair_keys) / total_checks if total_checks > 0 else 0.0

    if leaks:
        print('[Leakage Audit 1] ??????????:')
        for example in leaks[:max_examples]:
            print(' ', example)
        raise AssertionError(
            '???????????'
            f'? {len(inspected_pairs)} ? pair ???????'
            f'?? {len(leak_pair_keys)} ??????? {contamination_ratio:.6f}?'
        )

    return {
        'num_pairs_checked': len(inspected_pairs),
        'num_directional_checks': total_checks,
        'num_leaks': len(leak_pair_keys),
        'contamination_ratio': contamination_ratio,
    }


def build_quad_test_set_from_triplets(
    data: HeteroData,
    test_triplets: Sequence[Triplet],
) -> Tensor:
    holder = copy.deepcopy(data)
    holder.ho_pos_paths = torch.tensor(test_triplets, dtype=torch.long)
    dataset = PairPathBPRDataset(
        data=holder,
        positive_paths=holder.ho_pos_paths,
        known_positive_pairs=holder.ho_pos_paths,
        negative_strategy='random',
        use_pathway_quads=True,
    )

    quad_rows: List[List[int]] = []
    for path_tensor in dataset.path_bank.values():
        quad_rows.extend(path_tensor.tolist())

    unique_quads = list(dict.fromkeys(tuple(int(x) for x in row) for row in quad_rows))
    if not unique_quads:
        return torch.empty((0, 4), dtype=torch.long)
    return torch.tensor(unique_quads, dtype=torch.long)


def _edge_pairs_from_graph(
    data: HeteroData,
    src_type: str,
    dst_type: str,
) -> Dict[EdgeType, Set[Tuple[int, int]]]:
    pair_sets: Dict[EdgeType, Set[Tuple[int, int]]] = {}
    for edge_type, edge_index in data.edge_index_dict.items():
        edge_src_type, _, edge_dst_type = edge_type
        if edge_src_type != src_type or edge_dst_type != dst_type:
            continue

        edge_index_cpu = edge_index.detach().cpu().to(torch.long)
        src_global_ids = data[src_type].global_id.detach().cpu().to(torch.long)[edge_index_cpu[0]]
        dst_global_ids = data[dst_type].global_id.detach().cpu().to(torch.long)[edge_index_cpu[1]]
        pair_sets[edge_type] = {
            (int(src_id), int(dst_id))
            for src_id, dst_id in zip(src_global_ids.tolist(), dst_global_ids.tolist())
        }
    return pair_sets


def audit_graph_surgery_residuals(
    no_leakage_data: HeteroData,
    test_quads: Tensor,
    max_examples: int = 20,
) -> Dict[str, object]:
    if test_quads.dim() != 2 or test_quads.size(1) != 4:
        raise ValueError('`test_quads` ??? `(N, 4)`?')

    gene_to_drug = _edge_pairs_from_graph(no_leakage_data, 'gene/protein', 'drug')
    pathway_to_gene = _edge_pairs_from_graph(no_leakage_data, 'pathway', 'gene/protein')
    disease_to_gene = _edge_pairs_from_graph(no_leakage_data, 'disease', 'gene/protein')
    disease_to_pathway = _edge_pairs_from_graph(no_leakage_data, 'disease', 'pathway')
    drug_to_disease = _edge_pairs_from_graph(no_leakage_data, 'drug', 'disease')
    disease_to_drug = _edge_pairs_from_graph(no_leakage_data, 'disease', 'drug')

    residuals: List[Dict[str, object]] = []
    for drug_id, gene_id, pathway_id, disease_id in test_quads.tolist():
        for edge_type, pair_set in gene_to_drug.items():
            if (int(gene_id), int(drug_id)) in pair_set:
                residuals.append(
                    {
                        'kind': 'reverse_gene_to_drug',
                        'edge_type': edge_type,
                        'quad': (int(drug_id), int(gene_id), int(pathway_id), int(disease_id)),
                    }
                )

        for edge_type, pair_set in pathway_to_gene.items():
            if (int(pathway_id), int(gene_id)) in pair_set:
                residuals.append(
                    {
                        'kind': 'reverse_pathway_to_gene',
                        'edge_type': edge_type,
                        'quad': (int(drug_id), int(gene_id), int(pathway_id), int(disease_id)),
                    }
                )

        for edge_type, pair_set in disease_to_gene.items():
            if (int(disease_id), int(gene_id)) in pair_set:
                residuals.append(
                    {
                        'kind': 'reverse_disease_to_gene',
                        'edge_type': edge_type,
                        'quad': (int(drug_id), int(gene_id), int(pathway_id), int(disease_id)),
                    }
                )

        for edge_type, pair_set in disease_to_pathway.items():
            if (int(disease_id), int(pathway_id)) in pair_set:
                residuals.append(
                    {
                        'kind': 'reverse_disease_to_pathway',
                        'edge_type': edge_type,
                        'quad': (int(drug_id), int(gene_id), int(pathway_id), int(disease_id)),
                    }
                )

        for edge_type, pair_set in drug_to_disease.items():
            if (int(drug_id), int(disease_id)) in pair_set:
                residuals.append(
                    {
                        'kind': 'shortcut_drug_to_disease',
                        'edge_type': edge_type,
                        'quad': (int(drug_id), int(gene_id), int(pathway_id), int(disease_id)),
                    }
                )

        for edge_type, pair_set in disease_to_drug.items():
            if (int(disease_id), int(drug_id)) in pair_set:
                residuals.append(
                    {
                        'kind': 'shortcut_disease_to_drug',
                        'edge_type': edge_type,
                        'quad': (int(drug_id), int(gene_id), int(pathway_id), int(disease_id)),
                    }
                )

    if residuals:
        print('[Leakage Audit 2] ??????????:')
        for example in residuals[:max_examples]:
            print(' ', example)
        raise AssertionError(
            '???????????'
            f'?? {len(residuals)} ?????????? no-leak ????????'
        )

    return {
        'num_test_quads_checked': int(test_quads.size(0)),
        'num_residual_hits': 0,
    }


def collect_node_ids_present_in_edges(
    data: HeteroData,
    node_type: str,
) -> Set[int]:
    present_ids: Set[int] = set()
    for edge_type, edge_index in data.edge_index_dict.items():
        src_type, _, dst_type = edge_type
        if edge_index.numel() == 0:
            continue

        edge_index_cpu = edge_index.detach().cpu().to(torch.long)
        if src_type == node_type:
            src_global_ids = data[src_type].global_id.detach().cpu().to(torch.long)[edge_index_cpu[0]]
            present_ids.update(int(x) for x in src_global_ids.tolist())
        if dst_type == node_type:
            dst_global_ids = data[dst_type].global_id.detach().cpu().to(torch.long)[edge_index_cpu[1]]
            present_ids.update(int(x) for x in dst_global_ids.tolist())

    return present_ids


def audit_strict_cross_split_purity(
    split_mode: str,
    pair_splits: Mapping[str, Set[Pair]],
    no_leakage_data: HeteroData,
    max_examples: int = 20,
) -> Dict[str, object]:
    if split_mode not in {'cross_drug', 'cross_disease'}:
        return {
            'split_mode': split_mode,
            'skipped': True,
            'reason': 'strict cross-split purity only applies to cross_drug / cross_disease',
        }

    train_pairs = pair_splits['train']
    test_pairs = pair_splits['test']

    if split_mode == 'cross_disease':
        train_entity_ids = {int(disease_id) for _, disease_id in train_pairs}
        test_entity_ids = {int(disease_id) for _, disease_id in test_pairs}
        entity_node_type = 'disease'
        entity_label = 'disease'
    else:
        train_entity_ids = {int(drug_id) for drug_id, _ in train_pairs}
        test_entity_ids = {int(drug_id) for drug_id, _ in test_pairs}
        entity_node_type = 'drug'
        entity_label = 'drug'

    overlap_entities = sorted(train_entity_ids.intersection(test_entity_ids))
    if overlap_entities:
        print('[Leakage Audit 3] ?? split ???????:')
        print(' ', overlap_entities[:max_examples])
        raise AssertionError(
            f'Strict Cross-Split Purity ?????train/test {entity_label} ???????'
            f'????? {len(overlap_entities)}?'
        )

    entity_ids_present_in_graph = collect_node_ids_present_in_edges(
        data=no_leakage_data,
        node_type=entity_node_type,
    )
    leaked_train_graph_entities = sorted(test_entity_ids.intersection(entity_ids_present_in_graph))
    if leaked_train_graph_entities:
        print('[Leakage Audit 3] ???????????????????:')
        print(' ', leaked_train_graph_entities[:max_examples])
        raise AssertionError(
            f'Strict Cross-Split Purity ?????test {entity_label} ?????????????????'
            f'????? {len(leaked_train_graph_entities)}?'
        )

    return {
        'split_mode': split_mode,
        'entity_label': entity_label,
        'num_train_entities': len(train_entity_ids),
        'num_test_entities': len(test_entity_ids),
        'num_entity_overlap': 0,
        'num_graph_presence_overlap': 0,
    }


def audit_ot_pair_level_leakage(
    train_pairs: Iterable[Pair],
    ot_pairs: Iterable[Pair],
    max_examples: int = 20,
) -> Dict[str, object]:
    train_pair_set = {(int(drug_id), int(disease_id)) for drug_id, disease_id in train_pairs}
    ot_pair_set = {(int(drug_id), int(disease_id)) for drug_id, disease_id in ot_pairs}
    overlapped_pairs = sorted(train_pair_set.intersection(ot_pair_set))

    if overlapped_pairs:
        print('[Leakage Audit 4] ?? OT ? PrimeKG train ? pair-level ??:')
        print(' ', overlapped_pairs[:max_examples])
        raise AssertionError(
            'OT Pair-level Leakage ?????'
            f'?? {len(overlapped_pairs)} ? OT `(drug, disease)` pair ???? PrimeKG ?????'
        )

    return {
        'num_primekg_train_pairs': len(train_pair_set),
        'num_ot_pairs': len(ot_pair_set),
        'num_overlapped_pairs': 0,
    }


def summarise_pair_splits(pair_splits: Mapping[str, Set[Pair]]) -> Dict[str, int]:
    return {split_name: len(pair_set) for split_name, pair_set in pair_splits.items()}


def run_audits(args: argparse.Namespace) -> Dict[str, object]:
    text_records = load_entity_text_records(args.text_json)
    id_to_text, id_to_name, _ = build_text_maps(text_records)

    processor, _ = build_primekg_data(
        node_csv_path=args.nodes_csv,
        edge_csv_path=args.edges_csv,
        train_triplets=[],
    )
    split_mode, pair_splits, split_triplets = derive_split_triplets(
        processor=processor,
        processed_path=args.processed_path,
        edge_csv_path=args.edges_csv,
    )

    _, full_data = build_primekg_data(
        node_csv_path=args.nodes_csv,
        edge_csv_path=args.edges_csv,
        train_triplets=split_triplets['train'],
    )
    ot_triplets, ot_pairs = load_ot_triplets_and_pairs(
        ot_novel_csv=args.ot_novel_csv,
        global_entity2id=processor.global_entity2id,
    )
    heldout_triplets = split_triplets['valid'] + split_triplets['test'] + ot_triplets
    no_leakage_data = build_no_leakage_graph(
        data=full_data,
        heldout_triplets=heldout_triplets,
        split_mode=split_mode,
        pair_splits=pair_splits,
    )

    valid_pairs = pair_splits['valid']
    test_pairs = pair_splits['test']
    test_quads = build_quad_test_set_from_triplets(
        data=full_data,
        test_triplets=split_triplets['test'],
    )

    results: Dict[str, object] = {
        'config': {
            'processed_path': str(args.processed_path),
            'split_mode': split_mode,
            'text_json': str(args.text_json),
            'feature_strategy_name': args.feature_strategy_name,
            'nodes_csv': str(args.nodes_csv),
            'edges_csv': str(args.edges_csv),
            'ot_novel_csv': str(args.ot_novel_csv),
        },
        'split_pair_counts': summarise_pair_splits(pair_splits),
        'split_triplet_counts': {
            split_name: len(paths) for split_name, paths in split_triplets.items()
        },
        'text_feature_audit': audit_text_feature_contamination(
            id_to_text=id_to_text,
            id_to_name=id_to_name,
            valid_pairs=valid_pairs,
            test_pairs=test_pairs,
            max_examples=args.max_examples,
        ),
        'graph_surgery_audit': audit_graph_surgery_residuals(
            no_leakage_data=no_leakage_data,
            test_quads=test_quads,
            max_examples=args.max_examples,
        ),
        'strict_cross_split_audit': audit_strict_cross_split_purity(
            split_mode=split_mode,
            pair_splits=pair_splits,
            no_leakage_data=no_leakage_data,
            max_examples=args.max_examples,
        ),
    }

    if not args.skip_ot_audit:
        results['ot_pair_overlap_audit'] = audit_ot_pair_level_leakage(
            train_pairs=pair_splits['train'],
            ot_pairs=ot_pairs,
            max_examples=args.max_examples,
        )

    return results


def main() -> None:
    args = parse_args()
    results = run_audits(args)
    if args.report_json is not None:
        args.report_json.parent.mkdir(parents=True, exist_ok=True)
        args.report_json.write_text(json.dumps(results, indent=2), encoding='utf-8')
        print(f'????????: {args.report_json}')
    print('???????')


if __name__ == '__main__':
    main()
