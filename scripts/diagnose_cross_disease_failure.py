from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.train_quad_split_ho_probe import build_clean_graph_without_leakage
from scripts.train_split_probe import derive_ho_triplets_for_pair_splits, load_pair_splits
from src.pair_path_bpr_sampler import PairPathBPRDataset
from src.primekg_data_processor import PrimeKGDataProcessor

Pair = Tuple[int, int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Diagnose cross_disease degradation under strict pair-level clean protocol.'
    )
    parser.add_argument('--processed-path', type=Path, default=Path('data/PrimeKG/processed/primekg_indication_cross_disease.pt'))
    parser.add_argument('--nodes-csv', type=Path, default=Path('data/PrimeKG/nodes.csv'))
    parser.add_argument('--edges-csv', type=Path, default=Path('data/PrimeKG/edges.csv'))
    parser.add_argument('--disease-embeddings-pkl', type=Path, default=Path('thick_disease_text_embeddings.pkl'))
    parser.add_argument('--output-json', type=Path, default=Path('outputs/diagnose_cross_disease_failure_report.json'))
    parser.add_argument('--semantic-ood-threshold', type=float, default=0.5)
    parser.add_argument('--batch-size', type=int, default=512)
    return parser.parse_args()


def load_split_artifacts(processed_path: Path, nodes_csv: Path, edges_csv: Path):
    split_mode, pair_splits = load_pair_splits(processed_path)
    processor = PrimeKGDataProcessor(node_csv_path=nodes_csv, edge_csv_path=edges_csv)
    processor.build_entity_mappings()
    split_triplets = derive_ho_triplets_for_pair_splits(
        processor=processor,
        edge_csv_path=edges_csv,
        pair_splits=pair_splits,
    )
    full_data = processor.build_heterodata(ho_id_paths=split_triplets['train'], add_inverse_edges=False)
    heldout_triplets = split_triplets['valid'] + split_triplets['test']
    clean_data, total_removed_edges, leakage_edge_summary = build_clean_graph_without_leakage(
        data=full_data,
        heldout_triplets=heldout_triplets,
        split_mode=split_mode,
        pair_splits=pair_splits,
        graph_surgery_mode='direct_only',
    )
    return {
        'split_mode': split_mode,
        'pair_splits': pair_splits,
        'split_triplets': split_triplets,
        'processor': processor,
        'full_data': full_data,
        'clean_data': clean_data,
        'total_removed_edges': total_removed_edges,
        'leakage_edge_summary': leakage_edge_summary,
    }


def build_name_lookup(processor: PrimeKGDataProcessor) -> Dict[int, str]:
    return {int(global_id): record.name for global_id, record in processor.id2entity.items()}


def build_type_specific_degree_maps(clean_data) -> Dict[str, Dict[int, int]]:
    degree_maps: Dict[str, Dict[int, int]] = {}
    disease_type = 'disease'
    candidate_neighbor_types = [
        node_type for node_type in clean_data.node_types
        if node_type != disease_type and ('gene' in node_type or 'phenotype' in node_type)
    ]
    for neighbor_type in candidate_neighbor_types:
        degree_maps[neighbor_type] = {}

    disease_disease_degree: Dict[int, int] = {}
    all_non_drug_topology_degree: Dict[int, int] = {}

    for edge_type, edge_index in clean_data.edge_index_dict.items():
        if edge_index.numel() == 0:
            continue
        src_type, _, dst_type = edge_type
        edge_index_cpu = edge_index.detach().cpu().to(torch.long)
        src_global = clean_data[src_type].global_id.detach().cpu().to(torch.long)[edge_index_cpu[0]]
        dst_global = clean_data[dst_type].global_id.detach().cpu().to(torch.long)[edge_index_cpu[1]]

        if src_type == 'disease' and dst_type in degree_maps:
            counter = degree_maps[dst_type]
            for disease_id in src_global.tolist():
                counter[int(disease_id)] = counter.get(int(disease_id), 0) + 1
        elif dst_type == 'disease' and src_type in degree_maps:
            counter = degree_maps[src_type]
            for disease_id in dst_global.tolist():
                counter[int(disease_id)] = counter.get(int(disease_id), 0) + 1

        if src_type == 'disease' and dst_type == 'disease':
            for disease_id in src_global.tolist():
                disease_disease_degree[int(disease_id)] = disease_disease_degree.get(int(disease_id), 0) + 1
            for disease_id in dst_global.tolist():
                disease_disease_degree[int(disease_id)] = disease_disease_degree.get(int(disease_id), 0) + 1

        if src_type == 'disease' and dst_type != 'drug':
            for disease_id in src_global.tolist():
                all_non_drug_topology_degree[int(disease_id)] = all_non_drug_topology_degree.get(int(disease_id), 0) + 1
        if dst_type == 'disease' and src_type != 'drug':
            for disease_id in dst_global.tolist():
                all_non_drug_topology_degree[int(disease_id)] = all_non_drug_topology_degree.get(int(disease_id), 0) + 1

    degree_maps['disease_disease'] = disease_disease_degree
    degree_maps['all_non_drug_topology'] = all_non_drug_topology_degree
    return degree_maps


def summarize_degree_stats(
    disease_ids: Iterable[int],
    gene_degree_map: Mapping[int, int],
    phenotype_degree_map: Mapping[int, int],
    disease_disease_degree_map: Mapping[int, int],
    all_non_drug_topology_degree_map: Mapping[int, int],
) -> Dict[str, Any]:
    disease_ids = sorted({int(x) for x in disease_ids})
    if not disease_ids:
        return {
            'num_diseases': 0,
            'mean_gene_or_phenotype_degree': 0.0,
            'mean_gene_degree': 0.0,
            'mean_phenotype_degree': 0.0,
            'mean_disease_disease_degree': 0.0,
            'num_zero_gene_or_phenotype_degree': 0,
            'pct_zero_gene_or_phenotype_degree': 0.0,
        }

    gene_degrees = torch.tensor([int(gene_degree_map.get(d, 0)) for d in disease_ids], dtype=torch.float32)
    phenotype_degrees = torch.tensor([int(phenotype_degree_map.get(d, 0)) for d in disease_ids], dtype=torch.float32)
    disease_disease_degrees = torch.tensor([int(disease_disease_degree_map.get(d, 0)) for d in disease_ids], dtype=torch.float32)
    all_topology_degrees = torch.tensor([int(all_non_drug_topology_degree_map.get(d, 0)) for d in disease_ids], dtype=torch.float32)
    effective_degrees = gene_degrees + phenotype_degrees
    zero_count = int((effective_degrees == 0).sum().item())
    absolute_island_count = int((all_topology_degrees == 0).sum().item())

    return {
        'num_diseases': len(disease_ids),
        'mean_gene_or_phenotype_degree': float(effective_degrees.mean().item()),
        'mean_gene_degree': float(gene_degrees.mean().item()),
        'mean_phenotype_degree': float(phenotype_degrees.mean().item()),
        'mean_disease_disease_degree': float(disease_disease_degrees.mean().item()),
        'mean_all_non_drug_topology_degree': float(all_topology_degrees.mean().item()),
        'num_zero_gene_or_phenotype_degree': zero_count,
        'pct_zero_gene_or_phenotype_degree': float(zero_count / len(disease_ids)),
        'num_absolute_islands': absolute_island_count,
        'pct_absolute_islands': float(absolute_island_count / len(disease_ids)),
    }


def degree_collapse_analysis(artifacts: Mapping[str, Any]) -> Dict[str, Any]:
    pair_splits = artifacts['pair_splits']
    degree_maps = build_type_specific_degree_maps(artifacts['clean_data'])
    gene_degree_map = degree_maps.get('gene/protein', {})
    phenotype_keys = [key for key in degree_maps.keys() if 'phenotype' in key]
    phenotype_degree_map: Dict[int, int] = {}
    for key in phenotype_keys:
        for disease_id, count in degree_maps[key].items():
            phenotype_degree_map[disease_id] = phenotype_degree_map.get(disease_id, 0) + int(count)
    disease_disease_degree_map = degree_maps.get('disease_disease', {})
    all_non_drug_topology_degree_map = degree_maps.get('all_non_drug_topology', {})

    split_stats = {}
    zero_islands: List[int] = []
    absolute_islands: List[int] = []
    for split_name in ('train', 'valid', 'test'):
        disease_ids = [int(disease_id) for _, disease_id in pair_splits[split_name]]
        stats = summarize_degree_stats(
            disease_ids=disease_ids,
            gene_degree_map=gene_degree_map,
            phenotype_degree_map=phenotype_degree_map,
            disease_disease_degree_map=disease_disease_degree_map,
            all_non_drug_topology_degree_map=all_non_drug_topology_degree_map,
        )
        split_stats[split_name] = stats
        if split_name == 'test':
            zero_islands = sorted({
                int(disease_id)
                for _, disease_id in pair_splits['test']
                if int(gene_degree_map.get(int(disease_id), 0) + phenotype_degree_map.get(int(disease_id), 0)) == 0
            })
            absolute_islands = sorted({
                int(disease_id)
                for _, disease_id in pair_splits['test']
                if int(all_non_drug_topology_degree_map.get(int(disease_id), 0)) == 0
            })

    return {
        'neighbor_types_counted': ['gene/protein'] + phenotype_keys,
        'phenotype_node_types_present': phenotype_keys,
        'split_stats': split_stats,
        'test_zero_degree_disease_ids': zero_islands,
        'test_absolute_island_disease_ids': absolute_islands,
    }


def path_reachability_analysis(artifacts: Mapping[str, Any]) -> Dict[str, Any]:
    dataset = PairPathBPRDataset(
        data=artifacts['clean_data'],
        positive_paths=artifacts['clean_data'].ho_pos_paths,
        known_positive_pairs=torch.tensor(sorted(set().union(*artifacts['pair_splits'].values())), dtype=torch.long),
        negative_strategy='random',
        use_pathway_quads=True,
    )
    test_pairs = sorted({(int(d), int(c)) for d, c in artifacts['pair_splits']['test']})
    path_counts: List[int] = []
    unreachable_pairs: List[Pair] = []

    for pair in test_pairs:
        pair_paths = dataset.topology_path_bank.get(pair)
        count = 0 if pair_paths is None else int(pair_paths.size(0))
        path_counts.append(count)
        if count == 0:
            unreachable_pairs.append(pair)

    reachable_counts = [count for count in path_counts if count > 0]
    num_pairs = len(test_pairs)
    num_unreachable = len(unreachable_pairs)
    num_reachable = num_pairs - num_unreachable
    avg_reachable = float(sum(reachable_counts) / len(reachable_counts)) if reachable_counts else 0.0
    avg_all = float(sum(path_counts) / len(path_counts)) if path_counts else 0.0

    return {
        'num_test_pairs': num_pairs,
        'num_reachable_pairs': num_reachable,
        'num_unreachable_pairs': num_unreachable,
        'pct_unreachable_pairs': float(num_unreachable / num_pairs) if num_pairs else 0.0,
        'average_num_paths_among_reachable_pairs': avg_reachable,
        'average_num_paths_over_all_test_pairs': avg_all,
        'max_num_paths': max(path_counts) if path_counts else 0,
        'example_unreachable_pairs': [list(pair) for pair in unreachable_pairs[:20]],
    }


def load_embedding_dict(pkl_path: Path) -> Dict[str, torch.Tensor]:
    with pkl_path.open('rb') as f:
        data = json.load(f) if pkl_path.suffix.lower() == '.json' else None
    if data is not None:
        raise TypeError('Expected pickle file, got json.')
    import pickle
    with pkl_path.open('rb') as f:
        obj = pickle.load(f)
    out: Dict[str, torch.Tensor] = {}
    for key, value in obj.items():
        out[str(key)] = torch.as_tensor(value, dtype=torch.float32).view(-1)
    return out


def semantic_ood_analysis(artifacts: Mapping[str, Any], disease_embeddings_pkl: Path, threshold: float, batch_size: int) -> Dict[str, Any]:
    processor = artifacts['processor']
    embedding_dict = load_embedding_dict(disease_embeddings_pkl)

    train_disease_ids = sorted({int(disease_id) for _, disease_id in artifacts['pair_splits']['train']})
    test_disease_ids = sorted({int(disease_id) for _, disease_id in artifacts['pair_splits']['test']})

    train_raw_ids = [processor.id2entity[d].raw_id for d in train_disease_ids if processor.id2entity[d].raw_id in embedding_dict]
    test_pairs = [(d, processor.id2entity[d].raw_id) for d in test_disease_ids if processor.id2entity[d].raw_id in embedding_dict]
    if not train_raw_ids or not test_pairs:
        raise RuntimeError('Missing disease embeddings for train/test diseases.')

    train_matrix = torch.stack([embedding_dict[raw_id] for raw_id in train_raw_ids], dim=0)
    test_matrix = torch.stack([embedding_dict[raw_id] for _, raw_id in test_pairs], dim=0)
    train_matrix = F.normalize(train_matrix, p=2, dim=-1)
    test_matrix = F.normalize(test_matrix, p=2, dim=-1)

    max_sims: List[float] = []
    nearest_train_indices: List[int] = []
    for start in range(0, test_matrix.size(0), batch_size):
        batch = test_matrix[start:start + batch_size]
        sims = batch @ train_matrix.T
        vals, idx = sims.max(dim=1)
        max_sims.extend(vals.tolist())
        nearest_train_indices.extend(idx.tolist())

    severe = [i for i, sim in enumerate(max_sims) if sim < threshold]
    name_lookup = build_name_lookup(processor)
    examples = []
    for idx in severe[:20]:
        test_gid = test_pairs[idx][0]
        train_gid = train_disease_ids[nearest_train_indices[idx]]
        examples.append({
            'test_disease_global_id': int(test_gid),
            'test_disease_name': name_lookup.get(int(test_gid), ''),
            'nearest_train_disease_global_id': int(train_gid),
            'nearest_train_disease_name': name_lookup.get(int(train_gid), ''),
            'max_cosine_similarity': float(max_sims[idx]),
        })

    sims_tensor = torch.tensor(max_sims, dtype=torch.float32)
    return {
        'num_train_diseases': len(train_disease_ids),
        'num_test_diseases': len(test_disease_ids),
        'mean_max_cosine_similarity': float(sims_tensor.mean().item()),
        'median_max_cosine_similarity': float(sims_tensor.median().item()),
        'min_max_cosine_similarity': float(sims_tensor.min().item()),
        'num_below_threshold': len(severe),
        'pct_below_threshold': float(len(severe) / len(test_disease_ids)),
        'threshold': float(threshold),
        'examples_below_threshold': examples,
    }


def print_report(report: Mapping[str, Any]) -> None:
    print('# cross_disease Failure Diagnosis')
    print()
    degree = report['degree_collapse']
    print('## 1. Degree Collapse Analysis')
    print(f"Neighbor types counted: {degree['neighbor_types_counted']}")
    print('| Split | #Diseases | Mean gene/phenotype degree | Mean gene degree | Mean phenotype degree | Mean disease-disease degree | Mean all non-drug topology degree | #Zero effective degree | #Absolute islands |')
    print('|---|---:|---:|---:|---:|---:|---:|---:|---:|')
    for split_name in ('train', 'valid', 'test'):
        s = degree['split_stats'][split_name]
        print(
            f"| `{split_name}` | {s['num_diseases']} | {s['mean_gene_or_phenotype_degree']:.2f} | {s['mean_gene_degree']:.2f} | {s['mean_phenotype_degree']:.2f} | {s['mean_disease_disease_degree']:.2f} | {s['mean_all_non_drug_topology_degree']:.2f} | {s['num_zero_gene_or_phenotype_degree']} ({100.0*s['pct_zero_gene_or_phenotype_degree']:.2f}%) | {s['num_absolute_islands']} ({100.0*s['pct_absolute_islands']:.2f}%) |"
        )
    print(f"Test diseases with zero gene/phenotype degree: {len(degree['test_zero_degree_disease_ids'])}")
    print(f"Test diseases that are absolute topology islands: {len(degree['test_absolute_island_disease_ids'])}")
    print()

    reach = report['path_reachability']
    print('## 2. Path Reachability Analysis')
    print(f"Test positive pairs: {reach['num_test_pairs']}")
    print(f"Reachable pairs: {reach['num_reachable_pairs']} ({100.0 * reach['num_reachable_pairs'] / max(reach['num_test_pairs'],1):.2f}%)")
    print(f"Unreachable pairs: {reach['num_unreachable_pairs']} ({100.0 * reach['pct_unreachable_pairs']:.2f}%)")
    print(f"Average #paths among reachable pairs: {reach['average_num_paths_among_reachable_pairs']:.2f}")
    print(f"Average #paths over all test pairs: {reach['average_num_paths_over_all_test_pairs']:.2f}")
    print(f"Max #paths for a test pair: {reach['max_num_paths']}")
    print()

    sem = report['semantic_ood']
    print('## 3. Textual Semantic OOD')
    print(f"Train diseases with embeddings: {sem['num_train_diseases']}")
    print(f"Test diseases with embeddings: {sem['num_test_diseases']}")
    print(f"Mean max cosine similarity to train: {sem['mean_max_cosine_similarity']:.4f}")
    print(f"Median max cosine similarity to train: {sem['median_max_cosine_similarity']:.4f}")
    print(f"Min max cosine similarity to train: {sem['min_max_cosine_similarity']:.4f}")
    print(f"# test diseases below {sem['threshold']:.2f}: {sem['num_below_threshold']} ({100.0 * sem['pct_below_threshold']:.2f}%)")


def main() -> None:
    args = parse_args()
    artifacts = load_split_artifacts(
        processed_path=args.processed_path,
        nodes_csv=args.nodes_csv,
        edges_csv=args.edges_csv,
    )
    report = {
        'processed_path': str(args.processed_path),
        'split_mode': artifacts['split_mode'],
        'total_removed_edges': int(artifacts['total_removed_edges']),
        'leakage_edge_summary': artifacts['leakage_edge_summary'],
        'degree_collapse': degree_collapse_analysis(artifacts),
        'path_reachability': path_reachability_analysis(artifacts),
        'semantic_ood': semantic_ood_analysis(
            artifacts,
            disease_embeddings_pkl=args.disease_embeddings_pkl,
            threshold=args.semantic_ood_threshold,
            batch_size=args.batch_size,
        ),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print_report(report)
    print()
    print(f'Report saved to: {args.output_json}')


if __name__ == '__main__':
    main()
