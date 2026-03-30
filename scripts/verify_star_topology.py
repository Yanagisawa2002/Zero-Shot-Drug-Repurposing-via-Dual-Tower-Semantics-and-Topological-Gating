from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, DefaultDict, Dict, Iterable, List, Mapping, Sequence, Set, Tuple

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.train_quad_split_ho_probe import build_clean_graph_without_leakage
from scripts.train_split_probe import derive_ho_triplets_for_pair_splits, load_pair_splits
from src.pair_path_bpr_sampler import PairPathBPRDataset
from src.primekg_data_processor import PrimeKGDataProcessor


Pair = Tuple[int, int]
EdgeType = Tuple[str, str, str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Verify whether the cleaned PrimeKG behaves like a star topology around gene/protein hubs.'
    )
    parser.add_argument('--processed-path', type=Path, default=Path('data/PrimeKG/processed/primekg_indication_cross_disease.pt'))
    parser.add_argument('--nodes-csv', type=Path, default=Path('data/PrimeKG/nodes.csv'))
    parser.add_argument('--edges-csv', type=Path, default=Path('data/PrimeKG/edges.csv'))
    parser.add_argument('--output-json', type=Path, default=Path('outputs/verify_star_topology_report.json'))
    return parser.parse_args()


def load_split_artifacts(processed_path: Path, nodes_csv: Path, edges_csv: Path) -> Dict[str, Any]:
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
        'total_removed_edges': int(total_removed_edges),
        'leakage_edge_summary': leakage_edge_summary,
    }


def edge_type_audit(clean_data) -> Dict[str, Any]:
    all_edge_types = [tuple(edge_type) for edge_type in clean_data.edge_types]
    direct_drug_pathway: List[Dict[str, Any]] = []
    direct_disease_pathway: List[Dict[str, Any]] = []
    gene_connected: List[Dict[str, Any]] = []

    for edge_type, edge_index in clean_data.edge_index_dict.items():
        src_type, relation, dst_type = edge_type
        count = int(edge_index.size(1))
        item = {
            'edge_type': list(edge_type),
            'num_edges': count,
        }
        if {src_type, dst_type} == {'drug', 'pathway'}:
            direct_drug_pathway.append(item)
        if {src_type, dst_type} == {'disease', 'pathway'}:
            direct_disease_pathway.append(item)
        if 'gene/protein' in {src_type, dst_type}:
            gene_connected.append(item)

    return {
        'num_edge_types': len(all_edge_types),
        'all_edge_types': [list(edge_type) for edge_type in all_edge_types],
        'direct_drug_pathway_edge_types': direct_drug_pathway,
        'direct_disease_pathway_edge_types': direct_disease_pathway,
        'gene_connected_edge_types': sorted(gene_connected, key=lambda item: (-item['num_edges'], item['edge_type'])),
    }


def compute_incident_degree_by_type(clean_data) -> Dict[str, Dict[str, float]]:
    degree_by_type: Dict[str, torch.Tensor] = {
        node_type: torch.zeros(int(clean_data[node_type].num_nodes), dtype=torch.float32)
        for node_type in clean_data.node_types
    }

    for edge_type, edge_index in clean_data.edge_index_dict.items():
        if edge_index.numel() == 0:
            continue
        src_type, _, dst_type = edge_type
        src_indices = edge_index[0].detach().cpu().to(torch.long)
        dst_indices = edge_index[1].detach().cpu().to(torch.long)
        degree_by_type[src_type].scatter_add_(0, src_indices, torch.ones_like(src_indices, dtype=torch.float32))
        degree_by_type[dst_type].scatter_add_(0, dst_indices, torch.ones_like(dst_indices, dtype=torch.float32))

    summary: Dict[str, Dict[str, float]] = {}
    for node_type, degrees in degree_by_type.items():
        summary[node_type] = {
            'num_nodes': int(degrees.numel()),
            'mean_degree': float(degrees.mean().item()) if degrees.numel() > 0 else 0.0,
            'median_degree': float(degrees.median().item()) if degrees.numel() > 0 else 0.0,
            'max_degree': float(degrees.max().item()) if degrees.numel() > 0 else 0.0,
            'num_zero_degree': int((degrees == 0).sum().item()),
        }
    return summary


def build_neighbor_maps(clean_data) -> Tuple[Dict[int, Set[int]], Dict[int, Set[int]], Dict[int, Set[int]]]:
    drug_to_genes: DefaultDict[int, Set[int]] = defaultdict(set)
    disease_to_genes: DefaultDict[int, Set[int]] = defaultdict(set)
    gene_to_pathways: DefaultDict[int, Set[int]] = defaultdict(set)

    for edge_type, edge_index in clean_data.edge_index_dict.items():
        if edge_index.numel() == 0:
            continue
        src_type, _, dst_type = edge_type
        edge_index_cpu = edge_index.detach().cpu().to(torch.long)
        src_global_ids = clean_data[src_type].global_id.detach().cpu().to(torch.long)[edge_index_cpu[0]]
        dst_global_ids = clean_data[dst_type].global_id.detach().cpu().to(torch.long)[edge_index_cpu[1]]

        if {src_type, dst_type} == {'drug', 'gene/protein'}:
            if src_type == 'drug':
                pairs = zip(src_global_ids.tolist(), dst_global_ids.tolist())
            else:
                pairs = zip(dst_global_ids.tolist(), src_global_ids.tolist())
            for drug_id, gene_id in pairs:
                drug_to_genes[int(drug_id)].add(int(gene_id))
            continue

        if {src_type, dst_type} == {'disease', 'gene/protein'}:
            if src_type == 'disease':
                pairs = zip(src_global_ids.tolist(), dst_global_ids.tolist())
            else:
                pairs = zip(dst_global_ids.tolist(), src_global_ids.tolist())
            for disease_id, gene_id in pairs:
                disease_to_genes[int(disease_id)].add(int(gene_id))
            continue

        if {src_type, dst_type} == {'pathway', 'gene/protein'}:
            if src_type == 'gene/protein':
                pairs = zip(src_global_ids.tolist(), dst_global_ids.tolist())
            else:
                pairs = zip(dst_global_ids.tolist(), src_global_ids.tolist())
            for gene_id, pathway_id in pairs:
                gene_to_pathways[int(gene_id)].add(int(pathway_id))

    return dict(drug_to_genes), dict(disease_to_genes), dict(gene_to_pathways)


def linear_path_reachability(clean_data, pair_splits: Mapping[str, Iterable[Pair]]) -> Dict[str, Any]:
    dataset = PairPathBPRDataset(
        data=clean_data,
        positive_paths=clean_data.ho_pos_paths,
        known_positive_pairs=torch.tensor(sorted(set().union(*pair_splits.values())), dtype=torch.long),
        negative_strategy='random',
        use_pathway_quads=True,
    )
    test_pairs = sorted({(int(d), int(c)) for d, c in pair_splits['test']})
    path_counts: List[int] = []
    for pair in test_pairs:
        pair_paths = dataset.topology_path_bank.get(pair)
        count = 0 if pair_paths is None else int(pair_paths.size(0))
        path_counts.append(count)
    reachable = sum(count > 0 for count in path_counts)
    return {
        'num_test_pairs': len(test_pairs),
        'num_reachable_pairs': int(reachable),
        'pct_reachable_pairs': float(reachable / len(test_pairs)) if test_pairs else 0.0,
        'average_num_paths_over_all_test_pairs': float(sum(path_counts) / len(path_counts)) if path_counts else 0.0,
    }


def theoretical_star_reachability(clean_data, pair_splits: Mapping[str, Iterable[Pair]]) -> Dict[str, Any]:
    drug_to_genes, disease_to_genes, gene_to_pathways = build_neighbor_maps(clean_data)
    test_pairs = sorted({(int(drug_id), int(disease_id)) for drug_id, disease_id in pair_splits['test']})

    num_with_shared_gene = 0
    num_star_reachable = 0
    shared_gene_counts: List[int] = []
    star_gene_counts: List[int] = []
    star_path_counts: List[int] = []
    example_pairs: List[Dict[str, Any]] = []

    for drug_id, disease_id in test_pairs:
        drug_genes = drug_to_genes.get(int(drug_id), set())
        disease_genes = disease_to_genes.get(int(disease_id), set())
        shared_genes = drug_genes.intersection(disease_genes)
        shared_gene_counts.append(len(shared_genes))
        if shared_genes:
            num_with_shared_gene += 1

        star_genes = [gene_id for gene_id in shared_genes if gene_to_pathways.get(gene_id)]
        star_gene_counts.append(len(star_genes))
        total_paths_for_pair = sum(len(gene_to_pathways.get(gene_id, ())) for gene_id in star_genes)
        star_path_counts.append(total_paths_for_pair)
        if star_genes:
            num_star_reachable += 1
            if len(example_pairs) < 20:
                example_pairs.append({
                    'drug_id': int(drug_id),
                    'disease_id': int(disease_id),
                    'num_shared_genes': len(shared_genes),
                    'num_star_genes': len(star_genes),
                    'num_star_paths': int(total_paths_for_pair),
                })

    num_pairs = len(test_pairs)
    return {
        'num_test_pairs': num_pairs,
        'num_pairs_with_shared_gene': int(num_with_shared_gene),
        'pct_pairs_with_shared_gene': float(num_with_shared_gene / num_pairs) if num_pairs else 0.0,
        'num_star_reachable_pairs': int(num_star_reachable),
        'pct_star_reachable_pairs': float(num_star_reachable / num_pairs) if num_pairs else 0.0,
        'average_num_shared_genes': float(sum(shared_gene_counts) / len(shared_gene_counts)) if shared_gene_counts else 0.0,
        'average_num_star_genes': float(sum(star_gene_counts) / len(star_gene_counts)) if star_gene_counts else 0.0,
        'average_num_star_paths': float(sum(star_path_counts) / len(star_path_counts)) if star_path_counts else 0.0,
        'max_num_star_paths': int(max(star_path_counts)) if star_path_counts else 0,
        'example_star_reachable_pairs': example_pairs,
    }


def print_report(report: Mapping[str, Any]) -> None:
    print('# Star Topology Verification')
    print()

    edge_audit = report['edge_type_audit']
    print('## 1. Edge Type Audit')
    print(f"Total edge types: {edge_audit['num_edge_types']}")
    print(f"Direct drug-pathway edge types: {len(edge_audit['direct_drug_pathway_edge_types'])}")
    for item in edge_audit['direct_drug_pathway_edge_types']:
        print(f"  - {tuple(item['edge_type'])}: {item['num_edges']}")
    print(f"Direct disease-pathway edge types: {len(edge_audit['direct_disease_pathway_edge_types'])}")
    for item in edge_audit['direct_disease_pathway_edge_types']:
        print(f"  - {tuple(item['edge_type'])}: {item['num_edges']}")
    print('Gene/protein-connected edge types:')
    for item in edge_audit['gene_connected_edge_types']:
        print(f"  - {tuple(item['edge_type'])}: {item['num_edges']}")
    print()

    degree = report['degree_summary']
    print('## 2. Hub Centrality')
    print('| Node Type | #Nodes | Mean Degree | Median Degree | Max Degree | #Zero Degree |')
    print('|---|---:|---:|---:|---:|---:|')
    for node_type in ('gene/protein', 'drug', 'disease', 'pathway'):
        if node_type not in degree:
            continue
        stats = degree[node_type]
        print(
            f"| `{node_type}` | {stats['num_nodes']} | {stats['mean_degree']:.2f} | {stats['median_degree']:.2f} | {stats['max_degree']:.0f} | {stats['num_zero_degree']} |"
        )
    print()

    linear = report['linear_path_reachability']
    star = report['theoretical_star_reachability']
    print('## 3. Theoretical Star Reachability')
    print(f"Linear quad-path reachable pairs: {linear['num_reachable_pairs']} / {linear['num_test_pairs']} ({100.0 * linear['pct_reachable_pairs']:.2f}%)")
    print(f"Pairs with at least one shared drug-disease gene: {star['num_pairs_with_shared_gene']} / {star['num_test_pairs']} ({100.0 * star['pct_pairs_with_shared_gene']:.2f}%)")
    print(f"Star-reachable pairs (shared gene with pathway): {star['num_star_reachable_pairs']} / {star['num_test_pairs']} ({100.0 * star['pct_star_reachable_pairs']:.2f}%)")
    print(f"Average #shared genes per pair: {star['average_num_shared_genes']:.2f}")
    print(f"Average #star genes per pair: {star['average_num_star_genes']:.2f}")
    print(f"Average #star paths per pair: {star['average_num_star_paths']:.2f}")
    print(f"Max #star paths for a pair: {star['max_num_star_paths']}")


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
        'total_removed_edges': artifacts['total_removed_edges'],
        'leakage_edge_summary': artifacts['leakage_edge_summary'],
        'edge_type_audit': edge_type_audit(artifacts['clean_data']),
        'degree_summary': compute_incident_degree_by_type(artifacts['clean_data']),
        'linear_path_reachability': linear_path_reachability(artifacts['clean_data'], artifacts['pair_splits']),
        'theoretical_star_reachability': theoretical_star_reachability(artifacts['clean_data'], artifacts['pair_splits']),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print_report(report)
    print()
    print(f'Report saved to: {args.output_json}')


if __name__ == '__main__':
    main()
