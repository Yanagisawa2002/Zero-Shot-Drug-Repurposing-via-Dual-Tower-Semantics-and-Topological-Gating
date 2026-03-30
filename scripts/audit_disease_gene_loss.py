from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.train_quad_split_ho_probe import build_clean_graph_without_leakage
from scripts.train_split_probe import derive_ho_triplets_for_pair_splits, load_pair_splits
from src.primekg_data_processor import PrimeKGDataProcessor

DiseaseId = int
GeneId = int
EdgeType = Tuple[str, str, str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Audit whether cross_disease disease-gene edge loss comes from PrimeKG sparsity or strict-clean removal.'
    )
    parser.add_argument('--processed-path', type=Path, default=Path('data/PrimeKG/processed/primekg_indication_cross_disease.pt'))
    parser.add_argument('--nodes-csv', type=Path, default=Path('data/PrimeKG/nodes.csv'))
    parser.add_argument('--edges-csv', type=Path, default=Path('data/PrimeKG/edges.csv'))
    parser.add_argument('--output-json', type=Path, default=Path('outputs/audit_disease_gene_loss_report.json'))
    parser.add_argument('--print-max-diseases', type=int, default=20)
    return parser.parse_args()


def load_graphs(processed_path: Path, nodes_csv: Path, edges_csv: Path):
    split_mode, pair_splits = load_pair_splits(processed_path)
    if split_mode != 'cross_disease':
        raise ValueError(f'Expected cross_disease split, got {split_mode}')

    processor = PrimeKGDataProcessor(node_csv_path=nodes_csv, edge_csv_path=edges_csv)
    processor.build_entity_mappings()
    split_triplets = derive_ho_triplets_for_pair_splits(
        processor=processor,
        edge_csv_path=edges_csv,
        pair_splits=pair_splits,
    )
    raw_data = processor.build_heterodata(ho_id_paths=split_triplets['train'], add_inverse_edges=False)
    heldout_triplets = split_triplets['valid'] + split_triplets['test']
    clean_data, total_removed_edges, leakage_edge_summary = build_clean_graph_without_leakage(
        data=raw_data,
        heldout_triplets=heldout_triplets,
        split_mode=split_mode,
        pair_splits=pair_splits,
        graph_surgery_mode='direct_only',
    )
    return processor, pair_splits, raw_data, clean_data, total_removed_edges, leakage_edge_summary


def collect_disease_gene_adjacency(data) -> tuple[Dict[DiseaseId, set[GeneId]], Dict[DiseaseId, Counter[str]], Counter[str]]:
    disease_to_genes: Dict[DiseaseId, set[GeneId]] = defaultdict(set)
    disease_to_edge_types: Dict[DiseaseId, Counter[str]] = defaultdict(Counter)
    global_edge_type_counter: Counter[str] = Counter()

    for edge_type, edge_index in data.edge_index_dict.items():
        src_type, _, dst_type = edge_type
        if edge_index.numel() == 0:
            continue
        if {'disease', 'gene/protein'} != {src_type, dst_type}:
            continue

        edge_name = '|'.join(edge_type)
        edge_index_cpu = edge_index.detach().cpu().to(torch.long)
        src_global = data[src_type].global_id.detach().cpu().to(torch.long)[edge_index_cpu[0]]
        dst_global = data[dst_type].global_id.detach().cpu().to(torch.long)[edge_index_cpu[1]]

        if src_type == 'disease' and dst_type == 'gene/protein':
            disease_ids = src_global.tolist()
            gene_ids = dst_global.tolist()
        elif src_type == 'gene/protein' and dst_type == 'disease':
            disease_ids = dst_global.tolist()
            gene_ids = src_global.tolist()
        else:
            continue

        for disease_id, gene_id in zip(disease_ids, gene_ids):
            disease_int = int(disease_id)
            disease_to_genes[disease_int].add(int(gene_id))
            disease_to_edge_types[disease_int][edge_name] += 1
            global_edge_type_counter[edge_name] += 1

    return disease_to_genes, disease_to_edge_types, global_edge_type_counter


def summarize_island_diseases(
    test_disease_ids: Sequence[DiseaseId],
    raw_disease_to_genes: Mapping[DiseaseId, set[GeneId]],
    clean_disease_to_genes: Mapping[DiseaseId, set[GeneId]],
    raw_disease_to_edge_types: Mapping[DiseaseId, Counter[str]],
    processor: PrimeKGDataProcessor,
    max_print: int,
) -> Dict[str, Any]:
    island_disease_ids = sorted({int(d) for d in test_disease_ids if len(clean_disease_to_genes.get(int(d), set())) == 0})
    originally_missing = [d for d in island_disease_ids if len(raw_disease_to_genes.get(d, set())) == 0]
    removed_by_clean = [d for d in island_disease_ids if len(raw_disease_to_genes.get(d, set())) > 0]

    removed_edge_type_counter: Counter[str] = Counter()
    removed_examples: List[Dict[str, Any]] = []
    for disease_id in removed_by_clean:
        removed_edge_type_counter.update(raw_disease_to_edge_types.get(disease_id, Counter()))
        if len(removed_examples) < max_print:
            removed_examples.append({
                'disease_global_id': int(disease_id),
                'disease_raw_id': processor.id2entity[disease_id].raw_id,
                'disease_name': processor.id2entity[disease_id].name,
                'raw_gene_degree': len(raw_disease_to_genes.get(disease_id, set())),
                'clean_gene_degree': len(clean_disease_to_genes.get(disease_id, set())),
                'raw_edge_types': dict(raw_disease_to_edge_types.get(disease_id, Counter())),
            })

    island_examples: List[Dict[str, Any]] = []
    for disease_id in island_disease_ids[:max_print]:
        island_examples.append({
            'disease_global_id': int(disease_id),
            'disease_raw_id': processor.id2entity[disease_id].raw_id,
            'disease_name': processor.id2entity[disease_id].name,
            'raw_gene_degree': len(raw_disease_to_genes.get(disease_id, set())),
            'clean_gene_degree': len(clean_disease_to_genes.get(disease_id, set())),
            'raw_edge_types': dict(raw_disease_to_edge_types.get(disease_id, Counter())),
        })

    total_islands = len(island_disease_ids)
    return {
        'num_test_diseases': len(sorted(set(int(d) for d in test_disease_ids))),
        'island_disease_ids': island_disease_ids,
        'num_island_diseases': total_islands,
        'pct_island_diseases': float(total_islands / max(len(sorted(set(int(d) for d in test_disease_ids))), 1)),
        'num_originally_missing': len(originally_missing),
        'pct_originally_missing_among_islands': float(len(originally_missing) / max(total_islands, 1)),
        'num_removed_by_clean': len(removed_by_clean),
        'pct_removed_by_clean_among_islands': float(len(removed_by_clean) / max(total_islands, 1)),
        'removed_edge_type_counter': dict(removed_edge_type_counter),
        'island_examples': island_examples,
        'removed_examples': removed_examples,
    }


def print_report(report: Mapping[str, Any]) -> None:
    print('# Disease-Gene Loss Audit')
    print()
    print('## Graph Summary')
    print(f"- split_mode: {report['split_mode']}")
    print(f"- total_removed_direct_edges: {report['total_removed_edges']}")
    print(f"- leakage_edge_summary: {report['leakage_edge_summary']}")
    print()
    print('## Island Disease Audit')
    print(f"- test diseases: {report['summary']['num_test_diseases']}")
    print(f"- island diseases (clean gene degree = 0): {report['summary']['num_island_diseases']} ({report['summary']['pct_island_diseases']:.2%})")
    print(f"- originally had no gene edges in raw graph: {report['summary']['num_originally_missing']} ({report['summary']['pct_originally_missing_among_islands']:.2%} of islands)")
    print(f"- originally had gene edges, but clean graph lost them: {report['summary']['num_removed_by_clean']} ({report['summary']['pct_removed_by_clean_among_islands']:.2%} of islands)")
    print()
    print('## Raw Disease-Gene Edge Types Among Removed-by-Clean Diseases')
    if report['summary']['removed_edge_type_counter']:
        for edge_name, count in sorted(report['summary']['removed_edge_type_counter'].items(), key=lambda kv: (-kv[1], kv[0])):
            print(f'- {edge_name}: {count}')
    else:
        print('- None')
    print()
    print('## Example Island Diseases')
    for item in report['summary']['island_examples']:
        print(f"- {item['disease_raw_id']} | {item['disease_name']} | raw_gene_degree={item['raw_gene_degree']} | clean_gene_degree={item['clean_gene_degree']} | raw_edge_types={item['raw_edge_types']}")
    print()
    if report['summary']['removed_examples']:
        print('## Example Diseases Potentially Affected by Cleaning')
        for item in report['summary']['removed_examples']:
            print(f"- {item['disease_raw_id']} | {item['disease_name']} | raw_gene_degree={item['raw_gene_degree']} | clean_gene_degree={item['clean_gene_degree']} | raw_edge_types={item['raw_edge_types']}")
        print()


def main() -> None:
    args = parse_args()
    processor, pair_splits, raw_data, clean_data, total_removed_edges, leakage_edge_summary = load_graphs(
        processed_path=args.processed_path,
        nodes_csv=args.nodes_csv,
        edges_csv=args.edges_csv,
    )

    raw_disease_to_genes, raw_disease_to_edge_types, raw_edge_type_counter = collect_disease_gene_adjacency(raw_data)
    clean_disease_to_genes, clean_disease_to_edge_types, clean_edge_type_counter = collect_disease_gene_adjacency(clean_data)

    test_disease_ids = [int(disease_id) for _, disease_id in pair_splits['test']]
    summary = summarize_island_diseases(
        test_disease_ids=test_disease_ids,
        raw_disease_to_genes=raw_disease_to_genes,
        clean_disease_to_genes=clean_disease_to_genes,
        raw_disease_to_edge_types=raw_disease_to_edge_types,
        processor=processor,
        max_print=args.print_max_diseases,
    )

    report = {
        'split_mode': 'cross_disease',
        'processed_path': str(args.processed_path),
        'nodes_csv': str(args.nodes_csv),
        'edges_csv': str(args.edges_csv),
        'total_removed_edges': int(total_removed_edges),
        'leakage_edge_summary': leakage_edge_summary,
        'raw_disease_gene_edge_types': dict(raw_edge_type_counter),
        'clean_disease_gene_edge_types': dict(clean_edge_type_counter),
        'summary': summary,
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print_report(report)
    print(f'Saved report to: {args.output_json}')


if __name__ == '__main__':
    main()
