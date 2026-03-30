from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import DefaultDict, Dict, Iterable, List, Literal, Mapping, Optional, Sequence, Set, Tuple

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.process_open_targets import align_entities_to_primekg


TextStrategy = Literal['hybrid', 'name_only', 'graph_context_only', 'rich_text_only']


LEGACY_MODE_TO_STRATEGY: Dict[str, TextStrategy] = {
    'hybrid': 'hybrid',
    'name_only': 'name_only',
    'graph_context': 'graph_context_only',
    'ot_rich': 'rich_text_only',
}


@dataclass(frozen=True)
class EntityTextRecord:
    """供 `extract_pubmedbert_features.py` 使用的实体文本记录。"""

    global_id: int
    local_id: int
    node_type: str
    raw_id: str
    name: str
    text: str
    source: str


@dataclass(frozen=True)
class PrimeKGNodeRecord:
    """PrimeKG 节点表中的结构化记录。"""

    global_id: int
    local_id: int
    raw_id: str
    node_type: str
    name: str
    source: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='基于本地 PrimeKG + Open Targets 构建可选的 PubMedBERT rich-text 输入。'
    )
    parser.add_argument(
        '--nodes-csv',
        type=Path,
        default=Path('data/PrimeKG/nodes.csv'),
        help='PrimeKG nodes.csv 路径。',
    )
    parser.add_argument(
        '--edges-csv',
        type=Path,
        default=Path('data/PrimeKG/edges.csv'),
        help='PrimeKG edges.csv 路径。',
    )
    parser.add_argument(
        '--ot-phase-dir',
        type=Path,
        default=Path('open target'),
        help='本地 Open Targets phase xlsx 目录。',
    )
    parser.add_argument(
        '--ot-phase-glob',
        type=str,
        default='Open_target_Phase_*.xlsx',
        help='Open Targets phase 文件匹配模式。',
    )
    parser.add_argument(
        '--text-strategy',
        type=str,
        choices=('hybrid', 'name_only', 'graph_context_only', 'rich_text_only'),
        default='hybrid',
        help='??????????? hybrid?',
    )
    parser.add_argument(
        '--mode',
        type=str,
        choices=('name_only', 'ot_rich', 'graph_context', 'hybrid'),
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        '--max-neighbors',
        type=int,
        default=8,
        help='graph-context 模式下每类关系保留的邻居上限。',
    )
    parser.add_argument(
        '--output-json',
        type=Path,
        required=True,
        help='输出的 entity_text_records.json 路径。',
    )
    parser.add_argument(
        '--report-json',
        type=Path,
        default=None,
        help='可选的覆盖率报告输出路径；默认写到 output-json 同目录。',
    )
    parser.add_argument(
        '--no-type-prefix',
        action='store_true',
        help='若设置，则基础文本不再带 node_type 前缀。',
    )
    return parser.parse_args()


def resolve_text_strategy(args: argparse.Namespace) -> TextStrategy:
    """????????????? `--text-strategy`?"""

    legacy_mode = getattr(args, 'mode', None)
    text_strategy = args.text_strategy

    if legacy_mode is None:
        return text_strategy

    legacy_strategy = LEGACY_MODE_TO_STRATEGY[legacy_mode]
    if text_strategy != 'hybrid' and text_strategy != legacy_strategy:
        raise ValueError(
            f'`--text-strategy={text_strategy}` ????? `--mode={legacy_mode}` ???'
        )
    return legacy_strategy


def load_primekg_node_records(nodes_csv: Path) -> List[PrimeKGNodeRecord]:
    """读取 PrimeKG 节点表，并恢复全局 / 分类型局部索引。"""

    records: List[PrimeKGNodeRecord] = []
    local_id_by_type: Dict[str, int] = defaultdict(int)

    with nodes_csv.open('r', encoding='utf-8-sig', newline='') as file:
        reader = csv.DictReader(file)
        required_columns = {'id', 'type', 'name', 'source'}
        missing_columns = required_columns.difference(reader.fieldnames or [])
        if missing_columns:
            raise KeyError(f'nodes.csv 缺少必要列：{sorted(missing_columns)}')

        for global_id, row in enumerate(reader):
            node_type = row['type'].strip()
            local_id = local_id_by_type[node_type]
            local_id_by_type[node_type] += 1
            records.append(
                PrimeKGNodeRecord(
                    global_id=global_id,
                    local_id=local_id,
                    raw_id=row['id'].strip(),
                    node_type=node_type,
                    name=row['name'].strip(),
                    source=row['source'].strip(),
                )
            )
    return records


def build_ot_rich_text_by_raw_id(
    nodes_csv: Path,
    ot_phase_dir: Path,
    ot_phase_glob: str,
) -> Dict[str, str]:
    """从本地 Open Targets phase xlsx 中抽取 rich text，并对齐到 PrimeKG raw id。"""

    phase_paths = sorted(ot_phase_dir.glob(ot_phase_glob))
    if not phase_paths:
        return {}

    phase_frames: List[pd.DataFrame] = []
    for path in phase_paths:
        table = pd.read_excel(path)
        phase_match = re.search(r'phase[_\s]*(\d+)', path.name, re.I)
        phase = int(phase_match.group(1)) if phase_match else -1
        phase_frames.append(
            pd.DataFrame(
                {
                    'ot_drug_id': table['DrugBank_ID'],
                    'ot_target_id': table['targetId'],
                    'ot_disease_id': table['MONDO_ID'],
                    'ot_drug_name': table['prefName'],
                    'ot_target_name': table['approvedSymbol'],
                    'ot_disease_name': table['label'],
                    'approved_name': table['approvedName'],
                    'target_class': table['targetClass'],
                    'trade_names': table['tradeNames'],
                    'synonyms': table['synonyms'],
                    'drug_type': table['drugType'],
                    'mechanism_of_action': table['mechanismOfAction'],
                    'target_name': table['targetName'],
                    'ancestors': table['ancestors'],
                    'status': table['status'],
                    'phase': phase,
                }
            )
        )

    phase_df = pd.concat(phase_frames, ignore_index=True)
    phase_df = phase_df.dropna(subset=['ot_drug_id', 'ot_target_id', 'ot_disease_id']).copy()
    for column_name in ['ot_drug_id', 'ot_target_id', 'ot_disease_id']:
        phase_df[column_name] = phase_df[column_name].astype(str).str.strip()
    phase_df = phase_df.loc[(phase_df[['ot_drug_id', 'ot_target_id', 'ot_disease_id']] != '').all(axis=1)]
    phase_df = phase_df.sort_values(['phase']).drop_duplicates(
        subset=['ot_drug_id', 'ot_target_id', 'ot_disease_id'],
        keep='last',
    )

    aligned_df, _ = align_entities_to_primekg(
        triplets_df=phase_df,
        primekg_nodes_csv=nodes_csv,
    )
    if aligned_df.empty:
        return {}

    text_by_raw_id: Dict[str, str] = {}
    for raw_id, group in aligned_df.groupby('primekg_drug_id', dropna=True):
        text = build_drug_ot_text(group)
        if text:
            text_by_raw_id[str(raw_id)] = text
    for raw_id, group in aligned_df.groupby('primekg_target_id', dropna=True):
        text = build_target_ot_text(group)
        if text:
            text_by_raw_id[str(raw_id)] = text
    for raw_id, group in aligned_df.groupby('primekg_disease_id', dropna=True):
        text = build_disease_ot_text(group)
        if text:
            text_by_raw_id[str(raw_id)] = text
    return text_by_raw_id


def build_drug_ot_text(group: pd.DataFrame) -> str:
    trade_names = _collect_unique_chunks(group['trade_names'])
    synonyms = _collect_unique_chunks(group['synonyms'])
    drug_types = _collect_unique_chunks(group['drug_type'])
    moa = _collect_unique_chunks(group['mechanism_of_action'])
    statuses = _collect_unique_chunks(group['status'])
    max_phase = int(group['phase'].max()) if 'phase' in group.columns and not group['phase'].isna().all() else None

    segments: List[str] = []
    if trade_names:
        segments.append(f"trade names: {', '.join(trade_names[:8])}")
    if synonyms:
        segments.append(f"synonyms: {', '.join(synonyms[:12])}")
    if drug_types:
        segments.append(f"drug type: {', '.join(drug_types[:4])}")
    if moa:
        segments.append(f"mechanism of action: {'; '.join(moa[:4])}")
    if statuses:
        segments.append(f"open targets status: {', '.join(statuses[:4])}")
    if max_phase is not None and max_phase >= 0:
        segments.append(f"max clinical phase in open targets: {max_phase}")
    return '. '.join(segment for segment in segments if segment) + ('.' if segments else '')


def build_target_ot_text(group: pd.DataFrame) -> str:
    approved_names = _collect_unique_chunks(group['approved_name'])
    target_classes = _collect_unique_chunks(group['target_class'])
    target_names = _collect_unique_chunks(group['target_name'])

    segments: List[str] = []
    if approved_names:
        segments.append(f"approved names: {'; '.join(approved_names[:4])}")
    if target_names:
        segments.append(f"target names: {'; '.join(target_names[:4])}")
    if target_classes:
        segments.append(f"target classes: {', '.join(target_classes[:6])}")
    return '. '.join(segment for segment in segments if segment) + ('.' if segments else '')


def build_disease_ot_text(group: pd.DataFrame) -> str:
    ancestors = _collect_unique_chunks(group['ancestors'])
    statuses = _collect_unique_chunks(group['status'])

    segments: List[str] = []
    if ancestors:
        segments.append(f"ancestors: {'; '.join(ancestors[:4])}")
    if statuses:
        segments.append(f"open targets statuses: {', '.join(statuses[:4])}")
    return '. '.join(segment for segment in segments if segment) + ('.' if segments else '')


def build_graph_context_by_raw_id(
    node_records: Sequence[PrimeKGNodeRecord],
    edges_csv: Path,
    max_neighbors: int,
) -> Dict[str, str]:
    """? PrimeKG ??????? graph-context ???"""

    raw_id_to_name = {record.raw_id: record.name for record in node_records}
    context_buffer: DefaultDict[str, DefaultDict[str, List[str]]] = defaultdict(lambda: defaultdict(list))
    blacklisted_relations = {'indication', 'contraindication', 'off-label use'}

    with edges_csv.open('r', encoding='utf-8-sig', newline='') as file:
        reader = csv.DictReader(file)
        for row in reader:
            src_id = row['src_id'].strip()
            dst_id = row['dst_id'].strip()
            src_type = row['src_type'].strip()
            dst_type = row['dst_type'].strip()
            relation = row['rel'].strip()

            if relation in blacklisted_relations:
                # ??????? drug-disease ?????????????????
                continue

            src_name = raw_id_to_name.get(src_id)
            dst_name = raw_id_to_name.get(dst_id)
            if src_name is None or dst_name is None:
                continue

            if src_type == 'drug' and dst_type == 'gene/protein' and relation == 'targets':
                _append_unique_limited(context_buffer[src_id]['targets'], dst_name, max_neighbors)
                _append_unique_limited(context_buffer[dst_id]['targeted_by'], src_name, max_neighbors)
            elif src_type == 'disease' and dst_type == 'gene/protein' and relation == 'disease_protein':
                _append_unique_limited(context_buffer[src_id]['associated_genes'], dst_name, max_neighbors)
                _append_unique_limited(context_buffer[dst_id]['associated_diseases'], src_name, max_neighbors)
            elif src_type == 'gene/protein' and dst_type == 'gene/protein' and relation == 'protein_protein':
                _append_unique_limited(context_buffer[src_id]['interacts_with'], dst_name, max_neighbors)
            elif src_type == 'gene/protein' and dst_type == 'pathway' and relation == 'in_pathway':
                _append_unique_limited(context_buffer[src_id]['pathways'], dst_name, max_neighbors)
            elif src_type == 'pathway' and dst_type == 'pathway' and relation == 'pathway_pathway':
                _append_unique_limited(context_buffer[src_id]['related_pathways'], dst_name, max_neighbors)

    text_by_raw_id: Dict[str, str] = {}
    for record in node_records:
        relation_chunks = context_buffer.get(record.raw_id)
        if not relation_chunks:
            continue
        text = render_graph_context_text(relation_chunks)
        if text:
            text_by_raw_id[record.raw_id] = text
    return text_by_raw_id



def render_graph_context_text(relation_chunks: Mapping[str, Sequence[str]]) -> str:
    """??????????????????"""

    label_map = {
        'targets': 'targets',
        'targeted_by': 'targeted by drugs',
        'associated_genes': 'associated genes',
        'associated_diseases': 'associated diseases',
        'interacts_with': 'interacts with',
        'pathways': 'pathways',
        'related_pathways': 'related pathways',
    }
    segments: List[str] = []
    ordered_relation_keys = [
        'targets',
        'targeted_by',
        'associated_genes',
        'associated_diseases',
        'interacts_with',
        'pathways',
        'related_pathways',
    ]
    for relation_key in ordered_relation_keys:
        neighbors = relation_chunks.get(relation_key, [])
        if neighbors:
            segments.append(f"{label_map[relation_key]}: {', '.join(neighbors)}")
    return '. '.join(segment for segment in segments if segment) + ('.' if segments else '')



def build_name_phrase_index(
    node_records: Sequence[PrimeKGNodeRecord],
    node_type: str,
) -> Dict[str, Tuple[str, ...]]:
    """???????????????????????????"""

    phrases_by_prefix: DefaultDict[str, Set[str]] = defaultdict(set)
    for record in node_records:
        if record.node_type != node_type:
            continue
        token_tuple = tuple(_tokenize_phrase(record.name))
        if len(token_tuple) == 0:
            continue
        if len(token_tuple) == 1 and len(token_tuple[0]) <= 2:
            continue
        normalized_phrase = ' '.join(token_tuple)
        prefix = token_tuple[0][:4]
        if prefix:
            phrases_by_prefix[prefix].add(normalized_phrase)

    return {
        prefix: tuple(sorted(phrase_set, key=lambda item: (-len(item), item)))
        for prefix, phrase_set in phrases_by_prefix.items()
    }



def _tokenize_phrase(text: str) -> List[str]:
    return re.findall(r'[a-z0-9]+', str(text).casefold())



def contains_blocklisted_entity_name(
    text: str,
    phrase_index: Mapping[str, Sequence[str]],
) -> bool:
    if not text or not phrase_index:
        return False

    normalized_text = ' '.join(_tokenize_phrase(text))
    if not normalized_text:
        return False

    for token in normalized_text.split():
        prefix = token[:4]
        if not prefix:
            continue
        candidates = phrase_index.get(prefix)
        if not candidates:
            continue
        for candidate_text in candidates:
            if candidate_text and candidate_text in normalized_text:
                return True
    return False



def _split_text_segments(text: str) -> List[str]:
    segments = [segment.strip() for segment in re.split(r'\s*[.]\s*', text) if segment.strip()]
    return segments



def sanitize_auxiliary_text(
    node_type: str,
    text: str,
    drug_phrase_index: Mapping[str, Sequence[str]],
    disease_phrase_index: Mapping[str, Sequence[str]],
) -> str:
    """?? rich-text / graph-context ????????????????"""

    if not text:
        return ''

    if node_type in {'drug', 'gene/protein', 'pathway'}:
        forbidden_phrase_index = disease_phrase_index
    elif node_type == 'disease':
        forbidden_phrase_index = drug_phrase_index
    else:
        forbidden_phrase_index = {}

    if not forbidden_phrase_index:
        return text.strip()

    kept_segments: List[str] = []
    for segment in _split_text_segments(text):
        if contains_blocklisted_entity_name(segment, forbidden_phrase_index):
            continue
        kept_segments.append(segment)

    if not kept_segments:
        return ''

    return '. '.join(kept_segments) + '.'


def sanitize_base_text(
    node_type: str,
    text: str,
    drug_phrase_index: Mapping[str, Sequence[str]],
    disease_phrase_index: Mapping[str, Sequence[str]],
) -> str:
    """?????????????????? canonical name ?????????"""

    cleaned_text = sanitize_auxiliary_text(
        node_type=node_type,
        text=text,
        drug_phrase_index=drug_phrase_index,
        disease_phrase_index=disease_phrase_index,
    ).strip()
    if cleaned_text.endswith('.'):
        cleaned_text = cleaned_text[:-1].strip()
    if cleaned_text:
        return cleaned_text
    return f'{node_type} entity'



def build_entity_text_records(
    nodes_csv: Path,
    edges_csv: Path,
    text_strategy: TextStrategy,
    use_type_prefix: bool,
    max_neighbors: int,
    ot_phase_dir: Optional[Path],
    ot_phase_glob: str,
) -> tuple[List[EntityTextRecord], Dict[str, object]]:
    """???????? PubMedBERT ??????????"""

    node_records = load_primekg_node_records(nodes_csv)
    drug_phrase_index = build_name_phrase_index(node_records=node_records, node_type='drug')
    disease_phrase_index = build_name_phrase_index(node_records=node_records, node_type='disease')

    ot_text_by_raw_id: Dict[str, str] = {}
    graph_text_by_raw_id: Dict[str, str] = {}

    if text_strategy in {'rich_text_only', 'hybrid'} and ot_phase_dir is not None and ot_phase_dir.exists():
        ot_text_by_raw_id = build_ot_rich_text_by_raw_id(
            nodes_csv=nodes_csv,
            ot_phase_dir=ot_phase_dir,
            ot_phase_glob=ot_phase_glob,
        )
    if text_strategy in {'graph_context_only', 'hybrid'}:
        graph_text_by_raw_id = build_graph_context_by_raw_id(
            node_records=node_records,
            edges_csv=edges_csv,
            max_neighbors=max_neighbors,
        )

    records: List[EntityTextRecord] = []
    source_counter: DefaultDict[str, int] = defaultdict(int)
    for node in node_records:
        use_type_prefix_for_strategy = use_type_prefix if text_strategy == 'hybrid' else False
        base_text = sanitize_base_text(
            node_type=node.node_type,
            text=build_base_text(
                node.name,
                node.node_type,
                use_type_prefix=use_type_prefix_for_strategy,
            ),
            drug_phrase_index=drug_phrase_index,
            disease_phrase_index=disease_phrase_index,
        )
        ot_text = sanitize_auxiliary_text(
            node_type=node.node_type,
            text=ot_text_by_raw_id.get(node.raw_id, ''),
            drug_phrase_index=drug_phrase_index,
            disease_phrase_index=disease_phrase_index,
        )
        graph_text = sanitize_auxiliary_text(
            node_type=node.node_type,
            text=graph_text_by_raw_id.get(node.raw_id, ''),
            drug_phrase_index=drug_phrase_index,
            disease_phrase_index=disease_phrase_index,
        )
        final_text, source_tag = compose_final_text(
            text_strategy=text_strategy,
            base_text=base_text,
            ot_text=ot_text,
            graph_text=graph_text,
        )
        source_counter[source_tag] += 1
        records.append(
            EntityTextRecord(
                global_id=node.global_id,
                local_id=node.local_id,
                node_type=node.node_type,
                raw_id=node.raw_id,
                name=node.name,
                text=final_text,
                source=source_tag,
            )
        )

    report = build_text_report(
        node_records=node_records,
        text_strategy=text_strategy,
        ot_text_by_raw_id=ot_text_by_raw_id,
        graph_text_by_raw_id=graph_text_by_raw_id,
        source_counter=source_counter,
    )
    return records, report



def build_base_text(name: str, node_type: str, use_type_prefix: bool) -> str:
    clean_name = name.strip()
    return f'{node_type}: {clean_name}' if use_type_prefix else clean_name



def compose_final_text(
    text_strategy: TextStrategy,
    base_text: str,
    ot_text: str,
    graph_text: str,
) -> tuple[str, str]:
    if text_strategy == 'name_only':
        return base_text, 'name_only'
    if text_strategy == 'graph_context_only':
        return (
            (f'{base_text}. {graph_text}' if graph_text else base_text),
            ('name+graph_context' if graph_text else 'name_only_fallback'),
        )
    if text_strategy == 'rich_text_only':
        return (
            (f'{base_text}. {ot_text}' if ot_text else base_text),
            ('name+rich_text' if ot_text else 'name_only_fallback'),
        )

    segments = [base_text]
    source_parts = ['name']
    if ot_text:
        segments.append(ot_text)
        source_parts.append('rich_text')
    if graph_text:
        segments.append(graph_text)
        source_parts.append('graph_context')
    dedup_segments = list(dict.fromkeys(segment.strip() for segment in segments if segment.strip()))
    return ' '.join(dedup_segments), '+'.join(source_parts)


def build_text_report(
    node_records: Sequence[PrimeKGNodeRecord],
    text_strategy: TextStrategy,
    ot_text_by_raw_id: Mapping[str, str],
    graph_text_by_raw_id: Mapping[str, str],
    source_counter: Mapping[str, int],
) -> Dict[str, object]:
    total_nodes = len(node_records)
    coverage_by_type: Dict[str, Dict[str, float]] = {}
    records_by_type: DefaultDict[str, List[PrimeKGNodeRecord]] = defaultdict(list)
    for record in node_records:
        records_by_type[record.node_type].append(record)

    for node_type, typed_records in records_by_type.items():
        raw_ids = [record.raw_id for record in typed_records]
        ot_hits = sum(raw_id in ot_text_by_raw_id for raw_id in raw_ids)
        graph_hits = sum(raw_id in graph_text_by_raw_id for raw_id in raw_ids)
        coverage_by_type[node_type] = {
            'count': len(typed_records),
            'ot_rich_coverage': ot_hits / max(len(typed_records), 1),
            'graph_context_coverage': graph_hits / max(len(typed_records), 1),
        }

    return {
        'text_strategy': text_strategy,
        'mode': text_strategy,
        'num_nodes': total_nodes,
        'ot_rich_num_nodes': len(ot_text_by_raw_id),
        'graph_context_num_nodes': len(graph_text_by_raw_id),
        'source_counter': dict(source_counter),
        'coverage_by_type': coverage_by_type,
    }


def _append_unique_limited(buffer: List[str], value: str, limit: int) -> None:
    cleaned = value.strip()
    if cleaned and cleaned not in buffer and len(buffer) < limit:
        buffer.append(cleaned)


def _collect_unique_chunks(series: pd.Series) -> List[str]:
    chunks: List[str] = []
    for value in series.tolist():
        if pd.isna(value):
            continue
        raw_text = str(value).strip()
        if not raw_text:
            continue
        normalized = raw_text.replace('\n', ' ').strip('[]')
        pieces = re.split(r';|\||,(?=\s*[A-Za-z0-9])', normalized)
        if len(pieces) == 1:
            pieces = re.split(r"'\s+'|\"\s+\"", normalized)
        for piece in pieces:
            cleaned = piece.strip().strip("'\"")
            cleaned = re.sub(r'\s+', ' ', cleaned)
            if not cleaned or cleaned.casefold() in {'nan', 'none'}:
                continue
            if cleaned not in chunks:
                chunks.append(cleaned)
    return chunks


def run_builder(args: argparse.Namespace) -> None:
    if args.max_neighbors <= 0:
        raise ValueError('`--max-neighbors` 必须为正整数。')

    text_strategy = resolve_text_strategy(args)
    records, report = build_entity_text_records(
        nodes_csv=args.nodes_csv,
        edges_csv=args.edges_csv,
        text_strategy=text_strategy,
        use_type_prefix=not args.no_type_prefix,
        max_neighbors=args.max_neighbors,
        ot_phase_dir=args.ot_phase_dir,
        ot_phase_glob=args.ot_phase_glob,
    )

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open('w', encoding='utf-8') as file:
        json.dump([asdict(record) for record in records], file, ensure_ascii=False, indent=2)

    report_path = args.report_json or args.output_json.with_name(args.output_json.stem + '_report.json')
    with report_path.open('w', encoding='utf-8') as file:
        json.dump(report, file, ensure_ascii=False, indent=2)

    print(f'Built {len(records)} entity text records.')
    print(f'Text strategy: {text_strategy}')
    print(f'Output JSON: {args.output_json}')
    print(f'Report JSON: {report_path}')
    print('Source counter:', json.dumps(report['source_counter'], ensure_ascii=False))


def main() -> None:
    args = parse_args()
    run_builder(args)


if __name__ == '__main__':
    main()
