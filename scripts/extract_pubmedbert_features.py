from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer


PUBMEDBERT_MODEL_NAME = 'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext'


@dataclass(frozen=True)
class EntityTextRecord:
    """?? PrimeKG ??????????????"""

    global_id: int
    local_id: int
    node_type: str
    raw_id: str
    name: str
    text: str
    source: str = ''


class EntityTextDataset(Dataset[EntityTextRecord]):
    """????????????????"""

    def __init__(self, records: Sequence[EntityTextRecord]) -> None:
        self.records: List[EntityTextRecord] = list(records)
        if not self.records:
            raise ValueError('????????????? PubMedBERT ???')

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> EntityTextRecord:
        return self.records[index]


def collate_entity_batch(batch: Sequence[EntityTextRecord]) -> Dict[str, Any]:
    """
    ?????????? tokenizer / ???????????

    ?????? `global_ids`?`local_ids` ? `node_types`?
    ????? batch ????? embedding ?????????????????
    """

    if not batch:
        raise ValueError('? batch ?????????')

    return {
        'texts': [record.text for record in batch],
        'global_ids': torch.tensor([record.global_id for record in batch], dtype=torch.long),
        'local_ids': torch.tensor([record.local_id for record in batch], dtype=torch.long),
        'node_types': [record.node_type for record in batch],
        'raw_ids': [record.raw_id for record in batch],
        'names': [record.name for record in batch],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='?? PubMedBERT ? PrimeKG ????????????'
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        '--nodes-csv',
        type=Path,
        help='PrimeKG ? nodes.csv ??????????????',
    )
    input_group.add_argument(
        '--entity-map-path',
        type=Path,
        help=(
            '????????????????? .json / .pt?'
            '????? list[dict] ? {node_type: {entity_text: global_id}}?'
        ),
    )

    parser.add_argument(
        '--output-dir',
        type=Path,
        required=True,
        help='?????????????????????? .pt ?????',
    )
    parser.add_argument(
        '--model-name',
        type=str,
        default=PUBMEDBERT_MODEL_NAME,
        help='transformers ????????? PubMedBERT?',
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=64,
        help='????????GPU ???????????',
    )
    parser.add_argument(
        '--max-length',
        type=int,
        default=64,
        help='tokenizer ?????',
    )
    parser.add_argument(
        '--pooling',
        type=str,
        choices=('mean', 'cls'),
        default='mean',
        help='????????mean ? cls?',
    )
    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        help='?????? auto / cuda / cuda:0 / cpu?',
    )
    parser.add_argument(
        '--num-workers',
        type=int,
        default=0,
        help='DataLoader ? worker ???Windows ???? 0 ???',
    )
    parser.add_argument(
        '--no-type-prefix',
        action='store_true',
        help='?????????????????????????????',
    )
    return parser.parse_args()


def load_entity_records(
    nodes_csv: Optional[Path],
    entity_map_path: Optional[Path],
    use_type_prefix: bool,
) -> List[EntityTextRecord]:
    """?????????????????"""

    if nodes_csv is not None:
        return load_entity_records_from_nodes_csv(
            nodes_csv_path=nodes_csv,
            use_type_prefix=use_type_prefix,
        )
    if entity_map_path is not None:
        return load_entity_records_from_entity_map(
            entity_map_path=entity_map_path,
            use_type_prefix=use_type_prefix,
        )
    raise ValueError('???? `--nodes-csv` ? `--entity-map-path` ???')


def load_entity_records_from_nodes_csv(
    nodes_csv_path: Path,
    use_type_prefix: bool,
) -> List[EntityTextRecord]:
    """
    ? PrimeKG ? `nodes.csv` ?????

    ????????? `global_id`???? `PrimeKGDataProcessor.build_entity_mappings()`
    ???????????????? `local_id` ?????????
    """

    if not nodes_csv_path.exists():
        raise FileNotFoundError(f'??? nodes.csv: {nodes_csv_path}')

    records: List[EntityTextRecord] = []
    local_id_by_type: Dict[str, int] = {}

    with nodes_csv_path.open('r', encoding='utf-8-sig', newline='') as file:
        reader = csv.DictReader(file)
        required_columns = {'id', 'type', 'name', 'source'}
        missing_columns = required_columns.difference(reader.fieldnames or [])
        if missing_columns:
            raise KeyError(f'nodes.csv ??????{sorted(missing_columns)}')

        for global_id, row in enumerate(reader):
            node_type = row['type'].strip()
            raw_id = row['id'].strip()
            name = row['name'].strip()
            source = row['source'].strip()
            local_id = local_id_by_type.get(node_type, 0)
            local_id_by_type[node_type] = local_id + 1

            records.append(
                EntityTextRecord(
                    global_id=global_id,
                    local_id=local_id,
                    node_type=node_type,
                    raw_id=raw_id,
                    name=name,
                    text=build_entity_text(
                        name=name,
                        node_type=node_type,
                        use_type_prefix=use_type_prefix,
                    ),
                    source=source,
                )
            )

    return records


def load_entity_records_from_entity_map(
    entity_map_path: Path,
    use_type_prefix: bool,
) -> List[EntityTextRecord]:
    """
    ??????? entity map ????????

    ?????????
    1. `list[dict]`????????? `global_id`?`node_type`??? `text` ? `name`?
    2. `dict[node_type][entity_text] = global_id`????????? entity2id ???
    """

    payload = load_serialized_object(entity_map_path)

    if isinstance(payload, list):
        return parse_record_list_payload(payload=payload, use_type_prefix=use_type_prefix)
    if isinstance(payload, dict):
        return parse_typed_entity_map_payload(payload=payload, use_type_prefix=use_type_prefix)

    raise TypeError(
        '???? entity map ?????? list[dict] ? {node_type: {entity_text: global_id}}?'
    )


def parse_record_list_payload(
    payload: Sequence[Any],
    use_type_prefix: bool,
) -> List[EntityTextRecord]:
    """?? `list[dict]` ????????"""

    raw_records: List[Dict[str, Any]] = []
    for item in payload:
        if not isinstance(item, Mapping):
            raise TypeError('record list ?????????? dict-like ???')
        raw_records.append(dict(item))

    raw_records.sort(key=lambda item: int(item['global_id']))
    local_id_by_type: Dict[str, int] = {}
    records: List[EntityTextRecord] = []

    for item in raw_records:
        if 'global_id' not in item or 'node_type' not in item:
            raise KeyError('record list ??????????? `global_id` ? `node_type`?')

        node_type = str(item['node_type']).strip()
        name = str(item.get('name', item.get('text', item.get('entity_text', '')))).strip()
        text = str(item.get('text', '')).strip()
        raw_id = str(item.get('raw_id', name)).strip()
        source = str(item.get('source', '')).strip()
        global_id = int(item['global_id'])

        if not name and not text:
            raise ValueError('record list ??????????? `name` ? `text`?')

        local_id = int(item['local_id']) if 'local_id' in item else local_id_by_type.get(node_type, 0)
        if 'local_id' not in item:
            local_id_by_type[node_type] = local_id + 1

        if not text:
            text = build_entity_text(
                name=name,
                node_type=node_type,
                use_type_prefix=use_type_prefix,
            )

        records.append(
            EntityTextRecord(
                global_id=global_id,
                local_id=local_id,
                node_type=node_type,
                raw_id=raw_id,
                name=name or text,
                text=text,
                source=source,
            )
        )

    return records


def parse_typed_entity_map_payload(
    payload: Mapping[str, Any],
    use_type_prefix: bool,
) -> List[EntityTextRecord]:
    """
    ?? `{node_type: {entity_text: global_id}}` ??? entity2id?

    ???
    - ????? `global_id` ????????????????????????????????
    - ???????? `local_id` ? `global_id` ??????
    """

    grouped_candidates: Dict[str, Dict[int, List[str]]] = {}
    for node_type, typed_mapping in payload.items():
        if not isinstance(typed_mapping, Mapping):
            raise TypeError(
                'typed entity map ???? `{node_type: {entity_text: global_id}}` ????'
            )

        grouped_candidates[node_type] = {}
        for entity_text, global_id in typed_mapping.items():
            int_global_id = int(global_id)
            grouped_candidates[node_type].setdefault(int_global_id, []).append(str(entity_text))

    records: List[EntityTextRecord] = []
    for node_type, candidates_by_id in grouped_candidates.items():
        for local_id, global_id in enumerate(sorted(candidates_by_id.keys())):
            chosen_name = select_best_entity_text(
                candidates=candidates_by_id[global_id],
                node_type=node_type,
            )
            records.append(
                EntityTextRecord(
                    global_id=global_id,
                    local_id=local_id,
                    node_type=node_type,
                    raw_id=chosen_name,
                    name=chosen_name,
                    text=build_entity_text(
                        name=chosen_name,
                        node_type=node_type,
                        use_type_prefix=use_type_prefix,
                    ),
                    source='',
                )
            )

    records.sort(key=lambda record: record.global_id)
    return records


def select_best_entity_text(candidates: Sequence[str], node_type: str) -> str:
    """
    ???????????????????????

    ??????
    1. ??????? `::` ????
    2. ?????????????
    3. ???????????????
    """

    if not candidates:
        raise ValueError(f'???? `{node_type}` ???????????')

    normalized_candidates = [candidate.strip() for candidate in candidates if candidate.strip()]
    if not normalized_candidates:
        raise ValueError(f'???? `{node_type}` ?????????????')

    def score(alias: str) -> tuple[int, int, int]:
        contains_prefix = int('::' in alias)
        is_numeric = int(alias.replace('.', '').isdigit())
        return (contains_prefix, is_numeric, -len(alias))

    return min(normalized_candidates, key=score)


def build_entity_text(name: str, node_type: str, use_type_prefix: bool) -> str:
    """???? PubMedBERT ??????"""

    clean_name = name.strip()
    if not clean_name:
        raise ValueError('????????????')
    if not use_type_prefix:
        return clean_name
    return f'{node_type}: {clean_name}'


def load_serialized_object(path: Path) -> Any:
    """?? .json ? .pt ??????"""

    if not path.exists():
        raise FileNotFoundError(f'????????{path}')

    suffix = path.suffix.lower()
    if suffix == '.json':
        with path.open('r', encoding='utf-8') as file:
            return json.load(file)
    if suffix in {'.pt', '.pth'}:
        return torch.load(path, map_location='cpu')
    raise ValueError(f'?????????{path.suffix}???? .json / .pt / .pth?')


def resolve_device(device_arg: str) -> torch.device:
    """????????????????"""

    if device_arg == 'auto':
        if torch.cuda.is_available():
            return torch.device('cuda')
        return torch.device('cpu')

    device = torch.device(device_arg)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('????? CUDA??????????')
    return device


def extract_features(
    texts: Sequence[str],
    tokenizer: AutoTokenizer,
    model: AutoModel,
    device: torch.device,
    max_length: int,
    pooling: str,
) -> Tensor:
    """
    ????? batch ?? PubMedBERT ???

    ??????? `(batch_size, hidden_size)`?
    ????? PubMedBERT-base?`hidden_size = 768`?
    """

    encoded_inputs = tokenizer(
        list(texts),
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors='pt',
    )
    encoded_inputs = {
        key: value.to(device=device, non_blocking=True)
        for key, value in encoded_inputs.items()
    }

    with torch.no_grad():
        model_outputs = model(**encoded_inputs)
        last_hidden_state = model_outputs.last_hidden_state

        if pooling == 'cls':
            pooled_features = last_hidden_state[:, 0, :]
        elif pooling == 'mean':
            attention_mask = encoded_inputs['attention_mask'].unsqueeze(-1).to(last_hidden_state.dtype)
            masked_hidden_state = last_hidden_state * attention_mask
            summed_hidden_state = masked_hidden_state.sum(dim=1)
            token_counts = attention_mask.sum(dim=1).clamp(min=1.0)
            pooled_features = summed_hidden_state / token_counts
        else:
            raise ValueError(f'???? pooling ???{pooling}')

    return pooled_features.detach().cpu().to(torch.float32)


def sanitize_node_type_for_filename(node_type: str) -> str:
    """??????????????????????"""

    return node_type.replace('/', '_').replace(' ', '_').replace('-', '_')


def validate_record_id_layout(records: Sequence[EntityTextRecord]) -> None:
    """?? global/local ID ?????????"""

    if not records:
        raise ValueError('??????????? ID ???')

    sorted_global_ids = sorted(record.global_id for record in records)
    expected_global_ids = list(range(len(records)))
    if sorted_global_ids != expected_global_ids:
        raise ValueError(
            '??? global_id ???? 0 ??????????'
            f'????/?? ID ? {sorted_global_ids[0]} / {sorted_global_ids[-1]}?'
            f'???? {len(records)}?'
        )

    node_types = sorted({record.node_type for record in records})
    for node_type in node_types:
        local_ids = sorted(record.local_id for record in records if record.node_type == node_type)
        expected_local_ids = list(range(len(local_ids)))
        if local_ids != expected_local_ids:
            raise ValueError(
                f'???? `{node_type}` ? local_id ??? 0 ???????'
            )


def initialize_feature_buffers(
    records: Sequence[EntityTextRecord],
    hidden_size: int,
) -> tuple[Tensor, Dict[str, Tensor]]:
    """??????????????????"""

    validate_record_id_layout(records=records)

    all_features = torch.zeros((len(records), hidden_size), dtype=torch.float32)

    type_counts: Dict[str, int] = {}
    for record in records:
        type_counts[record.node_type] = max(type_counts.get(record.node_type, 0), record.local_id + 1)

    type_features = {
        node_type: torch.zeros((count, hidden_size), dtype=torch.float32)
        for node_type, count in type_counts.items()
    }
    return all_features, type_features


def build_metadata(
    records: Sequence[EntityTextRecord],
    model_name: str,
    pooling: str,
    hidden_size: int,
) -> Dict[str, Any]:
    """??????????????????"""

    type_counts: Dict[str, int] = {}
    for record in records:
        type_counts[record.node_type] = type_counts.get(record.node_type, 0) + 1

    return {
        'model_name': model_name,
        'pooling': pooling,
        'hidden_size': hidden_size,
        'num_nodes': len(records),
        'max_global_id': max(record.global_id for record in records),
        'node_type_counts': type_counts,
    }


def save_feature_tensors(
    output_dir: Path,
    all_features: Tensor,
    type_features: Mapping[str, Tensor],
    metadata: Mapping[str, Any],
) -> None:
    """?????????????????"""

    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(all_features, output_dir / 'all_node_features.pt')

    for node_type, feature_tensor in type_features.items():
        safe_node_type = sanitize_node_type_for_filename(node_type)
        torch.save(feature_tensor, output_dir / f'{safe_node_type}_features.pt')

    metadata_path = output_dir / 'feature_metadata.json'
    with metadata_path.open('w', encoding='utf-8') as file:
        json.dump(dict(metadata), file, ensure_ascii=False, indent=2)


def run_extraction(args: argparse.Namespace) -> None:
    """????? PubMedBERT ????????"""

    if args.batch_size <= 0:
        raise ValueError('`--batch-size` ???????')
    if args.max_length <= 0:
        raise ValueError('`--max-length` ???????')

    device = resolve_device(args.device)
    use_type_prefix = not args.no_type_prefix
    records = load_entity_records(
        nodes_csv=args.nodes_csv,
        entity_map_path=args.entity_map_path,
        use_type_prefix=use_type_prefix,
    )

    dataset = EntityTextDataset(records=records)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == 'cuda'),
        collate_fn=collate_entity_batch,
    )

    print(f'Loaded {len(records)} entities.')
    print(f'Loading tokenizer/model: {args.model_name}')
    print(f'Using device: {device}')

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModel.from_pretrained(args.model_name)
    model.to(device)
    model.eval()

    hidden_size = int(model.config.hidden_size)
    all_features, type_features = initialize_feature_buffers(records=records, hidden_size=hidden_size)

    for batch_index, batch in enumerate(dataloader, start=1):
        batch_features = extract_features(
            texts=batch['texts'],
            tokenizer=tokenizer,
            model=model,
            device=device,
            max_length=args.max_length,
            pooling=args.pooling,
        )
        global_ids = batch['global_ids']
        local_ids = batch['local_ids']
        node_types: Sequence[str] = batch['node_types']

        all_features[global_ids] = batch_features
        for row_index, node_type in enumerate(node_types):
            type_features[node_type][int(local_ids[row_index].item())] = batch_features[row_index]

        if batch_index % 10 == 0 or batch_index == len(dataloader):
            print(
                f'Processed batch {batch_index}/{len(dataloader)} '
                f'({int(global_ids[-1].item()) + 1}/{len(records)} entities).'
            )

    metadata = build_metadata(
        records=records,
        model_name=args.model_name,
        pooling=args.pooling,
        hidden_size=hidden_size,
    )
    save_feature_tensors(
        output_dir=args.output_dir,
        all_features=all_features,
        type_features=type_features,
        metadata=metadata,
    )

    print('Feature extraction finished.')
    print(f'All-node features: {args.output_dir / "all_node_features.pt"}')
    for node_type in sorted(type_features.keys()):
        safe_node_type = sanitize_node_type_for_filename(node_type)
        print(f'{node_type} features: {args.output_dir / f"{safe_node_type}_features.pt"}')


def main() -> None:
    args = parse_args()
    run_extraction(args)


if __name__ == '__main__':
    main()
