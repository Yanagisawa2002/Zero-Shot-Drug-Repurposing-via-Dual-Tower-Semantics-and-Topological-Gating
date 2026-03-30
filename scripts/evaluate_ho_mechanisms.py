from __future__ import annotations

import argparse
import copy
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import DefaultDict, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import pandas as pd
import torch
from sklearn.metrics import average_precision_score
from torch import Tensor
from torch_geometric.data import HeteroData

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation_utils import _extract_x_dict, _infer_model_device
from src.feature_utils import inject_features_to_graph
from src.primekg_data_processor import PrimeKGDataProcessor
from src.repurposing_rgcn import EmbeddingDict, EdgeType, RepurposingRGCN


PathQuad = Tuple[int, int, int, int]
MetricDict = Dict[str, float]
NodeIdPools = Dict[str, Tuple[int, ...]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Evaluate quad-level HO mechanisms with 1:4 full-node corruption.'
    )
    parser.add_argument('--checkpoint-path', type=Path, required=True)
    parser.add_argument('--positive-paths', type=Path, required=True)
    parser.add_argument('--output-json', type=Path, required=True)
    parser.add_argument('--nodes-csv', type=Path, default=Path('data/PrimeKG/nodes.csv'))
    parser.add_argument('--edges-csv', type=Path, default=Path('data/PrimeKG/edges.csv'))
    parser.add_argument(
        '--reference-edges-csv',
        type=Path,
        default=None,
        help='???????????????????? --edges-csv ???',
    )
    parser.add_argument('--feature-dir', type=Path, default=None)
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--in-channels', type=int, default=768)
    parser.add_argument('--hidden-channels', type=int, default=128)
    parser.add_argument('--out-dim', type=int, default=128)
    parser.add_argument('--scorer-hidden-dim', type=int, default=128)
    parser.add_argument('--scorer-output-hidden-dim', type=int, default=None)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--pathway-node-type', type=str, default='pathway')
    parser.add_argument('--pathway-dummy-global-id', type=int, default=0)
    parser.add_argument('--max-sampling-attempts', type=int, default=256)
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cpu', 'cuda'])
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()


def build_primekg_graph(
    node_csv_path: Path,
    edge_csv_path: Path,
    feature_dir: Optional[Path] = None,
    ho_attr_name: str = 'ho_pos_paths',
) -> HeteroData:
    """
    ??????? PrimeKG ????

    ????? HO ??????? `(drug, gene/protein, pathway, disease)`?
    ?????????????????????????????
    """

    processor = PrimeKGDataProcessor(node_csv_path=node_csv_path, edge_csv_path=edge_csv_path)
    processor.build_entity_mappings()
    data = processor.build_heterodata(
        ho_id_paths=[],
        ho_type_order=('drug', 'gene/protein', 'pathway', 'disease'),
        add_inverse_edges=False,
        ho_attr_name=ho_attr_name,
    )
    if feature_dir is not None:
        inject_features_to_graph(data=data, feature_dir=feature_dir)
    return data


def load_positive_paths(path: Path) -> Tensor:
    """
    ???? HO ???

    ???
    - `.pt/.pth`?????? Tensor???? `paths/test_paths/test_ho_paths/ho_pos_paths` ???
    - `.csv/.tsv`????? `drug/gene/pathway/disease`?`*_id`?`*_global_id`
    - `.json`????????? key ???
    """

    suffix = path.suffix.lower()
    if suffix in {'.pt', '.pth'}:
        payload = torch.load(path, map_location='cpu')
        tensor = _extract_path_tensor_from_payload(payload)
    elif suffix in {'.csv', '.tsv'}:
        sep = '\t' if suffix == '.tsv' else ','
        frame = pd.read_csv(path, sep=sep)
        tensor = _extract_path_tensor_from_dataframe(frame)
    elif suffix == '.json':
        payload = json.loads(path.read_text(encoding='utf-8'))
        tensor = _extract_path_tensor_from_payload(payload)
    else:
        raise ValueError(f'??????????: {path}')

    if tensor.dim() != 2 or tensor.size(1) not in {3, 4}:
        raise ValueError(
            '`positive_paths` ??? `(N, 3)` ? `(N, 4)` ??????'
            f'???? {tuple(tensor.shape)}?'
        )
    return tensor.detach().cpu().to(torch.long).contiguous()


def _extract_path_tensor_from_payload(payload: object) -> Tensor:
    if isinstance(payload, Tensor):
        return payload

    if isinstance(payload, dict):
        for candidate_key in ('paths', 'test_paths', 'test_ho_paths', 'ho_pos_paths', 'positive_paths'):
            if candidate_key in payload:
                candidate_value = payload[candidate_key]
                if isinstance(candidate_value, Tensor):
                    return candidate_value
                return torch.tensor(candidate_value, dtype=torch.long)

    if isinstance(payload, list):
        return torch.tensor(payload, dtype=torch.long)

    raise TypeError(f'????? {type(payload)} ????????')


def _extract_path_tensor_from_dataframe(frame: pd.DataFrame) -> Tensor:
    candidate_column_groups = [
        ['drug', 'gene', 'pathway', 'disease'],
        ['drug_id', 'gene_id', 'pathway_id', 'disease_id'],
        ['drug_global_id', 'gene_global_id', 'pathway_global_id', 'disease_global_id'],
        ['drug', 'gene', 'disease'],
        ['drug_id', 'gene_id', 'disease_id'],
        ['drug_global_id', 'gene_global_id', 'disease_global_id'],
    ]

    for columns in candidate_column_groups:
        if all(column in frame.columns for column in columns):
            return torch.tensor(frame[columns].to_numpy(), dtype=torch.long)

    integer_columns = [column for column in frame.columns if pd.api.types.is_integer_dtype(frame[column])]
    if len(integer_columns) >= 4:
        return torch.tensor(frame[integer_columns[:4]].to_numpy(), dtype=torch.long)
    if len(integer_columns) >= 3:
        return torch.tensor(frame[integer_columns[:3]].to_numpy(), dtype=torch.long)

    raise ValueError(
        'CSV/TSV ???????? HO ????'
        '??? `drug/gene/pathway/disease` ? `*_id` ???'
    )


def build_gene_to_pathways_from_graph(
    data: HeteroData,
    gene_node_type: str = 'gene/protein',
    pathway_node_type: str = 'pathway',
) -> Dict[int, Tuple[int, ...]]:
    """?????? `gene -> pathways` ????????"""

    gene_to_pathways_buffer: DefaultDict[int, List[int]] = defaultdict(list)

    for edge_type, edge_index in data.edge_index_dict.items():
        src_type, _, dst_type = edge_type
        if {src_type, dst_type} != {gene_node_type, pathway_node_type}:
            continue
        if edge_index.numel() == 0:
            continue

        edge_index_cpu = edge_index.detach().cpu().to(torch.long)
        src_global_ids = data[src_type].global_id.detach().cpu()[edge_index_cpu[0]]
        dst_global_ids = data[dst_type].global_id.detach().cpu()[edge_index_cpu[1]]

        if src_type == gene_node_type and dst_type == pathway_node_type:
            gene_pathway_pairs = zip(src_global_ids.tolist(), dst_global_ids.tolist())
        else:
            gene_pathway_pairs = zip(dst_global_ids.tolist(), src_global_ids.tolist())

        for gene_id, pathway_id in gene_pathway_pairs:
            gene_to_pathways_buffer[int(gene_id)].append(int(pathway_id))

    return {
        gene_id: tuple(dict.fromkeys(pathway_ids))
        for gene_id, pathway_ids in gene_to_pathways_buffer.items()
    }


def expand_paths_to_quads(
    path_tensor: Tensor,
    gene_to_pathways: Mapping[int, Sequence[int]],
    pathway_dummy_global_id: int = 0,
) -> Tensor:
    """
    ???? `(d, g, c)` ?????? `(d, g, p, c)`?

    ???????????????????
    ??? gene ???? pathway ???? `Dummy Pathway ID`?
    """

    if path_tensor.dim() != 2 or path_tensor.size(1) not in {3, 4}:
        raise ValueError('`path_tensor` ??? `(N, 3)` ? `(N, 4)`?')

    if path_tensor.size(1) == 4:
        return path_tensor.detach().cpu().to(torch.long).contiguous()

    expanded_paths: List[PathQuad] = []
    for drug_id, gene_id, disease_id in path_tensor.detach().cpu().to(torch.long).tolist():
        pathway_ids = tuple(int(x) for x in gene_to_pathways.get(int(gene_id), ()))
        if not pathway_ids:
            pathway_ids = (int(pathway_dummy_global_id),)
        for pathway_id in pathway_ids:
            expanded_paths.append((int(drug_id), int(gene_id), int(pathway_id), int(disease_id)))

    if not expanded_paths:
        return torch.empty((0, 4), dtype=torch.long)
    return torch.tensor(expanded_paths, dtype=torch.long)


def build_full_real_mechanism_quads(
    data: HeteroData,
    gene_to_pathways: Mapping[int, Sequence[int]],
    drug_node_type: str = 'drug',
    gene_node_type: str = 'gene/protein',
    disease_node_type: str = 'disease',
    pathway_dummy_global_id: int = 0,
) -> Set[PathQuad]:
    """
    ? PrimeKG ???????????????????

    ??????? topology-aware path bank ???
    - ????? `drug <-> gene` ??
    - ????? `gene <-> disease` ??
    - ??? gene ??????
    - ?? gene ?? pathway??? `(pathway=0)`
    """

    gene_to_drugs: DefaultDict[int, Set[int]] = defaultdict(set)
    gene_to_diseases: DefaultDict[int, Set[int]] = defaultdict(set)

    for edge_type, edge_index in data.edge_index_dict.items():
        src_type, _, dst_type = edge_type
        if edge_index.numel() == 0:
            continue

        edge_index_cpu = edge_index.detach().cpu().to(torch.long)
        src_global_ids = data[src_type].global_id.detach().cpu()[edge_index_cpu[0]]
        dst_global_ids = data[dst_type].global_id.detach().cpu()[edge_index_cpu[1]]

        if src_type == drug_node_type and dst_type == gene_node_type:
            for drug_id, gene_id in zip(src_global_ids.tolist(), dst_global_ids.tolist()):
                gene_to_drugs[int(gene_id)].add(int(drug_id))
            continue

        if src_type == gene_node_type and dst_type == drug_node_type:
            for gene_id, drug_id in zip(src_global_ids.tolist(), dst_global_ids.tolist()):
                gene_to_drugs[int(gene_id)].add(int(drug_id))
            continue

        if src_type == gene_node_type and dst_type == disease_node_type:
            for gene_id, disease_id in zip(src_global_ids.tolist(), dst_global_ids.tolist()):
                gene_to_diseases[int(gene_id)].add(int(disease_id))
            continue

        if src_type == disease_node_type and dst_type == gene_node_type:
            for disease_id, gene_id in zip(src_global_ids.tolist(), dst_global_ids.tolist()):
                gene_to_diseases[int(gene_id)].add(int(disease_id))
            continue

    real_quads: Set[PathQuad] = set()
    shared_gene_ids = set(gene_to_drugs.keys()).intersection(gene_to_diseases.keys())
    for gene_id in shared_gene_ids:
        pathway_ids = tuple(int(x) for x in gene_to_pathways.get(int(gene_id), ()))
        if not pathway_ids:
            pathway_ids = (int(pathway_dummy_global_id),)

        for drug_id in gene_to_drugs[gene_id]:
            for disease_id in gene_to_diseases[gene_id]:
                for pathway_id in pathway_ids:
                    real_quads.add((int(drug_id), int(gene_id), int(pathway_id), int(disease_id)))

    return real_quads


def collect_global_node_id_pools(
    data: HeteroData,
    pathway_node_type: str = 'pathway',
) -> NodeIdPools:
    """????????????????? ID ??"""

    return {
        'drug': tuple(int(x) for x in data['drug'].global_id.detach().cpu().tolist()),
        'gene/protein': tuple(int(x) for x in data['gene/protein'].global_id.detach().cpu().tolist()),
        'pathway': tuple(int(x) for x in data[pathway_node_type].global_id.detach().cpu().tolist()),
        'disease': tuple(int(x) for x in data['disease'].global_id.detach().cpu().tolist()),
    }


def sample_full_node_corruption_negatives(
    positive_quads: Tensor,
    node_id_pools: NodeIdPools,
    real_quad_set: Set[PathQuad],
    max_sampling_attempts: int = 256,
    generator: Optional[torch.Generator] = None,
) -> Tensor:
    """
    ?????????? 4 ????Drug/Gene/Pathway/Disease ??????

    ??????????? `real_quad_set` ??
    ????? `(N, 4, 4)` ???????????????
    0=Drug Corruption, 1=Gene Corruption, 2=Pathway Corruption, 3=Disease Corruption?
    """

    if positive_quads.dim() != 2 or positive_quads.size(1) != 4:
        raise ValueError('`positive_quads` ??? `(N, 4)`?')

    corruption_specs = [
        (0, node_id_pools['drug']),
        (1, node_id_pools['gene/protein']),
        (2, node_id_pools['pathway']),
        (3, node_id_pools['disease']),
    ]

    negative_groups: List[List[PathQuad]] = []
    for positive_quad in positive_quads.detach().cpu().to(torch.long).tolist():
        positive_tuple = tuple(int(x) for x in positive_quad)
        negative_group: List[PathQuad] = []

        for corrupt_position, candidate_pool in corruption_specs:
            sampled_negative = _sample_one_corrupted_quad(
                positive_quad=positive_tuple,
                corrupt_position=corrupt_position,
                candidate_pool=candidate_pool,
                real_quad_set=real_quad_set,
                max_sampling_attempts=max_sampling_attempts,
                generator=generator,
            )
            negative_group.append(sampled_negative)

        negative_groups.append(negative_group)

    return torch.tensor(negative_groups, dtype=torch.long)


def _sample_one_corrupted_quad(
    positive_quad: PathQuad,
    corrupt_position: int,
    candidate_pool: Sequence[int],
    real_quad_set: Set[PathQuad],
    max_sampling_attempts: int,
    generator: Optional[torch.Generator],
) -> PathQuad:
    current_value = int(positive_quad[corrupt_position])
    if not candidate_pool:
        raise RuntimeError('??????????????')

    for _ in range(max_sampling_attempts):
        sampled_index = torch.randint(
            low=0,
            high=len(candidate_pool),
            size=(1,),
            generator=generator,
        ).item()
        sampled_value = int(candidate_pool[sampled_index])
        if sampled_value == current_value:
            continue

        candidate_quad = list(positive_quad)
        candidate_quad[corrupt_position] = sampled_value
        candidate_tuple = tuple(candidate_quad)
        if candidate_tuple not in real_quad_set:
            return candidate_tuple  # type: ignore[return-value]

    for sampled_value in candidate_pool:
        sampled_value = int(sampled_value)
        if sampled_value == current_value:
            continue
        candidate_quad = list(positive_quad)
        candidate_quad[corrupt_position] = sampled_value
        candidate_tuple = tuple(candidate_quad)
        if candidate_tuple not in real_quad_set:
            return candidate_tuple  # type: ignore[return-value]

    raise RuntimeError(
        '???????????????????????????????????'
        f' positive_quad={positive_quad}, corrupt_position={corrupt_position}'
    )


def build_full_node_corruption_benchmark(
    positive_quads: Tensor,
    node_id_pools: NodeIdPools,
    real_quad_set: Set[PathQuad],
    max_sampling_attempts: int = 256,
    generator: Optional[torch.Generator] = None,
) -> Tuple[Tensor, Tensor]:
    """
    ?? 1:4 ????????

    ???
    - grouped_paths: `(N, 5, 4)`?????? `[pos, neg_d, neg_g, neg_p, neg_c]`
    - grouped_labels: `(N, 5)`?????? `[1, 0, 0, 0, 0]`
    """

    negative_quads = sample_full_node_corruption_negatives(
        positive_quads=positive_quads,
        node_id_pools=node_id_pools,
        real_quad_set=real_quad_set,
        max_sampling_attempts=max_sampling_attempts,
        generator=generator,
    )
    grouped_paths = torch.cat([positive_quads.unsqueeze(1), negative_quads], dim=1)
    grouped_labels = torch.zeros((positive_quads.size(0), 5), dtype=torch.long)
    grouped_labels[:, 0] = 1
    return grouped_paths, grouped_labels


def score_single_path_groups(
    model: RepurposingRGCN,
    data: HeteroData,
    grouped_paths: Tensor,
    batch_size: int = 512,
) -> Tensor:
    """
    ? `(N, 5, 4)` ????????? K=1 ???

    ???????????
    - `pair_ids`: `(drug, disease)`
    - `paths`: `(1, 4)`
    - `attention_mask=True`

    ???? `(N, 5)` ??????
    """

    if grouped_paths.dim() != 3 or grouped_paths.size(1) != 5 or grouped_paths.size(2) != 4:
        raise ValueError('`grouped_paths` ??? `(N, 5, 4)`?')

    was_training = model.training
    model.eval()
    device = _infer_model_device(model)
    graph_data = copy.deepcopy(data).to(device)

    try:
        with torch.no_grad():
            x_dict = _extract_x_dict(full_graph_data=graph_data)
            edge_index_dict: Mapping[EdgeType, Tensor] = graph_data.edge_index_dict
            node_embs_dict: EmbeddingDict = model.encode(
                x_dict=x_dict,
                edge_index_dict=edge_index_dict,
            )

            score_chunks: List[Tensor] = []
            grouped_paths_cpu = grouped_paths.detach().cpu().to(torch.long)
            num_groups = grouped_paths_cpu.size(0)
            for start in range(0, num_groups, batch_size):
                end = min(start + batch_size, num_groups)
                path_group_batch = grouped_paths_cpu[start:end].to(device)
                flat_paths = path_group_batch.view(-1, 1, 4)
                pair_ids = torch.stack(
                    [flat_paths[:, 0, 0], flat_paths[:, 0, 3]],
                    dim=1,
                )
                attention_mask = torch.ones(
                    (flat_paths.size(0), 1),
                    dtype=torch.bool,
                    device=device,
                )
                logits = model.score_batch(
                    node_embs_dict=node_embs_dict,
                    pair_ids=pair_ids,
                    paths=flat_paths,
                    attention_mask=attention_mask,
                )
                score_chunks.append(logits.detach().cpu().view(-1, 5))
    finally:
        if was_training:
            model.train()

    if not score_chunks:
        raise ValueError('??????????????')
    return torch.cat(score_chunks, dim=0)


def compute_imbalanced_metrics(group_scores: Tensor, grouped_labels: Tensor) -> MetricDict:
    """
    ?? 1:4 ??????????

    - AUPRC????????-????
    - Hit@1??? 5 ? 1 ????????????
    - MRR???? 5 ???????????
    """

    if group_scores.shape != grouped_labels.shape:
        raise ValueError(
            '`group_scores` ? `grouped_labels` ???????'
            f'???? {tuple(group_scores.shape)} vs {tuple(grouped_labels.shape)}?'
        )

    y_score = group_scores.reshape(-1).detach().cpu().numpy()
    y_true = grouped_labels.reshape(-1).detach().cpu().numpy()
    auprc = float(average_precision_score(y_true, y_score))

    sorted_indices = torch.argsort(group_scores, dim=1, descending=True)
    positive_rank_positions = (sorted_indices == 0).nonzero(as_tuple=False)[:, 1]
    hit_at_1 = float((positive_rank_positions == 0).float().mean().item())
    mrr = float((1.0 / (positive_rank_positions.to(torch.float32) + 1.0)).mean().item())

    return {
        'auprc': auprc,
        'hit_at_1': hit_at_1,
        'mrr': mrr,
        'num_groups': float(group_scores.size(0)),
        'num_candidates': float(group_scores.numel()),
    }


def evaluate_ho_mechanisms(
    model: RepurposingRGCN,
    eval_data: HeteroData,
    positive_quads: Tensor,
    reference_data: Optional[HeteroData] = None,
    pathway_node_type: str = 'pathway',
    pathway_dummy_global_id: int = 0,
    batch_size: int = 512,
    max_sampling_attempts: int = 256,
    seed: int = 42,
) -> Dict[str, object]:
    """
    ????? HO ????????

    `reference_data` ???????????????????? `eval_data`?
    ???????????????????????clean graph ?? + full graph ???????
    """

    reference_graph = reference_data if reference_data is not None else eval_data
    gene_to_pathways = build_gene_to_pathways_from_graph(
        data=reference_graph,
        pathway_node_type=pathway_node_type,
    )
    quad_tensor = expand_paths_to_quads(
        path_tensor=positive_quads,
        gene_to_pathways=gene_to_pathways,
        pathway_dummy_global_id=pathway_dummy_global_id,
    )
    real_quad_set = build_full_real_mechanism_quads(
        data=reference_graph,
        gene_to_pathways=gene_to_pathways,
        pathway_dummy_global_id=pathway_dummy_global_id,
    )
    node_id_pools = collect_global_node_id_pools(
        data=reference_graph,
        pathway_node_type=pathway_node_type,
    )

    generator = torch.Generator(device='cpu')
    generator.manual_seed(int(seed))
    grouped_paths, grouped_labels = build_full_node_corruption_benchmark(
        positive_quads=quad_tensor,
        node_id_pools=node_id_pools,
        real_quad_set=real_quad_set,
        max_sampling_attempts=max_sampling_attempts,
        generator=generator,
    )
    group_scores = score_single_path_groups(
        model=model,
        data=eval_data,
        grouped_paths=grouped_paths,
        batch_size=batch_size,
    )
    metrics = compute_imbalanced_metrics(
        group_scores=group_scores,
        grouped_labels=grouped_labels,
    )

    return {
        'metrics': metrics,
        'grouped_paths': grouped_paths,
        'grouped_labels': grouped_labels,
        'group_scores': group_scores,
        'num_real_reference_quads': len(real_quad_set),
    }


def load_model_from_checkpoint(
    checkpoint_path: Path,
    data: HeteroData,
    device: torch.device,
    in_channels: int = 768,
    hidden_channels: int = 128,
    out_dim: int = 128,
    scorer_hidden_dim: int = 128,
    scorer_output_hidden_dim: Optional[int] = None,
    dropout: float = 0.1,
    use_pathway_quads: bool = True,
    strict: bool = True,
) -> RepurposingRGCN:
    """
    ????????? `RepurposingRGCN`?

    ???? checkpoint?
    - ????? `state_dict`
    - ?? `model_state_dict` / `model_config` ???
    """

    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model_config: Dict[str, object] = {}

    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        if not isinstance(state_dict, dict):
            raise TypeError('`model_state_dict` ??????')
        maybe_config = checkpoint.get('model_config', {})
        if isinstance(maybe_config, dict):
            model_config = dict(maybe_config)
    elif isinstance(checkpoint, dict) and checkpoint and all(
        isinstance(value, Tensor) for value in checkpoint.values()
    ):
        state_dict = checkpoint
    else:
        raise TypeError(
            '???? checkpoint ?????? `state_dict` ??? `model_state_dict` ????'
        )

    cleaned_state_dict = _strip_module_prefix_from_state_dict(state_dict)
    resolved_hidden_channels = int(model_config.get('hidden_channels', model_config.get('hidden_dim', hidden_channels)))
    resolved_out_dim = int(model_config.get('out_dim', out_dim))
    resolved_scorer_hidden_dim = int(model_config.get('scorer_hidden_dim', scorer_hidden_dim))
    resolved_scorer_output_hidden_dim = model_config.get('scorer_output_hidden_dim', scorer_output_hidden_dim)
    if resolved_scorer_output_hidden_dim is not None:
        resolved_scorer_output_hidden_dim = int(resolved_scorer_output_hidden_dim)
    resolved_dropout = float(model_config.get('dropout', dropout))
    resolved_in_channels = int(model_config.get('in_channels', in_channels))
    resolved_use_pathway_quads = bool(model_config.get('use_pathway_quads', use_pathway_quads))

    model = RepurposingRGCN(
        data=data,
        in_channels=resolved_in_channels,
        hidden_channels=resolved_hidden_channels,
        out_dim=resolved_out_dim,
        scorer_hidden_dim=resolved_scorer_hidden_dim,
        scorer_output_hidden_dim=resolved_scorer_output_hidden_dim,
        dropout=resolved_dropout,
        use_pathway_quads=resolved_use_pathway_quads,
    ).to(device)
    model.load_state_dict(cleaned_state_dict, strict=strict)
    model.eval()
    return model


def _strip_module_prefix_from_state_dict(state_dict: Mapping[str, Tensor]) -> Dict[str, Tensor]:
    cleaned_state_dict: Dict[str, Tensor] = {}
    for key, value in state_dict.items():
        cleaned_key = key[7:] if key.startswith('module.') else key
        cleaned_state_dict[cleaned_key] = value
    return cleaned_state_dict


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == 'cpu':
        return torch.device('cpu')
    if device_arg == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('????? CUDA??????????')
        return torch.device('cuda')
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def summarize_results_for_json(results: Dict[str, object]) -> Dict[str, object]:
    grouped_paths = results['grouped_paths']
    grouped_labels = results['grouped_labels']
    group_scores = results['group_scores']
    assert isinstance(grouped_paths, Tensor)
    assert isinstance(grouped_labels, Tensor)
    assert isinstance(group_scores, Tensor)

    preview_count = min(5, grouped_paths.size(0))
    preview = []
    for group_index in range(preview_count):
        preview.append(
            {
                'paths': grouped_paths[group_index].tolist(),
                'labels': grouped_labels[group_index].tolist(),
                'scores': [float(x) for x in group_scores[group_index].tolist()],
            }
        )

    return {
        'metrics': results['metrics'],
        'num_real_reference_quads': results['num_real_reference_quads'],
        'preview_groups': preview,
    }


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    device = resolve_device(args.device)
    reference_edges_csv = args.reference_edges_csv or args.edges_csv

    eval_data = build_primekg_graph(
        node_csv_path=args.nodes_csv,
        edge_csv_path=args.edges_csv,
        feature_dir=args.feature_dir,
    )
    reference_data = build_primekg_graph(
        node_csv_path=args.nodes_csv,
        edge_csv_path=reference_edges_csv,
        feature_dir=None,
    )

    positive_paths = load_positive_paths(args.positive_paths)
    model = load_model_from_checkpoint(
        checkpoint_path=args.checkpoint_path,
        data=eval_data,
        device=device,
        in_channels=args.in_channels,
        hidden_channels=args.hidden_channels,
        out_dim=args.out_dim,
        scorer_hidden_dim=args.scorer_hidden_dim,
        scorer_output_hidden_dim=args.scorer_output_hidden_dim,
        dropout=args.dropout,
        use_pathway_quads=True,
    )

    results = evaluate_ho_mechanisms(
        model=model,
        eval_data=eval_data,
        positive_quads=positive_paths,
        reference_data=reference_data,
        pathway_node_type=args.pathway_node_type,
        pathway_dummy_global_id=args.pathway_dummy_global_id,
        batch_size=args.batch_size,
        max_sampling_attempts=args.max_sampling_attempts,
        seed=args.seed,
    )

    payload = {
        'config': {
            'checkpoint_path': str(args.checkpoint_path),
            'positive_paths': str(args.positive_paths),
            'nodes_csv': str(args.nodes_csv),
            'edges_csv': str(args.edges_csv),
            'reference_edges_csv': str(reference_edges_csv),
            'feature_dir': None if args.feature_dir is None else str(args.feature_dir),
            'batch_size': args.batch_size,
            'seed': args.seed,
        },
        **summarize_results_for_json(results),
    }
    args.output_json.write_text(json.dumps(payload, indent=2), encoding='utf-8')

    metrics = payload['metrics']
    print('HO Mechanism Evaluation Results:')
    print(f"  AUPRC = {metrics['auprc']:.4f}")
    print(f"  Hit@1 = {metrics['hit_at_1']:.4f}")
    print(f"  MRR   = {metrics['mrr']:.4f}")
    print(f"  Num Groups = {int(metrics['num_groups'])}")
    print(f"Saved results to: {args.output_json}")


if __name__ == '__main__':
    main()
