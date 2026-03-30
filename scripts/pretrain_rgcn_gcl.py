from __future__ import annotations

import argparse
import copy
import json
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch_geometric.data import HeteroData

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.feature_utils import inject_features_to_graph
from src.primekg_data_processor import PrimeKGDataProcessor
from src.repurposing_rgcn import EdgeType, EmbeddingDict, RepurposingRGCN


class ProjectionHead(nn.Module):
    """????????? MLP ????"""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class GCL_RGCN_Model(nn.Module):
    """
    ?????????

    ???
    - encoder: ??? RepurposingRGCN??????????
    - projection_heads: ?????????????? encoder ?????????
    """

    def __init__(
        self,
        encoder: RepurposingRGCN,
        target_node_types: Sequence[str],
        projection_hidden_dim: int,
        projection_dim: int,
        projection_dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.target_node_types = tuple(target_node_types)
        self.projection_heads = nn.ModuleDict(
            {
                node_type: ProjectionHead(
                    in_dim=self.encoder.out_dim,
                    hidden_dim=projection_hidden_dim,
                    out_dim=projection_dim,
                    dropout=projection_dropout,
                )
                for node_type in self.target_node_types
            }
        )

    def encode_view(self, graph_view: HeteroData) -> Tuple[EmbeddingDict, EmbeddingDict]:
        """???????????? encoder ????????"""

        x_dict: Optional[Mapping[str, Tensor]] = graph_view.x_dict if graph_view.x_dict else None
        node_embs_dict = self.encoder.encode(
            x_dict=x_dict,
            edge_index_dict=graph_view.edge_index_dict,
        )
        projected_dict: EmbeddingDict = {
            node_type: self.projection_heads[node_type](node_embs_dict[node_type])
            for node_type in self.target_node_types
        }
        return node_embs_dict, projected_dict

    def forward(
        self,
        view_1: HeteroData,
        view_2: HeteroData,
    ) -> Tuple[EmbeddingDict, EmbeddingDict]:
        """????????????????????"""

        _, proj_1 = self.encode_view(view_1)
        _, proj_2 = self.encode_view(view_2)
        return proj_1, proj_2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Pre-train RepurposingRGCN with graph contrastive learning.'
    )
    parser.add_argument('--nodes-csv', type=Path, default=Path('data/PrimeKG/nodes.csv'))
    parser.add_argument('--edges-csv', type=Path, default=Path('data/PrimeKG/edges.csv'))
    parser.add_argument(
        '--feature-dir',
        type=Path,
        default=Path('outputs/pubmedbert_hybrid_features_clean'),
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('outputs/pretrain_rgcn_gcl'),
    )
    parser.add_argument('--save-name', type=str, default='pretrained_rgcn.pth')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=1e-5)
    parser.add_argument('--hidden-channels', type=int, default=128)
    parser.add_argument('--out-dim', type=int, default=128)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--initial-residual-alpha', type=float, default=0.2)
    parser.add_argument('--encoder-type', type=str, default='rgcn', choices=['rgcn', 'mlp'])
    parser.add_argument('--projection-hidden-dim', type=int, default=128)
    parser.add_argument('--projection-dim', type=int, default=128)
    parser.add_argument('--projection-dropout', type=float, default=0.1)
    parser.add_argument('--edge-drop-prob', type=float, default=0.2)
    parser.add_argument('--feature-mask-prob', type=float, default=0.1)
    parser.add_argument('--temperature', type=float, default=0.2)
    parser.add_argument('--contrastive-batch-size', type=int, default=4096)
    parser.add_argument('--target-node-types', type=str, default='drug,disease')
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_target_node_types(raw_value: str) -> Tuple[str, ...]:
    node_types = tuple(part.strip() for part in raw_value.split(',') if part.strip())
    if not node_types:
        raise ValueError('`target_node_types` must contain at least one node type.')
    return node_types


def build_full_graph(
    node_csv_path: Path,
    edge_csv_path: Path,
    feature_dir: Path,
) -> Tuple[PrimeKGDataProcessor, HeteroData]:
    """???????????????????????"""

    processor = PrimeKGDataProcessor(node_csv_path=node_csv_path, edge_csv_path=edge_csv_path)
    processor.build_entity_mappings()
    data = processor.build_heterodata(ho_id_paths=[], add_inverse_edges=False)
    inject_features_to_graph(data=data, feature_dir=feature_dir)
    return processor, data


def augment_graph_view(
    data: HeteroData,
    edge_drop_prob: float,
    feature_mask_prob: float,
    generator: Optional[torch.Generator] = None,
) -> HeteroData:
    """
    ?????????
    1. ????
    2. ?????????????
    """

    if not 0.0 <= edge_drop_prob < 1.0:
        raise ValueError('`edge_drop_prob` must be in [0, 1).')
    if not 0.0 <= feature_mask_prob < 1.0:
        raise ValueError('`feature_mask_prob` must be in [0, 1).')

    augmented = copy.deepcopy(data)

    for edge_type, edge_index in data.edge_index_dict.items():
        num_edges = int(edge_index.size(1))
        if num_edges == 0:
            continue

        keep_mask = torch.rand(num_edges, generator=generator) > edge_drop_prob
        if not bool(keep_mask.any()):
            keep_mask[torch.randint(0, num_edges, (1,), generator=generator).item()] = True
        augmented[edge_type].edge_index = edge_index[:, keep_mask].clone()

    for node_type in data.node_types:
        if 'x' not in data[node_type]:
            continue
        x = data[node_type].x.clone()
        num_nodes = int(x.size(0))
        if num_nodes == 0:
            augmented[node_type].x = x
            continue

        node_mask = torch.rand(num_nodes, generator=generator) < feature_mask_prob
        if bool(node_mask.any()):
            x[node_mask] = 0.0
        augmented[node_type].x = x

    return augmented


def sample_node_indices(
    num_nodes: int,
    batch_size: int,
    device: torch.device,
) -> Tensor:
    """??????????????? mini-batch?"""

    if batch_size <= 0 or batch_size >= num_nodes:
        return torch.arange(num_nodes, device=device, dtype=torch.long)
    return torch.randperm(num_nodes, device=device)[:batch_size]


def info_nce_loss(
    z_view_1: Tensor,
    z_view_2: Tensor,
    temperature: float = 0.2,
) -> Tensor:
    """
    ???? InfoNCE ???

    ??????????????????
    ????batch ???????????????
    """

    if z_view_1.size(0) != z_view_2.size(0):
        raise ValueError('Positive pairs must have the same batch size in two views.')
    if z_view_1.size(0) == 0:
        raise ValueError('InfoNCE requires at least one positive pair.')
    if temperature <= 0.0:
        raise ValueError('`temperature` must be positive.')

    z_1 = F.normalize(z_view_1, p=2, dim=-1)
    z_2 = F.normalize(z_view_2, p=2, dim=-1)
    labels = torch.arange(z_1.size(0), device=z_1.device)

    logits_12 = torch.matmul(z_1, z_2.transpose(0, 1)) / temperature
    logits_21 = torch.matmul(z_2, z_1.transpose(0, 1)) / temperature

    loss_12 = F.cross_entropy(logits_12, labels)
    loss_21 = F.cross_entropy(logits_21, labels)
    return 0.5 * (loss_12 + loss_21)


def compute_contrastive_objective(
    proj_view_1: Mapping[str, Tensor],
    proj_view_2: Mapping[str, Tensor],
    target_node_types: Sequence[str],
    contrastive_batch_size: int,
    temperature: float,
) -> Tuple[Tensor, Dict[str, float]]:
    """????????? InfoNCE ???"""

    losses: List[Tensor] = []
    metrics: Dict[str, float] = {}

    for node_type in target_node_types:
        if node_type not in proj_view_1 or node_type not in proj_view_2:
            raise KeyError(f'Missing projected embeddings for node type `{node_type}`.')

        z_1_full = proj_view_1[node_type]
        z_2_full = proj_view_2[node_type]
        sampled_indices = sample_node_indices(
            num_nodes=int(z_1_full.size(0)),
            batch_size=contrastive_batch_size,
            device=z_1_full.device,
        )
        z_1 = z_1_full[sampled_indices]
        z_2 = z_2_full[sampled_indices]
        node_loss = info_nce_loss(z_1, z_2, temperature=temperature)
        losses.append(node_loss)
        metrics[f'{node_type}_loss'] = float(node_loss.detach().cpu().item())
        metrics[f'{node_type}_batch_nodes'] = float(sampled_indices.numel())

    total_loss = torch.stack(losses).mean()
    metrics['contrastive_loss'] = float(total_loss.detach().cpu().item())
    return total_loss, metrics


def move_view_to_device(graph_view: HeteroData, device: torch.device) -> HeteroData:
    return graph_view.to(device)


def pretrain_one_epoch(
    model: GCL_RGCN_Model,
    full_graph: HeteroData,
    optimizer: torch.optim.Optimizer,
    edge_drop_prob: float,
    feature_mask_prob: float,
    contrastive_batch_size: int,
    temperature: float,
    device: torch.device,
) -> Dict[str, float]:
    model.train()
    optimizer.zero_grad(set_to_none=True)

    view_1 = augment_graph_view(
        data=full_graph,
        edge_drop_prob=edge_drop_prob,
        feature_mask_prob=feature_mask_prob,
    )
    view_2 = augment_graph_view(
        data=full_graph,
        edge_drop_prob=edge_drop_prob,
        feature_mask_prob=feature_mask_prob,
    )

    view_1 = move_view_to_device(view_1, device=device)
    view_2 = move_view_to_device(view_2, device=device)

    proj_view_1, proj_view_2 = model(view_1=view_1, view_2=view_2)
    total_loss, metrics = compute_contrastive_objective(
        proj_view_1=proj_view_1,
        proj_view_2=proj_view_2,
        target_node_types=model.target_node_types,
        contrastive_batch_size=contrastive_batch_size,
        temperature=temperature,
    )

    total_loss.backward()
    optimizer.step()
    return metrics


def main() -> None:
    args = parse_args()
    set_random_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    target_node_types = parse_target_node_types(args.target_node_types)

    _, full_graph = build_full_graph(
        node_csv_path=args.nodes_csv,
        edge_csv_path=args.edges_csv,
        feature_dir=args.feature_dir,
    )

    missing_targets = [node_type for node_type in target_node_types if node_type not in full_graph.node_types]
    if missing_targets:
        raise KeyError(f'Target node types not found in graph: {missing_targets}')

    encoder = RepurposingRGCN(
        data=full_graph,
        in_channels=768,
        hidden_channels=args.hidden_channels,
        out_dim=args.out_dim,
        scorer_hidden_dim=args.out_dim,
        dropout=args.dropout,
        initial_residual_alpha=args.initial_residual_alpha,
        encoder_type=args.encoder_type,
        agg_type='attention',
        use_pathway_quads=False,
    )
    model = GCL_RGCN_Model(
        encoder=encoder,
        target_node_types=target_node_types,
        projection_hidden_dim=args.projection_hidden_dim,
        projection_dim=args.projection_dim,
        projection_dropout=args.projection_dropout,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    history: List[Dict[str, float]] = []
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        metrics = pretrain_one_epoch(
            model=model,
            full_graph=full_graph,
            optimizer=optimizer,
            edge_drop_prob=args.edge_drop_prob,
            feature_mask_prob=args.feature_mask_prob,
            contrastive_batch_size=args.contrastive_batch_size,
            temperature=args.temperature,
            device=device,
        )
        metrics['epoch'] = float(epoch)
        history.append(metrics)
        print(
            f"epoch={epoch:03d} "
            f"loss={metrics['contrastive_loss']:.4f} "
            + ' '.join(
                f"{node_type}_loss={metrics[f'{node_type}_loss']:.4f}"
                for node_type in target_node_types
            )
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    encoder_path = args.output_dir / args.save_name
    torch.save(model.encoder.state_dict(), encoder_path)

    metrics_path = args.output_dir / 'pretrain_metrics.json'
    payload = {
        'config': {
            'nodes_csv': str(args.nodes_csv),
            'edges_csv': str(args.edges_csv),
            'feature_dir': str(args.feature_dir),
            'epochs': args.epochs,
            'lr': args.lr,
            'weight_decay': args.weight_decay,
            'hidden_channels': args.hidden_channels,
            'out_dim': args.out_dim,
            'dropout': args.dropout,
            'initial_residual_alpha': args.initial_residual_alpha,
            'encoder_type': args.encoder_type,
            'projection_hidden_dim': args.projection_hidden_dim,
            'projection_dim': args.projection_dim,
            'projection_dropout': args.projection_dropout,
            'edge_drop_prob': args.edge_drop_prob,
            'feature_mask_prob': args.feature_mask_prob,
            'temperature': args.temperature,
            'contrastive_batch_size': args.contrastive_batch_size,
            'target_node_types': list(target_node_types),
            'device': str(device),
            'save_name': args.save_name,
        },
        'num_nodes_by_type': {
            node_type: int(full_graph[node_type].num_nodes)
            for node_type in full_graph.node_types
        },
        'total_time_sec': time.time() - start_time,
        'history': history,
        'encoder_path': str(encoder_path),
    }
    metrics_path.write_text(json.dumps(payload, indent=2), encoding='utf-8')

    print('Saved encoder weights to:', encoder_path)
    print('Saved metrics to:', metrics_path)


if __name__ == '__main__':
    main()
