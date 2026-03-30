from __future__ import annotations

import csv
import math
import pickle
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch_geometric.data import HeteroData
from torch_geometric.nn import HeteroConv, SAGEConv
from torch_geometric.utils import dropout_edge

from src.pair_aggregation_scorer import PairAggregationScorer


EdgeType = Tuple[str, str, str]
EmbeddingDict = Dict[str, Tensor]


class RepurposingRGCN(nn.Module):
    """
    Heterogeneous graph encoder + pair-level path scorer.

    Supported path schemas:
    - triplet mode: (drug, gene/protein, disease)
    - quad mode: (drug, gene/protein, pathway, disease)

    Key design points:
    - Optional PubMedBERT feature projection for each node type.
    - Optional MLP encoder ablation that skips message passing.
    - Initial residual connection inside each graph layer to reduce oversmoothing.
    - Dual-stream scorer that can use both graph-updated embeddings and raw embeddings.
    """

    def __init__(
        self,
        data: HeteroData,
        hidden_channels: Optional[int] = None,
        hidden_dim: Optional[int] = None,
        in_channels: int = 768,
        out_dim: Optional[int] = None,
        scorer_hidden_dim: Optional[int] = None,
        scorer_output_hidden_dim: Optional[int] = None,
        dropout: float = 0.1,
        conv_aggr: str = 'sum',
        encoder_type: str = 'rgcn',
        agg_type: str = 'attention',
        path_node_types: Sequence[str] = ('drug', 'gene/protein', 'disease'),
        use_pathway_quads: bool = False,
        pathway_node_type: str = 'pathway',
        pathway_dummy_global_id: int = 0,
        initial_residual_alpha: float = 0.2,
        triplet_text_embeddings_path: Optional[str | Path] = None,
        drug_morgan_fingerprints_path: Optional[str | Path] = None,
        drug_text_embeddings_path: Optional[str | Path] = None,
        disease_text_embeddings_path: Optional[str | Path] = None,
        nodes_csv_path: Optional[str | Path] = None,
        text_distill_alpha: float = 0.5,
        use_early_external_fusion: bool = False,
        dropedge_p: float = 0.15,
        ablate_gnn: bool = False,
    ) -> None:
        super().__init__()

        if hidden_channels is None and hidden_dim is None:
            raise ValueError('One of `hidden_channels` or `hidden_dim` must be provided.')
        if (
            hidden_channels is not None
            and hidden_dim is not None
            and int(hidden_channels) != int(hidden_dim)
        ):
            raise ValueError('`hidden_channels` and `hidden_dim` must match when both are set.')
        if encoder_type not in {'rgcn', 'mlp'}:
            raise ValueError("`encoder_type` must be either 'rgcn' or 'mlp'.")
        if agg_type not in {'attention', 'mean', 'max'}:
            raise ValueError("`agg_type` must be one of {'attention', 'mean', 'max'}." )
        if not 0.0 < float(initial_residual_alpha) < 1.0:
            raise ValueError('`initial_residual_alpha` must be strictly between 0 and 1.')
        if float(text_distill_alpha) < 0.0:
            raise ValueError('`text_distill_alpha` must be non-negative.')
        if not 0.0 <= float(dropedge_p) < 1.0:
            raise ValueError('`dropedge_p` must be in [0, 1).')

        resolved_hidden_channels = (
            int(hidden_channels) if hidden_channels is not None else int(hidden_dim)
        )
        self.node_types: List[str] = list(data.node_types)
        self.edge_types: List[EdgeType] = list(data.edge_types)
        self.hidden_channels = resolved_hidden_channels
        self.hidden_dim = resolved_hidden_channels
        self.in_channels = int(in_channels)
        self.out_dim = out_dim if out_dim is not None else resolved_hidden_channels
        self.dropout = float(dropout)
        self.encoder_type = encoder_type
        self.agg_type = agg_type
        self.use_pathway_quads = bool(use_pathway_quads)
        self.pathway_dummy_global_id = int(pathway_dummy_global_id)
        self.initial_residual_alpha = float(initial_residual_alpha)
        self.text_distill_alpha = float(text_distill_alpha)
        self.use_early_external_fusion = bool(use_early_external_fusion)
        self.dropedge_p = float(dropedge_p)
        self.ablate_gnn = bool(ablate_gnn)
        self.drug_fingerprint_dim = 1024
        self.disease_text_dim = 768
        self.triplet_text_embeddings_path = (
            None if triplet_text_embeddings_path is None else Path(triplet_text_embeddings_path)
        )
        self.drug_morgan_fingerprints_path = (
            None if drug_morgan_fingerprints_path is None else Path(drug_morgan_fingerprints_path)
        )
        self.drug_text_embeddings_path = (
            None if drug_text_embeddings_path is None else Path(drug_text_embeddings_path)
        )
        self.disease_text_embeddings_path = (
            None if disease_text_embeddings_path is None else Path(disease_text_embeddings_path)
        )
        self.nodes_csv_path = (
            Path('data/PrimeKG/nodes.csv')
            if nodes_csv_path is None
            and (
                triplet_text_embeddings_path is not None
                or drug_morgan_fingerprints_path is not None
                or drug_text_embeddings_path is not None
                or disease_text_embeddings_path is not None
            )
            else (None if nodes_csv_path is None else Path(nodes_csv_path))
        )

        base_path_node_types = tuple(getattr(data, 'ho_path_node_types', tuple(path_node_types)))
        self.path_node_types = self._resolve_path_node_types(
            base_path_node_types=base_path_node_types,
            fallback_pathway_node_type=pathway_node_type,
        )
        self.path_size = len(self.path_node_types)

        self.drug_node_type = self.path_node_types[0]
        self.gene_node_type = self.path_node_types[1]
        self.disease_node_type = self.path_node_types[-1]
        self.pathway_node_type = self.path_node_types[2] if self.use_pathway_quads else None
        self._validate_path_schema(data=data)

        self.node_embeddings = nn.ModuleDict(
            {
                node_type: nn.Embedding(int(data[node_type].num_nodes), self.hidden_channels)
                for node_type in self.node_types
            }
        )
        self.feature_projections = nn.ModuleDict(
            {
                node_type: nn.Linear(self.in_channels, self.hidden_channels)
                for node_type in self.node_types
            }
        )
        self.disease_proj = nn.Linear(self.disease_text_dim, self.hidden_channels)
        self.disease_external_norm = nn.LayerNorm(self.hidden_channels)
        self.encoder_output_projections = nn.ModuleDict(
            {
                node_type: (
                    nn.Identity()
                    if self.hidden_channels == self.out_dim
                    else nn.Linear(self.hidden_channels, self.out_dim)
                )
                for node_type in self.node_types
            }
        )

        self.initial_residual_projections_conv1 = nn.ModuleDict(
            {node_type: nn.Identity() for node_type in self.node_types}
        )
        self.initial_residual_projections_conv2 = nn.ModuleDict(
            {
                node_type: (
                    nn.Identity()
                    if self.hidden_channels == self.out_dim
                    else nn.Linear(self.hidden_channels, self.out_dim)
                )
                for node_type in self.node_types
            }
        )
        self.conv1_alpha_logit = nn.Parameter(torch.tensor(self._probability_to_logit(self.initial_residual_alpha)))
        self.conv2_alpha_logit = nn.Parameter(torch.tensor(self._probability_to_logit(self.initial_residual_alpha)))

        if self.encoder_type == 'rgcn':
            self.conv1 = HeteroConv(
                {
                    edge_type: SAGEConv(
                        (self.hidden_channels, self.hidden_channels),
                        self.hidden_channels,
                    )
                    for edge_type in self.edge_types
                },
                aggr=conv_aggr,
            )
            self.conv2 = HeteroConv(
                {
                    edge_type: SAGEConv(
                        (self.hidden_channels, self.hidden_channels),
                        self.out_dim,
                    )
                    for edge_type in self.edge_types
                },
                aggr=conv_aggr,
            )
        else:
            self.conv1 = None
            self.conv2 = None

        scorer_hidden_dim = scorer_hidden_dim if scorer_hidden_dim is not None else self.out_dim
        scorer_output_hidden_dim = (
            scorer_output_hidden_dim if scorer_output_hidden_dim is not None else scorer_hidden_dim
        )
        max_global_id = max(
            int(data[node_type].global_id.max().item())
            for node_type in self.node_types
        ) + 1
        self.register_buffer(
            'disease_external_feature_matrix',
            torch.empty((0, self.disease_text_dim), dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            'disease_external_feature_mask',
            torch.empty(0, dtype=torch.bool),
            persistent=False,
        )
        self._initialize_early_external_feature_buffers(max_global_id=max_global_id)

        scorer_drug_morgan_path = self.drug_morgan_fingerprints_path
        self.scorer = PairAggregationScorer(
            pair_emb_dim=self.out_dim,
            path_emb_dim=self.path_size * self.out_dim,
            hidden_dim=scorer_hidden_dim,
            output_hidden_dim=scorer_output_hidden_dim,
            dropout=self.dropout,
            agg_type=self.agg_type,
            triplet_text_embeddings_path=self.triplet_text_embeddings_path,
            drug_morgan_fingerprints_path=scorer_drug_morgan_path,
            drug_text_embeddings_path=self.drug_text_embeddings_path,
            disease_text_embeddings_path=self.disease_text_embeddings_path,
            nodes_csv_path=self.nodes_csv_path,
            max_global_id=max_global_id,
            use_external_late_fusion=scorer_drug_morgan_path is not None,
            ablate_gnn=self.ablate_gnn,
        )

        self._global_to_local_buffer_names: Dict[str, str] = {}
        self._local_global_id_buffer_names: Dict[str, str] = {}
        self._default_global_id_by_type: Dict[str, int] = {}
        self._register_global_to_local_buffers(data=data)
        self._cached_raw_node_embs_dict: Optional[EmbeddingDict] = None
        self.reset_parameters()

    @staticmethod
    def _probability_to_logit(probability: float) -> float:
        return math.log(probability / (1.0 - probability))

    def reset_parameters(self) -> None:
        for embedding in self.node_embeddings.values():
            nn.init.xavier_uniform_(embedding.weight)
        for projector in self.feature_projections.values():
            projector.reset_parameters()
        self.disease_proj.reset_parameters()
        self.disease_external_norm.reset_parameters()
        for output_projector in self.encoder_output_projections.values():
            if hasattr(output_projector, 'reset_parameters'):
                output_projector.reset_parameters()
        for projector in self.initial_residual_projections_conv1.values():
            if hasattr(projector, 'reset_parameters'):
                projector.reset_parameters()
        for projector in self.initial_residual_projections_conv2.values():
            if hasattr(projector, 'reset_parameters'):
                projector.reset_parameters()

        with torch.no_grad():
            initial_logit = self._probability_to_logit(self.initial_residual_alpha)
            self.conv1_alpha_logit.fill_(initial_logit)
            self.conv2_alpha_logit.fill_(initial_logit)
        if self.conv1 is not None:
            self.conv1.reset_parameters()
        if self.conv2 is not None:
            self.conv2.reset_parameters()
        self.scorer.reset_parameters()

    def encode_with_raw(
        self,
        x_dict: Optional[Mapping[str, Tensor]],
        edge_index_dict: Mapping[EdgeType, Tensor],
    ) -> Tuple[EmbeddingDict, EmbeddingDict]:
        initial_hidden_dict = self._prepare_input_features(x_dict=x_dict)
        raw_node_embs_dict = self._project_encoder_outputs(hidden_dict=initial_hidden_dict)

        if self.encoder_type == 'mlp':
            self._cached_raw_node_embs_dict = raw_node_embs_dict
            return raw_node_embs_dict, raw_node_embs_dict

        if self.conv1 is None or self.conv2 is None:
            raise RuntimeError('RGCN layers are not initialized.')

        effective_edge_index_dict = self._prepare_edge_index_dict_for_message_passing(edge_index_dict)
        hidden_after_conv1 = self.conv1(initial_hidden_dict, effective_edge_index_dict)
        hidden_after_conv1 = self._blend_with_initial_residual(
            previous_dict=initial_hidden_dict,
            updated_dict=hidden_after_conv1,
            initial_hidden_dict=initial_hidden_dict,
            residual_projectors=self.initial_residual_projections_conv1,
            alpha_logit=self.conv1_alpha_logit,
            apply_activation=True,
        )

        hidden_after_conv2 = self.conv2(hidden_after_conv1, effective_edge_index_dict)
        node_embs_dict = self._blend_with_initial_residual(
            previous_dict=hidden_after_conv1,
            updated_dict=hidden_after_conv2,
            initial_hidden_dict=initial_hidden_dict,
            residual_projectors=self.initial_residual_projections_conv2,
            alpha_logit=self.conv2_alpha_logit,
            apply_activation=False,
        )
        self._cached_raw_node_embs_dict = raw_node_embs_dict
        return node_embs_dict, raw_node_embs_dict

    def encode(
        self,
        x_dict: Optional[Mapping[str, Tensor]],
        edge_index_dict: Mapping[EdgeType, Tensor],
    ) -> EmbeddingDict:
        node_embs_dict, _ = self.encode_with_raw(x_dict=x_dict, edge_index_dict=edge_index_dict)
        return node_embs_dict

    def score_batch(
        self,
        node_embs_dict: Mapping[str, Tensor],
        pair_ids: Tensor,
        paths: Tensor,
        attention_mask: Tensor,
        return_attention: bool = False,
        return_distill_loss: bool = False,
        return_path_loss: bool = False,
    ) -> Tensor | Tuple[Tensor, ...]:
        self._validate_score_batch_inputs(
            pair_ids=pair_ids,
            paths=paths,
            attention_mask=attention_mask,
        )

        attention_mask = attention_mask.to(dtype=torch.bool)

        pair_drug_embs = self._gather_node_embeddings_by_global_ids(
            node_embs_dict=node_embs_dict,
            node_type=self.drug_node_type,
            global_ids=pair_ids[:, 0],
        )
        pair_disease_embs = self._gather_node_embeddings_by_global_ids(
            node_embs_dict=node_embs_dict,
            node_type=self.disease_node_type,
            global_ids=pair_ids[:, 1],
        )
        pair_embs = torch.stack([pair_drug_embs, pair_disease_embs], dim=1)

        path_component_embs: List[Tensor] = []
        for column_index, node_type in enumerate(self.path_node_types):
            component_valid_mask = attention_mask
            if self.use_pathway_quads and node_type == self.pathway_node_type:
                component_valid_mask = attention_mask & (
                    paths[:, :, column_index] != self.pathway_dummy_global_id
                )

            component_embs = self._gather_node_embeddings_by_global_ids(
                node_embs_dict=node_embs_dict,
                node_type=node_type,
                global_ids=paths[:, :, column_index],
                valid_mask=component_valid_mask,
            )
            path_component_embs.append(component_embs)

        paths_embs = torch.cat(path_component_embs, dim=-1)
        triplet_key_ids = torch.stack(
            [paths[:, :, 0], paths[:, :, -1], paths[:, :, 1]],
            dim=-1,
        )

        return self.scorer(
            pair_embs=pair_embs,
            paths_embs=paths_embs,
            attention_mask=attention_mask,
            triplet_key_ids=triplet_key_ids,
            drug_global_ids=pair_ids[:, 0],
            disease_global_ids=pair_ids[:, 1],
            return_attention=return_attention,
            return_distill_loss=return_distill_loss,
            return_path_loss=return_path_loss,
        )

    def score_paths(
        self,
        node_embs_dict: Mapping[str, Tensor],
        path_tensor: Tensor,
        return_distill_loss: bool = False,
        return_path_loss: bool = False,
    ) -> Tensor | Tuple[Tensor, ...]:
        if path_tensor.dim() == 1:
            path_tensor = path_tensor.unsqueeze(0)
        if path_tensor.dim() != 2 or path_tensor.size(1) != self.path_size:
            raise ValueError(
                '`path_tensor` must have shape `(batch_size, path_len)`, '
                f'got {tuple(path_tensor.shape)}.'
            )

        pair_ids = torch.stack([path_tensor[:, 0], path_tensor[:, -1]], dim=1)
        paths = path_tensor.unsqueeze(1)
        attention_mask = torch.ones(
            (path_tensor.size(0), 1),
            dtype=torch.bool,
            device=path_tensor.device,
        )
        return self.score_batch(
            node_embs_dict=node_embs_dict,
            pair_ids=pair_ids,
            paths=paths,
            attention_mask=attention_mask,
            return_distill_loss=return_distill_loss,
            return_path_loss=return_path_loss,
        )

    def forward(
        self,
        x_dict: Optional[Mapping[str, Tensor]],
        edge_index_dict: Mapping[EdgeType, Tensor],
        pos_pair_ids: Optional[Tensor] = None,
        pos_paths: Optional[Tensor] = None,
        pos_attention_mask: Optional[Tensor] = None,
        neg_pair_ids: Optional[Tensor] = None,
        neg_paths: Optional[Tensor] = None,
        neg_attention_mask: Optional[Tensor] = None,
        path_tensor: Optional[Tensor] = None,
        return_node_embs: bool = False,
        return_distill_loss: bool = False,
        return_path_loss: bool = False,
    ) -> EmbeddingDict | Tensor | Tuple[Tensor, ...]:
        node_embs_dict = self.encode(
            x_dict=x_dict,
            edge_index_dict=edge_index_dict,
        )

        if path_tensor is None and pos_pair_ids is None:
            return node_embs_dict

        if path_tensor is not None:
            path_outputs = self.score_paths(
                node_embs_dict=node_embs_dict,
                path_tensor=path_tensor,
                return_distill_loss=return_distill_loss,
                return_path_loss=return_path_loss,
            )
            if return_distill_loss and return_path_loss:
                path_logits, path_distill_loss, path_margin_loss = path_outputs
                if return_node_embs:
                    return node_embs_dict, path_logits, path_distill_loss, path_margin_loss
                return path_logits, path_distill_loss, path_margin_loss
            if return_distill_loss:
                path_logits, path_distill_loss = path_outputs
                if return_node_embs:
                    return node_embs_dict, path_logits, path_distill_loss
                return path_logits, path_distill_loss
            if return_path_loss:
                path_logits, path_margin_loss = path_outputs
                if return_node_embs:
                    return node_embs_dict, path_logits, path_margin_loss
                return path_logits, path_margin_loss
            path_logits = path_outputs
            if return_node_embs:
                return node_embs_dict, path_logits
            return path_logits

        if pos_pair_ids is None or pos_paths is None or pos_attention_mask is None:
            raise ValueError(
                'Pair-level forward requires `pos_pair_ids`, `pos_paths`, and `pos_attention_mask`.'
            )

        pos_outputs = self.score_batch(
            node_embs_dict=node_embs_dict,
            pair_ids=pos_pair_ids,
            paths=pos_paths,
            attention_mask=pos_attention_mask,
            return_distill_loss=return_distill_loss,
            return_path_loss=return_path_loss,
        )
        if return_distill_loss and return_path_loss:
            pos_scores, pos_distill_loss, pos_path_loss = pos_outputs
        elif return_distill_loss:
            pos_scores, pos_distill_loss = pos_outputs
            pos_path_loss = None
        elif return_path_loss:
            pos_scores, pos_path_loss = pos_outputs
            pos_distill_loss = None
        else:
            pos_scores = pos_outputs
            pos_distill_loss = None
            pos_path_loss = None

        if neg_pair_ids is None and neg_paths is None and neg_attention_mask is None:
            if return_distill_loss and return_path_loss:
                if return_node_embs:
                    return node_embs_dict, pos_scores, pos_distill_loss, pos_path_loss
                return pos_scores, pos_distill_loss, pos_path_loss
            if return_distill_loss:
                if return_node_embs:
                    return node_embs_dict, pos_scores, pos_distill_loss
                return pos_scores, pos_distill_loss
            if return_path_loss:
                if return_node_embs:
                    return node_embs_dict, pos_scores, pos_path_loss
                return pos_scores, pos_path_loss
            if return_node_embs:
                return node_embs_dict, pos_scores
            return pos_scores

        if neg_pair_ids is None or neg_paths is None or neg_attention_mask is None:
            raise ValueError(
                'Negative pair scoring requires `neg_pair_ids`, `neg_paths`, and `neg_attention_mask`.'
            )

        neg_outputs = self.score_batch(
            node_embs_dict=node_embs_dict,
            pair_ids=neg_pair_ids,
            paths=neg_paths,
            attention_mask=neg_attention_mask,
            return_distill_loss=return_distill_loss,
        )
        if return_distill_loss:
            neg_scores, neg_distill_loss = neg_outputs
            distill_loss = 0.5 * (pos_distill_loss + neg_distill_loss)
            if return_path_loss:
                if return_node_embs:
                    return node_embs_dict, pos_scores, neg_scores, distill_loss, pos_path_loss
                return pos_scores, neg_scores, distill_loss, pos_path_loss
            if return_node_embs:
                return node_embs_dict, pos_scores, neg_scores, distill_loss
            return pos_scores, neg_scores, distill_loss

        neg_scores = neg_outputs
        if return_path_loss:
            if return_node_embs:
                return node_embs_dict, pos_scores, neg_scores, pos_path_loss
            return pos_scores, neg_scores, pos_path_loss
        if return_node_embs:
            return node_embs_dict, pos_scores, neg_scores
        return pos_scores, neg_scores

    def _resolve_path_node_types(
        self,
        base_path_node_types: Sequence[str],
        fallback_pathway_node_type: str,
    ) -> Tuple[str, ...]:
        if self.use_pathway_quads:
            if len(base_path_node_types) == 4:
                return tuple(base_path_node_types)
            if len(base_path_node_types) == 3:
                return (
                    str(base_path_node_types[0]),
                    str(base_path_node_types[1]),
                    fallback_pathway_node_type,
                    str(base_path_node_types[2]),
                )
            raise ValueError('Pathway quad mode expects a 3-node or 4-node schema definition.')

        if len(base_path_node_types) != 3:
            raise ValueError('Triplet mode expects exactly 3 path node types.')
        return tuple(base_path_node_types)

    def _validate_path_schema(self, data: HeteroData) -> None:
        if self.drug_node_type != 'drug':
            raise ValueError(f'Expected first path node type to be `drug`, got `{self.drug_node_type}`.')
        if self.gene_node_type != 'gene/protein':
            raise ValueError(
                f'Expected second path node type to be `gene/protein`, got `{self.gene_node_type}`.'
            )
        if self.disease_node_type != 'disease':
            raise ValueError(
                f'Expected last path node type to be `disease`, got `{self.disease_node_type}`.'
            )
        if self.use_pathway_quads and self.pathway_node_type != 'pathway':
            raise ValueError(
                f'Expected pathway node type to be `pathway`, got `{self.pathway_node_type}`.'
            )

        required_node_types = [self.drug_node_type, self.gene_node_type, self.disease_node_type]
        if self.use_pathway_quads and self.pathway_node_type is not None:
            required_node_types.append(self.pathway_node_type)
        for node_type in required_node_types:
            if node_type not in data.node_types:
                raise KeyError(f'HeteroData is missing required node type `{node_type}`.')

    def _prepare_input_features(
        self,
        x_dict: Optional[Mapping[str, Tensor]],
    ) -> EmbeddingDict:
        prepared_dict: EmbeddingDict = {}
        for node_type in self.node_types:
            if x_dict is not None and node_type in x_dict and x_dict[node_type] is not None:
                raw_x = x_dict[node_type].float()
                if raw_x.dim() != 2:
                    raise ValueError(
                        f'Input features for `{node_type}` must be rank-2, got {tuple(raw_x.shape)}.'
                    )
                if raw_x.size(1) != self.in_channels:
                    raise ValueError(
                        f'Input features for `{node_type}` must match `in_channels`: '
                        f'expected={self.in_channels}, actual={raw_x.size(1)}.'
                    )
                prepared_dict[node_type] = self.feature_projections[node_type](raw_x)
            else:
                node_ids = torch.arange(
                    self.node_embeddings[node_type].num_embeddings,
                    device=self.node_embeddings[node_type].weight.device,
                )
                prepared_dict[node_type] = self.node_embeddings[node_type](node_ids)

        if self.use_early_external_fusion:
            self._inject_early_external_features(prepared_dict)
        return prepared_dict

    def _project_encoder_outputs(
        self,
        hidden_dict: Mapping[str, Tensor],
    ) -> EmbeddingDict:
        output_dict: EmbeddingDict = {}
        for node_type in self.node_types:
            output_dict[node_type] = self.encoder_output_projections[node_type](hidden_dict[node_type])
        return output_dict

    def _blend_with_initial_residual(
        self,
        previous_dict: Mapping[str, Tensor],
        updated_dict: Mapping[str, Tensor],
        initial_hidden_dict: Mapping[str, Tensor],
        residual_projectors: Mapping[str, nn.Module],
        alpha_logit: Tensor,
        apply_activation: bool,
    ) -> EmbeddingDict:
        output_dict: EmbeddingDict = {}
        alpha = torch.sigmoid(alpha_logit)
        for node_type in self.node_types:
            current_tensor = updated_dict.get(node_type, previous_dict[node_type])
            initial_residual = residual_projectors[node_type](initial_hidden_dict[node_type])
            blended_tensor = alpha * current_tensor + (1.0 - alpha) * initial_residual
            if apply_activation:
                blended_tensor = F.relu(blended_tensor)
                blended_tensor = F.dropout(
                    blended_tensor,
                    p=self.dropout,
                    training=self.training,
                )
            output_dict[node_type] = blended_tensor
        return output_dict

    def _register_global_to_local_buffers(self, data: HeteroData) -> None:
        max_global_id = max(
            int(data[node_type].global_id.max().item())
            for node_type in self.node_types
        )

        for type_index, node_type in enumerate(self.node_types):
            if 'global_id' not in data[node_type]:
                raise KeyError(f"`data['{node_type}']` is missing `global_id`.")

            global_ids = data[node_type].global_id.detach().cpu().to(torch.long)
            if global_ids.numel() == 0:
                raise ValueError(f'Node type `{node_type}` has no global ids.')

            mapping = torch.full((max_global_id + 1,), -1, dtype=torch.long)
            mapping[global_ids] = torch.arange(global_ids.numel(), dtype=torch.long)

            safe_node_type = node_type.replace('/', '_').replace('-', '_')
            buffer_name = f'_global_to_local_{type_index}_{safe_node_type}'
            local_global_ids_buffer_name = f'_local_global_ids_{type_index}_{safe_node_type}'
            self.register_buffer(buffer_name, mapping, persistent=False)
            self.register_buffer(local_global_ids_buffer_name, global_ids.clone(), persistent=False)
            self._global_to_local_buffer_names[node_type] = buffer_name
            self._local_global_id_buffer_names[node_type] = local_global_ids_buffer_name
            self._default_global_id_by_type[node_type] = int(global_ids[0].item())


    def _initialize_early_external_feature_buffers(self, max_global_id: int) -> None:
        if not self.use_early_external_fusion:
            return
        if self.nodes_csv_path is None:
            raise ValueError('`nodes_csv_path` is required when `use_early_external_fusion=True`.')
        if self.disease_text_embeddings_path is not None:
            matrix, mask = self._load_external_feature_matrix(
                embedding_path=self.disease_text_embeddings_path,
                expected_dim=self.disease_text_dim,
                max_global_id=max_global_id,
            )
            self.disease_external_feature_matrix = matrix
            self.disease_external_feature_mask = mask

    def _load_external_feature_matrix(
        self,
        embedding_path: Path,
        expected_dim: int,
        max_global_id: int,
    ) -> Tuple[Tensor, Tensor]:
        if self.nodes_csv_path is None:
            raise ValueError('`nodes_csv_path` is required for external feature loading.')
        if not embedding_path.exists():
            raise FileNotFoundError(f'External feature file does not exist: {embedding_path}')
        if not self.nodes_csv_path.exists():
            raise FileNotFoundError(f'PrimeKG nodes.csv does not exist: {self.nodes_csv_path}')

        raw_id_to_global = self._build_raw_id_to_global_id_mapping(self.nodes_csv_path)
        with embedding_path.open('rb') as file:
            feature_dict = pickle.load(file)
        if not isinstance(feature_dict, Mapping):
            raise TypeError(f'External feature file must contain a dictionary-like object: {embedding_path}')

        feature_matrix = torch.zeros((int(max_global_id), int(expected_dim)), dtype=torch.float32)
        feature_mask = torch.zeros(int(max_global_id), dtype=torch.bool)
        for raw_id, feature in feature_dict.items():
            raw_id = str(raw_id)
            if raw_id not in raw_id_to_global:
                continue
            feature_tensor = torch.as_tensor(feature, dtype=torch.float32).view(-1)
            if feature_tensor.numel() != int(expected_dim):
                raise ValueError(
                    'External feature shape mismatch: '
                    f'expected ({int(expected_dim)},), got {tuple(feature_tensor.shape)} for `{raw_id}`.'
                )
            global_id = int(raw_id_to_global[raw_id])
            feature_matrix[global_id] = feature_tensor.detach().clone()
            feature_mask[global_id] = True
        return feature_matrix, feature_mask

    def _build_raw_id_to_global_id_mapping(self, nodes_csv_path: Path) -> Dict[str, int]:
        raw_id_to_global: Dict[str, int] = {}
        with nodes_csv_path.open('r', encoding='utf-8-sig', newline='') as node_file:
            reader = csv.DictReader(node_file)
            for global_id, row in enumerate(reader):
                raw_id = str(row['id']).strip()
                raw_id_to_global[raw_id] = global_id
        return raw_id_to_global

    def _inject_early_external_features(self, prepared_dict: EmbeddingDict) -> None:
        self._overwrite_node_type_with_external_features(
            prepared_dict=prepared_dict,
            node_type=self.disease_node_type,
            feature_matrix=self.disease_external_feature_matrix,
            feature_mask=self.disease_external_feature_mask,
            projector=self.disease_proj,
            norm=self.disease_external_norm,
            apply_post_dropout=True,
        )

    def _overwrite_node_type_with_external_features(
        self,
        prepared_dict: EmbeddingDict,
        node_type: str,
        feature_matrix: Tensor,
        feature_mask: Tensor,
        projector: nn.Module,
        norm: Optional[nn.Module],
        apply_post_dropout: bool,
    ) -> None:
        if node_type not in prepared_dict or feature_matrix.numel() == 0 or feature_mask.numel() == 0:
            return

        local_global_ids = getattr(self, self._local_global_id_buffer_names[node_type])
        lookup_ids = local_global_ids.to(device=feature_matrix.device, dtype=torch.long)
        available_mask = feature_mask[lookup_ids]
        if not available_mask.any():
            return

        external_inputs = feature_matrix[lookup_ids[available_mask]].to(
            device=prepared_dict[node_type].device,
            dtype=prepared_dict[node_type].dtype,
        )
        projected_inputs = projector(external_inputs)
        if norm is not None:
            projected_inputs = norm(projected_inputs)
        if apply_post_dropout:
            projected_inputs = F.dropout(projected_inputs, p=self.dropout, training=self.training)

        updated_features = prepared_dict[node_type].clone()
        updated_features[available_mask.to(device=updated_features.device)] = projected_inputs
        prepared_dict[node_type] = updated_features

    def _prepare_edge_index_dict_for_message_passing(
        self,
        edge_index_dict: Mapping[EdgeType, Tensor],
    ) -> Mapping[EdgeType, Tensor]:
        if not self.training or self.dropedge_p <= 0.0:
            return edge_index_dict

        dropped_edge_index_dict: Dict[EdgeType, Tensor] = {}
        for edge_type, edge_index in edge_index_dict.items():
            if edge_index.numel() == 0:
                dropped_edge_index_dict[edge_type] = edge_index
                continue
            dropped_edge_index, _ = dropout_edge(
                edge_index,
                p=self.dropedge_p,
                training=True,
            )
            dropped_edge_index_dict[edge_type] = dropped_edge_index
        return dropped_edge_index_dict

    def _map_global_ids_to_local_indices(
        self,
        node_type: str,
        global_ids: Tensor,
    ) -> Tensor:
        if node_type not in self._global_to_local_buffer_names:
            raise KeyError(f'No global-id mapping registered for `{node_type}`.')

        mapping = getattr(self, self._global_to_local_buffer_names[node_type])
        global_ids = global_ids.to(device=mapping.device, dtype=torch.long)

        if global_ids.numel() == 0:
            return global_ids

        min_requested_id = int(global_ids.min().item())
        max_requested_id = int(global_ids.max().item())
        if min_requested_id < 0:
            raise IndexError(f'Negative global id requested for `{node_type}`: {min_requested_id}.')
        if max_requested_id >= mapping.numel():
            raise IndexError(
                f'Global id for `{node_type}` exceeds mapping range: '
                f'max_id={max_requested_id}, mapping_size={mapping.numel()}.'
            )

        local_indices = mapping[global_ids]
        if torch.any(local_indices < 0):
            invalid_global_ids = global_ids[local_indices < 0].detach().cpu().tolist()
            raise KeyError(
                f'Unknown global ids for `{node_type}`: {invalid_global_ids[:10]}.'
            )
        return local_indices

    def _gather_node_embeddings_by_global_ids(
        self,
        node_embs_dict: Mapping[str, Tensor],
        node_type: str,
        global_ids: Tensor,
        valid_mask: Optional[Tensor] = None,
    ) -> Tensor:
        if node_type not in node_embs_dict:
            raise KeyError(f'`node_embs_dict` is missing node type `{node_type}`.')

        device = getattr(self, self._global_to_local_buffer_names[node_type]).device
        safe_global_ids = global_ids.to(device=device, dtype=torch.long)
        safe_valid_mask: Optional[Tensor] = None

        if valid_mask is not None:
            safe_valid_mask = valid_mask.to(device=device, dtype=torch.bool)
            if safe_valid_mask.shape != safe_global_ids.shape:
                raise ValueError(
                    '`valid_mask` must have the same shape as `global_ids`: '
                    f'{tuple(safe_valid_mask.shape)} vs {tuple(safe_global_ids.shape)}.'
                )
            safe_global_ids = safe_global_ids.clone()
            safe_global_ids[~safe_valid_mask] = self._default_global_id_by_type[node_type]

        local_indices = self._map_global_ids_to_local_indices(
            node_type=node_type,
            global_ids=safe_global_ids,
        )
        gathered_embs = node_embs_dict[node_type][local_indices]

        if safe_valid_mask is not None:
            gathered_embs = gathered_embs * safe_valid_mask.to(dtype=gathered_embs.dtype).unsqueeze(-1)

        return gathered_embs

    def _validate_score_batch_inputs(
        self,
        pair_ids: Tensor,
        paths: Tensor,
        attention_mask: Tensor,
    ) -> None:
        if pair_ids.dim() != 2 or pair_ids.size(1) != 2:
            raise ValueError(
                '`pair_ids` must have shape `(batch_size, 2)`, '
                f'got {tuple(pair_ids.shape)}.'
            )
        if paths.dim() != 3 or paths.size(2) != self.path_size:
            raise ValueError(
                '`paths` must have shape `(batch_size, max_K, path_len)`, '
                f'with path_len={self.path_size}, got {tuple(paths.shape)}.'
            )
        if attention_mask.dim() != 2:
            raise ValueError(
                '`attention_mask` must have shape `(batch_size, max_K)`, '
                f'got {tuple(attention_mask.shape)}.'
            )
        if pair_ids.size(0) != paths.size(0) or pair_ids.size(0) != attention_mask.size(0):
            raise ValueError('`pair_ids`, `paths`, and `attention_mask` must share the same batch size.')
        if paths.size(1) != attention_mask.size(1):
            raise ValueError('`paths` and `attention_mask` must share the same max_K dimension.')
