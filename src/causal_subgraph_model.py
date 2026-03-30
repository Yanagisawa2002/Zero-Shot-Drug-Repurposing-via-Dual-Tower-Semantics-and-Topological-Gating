from __future__ import annotations

import csv
import pickle
from pathlib import Path
from typing import Dict, Mapping, MutableMapping, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch_geometric.data import HeteroData


EdgeType = Tuple[str, str, str]
NodeEmbeddingDict = Dict[str, Tensor]
EdgeMaskDict = Dict[EdgeType, Tensor]
ExternalFeatureSource = Optional[Path | str | Mapping[str, Tensor]]


class SubgraphGenerator(nn.Module):
    """
    ????????Edge Scorer??

    ????????? relation ?????
    - ??????? h_u
    - ???????? h_v
    - ?? relation embedding
    - ????? MLP ????????? logit
    - ??? sigmoid(logit / tau) ????? m_e ? [0, 1]

    ???? relation-wise ? edge_index_dict?????????? per-edge ? edge_type ???
    ??? edge_type bucket ???????? relation embedding?
    """

    def __init__(
        self,
        hidden_dim: int,
        edge_types: Sequence[EdgeType],
        relation_emb_dim: int = 64,
        scorer_hidden_dim: int = 128,
        temperature: float = 1.0,
    ) -> None:
        super().__init__()
        if hidden_dim <= 0:
            raise ValueError('`hidden_dim` must be positive.')
        if relation_emb_dim <= 0:
            raise ValueError('`relation_emb_dim` must be positive.')
        if scorer_hidden_dim <= 0:
            raise ValueError('`scorer_hidden_dim` must be positive.')
        if temperature <= 0.0:
            raise ValueError('`temperature` must be positive.')

        self.hidden_dim = int(hidden_dim)
        self.relation_emb_dim = int(relation_emb_dim)
        self.scorer_hidden_dim = int(scorer_hidden_dim)
        self.temperature = float(temperature)
        self.edge_types = [tuple(edge_type) for edge_type in edge_types]
        self.edge_type_to_id = {
            edge_type: relation_id for relation_id, edge_type in enumerate(self.edge_types)
        }

        self.relation_embeddings = nn.Embedding(len(self.edge_types), self.relation_emb_dim)
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * self.hidden_dim + self.relation_emb_dim, self.scorer_hidden_dim),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(self.scorer_hidden_dim, 1),
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.relation_embeddings.weight)
        for module in self.edge_mlp:
            if hasattr(module, 'reset_parameters'):
                module.reset_parameters()

    def forward(
        self,
        node_features: Mapping[str, Tensor],
        edge_index_dict: Mapping[EdgeType, Tensor],
        temperature: Optional[float] = None,
    ) -> EdgeMaskDict:
        tau = float(self.temperature if temperature is None else temperature)
        if tau <= 0.0:
            raise ValueError('`temperature` must stay positive during forward.')

        edge_mask_dict: EdgeMaskDict = {}
        for edge_type, edge_index in edge_index_dict.items():
            edge_type = tuple(edge_type)
            if edge_type not in self.edge_type_to_id:
                raise KeyError(f'Unknown edge_type for generator: {edge_type}')
            if edge_index.dim() != 2 or edge_index.size(0) != 2:
                raise ValueError(
                    f'edge_index for {edge_type} must have shape [2, E], got {tuple(edge_index.shape)}.'
                )

            num_edges = int(edge_index.size(1))
            if num_edges == 0:
                edge_mask_dict[edge_type] = edge_index.new_zeros((0,), dtype=torch.float32)
                continue

            src_type, _, dst_type = edge_type
            if src_type not in node_features or dst_type not in node_features:
                raise KeyError(f'Missing node features for edge type {edge_type}.')

            src_index = edge_index[0].to(device=node_features[src_type].device, dtype=torch.long)
            dst_index = edge_index[1].to(device=node_features[dst_type].device, dtype=torch.long)
            src_embs = node_features[src_type][src_index]
            dst_embs = node_features[dst_type][dst_index]

            relation_id = self.edge_type_to_id[edge_type]
            relation_emb = self.relation_embeddings.weight[relation_id].unsqueeze(0)
            relation_emb = relation_emb.expand(num_edges, -1)

            edge_inputs = torch.cat([src_embs, dst_embs, relation_emb], dim=-1)
            edge_logits = self.edge_mlp(edge_inputs).squeeze(-1)
            edge_mask = torch.sigmoid(edge_logits / tau)
            edge_mask_dict[edge_type] = edge_mask

        return edge_mask_dict


class MaskedRelationalConv(nn.Module):
    """
    ??????? relation-aware message passing layer?

    ?? relation ??????????
        m_e = W_r h_u
    ??? generator ???????????
        m_e = m_e * mask_e
    ??????????????????? root transform ???

    ???????????????????????????
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        node_types: Sequence[str],
        edge_types: Sequence[EdgeType],
        dropout: float = 0.1,
        use_layernorm: bool = True,
    ) -> None:
        super().__init__()
        if in_dim <= 0 or out_dim <= 0:
            raise ValueError('`in_dim` and `out_dim` must be positive.')

        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        self.node_types = list(node_types)
        self.edge_types = [tuple(edge_type) for edge_type in edge_types]
        self.dropout = float(dropout)
        self.use_layernorm = bool(use_layernorm)

        self.edge_type_to_module_key = {
            edge_type: f'relation_{relation_id}'
            for relation_id, edge_type in enumerate(self.edge_types)
        }
        self.relation_linears = nn.ModuleDict(
            {
                module_key: nn.Linear(self.in_dim, self.out_dim, bias=False)
                for module_key in self.edge_type_to_module_key.values()
            }
        )
        self.root_linears = nn.ModuleDict(
            {
                node_type: nn.Linear(self.in_dim, self.out_dim, bias=True)
                for node_type in self.node_types
            }
        )
        self.norms = nn.ModuleDict(
            {
                node_type: nn.LayerNorm(self.out_dim)
                for node_type in self.node_types
            }
        )
        self.dropout_layer = nn.Dropout(self.dropout)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for linear in self.relation_linears.values():
            linear.reset_parameters()
        for linear in self.root_linears.values():
            linear.reset_parameters()
        for norm in self.norms.values():
            norm.reset_parameters()

    def forward(
        self,
        node_features: Mapping[str, Tensor],
        edge_index_dict: Mapping[EdgeType, Tensor],
        edge_mask_dict: Mapping[EdgeType, Tensor],
    ) -> NodeEmbeddingDict:
        aggregated_messages: NodeEmbeddingDict = {
            node_type: node_features[node_type].new_zeros(
                (node_features[node_type].size(0), self.out_dim)
            )
            for node_type in self.node_types
        }
        message_weights: Dict[str, Tensor] = {
            node_type: node_features[node_type].new_zeros((node_features[node_type].size(0), 1))
            for node_type in self.node_types
        }

        for edge_type, edge_index in edge_index_dict.items():
            edge_type = tuple(edge_type)
            if edge_type not in self.edge_type_to_module_key:
                raise KeyError(f'Unknown edge_type for causal predictor: {edge_type}')
            if edge_type not in edge_mask_dict:
                raise KeyError(f'Missing edge mask for edge_type {edge_type}.')
            if edge_index.numel() == 0:
                continue

            src_type, _, dst_type = edge_type
            src_features = node_features[src_type]
            dst_features = node_features[dst_type]
            device = dst_features.device

            src_index = edge_index[0].to(device=device, dtype=torch.long)
            dst_index = edge_index[1].to(device=device, dtype=torch.long)
            edge_mask = edge_mask_dict[edge_type].to(device=device, dtype=dst_features.dtype)
            if edge_mask.dim() != 1 or edge_mask.numel() != edge_index.size(1):
                raise ValueError(
                    f'Edge mask for {edge_type} must have shape [E], got {tuple(edge_mask.shape)}.'
                )

            module_key = self.edge_type_to_module_key[edge_type]
            relation_messages = self.relation_linears[module_key](src_features[src_index])
            relation_messages = relation_messages * edge_mask.unsqueeze(-1)

            aggregated_messages[dst_type].index_add_(0, dst_index, relation_messages)
            message_weights[dst_type].index_add_(0, dst_index, edge_mask.unsqueeze(-1))

        updated_features: NodeEmbeddingDict = {}
        for node_type in self.node_types:
            root_term = self.root_linears[node_type](node_features[node_type])
            neighbor_term = aggregated_messages[node_type] / message_weights[node_type].clamp(min=1.0)
            hidden = root_term + neighbor_term
            if self.use_layernorm:
                hidden = self.norms[node_type](hidden)
            hidden = F.gelu(hidden)
            hidden = self.dropout_layer(hidden)
            updated_features[node_type] = hidden

        return updated_features


class CausalPredictor(nn.Module):
    """
    ??????????????

    ??? MaskedRelationalConv ??????????? generator ??????????
    ????????????????????
    """

    def __init__(
        self,
        hidden_dim: int,
        out_dim: int,
        node_types: Sequence[str],
        edge_types: Sequence[EdgeType],
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if num_layers <= 0:
            raise ValueError('`num_layers` must be positive.')
        if hidden_dim <= 0 or out_dim <= 0:
            raise ValueError('`hidden_dim` and `out_dim` must be positive.')

        self.hidden_dim = int(hidden_dim)
        self.out_dim = int(out_dim)
        self.node_types = list(node_types)
        self.edge_types = [tuple(edge_type) for edge_type in edge_types]
        self.num_layers = int(num_layers)

        layers = []
        for layer_index in range(self.num_layers):
            layer_in_dim = self.hidden_dim if layer_index == 0 else self.out_dim
            layer_out_dim = self.out_dim
            layers.append(
                MaskedRelationalConv(
                    in_dim=layer_in_dim,
                    out_dim=layer_out_dim,
                    node_types=self.node_types,
                    edge_types=self.edge_types,
                    dropout=dropout,
                    use_layernorm=True,
                )
            )
        self.layers = nn.ModuleList(layers)

    def reset_parameters(self) -> None:
        for layer in self.layers:
            layer.reset_parameters()

    def forward(
        self,
        node_features: Mapping[str, Tensor],
        edge_index_dict: Mapping[EdgeType, Tensor],
        edge_mask_dict: Mapping[EdgeType, Tensor],
    ) -> NodeEmbeddingDict:
        hidden_dict: NodeEmbeddingDict = {
            node_type: tensor for node_type, tensor in node_features.items()
        }
        for layer in self.layers:
            hidden_dict = layer(
                node_features=hidden_dict,
                edge_index_dict=edge_index_dict,
                edge_mask_dict=edge_mask_dict,
            )
        return hidden_dict


class CausalRepurposingNet(nn.Module):
    """
    ????????????

    ???????
    1. ????? Layer-0 ??? Early Fusion?
    2. SubgraphGenerator ????????????? M?
    3. CausalPredictor ?? M ???????????????

    ?? pair prediction ????
    - drug ??? h_drug
    - disease ??? h_disease
    - ?? 1024-d Morgan ???Late Fusion?

    ????????????????????????????????????
    """

    def __init__(
        self,
        data: HeteroData,
        hidden_dim: int = 128,
        out_dim: int = 128,
        predictor_num_layers: int = 2,
        dropout: float = 0.1,
        generator_temperature: float = 1.0,
        generator_relation_emb_dim: int = 64,
        generator_hidden_dim: int = 128,
        disease_text_embeddings: ExternalFeatureSource = None,
        drug_morgan_fingerprints: ExternalFeatureSource = None,
        nodes_csv_path: Optional[str | Path] = None,
        disease_node_type: str = 'disease',
        drug_node_type: str = 'drug',
        disease_text_dim: int = 768,
        drug_fingerprint_dim: int = 1024,
    ) -> None:
        super().__init__()
        if hidden_dim <= 0 or out_dim <= 0:
            raise ValueError('`hidden_dim` and `out_dim` must be positive.')
        if predictor_num_layers <= 0:
            raise ValueError('`predictor_num_layers` must be positive.')
        if disease_node_type not in data.node_types:
            raise KeyError(f'Disease node type `{disease_node_type}` is not present in data.node_types.')
        if drug_node_type not in data.node_types:
            raise KeyError(f'Drug node type `{drug_node_type}` is not present in data.node_types.')

        self.hidden_dim = int(hidden_dim)
        self.out_dim = int(out_dim)
        self.predictor_num_layers = int(predictor_num_layers)
        self.dropout = float(dropout)
        self.node_types = list(data.node_types)
        self.edge_types = [tuple(edge_type) for edge_type in data.edge_types]
        self.disease_node_type = disease_node_type
        self.drug_node_type = drug_node_type
        self.disease_text_dim = int(disease_text_dim)
        self.drug_fingerprint_dim = int(drug_fingerprint_dim)
        self.nodes_csv_path = None if nodes_csv_path is None else Path(nodes_csv_path)

        self.node_embeddings = nn.ModuleDict(
            {
                node_type: nn.Embedding(int(data[node_type].num_nodes), self.hidden_dim)
                for node_type in self.node_types
            }
        )
        self.disease_proj = nn.Sequential(
            nn.LayerNorm(self.disease_text_dim),
            nn.Linear(self.disease_text_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(0.5),
        )

        max_global_id = self._infer_max_global_id(data=data)
        self.register_buffer(
            'disease_feature_matrix',
            torch.empty((0, self.disease_text_dim), dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            'disease_feature_mask',
            torch.empty(0, dtype=torch.bool),
            persistent=False,
        )
        self.register_buffer(
            'drug_fingerprint_matrix',
            torch.empty((0, self.drug_fingerprint_dim), dtype=torch.float32),
            persistent=False,
        )

        self._global_to_local_buffer_names: Dict[str, str] = {}
        self._local_global_id_buffer_names: Dict[str, str] = {}
        self._default_global_id_by_type: Dict[str, int] = {}
        self._register_global_to_local_buffers(data=data, max_global_id=max_global_id)
        self._initialize_external_feature_buffers(
            max_global_id=max_global_id,
            disease_text_embeddings=disease_text_embeddings,
            drug_morgan_fingerprints=drug_morgan_fingerprints,
        )

        self.subgraph_generator = SubgraphGenerator(
            hidden_dim=self.hidden_dim,
            edge_types=self.edge_types,
            relation_emb_dim=generator_relation_emb_dim,
            scorer_hidden_dim=generator_hidden_dim,
            temperature=generator_temperature,
        )
        self.causal_predictor = CausalPredictor(
            hidden_dim=self.hidden_dim,
            out_dim=self.out_dim,
            node_types=self.node_types,
            edge_types=self.edge_types,
            num_layers=self.predictor_num_layers,
            dropout=self.dropout,
        )
        self.output_mlp = nn.Sequential(
            nn.Linear(2 * self.out_dim + self.drug_fingerprint_dim, self.out_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.out_dim, 1),
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for embedding in self.node_embeddings.values():
            nn.init.xavier_uniform_(embedding.weight)
        for module in self.disease_proj:
            if hasattr(module, 'reset_parameters'):
                module.reset_parameters()
        self.subgraph_generator.reset_parameters()
        self.causal_predictor.reset_parameters()
        for module in self.output_mlp:
            if hasattr(module, 'reset_parameters'):
                module.reset_parameters()

    def forward(
        self,
        pair_ids: Tensor,
        edge_index_dict: Mapping[EdgeType, Tensor],
        return_node_embeddings: bool = False,
    ) -> Tuple[Tensor, EdgeMaskDict] | Tuple[Tensor, EdgeMaskDict, NodeEmbeddingDict]:
        """
        ??
        - pair_ids: `(batch_size, 2)`???? `(drug_global_id, disease_global_id)`
        - edge_index_dict: ??? relation-wise ??

        ??
        - logits: `(batch_size,)`?? BCEWithLogitsLoss ??
        - edge_mask_dict: ?? relation ??????
        - ?? node_embs_dict: ??????????????/???
        """
        if pair_ids.dim() != 2 or pair_ids.size(1) != 2:
            raise ValueError(f'`pair_ids` must have shape [batch_size, 2], got {tuple(pair_ids.shape)}.')

        node_embs_dict, edge_mask_dict = self.encode_graph(
            edge_index_dict=edge_index_dict,
            device=pair_ids.device,
        )
        logits = self.score_pairs(
            node_embs_dict=node_embs_dict,
            pair_ids=pair_ids,
        )

        if return_node_embeddings:
            return logits, edge_mask_dict, node_embs_dict
        return logits, edge_mask_dict

    def encode_graph(
        self,
        edge_index_dict: Mapping[EdgeType, Tensor],
        device: Optional[torch.device] = None,
    ) -> Tuple[NodeEmbeddingDict, EdgeMaskDict]:
        target_device = device or next(self.parameters()).device
        initial_node_features = self._prepare_initial_node_features(device=target_device)
        edge_mask_dict = self.subgraph_generator(
            node_features=initial_node_features,
            edge_index_dict=edge_index_dict,
        )
        node_embs_dict = self.causal_predictor(
            node_features=initial_node_features,
            edge_index_dict=edge_index_dict,
            edge_mask_dict=edge_mask_dict,
        )
        return node_embs_dict, edge_mask_dict

    def score_pairs(
        self,
        node_embs_dict: Mapping[str, Tensor],
        pair_ids: Tensor,
    ) -> Tensor:
        if pair_ids.dim() != 2 or pair_ids.size(1) != 2:
            raise ValueError(f'`pair_ids` must have shape [batch_size, 2], got {tuple(pair_ids.shape)}.')

        drug_embs = self._gather_node_embeddings_by_global_ids(
            node_type=self.drug_node_type,
            node_embs_dict=node_embs_dict,
            global_ids=pair_ids[:, 0],
        )
        disease_embs = self._gather_node_embeddings_by_global_ids(
            node_type=self.disease_node_type,
            node_embs_dict=node_embs_dict,
            global_ids=pair_ids[:, 1],
        )
        drug_morgan_embs = self._lookup_drug_fingerprints(
            drug_global_ids=pair_ids[:, 0],
            device=drug_embs.device,
            dtype=drug_embs.dtype,
        )

        final_rep = torch.cat([drug_embs, disease_embs, drug_morgan_embs], dim=-1)
        return self.output_mlp(final_rep).squeeze(-1)

    def _prepare_initial_node_features(self, device: torch.device) -> NodeEmbeddingDict:
        prepared_dict: NodeEmbeddingDict = {}
        for node_type in self.node_types:
            num_nodes = self.node_embeddings[node_type].num_embeddings
            node_indices = torch.arange(num_nodes, device=device)
            prepared_dict[node_type] = self.node_embeddings[node_type](node_indices)

        self._inject_disease_early_features(prepared_dict)
        return prepared_dict

    def _inject_disease_early_features(self, prepared_dict: MutableMapping[str, Tensor]) -> None:
        if self.disease_feature_matrix.numel() == 0 or self.disease_feature_mask.numel() == 0:
            return
        if self.disease_node_type not in prepared_dict:
            return

        local_global_ids = getattr(self, self._local_global_id_buffer_names[self.disease_node_type])
        lookup_ids = local_global_ids.to(device=self.disease_feature_matrix.device, dtype=torch.long)
        available_mask = self.disease_feature_mask[lookup_ids]
        if not bool(available_mask.any().item()):
            return

        projected_inputs = self.disease_proj(self.disease_feature_matrix[lookup_ids[available_mask]])
        disease_features = prepared_dict[self.disease_node_type]
        updated_disease_features = disease_features.clone()
        updated_disease_features[available_mask.to(device=disease_features.device)] = projected_inputs.to(
            device=disease_features.device,
            dtype=disease_features.dtype,
        )
        prepared_dict[self.disease_node_type] = updated_disease_features

    def _initialize_external_feature_buffers(
        self,
        max_global_id: int,
        disease_text_embeddings: ExternalFeatureSource,
        drug_morgan_fingerprints: ExternalFeatureSource,
    ) -> None:
        if disease_text_embeddings is not None:
            disease_matrix, disease_mask = self._load_external_feature_matrix(
                source=disease_text_embeddings,
                expected_dim=self.disease_text_dim,
                max_global_id=max_global_id,
            )
            self.register_buffer('disease_feature_matrix', disease_matrix, persistent=False)
            self.register_buffer('disease_feature_mask', disease_mask, persistent=False)
        if drug_morgan_fingerprints is not None:
            drug_matrix, _ = self._load_external_feature_matrix(
                source=drug_morgan_fingerprints,
                expected_dim=self.drug_fingerprint_dim,
                max_global_id=max_global_id,
            )
            self.register_buffer('drug_fingerprint_matrix', drug_matrix, persistent=False)

    def _load_external_feature_matrix(
        self,
        source: ExternalFeatureSource,
        expected_dim: int,
        max_global_id: int,
    ) -> Tuple[Tensor, Tensor]:
        feature_dict = self._load_external_feature_dict(source)
        raw_id_to_global = self._build_raw_id_to_global_id_mapping(self.nodes_csv_path)

        feature_matrix = torch.zeros((int(max_global_id), int(expected_dim)), dtype=torch.float32)
        feature_mask = torch.zeros(int(max_global_id), dtype=torch.bool)
        for raw_id, raw_feature in feature_dict.items():
            if raw_id not in raw_id_to_global:
                continue
            feature_tensor = self._coerce_feature_tensor(raw_feature, expected_dim=expected_dim)
            if feature_tensor is None:
                continue
            global_id = int(raw_id_to_global[raw_id])
            feature_matrix[global_id] = feature_tensor.detach().clone()
            feature_mask[global_id] = True
        return feature_matrix, feature_mask

    def _load_external_feature_dict(self, source: ExternalFeatureSource) -> Dict[str, Tensor]:
        if source is None:
            return {}
        if isinstance(source, Mapping):
            return {str(key): value for key, value in source.items()}

        source_path = Path(source)
        if not source_path.exists():
            raise FileNotFoundError(f'External feature file does not exist: {source_path}')
        with source_path.open('rb') as feature_file:
            loaded = pickle.load(feature_file)
        if not isinstance(loaded, Mapping):
            raise TypeError(f'External feature file must contain a mapping, got {type(loaded)!r}.')
        return {str(key): value for key, value in loaded.items()}

    def _coerce_feature_tensor(self, raw_feature: object, expected_dim: int) -> Optional[Tensor]:
        feature_tensor = torch.as_tensor(raw_feature, dtype=torch.float32)
        if feature_tensor.numel() != expected_dim:
            return None
        return feature_tensor.view(expected_dim)

    def _build_raw_id_to_global_id_mapping(self, nodes_csv_path: Optional[Path]) -> Dict[str, int]:
        if nodes_csv_path is None:
            raise ValueError('`nodes_csv_path` is required for external feature loading.')
        if not nodes_csv_path.exists():
            raise FileNotFoundError(f'PrimeKG nodes.csv does not exist: {nodes_csv_path}')

        raw_id_to_global: Dict[str, int] = {}
        with nodes_csv_path.open('r', encoding='utf-8-sig', newline='') as node_file:
            reader = csv.DictReader(node_file)
            for global_id, row in enumerate(reader):
                raw_id = row.get('id')
                if raw_id:
                    raw_id_to_global[raw_id] = global_id
        return raw_id_to_global

    def _register_global_to_local_buffers(self, data: HeteroData, max_global_id: int) -> None:
        for type_index, node_type in enumerate(self.node_types):
            if 'global_id' in data[node_type]:
                global_ids = data[node_type].global_id.detach().cpu().to(torch.long)
            else:
                global_ids = torch.arange(int(data[node_type].num_nodes), dtype=torch.long)

            mapping = torch.full((max_global_id,), -1, dtype=torch.long)
            mapping[global_ids] = torch.arange(global_ids.numel(), dtype=torch.long)

            safe_node_type = self._sanitize_name(node_type)
            global_to_local_name = f'_global_to_local_{type_index}_{safe_node_type}'
            local_global_ids_name = f'_local_global_ids_{type_index}_{safe_node_type}'
            self.register_buffer(global_to_local_name, mapping, persistent=False)
            self.register_buffer(local_global_ids_name, global_ids.clone(), persistent=False)
            self._global_to_local_buffer_names[node_type] = global_to_local_name
            self._local_global_id_buffer_names[node_type] = local_global_ids_name
            if global_ids.numel() > 0:
                self._default_global_id_by_type[node_type] = int(global_ids[0].item())

    def _infer_max_global_id(self, data: HeteroData) -> int:
        max_global_id = 0
        for node_type in data.node_types:
            if 'global_id' in data[node_type] and data[node_type].global_id.numel() > 0:
                candidate = int(data[node_type].global_id.max().item()) + 1
            else:
                candidate = int(data[node_type].num_nodes)
            max_global_id = max(max_global_id, candidate)
        if max_global_id <= 0:
            raise ValueError('Unable to infer a positive max_global_id from the provided HeteroData.')
        return max_global_id

    def _map_global_ids_to_local_indices(self, node_type: str, global_ids: Tensor) -> Tensor:
        if node_type not in self._global_to_local_buffer_names:
            raise KeyError(f'Unknown node_type `{node_type}` for global-id lookup.')
        mapping = getattr(self, self._global_to_local_buffer_names[node_type])
        global_ids = global_ids.to(device=mapping.device, dtype=torch.long)
        if global_ids.numel() == 0:
            return global_ids
        if int(global_ids.min().item()) < 0 or int(global_ids.max().item()) >= mapping.numel():
            raise IndexError(f'Global ids for `{node_type}` fall outside the registered range.')
        local_indices = mapping[global_ids]
        if bool((local_indices < 0).any().item()):
            invalid_ids = global_ids[local_indices < 0].detach().cpu().tolist()
            raise KeyError(f'Unknown global ids for `{node_type}`: {invalid_ids[:10]}')
        return local_indices

    def _gather_node_embeddings_by_global_ids(
        self,
        node_type: str,
        node_embs_dict: Mapping[str, Tensor],
        global_ids: Tensor,
    ) -> Tensor:
        if node_type not in node_embs_dict:
            raise KeyError(f'Missing node embeddings for `{node_type}`.')
        local_indices = self._map_global_ids_to_local_indices(node_type=node_type, global_ids=global_ids)
        return node_embs_dict[node_type][local_indices.to(device=node_embs_dict[node_type].device)]

    def _lookup_drug_fingerprints(
        self,
        drug_global_ids: Tensor,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor:
        if self.drug_fingerprint_matrix.numel() == 0:
            return torch.zeros(
                (drug_global_ids.size(0), self.drug_fingerprint_dim),
                device=device,
                dtype=dtype,
            )

        lookup_ids = drug_global_ids.to(device=self.drug_fingerprint_matrix.device, dtype=torch.long)
        safe_ids = lookup_ids.clone()
        valid_mask = (safe_ids >= 0) & (safe_ids < self.drug_fingerprint_matrix.size(0))
        safe_ids[~valid_mask] = 0
        fingerprint_embs = self.drug_fingerprint_matrix[safe_ids]
        if bool((~valid_mask).any().item()):
            fingerprint_embs = fingerprint_embs.clone()
            fingerprint_embs[~valid_mask] = 0.0
        return fingerprint_embs.to(device=device, dtype=dtype)

    @staticmethod
    def _sanitize_name(name: str) -> str:
        safe = name.replace('/', '_').replace(' ', '_').replace('-', '_')
        return ''.join(ch if ch.isalnum() or ch == '_' else '_' for ch in safe)


def calc_sparsity_loss(edge_mask: Tensor | Mapping[EdgeType, Tensor]) -> Tensor:
    """
    ??????????
        L_sparse = mean(M)

    ??????? generator ??????????????????
    ????????? BCE ??????????
        total_loss = bce_loss + beta * calc_sparsity_loss(edge_mask)
    """
    if isinstance(edge_mask, Mapping):
        flattened_masks = [mask.reshape(-1) for mask in edge_mask.values() if mask.numel() > 0]
        if not flattened_masks:
            return torch.tensor(0.0, dtype=torch.float32)
        all_masks = torch.cat(flattened_masks, dim=0)
        return all_masks.float().mean()

    if edge_mask.numel() == 0:
        return edge_mask.new_tensor(0.0, dtype=torch.float32)
    return edge_mask.float().mean()
