from __future__ import annotations

import csv
import pickle
from pathlib import Path
from typing import Dict, Mapping, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn


TripletTextKey = Tuple[str, str, str]


class PairAggregationScorer(nn.Module):
    """
    Score a drug-disease pair by aggregating multiple mechanism paths.

    Inputs:
    - pair_embs: (batch_size, 2, pair_emb_dim)
    - paths_embs: (batch_size, max_K, path_emb_dim)
    - attention_mask: (batch_size, max_K)
    - triplet_key_ids: optional `(drug_global_id, disease_global_id, gene_global_id)`
      tensor with shape `(batch_size, max_K, 3)` for text-teacher matching
    - drug_global_ids: optional `(batch_size,)` drug ids for Morgan late fusion and drug-text lookup
    - disease_global_ids: optional `(batch_size,)` disease ids for disease-text lookup

    Aggregation modes:
    - attention: query-based attention pooling in projected text space
    - mean: mask-aware mean pooling
    - max: mask-aware max pooling

    Asymmetric modality routing:
    - Disease semantics are injected early into the encoder.
    - Drug Morgan fingerprints are injected late at the scorer head.
    - Drug and disease SapBERT mechanism texts are concatenated symmetrically at the scorer head.
    - Final scoring representation:
      [h_drug_gnn, h_disease_gnn, h_paths, morgan_raw_feat, drug_text, disease_text]

    Optional semi-supervised distillation:
    - Project every path feature into a 768-d text space
    - Match `(drug, disease, gene)` keys against external PubMedBERT triplet features
    - Distill predicted text-space features toward teacher embeddings when available
    """

    def __init__(
        self,
        pair_emb_dim: int,
        path_emb_dim: int,
        hidden_dim: int,
        query_hidden_dim: int | None = None,
        output_hidden_dim: int | None = None,
        dropout: float = 0.1,
        agg_type: str = 'attention',
        text_embedding_dim: int = 768,
        triplet_text_embeddings_path: Optional[str | Path] = None,
        drug_morgan_fingerprints_path: Optional[str | Path] = None,
        nodes_csv_path: Optional[str | Path] = None,
        max_global_id: Optional[int] = None,
        distill_loss_type: str = 'cosine',
        drug_fingerprint_dim: int = 1024,
        use_external_late_fusion: bool = True,
        drug_text_embeddings_path: Optional[str | Path] = None,
        disease_text_embeddings_path: Optional[str | Path] = None,
        drug_text_dim: int = 768,
        disease_text_dim: int = 768,
        ablate_gnn: bool = False,
    ) -> None:
        super().__init__()

        if pair_emb_dim <= 0:
            raise ValueError('`pair_emb_dim` must be positive.')
        if path_emb_dim <= 0:
            raise ValueError('`path_emb_dim` must be positive.')
        if hidden_dim <= 0:
            raise ValueError('`hidden_dim` must be positive.')
        if text_embedding_dim <= 0:
            raise ValueError('`text_embedding_dim` must be positive.')
        if drug_fingerprint_dim <= 0:
            raise ValueError('`drug_fingerprint_dim` must be positive.')
        if drug_text_dim <= 0:
            raise ValueError('`drug_text_dim` must be positive.')
        if disease_text_dim <= 0:
            raise ValueError('`disease_text_dim` must be positive.')
        if agg_type not in {'attention', 'mean', 'max'}:
            raise ValueError("`agg_type` must be one of {'attention', 'mean', 'max'}.")
        if distill_loss_type not in {'cosine', 'mse'}:
            raise ValueError("`distill_loss_type` must be one of {'cosine', 'mse'}.")

        self.pair_emb_dim = int(pair_emb_dim)
        self.path_emb_dim = int(path_emb_dim)
        self.hidden_dim = int(hidden_dim)
        self.query_hidden_dim = int(query_hidden_dim or hidden_dim)
        self.output_hidden_dim = int(output_hidden_dim or hidden_dim)
        self.dropout = float(dropout)
        self.agg_type = agg_type
        self.text_embedding_dim = int(text_embedding_dim)
        self.drug_fingerprint_dim = int(drug_fingerprint_dim)
        self.drug_text_dim = int(drug_text_dim)
        self.disease_text_dim = int(disease_text_dim)
        self.distill_loss_type = distill_loss_type
        self.use_external_late_fusion = bool(use_external_late_fusion)
        self.use_drug_text_late_fusion = drug_text_embeddings_path is not None
        self.use_disease_text_late_fusion = disease_text_embeddings_path is not None
        self.ablate_gnn = bool(ablate_gnn)

        self.query_mlp = nn.Sequential(
            nn.Linear(2 * self.pair_emb_dim, self.query_hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.query_hidden_dim, self.hidden_dim),
        )
        self.path_value_proj = nn.Linear(self.path_emb_dim, self.hidden_dim)
        self.text_projector = nn.Sequential(
            nn.Linear(self.path_emb_dim, self.text_embedding_dim),
            nn.ReLU(),
            nn.Linear(self.text_embedding_dim, self.text_embedding_dim),
        )
        self.query_text_proj = nn.Linear(self.hidden_dim, self.text_embedding_dim)
        self.attn_mlp = nn.Linear(self.text_embedding_dim, 1)
        self.path_gate = nn.Sequential(
            nn.Linear(self.hidden_dim, 1),
            nn.Sigmoid(),
        )
        self.path_scorer = nn.Sequential(
            nn.Linear(2 * self.pair_emb_dim + self.hidden_dim, 128),
            nn.GELU(),
            nn.Linear(128, 1),
        )
        self.path_loss_margin = 0.5

        final_input_dim = self.hidden_dim + 2 * self.pair_emb_dim
        if self.use_external_late_fusion:
            final_input_dim += self.drug_fingerprint_dim
        if self.use_drug_text_late_fusion:
            final_input_dim += self.drug_text_dim
        if self.use_disease_text_late_fusion:
            final_input_dim += self.disease_text_dim
        self.output_mlp = nn.Sequential(
            nn.Linear(final_input_dim, self.output_hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.output_hidden_dim, 1),
        )

        self.triplet_hash_base: int = 1
        self._teacher_is_enabled = False
        self._drug_fingerprints_enabled = False
        self.register_buffer(
            'triplet_teacher_codes',
            torch.empty(0, dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            'triplet_teacher_embeddings',
            torch.empty((0, self.text_embedding_dim), dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            'drug_fingerprint_matrix',
            torch.empty((0, self.drug_fingerprint_dim), dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            'drug_text_matrix',
            torch.empty((0, self.drug_text_dim), dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            'drug_text_mask',
            torch.empty(0, dtype=torch.bool),
            persistent=False,
        )
        self.register_buffer(
            'disease_text_matrix',
            torch.empty((0, self.disease_text_dim), dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            'disease_text_mask',
            torch.empty(0, dtype=torch.bool),
            persistent=False,
        )
        self._drug_text_enabled = False
        self._disease_text_enabled = False

        if triplet_text_embeddings_path is not None:
            self._initialize_triplet_text_teacher(
                triplet_text_embeddings_path=Path(triplet_text_embeddings_path),
                nodes_csv_path=Path(nodes_csv_path) if nodes_csv_path is not None else None,
                max_global_id=max_global_id,
            )
        if self.use_external_late_fusion and drug_morgan_fingerprints_path is not None:
            self._initialize_drug_fingerprints(
                drug_morgan_fingerprints_path=Path(drug_morgan_fingerprints_path),
                nodes_csv_path=Path(nodes_csv_path) if nodes_csv_path is not None else None,
                max_global_id=max_global_id,
            )
        if self.use_drug_text_late_fusion and drug_text_embeddings_path is not None:
            self._initialize_node_text_embeddings(
                embeddings_path=Path(drug_text_embeddings_path),
                nodes_csv_path=Path(nodes_csv_path) if nodes_csv_path is not None else None,
                max_global_id=max_global_id,
                expected_dim=self.drug_text_dim,
                matrix_attr='drug_text_matrix',
                mask_attr='drug_text_mask',
                enabled_attr='_drug_text_enabled',
            )
        if self.use_disease_text_late_fusion and disease_text_embeddings_path is not None:
            self._initialize_node_text_embeddings(
                embeddings_path=Path(disease_text_embeddings_path),
                nodes_csv_path=Path(nodes_csv_path) if nodes_csv_path is not None else None,
                max_global_id=max_global_id,
                expected_dim=self.disease_text_dim,
                matrix_attr='disease_text_matrix',
                mask_attr='disease_text_mask',
                enabled_attr='_disease_text_enabled',
            )

        self.reset_parameters()

    @property
    def has_text_teacher(self) -> bool:
        return self._teacher_is_enabled and self.triplet_teacher_codes.numel() > 0

    def reset_parameters(self) -> None:
        for module in self.query_mlp:
            if hasattr(module, 'reset_parameters'):
                module.reset_parameters()
        self.path_value_proj.reset_parameters()
        self.query_text_proj.reset_parameters()
        self.attn_mlp.reset_parameters()
        for module in self.path_gate:
            if hasattr(module, 'reset_parameters'):
                module.reset_parameters()
        for module in self.path_scorer:
            if hasattr(module, 'reset_parameters'):
                module.reset_parameters()
        for module in self.text_projector:
            if hasattr(module, 'reset_parameters'):
                module.reset_parameters()
        for module in self.output_mlp:
            if hasattr(module, 'reset_parameters'):
                module.reset_parameters()

    def forward(
        self,
        pair_embs: Tensor,
        paths_embs: Tensor,
        attention_mask: Tensor,
        triplet_key_ids: Optional[Tensor] = None,
        drug_global_ids: Optional[Tensor] = None,
        disease_global_ids: Optional[Tensor] = None,
        return_attention: bool = False,
        return_distill_loss: bool = False,
        return_path_loss: bool = False,
    ) -> Tensor | Tuple[Tensor, ...]:
        self._validate_inputs(
            pair_embs=pair_embs,
            paths_embs=paths_embs,
            attention_mask=attention_mask,
            triplet_key_ids=triplet_key_ids,
            drug_global_ids=drug_global_ids,
            disease_global_ids=disease_global_ids,
        )

        attention_mask = attention_mask.to(dtype=torch.bool)
        pair_context = torch.cat([pair_embs[:, 0, :], pair_embs[:, 1, :]], dim=-1)
        drug_fingerprint_embs = self._lookup_drug_fingerprints(
            drug_global_ids=drug_global_ids,
            batch_size=pair_embs.size(0),
            device=pair_embs.device,
            dtype=pair_embs.dtype,
        )
        drug_text_embs = self._lookup_node_text_embeddings(
            global_ids=drug_global_ids,
            batch_size=pair_embs.size(0),
            device=pair_embs.device,
            dtype=pair_embs.dtype,
            matrix_attr='drug_text_matrix',
            mask_attr='drug_text_mask',
            enabled_attr='_drug_text_enabled',
        )
        disease_text_embs = self._lookup_node_text_embeddings(
            global_ids=disease_global_ids,
            batch_size=pair_embs.size(0),
            device=pair_embs.device,
            dtype=pair_embs.dtype,
            matrix_attr='disease_text_matrix',
            mask_attr='disease_text_mask',
            enabled_attr='_disease_text_enabled',
        )

        path_hidden_states = self.path_value_proj(paths_embs)
        predicted_text_embs = self.text_projector(paths_embs)
        aggregated_paths, path_weights = self._aggregate_paths(
            pair_context=pair_context,
            path_hidden_states=path_hidden_states,
            predicted_text_embs=predicted_text_embs,
            attention_mask=attention_mask,
        )
        aggregated_paths, has_path_mask, _ = self._mask_and_gate_paths(
            aggregated_paths=aggregated_paths,
            attention_mask=attention_mask,
        )

        h_drug_gnn = pair_embs[:, 0, :]
        h_disease_gnn = pair_embs[:, 1, :]
        if self.ablate_gnn:
            h_drug_gnn = torch.zeros_like(h_drug_gnn)
            h_disease_gnn = torch.zeros_like(h_disease_gnn)
            aggregated_paths = torch.zeros_like(aggregated_paths)
        final_feature_blocks = [h_drug_gnn, h_disease_gnn, aggregated_paths]
        if self.use_external_late_fusion:
            final_feature_blocks.append(drug_fingerprint_embs)
        if self.use_drug_text_late_fusion:
            final_feature_blocks.append(drug_text_embs)
        if self.use_disease_text_late_fusion:
            final_feature_blocks.append(disease_text_embs)
        final_features = torch.cat(final_feature_blocks, dim=-1)
        logits = self.output_mlp(final_features).squeeze(-1)
        distill_loss = self._compute_distillation_loss(
            predicted_text_embs=predicted_text_embs,
            triplet_key_ids=triplet_key_ids,
            attention_mask=attention_mask,
        )
        path_loss = self._compute_path_margin_loss(
            h_drug_gnn=h_drug_gnn,
            h_disease_gnn=h_disease_gnn,
            aggregated_paths=aggregated_paths,
            has_path_mask=has_path_mask,
        ) if return_path_loss else logits.new_zeros(())

        if return_attention and return_distill_loss and return_path_loss:
            return logits, path_weights, distill_loss, path_loss
        if return_attention and return_distill_loss:
            return logits, path_weights, distill_loss
        if return_attention and return_path_loss:
            return logits, path_weights, path_loss
        if return_attention:
            return logits, path_weights
        if return_distill_loss and return_path_loss:
            return logits, distill_loss, path_loss
        if return_distill_loss:
            return logits, distill_loss
        if return_path_loss:
            return logits, path_loss
        return logits

    def _aggregate_paths(
        self,
        pair_context: Tensor,
        path_hidden_states: Tensor,
        predicted_text_embs: Tensor,
        attention_mask: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        if self.agg_type == 'attention':
            return self._attention_pool_paths(
                pair_context=pair_context,
                path_hidden_states=path_hidden_states,
                predicted_text_embs=predicted_text_embs,
                attention_mask=attention_mask,
            )
        if self.agg_type == 'mean':
            return self._mean_pool_paths(
                path_hidden_states=path_hidden_states,
                attention_mask=attention_mask,
            )
        if self.agg_type == 'max':
            return self._max_pool_paths(
                path_hidden_states=path_hidden_states,
                attention_mask=attention_mask,
            )
        raise RuntimeError(f'Unsupported agg_type: {self.agg_type}')

    def _mask_and_gate_paths(
        self,
        aggregated_paths: Tensor,
        attention_mask: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        has_path_mask = attention_mask.any(dim=1, keepdim=True)
        zero_paths = torch.zeros_like(aggregated_paths)
        silent_paths = torch.where(has_path_mask, aggregated_paths, zero_paths)

        path_gates = self.path_gate(silent_paths)
        zero_gates = torch.zeros_like(path_gates)
        path_gates = torch.where(has_path_mask, path_gates, zero_gates)
        gated_paths = torch.where(has_path_mask, silent_paths * path_gates, zero_paths)
        return gated_paths, has_path_mask, path_gates

    def _compute_path_margin_loss(
        self,
        h_drug_gnn: Tensor,
        h_disease_gnn: Tensor,
        aggregated_paths: Tensor,
        has_path_mask: Tensor,
    ) -> Tensor:
        if not self.training:
            return aggregated_paths.new_zeros(())

        valid_mask = has_path_mask.squeeze(-1)
        if valid_mask.dim() != 1:
            valid_mask = valid_mask.view(-1)
        if int(valid_mask.sum().item()) <= 1:
            return aggregated_paths.new_zeros(())

        pos_drug = h_drug_gnn[valid_mask]
        pos_disease = h_disease_gnn[valid_mask]
        pos_paths = aggregated_paths[valid_mask]
        num_valid = pos_paths.size(0)
        if num_valid <= 1:
            return aggregated_paths.new_zeros(())

        shuffle_indices = torch.roll(
            torch.arange(num_valid, device=pos_paths.device),
            shifts=1,
            dims=0,
        )
        neg_paths = pos_paths[shuffle_indices]

        pos_inputs = torch.cat([pos_drug, pos_disease, pos_paths], dim=-1)
        neg_inputs = torch.cat([pos_drug, pos_disease, neg_paths], dim=-1)
        pos_path_scores = self.path_scorer(pos_inputs).squeeze(-1)
        neg_path_scores = self.path_scorer(neg_inputs).squeeze(-1)
        margin_gap = pos_path_scores - neg_path_scores
        return torch.clamp(self.path_loss_margin - margin_gap, min=0.0).mean()


    def _attention_pool_paths(
        self,
        pair_context: Tensor,
        path_hidden_states: Tensor,
        predicted_text_embs: Tensor,
        attention_mask: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        query = self.query_mlp(pair_context)
        query_text = self.query_text_proj(query)
        raw_attention_scores = self.attn_mlp(
            torch.tanh(predicted_text_embs + query_text.unsqueeze(1))
        ).squeeze(-1)

        masked_attention_scores = raw_attention_scores.masked_fill(~attention_mask, -1e9)
        attention_weights = torch.softmax(masked_attention_scores, dim=1)
        attention_weights = attention_weights * attention_mask.to(dtype=path_hidden_states.dtype)
        normalizer = attention_weights.sum(dim=1, keepdim=True).clamp_min(1e-12)
        attention_weights = attention_weights / normalizer

        aggregated_paths = torch.sum(
            attention_weights.unsqueeze(-1) * path_hidden_states,
            dim=1,
        )
        return aggregated_paths, attention_weights

    def _mean_pool_paths(
        self,
        path_hidden_states: Tensor,
        attention_mask: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        path_mask = attention_mask.to(dtype=path_hidden_states.dtype)
        masked_paths = path_hidden_states * path_mask.unsqueeze(-1)
        valid_counts = path_mask.sum(dim=1, keepdim=True).clamp_min(1.0)

        aggregated_paths = masked_paths.sum(dim=1) / valid_counts
        mean_weights = path_mask / valid_counts
        return aggregated_paths, mean_weights

    def _max_pool_paths(
        self,
        path_hidden_states: Tensor,
        attention_mask: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        valid_any = attention_mask.any(dim=1, keepdim=True)
        masked_paths = path_hidden_states.masked_fill(~attention_mask.unsqueeze(-1), -1e9)
        max_values, max_indices = torch.max(masked_paths, dim=1)
        aggregated_paths = torch.where(valid_any, max_values, torch.zeros_like(max_values))

        path_selection_weights = torch.zeros(
            (path_hidden_states.size(0), path_hidden_states.size(1)),
            dtype=path_hidden_states.dtype,
            device=path_hidden_states.device,
        )
        path_selection_weights.scatter_add_(
            dim=1,
            index=max_indices,
            src=torch.ones_like(max_indices, dtype=path_hidden_states.dtype),
        )
        path_selection_weights = path_selection_weights / float(path_hidden_states.size(-1))
        path_selection_weights = path_selection_weights * valid_any.to(dtype=path_hidden_states.dtype)
        return aggregated_paths, path_selection_weights

    def _initialize_drug_fingerprints(
        self,
        drug_morgan_fingerprints_path: Path,
        nodes_csv_path: Optional[Path],
        max_global_id: Optional[int],
    ) -> None:
        if nodes_csv_path is None:
            raise ValueError('`nodes_csv_path` is required when `drug_morgan_fingerprints_path` is provided.')
        if max_global_id is None or max_global_id <= 0:
            raise ValueError('`max_global_id` must be provided when enabling drug fingerprints.')
        if not drug_morgan_fingerprints_path.exists():
            raise FileNotFoundError(
                f'Drug Morgan fingerprint file does not exist: {drug_morgan_fingerprints_path}'
            )
        if not nodes_csv_path.exists():
            raise FileNotFoundError(f'PrimeKG nodes.csv does not exist: {nodes_csv_path}')

        raw_id_to_global = self._build_raw_id_to_global_id_mapping(nodes_csv_path)
        with drug_morgan_fingerprints_path.open('rb') as file:
            fingerprint_dict = pickle.load(file)
        if not isinstance(fingerprint_dict, Mapping):
            raise TypeError('`drug_morgan_fingerprints.pkl` must contain a dictionary-like object.')

        fingerprint_matrix = torch.zeros((int(max_global_id), self.drug_fingerprint_dim), dtype=torch.float32)
        for raw_drug_id, fingerprint in fingerprint_dict.items():
            raw_drug_id = str(raw_drug_id)
            if raw_drug_id not in raw_id_to_global:
                continue

            fingerprint_tensor = torch.as_tensor(fingerprint, dtype=torch.float32).view(-1)
            if fingerprint_tensor.numel() != self.drug_fingerprint_dim:
                raise ValueError(
                    'Drug fingerprint shape mismatch: '
                    f'expected ({self.drug_fingerprint_dim},), got {tuple(fingerprint_tensor.shape)}.'
                )
            fingerprint_matrix[int(raw_id_to_global[raw_drug_id])] = fingerprint_tensor.detach().clone()

        self.drug_fingerprint_matrix = fingerprint_matrix
        self._drug_fingerprints_enabled = True

    def _lookup_drug_fingerprints(
        self,
        drug_global_ids: Optional[Tensor],
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor:
        fingerprint_batch = torch.zeros(
            (batch_size, self.drug_fingerprint_dim),
            device=device,
            dtype=dtype,
        )
        if (
            not self.use_external_late_fusion
            or not self._drug_fingerprints_enabled
            or drug_global_ids is None
            or self.drug_fingerprint_matrix.numel() == 0
        ):
            return fingerprint_batch

        lookup_ids = drug_global_ids.to(device=self.drug_fingerprint_matrix.device, dtype=torch.long)
        valid_mask = (lookup_ids >= 0) & (lookup_ids < self.drug_fingerprint_matrix.size(0))
        if not valid_mask.any():
            return fingerprint_batch

        gathered = self.drug_fingerprint_matrix[lookup_ids[valid_mask]].to(device=device, dtype=dtype)
        fingerprint_batch[valid_mask.to(device=device)] = gathered
        return fingerprint_batch.detach()


    def _initialize_node_text_embeddings(
        self,
        embeddings_path: Path,
        nodes_csv_path: Optional[Path],
        max_global_id: Optional[int],
        expected_dim: int,
        matrix_attr: str,
        mask_attr: str,
        enabled_attr: str,
    ) -> None:
        if nodes_csv_path is None:
            raise ValueError('`nodes_csv_path` is required when node text embeddings are provided.')
        if max_global_id is None or max_global_id <= 0:
            raise ValueError('`max_global_id` must be provided when enabling node text embeddings.')
        if not embeddings_path.exists():
            raise FileNotFoundError(f'Node text embedding file does not exist: {embeddings_path}')
        if not nodes_csv_path.exists():
            raise FileNotFoundError(f'PrimeKG nodes.csv does not exist: {nodes_csv_path}')

        raw_id_to_global = self._build_raw_id_to_global_id_mapping(nodes_csv_path)
        with embeddings_path.open('rb') as file:
            embedding_dict = pickle.load(file)
        if not isinstance(embedding_dict, Mapping):
            raise TypeError(f'Node text embedding file must contain a dictionary-like object: {embeddings_path}')

        text_matrix = torch.zeros((int(max_global_id), int(expected_dim)), dtype=torch.float32)
        text_mask = torch.zeros(int(max_global_id), dtype=torch.bool)
        for raw_id, embedding in embedding_dict.items():
            raw_id = str(raw_id)
            if raw_id not in raw_id_to_global:
                continue
            embedding_tensor = torch.as_tensor(embedding, dtype=torch.float32).view(-1)
            if embedding_tensor.numel() != int(expected_dim):
                raise ValueError(
                    'Node text embedding shape mismatch: ' 
                    f'expected ({int(expected_dim)},), got {tuple(embedding_tensor.shape)}.'
                )
            global_id = int(raw_id_to_global[raw_id])
            text_matrix[global_id] = embedding_tensor.detach().clone()
            text_mask[global_id] = True

        setattr(self, matrix_attr, text_matrix)
        setattr(self, mask_attr, text_mask)
        setattr(self, enabled_attr, True)

    def _lookup_node_text_embeddings(
        self,
        global_ids: Optional[Tensor],
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
        matrix_attr: str,
        mask_attr: str,
        enabled_attr: str,
    ) -> Tensor:
        text_matrix = getattr(self, matrix_attr)
        text_mask = getattr(self, mask_attr)
        expected_dim = int(text_matrix.size(1)) if text_matrix.dim() == 2 else 0
        text_batch = torch.zeros((batch_size, expected_dim), device=device, dtype=dtype)
        if not getattr(self, enabled_attr) or global_ids is None or text_matrix.numel() == 0:
            return text_batch

        lookup_ids = global_ids.to(device=text_matrix.device, dtype=torch.long)
        valid_mask = (lookup_ids >= 0) & (lookup_ids < text_matrix.size(0))
        if valid_mask.any():
            clamped_ids = lookup_ids.clamp(min=0, max=max(text_matrix.size(0) - 1, 0))
            valid_mask = valid_mask & text_mask[clamped_ids]
        if not valid_mask.any():
            return text_batch

        gathered = text_matrix[lookup_ids[valid_mask]].to(device=device, dtype=dtype)
        text_batch[valid_mask.to(device=device)] = gathered
        return text_batch.detach()

    def _initialize_triplet_text_teacher(
        self,
        triplet_text_embeddings_path: Path,
        nodes_csv_path: Optional[Path],
        max_global_id: Optional[int],
    ) -> None:
        if nodes_csv_path is None:
            raise ValueError('`nodes_csv_path` is required when `triplet_text_embeddings_path` is provided.')
        if max_global_id is None or max_global_id <= 0:
            raise ValueError('`max_global_id` must be provided when enabling triplet text distillation.')
        if not triplet_text_embeddings_path.exists():
            raise FileNotFoundError(
                f'Triplet text embedding file does not exist: {triplet_text_embeddings_path}'
            )
        if not nodes_csv_path.exists():
            raise FileNotFoundError(f'PrimeKG nodes.csv does not exist: {nodes_csv_path}')

        raw_id_to_global = self._build_raw_id_to_global_id_mapping(nodes_csv_path)
        with triplet_text_embeddings_path.open('rb') as file:
            embedding_dict = pickle.load(file)
        if not isinstance(embedding_dict, Mapping):
            raise TypeError('`triplet_text_embeddings.pkl` must contain a dictionary-like object.')

        code_to_embedding: Dict[int, Tensor] = {}
        self.triplet_hash_base = int(max_global_id)
        for raw_key, embedding in embedding_dict.items():
            if not isinstance(raw_key, tuple) or len(raw_key) != 3:
                continue
            if raw_key[0] not in raw_id_to_global or raw_key[1] not in raw_id_to_global or raw_key[2] not in raw_id_to_global:
                continue

            teacher_embedding = torch.as_tensor(embedding, dtype=torch.float32)
            if teacher_embedding.dim() != 1 or teacher_embedding.size(0) != self.text_embedding_dim:
                raise ValueError(
                    'Teacher triplet text embedding shape mismatch: '
                    f'expected ({self.text_embedding_dim},), got {tuple(teacher_embedding.shape)}.'
                )

            encoded_key = self._encode_triplet_code_values(
                drug_global_id=raw_id_to_global[raw_key[0]],
                disease_global_id=raw_id_to_global[raw_key[1]],
                gene_global_id=raw_id_to_global[raw_key[2]],
            )
            code_to_embedding[encoded_key] = teacher_embedding

        if not code_to_embedding:
            self.triplet_teacher_codes = torch.empty(0, dtype=torch.long)
            self.triplet_teacher_embeddings = torch.empty(
                (0, self.text_embedding_dim),
                dtype=torch.float32,
            )
            self._teacher_is_enabled = False
            return

        sorted_codes = sorted(code_to_embedding.keys())
        sorted_embeddings = [code_to_embedding[code] for code in sorted_codes]
        self.triplet_teacher_codes = torch.tensor(sorted_codes, dtype=torch.long)
        self.triplet_teacher_embeddings = torch.stack(sorted_embeddings, dim=0)
        self._teacher_is_enabled = True

    def _build_raw_id_to_global_id_mapping(self, nodes_csv_path: Path) -> Dict[str, int]:
        raw_id_to_global: Dict[str, int] = {}
        with nodes_csv_path.open('r', encoding='utf-8-sig', newline='') as node_file:
            reader = csv.DictReader(node_file)
            for global_id, row in enumerate(reader):
                raw_id = str(row['id']).strip()
                raw_id_to_global[raw_id] = global_id
        return raw_id_to_global

    def _encode_triplet_code_values(
        self,
        drug_global_id: int,
        disease_global_id: int,
        gene_global_id: int,
    ) -> int:
        hash_base = int(self.triplet_hash_base)
        return (
            int(drug_global_id) * hash_base * hash_base
            + int(disease_global_id) * hash_base
            + int(gene_global_id)
        )

    def _encode_triplet_key_ids(self, triplet_key_ids: Tensor) -> Tensor:
        hash_base = int(self.triplet_hash_base)
        drug_ids = triplet_key_ids[..., 0].to(dtype=torch.long)
        disease_ids = triplet_key_ids[..., 1].to(dtype=torch.long)
        gene_ids = triplet_key_ids[..., 2].to(dtype=torch.long)
        return drug_ids * hash_base * hash_base + disease_ids * hash_base + gene_ids

    def _lookup_teacher_text_embeddings(
        self,
        triplet_key_ids: Tensor,
        attention_mask: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        predicted_shape = (*triplet_key_ids.shape[:2], self.text_embedding_dim)
        if not self.has_text_teacher:
            empty_embeddings = self.triplet_teacher_embeddings.new_zeros(predicted_shape)
            empty_mask = attention_mask.new_zeros(attention_mask.shape)
            return empty_embeddings, empty_mask

        encoded_keys = self._encode_triplet_key_ids(triplet_key_ids).reshape(-1)
        teacher_codes = self.triplet_teacher_codes
        teacher_embeddings = self.triplet_teacher_embeddings

        insertion_indices = torch.searchsorted(teacher_codes, encoded_keys)
        clamped_indices = insertion_indices.clamp(max=max(int(teacher_codes.numel()) - 1, 0))
        hit_mask_flat = insertion_indices < teacher_codes.numel()
        if teacher_codes.numel() > 0:
            hit_mask_flat = hit_mask_flat & (teacher_codes[clamped_indices] == encoded_keys)

        true_text_embeddings = teacher_embeddings.new_zeros((encoded_keys.numel(), self.text_embedding_dim))
        if hit_mask_flat.any():
            true_text_embeddings[hit_mask_flat] = teacher_embeddings[clamped_indices[hit_mask_flat]]

        hit_mask = hit_mask_flat.view(attention_mask.shape) & attention_mask
        true_text_embeddings = true_text_embeddings.view(*predicted_shape)
        true_text_embeddings = true_text_embeddings * hit_mask.unsqueeze(-1).to(dtype=true_text_embeddings.dtype)
        return true_text_embeddings, hit_mask

    def _compute_distillation_loss(
        self,
        predicted_text_embs: Tensor,
        triplet_key_ids: Optional[Tensor],
        attention_mask: Tensor,
    ) -> Tensor:
        zero = predicted_text_embs.new_zeros(())
        if triplet_key_ids is None or not self.has_text_teacher:
            return zero

        true_text_embs, hit_mask = self._lookup_teacher_text_embeddings(
            triplet_key_ids=triplet_key_ids,
            attention_mask=attention_mask,
        )
        if not hit_mask.any():
            return zero

        predicted_valid = predicted_text_embs[hit_mask]
        teacher_valid = true_text_embs[hit_mask]
        if self.distill_loss_type == 'mse':
            return F.mse_loss(predicted_valid, teacher_valid)

        cosine_similarities = F.cosine_similarity(predicted_valid, teacher_valid, dim=-1, eps=1e-8)
        return 1.0 - cosine_similarities.mean()

    def _validate_inputs(
        self,
        pair_embs: Tensor,
        paths_embs: Tensor,
        attention_mask: Tensor,
        triplet_key_ids: Optional[Tensor],
        drug_global_ids: Optional[Tensor],
        disease_global_ids: Optional[Tensor],
    ) -> None:
        if pair_embs.dim() != 3:
            raise ValueError(
                '`pair_embs` must have shape `(batch_size, 2, pair_emb_dim)`, '
                f'got {tuple(pair_embs.shape)}.'
            )
        if pair_embs.size(1) != 2:
            raise ValueError('`pair_embs` second dimension must be 2 for [drug, disease].')
        if pair_embs.size(2) != self.pair_emb_dim:
            raise ValueError(
                '`pair_embs` last dimension does not match `pair_emb_dim`: '
                f'expected={self.pair_emb_dim}, actual={pair_embs.size(2)}.'
            )

        if paths_embs.dim() != 3:
            raise ValueError(
                '`paths_embs` must have shape `(batch_size, max_K, path_emb_dim)`, '
                f'got {tuple(paths_embs.shape)}.'
            )
        if paths_embs.size(2) != self.path_emb_dim:
            raise ValueError(
                '`paths_embs` last dimension does not match `path_emb_dim`: '
                f'expected={self.path_emb_dim}, actual={paths_embs.size(2)}.'
            )

        if attention_mask.dim() != 2:
            raise ValueError(
                '`attention_mask` must have shape `(batch_size, max_K)`, '
                f'got {tuple(attention_mask.shape)}.'
            )
        if pair_embs.size(0) != paths_embs.size(0) or pair_embs.size(0) != attention_mask.size(0):
            raise ValueError('`pair_embs`, `paths_embs`, and `attention_mask` must share the same batch size.')
        if paths_embs.size(1) != attention_mask.size(1):
            raise ValueError('`paths_embs` and `attention_mask` must share the same max_K dimension.')

        if triplet_key_ids is not None:
            if triplet_key_ids.dim() != 3 or triplet_key_ids.size(-1) != 3:
                raise ValueError(
                    '`triplet_key_ids` must have shape `(batch_size, max_K, 3)`, '
                    f'got {tuple(triplet_key_ids.shape)}.'
                )
            if triplet_key_ids.shape[:2] != attention_mask.shape:
                raise ValueError(
                    '`triplet_key_ids` must align with `attention_mask` in the first two dimensions: '
                    f'{tuple(triplet_key_ids.shape[:2])} vs {tuple(attention_mask.shape)}.'
                )

        if drug_global_ids is not None:
            if drug_global_ids.dim() != 1:
                raise ValueError(
                    '`drug_global_ids` must have shape `(batch_size,)`, '
                    f'got {tuple(drug_global_ids.shape)}.'
                )
            if drug_global_ids.size(0) != pair_embs.size(0):
                raise ValueError(
                    '`drug_global_ids` must align with the batch size of `pair_embs`: '
                    f'{drug_global_ids.size(0)} vs {pair_embs.size(0)}.'
                )

        if disease_global_ids is not None:
            if disease_global_ids.dim() != 1:
                raise ValueError(
                    '`disease_global_ids` must have shape `(batch_size,)`, '
                    f'got {tuple(disease_global_ids.shape)}.'
                )
            if disease_global_ids.size(0) != pair_embs.size(0):
                raise ValueError(
                    '`disease_global_ids` must align with the batch size of `pair_embs`: '
                    f'{disease_global_ids.size(0)} vs {pair_embs.size(0)}.'
                )


