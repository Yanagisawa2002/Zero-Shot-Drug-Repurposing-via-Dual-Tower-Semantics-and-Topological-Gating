# Current SOTA Model Summary

## 1. Scope

This document summarizes the current strongest model configuration in this repository as of 2026-03-27.

Current frozen SOTA snapshot:
- `frozen_experiments/asym_route_dropedge_sota_20260327`

Core idea:
- Disease semantic prior is injected at the graph encoder input stage (early fusion).
- Drug chemical prior is injected at the scorer stage (late fusion).
- Multi-path mechanism evidence is aggregated with attention.
- A weak triplet-text teacher is used as auxiliary distillation.
- DropEdge regularization is applied only during training.

This is the configuration that currently gives the best overall OOD behavior, especially on `cross_disease`.

## 2. Main Files

Core implementation files:
- `src/repurposing_rgcn.py`
- `src/pair_aggregation_scorer.py`
- `src/pair_path_bpr_sampler.py`
- `src/training_utils.py`
- `src/graph_surgery.py`
- `src/evaluation_utils.py`
- `scripts/train_quad_split_ho_probe.py`
- `scripts/train_quad_split_ot_probe.py`
- `scripts/evaluate_ho_mechanisms.py`

External feature files actually used by the model:
- `drug_morgan_fingerprints.pkl`
- `thick_disease_text_embeddings.pkl`
- `triplet_text_embeddings.pkl`
- `outputs/pubmedbert_hybrid_features_clean`

## 3. Data Representation

### 3.1 Graph Backbone

The base graph is PrimeKG, loaded as a PyG `HeteroData` object.

The model uses heterogeneous node types including at least:
- `drug`
- `gene/protein`
- `disease`
- `pathway`

The current mechanism unit is a star-style quad:
- `(drug, gene/protein, pathway, disease)`

This quad is not treated as a graph edge. It is treated as a mechanism path candidate used by the scorer.

### 3.2 Pair-Level Training Target

The prediction target is still pairwise drug-disease association.

For each positive `(drug, disease)` pair, the model collects multiple mechanism paths between them and scores the pair through those paths.

### 3.3 Multi-Path Batch Structure

The DataLoader produces pair-level batches with:
- `pos_pair_ids`: shape `(B, 2)`
- `pos_paths`: shape `(B, K, 4)`
- `pos_attention_mask`: shape `(B, K)`
- `neg_pair_ids`: shape `(B, 2)`
- `neg_paths`: shape `(B, K_neg, 4)`
- `neg_attention_mask`: shape `(B, K_neg)`

Path column order in quad mode is:
- column 0: `drug`
- column 1: `gene/protein`
- column 2: `pathway`
- column 3: `disease`

## 4. Data Sampling and Negatives

### 4.1 PairPathBPRDataset

Implemented in:
- `src/pair_path_bpr_sampler.py`

The sampler works at the pair level rather than the single-path level.

For each positive pair, it collects all available mechanism paths and forms a multi-path sample.

### 4.2 Topology-Aware Hard Negatives

The sampler precomputes graph topology so that negative pairs are not trivial no-path negatives.

For `cross_drug` and `cross_disease`, negatives are sampled from graph-connected candidate sets so that the fake pair still tends to have real mechanism paths.

This avoids the old failure mode where the scorer could exploit dummy-path absence as a shortcut.

### 4.3 Pathway Expansion

In quad mode, each `(drug, gene, disease)` mechanism is expanded into one or more `(drug, gene, pathway, disease)` quads by enumerating pathways connected to the gene.

If a gene has no pathway edge, a dummy pathway id is used so that the real triplet mechanism is not dropped.

## 5. Leakage-Control Protocol

### 5.1 Current Choice: Direct-Only Graph Surgery

Implemented in:
- `src/graph_surgery.py`

Current training and evaluation use:
- `graph_surgery_mode = direct_only`

This means:
- remove direct held-out `drug-disease` shortcut edges
- keep lower-level mechanism edges such as:
  - `drug-gene`
  - `gene-disease`
  - `gene-pathway`

This was chosen because the earlier strict graph surgery over-deleted mechanism context and hurt realistic performance.

### 5.2 What Is Removed

Only direct leakage relations are cut for held-out pairs, e.g.:
- `indication`
- `off-label use`
- `contraindication`
- and their reverse directions

### 5.3 What Is Preserved

The following remain available to message passing:
- drug-target-gene edges
- disease-gene edges
- gene-pathway edges
- other biological context edges

This preserves biologically realistic background structure while still removing the most direct label shortcut.

## 6. Node Feature Initialization

### 6.1 Generic Node Inputs

Every node type has two possible base sources:
- projected external feature `x_dict[node_type]` if provided
- otherwise a learned random embedding `nn.Embedding(num_nodes, hidden_dim)`

There is also a generic per-type feature projector:
- `nn.Linear(in_channels, hidden_channels)`

### 6.2 Disease Early Fusion

This is the key early-fusion branch that is currently kept.

Source file:
- `thick_disease_text_embeddings.pkl`

Construction logic:
- disease names are enriched with ontology-like context from Open Targets
- only safe disease-intrinsic fields are used
- no drugs or treatment labels are injected

Encoder-side injection:
- disease 768-dim text vector
- `disease_proj: Linear(768 -> hidden_dim)`
- then `LayerNorm`
- then optional dropout in the overwrite step using model dropout

This disease feature overwrites the disease rows in `h^(0)` before graph message passing.

### 6.3 Drug Late Fusion

Drug Morgan fingerprints are **not** injected into the graph encoder input.

Instead, they are used as a late-fusion wide branch at the scorer stage.

Source file:
- `drug_morgan_fingerprints.pkl`

Dimension:
- `1024`

Reason:
- earlier experiments showed that injecting Morgan into `h^(0)` caused excessive smoothing of drug chemical information
- keeping Morgan outside the graph helps preserve sharp chemical discrimination

### 6.4 Gene Text Features

Gene text embedding extraction was implemented and tested, but it is **not part of the current frozen SOTA**.

Reason:
- even after template cleanup and mean pooling, the gene text embeddings remained highly anisotropic globally
- adding them to the current SOTA line did not yield stable gains

## 7. Encoder Architecture

### 7.1 Encoder Type

Current SOTA uses:
- `encoder_type = rgcn`

The actual implementation is a 2-layer heterogeneous GraphSAGE-style encoder built with `HeteroConv`.

Layer 1:
- relation-specific `SAGEConv((hidden_dim, hidden_dim) -> hidden_dim)`

Layer 2:
- relation-specific `SAGEConv((hidden_dim, hidden_dim) -> out_dim)`

Aggregation across relations:
- `sum`

### 7.2 Initial Residual Connection

To reduce oversmoothing, each graph layer blends graph-updated features with the initial hidden state.

For each layer `l`:
- compute graph update `H_temp`
- blend with initial hidden features `H0`
- `H_l = alpha * H_temp + (1 - alpha) * H0_proj`

Where:
- `alpha` is learnable
- implemented through `conv1_alpha_logit` and `conv2_alpha_logit`
- initialized from `initial_residual_alpha = 0.2`

This keeps the encoder anchored to the original semantic input.

### 7.3 DropEdge

Current SOTA adds training-only DropEdge:
- `dropedge_p = 0.15`

Implementation details:
- applied before message passing
- applied relation by relation to the hetero `edge_index_dict`
- active only when `self.training == True`
- eval uses the full graph

This is currently one of the most important contributors to OOD gains, especially in `cross_disease`.

## 8. Scorer Architecture

### 8.1 Input to the Scorer

For each pair batch, `RepurposingRGCN.score_batch()` extracts:
- `h_drug_gnn`
- `h_disease_gnn`
- per-path concatenated node features for all paths

In quad mode, each path embedding is:
- concat of `drug`, `gene/protein`, `pathway`, `disease` node embeddings

### 8.2 Path Representation Branch

Inside `PairAggregationScorer`:
- path embeddings first go through `path_value_proj`
- another branch goes through `text_projector`

The projected text-space path features are used for two purposes:
- attention scoring
- optional weak distillation toward teacher triplet text embeddings

### 8.3 Attention Pooling

Current SOTA uses:
- `agg_type = attention`

Mechanism:
- build `pair_context` from graph-updated pair embeddings
- project pair context into the text space
- compute attention logits for each path
- apply mask over padded paths
- softmax over valid paths
- weighted sum of path hidden states to produce `h_paths`

### 8.4 Final Pair Classification Head

Current final representation is:
- `h_drug_gnn`
- `h_disease_gnn`
- `h_paths`
- `morgan_raw_feat`

That is:
- graph-updated drug node embedding
- graph-updated disease node embedding
- attention-aggregated path embedding
- raw 1024-dim Morgan fingerprint

This is the current asymmetric-routing classifier:
- disease semantics enter early through the graph
- drug chemistry enters late at the scorer

## 9. Weak Outer-HO Text Distillation

### 9.1 Teacher Source

Teacher file:
- `triplet_text_embeddings.pkl`

This contains PubMedBERT embeddings of cleaned triplet-level mechanism text:
- keyed by `(drug_id, disease_id, protein_id)`

### 9.2 Student Mechanism

For each candidate path in a batch:
- scorer predicts a path text embedding via `text_projector`
- if the `(drug, disease, gene)` triplet key exists in teacher memory, the model computes distillation loss

Current loss uses similarity matching between:
- predicted path text feature
- teacher triplet text feature

### 9.3 Why It Is Weak Distillation

Current weight:
- `text_distill_alpha = 0.2`

This means the text teacher is treated as an auxiliary regularizer, not the main learning objective.

## 10. Training Objective

Implemented in:
- `src/training_utils.py`

Current SOTA uses:
- `primary_loss_type = bce`

For each batch:
- compute positive scores
- compute negative scores
- primary loss = BCE over concatenated pos/neg logits
- total loss = primary loss + `alpha * distill_loss`

So the actual objective is:
- `TotalLoss = BCE + 0.2 * DistillLoss`

Why BCE rather than BPR in the final line:
- BCE was more stable for the final strongest configuration when combined with late-fused Morgan and weak text distillation

## 11. Evaluation Protocols

### 11.1 Internal Pairwise Evaluation

Implemented in:
- `src/evaluation_utils.py`

For a given split, evaluation uses the held-out positive mechanisms and three 1:1 negative settings:
- `random`
- `cross_drug`
- `cross_disease`

Metrics:
- Pairwise Accuracy
- AUROC
- AUPRC

Important interpretation:
- for each split, the main reported internal metric is the matching setting:
  - `random` split -> `random`
  - `cross_drug` split -> `cross_drug`
  - `cross_disease` split -> `cross_disease`

### 11.2 OT External Generalization

External OT evaluation uses pair-clean novel triplets relative to the corresponding training split.

This means OT novelty is enforced at the pair level, not only at the triplet level.

### 11.3 HO Mechanism Evaluation

Implemented in:
- `scripts/evaluate_ho_mechanisms.py`

Task:
- each real positive quad is evaluated against four independently corrupted negatives:
  - corrupt drug
  - corrupt gene
  - corrupt pathway
  - corrupt disease

This forms a `1:4` mechanism ranking benchmark.

Metrics:
- AUPRC
- Hit@1
- MRR

This is much harder than pairwise association prediction and is used to measure fine-grained mechanism precision.

## 12. Current Frozen SOTA Results

These are the best-checkpoint results from the frozen snapshot.

### 12.1 Random Split

- best epoch: `40`
- pair AUROC: `0.9778`
- OT AUROC: `0.8639`
- HO AUPRC: `0.3540`

### 12.2 Cross-Drug Split

- best epoch: `40`
- pair AUROC: `0.8615`
- OT AUROC: `0.8419`
- HO AUPRC: `0.3134`

### 12.3 Cross-Disease Split

- best epoch: `50`
- pair AUROC: `0.8661`
- OT AUROC: `0.7683`
- HO AUPRC: `0.2987`

## 13. What the Ablations Show

Paper-level core ablations have already been run around this SOTA line.

Main conclusions:
- `DropEdge` is a major contributor, especially for `cross_disease`
- `Morgan` is also a major contributor, especially for OOD and HO stability
- `Disease Early Fusion` helps HO and overall semantic grounding, but is not uniformly best for every OT metric in every split
- `Triplet Distillation` is useful but task-dependent; it is not the single most critical component

Interpretation:
- the hardest OOD gains are mostly driven by structural regularization plus chemistry
- weak text supervision helps, but it is not the main engine of the final gains

## 14. Final Architectural Summary

The current model can be summarized as:

1. Build a clean training graph by removing direct held-out drug-disease shortcut edges.
2. Construct pair-level multi-path quad samples with topology-aware hard negatives.
3. Initialize node states with:
   - projected node features if available
   - otherwise learned embeddings
4. Inject thick disease text into disease node rows at encoder input.
5. Run 2-layer hetero GraphSAGE with:
   - initial residual blending
   - training-only DropEdge
6. For each pair, extract all mechanism paths and attention-pool them.
7. Concatenate:
   - graph drug embedding
   - graph disease embedding
   - aggregated path embedding
   - raw drug Morgan fingerprint
8. Predict pairwise association score.
9. Train with BCE plus weak triplet-text distillation.
10. Evaluate on internal pairwise, external OT, and HO mechanism ranking.

## 15. Practical Reproduction Recipe

If reproducing the current SOTA, keep these choices fixed:
- `graph_surgery_mode = direct_only`
- `encoder_type = rgcn`
- `agg_type = attention`
- `use_early_external_fusion = True`
- disease early fusion on
- drug Morgan late fusion on
- `dropedge_p = 0.15`
- `primary_loss_type = bce`
- `text_distill_alpha = 0.2`
- `epochs = 60`
- choose checkpoint by validation AUROC on the split-matched setting

## 16. Components That Are Deliberately Not in the Current SOTA

These were tested but are not part of the frozen strongest line:
- gene text early fusion
- ontology smoothing edge bias
- test-time adaptation
- wide-and-deep re-injection of disease semantics at the scorer top
- path L2 normalization + path dropout branch
- several heavier distillation variants

They are excluded because they did not produce stable net gains relative to the current SOTA.

