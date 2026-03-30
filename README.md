# Zero-Shot Drug Repurposing via Dual-Tower Semantics and Topological Gating

This repository contains a research codebase for **zero-shot drug repurposing** with **multi-modal biomedical knowledge graphs**. The project studies how to combine heterogeneous graph topology, molecular fingerprints, and biomedical mechanism text under a **strict pair-level clean** protocol that removes direct drug-disease shortcut edges.

The current public code focuses on three evaluation regimes:
- internal pairwise prediction
- external Open Targets generalization
- held-out high-order mechanism ranking

## Method Overview

The main modeling line combines:
- **Disease-side semantic prior (Early Fusion):** disease mechanism text is encoded with SapBERT and injected into disease nodes before GNN message passing.
- **Drug-side chemistry anchor (Late Fusion):** Morgan fingerprints are fused only at the scorer stage to preserve sharp chemical discrimination.
- **Dual-tower semantic alignment:** optional SapBERT drug text and SapBERT disease text can be fused symmetrically at the scoring head.
- **Topological gating:** path-level aggregation uses zero-path masking and adaptive gating so empty mechanism sets do not inject noise.
- **Strict pair-level clean graph surgery:** held-out drug-disease shortcut edges are removed before training.

## Visual Summary

The figures below are generated from multi-seed experiment summaries under the strict pair-level clean protocol.

![Performance overview](assets/readme/performance_overview.svg)

![Cross-disease focus](assets/readme/cross_disease_focus.svg)

Figure generation script:
- `python scripts/generate_readme_figures.py`

## Selected Results

### Dual-tower main line across the three splits

| Split | Pair AUROC | OT AUROC | HO AUPRC |
|---|---:|---:|---:|
| `random` | `0.9873 ¡À 0.0018` | `0.8500 ¡À 0.0109` | `0.3323 ¡À 0.0054` |
| `cross_drug` | `0.8854 ¡À 0.0202` | `0.8122 ¡À 0.0161` | `0.3194 ¡À 0.0072` |
| `cross_disease` | `0.8014 ¡À 0.0134` | `0.6956 ¡À 0.0240` | `0.3078 ¡À 0.0039` |

### Cross-disease ablation snapshot

| Variant | Pair AUROC | OT AUROC | HO AUPRC |
|---|---:|---:|---:|
| `Pure GNN` | `0.6423 ¡À 0.0211` | `0.7482 ¡À 0.0199` | `0.2409 ¡À 0.0167` |
| `Single Tower` | `0.7613 ¡À 0.0523` | `0.7355 ¡À 0.0304` | `0.3236 ¡À 0.0165` |
| `Dual Tower` | `0.8014 ¡À 0.0134` | `0.6956 ¡À 0.0240` | `0.3078 ¡À 0.0039` |
| `No-GNN` | `0.8828 ¡À 0.0154` | `0.7236 ¡À 0.0063` | `0.3197 ¡À 0.0019` |
| `Path Loss` | `0.8224 ¡À 0.0074` | `0.7161 ¡À 0.0089` | `0.3193 ¡À 0.0091` |

## Main Takeaways

- Disease-side semantic priors are the strongest driver of **zero-shot cross-disease** generalization.
- Dual-tower fusion improves internal pairwise performance over a pure graph baseline, but introduces a non-trivial tradeoff with external OT calibration.
- Strict pair-level clean evaluation materially changes conclusions compared with looser graph-cleaning settings.
- Path-gated mechanism aggregation is important when many test pairs have zero valid mechanism paths after strict cleaning.

## Repository Layout

- `src/`: model, graph surgery, samplers, and training utilities
- `scripts/`: training, evaluation, audits, feature extraction, and figure generation
- `tests/`: regression tests for core logic
- `assets/readme/`: README figures and compact summary metrics used for public presentation

## Quick Start

1. Create a Python environment and install dependencies from `requirements.txt`.
2. Place required graph/data files under `data/` and local model weights under `models/`.
3. Run training with `python scripts/train_quad_split_ho_probe.py`.
4. Run audits and evaluations with scripts under `scripts/`.
5. Regenerate README figures with `python scripts/generate_readme_figures.py`.

## Data and Model Weights

This repository is intentionally **code-first**. Large datasets, local model weights, checkpoints, and intermediate embeddings are not tracked in Git.

Examples of excluded assets:
- PrimeKG raw and processed graph files
- Open Targets export files
- local SapBERT / PubMedBERT checkpoints
- experiment logs and frozen training snapshots
- derived `.pkl`, `.pt`, `.pth`, `.db`, `.sqlite`, and large output artifacts

If you want to reproduce the experiments, prepare the required files locally under `data/`, `models/`, and `outputs/` following the licenses of the upstream resources.

## Why This Repo Is Useful on a Resume

This project is structured as a research engineering codebase rather than a small tutorial package. The strongest engineering contributions are:
- graph-data cleaning and leakage prevention
- multimodal feature routing under OOD constraints
- mechanism-aware evaluation beyond plain AUROC
- large ablation orchestration with reproducibility-focused scripts and tests

## License and Redistribution

Before redistributing any data or derived artifacts, verify the license terms of the upstream sources you used to build them. This repository should only include materials you are legally allowed to publish.
