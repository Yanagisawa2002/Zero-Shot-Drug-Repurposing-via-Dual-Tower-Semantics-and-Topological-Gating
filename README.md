# Multi-Modal Knowledge Graph Drug Repositioning

This repository contains a research codebase for drug repositioning with multi-modal biomedical knowledge graphs. The project combines graph neural networks, molecular fingerprints, and biomedical text representations to study internal pairwise prediction, external Open Targets generalization, and high-order mechanism ranking.

## What This Project Implements

- Heterogeneous graph encoder for PrimeKG-style biomedical graphs
- Strict pair-level graph surgery to prevent direct drug-disease leakage
- Asymmetric modality routing and dual-tower text/chemistry fusion variants
- DropEdge, path gating, distillation, and pair-fixed HO evaluation
- Systematic multi-seed training and ablation pipelines

## Repository Layout

- `src/`: model, graph surgery, samplers, training utilities
- `scripts/`: training, evaluation, feature extraction, and audit scripts
- `tests/`: regression tests for core training and scoring logic

## Data and Model Weights

This public repository is intentionally code-first. Large datasets, intermediate embeddings, checkpoints, and local model weights are **not** tracked in Git.

Examples of excluded assets:

- PrimeKG raw and processed graph files
- Open Targets export files
- Frozen experiment directories and training logs
- SapBERT / PubMedBERT local weights
- Precomputed `.pkl`, `.pt`, and checkpoint artifacts

If you want to reproduce the experiments, prepare the required files locally under `data/`, `models/`, and `outputs/` following your own licensed access to the upstream resources.

## Quick Start

1. Create a Python environment with PyTorch, PyG, transformers, pandas, scikit-learn, and pytest.
2. Place the required graph/data files under `data/` and local model weights under `models/`.
3. Run training via `scripts/train_quad_split_ho_probe.py`.
4. Run audits / evaluations via scripts in `scripts/`.

## Notes for Recruiters / Reviewers

This repo is structured as a research engineering project rather than a polished library package. The main value is in:

- graph-data cleaning and leakage prevention
- multimodal feature injection and ablation design
- evaluation under strict OOD and mechanism-ranking settings
- experiment automation and reproducibility checks

## License and Redistribution

Before redistributing any data or derived artifacts, verify the license terms of the upstream sources you used to build them. This repository should only include materials you are legally allowed to publish.
