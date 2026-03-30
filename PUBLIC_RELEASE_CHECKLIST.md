# Public Release Checklist

Use this checklist before pushing the repository to GitHub.

## Keep in the public repo

- `src/`
- `scripts/`
- `tests/`
- `README.md`
- `requirements.txt`
- high-level summary docs that do not redistribute restricted data

## Do not commit

- `data/`
- `models/`
- `outputs/`
- `frozen_experiments/`
- `open target/`
- local `.pkl`, `.pt`, `.pth`, `.db`, `.sqlite`, `.log` artifacts
- temporary folders such as `feature_utils_*`, `pytest-cache-files-*`, `tmp_test_artifacts/`

## Before pushing

1. Verify `.gitignore` is active and excludes large artifacts.
2. Scan for hard-coded local paths like `D:/Models/gemini` and `C:/Users/...`.
3. Confirm no API keys, tokens, or credentials are present.
4. Confirm you are allowed to reference upstream datasets and models.
5. Keep README focused on:
   - project goal
   - core method
   - setup
   - how to run one training job
   - what users must download themselves

## Recommended repo structure

- `src/`: model and training code
- `scripts/`: train, eval, audit, and feature extraction entry points
- `tests/`: regression tests
- `README.md`: public-facing overview
- `requirements.txt`: minimum dependencies

## Recommended README sections

- Overview
- Method summary
- Repository layout
- Setup
- Data preparation
- Training
- Evaluation
- Notes on licenses / redistribution

## Recommended final sanity checks

- `python -m py_compile scripts/*.py src/*.py`
- `pytest -q tests`
- `git status`
