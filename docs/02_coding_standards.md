# 0. Coding Standards

The baseline for this repo is the master standard at
`~/Documents/GitHub/coding-standards/coding_standards.md`. This file records
**only** where this project deliberately differs. Anything not listed here
follows the master.

## Doc shape: A

Master §2 asks a repo that is both a modelling project and a product to pick a
centre of gravity and declare it. This repo picks **Shape A**. Six of the ten
numbered docs are modelling docs, and the README leads with the champion model
and its metrics. The FastAPI + React app in `app/` is the product surface, not
the centre of gravity.

## Root directories beyond master §1

Master §1 asks for a small root. Two additions are deliberate:

- `app/` — the product runtime (FastAPI backend, React frontend, artifacts).
- `kaggle/` — the Kaggle run records. Each directory is one training run's
  script plus its notebook, kept so a result traces to the code that produced it.

`data/`, `results/`, and `models/` remain untracked, per master §1 and §8.

## `E402` is intended in `kaggle/`

Scripts under `kaggle/` mirror notebook cell order, where imports follow the
configuration block so the config is visible at the top of the run log. This is
the intended Kaggle style, not debt. Enforced as a `per-file-ignores` entry in
`pyproject.toml` rather than being "fixed".

## Four archived notebooks retain outputs

Master §4 permits keeping notebook outputs when they are intentionally preserved
as evidence. These four are, and their outputs must not be cleared:

- `notebooks/05_confidence_decision_layer.ipynb`
- `notebooks/06_food_recognition_demo_inference.ipynb`
- `notebooks/08_detection_to_foodlens_pipeline.ipynb`
- `notebooks/archive/03_modern_backbone_comparison.ipynb`

Every other notebook follows the master rule: clear outputs when code changes.
