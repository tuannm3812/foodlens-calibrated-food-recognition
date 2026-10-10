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

Master §1 asks for a small root. Three additions are deliberate:

- `app/` — the product runtime (FastAPI backend, React frontend, artifacts).
- `kaggle/` — the Kaggle run records. Each directory is one training run's
  script plus its notebook, kept so a result traces to the code that produced it.
- `configs/` — one tracked file, `expanded_taxonomy_v1.json`, the canonical
  label mapping for the 130-class expanded taxonomy. Notebooks read it, so the
  taxonomy work is not reproducible without it.

`data/`, `results/`, and `models/` remain untracked, per master §1 and §8.

## `E402` is intended in `kaggle/`

Scripts under `kaggle/` mirror notebook cell order, where imports follow the
configuration block so the config is visible at the top of the run log. This is
the intended Kaggle style, not debt. Enforced as a `per-file-ignores` entry in
`pyproject.toml` rather than being "fixed".

`E501` is ignored in the same directory for the same reason — those scripts
keep long configuration and metric-printing lines that mirror the notebook
cells they came from.

Jupyter notebooks are excluded from ruff altogether (`extend-exclude` in
`pyproject.toml`). Cell-based code inherently trips `E402`, `F821` (a name
bound in an earlier cell) and `F404`, and notebooks here are execution
records rather than maintained source.

`B008` is ignored in `app/backend/api.py`. A call in an argument default —
`File(...)`, `Form(...)`, `Depends(...)` — is how FastAPI declares request
parameters, so "fixing" it would break request parsing.

## Seven notebooks retain outputs

Master §4 permits keeping notebook outputs when they are intentionally preserved
as evidence. These seven do, and their outputs must not be cleared:

- `notebooks/05_confidence_decision_layer.ipynb`
- `notebooks/06_food_recognition_demo_inference.ipynb`
- `notebooks/08_detection_to_foodlens_pipeline.ipynb`
- `notebooks/archive/01_food101_baseline_transfer_finetuning.ipynb`
- `notebooks/archive/02_resnet50_training_refinements.ipynb`
- `notebooks/archive/03_modern_backbone_comparison.ipynb`
- `notebooks/archive/07_multi_food_detection_exploration.ipynb`

Three of the seven are active notebooks, not archived ones, so this is not an
"archive only" exemption. Every other tracked notebook currently has no outputs,
and the master rule applies to them: clear outputs when code changes.
