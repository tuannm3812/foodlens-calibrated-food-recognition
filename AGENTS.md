# foodlens-calibrated-food-recognition

Calibrated food **image recognition**: Food-101 classifiers whose confidence is
temperature-scaled and routed through a decision layer (auto-accept / suggest /
confirm / review), plus a FastAPI + React workbench for multi-food analysis.

Not to be confused with `1. Study/ai-meal-planner`, another food-domain FastAPI
+ ML repo: that one plans meals from user profiles. Neither shares code.

## Standards

Follow the master standard at `~/Documents/GitHub/coding-standards/`.
Project-specific rules and deliberate overrides: @docs/0_coding_standards.md

## Deltas from the master

- **Shape A, despite shipping a product app.** Six of the ten numbered docs are
  modelling docs and the README leads with the champion model, so the centre of
  gravity is modelling. `app/` is the product surface, not the centre.
- **`app/` and `kaggle/` are legitimate root directories** beyond master §1's
  list. The product runtime and the Kaggle run records both need a home in git.
- **`E402` is intended in `kaggle/`.** Those scripts mirror notebook cell order,
  where imports follow the configuration block. Enforced as a per-file-ignore.
- **Seven notebooks deliberately retain outputs** as evidence, which
  master §4 permits but does not default to.

## Evidence locations

- `docs/3_model_results.md` — every metric; any accuracy claim traces to a row here
- `docs/8_runtime_contract.md` — the API surface and runtime status semantics
- `docs/9_agent_log.md` — append-only session history

## Current state

As of 2026-09-10: product champion is **ResNet50 FT-V2** (83.90% is A3b's
number, not the champion's — see `docs/3_model_results.md`). **A3b
ConvNeXt-Tiny is the accuracy leader but is blocked from promotion** pending
decision-layer recalibration. Do not promote it without redoing calibration.

## Open risks

- `app/backend/inference.py` is 886 lines doing artifact loading, detection,
  classification, and response assembly. Decomposition is planned as S2 — don't
  bolt more onto it.
- **S1 resolved the triplicated recalibration CLI**: one copy now, at
  `scripts/recalibrate_decision_layer.py`. The four training scripts stay
  duplicated by design — immutable run records, and Kaggle needs
  self-contained notebooks — see `docs/0_coding_standards.md`. Their `.py`
  mirrors sit at 97.8-98.1% of their notebooks; reconciling or removing them
  is still an open decision.
- `yolo11n.pt` (5.6 MB, repo root) is gitignored and required at runtime for
  live detection. A fresh clone will not have it — see `app/backend/README.md`.
