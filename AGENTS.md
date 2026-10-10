# foodlens-calibrated-food-recognition

Calibrated food **image recognition**: Food-101 classifiers whose confidence is
temperature-scaled and routed through a decision layer (auto-accept / suggest /
confirm / review), plus a FastAPI + React workbench for multi-food analysis.

Not `1. Study/ai-meal-planner` (also food + FastAPI + ML, but plans meals).

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
- `docs/10_decisions.md` — key decisions, open ones first; `9_agent_log.md` is the trail

## Current state

As of 2026-10-10: champion **A3b ConvNeXt-Tiny** (promoted over ResNet50
FT-V2; more accurate, ECE 0.0556 vs 0.0265 — D-013). Deploy models and policies
only with `scripts/deploy_decision_policy.py`, then restart the API and run
`--verify-live`; copied files fail silently. Roll back with its `--restore`.

## Open risks

- `app/backend/inference.py` is still 636 lines of orchestration. Deploy logic
  lives in `app/deployment/` behind a thin CLI; keep its import rules, which
  `tests/test_deployment_boundaries.py` enforces.
- **S1 resolved the triplicated recalibration CLI**: one copy now, at
  `scripts/recalibrate_decision_layer.py`. The four training scripts stay
  duplicated by design — immutable run records, and Kaggle needs
  self-contained notebooks — see `docs/0_coding_standards.md`. A run's record
  is its kernel `code_file` (A4: the `.py`); the other file is a drifting
  mirror, and what to do with mirrors is open decision O-3.
- `yolo11n.pt` (5.6 MB, repo root) is gitignored and required at runtime for
  live detection. A fresh clone will not have it — see `app/backend/README.md`.
