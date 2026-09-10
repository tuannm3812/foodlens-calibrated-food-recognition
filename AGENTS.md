# foodlens-calibrated-food-recognition

Calibrated food **image recognition**: Food-101 classifiers whose confidence is
temperature-scaled and routed through a decision layer (auto-accept / suggest /
confirm / review), plus a FastAPI + React workbench for multi-food analysis.

Not to be confused with `1. Study/ai-meal-planner`, which is also a food-domain
FastAPI + ML repo with notebooks and a frontend. That one plans meals from user
profiles. This one recognises dishes in images and decides how much to trust the
prediction. Neither shares code with the other.

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
- `kaggle/*/recalibrate_decision_layer.py` exists as three byte-identical copies,
  and the four training scripts are near-duplicates — 46 to 282 changed lines
  between any pair of files that are each about 700 lines long. Fixing this is S1.
- `yolo11n.pt` (5.6 MB, repo root) is gitignored and required at runtime for
  live detection. A fresh clone will not have it — see `app/backend/README.md`.
