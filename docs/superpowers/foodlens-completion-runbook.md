# FoodLens Completion Runbook

## 1) Objective

Finish all tracks with deterministic checks, independent agent ownership, and a
single source of truth for evidence.

## 2) Mandatory baseline checks

- Install backend deps in a normal environment:
  - `pip install -r app/backend/requirements.txt`
  - `pip install -r app/backend/requirements-dev.txt` (optional for tests)
  - `pip install -r app/backend/requirements-detector.txt` (optional)
  - `cd app/frontend && npm install`
- Start backend:
  - `python3 -m uvicorn app.backend.api:app --reload --port 8000`
- Start frontend:
  - `cd app/frontend && npm run dev`

## 3) Acceptance checklists by track

### Track A — Modeling + Decision-Readiness
- Confirm backend still runs with:
  - `MODEL_NAME = \"resnet50_ft_v2\"`
  - `read_policy()` returning `decision_policy.json` when present.
- Validate docs state that ConvNeXt-Tiny is accuracy leader but promoted only after
  E2 recalibration and product criteria check.

### Track B — API + Runtime Hardening
- Confirm route and contract:
  - `GET /health`
  - `GET /runtime/status`
  - `POST /predict/image`
  - `POST /predict/multi-food/image`
  - `POST /predict/multi-food/image-url`
  - `POST /predict/multi-food/youtube-url`
- Record field meanings:
  - `classifier.status`
  - `classifier.artifact_status`
  - `detector.status`
  - `detector.dependency_available`
  - `multi_food.mode`
  - `multi_food.detector_status`

### Track C — Multi-Food Product Pipeline
- Verify crop-level contracts:
  - proposal role mapping stays valid
  - full-image fallback includes whole-image region
  - detector-only fallback uses `live_yolo_classifier_fallback`
- Verify YouTube combine semantics:
  - responses are concatenated
  - `source_id` and `detection_index` are rewritten deterministically

### Track D — Frontend Workbench
- Validate source mode and URL mode flows:
  - image upload
  - video mode sampling
  - image URL submission
  - YouTube URL submission
- Confirm fallback behavior:
  - 4xx/validation failures render as errors
  - other failures render local demo data

### Track E — QA + Release Readiness
- Execute smoke pass:
  - `pytest tests/backend/test_api_contract.py -q`
  - `pytest tests/backend/test_url_ingestion.py -q`
  - `pytest tests/backend/test_decision.py -q`
  - `cd app/frontend && npm run build`
- Collect command output as release evidence.

### Track F — Docs + Packaging
- Confirm completion notes describe:
  - champion policy rationale
  - onboarding prerequisites
  - detector/model artifact requirements
- Confirm model artifacts expected under:
  - `app/artifacts/resnet50_ft_v2_best.pth`
  - `app/artifacts/class_names.json`
  - `app/artifacts/calibration.json`
  - `app/artifacts/decision_policy.json`
  - `app/artifacts/hard_classes.json`
  - `app/artifacts/confusion_pairs.json`
- Confirm optional runtime weight path strategy:
  - `FOODLENS_DETECTOR_WEIGHTS` override
  - yolo weight auto-discovery from repo root

## 4) Final handoff

- Update [docs/superpowers/plans/2026-06-11-foodlens-subagent-execution-board.md]
  with each track checkmark and evidence references.
- Add one-page completion summary covering:
  - what passed
  - what is deferred
  - known risks before deployment
- Tag blockers explicitly if any remain unresolved.
