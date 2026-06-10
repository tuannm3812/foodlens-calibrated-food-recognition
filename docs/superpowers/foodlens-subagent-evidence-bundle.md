# FoodLens Multi-Agent Evidence Bundle

## Launch Matrix (Execute when you say “Go”)

### Track A — Modeling + Decision-Readiness
- Confirm artifact precedence and policy source:
  - [ ] `cat app/backend/decision.py`
  - [ ] `cat app/backend/inference.py`
- Confirm champion rationale:
  - [ ] Validate current product default remains `resnet50_ft_v2`.
  - [ ] Record evidence that ConvNeXt-Tiny needs ECE recalibration before promotion.

### Track B — API + Runtime Hardening
- Backend status contract:
  - [ ] Confirm route availability in `app/backend/api.py`.
  - [ ] Confirm runtime contract fields in `app/backend/inference.py::runtime_status`.
- Failure mode consistency:
  - [ ] Confirm URL endpoints map validation/dependency/ingestion errors to expected HTTP status codes.
  - [ ] Confirm image/video fallback reasons remain deterministic (`missing_artifacts`, `video_mock`, etc.).

### Track C — Multi-Food Product Pipeline
- End-to-end regional behavior:
  - [ ] Confirm detector + fallback pipeline in `app/backend/inference.py`.
  - [ ] Verify region role assignment and whole-image fallback path.
- Frame aggregation and YouTube ingestion:
  - [ ] Verify `combine_frame_responses` behavior in `app/backend/youtube_ingestion.py`.

### Track D — Frontend Workbench
- Workflow and UX sanity checks:
  - [ ] `app/frontend/src/state/useAnalyzer.ts`
  - [ ] `app/frontend/src/components/UploadControls.tsx`
  - [ ] `app/frontend/src/components/AnalyzerWorkbench.tsx`
  - [ ] `app/frontend/src/api/foodlensClient.ts`
- Validate source-label + decision-band rendering.

### Track E — QA + Release Readiness
- Contract tests and smoke order:
  - [ ] `pytest tests/backend/test_api_contract.py -q`
  - [ ] `pytest tests/backend/test_url_ingestion.py -q`
  - [ ] `pytest tests/backend/test_decision.py -q`
  - [ ] Frontend quick run: `cd app/frontend && npm run build`
- Risk register update:
  - [ ] Document blocked paths: detector quality, multi-food label coverage, optional ffmpeg dependency.

### Track F — Docs + Packaging
- Documentation and onboarding:
  - [ ] Update top-level completion note in `README.md`.
  - [ ] Update roadmap in `docs/05_next_steps.md` and include artifact/run requirements.
  - [ ] Keep all model decision and fallback assumptions explicit in one page.

## Completion Output Template

Use this for each track:

1. What was implemented / confirmed.
2. Files touched.
3. Command evidence (pass/fail + summary).
4. Risks and follow-ups.

