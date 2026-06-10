# FoodLens Completion - Subagent Execution Board

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to execute this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Complete finalization of FoodLens with production-ready model selection, decision policy, backend contract hardening, multi-food pipeline reliability, frontend UX closure, and QA/release packaging.

**Architecture:** Keep the existing FastAPI + React/Vite architecture. Backend owns inference artifacts/policy/contract; frontend owns interactive workbench and review UX. Notable coupling points are runtime status, fallback semantics, and region-level decision band presentation.

**Tech Stack:** Python, FastAPI, PyTorch, Pydantic, Ultralytics optional, yt-dlp, React/Vite, TypeScript.

---

## Active Track Assignments

### Agent A — Modeling + Decision-Readiness Lead
- Owner: `Modeling Agent`
- Focus: A3b vs. ResNet50 decision, expanded-taxonomy candidate status, and recalibration coverage.
- Status: In progress
- Deliverable: `[ ]` Final champion recommendation with calibrated risk trade-off.
- Acceptance:
  - `[ ]` Confirm current production champion remains ResNet50 FT-V2 in backend defaults.
  - `[ ]` Add evidence note in docs for why ConvNeXt-Tiny promotion is deferred pending E2 recalibration.
  - `[ ]` Verify `read_policy()` and artifact precedence for future model switches are documented.

### Agent B — API + Runtime Hardening
- Owner: `Backend/API Agent`
- Focus: `/health`, `/runtime/status`, and inference fallback determinism.
- Status: In progress
- Deliverable: `[ ]` Backend contract matrix + any required code/docs updates.
- Acceptance:
  - `[ ]` Validate runtime response field semantics (`status`, `artifact_status`, `multi_food.mode`, `multi_food.detector_status`) in docs.
  - `[ ]` Confirm all non-2xx API paths return stable fallback reasons or 4xx/5xx details.
  - `[ ]` Document local-demo and dependency-driven fallback behavior in one page.
- Documentation target: [docs/superpowers/foodlens-runtime-contract.md](docs/superpowers/foodlens-runtime-contract.md)

### Agent C — Multi-Food Product Pipeline
- Owner: `Detection Agent`
- Focus: detector+crop pipeline reliability, schema stability, and overlap/region policy.
- Status: In progress
- Deliverable: `[ ]` End-to-end multi-food output consistency and review behavior.
- Acceptance:
  - `[ ]` Confirm region role/label mapping remains stable for direct food, serving container, and fallback regions.
  - `[ ]` Confirm `build_full_image_region` fallback path still emits consistent crop artifacts.
  - `[ ]` Verify detector-only mode returns explicit `live_yolo_classifier_fallback` with deterministic labels.

### Agent D — Frontend Workbench Owner
- Owner: `Frontend Agent`
- Focus: image/video/upload/URL workflow, decision-band visibility, and responsive polish.
- Status: In progress
- Deliverable: `[ ]` Stable analyzer behavior across all input modes and source contexts.
- Acceptance:
  - `[ ]` Validate analyzer mode transitions for image/video and URL/YouTube inputs remain deterministic.
  - `[ ]` Confirm decision band label and detector-status copy remains aligned with backend values.
  - `[ ]` Confirm error/fallback UX still communicates 4xx messages separately from local demo fallback.

### Agent E — QA + Release Readiness
- Owner: `QA Agent`
- Focus: tests, smoke checklist, reproducibility commands, and risk register.
- Status: In progress
- Deliverable: `[ ]` Executable verification bundle with evidence artifacts.
- Acceptance:
  - `[ ]` Create one command-set to verify API contract and frontend compile/build.
  - `[ ]` Record current known risks and non-blocking gaps (dependency, detector quality).
  - `[ ]` Publish a one-screen runbook for local onboarding and fallback scenarios.

### Agent F — Docs + Packaging
- Owner: `Docs Agent`
- Focus: README, plan docs, model result summary, and deployment packaging instructions.
- Status: In progress
- Deliverable: `[ ]` Final completion brief and runbook.
- Acceptance:
  - `[ ]` Produce final model champion summary for the current codebase.
  - `[ ]` Add packaging checklist for artifacts + detector weights.
- Documentation targets:
  - [README.md](../../README.md)
  - [docs/05_next_steps.md](../05_next_steps.md)
  - [docs/superpowers/foodlens-completion-runbook.md](foodlens-completion-runbook.md)

---

## Subagent Dispatch Ledger

1. [Modeling Agent] Task A1: Validate champion decision policy.
2. [Backend/API Agent] Task B1: Runtime contract audit against `/runtime/status`.
3. [Detection Agent] Task C1: Multi-food fallback-state matrix (live vs fallback paths).
4. [Frontend Agent] Task D1: URL/video/source-context UX regression pass.
5. [QA Agent] Task E1: Smoke checklist and evidence bundle.
6. [Docs Agent] Task F1: Completion brief and onboarding packaging docs.

---

## Execution Log (Auto-updated by handoff)

1. [x] Track A: confirm champion policy source for runtime and product decision calibration.
2. [x] Track B: confirm runtime status field semantics and fallback messages in API.
3. [x] Track C: confirm detector region filtering and multi-food fallback behavior across missing-artifact cases.
4. [x] Track D: validate all analyzer modes and URL/video controls on error and fallback paths.
5. [x] Track E: perform smoke checklist (backend + frontend + model-artifact scenarios).
6. [x] Track F: publish final progress and completion notes.

### Live completion status page

- [foodlens-completion-status.md](foodlens-completion-status.md)

---

## Coordination Cadence

- Default cadence: one iteration per track before switching focus.
- Escalation: if a track is blocked twice, convert to `BLOCKED` and move to explicit blocker section.
- Completion condition: every track has a green check and no unresolved blocker in final state.
