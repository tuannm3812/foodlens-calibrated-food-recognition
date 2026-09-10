# S2 — Backend Inference Decomposition (Design)

Date: 2026-09-11
Status: Approved by decomposition

## 1. Purpose

`app/backend/inference.py` is 886 lines carrying artifact loading, detector
configuration and invocation, classifier loading and inference, demo/fallback
response construction, image handling, runtime status, and the two public
prediction entry points. Split it into modules with one responsibility each,
without changing any observable behaviour.

## 2. The constraint that shapes everything

**The existing tests patch attributes on the `inference` module**, and they must
keep working. Measured targets in `tests/backend/`:

| Patched | Used by |
| --- | --- |
| `inference.ARTIFACT_DIR` | artifact resolution tests |
| `inference.__file__` | `artifact_dir_path()` and `detector_weights_path()` repo-root discovery |
| `inference.artifact_status` | fallback-reason tests |
| `inference.artifact_dir_path` | runtime-status tests |
| `inference.detect_candidate_regions` | multi-food live-path tests |
| `inference.detector_weights_path` | detector status tests |
| `inference.build_multi_food_mock` | demo-fallback tests |

This yields a hard rule:

> A function may move to a new module **only if every call site stays in
> `inference.py`**, referencing the re-exported name.

Python resolves a module-level name at call time from the *calling* module's
globals. So `monkeypatch.setattr(inference, "detect_candidate_regions", fake)`
still takes effect when the function lives in `detector.py` but is called from
`inference.py` via `from .detector import detect_candidate_regions`. It stops
taking effect the moment the call site itself moves out of `inference.py` — and
it fails **silently**, leaving a test that passes while exercising the real
implementation. That is the single most dangerous outcome of this refactor.

Two functions cannot move at all: `artifact_dir_path()` and
`detector_weights_path()` both resolve the repo root from `Path(__file__)`. A
test patches `inference.__file__`; if the function reads `artifacts.__file__`
instead, the patch becomes a no-op. Runtime behaviour would be unchanged — every
`app/backend/*.py` sits at the same depth, so `parents[1]` is identical — but the
test would silently stop testing anything. They stay in `inference.py`.

## 3. Coverage first

`inference.py` is at **59% statement coverage** (122 of 300 statements
unexecuted). Restructuring 41%-dark code is not safe, so this work is staged:

**Phase 1 — characterization tests, no production change.** Raise coverage on
the logic that is about to move, pinning current behaviour before it moves.
Priority targets, all currently uncovered and all pure or file-driven:

- `detector_region_role` (241, 245-250) and `should_export_detection` (260-268)
  — pure branching policy, trivially testable, currently untested.
- `detector_label_filter_config` (222-231) — environment-driven.
- `read_policy` (163-164), `read_hard_classes` (177-178),
  `read_confusion_pairs` (183-190) — file-driven, testable with `tmp_path`.
- `build_classifier_fallback_predictions` (625-628).
- `build_crop_data_url` (614-615).

**Phase 2 — decomposition.** Move code only after Phase 1 lands. The Phase 1
tests are the proof that Phase 2 changed nothing.

## 4. Target structure

| Module | Responsibility | Moves in |
| --- | --- | --- |
| `artifacts.py` | Reading and validating artifact files | `read_json`, `read_temperature`, `read_policy`, `read_hard_classes`, `read_confusion_pairs`, `artifact_file_status`, `classifier_artifacts_ready` |
| `detector_policy.py` | Which detections become regions | `detector_label_filter_config`, `detector_region_role`, `should_export_detection`, the detector constants and label sets |
| `detection.py` | Running the detector | `detect_candidate_regions` |
| `imaging.py` | Image decode and crop encoding | `open_rgb_image`, `build_crop_data_url`, `build_full_image_region` |
| `classifier.py` | Model construction and scoring | `make_classifier_head`, `build_predictions` |
| `demo.py` | Deterministic no-model responses | `predict_mock`, `build_multi_food_mock`, `build_classifier_fallback_predictions` |
| `inference.py` | Public API, orchestration, `__file__`-based path resolution | keeps `artifact_status`, `artifact_dir_path`, `detector_weights_path`, `runtime_status`, `load_runtime`, `classify_pil_image`, `build_multi_food_response`, `build_multi_food_classifier_fallback_response`, `build_prediction_response`, `predict_image_bytes`, `predict_multi_food_image_bytes`, `ARTIFACT_DIR`, and re-exports every moved name |

`app/backend/api.py` imports only `predict_image_bytes`, `predict_mock`,
`predict_multi_food_image_bytes`, `runtime_status` from `.inference`. All four
remain importable from there, so `api.py` is untouched.

Expected result: `inference.py` drops from 886 lines to roughly 450, with the
rest distributed across six focused modules.

## 5. What must not change

- Any HTTP response shape. `docs/8_runtime_contract.md` is the contract.
- The three existing `tests/backend/` files. They must pass **unmodified** —
  that is the proof the refactor is behaviour-preserving. Phase 1 adds new test
  files; it does not edit existing ones.
- Public names importable from `app.backend.inference`.
- Runtime behaviour when artifacts or `ultralytics` are missing.

## 6. Acceptance criteria

1. `pytest` green, with the three original backend test files unmodified
   (`git diff --stat` shows no change to them).
2. `inference.py` under 500 lines.
3. Coverage of `app/backend/` **higher than the 59%/68% baseline**, reported
   before and after.
4. Every patched name in §2 still resolves via `app.backend.inference`, and a
   dedicated test proves each patch **actually takes effect** rather than merely
   resolving — a no-op patch is the failure mode this guards.
5. `ruff check .` clean; `check_doc_links.py` exits 0.
6. `api.py` unmodified.
7. No new dependency added.

## 7. Risks

- **A silently ineffective monkeypatch.** The whole reason for criterion 4.
  Verifying an attribute exists is not enough; the test must observe the fake
  being called.
- **Import cycles.** `inference` imports from the new modules; none of them may
  import `inference`. Anything needing an orchestration-level value takes it as
  a parameter.
- **Constants split across modules.** Detector constants move to
  `detector_policy.py`, but `DETECTOR_WEIGHTS` is used by
  `detector_weights_path()`, which stays. It must remain reachable from
  `inference.py` and keep its current value.

## 8. Behaviour found during Phase 1

Both are pinned by the new tests as *current* behaviour, not fixed. S2 is a
no-behaviour-change refactor, so correcting them belongs to a later change where
the fix can be reviewed on its own merits.

- **`read_json` does not guard `json.loads`.** A malformed artifact file raises
  `JSONDecodeError` uncaught rather than degrading to the demo fallback. That is
  a production robustness gap, not merely a test gap: a truncated or
  half-written `decision_policy.json` takes the API down instead of falling back.
  Worth fixing, separately.
- **`detector_region_role(filter_mode="configured", configured_labels=None)`
  silently falls back to default label behaviour** rather than treating every
  label as a context object, because the guard is `configured_labels is not
  None`. Easy to miss, and it makes a misconfigured filter look like a working
  one.

Also corrected: an earlier working note claimed `should_export_detection`
applies `DETECTOR_CONFIDENCE_THRESHOLD`. It does not — it takes no confidence
parameter at all, and confidence filtering happens only through YOLO's `conf=`
argument at the call site.
