# 8. Runtime Contract

## `/runtime/status`

Response shape:

- `classifier`
  - `status`: `"ready"` when both `resnet50_ft_v2_best.pth` and `class_names.json` are
    present under resolved artifact directory, otherwise `"missing_artifacts"`.
  - `artifact_status`: `"ready"` or `"mock"` mirror of classifier readiness.
  - `artifact_dir`: resolved directory path used for runtime.
  - `artifacts`: per-file status (`exists`, `size_bytes`, `path`) for:
    - `checkpoint`
    - `class_names`
    - `calibration`
    - `decision_policy`
    - `hard_classes`
    - `confusion_pairs`
- `detector`
  - `status`: `"ready"` when `ultralytics` import is available, otherwise `"missing_dependency"`.
  - `dependency`: fixed string (`"ultralytics"`).
  - `dependency_available`: boolean import check result.
  - `weights_path`: resolved detector path.
    - explicit override via `FOODLENS_DETECTOR_WEIGHTS`
    - repo/root discovery fallback
    - otherwise package default path string
  - `weights_found`: whether `weights_path` exists on disk.
  - `weights_source`: `"environment"`, `"auto_discovered"`, or `"ultralytics_default"`.
  - `label_filter`:
    - `mode`: `"all"`, `"configured"`, or `"default"`
    - `labels`: configured labels when mode is `"configured"`
- `multi_food`
  - `mode`:
    - `"live_yolo_classifier"` when classifier and detector dependency are both available.
    - `"detector_only_classifier_fallback"` when detector is available but classifier artifacts are not.
    - `"demo_fallback"` when detector dependency is unavailable.
  - `detector_status`:
    - `"live_yolo"` for full classifier+detector path
    - `"live_yolo_classifier_fallback"` for detector-only path with classifier fallback labels
    - `"fallback_demo"` when both paths are unavailable

## Multi-food response fields

`/predict/multi-food/*` responses use `MultiFoodPredictionResponse` with:

- `detector_status`:
  - `"fallback_demo"` for deterministic mock data
  - `"live_yolo_classifier_fallback"` when detector proposals exist but classification is unavailable
  - `"live_yolo"` for full live path
  - `"live_yolo_whole_image_fallback"` when no detections pass filters and full image is used as one region
- `artifact_status`: `"ready"` or `"mock"` to indicate classifier-artifact dependence
- `fallback_reason`: present when `artifact_status` is `"mock"` or detection path is constrained

Common fallback reasons:

- `"missing_artifacts"`
- `"missing_classifier_artifacts"`
- `"detector_runtime_unavailable"`
- `"detector_inference_error"`
- `"invalid_image"`
- `"no_detector_regions"`
- `"classifier_load_error"`
- `"classifier_inference_error"`
- `"video_mock"`

## URL ingestion behavior

- Image URL endpoint requires public direct image URLs and validates host/IP safety.
- YouTube URL endpoint requires public URL and local optional dependencies:
  - `yt-dlp`
  - `ffmpeg`
- URL input errors return `400`.
- Missing media dependencies return `503`.
- Ingestion failures return `400` for user-facing URL/media issues.

## Detector weights

Live detection needs `yolo11n.pt` — the YOLO11-nano checkpoint from
Ultralytics, 5.6 MB. It is gitignored by the `*.pt` rule, so a fresh clone
does not have it, and `weights_found` reports `false` until it is present.

Resolution order, reflected in `weights_source`:

1. `"environment"` — the `FOODLENS_DETECTOR_WEIGHTS` environment variable.
2. `"auto_discovered"` — `yolo11n.pt` found at the repo root.
3. `"ultralytics_default"` — the package default path. Ultralytics downloads
   the checkpoint on first use when it resolves here.

With no weights and no `ultralytics` install, `/predict/multi-food/*` serves
the deterministic demo fallback (`detector_status: "fallback_demo"`). That is
the expected state in CI, which does not install the detector extra.

## Deployed decision policy

`app/artifacts/` is gitignored, so the policy the backend serves is not in
version control. This section records which one is deployed and how to verify
it.

**Current state (deployed 2026-10-09 UTC):** champion **ResNet50 FT-V2** with
the corrected decision policy from the symmetric closure run
`results/accuracy_phase1/champion_resnet50_ft_v2/decision_layer_closure_2026-10-10/`.

| Field | Value |
| --- | --- |
| `auto_confidence` / `suggest_confidence` / `margin_threshold` | 0.70 / 0.35 / 0.05 |
| Hard classes | 11, bottom 10% by validation F1 |
| Confusion pairs | 40, top by validation error count |
| Source provenance SHA-256 | `b1068b6e4c999a9f6e6ff4cdd8bcd7af916b9a480347327ad84df91df2fdcce5` |
| Deployed `decision_policy.json` SHA-256 | `377a98891a443faa3f2e76c3b9a1fbb9f963ec93447f693be6397b8a1673e465` |
| Deployed `hard_classes.json` SHA-256 | `541b414421e18985b69dd5cfc357704bb482bdc77a8972e9d6d780d522ce329b` |
| Deployed `confusion_pairs.json` SHA-256 | `80fc4b0adbdc7908c15c07aec41e715258357297859c3d6151602815d5793f24` |

Measured on the held-out test split with that policy: auto-accept 61.20% at
94.58% top-1, review 2.54%. Section 16 of `3_model_results.md` holds the full
tables. The model checkpoint, `class_names.json` and `calibration.json`
(temperature 0.958111) were not changed.

The replaced legacy policy (margin 0.40, 15 hard classes, 30 pairs, written
2026-05-30 by Notebook 5 with the withdrawn methodology) is preserved in
`app/artifacts/replaced_20261009T224436Z/`. Measured the corrected way, it
auto-accepted 59.02% at 94.83% and routed 11.52% of images to review — not the
58.02% / 96.47% it had been credited with.

**Deploying a policy.** Use `scripts/deploy_decision_policy.py --source <run dir>`
(add `--dry-run` to validate only). Never copy recalibration outputs by hand:
the recalibration script writes `decision_policy.json` as a one-element list,
while `read_policy()` expects a dict. A verbatim copy makes `load_runtime()`
raise, and its `except Exception` turns every request into a demo fallback with
`fallback_reason: classifier_load_error` — while `artifact_status` still
reports `ready`. The deploy script converts the format, validates labels against
the served model's class names, backs up what it replaces, reads the result back
through the backend's own readers, and writes
`app/artifacts/deployment_provenance.json`.

**Verifying the deployed state.** Compare `deployment_provenance.json` against
the hashes above. The 2026-10-09 deployment was also verified end to end: 400
test images posted to `/predict/image` (sent as PNG to avoid re-encoding)
agreed with the closure run's offline routing on top-1 class and decision band
for 400 of 400, with no fallbacks.
