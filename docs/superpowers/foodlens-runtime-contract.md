# FoodLens Runtime Contract

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
