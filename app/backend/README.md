# FoodLens Backend

This backend provides the FoodLens API contract while the real PyTorch inference
service is being prepared.

## Run Locally

Install runtime dependencies:

```bash
pip install -r requirements.txt
```

Install development dependencies before running tests:

```bash
pip install -r requirements-dev.txt
```

Install the optional detector dependency for live multi-food analysis:

```bash
pip install -r requirements-detector.txt
```

Install the analysis-tooling dependencies (numpy/pandas) needed to run
`scripts/recalibrate_decision_layer.py` -- these are not part of the deployed
backend, so they are kept out of `requirements.txt`:

```bash
pip install -r requirements-analysis.txt
```

`requirements-lock.txt` is a local reproducibility snapshot, not a CI
constraint: it records the exact versions the pinned dev environment resolved
to on macOS/arm64, for reproducing a known-good local environment. CI
deliberately installs the unpinned files above so upstream breakage surfaces
early rather than being masked by a frozen, platform-specific resolution.

Live detection also needs the YOLO weights file `yolo11n.pt` in the repo root.
It is gitignored (5.6 MB) so a fresh clone will not have it, or set
`FOODLENS_DETECTOR_WEIGHTS` to an existing path. The demo fallback is driven
by the `ultralytics` dependency, not by the weights file: when `ultralytics`
is absent -- as it is in CI, which does not install
`requirements-detector.txt` -- the backend serves the deterministic demo
fallback. When `ultralytics` is installed but the local checkpoint is
missing, `detect_candidate_regions()` calls `YOLO()` with no existence gate,
so Ultralytics attempts to download the checkpoint; a failed download or
failed inference surfaces as an error rather than a silent fallback.

Start the API:

```bash
uvicorn app.backend.api:app --reload --port 8000
```

Run backend tests:

```bash
python3 -m pytest tests/backend -v
```

Health check:

```text
http://127.0.0.1:8000/health
```

## Endpoints

```text
GET /health
GET /runtime/status
POST /predict/image
POST /predict/multi-food/image
POST /predict/multi-food/image-url
POST /predict/multi-food/youtube-url
POST /predict/video
```

The runtime status endpoint reports classifier artifact readiness, optional
calibration/policy artifacts, detector dependency availability, detector weight
resolution, and the effective multi-food mode. Use it when diagnosing why an
environment is returning live inference, detector-only classifier fallback, or
demo fallback responses.

The single-image endpoint uses real artifacts when available and fallback
predictions when they are not. The multi-food upload and direct image URL
endpoints return the Notebook 8 app contract with detected regions, crop-level
predictions, decision bands, and artifact references. The YouTube URL endpoint
samples frames server-side and sends those frames through the same multi-food
image path.

The multi-food path uses live YOLO proposals plus crop classification when
`ultralytics` and classifier artifacts are available, marking responses with
`detector_status: live_yolo`. When YOLO is available but classifier artifacts
are missing, it still returns real uploaded-image crops and marks classifier
labels with `detector_status: live_yolo_classifier_fallback` and
`fallback_reason: missing_classifier_artifacts`. It falls back to a
deterministic prototype response marked with `detector_status: fallback_demo`
when image decoding or the detector runtime is unavailable. Fallback responses
include `fallback_reason` so clients can distinguish deterministic demo data,
detector-only crops, and live inference.

You can switch the detector label acceptance policy with
`FOODLENS_DETECTOR_LABELS`:

- unset: use the default COCO-aware food labels (`apple`, `banana`, `bowl`, ...).
- `FOODLENS_DETECTOR_LABELS="*"`: accept every detector label.
- `FOODLENS_DETECTOR_LABELS="label1,label2,label3"`: accept only the listed labels.

This is useful when you start using a food-tuned detector with class names that are
outside the default list.

The legacy video upload endpoint remains deterministic mock output and returns
`fallback_reason: video_mock` until live backend video inference is implemented.

The frontend implements video review by sampling key frames client-side and
calling `POST /predict/multi-food/image` for each extracted frame.

## Real Inference Integration

To move from mock inference to real inference, place artifacts outside git under
`app/artifacts/`.

Required artifacts:

- the classifier checkpoint: `resnet50_ft_v2_best.pth`, or the file an
  optional `model.json` manifest names
- ordered class names
- calibration temperature
- decision policy
- hard-class list
- confusion-pair list

`model.json` selects the classifier: `{"architecture": "resnet50" |
"convnext_tiny", "checkpoint": "<file name>", "model_name": "<name>"}`, plus an
optional `model_run` the deploy script records. Without it the backend serves
ResNet50 from `resnet50_ft_v2_best.pth` as `resnet50_ft_v2`, exactly as before
the manifest existed; a present but invalid manifest fails with
`classifier_load_error` rather than falling back to ResNet50. Do not write it by
hand: promote a model together with its policy through
`scripts/deploy_decision_policy.py --model-run`, and roll back with `--restore`.
That script is a thin CLI over `app/deployment/`, which reads the artifacts
through this package's own readers; nothing in `app/backend/` imports
`app/deployment/`.
`/runtime/status` reports the served model in its `model` block. See
`docs/8_runtime_contract.md`.

The multi-food path also uses detector weights through the `ultralytics` runtime.
Set `FOODLENS_DETECTOR_WEIGHTS` to override the default `yolo11n.pt` detector.
When the environment variable is not set, the backend searches parent
directories for `yolo11n.pt` before allowing Ultralytics to use its default
download behavior.

Classifier artifacts are resolved in this order:

1. `FOODLENS_ARTIFACT_DIR` when set.
2. The local worktree `app/artifacts/` directory when it contains the required
   checkpoint and class-name files.
3. Parent repository `app/artifacts/` directories when working from a git
   worktree.

This keeps model files out of branch worktrees while still allowing the API to
run live inference from artifacts in the main repository checkout.
