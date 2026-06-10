# FoodLens Backend

This backend provides the FoodLens API contract while the real PyTorch inference
service is being prepared.

## Run Locally

Install runtime dependencies in your preferred environment:

```bash
pip install -r app/backend/requirements.txt
```

Install backend development dependencies before running tests:

```bash
pip install -r app/backend/requirements-dev.txt
```

Install the optional detector dependency when testing live multi-food image
analysis:

```bash
pip install -r app/backend/requirements-detector.txt
```

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

- `resnet50_ft_v2_best.pth`
- ordered class names
- calibration temperature
- decision policy
- hard-class list
- confusion-pair list

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
