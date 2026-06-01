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
POST /predict/image
POST /predict/multi-food/image
POST /predict/video
```

The single-image endpoint uses real artifacts when available and fallback
predictions when they are not. The video endpoint remains deterministic mock
output and returns `fallback_reason: video_mock` until live video inference is
implemented. The multi-food endpoint returns the Notebook 8 app contract with
detected regions, crop-level predictions, decision bands, and artifact
references. It uses live YOLO proposals plus crop classification when
`ultralytics` is installed and marks responses with `detector_status:
live_yolo`. It falls back to a deterministic prototype response marked with
`detector_status: fallback_demo` when the detector runtime is unavailable.
Fallback responses include `fallback_reason` so clients can distinguish
deterministic demo data from live inference.

The frontend implements video review by sampling key frames client-side and
calling `POST /predict/multi-food/image` for each extracted frame.

## Real Inference Integration

To move from mock inference to real inference, place artifacts outside git under
`app/artifacts/` and update `app/backend/inference.py`.

Required artifacts:

- `resnet50_ft_v2_best.pth`
- ordered class names
- calibration temperature
- decision policy
- hard-class list
- confusion-pair list

The multi-food path also uses detector weights through the `ultralytics` runtime.
Set `FOODLENS_DETECTOR_WEIGHTS` to override the default `yolo11n.pt` detector.
