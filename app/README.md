# FoodLens App

FoodLens is the product direction for this project: a polished food-recognition
assistant that turns image or video inputs into calibrated crop-level
predictions and user-facing actions.

## Current Prototype

The main frontend is the React/Vite Analyzer Workbench:

```text
app/frontend
```

It includes:

- image upload preview;
- video key-frame sampling;
- single-image and multi-region result views;
- calibrated-confidence style display;
- the four decision bands: auto-accept, suggest, confirm, and review.

Run it locally:

```bash
cd app/frontend
npm install
npm run dev
```

The old static prototype is archived under `app/frontend-static`.

The frontend calls the local FoodLens API when it is running and falls back to
deterministic demo predictions when it is not. Backend fallback responses expose
`fallback_reason` so the UI can explain why demo data is being shown.

Frontend checks:

```bash
cd app/frontend
npm test
npm run typecheck
npm run build
```

## Backend Prototype

The backend is a small FastAPI service:

```text
app/backend/api.py
```

Run it locally:

```bash
pip install -r app/backend/requirements.txt
pip install -r app/backend/requirements-dev.txt
uvicorn app.backend.api:app --reload --port 8000
```

For live multi-food detection, install the optional detector runtime as well:

```bash
pip install -r app/backend/requirements-detector.txt
```

Backend tests:

```bash
python3 -m pytest tests/backend -v
```

Endpoints:

```text
GET  /health
GET  /runtime/status
POST /predict/image
POST /predict/multi-food/image
POST /predict/multi-food/image-url
POST /predict/multi-food/youtube-url
POST /predict/video
```

`/runtime/status` reports classifier artifact readiness, optional calibration
and decision-policy files, detector dependency availability, detector weight
resolution, and the effective multi-food mode.

The backend uses the project ResNet50 FT-V2 artifacts for image classification
when they are present. If artifacts or runtime dependencies are missing, it
falls back to deterministic mock predictions so the frontend still works. The
direct image URL and YouTube URL endpoints validate and ingest remote media
before sending images or sampled frames through the same multi-food contract.
The legacy video upload endpoint is explicitly marked as
`fallback_reason: video_mock` until live backend video inference is implemented.

The multi-food endpoint follows the Notebook 8 response contract so the app can
render detected regions, crop-level FoodLens predictions, decision bands, and
artifact references. When `ultralytics` and the model artifacts are available,
it runs YOLO proposals followed by the FoodLens crop classifier and returns
`detector_status: live_yolo`. Otherwise, it falls back to a deterministic
prototype response with `detector_status: fallback_demo`.

For video uploads, the frontend samples up to three key frames and sends each
frame through the same multi-food image endpoint. Video summaries always ask for
confirmation because sampled frames are less reliable than direct image
analysis.

## Artifact Requirements

Real inference needs:

- `resnet50_ft_v2_best.pth`;
- `class_names.json`;
- `calibration.json`;
- `decision_policy.json`;
- `hard_classes.json`;
- `confusion_pairs.json`.

Artifacts should stay out of git and be placed under `app/artifacts/` or a
local model path when running live inference.

Source:

- Download `resnet50_ft_v2_best.pth` from the ResNet50 FT-V2 Kaggle model
  artifact.
- Download the JSON files from Notebook 6 output under
  `results/food_recognition_demo/`.
- For convenience, Notebook 6 also creates
  `foodlens_app_artifacts.zip` with the JSON files and demo CSVs.

Expected JSON shapes:

```json
{"temperature": 0.958111}
```

```json
{"auto_confidence": 0.7, "suggest_confidence": 0.35, "margin_threshold": 0.4}
```

```json
["chocolate_mousse", "steak", "pork_chop"]
```

```json
[["steak", "filet_mignon"], ["tuna_tartare", "beef_tartare"]]
```

## Next Build Step

The next implementation step is improving the detector quality with a
food-specific detection or segmentation model, because generic COCO detectors
still miss many plated foods and often localize containers instead of dishes.
