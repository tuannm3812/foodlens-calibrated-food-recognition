# FoodLens App

FoodLens is the product direction for this project: a polished food-recognition
assistant that turns image or video inputs into calibrated predictions and
user-facing actions.

## Current Prototype

The first frontend prototype is a static web app:

```text
app/frontend/index.html
```

It includes:

- image upload preview;
- video upload concept mode;
- mocked top-k predictions;
- calibrated-confidence style display;
- the four decision bands: auto-accept, suggest, confirm, and review.

Open `app/frontend/index.html` in a browser to review the concept.

The frontend calls the local FoodLens API when it is running and falls back to
mock predictions when it is not.

## Backend Prototype

The backend is a small FastAPI service:

```text
app/backend/api.py
```

Run it locally:

```bash
pip install fastapi uvicorn python-multipart
uvicorn app.backend.api:app --reload --port 8000
```

Endpoints:

```text
GET /health
POST /predict/image
POST /predict/video
```

The backend uses the project ResNet50 FT-V2 artifacts when they are present. If
artifacts or runtime dependencies are missing, it falls back to deterministic
mock predictions so the frontend still works.

## Artifact Requirements

Real inference needs:

- `resnet50_ft_v2_best.pth`;
- `class_names.json`;
- `calibration.json`;
- `decision_policy.json`;
- `hard_classes.json`;
- `confusion_pairs.json`.

Artifacts should stay out of git and be placed under `app/artifacts/` or a
local model path when the backend is added.

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

The next implementation step is exporting the Notebook 6 metadata artifacts
into the JSON files above and placing the checkpoint under `app/artifacts/`.
