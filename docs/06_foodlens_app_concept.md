# 6. FoodLens App Concept

## 1. Product Vision

**FoodLens** is a food-recognition assistant that identifies dishes from images
or short videos and turns model predictions into practical user actions.

The app should not behave like a raw classifier that always returns one answer.
It should use the project decision layer to decide whether to:

- **auto-accept** a confident, low-risk prediction;
- **suggest** ranked food labels for ambiguous cases;
- **confirm** uncertain or hard-class predictions with the user;
- **review** known confusion patterns that need extra care.

This product direction matches the strongest evidence from the modeling work:
the model has useful top-1 performance, very strong top-5 performance, improved
calibration, and clear hard-class risk patterns.

## 2. Target Users

FoodLens can support several realistic user groups:

| User group | Need |
| --- | --- |
| Food diary users | quickly identify meals and correct labels when needed |
| Recipe or menu platforms | enrich uploaded food photos with searchable labels |
| Restaurants or creators | tag food images for catalogs, menus, or social content |
| Data-labeling teams | speed up image labeling while preserving human review |

The strongest first use case is **assisted food tagging**, where the user can
confirm or correct the model output.

## 3. MVP Scope

The first version should focus on **image recognition**, not video. Image mode
is enough to prove the product value and reuse the current Notebook 6 inference
logic cleanly.

MVP features:

1. Upload or capture one food image.
2. Run the ResNet50 FT-V2 inference pipeline.
3. Return top-k predictions with calibrated confidence.
4. Apply the decision policy.
5. Show the recommended action to the user.
6. Let the user confirm or correct the prediction.
7. Store lightweight prediction history for review.

Out of scope for MVP:

- nutrition estimation;
- ingredient detection;
- multi-food plate segmentation;
- real-time video;
- user account system;
- production monitoring.

## 4. User Experience

### 4.1 Easy Prediction

Example:

```text
Prediction: miso_soup
Confidence: 99.17%
Decision: Auto-accept
Action: Accept the top prediction automatically.
```

The app should show the accepted label clearly, while still allowing the user to
open the ranked alternatives if they want to correct it.

### 4.2 Ambiguous Prediction

Example:

```text
Prediction: ice_cream
Confidence: 54.76%
Decision: Suggest
Action: Show ranked suggestions for user selection.

Alternatives:
1. ice_cream
2. frozen_yogurt
3. churros
4. cannoli
5. dumplings
```

The app should present the top choices as selectable options instead of forcing
one final answer.

### 4.3 Hard-Class Prediction

Example:

```text
Prediction: grilled_salmon
Confidence: 62.51%
Decision: Confirm
Action: Ask the user to confirm because this is a hard predicted class.
```

The app should ask for confirmation and make correction easy.

### 4.4 Known Confusion

Example:

```text
Prediction: filet_mignon
Actual demo class: steak
Decision: Review
Action: Flag for review because this matches a known confusion pair.
```

In a real user upload, the true class is usually unknown, so review should rely
on predicted-class risk, low margin, historical confusion groups, or user
feedback. In evaluation demos, known labels can trigger review directly.

## 5. System Architecture

A clean first implementation can use a small web app plus an inference API:

```text
foodlens/
  app/
    backend/
      api.py
      inference.py
      schemas.py
    frontend/
      upload UI
      result UI
      feedback UI
    artifacts/
      class_names.json
      decision_policy.json
      calibration.json
```

Recommended stack:

| Layer | Recommendation | Reason |
| --- | --- | --- |
| Backend | FastAPI | simple Python inference service |
| Model runtime | PyTorch | matches current training workflow |
| Frontend | React or Next.js | fast image-upload prototype |
| Storage | local JSON/SQLite first | enough for prediction history and feedback |
| Deployment | local or lightweight cloud VM | avoid premature infrastructure complexity |

## 6. Inference Contract

The app should expose one primary endpoint:

```text
POST /predict/image
```

Example response:

```json
{
  "model_name": "resnet50_ft_v2",
  "temperature": 0.958111,
  "top_predictions": [
    {"rank": 1, "class_name": "miso_soup", "confidence": 0.9917},
    {"rank": 2, "class_name": "ramen", "confidence": 0.0008}
  ],
  "decision": {
    "band": "auto_accept",
    "recommended_action": "Accept the top prediction automatically.",
    "top_1_top_2_margin": 0.9909
  }
}
```

The API should also return enough metadata for auditability:

- model version;
- calibration temperature;
- decision-policy version;
- preprocessing settings;
- inference timestamp;
- optional user feedback.

## 7. Video Direction

Video should be a second-phase feature after image mode is stable.

Initial video method:

1. Accept a short food video.
2. Sample frames at a fixed interval.
3. Run the image classifier on each sampled frame.
4. Aggregate predictions with confidence-weighted voting.
5. Return a video-level top-k prediction and decision band.

Example video output:

```text
Top dish: sushi
Decision: Suggest
Frame agreement:
- sushi: 8 frames
- sashimi: 2 frames
- ramen: 1 frame
```

Video risks:

- motion blur;
- non-food frames;
- multiple dishes in one scene;
- changing camera angle and lighting;
- higher inference cost.

For that reason, video should not be the MVP.

## 8. Model And Product Limitations

FoodLens should communicate uncertainty honestly.

Current limitations:

- Food-101 only covers **101 classes**, so out-of-scope dishes may be forced
  into the closest known class.
- Similar dishes remain hard: steak-like dishes, tartare-style dishes, and
  dessert families.
- The model classifies the dominant dish but does not detect multiple foods in
  one image.
- Confidence is calibrated for the Food-101 evaluation setup, not guaranteed
  for every real-world camera image.

These limitations support the decision-layer design: users should see
suggestions, confirmations, and review states when the model is uncertain.

## 9. Build Roadmap

### Phase 1: Inference Package

- Move Notebook 6 inference logic into reusable Python functions.
- Export `class_names.json`, `calibration.json`, and `decision_policy.json`.
- Add a small CLI or local script for single-image prediction.

### Phase 2: Backend API

- Build a FastAPI `/predict/image` endpoint.
- Add request and response schemas.
- Add basic validation for file type and image size.
- Return top-k predictions, decision band, and recommended action.

### Phase 3: Web MVP

- Build image upload UI.
- Show top-k predictions and decision action.
- Add user confirmation/correction controls.
- Save prediction history locally.

### Phase 4: Feedback Loop

- Store corrected labels.
- Export feedback examples for future evaluation.
- Track which classes users correct most often.

### Phase 5: Video Prototype

- Add short-video upload.
- Sample frames.
- Aggregate frame predictions.
- Compare video-level output against image-only output.

## 10. Success Criteria

FoodLens should be judged by product behavior, not only model accuracy.

MVP success metrics:

- image upload works reliably;
- top-k predictions match Notebook 6 behavior;
- calibrated confidence is shown consistently;
- decision bands are applied correctly;
- users can confirm or correct a prediction;
- prediction history can be reviewed;
- the app clearly handles uncertainty instead of pretending every prediction is
  equally reliable.

## 11. Recommended Next Step

The first app foundation has started with a static frontend concept and
artifact-aware FastAPI backend:

```text
app/frontend/index.html
app/backend/api.py
```

It is intentionally not Streamlit. The prototype focuses on a polished product
interface for upload, preview, top-k predictions, calibrated confidence, and the
four FoodLens decision bands. The backend uses the project ResNet50 FT-V2
artifacts when they are available and falls back to mock predictions only when
artifacts or runtime dependencies are missing.

The next technical step is to export the project artifacts into app-ready files:

1. Place model artifacts under `app/artifacts/` outside git.
2. Export `class_names.json`, `calibration.json`, `decision_policy.json`,
   `hard_classes.json`, and `confusion_pairs.json`.
3. Keep Notebook 6 as the experiment/demo record.
4. Improve the React frontend using the Stitch and AI Studio design exports.
5. Build the image-upload MVP before starting video.

This keeps FoodLens grounded in the current model evidence while opening a
clear path from research notebook to usable application.
