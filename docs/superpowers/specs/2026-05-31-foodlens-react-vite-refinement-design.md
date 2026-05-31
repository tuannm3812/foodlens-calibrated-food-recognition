# FoodLens React/Vite Refinement Design

Date: 2026-05-31

## 1. Goal

Refine the FoodLens application into a portfolio-quality prototype while
keeping the implementation grounded in the existing project. The pass will
retain the FastAPI backend, replace the main static frontend with a React/Vite
frontend, archive the current static implementation for reference, and add
focused tests and documentation.

The outcome should show a complete machine-learning product workflow:
food-image upload, multi-food crop review, calibrated decision bands, fallback
transparency, and clear local run instructions.

## 2. Chosen Approach

Use a vertical-slice rebuild.

1. Move the current static frontend from `app/frontend` to
   `app/frontend-static`.
2. Create the React/Vite frontend in `app/frontend`.
3. Define a small typed API client for the FoodLens backend.
4. Build one complete image-analysis flow first:
   upload or sample image, call `POST /predict/multi-food/image`, render the
   preview overlay, strongest-crop summary, top-k ranking, model status, and
   crop review grid.
5. Add focused backend and frontend tests around that slice.
6. Port the existing client-side video-frame sampling flow after the image
   flow is stable.
7. Update app documentation and run instructions.

This avoids a broad rewrite while proving the new architecture through one
complete user-facing path.

## 3. Frontend Design

The React app will use an Analyzer Workbench layout. The first screen remains
the usable analysis tool, not a landing page.

Primary layout:

- Left column: image or video preview, detection overlay, upload controls,
  sample control, clear control, and image/video mode switch.
- Right column: decision badge, top prediction, confidence display, top-k
  ranking, recommended action, model name, temperature, artifact status, and
  detector status.
- Lower section: multi-food crop review cards sorted by crop confidence.

Visual tone:

- Culinary Studio: warm and food-forward, but restrained enough to remain a
  serious analysis tool.
- Use stable dimensions for preview, overlays, buttons, rankings, and crop
  cards so loading and dynamic result states do not shift the layout.
- Avoid marketing-page composition. The analyzer is the primary experience.

Suggested frontend modules:

- `src/api/foodlensClient.ts`: API calls and response normalization.
- `src/api/types.ts`: backend response types used by the UI.
- `src/components/AnalyzerWorkbench.tsx`: top-level analyzer composition.
- `src/components/PreviewStage.tsx`: media preview and detection overlay.
- `src/components/UploadControls.tsx`: mode, upload, sample, and clear
  controls.
- `src/components/DecisionSummary.tsx`: decision band, top label, confidence,
  action copy, and model/runtime metadata.
- `src/components/PredictionRanking.tsx`: top-k prediction list.
- `src/components/CropReviewGrid.tsx`: detected crop cards.
- `src/state/useAnalyzer.ts`: local analyzer state and image/video workflows.
- `src/styles/`: global CSS and design tokens.

React state should track:

- selected file and object URL;
- active mode: image or video;
- loading/analyzing status;
- latest normalized result;
- backend error or fallback reason;
- whether the displayed result is live, backend fallback, or local demo data.

## 4. Backend Design

The backend remains FastAPI. The refinement should improve the interface React
depends on without turning the backend into a new service.

Backend focus areas:

- Make response metadata stable and explicit:
  `artifact_status`, `detector_status`, and a frontend-readable fallback reason
  where useful.
- Keep the multi-food endpoint aligned with the Notebook 8 app contract.
- Keep real inference optional. Missing artifacts, missing detector runtime, or
  real inference errors should return deterministic fallback data for the demo
  path unless the request is invalid.
- Improve locality of decision and response-building code only where it helps
  tests. A split from `inference.py` is acceptable for decision logic,
  artifact/runtime status, or response builders; unrelated restructuring is out
  of scope.

The backend should continue exposing:

- `GET /health`
- `POST /predict/image`
- `POST /predict/multi-food/image`
- `POST /predict/video`

The React vertical slice will primarily consume
`POST /predict/multi-food/image`.

## 5. Data Flow

Image flow:

1. User selects an image or loads the sample image.
2. React renders the preview immediately and sets the analyzer status to
   loading.
3. The API client posts the image to `/predict/multi-food/image`.
4. The client normalizes the backend response into a UI result.
5. The UI renders detection boxes, a strongest-crop summary, top-k ranking,
   model/runtime status, and crop cards.
6. If the backend is offline or returns an invalid response, React renders the
   local demo fallback and labels it clearly as local demo data.

Video flow:

1. User selects a short video.
2. React samples up to three frames in the browser.
3. Each frame is sent to `/predict/multi-food/image`.
4. React combines region predictions across frames and labels the source frame
   for each crop.
5. The same Analyzer Workbench components render the combined result.

## 6. Error Handling And Fallbacks

User-facing states should be explicit:

- Idle: no input selected.
- Analyzing: request or frame sampling in progress.
- Live result: backend returned live inference or live detector status.
- Backend fallback: backend responded with deterministic fallback data.
- Local demo fallback: backend unavailable or response invalid, so the frontend
  used bundled demo data.
- Error: invalid file, unsupported media, or unrecoverable UI workflow failure.

Fallbacks should not be hidden. The UI should show whether a result came from
live inference, backend fallback, or local demo data.

## 7. Testing

Backend tests:

- Decision band behavior for auto-accept, suggest, confirm, and review.
- Artifact-missing fallback for single-image and multi-food image paths.
- Multi-food response shape, including crop count, prediction fields, detector
  metadata, and artifact metadata.

Frontend tests:

- API response normalization for backend multi-food responses.
- Local demo fallback normalization.
- Analyzer rendering states for idle, analyzing, result, and fallback.

Manual verification:

- Start FastAPI and Vite locally.
- Load the React app in a browser.
- Verify sample image, uploaded image, fallback labeling, overlay rendering, and
  crop review layout.
- Verify video frame sampling after the image slice is complete.

## 8. Documentation

Update documentation after implementation:

- `app/README.md`: new FastAPI + React/Vite run commands, static archive note,
  artifact requirements, fallback behavior, and test commands.
- `app/backend/README.md`: clarify backend endpoints, fallback behavior, and
  optional detector/runtime dependencies.
- `README.md` if needed: update the app section to point at the React frontend
  and describe the current FoodLens prototype accurately.

## 9. Out Of Scope

- Replacing FastAPI.
- Adding accounts, persistence, cloud deployment, or production monitoring.
- Training a new model or changing the champion model.
- Replacing the detector with a food-specific detector.
- Adding nutrition estimation or ingredient detection.
- Building a marketing landing page.

## 10. Approval Notes

Approved choices:

- Balanced portfolio-quality refinement.
- Keep FastAPI.
- Upgrade the main frontend to React/Vite.
- Archive the current static frontend and make `app/frontend` the React app.
- Use an Analyzer Workbench layout.
- Use a restrained Culinary Studio visual tone.
- Use a vertical-slice rebuild.
- Include focused backend and frontend tests in the first implementation plan.
