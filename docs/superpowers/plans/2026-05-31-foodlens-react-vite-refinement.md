# FoodLens React/Vite Refinement Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a portfolio-quality FoodLens prototype by archiving the static frontend, creating a React/Vite Analyzer Workbench, tightening the FastAPI response contract, and adding focused backend/frontend tests.

**Architecture:** Keep FastAPI as the inference API and make `app/frontend` the React/Vite app. Archive the current static app in `app/frontend-static`. Build one complete image-analysis slice first, then add video frame sampling and documentation.

**Tech Stack:** FastAPI, Pydantic, pytest, React, TypeScript, Vite, Vitest, Testing Library, CSS.

---

## File Structure

Backend:

- Modify `app/backend/schemas.py`: add explicit fallback metadata to response models.
- Create `app/backend/decision.py`: isolate decision-band logic behind a small function.
- Modify `app/backend/inference.py`: use `decision.py`, set fallback metadata, preserve existing runtime behavior.
- Create `app/backend/requirements-dev.txt`: backend test dependencies.
- Create `tests/backend/test_decision.py`: decision-band tests.
- Create `tests/backend/test_api_contract.py`: fallback and response-shape tests.

Frontend archive and React app:

- Move existing `app/frontend/index.html`, `app/frontend/styles.css`, and `app/frontend/app.js` to `app/frontend-static/`.
- Create `app/frontend/package.json`, `tsconfig.json`, `tsconfig.node.json`, `vite.config.ts`, `index.html`.
- Create `app/frontend/src/main.tsx`, `src/App.tsx`, `src/styles.css`.
- Create `app/frontend/src/api/types.ts`: typed API response and normalized UI result types.
- Create `app/frontend/src/api/demoData.ts`: local demo fallback data.
- Create `app/frontend/src/api/foodlensClient.ts`: API calls, validation, normalization, and fallback helpers.
- Create `app/frontend/src/state/useAnalyzer.ts`: analyzer workflow state.
- Create `app/frontend/src/components/AnalyzerWorkbench.tsx`.
- Create `app/frontend/src/components/PreviewStage.tsx`.
- Create `app/frontend/src/components/UploadControls.tsx`.
- Create `app/frontend/src/components/DecisionSummary.tsx`.
- Create `app/frontend/src/components/PredictionRanking.tsx`.
- Create `app/frontend/src/components/CropReviewGrid.tsx`.
- Create `app/frontend/src/components/StatusNotice.tsx`.
- Create `app/frontend/src/test/setup.ts`.
- Create `app/frontend/src/api/foodlensClient.test.ts`.
- Create `app/frontend/src/components/AnalyzerWorkbench.test.tsx`.

Docs:

- Modify `app/README.md`: React/Vite run commands, archived static app note, fallback states, tests.
- Modify `app/backend/README.md`: backend test command and fallback metadata.
- Modify `README.md`: point the app section at the React frontend.

## Task 1: Backend Decision Module And Tests

**Files:**

- Create: `app/backend/decision.py`
- Modify: `app/backend/inference.py`
- Create: `app/backend/requirements-dev.txt`
- Create: `tests/backend/test_decision.py`

- [ ] **Step 1: Add backend dev requirements**

Create `app/backend/requirements-dev.txt`:

```text
-r requirements.txt
pytest
httpx
```

- [ ] **Step 2: Write failing decision tests**

Create `tests/backend/test_decision.py`:

```python
from app.backend.decision import build_decision
from app.backend.schemas import Prediction


def predictions(top_label: str, top_score: float, second_score: float) -> list[Prediction]:
    return [
        Prediction(rank=1, class_name=top_label, confidence=top_score),
        Prediction(rank=2, class_name="second_choice", confidence=second_score),
    ]


def test_auto_accept_when_confident_not_hard_and_margin_is_large() -> None:
    decision = build_decision(
        mode="image",
        predictions=predictions("miso_soup", 0.91, 0.12),
        policy={
            "auto_confidence": 0.70,
            "suggest_confidence": 0.35,
            "margin_threshold": 0.40,
        },
        hard_classes={"steak"},
        confusion_pairs=set(),
    )

    assert decision.band == "auto_accept"
    assert decision.title == "Auto-accept"
    assert decision.top_1_top_2_margin == 0.79


def test_suggest_when_confident_but_margin_is_small() -> None:
    decision = build_decision(
        mode="image",
        predictions=predictions("ramen", 0.68, 0.42),
        policy={
            "auto_confidence": 0.70,
            "suggest_confidence": 0.35,
            "margin_threshold": 0.40,
        },
        hard_classes={"steak"},
        confusion_pairs=set(),
    )

    assert decision.band == "suggest"
    assert decision.title == "Show suggestions"


def test_confirm_when_prediction_is_hard_class_below_auto_threshold() -> None:
    decision = build_decision(
        mode="image",
        predictions=predictions("steak", 0.64, 0.12),
        policy={
            "auto_confidence": 0.70,
            "suggest_confidence": 0.35,
            "margin_threshold": 0.40,
        },
        hard_classes={"steak"},
        confusion_pairs=set(),
    )

    assert decision.band == "confirm"
    assert decision.title == "Confirm dish"


def test_review_when_prediction_has_confusion_risk_and_small_margin() -> None:
    decision = build_decision(
        mode="image",
        predictions=predictions("filet_mignon", 0.57, 0.31),
        policy={
            "auto_confidence": 0.70,
            "suggest_confidence": 0.35,
            "margin_threshold": 0.40,
        },
        hard_classes=set(),
        confusion_pairs={("steak", "filet_mignon")},
    )

    assert decision.band == "review"
    assert decision.title == "Review prediction"


def test_video_predictions_always_require_confirmation() -> None:
    decision = build_decision(
        mode="video",
        predictions=predictions("sushi", 0.88, 0.05),
        policy={
            "auto_confidence": 0.70,
            "suggest_confidence": 0.35,
            "margin_threshold": 0.40,
        },
        hard_classes=set(),
        confusion_pairs=set(),
    )

    assert decision.band == "confirm"
    assert decision.title == "Confirm dish"
```

- [ ] **Step 3: Run test to verify it fails**

Run:

```bash
python -m pytest tests/backend/test_decision.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'app.backend.decision'`.

- [ ] **Step 4: Create the decision module**

Create `app/backend/decision.py`:

```python
"""Decision-band helpers for FoodLens predictions."""

from .schemas import Decision, Prediction


DEFAULT_HARD_CLASSES = {
    "chocolate_mousse",
    "steak",
    "pork_chop",
    "bread_pudding",
    "tuna_tartare",
}

DEFAULT_POLICY = {
    "auto_confidence": 0.70,
    "suggest_confidence": 0.35,
    "margin_threshold": 0.40,
}


def build_decision(
    mode: str,
    predictions: list[Prediction],
    policy: dict[str, float] | None = None,
    hard_classes: set[str] | None = None,
    confusion_pairs: set[tuple[str, str]] | None = None,
) -> Decision:
    """Build a FoodLens decision output from ranked predictions."""
    active_policy = policy or DEFAULT_POLICY
    active_hard_classes = hard_classes or DEFAULT_HARD_CLASSES
    active_confusion_pairs = confusion_pairs or set()
    top_1 = predictions[0]
    top_2 = predictions[1]
    margin = round(top_1.confidence - top_2.confidence, 10)
    predicted_label = top_1.class_name
    risky_prediction = any(predicted_label in pair for pair in active_confusion_pairs)

    if mode == "video":
        return Decision(
            band="confirm",
            title="Confirm dish",
            recommended_action=(
                "Ask the user to confirm because sampled frames are not fully aligned."
            ),
            top_1_top_2_margin=margin,
        )

    if risky_prediction and margin < active_policy["margin_threshold"]:
        return Decision(
            band="review",
            title="Review prediction",
            recommended_action="Flag for review because this matches a known confusion risk.",
            top_1_top_2_margin=margin,
        )

    if (
        predicted_label in active_hard_classes
        and top_1.confidence < active_policy["auto_confidence"]
    ):
        return Decision(
            band="confirm",
            title="Confirm dish",
            recommended_action="Ask the user to confirm because this is a hard predicted class.",
            top_1_top_2_margin=margin,
        )

    if (
        top_1.confidence >= active_policy["auto_confidence"]
        and margin >= active_policy["margin_threshold"]
        and predicted_label not in active_hard_classes
    ):
        return Decision(
            band="auto_accept",
            title="Auto-accept",
            recommended_action="Accept the top prediction automatically.",
            top_1_top_2_margin=margin,
        )

    if top_1.confidence >= active_policy["suggest_confidence"]:
        return Decision(
            band="suggest",
            title="Show suggestions",
            recommended_action="Show ranked suggestions for user selection.",
            top_1_top_2_margin=margin,
        )

    return Decision(
        band="confirm",
        title="Confirm dish",
        recommended_action="Ask the user to confirm before applying a label.",
        top_1_top_2_margin=margin,
    )
```

- [ ] **Step 5: Modify `app/backend/inference.py` imports and constants**

Remove `Decision` from the `.schemas` import list. Add this import near the
local imports:

```python
from .decision import DEFAULT_HARD_CLASSES, DEFAULT_POLICY, build_decision
```

Delete the `DEFAULT_POLICY` and `DEFAULT_HARD_CLASSES` definitions from
`app/backend/inference.py`. Delete the existing `build_decision` function from
`app/backend/inference.py`.

- [ ] **Step 6: Run decision tests**

Run:

```bash
python -m pytest tests/backend/test_decision.py -v
```

Expected: 5 passed.

- [ ] **Step 7: Commit**

```bash
git add app/backend/decision.py app/backend/inference.py app/backend/requirements-dev.txt tests/backend/test_decision.py
git commit -m "test: cover FoodLens decision bands"
```

## Task 2: Backend Contract Metadata And API Tests

**Files:**

- Modify: `app/backend/schemas.py`
- Modify: `app/backend/inference.py`
- Create: `tests/backend/test_api_contract.py`

- [ ] **Step 1: Write failing API contract tests**

Create `tests/backend/test_api_contract.py`:

```python
from fastapi.testclient import TestClient

from app.backend.api import app


client = TestClient(app)


def test_health_contract() -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_single_image_missing_artifacts_returns_mock_with_fallback_reason() -> None:
    response = client.post(
        "/predict/image",
        files={"file": ("sample.jpg", b"not-a-real-image", "image/jpeg")},
    )

    body = response.json()
    assert response.status_code == 200
    assert body["artifact_status"] == "mock"
    assert body["fallback_reason"] == "missing_artifacts"
    assert body["top_predictions"][0]["class_name"] == "steak"
    assert body["decision"]["band"] == "suggest"


def test_multi_food_missing_artifacts_returns_demo_contract() -> None:
    response = client.post(
        "/predict/multi-food/image",
        files={"file": ("sample.jpg", b"not-a-real-image", "image/jpeg")},
    )

    body = response.json()
    first_prediction = body["predictions"][0]
    assert response.status_code == 200
    assert body["artifact_status"] == "mock"
    assert body["detector_status"] == "fallback_demo"
    assert body["fallback_reason"] == "missing_artifacts"
    assert body["crop_count"] == len(body["predictions"])
    assert first_prediction["bbox"]["source_width"] > 0
    assert first_prediction["detector"]["proposal_role"] in {
        "serving_container",
        "direct_food",
        "fallback_region",
        "context_object",
    }
    assert first_prediction["foodlens"]["top_k_predictions"][0][0]
    assert "crop_path" in first_prediction["artifacts"]
```

- [ ] **Step 2: Run API tests to verify they fail**

Run:

```bash
python -m pytest tests/backend/test_api_contract.py -v
```

Expected: FAIL because `fallback_reason` is absent.

- [ ] **Step 3: Add fallback metadata to schemas**

In `app/backend/schemas.py`, add `fallback_reason` to both response models:

```python
class PredictionResponse(BaseModel):
    """FoodLens prediction response."""

    model_name: str
    mode: str
    temperature: float
    top_predictions: list[Prediction]
    decision: Decision
    artifact_status: str
    fallback_reason: Optional[str] = None
```

```python
class MultiFoodPredictionResponse(BaseModel):
    """App-ready multi-food prediction response."""

    model: str
    temperature: float
    top_k: int
    decision_thresholds: dict[str, float]
    detector_status: str
    crop_count: int
    predictions: list[MultiFoodPrediction]
    artifact_status: str
    fallback_reason: Optional[str] = None
```

- [ ] **Step 4: Set fallback metadata in inference responses**

In `app/backend/inference.py`, change `predict_mock` to accept a fallback
reason:

```python
def predict_mock(
    mode: str = "image",
    fallback_reason: str = "missing_artifacts",
) -> PredictionResponse:
    """Return a deterministic mock prediction response."""
    raw_predictions = MOCK_VIDEO_PREDICTIONS if mode == "video" else MOCK_IMAGE_PREDICTIONS
    predictions = build_predictions(raw_predictions)
    return PredictionResponse(
        model_name=MODEL_NAME,
        mode=mode,
        temperature=TEMPERATURE,
        top_predictions=predictions,
        decision=build_decision(mode, predictions),
        artifact_status=artifact_status(),
        fallback_reason=fallback_reason,
    )
```

In `build_multi_food_mock`, add a parameter and response field:

```python
def build_multi_food_mock(
    fallback_reason: str = "missing_artifacts",
) -> MultiFoodPredictionResponse:
    """Return a deterministic Notebook 8-style multi-food response."""
```

Set the response field:

```python
        artifact_status=artifact_status(),
        fallback_reason=fallback_reason,
```

In `build_multi_food_response`, add `fallback_reason=None`.

In `build_prediction_response`, add `fallback_reason=None`.

In `predict_multi_food_image_bytes`, distinguish fallback reasons:

```python
    if artifact_status() != "ready":
        return build_multi_food_mock(fallback_reason="missing_artifacts")

    try:
        runtime = load_runtime()
        image = runtime["image_class"].open(BytesIO(image_bytes)).convert("RGB")
        detections = detect_candidate_regions(image)
        if not detections:
            detections = [build_full_image_region(image)]
        return build_multi_food_response(image, detections, runtime)
    except RuntimeError:
        return build_multi_food_mock(fallback_reason="detector_runtime_unavailable")
    except Exception:
        return build_multi_food_mock(fallback_reason="inference_error")
```

In `predict_image_bytes`, distinguish missing artifacts from inference errors:

```python
    if artifact_status() != "ready":
        return predict_mock(mode="image", fallback_reason="missing_artifacts")

    try:
        runtime = load_runtime()
        image = runtime["image_class"].open(BytesIO(image_bytes)).convert("RGB")
        return build_prediction_response(image, runtime)
    except Exception:
        return predict_mock(mode="image", fallback_reason="inference_error")
```

- [ ] **Step 5: Run backend tests**

Run:

```bash
python -m pytest tests/backend -v
```

Expected: all backend tests pass.

- [ ] **Step 6: Commit**

```bash
git add app/backend/schemas.py app/backend/inference.py tests/backend/test_api_contract.py
git commit -m "feat: expose FoodLens fallback metadata"
```

## Task 3: Archive Static Frontend And Scaffold React/Vite

**Files:**

- Move: `app/frontend/index.html` to `app/frontend-static/index.html`
- Move: `app/frontend/styles.css` to `app/frontend-static/styles.css`
- Move: `app/frontend/app.js` to `app/frontend-static/app.js`
- Create: `app/frontend/package.json`
- Create: `app/frontend/tsconfig.json`
- Create: `app/frontend/tsconfig.node.json`
- Create: `app/frontend/vite.config.ts`
- Create: `app/frontend/index.html`
- Create: `app/frontend/src/main.tsx`
- Create: `app/frontend/src/App.tsx`
- Create: `app/frontend/src/styles.css`
- Create: `app/frontend/src/test/setup.ts`

- [ ] **Step 1: Move static frontend files**

Run:

```bash
mkdir -p app/frontend-static
mv app/frontend/index.html app/frontend-static/index.html
mv app/frontend/styles.css app/frontend-static/styles.css
mv app/frontend/app.js app/frontend-static/app.js
mkdir -p app/frontend/src/test
```

Expected: `app/frontend` contains no old static JS/CSS files, and
`app/frontend-static` contains the archived prototype.

- [ ] **Step 2: Create React package metadata**

Create `app/frontend/package.json`:

```json
{
  "name": "foodlens-frontend",
  "version": "0.1.0",
  "private": true,
  "type": "module",
  "scripts": {
    "dev": "vite --host 127.0.0.1 --port 5173",
    "build": "tsc && vite build",
    "preview": "vite preview --host 127.0.0.1 --port 4173",
    "test": "vitest run",
    "test:watch": "vitest"
  },
  "dependencies": {
    "@vitejs/plugin-react": "^5.0.0",
    "vite": "^7.0.0",
    "typescript": "^5.8.0",
    "react": "^19.1.0",
    "react-dom": "^19.1.0",
    "lucide-react": "^0.468.0"
  },
  "devDependencies": {
    "@testing-library/jest-dom": "^6.6.0",
    "@testing-library/react": "^16.1.0",
    "@testing-library/user-event": "^14.5.0",
    "@types/react": "^19.1.0",
    "@types/react-dom": "^19.1.0",
    "jsdom": "^25.0.0",
    "vitest": "^3.0.0"
  }
}
```

- [ ] **Step 3: Create TypeScript and Vite config**

Create `app/frontend/tsconfig.json`:

```json
{
  "compilerOptions": {
    "target": "ES2020",
    "useDefineForClassFields": true,
    "lib": ["DOM", "DOM.Iterable", "ES2020"],
    "allowJs": false,
    "skipLibCheck": true,
    "esModuleInterop": true,
    "allowSyntheticDefaultImports": true,
    "strict": true,
    "forceConsistentCasingInFileNames": true,
    "module": "ESNext",
    "moduleResolution": "Node",
    "resolveJsonModule": true,
    "isolatedModules": true,
    "noEmit": true,
    "jsx": "react-jsx"
  },
  "include": ["src"],
  "references": [{ "path": "./tsconfig.node.json" }]
}
```

Create `app/frontend/tsconfig.node.json`:

```json
{
  "compilerOptions": {
    "composite": true,
    "module": "ESNext",
    "moduleResolution": "Node",
    "allowSyntheticDefaultImports": true
  },
  "include": ["vite.config.ts"]
}
```

Create `app/frontend/vite.config.ts`:

```typescript
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  test: {
    environment: "jsdom",
    setupFiles: "./src/test/setup.ts",
    globals: true,
  },
});
```

- [ ] **Step 4: Create React entry files**

Create `app/frontend/index.html`:

```html
<!doctype html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>FoodLens</title>
  </head>
  <body>
    <div id="root"></div>
    <script type="module" src="/src/main.tsx"></script>
  </body>
</html>
```

Create `app/frontend/src/main.tsx`:

```typescript
import { StrictMode } from "react";
import { createRoot } from "react-dom/client";

import App from "./App";
import "./styles.css";

createRoot(document.getElementById("root") as HTMLElement).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
```

Create `app/frontend/src/App.tsx`:

```typescript
export default function App() {
  return (
    <main className="app-shell">
      <section className="scaffold-panel">
        <p className="eyebrow">FoodLens</p>
        <h1>Analyzer Workbench</h1>
        <p>
          React/Vite scaffold is ready. The analyzer slice is added in the next
          task.
        </p>
      </section>
    </main>
  );
}
```

Create `app/frontend/src/styles.css`:

```css
:root {
  color: #271714;
  background: #fffaf6;
  font-family:
    Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI",
    sans-serif;
}

* {
  box-sizing: border-box;
}

body {
  margin: 0;
  min-height: 100vh;
}

button,
input {
  font: inherit;
}

.app-shell {
  display: grid;
  min-height: 100vh;
  place-items: center;
  padding: 32px;
}

.scaffold-panel {
  width: min(100%, 720px);
  padding: 32px;
  border: 1px solid #ead6cd;
  border-radius: 8px;
  background: #ffffff;
}

.eyebrow {
  margin: 0 0 8px;
  color: #b42318;
  font-size: 0.78rem;
  font-weight: 700;
  letter-spacing: 0.08em;
  text-transform: uppercase;
}

h1 {
  margin: 0;
  font-size: clamp(2rem, 5vw, 4rem);
  letter-spacing: 0;
}
```

Create `app/frontend/src/test/setup.ts`:

```typescript
import "@testing-library/jest-dom/vitest";
```

- [ ] **Step 5: Install frontend dependencies**

Run:

```bash
cd app/frontend
npm install
```

Expected: `node_modules` and `package-lock.json` are created.

- [ ] **Step 6: Run frontend build**

Run:

```bash
cd app/frontend
npm run build
```

Expected: TypeScript compiles and Vite creates `dist/`.

- [ ] **Step 7: Commit**

```bash
git add app/frontend app/frontend-static
git commit -m "feat: scaffold React FoodLens frontend"
```

## Task 4: Frontend API Client And Normalization Tests

**Files:**

- Create: `app/frontend/src/api/types.ts`
- Create: `app/frontend/src/api/demoData.ts`
- Create: `app/frontend/src/api/foodlensClient.ts`
- Create: `app/frontend/src/api/foodlensClient.test.ts`

- [ ] **Step 1: Create API types**

Create `app/frontend/src/api/types.ts`:

```typescript
export type DecisionBand = "auto_accept" | "suggest" | "confirm" | "review";
export type ResultSource = "live" | "backend_fallback" | "local_demo";

export type BoundingBox = {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
  source_width: number;
  source_height: number;
};

export type BackendRegionPrediction = {
  source_id: string;
  detection_index: number;
  bbox?: BoundingBox;
  detector: {
    label: string;
    proposal_role: string;
    confidence: number;
    crop_area_ratio: number;
  };
  foodlens: {
    top_label: string;
    top_confidence: number;
    decision_band: DecisionBand;
    top_k_predictions: Array<[string, number]>;
  };
  artifacts: {
    crop_path: string;
    crop_artifact_path: string;
    figure_path: string;
    crop_data_url?: string | null;
  };
};

export type BackendMultiFoodResponse = {
  model: string;
  temperature: number;
  top_k: number;
  decision_thresholds: Record<string, number>;
  detector_status: string;
  crop_count: number;
  predictions: BackendRegionPrediction[];
  artifact_status: string;
  fallback_reason?: string | null;
};

export type UiRegionPrediction = BackendRegionPrediction & {
  displayIndex: number;
};

export type AnalyzerResult = {
  modelName: string;
  temperature: number;
  detectorStatus: string;
  artifactStatus: string;
  fallbackReason?: string;
  source: ResultSource;
  strongestLabel: string;
  strongestConfidence: number;
  decisionBand: DecisionBand;
  actionCopy: string;
  topPredictions: Array<[string, number]>;
  regions: UiRegionPrediction[];
};
```

- [ ] **Step 2: Create local demo fallback data**

Create `app/frontend/src/api/demoData.ts`:

```typescript
import type { BackendMultiFoodResponse } from "./types";

export const LOCAL_DEMO_RESPONSE: BackendMultiFoodResponse = {
  model: "resnet50_ft_v2",
  temperature: 0.958111,
  top_k: 5,
  decision_thresholds: {
    auto_accept: 0.85,
    suggest: 0.5,
  },
  detector_status: "local_demo",
  crop_count: 3,
  artifact_status: "mock",
  fallback_reason: "frontend_local_demo",
  predictions: [
    {
      source_id: "demo_shared_plate",
      detection_index: 0,
      bbox: {
        x1: 72,
        y1: 54,
        x2: 420,
        y2: 332,
        source_width: 640,
        source_height: 420,
      },
      detector: {
        label: "bowl",
        proposal_role: "serving_container",
        confidence: 0.54,
        crop_area_ratio: 0.36,
      },
      foodlens: {
        top_label: "ravioli",
        top_confidence: 0.972,
        decision_band: "auto_accept",
        top_k_predictions: [
          ["ravioli", 0.972],
          ["gnocchi", 0.018],
          ["lasagna", 0.004],
        ],
      },
      artifacts: {
        crop_path: "local-demo/ravioli.jpg",
        crop_artifact_path: "app://local-demo/ravioli.jpg",
        figure_path: "local-demo/figure.jpg",
      },
    },
    {
      source_id: "demo_shared_plate",
      detection_index: 1,
      bbox: {
        x1: 355,
        y1: 92,
        x2: 596,
        y2: 338,
        source_width: 640,
        source_height: 420,
      },
      detector: {
        label: "cake",
        proposal_role: "direct_food",
        confidence: 0.58,
        crop_area_ratio: 0.22,
      },
      foodlens: {
        top_label: "falafel",
        top_confidence: 0.241,
        decision_band: "confirm",
        top_k_predictions: [
          ["falafel", 0.241],
          ["donuts", 0.195],
          ["garlic_bread", 0.112],
        ],
      },
      artifacts: {
        crop_path: "local-demo/falafel.jpg",
        crop_artifact_path: "app://local-demo/falafel.jpg",
        figure_path: "local-demo/figure.jpg",
      },
    },
    {
      source_id: "demo_shared_plate",
      detection_index: 2,
      bbox: {
        x1: 248,
        y1: 255,
        x2: 526,
        y2: 402,
        source_width: 640,
        source_height: 420,
      },
      detector: {
        label: "bowl",
        proposal_role: "serving_container",
        confidence: 0.44,
        crop_area_ratio: 0.15,
      },
      foodlens: {
        top_label: "ramen",
        top_confidence: 0.768,
        decision_band: "suggest",
        top_k_predictions: [
          ["ramen", 0.768],
          ["pho", 0.034],
          ["miso_soup", 0.023],
        ],
      },
      artifacts: {
        crop_path: "local-demo/ramen.jpg",
        crop_artifact_path: "app://local-demo/ramen.jpg",
        figure_path: "local-demo/figure.jpg",
      },
    },
  ],
};
```

- [ ] **Step 3: Write failing API client tests**

Create `app/frontend/src/api/foodlensClient.test.ts`:

```typescript
import { describe, expect, it } from "vitest";

import { LOCAL_DEMO_RESPONSE } from "./demoData";
import { normalizeMultiFoodResponse, toLocalDemoResult } from "./foodlensClient";

describe("normalizeMultiFoodResponse", () => {
  it("sorts regions by confidence and labels backend fallback", () => {
    const result = normalizeMultiFoodResponse({
      ...LOCAL_DEMO_RESPONSE,
      detector_status: "fallback_demo",
      fallback_reason: "missing_artifacts",
    });

    expect(result.source).toBe("backend_fallback");
    expect(result.fallbackReason).toBe("missing_artifacts");
    expect(result.strongestLabel).toBe("ravioli");
    expect(result.strongestConfidence).toBe(0.972);
    expect(result.decisionBand).toBe("auto_accept");
    expect(result.regions[0].displayIndex).toBe(1);
  });

  it("returns a local demo result when the frontend fallback is used", () => {
    const result = toLocalDemoResult();

    expect(result.source).toBe("local_demo");
    expect(result.detectorStatus).toBe("local_demo");
    expect(result.actionCopy).toContain("local demo");
  });
});
```

- [ ] **Step 4: Run API client tests to verify they fail**

Run:

```bash
cd app/frontend
npm test -- src/api/foodlensClient.test.ts
```

Expected: FAIL because `foodlensClient.ts` does not exist.

- [ ] **Step 5: Create the API client**

Create `app/frontend/src/api/foodlensClient.ts`:

```typescript
import { LOCAL_DEMO_RESPONSE } from "./demoData";
import type {
  AnalyzerResult,
  BackendMultiFoodResponse,
  ResultSource,
  UiRegionPrediction,
} from "./types";

const API_BASE_URL = "http://127.0.0.1:8000";

const DECISION_ACTIONS = {
  auto_accept: "Accept the strongest crop label while keeping alternatives available.",
  suggest: "Show ranked suggestions for the user to select.",
  confirm: "Ask the user to confirm before applying a label.",
  review: "Flag this result for extra review because it matches a known risk.",
} as const;

function resultSource(response: BackendMultiFoodResponse): ResultSource {
  if (response.detector_status === "local_demo") {
    return "local_demo";
  }

  if (response.fallback_reason || response.detector_status.includes("fallback")) {
    return "backend_fallback";
  }

  return "live";
}

export function normalizeMultiFoodResponse(
  response: BackendMultiFoodResponse,
): AnalyzerResult {
  const regions: UiRegionPrediction[] = response.predictions
    .slice()
    .sort((a, b) => b.foodlens.top_confidence - a.foodlens.top_confidence)
    .map((prediction, index) => ({
      ...prediction,
      displayIndex: index + 1,
    }));
  const strongest = regions[0];

  return {
    modelName: `${response.model} · Multi-food`,
    temperature: response.temperature,
    detectorStatus: response.detector_status,
    artifactStatus: response.artifact_status,
    fallbackReason: response.fallback_reason ?? undefined,
    source: resultSource(response),
    strongestLabel: strongest?.foodlens.top_label ?? "no_detection",
    strongestConfidence: strongest?.foodlens.top_confidence ?? 0,
    decisionBand: strongest?.foodlens.decision_band ?? "confirm",
    actionCopy: strongest
      ? DECISION_ACTIONS[strongest.foodlens.decision_band]
      : "No usable crop was returned. Ask the user to try another image.",
    topPredictions: strongest?.foodlens.top_k_predictions ?? [["no_detection", 0]],
    regions,
  };
}

export function toLocalDemoResult(): AnalyzerResult {
  return {
    ...normalizeMultiFoodResponse(LOCAL_DEMO_RESPONSE),
    source: "local_demo",
    detectorStatus: "local_demo",
    fallbackReason: "frontend_local_demo",
    actionCopy:
      "Showing local demo data because the API is unavailable or returned an invalid response.",
  };
}

export async function predictMultiFoodImage(file: File): Promise<AnalyzerResult> {
  const formData = new FormData();
  formData.append("file", file);

  const response = await fetch(`${API_BASE_URL}/predict/multi-food/image`, {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    throw new Error(`FoodLens API returned ${response.status}`);
  }

  const body = (await response.json()) as BackendMultiFoodResponse;
  if (!Array.isArray(body.predictions)) {
    throw new Error("FoodLens API returned an invalid multi-food response.");
  }

  return normalizeMultiFoodResponse(body);
}
```

- [ ] **Step 6: Run frontend tests**

Run:

```bash
cd app/frontend
npm test -- src/api/foodlensClient.test.ts
```

Expected: 2 passed.

- [ ] **Step 7: Commit**

```bash
git add app/frontend/src/api
git commit -m "test: cover FoodLens API normalization"
```

## Task 5: React Analyzer Workbench Image Slice

**Files:**

- Modify: `app/frontend/src/App.tsx`
- Modify: `app/frontend/src/styles.css`
- Create: `app/frontend/src/state/useAnalyzer.ts`
- Create: `app/frontend/src/components/AnalyzerWorkbench.tsx`
- Create: `app/frontend/src/components/PreviewStage.tsx`
- Create: `app/frontend/src/components/UploadControls.tsx`
- Create: `app/frontend/src/components/DecisionSummary.tsx`
- Create: `app/frontend/src/components/PredictionRanking.tsx`
- Create: `app/frontend/src/components/CropReviewGrid.tsx`
- Create: `app/frontend/src/components/StatusNotice.tsx`
- Create: `app/frontend/src/components/AnalyzerWorkbench.test.tsx`

- [ ] **Step 1: Write failing component tests**

Create `app/frontend/src/components/AnalyzerWorkbench.test.tsx`:

```typescript
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it } from "vitest";

import { AnalyzerWorkbench } from "./AnalyzerWorkbench";

describe("AnalyzerWorkbench", () => {
  it("renders the idle analyzer controls", () => {
    render(<AnalyzerWorkbench />);

    expect(screen.getByRole("heading", { name: "Analyzer Workbench" })).toBeInTheDocument();
    expect(screen.getByText("No input selected")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Sample" })).toBeInTheDocument();
  });

  it("renders local demo fallback after selecting sample", async () => {
    const user = userEvent.setup();
    render(<AnalyzerWorkbench />);

    await user.click(screen.getByRole("button", { name: "Sample" }));

    expect(screen.getByText("ravioli")).toBeInTheDocument();
    expect(screen.getByText("Local demo")).toBeInTheDocument();
    expect(screen.getByText("Detected regions")).toBeInTheDocument();
  });
});
```

- [ ] **Step 2: Run component tests to verify they fail**

Run:

```bash
cd app/frontend
npm test -- src/components/AnalyzerWorkbench.test.tsx
```

Expected: FAIL because `AnalyzerWorkbench.tsx` does not exist.

- [ ] **Step 3: Create analyzer state hook**

Create `app/frontend/src/state/useAnalyzer.ts`:

```typescript
import { useEffect, useState } from "react";

import {
  predictMultiFoodImage,
  toLocalDemoResult,
} from "../api/foodlensClient";
import type { AnalyzerResult } from "../api/types";

type AnalyzerStatus = "idle" | "analyzing" | "result" | "error";
type AnalyzerMode = "image" | "video";

export function useAnalyzer() {
  const [mode, setMode] = useState<AnalyzerMode>("image");
  const [status, setStatus] = useState<AnalyzerStatus>("idle");
  const [previewUrl, setPreviewUrl] = useState<string>();
  const [result, setResult] = useState<AnalyzerResult>();
  const [message, setMessage] = useState("No input selected");

  useEffect(() => {
    return () => {
      if (previewUrl) {
        URL.revokeObjectURL(previewUrl);
      }
    };
  }, [previewUrl]);

  function clear() {
    if (previewUrl) {
      URL.revokeObjectURL(previewUrl);
    }
    setPreviewUrl(undefined);
    setResult(undefined);
    setStatus("idle");
    setMessage("No input selected");
  }

  function loadSample() {
    setPreviewUrl(undefined);
    setResult(toLocalDemoResult());
    setStatus("result");
    setMessage("Local demo");
  }

  async function analyzeImage(file: File) {
    if (previewUrl) {
      URL.revokeObjectURL(previewUrl);
    }
    setPreviewUrl(URL.createObjectURL(file));
    setStatus("analyzing");
    setMessage("Analyzing image");

    try {
      const apiResult = await predictMultiFoodImage(file);
      setResult(apiResult);
      setStatus("result");
      setMessage(apiResult.source === "live" ? "Live result" : "Backend fallback");
    } catch {
      setResult(toLocalDemoResult());
      setStatus("result");
      setMessage("Local demo");
    }
  }

  return {
    mode,
    setMode,
    status,
    previewUrl,
    result,
    message,
    clear,
    loadSample,
    analyzeImage,
  };
}
```

- [ ] **Step 4: Create AnalyzerWorkbench composition**

Create `app/frontend/src/components/AnalyzerWorkbench.tsx`:

```typescript
import { useAnalyzer } from "../state/useAnalyzer";
import { CropReviewGrid } from "./CropReviewGrid";
import { DecisionSummary } from "./DecisionSummary";
import { PreviewStage } from "./PreviewStage";
import { StatusNotice } from "./StatusNotice";
import { UploadControls } from "./UploadControls";

export function AnalyzerWorkbench() {
  const analyzer = useAnalyzer();

  return (
    <main className="app-shell">
      <header className="app-header">
        <div>
          <p className="eyebrow">FoodLens</p>
          <h1>Analyzer Workbench</h1>
        </div>
        <StatusNotice message={analyzer.message} source={analyzer.result?.source} />
      </header>

      <section className="workbench" aria-label="FoodLens analysis workspace">
        <div className="left-column">
          <PreviewStage
            previewUrl={analyzer.previewUrl}
            regions={analyzer.result?.regions ?? []}
          />
          <UploadControls
            mode={analyzer.mode}
            onModeChange={analyzer.setMode}
            onImageSelected={analyzer.analyzeImage}
            onSample={analyzer.loadSample}
            onClear={analyzer.clear}
            disabled={analyzer.status === "analyzing"}
          />
        </div>
        <DecisionSummary result={analyzer.result} status={analyzer.status} />
      </section>

      <CropReviewGrid regions={analyzer.result?.regions ?? []} />
    </main>
  );
}
```

- [ ] **Step 5: Create child components**

Create `app/frontend/src/components/StatusNotice.tsx`:

```typescript
import type { ResultSource } from "../api/types";

const SOURCE_LABELS: Record<ResultSource, string> = {
  live: "Live result",
  backend_fallback: "Backend fallback",
  local_demo: "Local demo",
};

export function StatusNotice({
  message,
  source,
}: {
  message: string;
  source?: ResultSource;
}) {
  return (
    <div className="status-notice">
      <span>{source ? SOURCE_LABELS[source] : message}</span>
    </div>
  );
}
```

Create `app/frontend/src/components/PreviewStage.tsx`:

```typescript
import type { UiRegionPrediction } from "../api/types";

export function PreviewStage({
  previewUrl,
  regions,
}: {
  previewUrl?: string;
  regions: UiRegionPrediction[];
}) {
  return (
    <section className="preview-stage" aria-label="Image preview">
      {previewUrl ? (
        <img src={previewUrl} alt="Selected food preview" />
      ) : (
        <div className="preview-empty">
          <strong>No input selected</strong>
          <span>Upload a food image or use the sample data.</span>
        </div>
      )}
      <div className="overlay" aria-hidden="true">
        {regions.slice(0, 8).map((region) => {
          if (!region.bbox) {
            return null;
          }
          const { bbox } = region;
          return (
            <span
              className="overlay-box"
              key={`${region.source_id}-${region.detection_index}`}
              style={{
                left: `${(bbox.x1 / bbox.source_width) * 100}%`,
                top: `${(bbox.y1 / bbox.source_height) * 100}%`,
                width: `${((bbox.x2 - bbox.x1) / bbox.source_width) * 100}%`,
                height: `${((bbox.y2 - bbox.y1) / bbox.source_height) * 100}%`,
              }}
            >
              {String(region.displayIndex).padStart(2, "0")}
            </span>
          );
        })}
      </div>
    </section>
  );
}
```

Create `app/frontend/src/components/UploadControls.tsx`:

```typescript
export function UploadControls({
  mode,
  onModeChange,
  onImageSelected,
  onSample,
  onClear,
  disabled,
}: {
  mode: "image" | "video";
  onModeChange: (mode: "image" | "video") => void;
  onImageSelected: (file: File) => void;
  onSample: () => void;
  onClear: () => void;
  disabled: boolean;
}) {
  return (
    <div className="controls">
      <div className="segmented" aria-label="Input mode">
        <button
          className={mode === "image" ? "active" : ""}
          type="button"
          onClick={() => onModeChange("image")}
        >
          Image
        </button>
        <button
          className={mode === "video" ? "active" : ""}
          type="button"
          onClick={() => onModeChange("video")}
        >
          Video
        </button>
      </div>
      <label className="upload-button">
        Upload
        <input
          accept="image/jpeg,image/png,image/webp,image/heic"
          disabled={disabled || mode !== "image"}
          type="file"
          onChange={(event) => {
            const file = event.target.files?.[0];
            if (file) {
              onImageSelected(file);
            }
          }}
        />
      </label>
      <button type="button" onClick={onSample} disabled={disabled}>
        Sample
      </button>
      <button type="button" onClick={onClear} disabled={disabled}>
        Clear
      </button>
    </div>
  );
}
```

Create `app/frontend/src/components/DecisionSummary.tsx`:

```typescript
import type { AnalyzerResult, DecisionBand } from "../api/types";
import { PredictionRanking } from "./PredictionRanking";

const DECISION_LABELS: Record<DecisionBand, string> = {
  auto_accept: "Auto-accept",
  suggest: "Suggest",
  confirm: "Confirm",
  review: "Review",
};

function formatLabel(label: string) {
  return label.replaceAll("_", " ");
}

function formatConfidence(value: number) {
  return `${(value * 100).toFixed(2)}%`;
}

export function DecisionSummary({
  result,
  status,
}: {
  result?: AnalyzerResult;
  status: string;
}) {
  const decision = result?.decisionBand ?? "confirm";

  return (
    <aside className="decision-summary" aria-label="Prediction result">
      <div className="summary-topline">
        <span className={`decision-badge ${decision}`}>{result ? DECISION_LABELS[decision] : "Waiting"}</span>
        <strong>{formatConfidence(result?.strongestConfidence ?? 0)}</strong>
      </div>
      <h2>{result ? formatLabel(result.strongestLabel) : "No image selected"}</h2>
      <p>{result ? result.actionCopy : status === "analyzing" ? "Analyzing image." : "FoodLens will show calibrated predictions after upload."}</p>
      <PredictionRanking predictions={result?.topPredictions ?? []} />
      <dl className="model-meta">
        <div>
          <dt>Model</dt>
          <dd>{result?.modelName ?? "ResNet50 FT-V2"}</dd>
        </div>
        <div>
          <dt>Temp</dt>
          <dd>{result ? result.temperature.toFixed(6) : "0.958111"}</dd>
        </div>
        <div>
          <dt>Detector</dt>
          <dd>{result?.detectorStatus ?? "Waiting"}</dd>
        </div>
      </dl>
    </aside>
  );
}
```

Create `app/frontend/src/components/PredictionRanking.tsx`:

```typescript
function formatLabel(label: string) {
  return label.replaceAll("_", " ");
}

function formatConfidence(value: number) {
  return `${(value * 100).toFixed(2)}%`;
}

export function PredictionRanking({
  predictions,
}: {
  predictions: Array<[string, number]>;
}) {
  return (
    <div className="ranking">
      <div className="ranking-head">
        <span>Candidate ranking</span>
        <span>Score</span>
      </div>
      {predictions.map(([label, confidence]) => (
        <div className="ranking-row" key={label}>
          <strong>{formatLabel(label)}</strong>
          <span>{formatConfidence(confidence)}</span>
        </div>
      ))}
    </div>
  );
}
```

Create `app/frontend/src/components/CropReviewGrid.tsx`:

```typescript
import type { DecisionBand, UiRegionPrediction } from "../api/types";

const DECISION_LABELS: Record<DecisionBand, string> = {
  auto_accept: "Auto-accept",
  suggest: "Suggest",
  confirm: "Confirm",
  review: "Review",
};

function formatLabel(label: string) {
  return label.replaceAll("_", " ");
}

function formatConfidence(value: number) {
  return `${(value * 100).toFixed(2)}%`;
}

export function CropReviewGrid({ regions }: { regions: UiRegionPrediction[] }) {
  if (regions.length === 0) {
    return null;
  }

  return (
    <section className="crop-panel" aria-label="Detected food regions">
      <div className="panel-heading">
        <div>
          <p className="eyebrow">Multi-food review</p>
          <h2>Detected regions</h2>
        </div>
        <strong>{regions.length} crops</strong>
      </div>
      <div className="crop-grid">
        {regions.map((region) => (
          <article className="crop-card" key={`${region.source_id}-${region.detection_index}`}>
            <div className="crop-index">{String(region.displayIndex).padStart(2, "0")}</div>
            <div>
              <span className={`decision-badge ${region.foodlens.decision_band}`}>
                {DECISION_LABELS[region.foodlens.decision_band]}
              </span>
              <h3>{formatLabel(region.foodlens.top_label)}</h3>
              <p>
                {formatLabel(region.detector.label)} ·{" "}
                {formatConfidence(region.foodlens.top_confidence)}
              </p>
            </div>
          </article>
        ))}
      </div>
    </section>
  );
}
```

- [ ] **Step 6: Replace `App.tsx`**

Modify `app/frontend/src/App.tsx`:

```typescript
import { AnalyzerWorkbench } from "./components/AnalyzerWorkbench";

export default function App() {
  return <AnalyzerWorkbench />;
}
```

- [ ] **Step 7: Replace CSS with Analyzer Workbench styling**

Replace `app/frontend/src/styles.css` with:

```css
:root {
  --bg: #fffaf6;
  --paper: #ffffff;
  --panel: #fff6f0;
  --ink: #271714;
  --muted: #6b4a3f;
  --line: #ead6cd;
  --red: #b42318;
  --green: #2f6f32;
  --gold: #9a7000;
  --berry: #8a2b52;
  color: var(--ink);
  background: var(--bg);
  font-family:
    Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI",
    sans-serif;
}

* {
  box-sizing: border-box;
}

body {
  margin: 0;
  min-height: 100vh;
}

button,
input {
  font: inherit;
}

button {
  min-height: 36px;
  border: 1px solid var(--line);
  border-radius: 8px;
  background: var(--paper);
  color: var(--ink);
  cursor: pointer;
  font-weight: 700;
}

button:disabled {
  cursor: wait;
  opacity: 0.6;
}

.app-shell {
  width: min(100% - 40px, 1240px);
  margin: 0 auto;
  padding: 24px 0 32px;
}

.app-header,
.workbench,
.panel-heading,
.summary-topline,
.ranking-head,
.ranking-row,
.controls,
.model-meta {
  display: grid;
}

.app-header {
  grid-template-columns: 1fr auto;
  gap: 18px;
  align-items: end;
  margin-bottom: 18px;
}

.eyebrow {
  margin: 0 0 6px;
  color: var(--red);
  font-size: 0.74rem;
  font-weight: 800;
  letter-spacing: 0.08em;
  text-transform: uppercase;
}

h1,
h2,
h3,
p {
  margin-top: 0;
}

h1 {
  margin-bottom: 0;
  font-size: clamp(2rem, 4vw, 3.4rem);
  letter-spacing: 0;
  line-height: 0.96;
}

.status-notice {
  min-width: 150px;
  padding: 10px 12px;
  border: 1px solid var(--line);
  border-radius: 8px;
  background: var(--paper);
  color: var(--muted);
  font-weight: 800;
  text-align: center;
}

.workbench {
  grid-template-columns: minmax(0, 1fr) minmax(320px, 380px);
  gap: 18px;
  align-items: start;
}

.preview-stage {
  position: relative;
  display: grid;
  place-items: center;
  min-height: 340px;
  aspect-ratio: 16 / 10;
  overflow: hidden;
  border: 1px solid var(--line);
  border-radius: 8px;
  background: #f1dfd5;
}

.preview-stage img {
  width: 100%;
  height: 100%;
  object-fit: contain;
}

.preview-empty {
  display: grid;
  gap: 8px;
  justify-items: center;
  color: var(--muted);
  text-align: center;
}

.preview-empty strong {
  color: var(--ink);
  font-size: 1.25rem;
}

.overlay {
  position: absolute;
  inset: 0;
  pointer-events: none;
}

.overlay-box {
  position: absolute;
  display: grid;
  place-items: start;
  border: 2px solid var(--red);
  border-radius: 6px;
  color: white;
  font-size: 0.72rem;
  font-weight: 800;
  text-shadow: 0 1px 8px rgba(39, 23, 20, 0.7);
}

.controls {
  grid-template-columns: auto auto auto auto;
  gap: 10px;
  align-items: center;
  margin-top: 12px;
}

.segmented {
  display: inline-grid;
  grid-template-columns: repeat(2, 1fr);
  border: 1px solid var(--line);
  border-radius: 8px;
  overflow: hidden;
}

.segmented button {
  border: 0;
  border-radius: 0;
}

.segmented .active {
  background: var(--ink);
  color: white;
}

.upload-button {
  position: relative;
  display: inline-grid;
  place-items: center;
  min-height: 36px;
  padding: 0 16px;
  border: 1px solid var(--line);
  border-radius: 8px;
  background: var(--red);
  color: white;
  cursor: pointer;
  font-weight: 800;
}

.upload-button input {
  position: absolute;
  inset: 0;
  opacity: 0;
}

.decision-summary,
.crop-panel {
  border: 1px solid var(--line);
  border-radius: 8px;
  background: var(--paper);
}

.decision-summary {
  padding: 18px;
}

.summary-topline,
.ranking-head,
.ranking-row,
.model-meta {
  grid-template-columns: 1fr auto;
  gap: 12px;
  align-items: center;
}

.decision-badge {
  display: inline-flex;
  width: max-content;
  min-height: 28px;
  align-items: center;
  padding: 0 10px;
  border-radius: 8px;
  color: white;
  font-size: 0.72rem;
  font-weight: 800;
  text-transform: uppercase;
}

.decision-badge.auto_accept {
  background: var(--green);
}

.decision-badge.suggest {
  background: var(--gold);
}

.decision-badge.confirm {
  background: var(--red);
}

.decision-badge.review {
  background: var(--berry);
}

.decision-summary h2 {
  margin: 16px 0 8px;
  font-size: 2rem;
  text-transform: capitalize;
}

.decision-summary p {
  color: var(--muted);
  line-height: 1.5;
}

.ranking {
  margin-top: 18px;
}

.ranking-head {
  padding-bottom: 8px;
  border-bottom: 1px solid var(--line);
  color: var(--muted);
  font-size: 0.72rem;
  font-weight: 800;
  text-transform: uppercase;
}

.ranking-row {
  min-height: 38px;
  border-bottom: 1px solid var(--line);
  text-transform: capitalize;
}

.ranking-row:last-child {
  border-bottom: 0;
}

.model-meta {
  grid-template-columns: repeat(3, 1fr);
  margin-top: 18px;
}

.model-meta div {
  min-width: 0;
  padding: 10px;
  background: var(--panel);
}

.model-meta dt {
  color: var(--muted);
  font-size: 0.68rem;
  font-weight: 800;
  text-transform: uppercase;
}

.model-meta dd {
  margin: 5px 0 0;
  overflow-wrap: anywhere;
  font-size: 0.78rem;
  font-weight: 700;
}

.crop-panel {
  margin-top: 18px;
  padding: 16px;
}

.panel-heading {
  grid-template-columns: 1fr auto;
  gap: 16px;
  align-items: end;
  margin-bottom: 14px;
}

.panel-heading h2 {
  margin-bottom: 0;
}

.crop-grid {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 12px;
}

.crop-card {
  display: grid;
  grid-template-columns: 48px 1fr;
  gap: 12px;
  min-height: 116px;
  padding: 12px;
  border: 1px solid var(--line);
  border-radius: 8px;
  background: #fffdfb;
}

.crop-index {
  display: grid;
  place-items: center;
  width: 38px;
  height: 38px;
  border-radius: 8px;
  background: var(--panel);
  color: var(--red);
  font-weight: 900;
}

.crop-card h3 {
  margin: 10px 0 5px;
  text-transform: capitalize;
}

.crop-card p {
  color: var(--muted);
  font-size: 0.84rem;
}

@media (max-width: 920px) {
  .app-header,
  .workbench,
  .crop-grid {
    grid-template-columns: 1fr;
  }
}

@media (max-width: 620px) {
  .controls,
  .model-meta {
    grid-template-columns: 1fr;
  }
}
```

- [ ] **Step 8: Run frontend tests and build**

Run:

```bash
cd app/frontend
npm test
npm run build
```

Expected: all tests pass and Vite build succeeds.

- [ ] **Step 9: Commit**

```bash
git add app/frontend/src
git commit -m "feat: build React analyzer image slice"
```

## Task 6: Video Frame Sampling And Documentation

**Files:**

- Modify: `app/frontend/src/state/useAnalyzer.ts`
- Modify: `app/frontend/src/components/UploadControls.tsx`
- Modify: `app/frontend/src/api/foodlensClient.ts`
- Modify: `app/README.md`
- Modify: `app/backend/README.md`
- Modify: `README.md`

- [ ] **Step 1: Add video helper functions to API client**

Add this export to `app/frontend/src/api/foodlensClient.ts`:

```typescript
export function combineFrameResults(results: AnalyzerResult[]): AnalyzerResult {
  const first = results[0] ?? toLocalDemoResult();
  const regions = results.flatMap((result, frameIndex) =>
    result.regions.map((region) => ({
      ...region,
      source_id: `video frame ${frameIndex + 1}`,
    })),
  );
  const strongest = regions
    .slice()
    .sort((a, b) => b.foodlens.top_confidence - a.foodlens.top_confidence)[0];

  return {
    ...first,
    modelName: first.modelName.replace("Multi-food", "Video review"),
    detectorStatus: `${first.detectorStatus} · ${results.length} frames`,
    strongestLabel: strongest?.foodlens.top_label ?? first.strongestLabel,
    strongestConfidence:
      strongest?.foodlens.top_confidence ?? first.strongestConfidence,
    decisionBand: strongest?.foodlens.decision_band ?? first.decisionBand,
    topPredictions: strongest?.foodlens.top_k_predictions ?? first.topPredictions,
    regions: regions.map((region, index) => ({
      ...region,
      displayIndex: index + 1,
    })),
  };
}
```

- [ ] **Step 2: Extend analyzer state for video files**

In `app/frontend/src/state/useAnalyzer.ts`, add imports:

```typescript
import { combineFrameResults } from "../api/foodlensClient";
```

Add these helper functions above `useAnalyzer`:

```typescript
function waitForEvent(element: HTMLVideoElement, eventName: string) {
  return new Promise<void>((resolve, reject) => {
    const onSuccess = () => {
      element.removeEventListener(eventName, onSuccess);
      element.removeEventListener("error", onError);
      resolve();
    };
    const onError = () => {
      element.removeEventListener(eventName, onSuccess);
      element.removeEventListener("error", onError);
      reject(new Error(`Video ${eventName} failed.`));
    };
    element.addEventListener(eventName, onSuccess, { once: true });
    element.addEventListener("error", onError, { once: true });
  });
}

async function frameToFile(video: HTMLVideoElement, index: number) {
  const scale = Math.min(1, 640 / Math.max(video.videoWidth, video.videoHeight));
  const canvas = document.createElement("canvas");
  canvas.width = Math.max(1, Math.round(video.videoWidth * scale));
  canvas.height = Math.max(1, Math.round(video.videoHeight * scale));
  const context = canvas.getContext("2d");
  if (!context) {
    throw new Error("Canvas context unavailable.");
  }
  context.drawImage(video, 0, 0, canvas.width, canvas.height);

  const blob = await new Promise<Blob>((resolve, reject) => {
    canvas.toBlob(
      (candidate) => {
        if (candidate) {
          resolve(candidate);
        } else {
          reject(new Error("Could not extract frame."));
        }
      },
      "image/jpeg",
      0.88,
    );
  });

  return new File([blob], `foodlens_video_frame_${index + 1}.jpg`, {
    type: "image/jpeg",
  });
}
```

Add this function inside `useAnalyzer` before the return:

```typescript
  async function analyzeVideo(file: File) {
    if (previewUrl) {
      URL.revokeObjectURL(previewUrl);
    }
    const source = URL.createObjectURL(file);
    setPreviewUrl(source);
    setStatus("analyzing");
    setMessage("Sampling video frames");

    try {
      const video = document.createElement("video");
      video.src = source;
      video.muted = true;
      video.playsInline = true;
      video.preload = "auto";
      await waitForEvent(video, "loadedmetadata");
      const duration =
        Number.isFinite(video.duration) && video.duration > 0 ? video.duration : 1;
      const frameCount = Math.min(3, Math.max(1, Math.ceil(duration / 2)));
      const frameTimes = Array.from({ length: frameCount }, (_, index) =>
        frameCount === 1
          ? Math.min(0.1, duration * 0.5)
          : Math.min(duration - 0.05, (duration * (index + 1)) / (frameCount + 1)),
      );
      const frameResults = [];

      for (const [index, frameTime] of frameTimes.entries()) {
        video.currentTime = Math.max(0, frameTime);
        await waitForEvent(video, "seeked");
        const frameFile = await frameToFile(video, index);
        frameResults.push(await predictMultiFoodImage(frameFile));
      }

      setResult(combineFrameResults(frameResults));
      setStatus("result");
      setMessage("Video review");
    } catch {
      setResult(toLocalDemoResult());
      setStatus("result");
      setMessage("Local demo");
    }
  }
```

Return `analyzeVideo` from the hook.

- [ ] **Step 3: Wire video upload control**

In `app/frontend/src/components/UploadControls.tsx`, add an `onVideoSelected`
prop:

```typescript
  onVideoSelected: (file: File) => void;
```

Replace the upload input with mode-aware accept and handler:

```typescript
        <input
          accept={
            mode === "image"
              ? "image/jpeg,image/png,image/webp,image/heic"
              : "video/mp4,video/quicktime,video/webm"
          }
          disabled={disabled}
          type="file"
          onChange={(event) => {
            const file = event.target.files?.[0];
            if (!file) {
              return;
            }
            if (mode === "video") {
              onVideoSelected(file);
            } else {
              onImageSelected(file);
            }
          }}
        />
```

In `AnalyzerWorkbench.tsx`, pass the handler:

```typescript
            onVideoSelected={analyzer.analyzeVideo}
```

- [ ] **Step 4: Update docs**

Update `app/README.md` app section to include:

```markdown
## Frontend

The main frontend is a React/Vite Analyzer Workbench in `app/frontend`.

Run it locally:

```bash
cd app/frontend
npm install
npm run dev
```

The previous static prototype is archived in `app/frontend-static` for
reference.
```

Update `app/backend/README.md` with:

```markdown
## Test

Install backend dev dependencies:

```bash
pip install -r app/backend/requirements-dev.txt
```

Run backend tests:

```bash
python -m pytest tests/backend -v
```

Responses include `fallback_reason` when deterministic demo data is returned.
The frontend uses this to label backend fallback and local demo states.
```

Update the app paragraph in `README.md` to state:

```markdown
The current FoodLens app uses a FastAPI backend and a React/Vite Analyzer
Workbench under `app/frontend`. The archived static prototype remains under
`app/frontend-static` for reference.
```

- [ ] **Step 5: Run full verification**

Run:

```bash
python -m pytest tests/backend -v
cd app/frontend
npm test
npm run build
```

Expected: backend tests pass, frontend tests pass, and Vite build succeeds.

- [ ] **Step 6: Commit**

```bash
git add app/frontend/src app/README.md app/backend/README.md README.md
git commit -m "feat: add video review flow and docs"
```

## Task 7: Browser Verification

**Files:**

- Modify only files needed to fix issues discovered during verification.

- [ ] **Step 1: Start backend**

Run:

```bash
uvicorn app.backend.api:app --reload --port 8000
```

Expected: API available at `http://127.0.0.1:8000/health`.

- [ ] **Step 2: Start frontend**

Run:

```bash
cd app/frontend
npm run dev
```

Expected: Vite serves the app at `http://127.0.0.1:5173`.

- [ ] **Step 3: Verify browser workflow**

Open `http://127.0.0.1:5173` and verify:

- The first screen is the Analyzer Workbench.
- The sample button renders local demo data and labels it as Local demo.
- Uploading an image renders the preview, decision summary, crop cards, and
  backend fallback label when artifacts are missing.
- Detection boxes stay inside the preview stage.
- Responsive layout works around 1280px, 920px, and 390px widths.

- [ ] **Step 4: Fix verification issues with focused patches**

For CSS overlap or responsive issues, edit only `app/frontend/src/styles.css`.
For state or rendering issues, edit only the relevant component or
`useAnalyzer.ts`.

- [ ] **Step 5: Re-run verification commands**

Run:

```bash
python -m pytest tests/backend -v
cd app/frontend
npm test
npm run build
```

Expected: all tests and build pass after browser fixes.

- [ ] **Step 6: Commit browser verification fixes**

If files changed:

```bash
git add app/frontend/src
git commit -m "fix: polish FoodLens analyzer verification issues"
```

If no files changed:

```bash
git status --short
```

Expected: no uncommitted implementation changes.

## Self-Review

Spec coverage:

- React/Vite replaces the main `app/frontend`: Task 3.
- Static frontend is archived: Task 3.
- Analyzer Workbench image vertical slice: Tasks 4 and 5.
- Culinary Studio visual tone: Task 5 CSS.
- Backend contract and fallback metadata: Tasks 1 and 2.
- Focused backend and frontend tests: Tasks 1, 2, 4, and 5.
- Video frame sampling after image slice: Task 6.
- Documentation updates: Task 6.
- Browser verification: Task 7.

Red-flag scan:

- This plan avoids ambiguous markers and provides exact files, commands, and
  code for each implementation step.

Type consistency:

- Backend response field names match `schemas.py` and frontend `types.ts`.
- Frontend normalized fields match `AnalyzerResult` and component props.
- Decision bands use `auto_accept`, `suggest`, `confirm`, and `review`
  consistently across backend and frontend.
