# FoodLens Precision Lab Product Shell Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rework the React/Vite FoodLens frontend into the approved Precision Lab Product Shell while preserving the existing analyzer behavior.

**Architecture:** Keep the current React state, API client, and backend contract unchanged. Add small presentational components for the product shell and decision policy strip, then update existing analyzer components and CSS to match the approved layout. Use tests to lock the shell structure, result hierarchy, and existing video/upload behavior.

**Tech Stack:** React, TypeScript, Vite, Vitest, Testing Library, CSS, lucide-react.

---

## File Structure

Modify:

- `app/frontend/src/components/AnalyzerWorkbench.tsx`: product shell composition, title/status row, workspace placement, decision strip placement.
- `app/frontend/src/components/DecisionSummary.tsx`: result card hierarchy with decision badge, confidence, top label, confidence track, ranking, metadata.
- `app/frontend/src/components/PreviewStage.tsx`: preserve media/overlay behavior while adding stable Precision Lab structure.
- `app/frontend/src/components/UploadControls.tsx`: compact control bar classes and accessible labels.
- `app/frontend/src/components/StatusNotice.tsx`: compact title-row status panel.
- `app/frontend/src/components/CropReviewGrid.tsx`: align the crop grid class with the product shell.
- `app/frontend/src/components/AnalyzerWorkbench.test.tsx`: shell, nav, decision strip, status, and regression assertions.
- `app/frontend/src/styles.css`: Precision Lab tokens, shell layout, result card, preview, controls, decision strip, crop grid, responsive rules.

Create:

- `app/frontend/src/components/ProductHeader.tsx`: FoodLens brand and modern nav.
- `app/frontend/src/components/DecisionPolicyStrip.tsx`: four decision policy tiles.

Do not modify:

- `app/backend/*`
- `app/frontend/src/api/*`, unless tests reveal a real presentation bug.
- `app/frontend/src/state/useAnalyzer.ts`, unless tests reveal a real presentation bug.

## Task 1: Lock Product Shell Structure With Tests

**Files:**

- Modify: `app/frontend/src/components/AnalyzerWorkbench.test.tsx`

- [ ] **Step 1: Add shell and decision strip assertions**

In `app/frontend/src/components/AnalyzerWorkbench.test.tsx`, update the Testing
Library import so it includes `within`:

```tsx
import { act, render, renderHook, screen, within } from "@testing-library/react";
```

Then replace the idle analyzer test with this version:

```tsx
  it("renders the Precision Lab product shell", () => {
    render(<AnalyzerWorkbench />);
    const decisionPolicy = screen.getByLabelText("Decision policy");

    expect(screen.getByRole("banner")).toBeInTheDocument();
    expect(screen.getByRole("link", { name: "FoodLens home" })).toBeInTheDocument();
    expect(screen.getByRole("navigation", { name: "Product navigation" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Analyze" })).toHaveAttribute(
      "aria-current",
      "page",
    );
    expect(screen.getByRole("button", { name: "Review" })).toBeDisabled();
    expect(screen.getByRole("button", { name: "Models" })).toBeDisabled();
    expect(screen.getByRole("heading", { name: "Analysis Result" })).toBeInTheDocument();
    expect(screen.getByText("Live API · Image/video upload · Calibrated crop review")).toBeInTheDocument();
    expect(decisionPolicy).toBeInTheDocument();
    expect(within(decisionPolicy).getByText("Auto-accept")).toBeInTheDocument();
    expect(within(decisionPolicy).getByText("Suggest")).toBeInTheDocument();
    expect(within(decisionPolicy).getByText("Confirm")).toBeInTheDocument();
    expect(within(decisionPolicy).getByText("Review")).toBeInTheDocument();
    expect(screen.getByText("No input selected")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Sample" })).toBeInTheDocument();
  });
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
cd app/frontend
npm test -- src/components/AnalyzerWorkbench.test.tsx
```

Expected: FAIL because `ProductHeader`, the `Analysis Result` heading, disabled `Review`/`Models` nav buttons, and decision policy strip do not exist yet.

- [ ] **Step 3: Leave failing tests uncommitted**

Do not commit failing tests separately. Continue to Task 2, then commit the passing shell slice.

## Task 2: Add Product Header And Decision Policy Components

**Files:**

- Create: `app/frontend/src/components/ProductHeader.tsx`
- Create: `app/frontend/src/components/DecisionPolicyStrip.tsx`
- Modify: `app/frontend/src/components/AnalyzerWorkbench.tsx`

- [ ] **Step 1: Create `ProductHeader.tsx`**

Create `app/frontend/src/components/ProductHeader.tsx`:

```tsx
import { Activity, FlaskConical, ListChecks } from "lucide-react";

const NAV_ITEMS = [
  {
    label: "Analyze",
    icon: Activity,
    active: true,
    disabled: false,
  },
  {
    label: "Review",
    icon: ListChecks,
    active: false,
    disabled: true,
  },
  {
    label: "Models",
    icon: FlaskConical,
    active: false,
    disabled: true,
  },
] as const;

export function ProductHeader() {
  return (
    <header className="product-header">
      <a className="product-brand" href="#" aria-label="FoodLens home">
        FoodLens
      </a>
      <nav className="product-nav" aria-label="Product navigation">
        {NAV_ITEMS.map((item) => {
          const Icon = item.icon;

          return (
            <button
              key={item.label}
              type="button"
              className={item.active ? "is-active" : ""}
              aria-current={item.active ? "page" : undefined}
              disabled={item.disabled}
            >
              <Icon size={15} aria-hidden="true" />
              {item.label}
            </button>
          );
        })}
      </nav>
      <div className="product-header__meta" aria-label="Prototype status">
        <span>Prototype</span>
      </div>
    </header>
  );
}
```

- [ ] **Step 2: Create `DecisionPolicyStrip.tsx`**

Create `app/frontend/src/components/DecisionPolicyStrip.tsx`:

```tsx
const POLICIES = [
  {
    label: "Auto-accept",
    description: "High confidence",
    tone: "auto",
  },
  {
    label: "Suggest",
    description: "Ranked choices",
    tone: "suggest",
  },
  {
    label: "Confirm",
    description: "User check",
    tone: "confirm",
  },
  {
    label: "Review",
    description: "Known risk",
    tone: "review",
  },
] as const;

export function DecisionPolicyStrip() {
  return (
    <section className="decision-policy" aria-label="Decision policy">
      {POLICIES.map((policy) => (
        <article
          key={policy.label}
          className={`decision-policy__tile decision-policy__tile--${policy.tone}`}
        >
          <span className="decision-policy__dot" aria-hidden="true" />
          <span>{policy.label}</span>
          <strong>{policy.description}</strong>
        </article>
      ))}
    </section>
  );
}
```

- [ ] **Step 3: Update `AnalyzerWorkbench.tsx` composition**

Replace the contents of `app/frontend/src/components/AnalyzerWorkbench.tsx` with:

```tsx
import { CropReviewGrid } from "./CropReviewGrid";
import { DecisionPolicyStrip } from "./DecisionPolicyStrip";
import { DecisionSummary } from "./DecisionSummary";
import { PreviewStage } from "./PreviewStage";
import { ProductHeader } from "./ProductHeader";
import { StatusNotice } from "./StatusNotice";
import { UploadControls } from "./UploadControls";
import { useAnalyzer } from "../state/useAnalyzer";

export function AnalyzerWorkbench() {
  const analyzer = useAnalyzer();

  return (
    <div className="app-shell">
      <ProductHeader />
      <main className="workbench-shell">
        <section className="workbench-title-row" aria-label="Analysis overview">
          <div>
            <p className="eyebrow">Analyze</p>
            <h1>Analysis Result</h1>
            <p className="workbench-subtitle">
              Live API · Image/video upload · Calibrated crop review
            </p>
          </div>
          <StatusNotice
            status={analyzer.status}
            message={analyzer.message}
            source={analyzer.result?.source}
          />
        </section>

        <section className="workbench-layout" aria-label="FoodLens analysis workspace">
          <div className="workbench-primary">
            <PreviewStage
              mode={analyzer.mode}
              previewUrl={analyzer.previewUrl}
              result={analyzer.result}
            />
            <UploadControls
              mode={analyzer.mode}
              status={analyzer.status}
              onModeChange={analyzer.setMode}
              onUploadImage={(file) => {
                void analyzer.analyzeImage(file);
              }}
              onVideoSelected={(file) => {
                void analyzer.analyzeVideo(file);
              }}
              onSample={analyzer.loadSample}
              onClear={analyzer.clear}
            />
          </div>
          <DecisionSummary result={analyzer.result} />
        </section>

        <DecisionPolicyStrip />
        <CropReviewGrid result={analyzer.result} />
      </main>
    </div>
  );
}
```

- [ ] **Step 4: Run shell tests**

Run:

```bash
cd app/frontend
npm test -- src/components/AnalyzerWorkbench.test.tsx
```

Expected: PASS for shell tests after CSS-independent markup is present.

- [ ] **Step 5: Run typecheck**

Run:

```bash
cd app/frontend
npm run typecheck
```

Expected: PASS.

- [ ] **Step 6: Commit shell structure**

Run:

```bash
git add app/frontend/src/components/ProductHeader.tsx app/frontend/src/components/DecisionPolicyStrip.tsx app/frontend/src/components/AnalyzerWorkbench.tsx app/frontend/src/components/AnalyzerWorkbench.test.tsx
git commit -m "feat: add Precision Lab product shell"
```

## Task 3: Strengthen Decision Summary And Status Presentation

**Files:**

- Modify: `app/frontend/src/components/DecisionSummary.tsx`
- Modify: `app/frontend/src/components/StatusNotice.tsx`
- Modify: `app/frontend/src/components/AnalyzerWorkbench.test.tsx`

- [ ] **Step 1: Add accessible result hierarchy assertions**

Add this test under the `AnalyzerWorkbench` describe block:

```tsx
  it("keeps the result card hierarchy after loading the sample", async () => {
    const user = userEvent.setup();
    render(<AnalyzerWorkbench />);

    await user.click(screen.getByRole("button", { name: "Sample" }));

    expect(screen.getByLabelText("Decision summary")).toHaveClass("decision-card");
    expect(screen.getByText("Local demo")).toBeInTheDocument();
    expect(screen.getByText("ravioli")).toBeInTheDocument();
    expect(screen.getByText("97.2%")).toBeInTheDocument();
    expect(screen.getByText("Detected regions")).toBeInTheDocument();
    expect(screen.getByText("Decision")).toBeInTheDocument();
    expect(screen.getByText("Confidence")).toBeInTheDocument();
    expect(screen.getByText("Model")).toBeInTheDocument();
    expect(screen.getByText("Detector")).toBeInTheDocument();
  });
```

- [ ] **Step 2: Run the targeted test to verify it fails if labels are missing**

Run:

```bash
cd app/frontend
npm test -- src/components/AnalyzerWorkbench.test.tsx -t "result card hierarchy"
```

Expected: FAIL if `Confidence` is not rendered as a separate label.

- [ ] **Step 3: Replace `DecisionSummary.tsx`**

Replace `app/frontend/src/components/DecisionSummary.tsx` with:

```tsx
import type { AnalyzerResult, DecisionBand } from "../api/types";
import { PredictionRanking } from "./PredictionRanking";

type DecisionSummaryProps = {
  result: AnalyzerResult | null;
};

const DECISION_LABELS: Record<DecisionBand, string> = {
  auto_accept: "Auto accept",
  suggest: "Suggest",
  confirm: "Confirm",
  review: "Review",
};

function formatConfidence(value: number): string {
  return `${Math.round(value * 1000) / 10}%`;
}

export function DecisionSummary({ result }: DecisionSummaryProps) {
  if (!result) {
    return (
      <section className="decision-card decision-card--empty" aria-label="Decision summary">
        <p className="eyebrow">Decision</p>
        <h2>No input selected</h2>
        <p className="muted-copy">
          Upload a plated dish image, select a short video, or load the sample to
          review FoodLens crop decisions.
        </p>
      </section>
    );
  }

  const confidence = formatConfidence(result.strongestConfidence);

  return (
    <section className="decision-card" aria-label="Decision summary">
      <div className="decision-card__topline">
        <div>
          <span className="metric-label">Decision</span>
          <span className={`decision-badge decision-badge--${result.decisionBand}`}>
            {DECISION_LABELS[result.decisionBand]}
          </span>
        </div>
        <div className="confidence-metric">
          <span className="metric-label">Confidence</span>
          <strong>{confidence}</strong>
        </div>
      </div>

      <div className="decision-card__prediction">
        <h2>{result.strongestLabel}</h2>
        <p>{result.actionCopy}</p>
      </div>

      <div className="confidence-track" aria-hidden="true">
        <span style={{ width: confidence }} />
      </div>

      <PredictionRanking predictions={result.topPredictions} />

      <dl className="model-metadata">
        <div>
          <dt>Model</dt>
          <dd>{result.modelName}</dd>
        </div>
        <div>
          <dt>Temperature</dt>
          <dd>{result.temperature.toFixed(3)}</dd>
        </div>
        <div>
          <dt>Detector</dt>
          <dd>{result.detectorStatus}</dd>
        </div>
        <div>
          <dt>Artifacts</dt>
          <dd>{result.artifactStatus}</dd>
        </div>
      </dl>
    </section>
  );
}
```

- [ ] **Step 4: Replace `StatusNotice.tsx`**

Replace `app/frontend/src/components/StatusNotice.tsx` with:

```tsx
import type { ResultSource } from "../api/types";
import type { AnalyzerStatus } from "../state/useAnalyzer";

const SOURCE_LABELS: Record<ResultSource, string> = {
  live: "Live API",
  backend_fallback: "Backend fallback",
  local_demo: "Local demo",
};

const STATUS_LABELS: Record<AnalyzerStatus, string> = {
  idle: "Ready",
  loading: "Analyzing",
  ready: "Result ready",
  error: "Needs attention",
};

type StatusNoticeProps = {
  status: AnalyzerStatus;
  message: string;
  source?: ResultSource;
};

export function StatusNotice({ status, message, source }: StatusNoticeProps) {
  const sourceLabel = source ? SOURCE_LABELS[source] : "Workbench";

  return (
    <aside className={`status-notice status-notice--${status}`} aria-live="polite">
      <span className="status-notice__label">{STATUS_LABELS[status]}</span>
      <strong>{sourceLabel}</strong>
      <span className="status-notice__message">{message}</span>
    </aside>
  );
}
```

- [ ] **Step 5: Run targeted and full frontend tests**

Run:

```bash
cd app/frontend
npm test -- src/components/AnalyzerWorkbench.test.tsx
npm test
```

Expected: both PASS.

- [ ] **Step 6: Commit result card changes**

Run:

```bash
git add app/frontend/src/components/DecisionSummary.tsx app/frontend/src/components/StatusNotice.tsx app/frontend/src/components/AnalyzerWorkbench.test.tsx
git commit -m "feat: refine FoodLens decision summary"
```

## Task 4: Apply Precision Lab CSS And Responsive Layout

**Files:**

- Modify: `app/frontend/src/styles.css`
- Modify: `app/frontend/src/components/PreviewStage.tsx`
- Modify: `app/frontend/src/components/UploadControls.tsx`
- Modify: `app/frontend/src/components/CropReviewGrid.tsx`

- [ ] **Step 1: Preserve preview and upload behavior tests**

Run:

```bash
cd app/frontend
npm test -- src/components/AnalyzerWorkbench.test.tsx -t "video previews|video upload"
```

Expected: PASS before CSS/markup class changes.

- [ ] **Step 2: Update `PreviewStage.tsx` copy and stable empty state**

In `app/frontend/src/components/PreviewStage.tsx`, replace the empty state block:

```tsx
        ) : (
          <div className="preview-stage__empty">
            <span>Awaiting {mode} input</span>
          </div>
        )}
```

with:

```tsx
        ) : (
          <div className="preview-stage__empty">
            <span className="preview-stage__empty-icon" aria-hidden="true">
              +
            </span>
            <strong>No subject detected</strong>
            <span>Upload a food {mode} or load the sample to begin analysis.</span>
          </div>
        )}
```

- [ ] **Step 3: Update `UploadControls.tsx` label classes only**

Keep behavior unchanged. In `app/frontend/src/components/UploadControls.tsx`, ensure the returned section still has `className="upload-controls"` and the two groups still use:

```tsx
<div className="segmented-control" role="group" aria-label="Input mode">
```

and:

```tsx
<div className="control-row">
```

Do not remove the `Upload` label text because tests use `getByLabelText("Upload")`.

- [ ] **Step 4: Update `CropReviewGrid.tsx` grid class**

In `app/frontend/src/components/CropReviewGrid.tsx`, keep the heading text
`Detected regions` unchanged. Replace the crop grid wrapper class:

```tsx
<div className="crop-review__grid">
```

with:

```tsx
<div className="crop-grid">
```

- [ ] **Step 5: Replace `styles.css` with Precision Lab styles**

Replace `app/frontend/src/styles.css` with a CSS file that includes these required sections and selectors:

```css
:root {
  color: #1f2420;
  background: #f7f8f6;
  font-family:
    Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI",
    sans-serif;
  font-synthesis: none;
  text-rendering: optimizeLegibility;
  --lab-bg: #f7f8f6;
  --lab-surface: #ffffff;
  --lab-surface-soft: #fbfcfa;
  --lab-border: #d9dfd7;
  --lab-border-strong: #cbd5cb;
  --lab-ink: #181d19;
  --lab-muted: #5c665c;
  --lab-red: #d42020;
  --lab-green: #2f7d59;
  --lab-gold: #946200;
  --lab-blue: #315f8f;
  --lab-radius: 8px;
}
```

The replacement must define styles for all of these selectors because current components use them:

```css
.app-shell
.product-header
.product-brand
.product-nav
.product-header__meta
.workbench-shell
.workbench-title-row
.workbench-subtitle
.status-notice
.workbench-layout
.workbench-primary
.preview-stage
.preview-stage__frame
.preview-image-layer
.preview-stage__image
.preview-overlay-layer
.preview-stage__empty
.preview-stage__empty-icon
.bbox-overlay
.upload-controls
.segmented-control
.control-row
.upload-button
.decision-card
.decision-card--empty
.decision-card__topline
.decision-card__prediction
.decision-badge
.confidence-metric
.confidence-track
.ranking-list
.model-metadata
.decision-policy
.decision-policy__tile
.decision-policy__dot
.crop-review
.crop-grid
.crop-card
```

Responsive rules must include:

```css
@media (max-width: 900px) {
  .product-header,
  .workbench-title-row,
  .workbench-layout {
    grid-template-columns: 1fr;
  }

  .product-nav {
    justify-content: flex-start;
    overflow-x: auto;
  }

  .decision-policy {
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }
}

@media (max-width: 640px) {
  .app-shell {
    padding: 0;
  }

  .workbench-shell {
    padding: 16px;
  }

  .decision-policy,
  .crop-grid {
    grid-template-columns: 1fr;
  }
}
```

- [ ] **Step 6: Run tests, typecheck, and build**

Run:

```bash
cd app/frontend
npm test
npm run typecheck
npm run build
```

Expected: all PASS.

- [ ] **Step 7: Commit CSS/layout polish**

Run:

```bash
git add app/frontend/src/styles.css app/frontend/src/components/PreviewStage.tsx app/frontend/src/components/UploadControls.tsx app/frontend/src/components/CropReviewGrid.tsx
git commit -m "style: apply Precision Lab frontend shell"
```

## Task 5: Browser Verification And Final Fixes

**Files:**

- Modify only files needed to fix visual or test regressions found during verification.

- [ ] **Step 1: Start dev servers**

Run backend:

```bash
python3 -m uvicorn app.backend.api:app --reload --port 8000
```

Run frontend in another session:

```bash
cd app/frontend
npm run dev
```

Expected:

- FastAPI available at `http://127.0.0.1:8000/health`.
- Vite available at its printed local URL, normally `http://127.0.0.1:5173/`.

- [ ] **Step 2: Verify HTTP probes**

Run:

```bash
curl -sS http://127.0.0.1:8000/health
curl -sS -I http://127.0.0.1:5173/
```

Expected:

```json
{"status":"ok"}
```

and an HTTP 200 response from Vite.

- [ ] **Step 3: Inspect in browser or fallback with screenshots**

Use the Browser plugin if available. If the Browser plugin is unavailable, use the best available local browser automation fallback. Verify:

- desktop width around 1280px: header, nav, title/status row, preview-left/result-right layout, decision strip, crop grid;
- mobile width around 390px: layout stacks without overlap;
- sample image flow: result card shows local demo state, confidence, ranking, metadata, crop cards;
- video mode: upload accepts video, preview renders a video element, result policy remains confirm after frame analysis;
- no visible text overlaps or oversized hero treatment.

- [ ] **Step 4: Fix visual regressions with focused patches**

If verification reveals issues, patch only the affected component or CSS selector. After each fix, run:

```bash
cd app/frontend
npm test
npm run typecheck
npm run build
```

Expected: all PASS.

- [ ] **Step 5: Run final frontend verification**

Run:

```bash
cd app/frontend
npm test
npm run typecheck
npm run build
```

Expected: all PASS.

- [ ] **Step 6: Commit verification fixes**

If Step 4 changed files, run:

```bash
git add app/frontend/src
git commit -m "fix: polish Precision Lab shell verification"
```

If Step 4 changed nothing, do not create an empty commit.

## Final Verification

Run from the repository root:

```bash
python3 -m pytest tests/backend -v
```

Run from `app/frontend`:

```bash
npm test
npm run typecheck
npm run build
```

Expected:

- backend tests pass;
- frontend tests pass;
- TypeScript passes;
- Vite production build succeeds;
- worktree is clean except ignored build artifacts.
