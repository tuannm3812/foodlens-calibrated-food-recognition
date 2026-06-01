# FoodLens Precision Lab Product Shell Design

Date: 2026-06-01

## 1. Goal

Polish the React/Vite FoodLens frontend so it feels stronger than the archived
static UI while keeping the improved implementation structure from the
refinement branch.

The current React app is technically cleaner, but the archived static UI has a
better product feel: clearer FoodLens branding, denser analysis workspace,
stronger result card hierarchy, top navigation, and a more intentional decision
policy strip. This pass should bring those strengths into React with a more
modern Precision Lab visual style.

## 2. Chosen Direction

Use a Product Shell Rebuild.

Keep the existing React/Vite app, typed API client, backend fallback metadata,
image flow, video frame sampling, and tests. Rework the frontend shell and
component presentation around the archived static UI's stronger layout:

- compact product header;
- modern navigation;
- analysis title and status row;
- large preview stage with overlay;
- right-side result card;
- upload and mode controls below the preview;
- decision policy strip;
- crop review section below the primary workspace.

This is a UI polish pass, not another backend rewrite.

## 3. Visual Direction

Use a Precision Lab style.

The UI should feel modern, precise, and product-ready:

- near-white and white work surfaces;
- subtle cool-gray borders;
- restrained shadows used only for primary preview/result surfaces;
- tighter spacing than the current React workbench;
- red used as the FoodLens identity color and high-priority signal;
- dark neutral text for hierarchy;
- no decorative gradients, marketing hero layout, or oversized headline
  treatment.

The visual language should remain food-aware through the brand and result
content, but it should read more like a polished analysis product than a
restaurant or recipe site.

## 4. Navigation And Shell

Replace the old placeholder navigation with real or near-term sections:

- `Analyze`: active primary screen.
- `Review`: future review queue/history surface.
- `Models`: future model/runtime/artifact surface.

The nav items do not need full routes in this pass unless the existing React
structure makes it trivial. They may render as inactive shell links or buttons,
but the active `Analyze` state should be clear.

Header requirements:

- `FoodLens` remains a strong first-viewport brand signal.
- Header height stays compact.
- Header actions may be icon-style placeholders if useful, but should not imply
  unsupported account functionality.

## 5. Component Design

Keep the component boundaries created during the React refinement, but adjust
their rendered structure and CSS.

Primary components:

- `AnalyzerWorkbench`: owns the product shell layout, header, title row,
  primary workspace, decision strip, and crop review placement.
- `StatusNotice`: becomes a compact status panel in the title row, not the
  dominant right-side element.
- `PreviewStage`: keeps image/video rendering and overlay behavior. Its visual
  treatment should move closer to the archived static preview stage: gridded
  empty state, stable aspect ratio, precise overlay boxes, and no layout shift.
- `UploadControls`: remains functional, but should look like a compact control
  bar under the preview.
- `DecisionSummary`: should become the strongest visual result surface. It
  should lead with the decision band and confidence, then top label, action
  copy, confidence track, ranked predictions, and model/runtime metadata.
- `CropReviewGrid`: remains below the primary workspace and should visually
  align with the product shell instead of feeling like a separate page section.

The implementation should avoid changing API types or analyzer state unless a
small shape adjustment is required for presentation.

## 6. Data Flow

Data flow remains unchanged from the React refinement:

1. User selects image, video, or sample.
2. React renders the preview immediately.
3. Image uploads call `POST /predict/multi-food/image`.
4. Video uploads sample frames in-browser and call the same image endpoint per
   frame.
5. API responses normalize through `foodlensClient`.
6. The UI renders preview overlays, result summary, rankings, runtime status,
   fallback state, and crop cards.

No backend contract changes are planned.

## 7. Error Handling And Fallbacks

Fallback transparency remains required.

The UI should preserve these states:

- idle;
- analyzing;
- live result;
- backend fallback;
- local demo fallback;
- unsupported or failed media workflow.

The compact status panel should make the current state visible without
overpowering the analysis result.

Video summaries must continue to require confirmation and must not surface
`auto_accept` from sampled still-frame crops.

## 8. Responsive Behavior

Desktop:

- header and nav remain single-row;
- primary workspace uses preview-left and result-right;
- decision strip spans below the workspace;
- crop review grid follows below.

Tablet/mobile:

- layout stacks in this order: header, title/status, preview, controls, result,
  decision strip, crop review;
- navigation can scroll horizontally or compress spacing, but text must not
  overlap;
- preview, buttons, badges, result labels, and crop cards must keep stable
  dimensions and avoid layout shift.

## 9. Testing

Frontend tests should be updated only where UI text or structure changes affect
current assertions.

Required checks:

- existing API normalization tests continue to pass;
- analyzer smoke/render tests continue to pass;
- video upload routing still renders the video preview;
- decision summary still reflects fallback/local/live states;
- production build and typecheck pass.

Manual verification:

- run Vite locally;
- inspect the analyzer at desktop and mobile widths;
- verify sample image, image upload, video preview, fallback messaging, overlay
  alignment, result card hierarchy, decision strip, and crop grid.

## 10. Out Of Scope

- New backend endpoints or inference behavior.
- Real navigation routes for Review or Models.
- Accounts, notifications, persistence, or archive/database screens.
- Marketing landing page.
- New image assets.
- Replacing the React state model or typed API client.

## 11. Approval Notes

Approved choices:

- Use the hybrid direction: old UI's stronger visual shell with the new
  React/Vite implementation.
- Modernize it with Precision Lab styling.
- Use modern navigation labels: `Analyze`, `Review`, `Models`.
- Prefer Product Shell Rebuild over a shallow reskin or full recomposition.
