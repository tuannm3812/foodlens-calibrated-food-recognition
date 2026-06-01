# Workbench Layout Polish Design

## Goal

Polish the merged React/Vite FoodLens workbench so the first screen feels less repetitive, more aligned, and less like a framed page inside another page.

## Approved Changes

- Keep the top navigation label `Analyze`, but change the analyze page heading from `Analysis Result` to `Food Recognition`.
- Keep result-specific wording inside the decision/result cards instead of repeating it in the page title.
- Restructure upload controls into a clearer two-row layout:
  - First row: `Image` / `Video` segmented mode control beside the URL field.
  - Second row: `Upload`, `Sample`, and `Clear` actions.
- Loosen the outer workbench pane by removing the heavy bordered shell treatment and letting the preview, controls, decision, policy, runtime, and review sections read as the functional surfaces.
- Preserve the current backend/API behavior and all existing upload, sample, clear, direct image URL, and YouTube URL flows.

## Non-Goals

- No backend changes.
- No new navigation destinations.
- No redesign of crop cards, runtime data, or model metadata content in this pass.

## Verification

- Component tests should assert the new heading copy and the upload control grouping.
- Existing URL upload tests must continue to pass.
- A new UI snapshot should be captured after implementation for comparison.
