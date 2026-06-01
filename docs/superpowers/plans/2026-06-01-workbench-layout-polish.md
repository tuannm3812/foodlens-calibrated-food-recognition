# Workbench Layout Polish Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Polish the FoodLens workbench layout by removing duplicate analyze/result wording, aligning URL input with mode selection, and reducing the outer pane framing.

**Architecture:** This is a frontend-only React/CSS refinement. `AnalyzerWorkbench.tsx` owns page copy, `UploadControls.tsx` owns the mode/URL/action control structure, and `styles.css` owns the visual shell and responsive layout.

**Tech Stack:** React, TypeScript, Vite, Vitest, Testing Library, CSS.

---

### Task 1: Page Copy And Control Structure

**Files:**
- Modify: `app/frontend/src/components/AnalyzerWorkbench.test.tsx`
- Modify: `app/frontend/src/components/AnalyzerWorkbench.tsx`
- Modify: `app/frontend/src/components/UploadControls.tsx`

- [ ] **Step 1: Write failing component tests**

Assert that the analyze page heading is `Food Recognition`, that `Analysis Result` is not the main heading, and that the upload controls expose a mode/URL row plus action row.

- [ ] **Step 2: Run targeted frontend tests**

Run: `npm test -- src/components/AnalyzerWorkbench.test.tsx`

Expected: FAIL before implementation because the heading and control row class do not exist yet.

- [ ] **Step 3: Update React components**

Change analyze page copy and wrap the segmented mode control plus URL form in a new `.upload-controls__input-row`.

- [ ] **Step 4: Re-run targeted tests**

Run: `npm test -- src/components/AnalyzerWorkbench.test.tsx`

Expected: PASS.

### Task 2: Visual Layout Polish

**Files:**
- Modify: `app/frontend/src/styles.css`

- [ ] **Step 1: Adjust CSS shell and controls**

Remove the heavy outer workbench border/background/card shadow, tighten the title row, and align `.upload-controls__input-row` as `grid-template-columns: auto minmax(280px, 1fr)`.

- [ ] **Step 2: Run full frontend checks**

Run: `npm test`, `npm run typecheck`, and `npm run build`.

Expected: all pass.

- [ ] **Step 3: Capture a refined snapshot**

Run the app locally and capture `docs/ui-snapshots/foodlens-refined-layout-polish-2026-06-01.png`.

- [ ] **Step 4: Update snapshot README and commit**

Update `docs/ui-snapshots/README.md`, then commit the implementation and snapshot.
