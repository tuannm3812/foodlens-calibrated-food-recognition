# UI Snapshots

Visual reference captures used while comparing application refinements.

## Folder convention

- Store screenshots under a `YYYY-MM-DD/` folder for the capture date.
- Keep filenames descriptive: `{app-or-page}-{state-or-purpose}-{YYYY-MM-DD}.png`.
- Add every committed screenshot to this index with the viewport, source URL or app state, and why it was captured.

## 2026-06-01

- [`foodlens-main-old-ui-2026-06-01.png`](2026-06-01/foodlens-main-old-ui-2026-06-01.png) - Old `main` branch FoodLens UI, captured at `1440x1100` from `http://127.0.0.1:5174/` in the initial analysis state.
- [`foodlens-refined-ui-2026-06-01.png`](2026-06-01/foodlens-refined-ui-2026-06-01.png) - Refined React/Vite FoodLens UI after merging `foodlens-react-vite-refinement` into `main`, captured at `1440x1100` from `http://127.0.0.1:5174/` in the initial analysis state.
- [`foodlens-refined-layout-polish-2026-06-01.png`](2026-06-01/foodlens-refined-layout-polish-2026-06-01.png) - Refined UI after the layout polish pass, captured at `1440x1100`, showing the `Food Recognition` heading, aligned mode/URL controls, and a lighter unframed workspace shell.

## 2026-06-02

- [`foodlens-west-bookings-polish-2026-06-02.png`](2026-06-02/foodlens-west-bookings-polish-2026-06-02.png) - Refined analyzed-state UI using the West Kitchen image URL, captured at `1440x1013`, showing labeled crop overlays and first-viewport result context.
- [`foodlens-west-bookings-aligned-2026-06-02.png`](2026-06-02/foodlens-west-bookings-aligned-2026-06-02.png) - Alignment pass for the analyzed West Kitchen image state, captured at `1440x1013`, with the result status aligned to the decision panel and crop cards aligned with the selected crop detail.
- [`foodlens-west-bookings-crop-polish-2026-06-02.png`](2026-06-02/foodlens-west-bookings-crop-polish-2026-06-02.png) - Crop review polish pass for the analyzed West Kitchen image state, captured at `1440x1013`, with denser crop cards, structured selected crop metadata, and compact runtime readiness chips.
- [`foodlens-west-bookings-analysis-polish-2026-06-02.png`](2026-06-02/foodlens-west-bookings-analysis-polish-2026-06-02.png) - Analysis polish pass for the analyzed West Kitchen image state, captured at `1440x1013`, with crop confidence bars, crop status pills, and a richer selected crop summary.
