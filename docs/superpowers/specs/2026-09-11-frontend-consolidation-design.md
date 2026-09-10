# S3 — Frontend Consolidation (Design)

Date: 2026-09-11
Status: Approved by decomposition; scope corrected after investigation

## 1. Scope correction

The S0 decomposition described S3 as *"Resolve the duplicate `frontend-static/`,
split the 1232-line stylesheet and the two ~470-line modules."*

**`app/frontend-static/` is not duplication.** It is an archived prototype,
created deliberately by the 2026-05-31 React/Vite refinement plan and documented
as such in two places: `README.md:165` ("Archived static prototype") and
`app/README.md:31` ("The old static prototype is archived under
`app/frontend-static`"). Nothing imports it, nothing builds it, and no CI job
touches it.

Deleting it is defensible — git history preserves it, and it is 1,967 lines of
unmaintained code that no linter checks. But it is the user's reference
material, deliberately kept, and removing someone's archive while they are away
is not a call to make unasked. **This design leaves it in place and raises the
question in the pull request.**

So S3 is the two real problems: one oversized stylesheet and one oversized API
client.

## 2. Targets

| File | Lines | Problem |
| --- | ---: | --- |
| `src/styles.css` | 1,232 | 144 selectors, no internal structure |
| `src/api/foodlensClient.ts` | 478 | Four unrelated responsibilities in one module |
| `src/state/useAnalyzer.ts` | 453 | Hook plus four standalone helpers |

## 3. `foodlensClient.ts` — four responsibilities

Reading the module, the seams are unambiguous:

| New module | Contents |
| --- | --- |
| `api/labels.ts` | `DECISION_ACTIONS`, `DECISION_BANDS`, `DETECTOR_STATUS_LABELS`, `FALLBACK_REASON_LABELS`, `DETECTOR_ROLE_LABELS`, `labelFromToken`, `detectorStatusLabel`, `modelNameLabel`, `fallbackReasonLabel`, `artifactStatusLabel`, `detectorLabel`, `detectorRoleLabel`, `regionStatusLabel`, `actionCopyForDecisionBand` |
| `api/guards.ts` | `isRecord`, `isNumber`, `isString`, `isDecisionBand`, `isBoundingBox`, `isTopKPrediction`, `isDecisionThresholds`, `isBackendRegionPrediction`, `isBackendMultiFoodResponse`, `isBackendRuntimeStatus` |
| `api/normalize.ts` | `resultSource`, `normalizeMultiFoodResponse`, `toLocalDemoResult`, `combineFrameResults` |
| `api/foodlensClient.ts` (kept) | `API_BASE_URL`, `FoodLensApiError`, `parseErrorMessage`, `postUrlPrediction`, `predictMultiFoodImage`, `predictMultiFoodImageUrl`, `predictMultiFoodYoutubeUrl`, `fetchRuntimeStatus`, `isUserInputApiError` — HTTP transport and nothing else |

`foodlensClient.ts` re-exports `normalizeMultiFoodResponse`,
`toLocalDemoResult`, `combineFrameResults` and `isUserInputApiError`, because
`foodlensClient.test.ts` and the components import them from there today. No
import site outside the `api/` directory changes.

## 4. `useAnalyzer.ts` — extract the pure helpers

`createPreviewUrl`, `waitForEvent`, `videoSampleTimes` and `sourceHost` are
standalone functions that happen to live beside the hook. They move to
`state/analyzerHelpers.ts`. The hook itself stays — its `useState` block and
effects are one cohesive unit and splitting them would obscure, not clarify.

`videoSampleTimes` is pure arithmetic and currently has no direct test; the move
makes it trivially testable, and this design adds that test.

## 5. `styles.css` — split with the cascade preserved

Two constraints:

- **Cascade order is behaviour.** 144 selectors, some overriding earlier ones.
  Any reordering is a visual regression that no test would catch.
- **`styles.test.ts` reads the file directly.** Its `cssRule()` helper does
  `readFileSync(resolve(__dirname, "styles.css"))` and regex-matches a selector.
  Split the file and every assertion silently returns `""` — and `""` still
  *contains* nothing, so some assertions could pass vacuously. This is the same
  silent-no-op hazard S2 dealt with, in a different costume.

Approach: move the rules into `src/styles/` as themed partials, and keep
`src/styles.css` as an entry file containing only `@import` statements **in the
exact original order**. Vite inlines these at build time, so the cascade is
byte-equivalent.

Partitioning follows the existing top-to-bottom grouping, so no rule crosses
another:

`tokens.css` (`:root`) · `shell.css` · `header.css` · `workbench.css` ·
`preview.css` · `controls.css` · `decision.css` · `crops.css` · `responsive.css`

The exact cut points are chosen during implementation by reading the file; the
binding rule is that concatenating the partials in `@import` order reproduces
the original rule sequence.

`styles.test.ts`'s `cssRule()` is updated to read and concatenate all partials
in `@import` order. **Every assertion stays byte-identical** — only the source of
the text changes. A test proving the concatenation matches the original file's
rule order is added, so a future partial added in the wrong place fails loudly.

## 6. What must not change

- Rendered output. No component markup or class name changes.
- The public import surface of `api/foodlensClient.ts`.
- Any assertion in `styles.test.ts`, `foodlensClient.test.ts`,
  `AnalyzerWorkbench.test.tsx` or `App.test.tsx`.
- `app/frontend-static/`.

## 7. Acceptance criteria

1. `npm run typecheck`, `npm run build`, `npm test` all pass — 53 tests, plus
   the new ones.
2. Concatenating `src/styles/*.css` in `@import` order yields the same sequence
   of rules as the pre-split `styles.css`. Verified by a test, not by eye.
3. No file in `src/` exceeds 400 lines.
4. Assertions in the four existing test files are unmodified (`git diff` shows
   changes only to `cssRule()`'s file reading).
5. `git diff --stat -- app/frontend-static` is empty.
6. Backend untouched: `git diff --stat -- app/backend tests` is empty.

## 8. Risks

- **A vacuous CSS assertion.** If `cssRule()` returns `""`, `toContain` fails
  loudly for non-empty expectations — but a future assertion expecting absence
  would pass vacuously. Criterion 2's order test is the real guard.
- **`@import` resolution.** Vite resolves relative `@import` at build; if the
  build inlines them differently than expected, `npm run build` catches it.
  Criterion 1 covers this.
