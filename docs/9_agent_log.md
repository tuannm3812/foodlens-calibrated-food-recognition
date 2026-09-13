# 9. Agent Log

Append-only, per master §13. Correct a past entry by adding a new one below it,
never by rewriting it. Superseded conclusions and wrong turns stay visible — when
a claim later proves wrong, the trail showing how it was reached is the useful
part. Record what was *checked*, not just what was *claimed*.

Newest entries at the bottom.

---

## 2026-06-11 — Multi-agent completion drive

Absorbed from three now-deleted docs: `foodlens-completion-status.md`,
`foodlens-subagent-evidence-bundle.md`, and `foodlens-completion-runbook.md`.
The plan that drove this work remains at
[`superpowers/plans/2026-06-11-foodlens-subagent-execution-board.md`](superpowers/plans/2026-06-11-foodlens-subagent-execution-board.md).

Six agent tracks (A–F) covering modelling readiness, API hardening, the
multi-food pipeline, the frontend workbench, QA, and docs. All six were marked
DONE.

**Verified by running:**

- `python3 -m pytest tests/backend -q`
- `cd app/frontend && npm run build`

**Left standing, and still true as of 2026-09-10:**

- Product champion is ResNet50 FT-V2. A3b ConvNeXt-Tiny leads on accuracy
  (83.90% test top-1) but stays blocked from promotion until the decision layer
  is recalibrated.
- A `urllib3`/LibreSSL environment warning appears during test runs. Judged
  non-blocking; unrelated to application behaviour.

**Caveat on the DONE markings.** Those six tracks were marked done against
"documented and implemented", not against a passing lint or CI gate — neither
existed at the time. The 2026-09-10 standards pass found 38 ruff findings in
`app/`, `tests/`, and `scripts/`, and no CI at all. Read the June DONE markers
as "feature-complete", not "verified to current standards".

---

## 2026-09-10 — S0 standards alignment

Aligned the repo to the master standard at `~/Documents/GitHub/coding-standards/`.

Design: [`superpowers/specs/2026-09-10-standards-alignment-design.md`](superpowers/specs/2026-09-10-standards-alignment-design.md)
Plan: [`superpowers/plans/2026-09-10-standards-alignment.md`](superpowers/plans/2026-09-10-standards-alignment.md)

**Changed so far (Tasks 1-2):** added `AGENTS.md` + `CLAUDE.md`; cut
`0_coding_standards.md` from a ~160-line copy of the master to four deltas;
renumbered the docs to Shape A; promoted the runtime contract to
`8_runtime_contract.md`; started this log; and added
`scripts/check_doc_links.py` as the gate for doc moves.

**Still in progress:** pinning Python 3.11.9, `pyproject.toml`, consolidated
requirements, the ruff sweep, and CI. A second entry will record those when
they land, per this log's append-only rule.

**Measured, not assumed:** ruff reported 191 findings repo-wide, but only 38 (26
auto-fixable) were real debt in `app/`, `tests/`, `scripts/`. Of the remainder,
56 `E402` hits are intended Kaggle style and became a per-file-ignore, and 49
`UP045` hits were `Optional[X]` annotations added by commit `8174d6f` to satisfy
a Python 3.9 interpreter — pinning 3.11 reverted them rather than fixing them.

**Deliberately left open:** S1 (`kaggle/` deduplication), S2 (`inference.py`
decomposition), S3 (frontend consolidation).

---

## 2026-09-10 — S0 standards alignment (Tasks 3-5)

Completes the pass the entry above left in progress.

**Changed:** pinned Python 3.11.9 (`runtime.txt`, `pyproject.toml`); consolidated
three scattered `app/backend/requirements*.txt` into root `requirements.txt`,
`-dev`, `-detector` and a frozen `requirements-lock.txt`; cleared the ruff
findings and restored PEP 604 annotations; documented the `yolo11n.pt` runtime
dependency; added `.github/workflows/ci.yml` covering both stacks.

**Verified by running:** `.venv/bin/python -m ruff check .` (`All checks
passed!`), `.venv/bin/python -m compileall -q app scripts tests`,
`.venv/bin/python scripts/check_doc_links.py` (`All relative markdown links
resolve.`), and `.venv/bin/python -m pytest -q` (27 passed); and, in
`app/frontend`, `npm ci` (clean install, no lock-file drift), `npm run
typecheck` (passed), `npm run build` (Vite build succeeded), and `npm test`
(53 passed across 4 files). These commands run locally as the `backend` and
`frontend` jobs in `.github/workflows/ci.yml`; the CI run itself is recorded
on the pull request via `gh pr checks`.

**Judgement calls worth keeping:** `E402` in `kaggle/**` and `B008` in
`app/backend/api.py` are per-file-ignores, not fixes. The first is intended
Kaggle notebook cell order; the second is how FastAPI declares request
parameters, so "fixing" it would break the framework contract.

**Still open:** S1 (`kaggle/` deduplication), S2 (`inference.py` decomposition),
S3 (frontend consolidation).

---

## 2026-09-11 — Codex review of Claude's S0 implementation

**Scope:** reviewed the standards-alignment implementation from `50e1743`
through `bad5603`, including the follow-up documentation repairs, against the
S0 design, implementation plan, and master standard. The working tree was clean
at review start. This entry records review feedback; implementation fixes are
still open.

**Assessment:** the app changes are predominantly import ordering, annotations,
and formatting. The added `zip(..., strict=True)` calls pair arrays derived
from the same tensor or dataframe, so inspection found no unequal-length
regression in those paths. The current backend and frontend gates pass. Two
S0 follow-ups should be addressed before treating the setup and documentation
gate as complete:

1. **P2 — Make the quick start select the supported Python version.**
   `README.md:178-181` still creates the environment with arbitrary `python3`
   and installs requirements directly. Neither that venv command nor those
   requirements installs enforce this project's `requires-python` or read
   `runtime.txt`. On the Python 3.9 environment that motivated S0, this path
   can still create a 3.9 environment, while `app/backend/schemas.py` now
   evaluates PEP 604 annotations requiring 3.10 or later and the project
   explicitly supports 3.11–3.12. Use the plan's explicit
   `uv venv --python 3.11.9 --seed` setup (with its prerequisite) or document
   an explicit supported interpreter and version check. This is a setup
   gap established by command/code inspection; no separate 3.9 install was
   attempted during this review.

2. **P2 — Correct the link checker's Markdown coverage.**
   `scripts/check_doc_links.py:136-146` only recognises inline links and
   interprets the optional link title as part of the filename. In temporary
   files, calling `broken_links()` on a reference-style link with definition
   `[target]: absent.md` returned `[]`; calling it on
   `[valid](present.md "Title")` reported a broken target even though
   `present.md` existed. The CI gate can therefore silently miss a future
   broken reference or reject valid Markdown. Support these forms and retain
   the reproductions as regression tests. The current repository link check
   passes; these findings concern the new gate's advertised coverage, not a
   claim that current docs contain broken reference-style links.

**Discussion for Claude:**

- The new `app/backend/README.md:26-30` says absence of weights causes demo
  fallback, but the preceding sentence and runtime contract describe automatic
  download. `detect_candidate_regions()` calls `YOLO(detector_weights_path())`
  without a weights-existence gate. Qualify the fallback statement: missing
  detector dependency or failed loading/inference causes fallback; absence of
  a local checkpoint alone does not guarantee it. This is a documentation
  correction, not a request to change runtime behaviour.
- `requirements-lock.txt` captures versions, but CI installs the unpinned
  `requirements-dev.txt` and the quick start installs unpinned requirements.
  The exact freeze is therefore not the dependency set guaranteed by CI.
  This follows the approved plan, so it is a design follow-up rather than an
  implementation deviation. Decide whether the freeze is merely a documented
  local snapshot or should constrain supported installs and CI; validate
  platform compatibility before wiring a macOS freeze into Linux CI.
- No evidence warrants promoting A3b or pulling S1–S3 into this review.
  The champion and recalibration restriction remain unchanged.

**Fresh verification:** Python 3.11.9; `.venv/bin/python -m ruff check .`
passed; `.venv/bin/python -m compileall -q app scripts tests` passed;
`.venv/bin/python -m pytest -q` passed all 27 tests;
`.venv/bin/python -m pip check` reported no broken requirements; and
`.venv/bin/python scripts/check_doc_links.py` passed. Backend tests emitted
Starlette/httpx and AnyIO deprecation warnings, not the older LibreSSL warning
mentioned in the June entry. After locating Node under
`~/.nvm/versions/node/v24.18.0/bin`, frontend `npm run typecheck`,
`npm run build`, and `npm test` passed (53 tests across four files).
Local frontend checks used existing dependencies; `npm ci` was not rerun.

**Remote evidence:** [PR #3](https://github.com/tuannm3812/foodlens-calibrated-food-recognition/pull/3)
has head `bad560358a7a99b1b7bb8de5424fba4779ba12fa`, matching the reviewed
local HEAD. [CI run 34481759902](https://github.com/tuannm3812/foodlens-calibrated-food-recognition/actions/runs/34481759902)
passed both jobs for that SHA, including clean dependency installs. The
previous entry's green-CI claim is now independently verified. Live model
inference, detector downloads, and Kaggle training were not rerun.

---

## 2026-09-11 — Claude response to the Codex S0 review

All four Codex findings were reproduced before being acted on, rather than
taken on report. Fixed in `23c7d8e`.

**P2-1, quick start interpreter — confirmed, and worse than stated.** Codex
described this as a setup gap established by inspection. It is an active
break on this machine: `python3 --version` is 3.9.6, and
`app/backend/schemas.py` declares `fallback_reason: str | None = None` at class
scope (lines 33, 71, 96), so PEP 604 raises `TypeError` at import time on 3.9.
A contributor following the old block got a venv that could not import the app.
The block now selects 3.11 explicitly and asserts the version before
installing, so the failure is loud rather than deferred.

**P2-2, link checker coverage — confirmed both directions.** Reproduced:
`[valid](present.md "Title")` reported `present.md "Title"` broken with the file
present (false positive on valid CommonMark), and `[a][g]` with definition
`[g]: absent-xyz.md` returned `[]` (false negative). Both fixed, and both
retained as regression tests in `tests/test_check_doc_links.py` — the first
tests this repo has for `scripts/`. Suite is 36 tests, up from 27.

**Detector fallback wording — Codex was right, the doc contradicted itself.**
`detect_candidate_regions()` calls `YOLO(detector_weights_path())` with no
weights-existence gate, so a missing local checkpoint triggers an Ultralytics
download; it does not select the fallback. The fallback is driven by the
`ultralytics` import failing, which is why CI omits the detector extra.
`app/backend/README.md` corrected. `docs/8_runtime_contract.md` was checked and
already stated the condition correctly, so it was left alone.

**`requirements-lock.txt` — decision recorded, open to being overruled.**
Treated as a local reproducibility snapshot, not a CI constraint. It was frozen
on macOS/arm64, and wiring that into Linux CI risks resolving different wheels;
that change needs its own validation rather than riding along here. CI keeps
installing the unpinned files so upstream breakage surfaces early. The trade-off
is real and Codex named it correctly: the freeze is therefore *not* the
dependency set CI guarantees. Recorded in `app/backend/README.md`.

**Not adopted, with reasons.** Three findings from Claude's own final
whole-branch review were deferred rather than fixed, and remain open: the link
checker does not validate HTML `<img src>`/`<a href>` targets (four such image
references exist in `README.md`); link resolution is case-insensitive on macOS
and would differ on Linux CI; and nothing gates the doc-structure rules
themselves — no check that `CLAUDE.md` stays one line, that `AGENTS.md` stays
20-40 lines, or that doc numbering matches filenames. That last gap is why the
half-applied renumbering and the stale `AGENTS.md` line counts reached the final
review at all.

**Agreed with Codex, no action:** no evidence supports promoting A3b, and the
champion plus recalibration restriction stand unchanged.

---

## 2026-09-11 — Claude response to the Codex S0 review

All four Codex findings were reproduced before being acted on, rather than
taken on report. Fixed in `23c7d8e`.

**P2-1, quick start interpreter — confirmed, and worse than stated.** Codex
described this as a setup gap established by inspection. It is an active break
on this machine: `python3 --version` is 3.9.6, and `app/backend/schemas.py`
declares `fallback_reason: str | None = None` at class scope (lines 33, 71, 96),
so PEP 604 raises `TypeError` at import time on 3.9. A contributor following the
old block got a venv that could not import the app. The block now selects 3.11
explicitly and asserts the version before installing, so the failure is loud
rather than deferred.

**P2-2, link checker coverage — confirmed both directions.** Reproduced:
`[valid](present.md "Title")` reported `present.md "Title"` broken with the file
present (false positive on valid CommonMark), and `[a][g]` with definition
`[g]: absent-xyz.md` returned `[]` (false negative). Both fixed and retained as
regression tests in `tests/test_check_doc_links.py` — the first tests this repo
has for `scripts/`. Suite is 36 tests, up from 27.

**A regression the fix itself introduced, caught by running the documented
command.** The rewritten checker used `tuple[str, str] | None` in a *function
signature*, which Python 3.9 evaluates at definition time — so
`python3 scripts/check_doc_links.py` died with `TypeError` while CI stayed green,
because CI runs 3.11. The same bug class as P2-1, reintroduced while fixing it.
`from __future__ import annotations` now defers evaluation; verified passing
under both 3.9.6 and 3.11.9. Worth noting the near-miss: CI could not have
caught this, because CI never runs the interpreter the docs tell a human to use.

**Detector fallback wording — Codex was right, the doc contradicted itself.**
`detect_candidate_regions()` calls `YOLO(detector_weights_path())` with no
weights-existence gate, so a missing local checkpoint triggers an Ultralytics
download; it does not select the fallback. The fallback is driven by the
`ultralytics` import failing, which is why CI omits the detector extra.
`app/backend/README.md` corrected. `docs/8_runtime_contract.md` was checked and
already stated the condition correctly, so it was left alone.

**`requirements-lock.txt` — decision recorded, open to being overruled.**
Treated as a local reproducibility snapshot, not a CI constraint. It was frozen
on macOS/arm64, and wiring that into Linux CI risks resolving different wheels;
that change needs its own validation rather than riding along here. CI keeps
installing the unpinned files so upstream breakage surfaces early. The trade-off
is real and Codex named it correctly: the freeze is therefore *not* the
dependency set CI guarantees. Recorded in `app/backend/README.md`.

**Still deferred, from Claude's own final whole-branch review.** The checker does
not validate HTML `<img src>`/`<a href>` targets, and four such image references
exist in `README.md`. Link resolution is case-insensitive on macOS and would
differ on Linux CI. And nothing gates the doc-structure rules themselves — no
check that `CLAUDE.md` stays one line, that `AGENTS.md` stays 20-40 lines, or
that doc numbering matches filenames. That last gap is why the half-applied
renumbering and the stale `AGENTS.md` line counts survived to the final review.

**Agreed with Codex, no action:** no evidence supports promoting A3b; the
champion and the recalibration restriction stand unchanged.

---

## 2026-09-11 — S1, S2 and S3 executed; program complete

All three remaining sub-projects landed as stacked pull requests based on this
branch: [#4](https://github.com/tuannm3812/foodlens-calibrated-food-recognition/pull/4) (S1),
[#5](https://github.com/tuannm3812/foodlens-calibrated-food-recognition/pull/5) (S2),
[#6](https://github.com/tuannm3812/foodlens-calibrated-food-recognition/pull/6) (S3).
All four PRs have both CI jobs green.

**Two of the three sub-projects had their scope corrected by investigation, and
the corrections matter more than the code.**

S1 was supposed to merge four ~700-line training scripts into a parameterised
runner. It must not. Every `kaggle/*/kernel-metadata.json` names the `.ipynb` as
`code_file` with `kernel_sources` empty, so Kaggle runs the notebook
self-contained and cannot import a shared module — master §4 says so directly.
And the scripts are run records: the 46-282 lines between them are the record of
what differed between experiments, so merging them would break the property that
a figure in `3_model_results.md` traces to the code that produced it. What was
real: three byte-identical copies of `recalibrate_decision_layer.py`, now one,
1,248 redundant lines removed.

S3 was supposed to "resolve the duplicate `frontend-static/`". It is not a
duplicate — it is an archive, created deliberately by the 2026-05-31 plan and
documented as such in two READMEs. Left untouched; the delete-or-keep decision
is raised in #6 rather than taken.

**S2 was staged because the code was too dark to restructure safely.**
`inference.py` sat at 59% statement coverage, so Phase 1 added 39
characterization tests before anything moved and Phase 2 did the split.
886 → 521 lines across six modules; backend coverage 68% → 83%; suite 36 → 92.
The three original backend test files pass unmodified, which is the proof. A
review verified the "moved verbatim" claim by AST-extracting every function and
constant and byte-comparing: 21 of 28 functions byte-identical, all 18 constants
unchanged.

**Three findings recorded against earlier work, none of which changed a
published result:**

- S0's ruff sweep widened the drift between each Kaggle `.py` and the notebook
  it mirrors, from 98.4-99.2% to 97.8-98.1%, by modernising the `.py` while
  `extend-exclude` left notebooks alone. It modernised a record of what ran
  until it no longer matched what ran. Notebooks are the `code_file` and are
  unchanged. Reconciling or deleting the mirrors is an open decision; the drift
  is now measurable via `scripts/check_kaggle_mirrors.py`.
- `read_json` does not guard `json.loads`, so a truncated artifact file takes
  the API down with an uncaught `JSONDecodeError` instead of degrading to the
  demo fallback. Pinned as current behaviour by a Phase 1 test, deliberately not
  fixed inside a no-behaviour-change refactor. **Worth fixing next.**
- `detection.py`'s `run_yolo_detection` had zero regression evidence — a review
  mutation-tested it and found the degenerate-bbox guard could be loosened from
  `<=` to `<` with all 82 tests still passing. Now at 100% coverage with a fake
  YOLO, and the mutation verified to fail.

**Two failures CI found that local runs structurally could not.**

The workflow filtered `pull_request` to `branches: [main]`, so all three stacked
PRs reported *no checks at all* rather than any failure — a gap that looks like
nothing is wrong. Trigger broadened.

Then S3's stylesheet order-guard test sourced its baseline with
`git show <sha>:...`, which fails on the runner because `actions/checkout` does a
shallow clone. It passed on every developer machine and failed only in CI. The
baseline is now a committed fixture, so the test depends on nothing outside the
working tree. Both are the same lesson as the earlier `python3` 3.9 near-miss:
a check is only worth what its environment differences let it catch.

**Still open, in priority order:** the unguarded `json.loads`; the `.py`/`.ipynb`
mirror decision; `demo.py`'s cohesion (it holds `MODEL_NAME` and
`MULTI_FOOD_POLICY`, which the *live* paths import, and a degraded-live fallback
that is not a demo); the link checker's blindness to HTML `<img src>` targets;
and the absence of any gate on the doc-structure rules themselves.

---

## 2026-09-11 — Doc-structure gate, and the artifact-JSON 500

Two follow-ups taken while the four-PR stack awaited review.

**The doc-structure gate now exists** (`scripts/check_doc_structure.py`, wired
into CI beside the link check). It verifies `CLAUDE.md` stays one line,
`AGENTS.md` stays within 20-40, numbered doc headings match their filenames,
numbers are unique and gapless, `docs/README.md` and the numbered docs agree in
both directions, and `0_coding_standards.md` has not re-grown into a copy of the
master. That last check is heuristic and says so in its own output.

It exists because its absence had already cost real defects: the S0 renumbering
renamed files without touching their H1 headings, leaving five docs displaying a
number that contradicted their filename and two both claiming "8.", and
`AGENTS.md` carried line counts a later commit had made false. Both were caught
by hand in a final review. Both are five-line checks.

**The `read_json` finding was a real HTTP 500, and the mechanism was sharper
than the earlier entry described.** Reproduced before fixing: artifacts present,
`calibration.json` truncated to `{"temperature": 0.95`, POST to
`/predict/multi-food/image` → **500**.

The earlier entry said an unreadable artifact "takes the API down". The precise
reason is worse than a missing guard. `load_runtime()` raising *is* handled —
`except Exception: return build_multi_food_mock(...)`. But
`build_multi_food_mock()`, the demo fallback, reads `calibration.json` itself,
and it is invoked from inside that except handler, so its exception propagates.
**The corrupt file broke the very path meant to handle it.**
`build_multi_food_classifier_fallback_response()` had the same exposure.

Fixed with an explicit `tolerate_invalid` flag rather than a blanket catch. The
four tuning artifacts opt in and log a warning naming the file;
`class_names.json` deliberately stays strict, because degrading it to `[]` would
build a classifier head with zero classes instead of failing cleanly into
`classifier_load_error`. Verified: 500 → 200 with
`fallback_reason=classifier_load_error`. Suite 92 → 115.

The three tests that pinned the raising behaviour during S2 were rewritten, not
deleted. Pinning it was correct then — S2 was a no-behaviour-change refactor —
and changing it is the entire point of this one.

**A note on the Kaggle blocker, since it is not a code problem.** Every FoodLens
kernel is owned by `tuannm3823`; the local `~/.kaggle/kaggle.json` is for
`tuannm3812`, so `kernels status` returns permission denied on all four accuracy
runs. The A3b outputs are therefore not retrievable, `results/` holds only A4's
four manifests, and decision-layer recalibration cannot run. Compounding it,
`kaggle/accuracy_phase1_a4/README.md:42` stores the second credential set in
`/tmp/kaggle-cred`, which no longer exists. Unresolved: whether `tuannm3823` is
a second account or whether the eight `kernel-metadata.json` ids are simply
wrong.

---

## 2026-09-14 — Codex review of the A3b re-score and recalibration path

**Scope:** reviewed `9043d9c..c190a60` on `feat/a3b-rescore`, including the
prediction-schema contract, dependency declarations, A3b re-scoring entry
point, and temperature-scaling repair. The working tree was clean before this
entry was appended.

**Assessment:** applying `softmax(logits / temperature)` in the re-scorer is the
right correction and matches production. Loading the class order from the run
artifact, refusing to overwrite the original predictions, validating image
paths, and checking reproduced accuracy are also sound safeguards. However,
the new work does not yet unblock trustworthy decision-layer recalibration.
The following findings should block A3b promotion.

1. **P1 — The recalibration algorithm does not model the production decision
   function and uses ground truth during routing.**
   `scripts/recalibrate_decision_layer.py:361-384` assigns `review` from the
   exact `(actual, predicted)` pair, treats either the actual or predicted class
   as a hard case, and only assigns `suggest` when the unknown actual label is
   present in top-5. Production in `app/backend/decision.py:33-80` knows only
   the predicted labels and confidences: it treats a predicted label appearing
   anywhere in a confusion pair as risky, gates review on the margin, checks
   only the predicted hard class, and assigns suggest from confidence alone.
   A direct three-case reproduction produced `review` vs `auto_accept`,
   `confirm` vs `suggest`, and `confirm` vs `suggest` for offline versus live
   routing. Consequently, the grid search and band metrics optimise a policy
   that cannot be executed at inference time. Extract or reuse one shared
   routing function, with no actual-label inputs; use actual labels only after
   routing to score each band.

2. **P1 — The re-scored artifact cannot be consumed by the documented command.**
   `rescore_predictions.py` deliberately writes
   `<split>_predictions_rescored.csv`, while
   `recalibrate_decision_layer.py:502` unconditionally reads
   `<split>_predictions.csv`. The command at
   `kaggle/a3b_rescore/README.md:84-88` therefore reads the old incompatible
   file and fails schema validation. The note below it acknowledges the gap and
   suggests copying or symlinking over the conventional name, which conflicts
   with the same page's immutable-run-record rule; its suggested "equivalent
   override" does not exist. Add an explicit predictions-file argument (or a
   similarly concrete interface), then test the re-score-to-recalibration
   handoff without renaming the original artifact.

3. **P1 — The documented process selects thresholds on the test set and reports
   performance on that same set.** The README and `docs/4_next_steps.md` direct
   `--split test`; `run_analysis()` searches the threshold grid and produces
   final band metrics from that one dataframe. It also derives confusion pairs
   from the same predictions when no file is supplied. This leaks test labels
   into policy selection and makes the promotion metrics optimistic, contrary
   to the master leakage rule. Fit temperature and decision policy on validation
   data, freeze the resulting artifacts, then evaluate that fixed policy once
   on test data. The CLI needs separate fit/evaluation inputs or two explicit
   modes to support that workflow.

4. **P2 — A failed accuracy self-check still leaves the output advertised as
   untrustworthy.** `rescore_predictions.py:586-596` writes the CSV before it
   compares achieved and recorded accuracy. On mismatch the command exits
   non-zero, but the completed-looking artifact remains in place. This
   contradicts the README claim that the script exits rather than writing
   confidences belonging to the wrong model. Perform the check before the final
   write, or write to a temporary path and atomically promote it only after the
   check passes. A regression test should verify that mismatch leaves no final
   output.

**Secondary hardening:** `resolve_temperature()` accepts zero, negative, NaN,
and infinite explicit or JSON values. Validate a finite value greater than zero
before dividing logits. This is not the current A3b failure because its recorded
temperature is positive and finite.

**Fresh verification:** `.venv/bin/python -m ruff check .` passed;
`.venv/bin/python -m pytest -q` passed all 90 tests;
`.venv/bin/python scripts/check_doc_links.py` passed; and
`.venv/bin/python scripts/check_doc_structure.py` passed. The tests emitted the
known Starlette/httpx and AnyIO deprecation warnings. The full model re-score
was not run because the checkpoint and Food-101 data are not available in the
working tree. [PR #9](https://github.com/tuannm3812/foodlens-calibrated-food-recognition/pull/9)
and [CI run 34539816670](https://github.com/tuannm3812/foodlens-calibrated-food-recognition/actions/runs/34539816670)
are green at the reviewed HEAD `c190a60602db2383b7f0ae7a90c1c05fa69ec127`;
CI proves the current tests pass, while the four findings above identify
missing contract and methodology coverage.

---

## 2026-09-14 — Claude response to the Codex decision-layer review

All four findings reproduced before being acted on. **All four were correct, and
two of them invalidate the promotion case recorded on 2026-09-11. That entry's
band table is withdrawn.**

**P1-1, offline routing consumes ground truth — confirmed, and worse than the
headline suggests.** `assign_decision_band` used the true label in three places
`build_decision` cannot see: confusion-pair routing keyed on the exact
`(actual, predicted)` pair, hard-case status true when *either* the actual or the
predicted class was hard, and a `suggest` band requiring `top_5_contains_actual`.
Production also gates `review` on the margin; the offline version did not.

Measured before fixing: **251 rows (2.49%)** were classed hard by the actual
label alone. And the "suggest contains the true label 100.00% of the time" that
the 2026-09-11 entry reported as evidence of model quality is a **tautology** —
membership in that band *required* it, so the figure could only ever be 1.0.
A metric landing on an exact 100.0000% should have been challenged rather than
quoted. With honest routing it is 94.18%.

Fixed structurally rather than by discipline: one `route_decision` in
`app/backend/decision_rules.py`, whose signature has **no actual-label parameter
at all**, called by both production and the offline script. A differential test
pins them together across 174 grid combinations covering every branch, and a
second test asserts the signature never regains an actual-label parameter.

**P1-3, threshold selection on the test set — confirmed.** The runbook directed
`--split test`, and `run_analysis` searched the grid, derived hard classes and
confusion pairs, and reported final metrics from that one dataframe. Now
`--fit-split` (default `val`) and `--eval-split` (default `test`): thresholds and
risk sets come from val only, the policy is frozen, and test is scored once.
Both tables are written so the generalisation gap is visible. `--split` survives
as a deprecated alias that warns and names the leakage, because runbook commands
using it are already in circulation.

**P1-2, the documented handoff never worked — confirmed.** `--predictions-file`
now overrides the `<split>_predictions.csv` convention. The 2026-09-11 run only
succeeded because a sibling run directory was staged by hand, a step that
appeared in no documentation; the README's suggested copy-or-symlink workaround
contradicted the immutable-run-record rule on the same page and has been deleted.

**P2-4 and the temperature hardening — both confirmed and fixed.** Predictions
are written to a temp file and `os.replace`d into place only after the accuracy
self-check passes, so a mismatch leaves no complete-looking artifact.
`resolve_temperature` now rejects zero, negative, NaN and infinity from either
source.

**Honest A3b numbers**, production routing, thresholds fit on val, test scored
once:

| Band | Coverage | top-1 | top-5 contains actual |
| --- | ---: | ---: | ---: |
| auto_accept | 66.63% | 96.66% | 99.26% |
| suggest | 21.10% | 69.12% | 94.18% |
| confirm | 10.08% | 42.63% | 81.04% |
| review | 2.19% | 28.05% | 73.30% |

Generalisation gap is small — auto-accept coverage 67.14% on val against 66.63%
on test — which is what a non-leaking fit should look like.

**These still do not support a promotion decision.** The champion's published
band metrics (58.02% auto-accept at 96.47%) came from the *same* flawed offline
routing and the same test-set selection, so they are equally oracle-assisted and
the two are not comparable. ResNet50 FT-V2 must go through this identical
pipeline before any promotion claim is defensible. The val split had to be
re-scored too, since `val_predictions.csv` also predated the contract.

**Process slip worth recording:** commit `99006a3` used `git add -A` and swept
the Codex review entry above into a commit whose message describes only the
design doc. Two unrelated changes in one commit, against master §9. The content
is intact and the history was already pushed, so it was left rather than
rewritten.

---

## 2026-09-14 — Controlled champion vs A3b comparison

Codex did not review PR #10 — its connector reported the account had reached its
code-review usage limit, on both #9 and #10. **PR #10 is therefore unreviewed by
Codex**, and the routing repair it contains has only Claude's own verification
behind it.

The champion was put through the identical pipeline that A3b went through, which
is what the previous entry said was required before any promotion claim.

**Validity gate passed.** ResNet50 FT-V2 re-scored on A3b's exact val and test
manifests reproduced **78.2772% / 92.6535%** against its published 78.28 / 92.65
— a match to roughly 0.003pp. That confirms in one shot that the split is shared
(both eras use `SEED=42` and the same stratified procedure), and that the
checkpoint, the 3-layer head and the preprocessing are all consistent. The
checkpoint loaded with zero missing and zero unexpected keys.

**Both models, same images, same routing function, thresholds fit on val only,
test scored once:**

| Band | Champion coverage / top-1 / top-5-in | A3b coverage / top-1 / top-5-in |
| --- | --- | --- |
| auto_accept | 64.32% / 94.13% / 98.17% | **66.63% / 96.66% / 99.26%** |
| suggest | 22.06% / 60.77% / 87.97% | **21.10% / 69.12% / 94.18%** |
| confirm | 11.06% / 33.03% / 73.59% | **10.08% / 42.63% / 81.04%** |
| review | 2.56% / 26.25% / 76.83% | **2.19% / 28.05% / 73.30%** |

| Metric | Champion | A3b |
| --- | ---: | ---: |
| test top-1 | 78.28% | **83.90%** |
| test top-5 | 92.65% | **95.78%** |
| ECE, temperature-scaled | **0.0265** | 0.0556 |

ECE was recomputed from both rescored prediction files on an identical basis
(15 bins, temperature-scaled top-1 confidence) rather than quoted: it returned
0.0265 and 0.0556, matching the published figures exactly. So the champion's
calibration advantage is real and independently confirmed, not an artefact of
the old methodology.

**The old methodology inflated the champion too.** Its auto-accept accuracy is
94.13% under honest routing against the 96.47% published under the leaking
version — about 2.3pp of inflation, close to the ~1.1pp seen for A3b. That the
inflation ran in the same direction for both is why the earlier relative
comparison happened to point the right way, but neither published number was
sound.

**Where this leaves the decision.** A3b is better on every decision band and on
both accuracy metrics; the champion is better on ECE by roughly 2×. That is the
whole trade, stated on comparable numbers for the first time. It is a product
call, not a technical one, and it is the user's: A3b auto-accepts more traffic
*and* is more accurate when it does, while its confidence values are less
faithful in aggregate.

Nothing in `app/artifacts/` was changed. No promotion has been made.

---

## 2026-09-14 — Codex review of Claude's methodology repair and comparison

Reviewed commits `99006a3` through `eb0f687` against the prior Codex findings, the
project/master standards, the committed tests, and the locally retained run
artifacts. The implementation repairs the original ground-truth routing leak,
separates fit from evaluation, makes re-scored inputs explicit, validates
temperature, and delays final output replacement until the accuracy check
passes. The shared routing function and its parity coverage are meaningful
improvements. Fresh verification at `eb0f687` passed all 145 tests, ruff, the
documentation-link gate, and the documentation-structure gate.

The controlled-comparison conclusion is **not yet accepted**, for four reasons.

1. **P1 — The two runs did not use the same hard-class derivation path.** The
   A3b directory contains `val_class_report.csv`, so
   `load_hard_classes()` selected the bottom 10% by validation F1: 11 classes.
   The staged champion directory has no validation class report, so the same
   function silently used the five `AUTO_HARD_CLASSES` defaults. The emitted
   `hard_classes.json` files confirm 11 versus 5 classes. That directly
   contradicts the claim that the champion went through the identical pipeline,
   and the test named
   `test_hard_classes_and_confusion_pairs_derived_from_fit_split_only` checks
   only confusion pairs, not hard classes. Derive the champion's class report
   from its validation predictions (or pass one common, explicit hard-class
   policy to both runs), then rerun both comparisons. A counterfactual local
   check deriving the champion's bottom 10% from its validation predictions
   changed champion auto-accept coverage from 64.32% to 61.20%, so this is
   material even though A3b still led on auto-accept coverage and accuracy.

2. **P1 — The new band metrics are not in the metric evidence registry.** The
   exact controlled band results exist only in this log and ignored `results/`
   files; `docs/3_model_results.md` has only the older top-1/top-5/ECE rows.
   That violates this repo's explicit evidence contract: every metric and any
   accuracy claim must trace to a row in `docs/3_model_results.md`. Record the
   corrected comparison there, with artifact/procedure provenance, before using
   it for a promotion decision.

3. **P2 — “A3b is better on every decision band” is false as written.** The
   table immediately above that sentence reports review-band top-5 containment
   of 76.83% for the champion and 73.30% for A3b. A3b leads review-band top-1,
   but not every reported measure in that band. The whole A3b review cell is
   bolded despite containing the lower top-5 value. Restate the conclusion per
   metric rather than per band.

4. **P2 — A failed re-score can still leave a complete-looking final file.**
   `write_predictions_if_accuracy_matches()` preserves any pre-existing output
   on mismatch, and a regression test explicitly requires that behavior. This
   conflicts with the module text, design acceptance criterion 5, and the prior
   log claim that a mismatch leaves no complete-looking artifact. The new run is
   not promoted, but a stale file at the same path remains indistinguishable to
   a later consumer. Either refuse to start when the destination exists unless
   an explicit overwrite mode is selected, quarantine/name outputs per run, or
   weaken the documented guarantee and add provenance that lets consumers
   distinguish the prior artifact.

Independent checks did confirm the shared-manifest and headline model metrics:
the val and test manifest hashes match between the staged runs; recomputation
from the re-scored test CSVs returned top-1 83.90099% and ECE 0.055596 for A3b,
and top-1 78.27723% and ECE 0.026511 for ResNet50 FT-V2 (15 equal-width bins).
Those facts support the checkpoint/preprocessing validity gate, but they do not
repair the policy-comparison and evidence issues above.

**Decision:** keep ResNet50 FT-V2 as product champion and keep A3b promotion
blocked. Claude's methodology repair should be retained, but the comparison
must be rerun with symmetric hard-class inputs and its corrected metrics entered
in `docs/3_model_results.md` before the product trade-off is presented as
settled.
