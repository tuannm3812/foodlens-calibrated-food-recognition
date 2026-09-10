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
