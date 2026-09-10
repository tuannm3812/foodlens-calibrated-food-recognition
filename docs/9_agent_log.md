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
