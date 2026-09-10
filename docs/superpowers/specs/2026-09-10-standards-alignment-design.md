# S0 — Standards Alignment & Scaffolding (Design)

Date: 2026-09-10
Status: Approved, ready for implementation planning

## 1. Purpose

Bring this repository into line with the master coding standard at
`~/Documents/GitHub/coding-standards/coding_standards.md`, and install the
tooling that makes the alignment hold. This is the first of four sub-projects
in a larger refactor program; it comes first because it defines what "done"
means for the other three.

Success is measurable: a fresh agent session opens the repo and immediately
knows the standard, the champion model, and the open risks; and a CI run
proves the Python and TypeScript stacks both build, lint, and test clean.

## 2. Scope

### In scope

1. Agent instruction layer — `AGENTS.md` and `CLAUDE.md`.
2. Docs restructure to Shape A numbering, with an append-only agent log.
3. Environment and packaging — Python 3.11 pin, `pyproject.toml`,
   consolidated requirements.
4. CI — GitHub Actions covering backend and frontend.
5. Repo hygiene — undeclared artifacts, stale worktree, notebook output rule.

### Explicit non-goals

These are deferred to their own spec/plan/implementation cycles:

- **S1** — deduplicating `kaggle/` training scripts.
- **S2** — decomposing the 883-line `app/backend/inference.py`.
- **S3** — consolidating the duplicate frontend and splitting the
  1232-line stylesheet.
- Rewriting git history. The 44 MB `.git` is tolerable and master §9 warns
  against force-pushing shared history.
- Any model, accuracy, or calibration work.

## 3. Current State

Measured on 2026-09-10.

| Standard | Rule | This repo today |
| --- | --- | --- |
| §13 | Layer 2 is `AGENTS.md` plus a one-line `CLAUDE.md` | Neither file exists |
| §13 | "Never copy this file into a project" | `docs/02_coding_standards.md` is a ~160-line copy of the master |
| §2 | Single-digit numbering, declared shape | Two-digit `01_`–`08_`, plus a parallel `docs/superpowers/` tree |
| §13 | Append-only agent collaboration log | Four overlapping, rewritable status docs, already stale |
| §9 | `<type>(<scope>)` conventional commits | Types present, scopes missing |
| §4/§8 | Clear notebook outputs when code changes | Seven notebooks retain outputs, three of them active (deliberate — see §4.6) |
| — | Siblings pin Python 3.11 | README claims 3.11; local interpreter is 3.9 |

### Lint debt

`ruff check --select E,F,I,B,UP --line-length 100 --target-version py311`:

| Scope | Errors | Auto-fixable | Notes |
| --- | ---: | ---: | --- |
| `app/`, `tests/`, `scripts/` | 38 | 26 | Real debt, small |
| `kaggle/` | 153 | 42 | 56 are `E402`, which is intended style |
| **Total** | **191** | **68** | |

Two findings shape the plan:

- **`E402` (56 hits) is correct, not debt.** Kaggle-mirrored scripts place
  imports after a configuration block on purpose. This becomes a
  `per-file-ignores` entry, not a fix.
- **`UP045` (49 hits) is an artifact of the environment drift.** These
  `Optional[X]` annotations came from commit `8174d6f`
  ("fix py3.9 type hints"), written to satisfy the 3.9 interpreter. Pinning
  3.11 lets them revert to PEP 604 `X | None`. The unpinned environment
  created the debt; pinning removes it.

## 4. Design

### 4.1 Agent instruction layer

Create `AGENTS.md` at the repo root from
`~/Documents/GitHub/coding-standards/templates/AGENTS.md.template`, 20–40
lines, and `CLAUDE.md` containing exactly one line: `@AGENTS.md`.

`AGENTS.md` must state what this repo is **not**. `ai-meal-planner` in the
same `1. Study/` folder is also a food-domain FastAPI + ML repo with
notebooks and a frontend; it is a genuine confusion risk for a fresh session.
FoodLens does calibrated image recognition and decision routing. It does not
plan meals.

Sections, per the template:

- Identity, including the `ai-meal-planner` distinction.
- **Standards** — reference the master, import `@docs/0_coding_standards.md`.
  Reference only; never paste.
- **Deltas from the master** — the four in §4.2.
- **Evidence locations** — `docs/3_model_results.md` for metrics,
  `docs/8_runtime_contract.md` for the API surface,
  `docs/9_agent_log.md` for session history.
- **Current state** — champion is ResNet50 FT-V2; A3b ConvNeXt-Tiny is the
  accuracy leader but blocked from promotion pending decision-layer
  recalibration. Timestamped per master §7.
- **Open risks** — the S1–S3 debt this pass deliberately leaves standing.

### 4.2 Docs restructure

**Shape: A.** The repo trains and evaluates models and also ships a product
app, so master §2 requires picking the centre of gravity and declaring it.
Six of eight numbered docs are modelling-oriented and the README leads with
the champion model and its metrics, so Shape A is the centre of gravity. This
is recorded in `0_coding_standards.md`.

Renumbering is permitted here. Master §2 says not to renumber existing repos,
except "repos being substantially reworked anyway" — which this program is.

| Current | Becomes |
| --- | --- |
| `docs/02_coding_standards.md` | `docs/0_coding_standards.md`, cut to deltas only |
| `docs/01_project_instructions.md` | `docs/1_instructions.md` |
| `docs/03_modeling_approach.md` | `docs/2_modeling_approach.md` |
| `docs/04_model_results.md` | `docs/3_model_results.md` |
| `docs/05_next_steps.md` | `docs/4_next_steps.md` |
| `docs/06_foodlens_app_concept.md` | `docs/5_app_concept.md` |
| `docs/07_multi_food_detection_plan.md` | `docs/6_multi_food_detection_plan.md` |
| `docs/08_model_accuracy_improvement_plan.md` | `docs/7_accuracy_improvement_plan.md` |
| `docs/superpowers/foodlens-runtime-contract.md` | `docs/8_runtime_contract.md` (promoted) |
| — | `docs/9_agent_log.md` (new, append-only) |

**Retired into the agent log.** Three hand-written status docs at the root of
`docs/superpowers/` overlap heavily and are already stale:
`foodlens-completion-status.md`, `foodlens-subagent-evidence-bundle.md`,
`foodlens-completion-runbook.md`. Their durable content becomes a single
dated entry in `docs/9_agent_log.md` recording the June 2026 completion
drive: what was done, what was verified, what stayed open. The files are then
deleted.

**Kept untouched.** `docs/superpowers/specs/` and `docs/superpowers/plans/`
— the brainstorming and writing-plans skills write there, and the dated
design records are worth keeping. This includes
`plans/2026-06-11-foodlens-subagent-execution-board.md`, which is a plan and
stays in `plans/`. Also kept: `docs/ops/`, `docs/releases/`,
`docs/ui-snapshots/`.

`docs/README.md` is rewritten as the index for the new numbering.

**Deltas to record in `0_coding_standards.md`.** The file drops from ~160
lines of restated master to only these genuine differences:

1. Shape A is declared despite the repo shipping a product app.
2. `app/` and `kaggle/` are legitimate root directories beyond master §1's
   list, because the product runtime and the Kaggle run records both need a
   home in git.
3. `E402` is intended in Kaggle-mirrored scripts and notebooks; imports
   follow the configuration block by design.
4. Seven notebooks deliberately retain outputs as evidence — three active,
   four archived — which master §4 permits but does not default to.

### 4.3 Cross-reference integrity

Across tracked files there are **65 references** to paths this pass changes:
46 to the numbered docs and 19 to the retired or promoted `superpowers/`
docs. Thirty of those live in files that are rewritten or deleted anyway
(`docs/README.md` becomes the new index; `docs/02_coding_standards.md` is cut
to deltas; the three status docs are deleted), leaving **35 targeted link
edits**:

| File | Numbered-doc refs | `superpowers/` refs |
| --- | ---: | ---: |
| `README.md` | 12 | 6 |
| `docs/05_next_steps.md` → `4_next_steps.md` | 2 | — |
| `docs/superpowers/plans/2026-06-11-foodlens-subagent-execution-board.md` | 2 | 6 |
| `docs/releases/github-release-v1.0.0-foodlens-completion.md` | — | 3 |
| Four notebooks (see below) | 4 | — |

`CHANGELOG.md` lines 6–7 mention `docs/superpowers/` as a directory in prose
rather than as a link. That statement stays true — `specs/` and `plans/`
remain — so it needs no edit.

They fall into three groups, handled differently:

- **Live docs** (`README.md`, `docs/README.md`, the numbered docs) — update
  all links to the new paths.
- **Historical records** (`docs/releases/*`, `CHANGELOG.md`,
  `docs/superpowers/plans/*`) — update the *paths* so links resolve, but do
  not alter any claim, metric, or narrative. These are records; fixing a
  path is maintenance, changing a statement is falsification.
- **Notebooks** — four notebooks reference
  `08_model_accuracy_improvement_plan.md` in markdown cells:
  `notebooks/archive/09_food101_accuracy_phase1_a1_resnet50_ft_v3.ipynb`,
  `notebooks/archive/10_food101_accuracy_phase1_a3_convnext_tiny.ipynb`, and
  their two byte-level mirrors at
  `kaggle/accuracy_phase1/09_food101_accuracy_phase1_a1_resnet50_ft_v3.ipynb`
  and
  `kaggle/accuracy_phase1_a3/10_food101_accuracy_phase1_a3_convnext_tiny.ipynb`.
  Edit the markdown cell only. This repo's own
  §4 states that when only markdown or documentation changes, existing
  notebook outputs may be kept — so no re-execution or output clearing is
  required. The kaggle mirrors must be updated identically to their
  `notebooks/archive/` counterparts.

A link check over all tracked markdown is part of the acceptance criteria in
§5, so a missed reference fails rather than ships.

### 4.4 Environment and packaging

Follow the `ai-meal-planner` precedent rather than inventing conventions.

- `runtime.txt` → `python-3.11.9`.
- `pyproject.toml` with:
  - `requires-python = ">=3.11,<3.13"`.
  - `[tool.ruff]` — `line-length = 100`, `target-version = "py311"`.
  - `[tool.ruff.lint]` — `select = ["E", "F", "I", "B", "UP"]`.
  - `[tool.ruff.lint.per-file-ignores]` — `"kaggle/**" = ["E402", "E501"]`,
    with a comment explaining that Kaggle scripts mirror notebook cell order.
  - `[tool.pytest.ini_options]` — `testpaths = ["tests"]`, `addopts = "-q"`.
- Consolidate the three scattered `app/backend/requirements*.txt` into root
  `requirements.txt`, `requirements-dev.txt`, and `requirements-detector.txt`.
  Add `requirements-lock.txt`, matching the `unsw-ma-hackathon-2026`
  precedent.
- Rebuild `.venv` on 3.11.9. The current one is empty and unusable.
- Revert the 49 `Optional[X]` annotations to PEP 604 `X | None` and clear the
  remaining 38 `app`/`tests`/`scripts` findings, 26 of which are auto-fixable.

### 4.5 CI

`.github/workflows/ci.yml`, triggered on push and pull request to `main`,
with two independent jobs so a frontend failure does not mask a backend one:

- **backend** — checkout, `setup-python` 3.11, install `requirements.txt`
  and `requirements-dev.txt`, then `ruff check .`,
  `python -m compileall app scripts tests`, `pytest -q`.
- **frontend** — checkout, `setup-node`, `npm ci` in `app/frontend`, then
  `npm run typecheck`, `npm run build`, `npm test`.

`requirements-detector.txt` (ultralytics) is not installed in CI. The backend
tests exercise the demo-fallback path and do not require detector weights;
keeping it out holds CI fast and avoids a large download.

### 4.6 Repo hygiene

- **`yolo11n.pt`** — 5.6 MB sitting in the repo root, correctly gitignored
  by `*.pt` but entirely undeclared, so a fresh clone cannot know it is
  needed or where it came from. Document its provenance, version, and
  download step in `docs/8_runtime_contract.md` and `app/backend/README.md`.
- **`.worktrees/foodlens-react-vite-refinement`** — a stale worktree from
  completed June work, in two independent pieces:
  1. A git *registration* pointing at
     `/Users/tuanm.nguyen/Documents/multi-class-food-recognition/.worktrees/...`
     — a different username and a different repo name, left over from a path
     migration. Git reports it `prunable (gitdir file points to non-existent
     location)`, so `git worktree prune` clears it. `git worktree remove`
     would fail on a path that does not exist.
  2. An orphaned *directory* at `.worktrees/foodlens-react-vite-refinement`
     in this repo, which git does not track as a worktree at all (it is
     gitignored) and which must be deleted directly.

  Safe to remove: the `foodlens-react-vite-refinement` branch is already
  merged into `main` and `git log main..foodlens-react-vite-refinement`
  is empty, so no unique commits are lost. The branch itself is then deleted.
- **Notebook outputs** — the seven notebooks that have them keep them. This is
  permitted by §4 as intentionally preserved evidence; the reason is written
  into `0_coding_standards.md` so a future audit reads it as a decision
  rather than an oversight.
- **`.gitignore`** — no change needed. The `app/artifacts/*` plus
  `!app/artifacts/.gitkeep` pattern already matches the negation style master
  §8 prescribes.

## 5. Acceptance Criteria

Each is a command with observable output. None may be claimed without
running it.

1. `ruff check .` exits 0.
2. `pytest -q` passes; the three existing backend test files still pass
   unchanged, proving the packaging move broke no imports.
3. `cd app/frontend && npm run typecheck && npm run build && npm test` all
   pass.
4. `python --version` in the rebuilt `.venv` reports 3.11.9.
5. Every relative markdown link in tracked `.md` files resolves to an
   existing file.
6. `AGENTS.md` and `CLAUDE.md` exist; `CLAUDE.md` is exactly `@AGENTS.md`.
7. `docs/0_coding_standards.md` contains only the four deltas in §4.2 — no
   restatement of the master.
8. CI is green on a pull request.
9. `git status --short` is clean, with no stray artifacts staged.

## 6. Risks

- **The 3.11 rebuild surfaces a dependency conflict.** `torch`,
  `torchvision`, and `ultralytics` are unpinned in the current requirements
  files. Mitigation: pin at lock time and record resolved versions in
  `requirements-lock.txt`. If a conflict blocks the rebuild, the lock file
  documents the working set rather than leaving it implicit.
- **Renumbering breaks a link nobody notices.** Mitigated by acceptance
  criterion 5, which fails the pass rather than shipping a broken index.
- **Consolidating requirements changes what the backend imports at runtime.**
  Mitigated by acceptance criterion 2 — the existing tests must pass
  unmodified.

## 7. Commit Strategy

Master §9 asks for one coherent change per commit. This pass splits into six
(the implementation plan splits step 4 below into a mechanical `ruff --fix`
sweep and a hand-edited follow-up, so the judgement calls stay reviewable):

1. `docs(standards): add AGENTS.md and cut coding standards to deltas`
2. `docs(structure): renumber docs to Shape A and add agent log`
3. `build(python): pin 3.11, add pyproject, consolidate requirements`
4. `style(lint): clear ruff findings and restore PEP 604 annotations`
5. `ci(github): add backend and frontend workflow`

Commits 3 and 4 are separated deliberately: the mechanical lint sweep would
otherwise bury the packaging decisions in a large diff.
