# S0 Standards Alignment & Scaffolding Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Bring this repo into line with the master coding standard at `~/Documents/GitHub/coding-standards/coding_standards.md`, and install the tooling (Python 3.11 pin, ruff, pytest config, CI) that keeps the alignment from drifting again.

**Architecture:** Six independent commits, each with its own verification. (Spec §7 anticipated five; Task 4 splits its mechanical `ruff --fix` sweep from its hand-edited fixes, so the judgement calls stay reviewable instead of drowning in a 49-file diff.) The agent instruction layer comes first because it is what a fresh session reads. Docs renumbering follows, with a link-integrity check that fails loudly rather than shipping a broken index. Packaging and lint are split deliberately so a mechanical 49-file annotation sweep does not bury the dependency decisions. CI lands last, once there is something green for it to prove.

**Tech Stack:** Python 3.11.9 (via `uv`), ruff, pytest, FastAPI, PyTorch, Node 24 / npm 11, Vite, Vitest, GitHub Actions.

**Source spec:** [`docs/superpowers/specs/2026-09-10-standards-alignment-design.md`](../specs/2026-09-10-standards-alignment-design.md)

## Global Constraints

- **Python floor:** `requires-python = ">=3.11,<3.13"`. Pin exactly `3.11.9`.
- **Ruff config:** `line-length = 100`, `target-version = "py311"`, `select = ["E", "F", "I", "B", "UP"]`.
- **Ruff per-file-ignores:** `"kaggle/**" = ["E402", "E501"]` — Kaggle scripts mirror notebook cell order, where imports follow the config block by design. This is intended style, not debt.
- **Doc shape:** Shape A (master §2). Single-digit numbering: `0_`, `1_`, `2_`, …
- **`CLAUDE.md` is exactly one line:** `@AGENTS.md`. No second source of truth.
- **`AGENTS.md` length:** 20–40 lines. It references the master standard; it never pastes it.
- **Historical records get path fixes, never claim edits.** `docs/releases/*`, `CHANGELOG.md`, and `docs/superpowers/plans/*` may have broken links repaired. No metric, claim, or narrative in them may be altered.
- **Notebook edits are markdown-cell-only.** This repo's §4 permits keeping existing outputs when only markdown changes. Never re-execute a notebook in this pass.
- **Commit convention:** `<type>(<scope>): <imperative summary>` (master §9), with material detail in the body.
- **Never claim a command passed without running it and reading its output.**

---

## File Structure

**Created:**

| Path | Responsibility |
| --- | --- |
| `AGENTS.md` | Layer 2 agent instructions: identity, deltas, evidence locations, state, risks |
| `CLAUDE.md` | One line: `@AGENTS.md` |
| `docs/9_agent_log.md` | Append-only session history; absorbs the three retired status docs |
| `pyproject.toml` | Python metadata, ruff config, pytest config |
| `runtime.txt` | `python-3.11.9` |
| `requirements.txt`, `requirements-dev.txt`, `requirements-detector.txt`, `requirements-lock.txt` | Consolidated dependency surface at repo root |
| `.github/workflows/ci.yml` | Backend and frontend jobs |

**Renamed:** eight `docs/0N_*.md` → `docs/N_*.md`, plus `docs/superpowers/foodlens-runtime-contract.md` → `docs/8_runtime_contract.md`.

**Deleted:** `docs/superpowers/foodlens-completion-status.md`, `foodlens-subagent-evidence-bundle.md`, `foodlens-completion-runbook.md`, `app/backend/requirements*.txt`.

**Modified:** `README.md`, `docs/README.md`, `docs/0_coding_standards.md` (cut to deltas), `docs/4_next_steps.md`, `docs/releases/github-release-v1.0.0-foodlens-completion.md`, `docs/superpowers/plans/2026-06-11-foodlens-subagent-execution-board.md`, `app/backend/README.md`, four notebooks, and ~12 Python files for lint.

---

## Task 1: Agent instruction layer

**Files:**
- Create: `AGENTS.md`, `CLAUDE.md`
- Modify: `docs/02_coding_standards.md` → cut to deltas (renamed in Task 2)

**Interfaces:**
- Consumes: nothing.
- Produces: `AGENTS.md` referencing `@docs/0_coding_standards.md`. **Task 2 creates that path.** Until Task 2 lands, the import target does not exist — this is expected and is resolved by Task 2's link check.

- [ ] **Step 1: Read the template**

```bash
cat ~/Documents/GitHub/coding-standards/templates/AGENTS.md.template
```

Expected: the 20–40 line skeleton with Standards / Deltas / Evidence locations / Current state / Open risks sections.

- [ ] **Step 2: Write `AGENTS.md`**

Create `AGENTS.md` at the repo root with exactly this content:

```markdown
# foodlens-calibrated-food-recognition

Calibrated food **image recognition**: Food-101 classifiers whose confidence is
temperature-scaled and routed through a decision layer (auto-accept / suggest /
confirm / review), plus a FastAPI + React workbench for multi-food analysis.

Not to be confused with `1. Study/ai-meal-planner`, which is also a food-domain
FastAPI + ML repo with notebooks and a frontend. That one plans meals from user
profiles. This one recognises dishes in images and decides how much to trust the
prediction. Neither shares code with the other.

## Standards

Follow the master standard at `~/Documents/GitHub/coding-standards/`.
Project-specific rules and deliberate overrides: @docs/0_coding_standards.md

## Deltas from the master

- **Shape A, despite shipping a product app.** Six of the ten numbered docs are
  modelling docs and the README leads with the champion model, so the centre of
  gravity is modelling. `app/` is the product surface, not the centre.
- **`app/` and `kaggle/` are legitimate root directories** beyond master §1's
  list. The product runtime and the Kaggle run records both need a home in git.
- **`E402` is intended in `kaggle/`.** Those scripts mirror notebook cell order,
  where imports follow the configuration block. Enforced as a per-file-ignore.
- **Four archived notebooks deliberately retain outputs** as evidence, which
  master §4 permits but does not default to.

## Evidence locations

- `docs/3_model_results.md` — every metric; any accuracy claim traces to a row here
- `docs/8_runtime_contract.md` — the API surface and runtime status semantics
- `docs/9_agent_log.md` — append-only session history

## Current state

As of 2026-09-10: product champion is **ResNet50 FT-V2** (83.90% is A3b's
number, not the champion's — see `docs/3_model_results.md`). **A3b
ConvNeXt-Tiny is the accuracy leader but is blocked from promotion** pending
decision-layer recalibration. Do not promote it without redoing calibration.

## Open risks

- `app/backend/inference.py` is 883 lines doing artifact loading, detection,
  classification, and response assembly. Decomposition is planned as S2 — don't
  bolt more onto it.
- `kaggle/recalibrate_decision_layer.py` exists as three byte-identical copies,
  and the four training scripts differ by ~74 lines. Fixing this is S1.
- `yolo11n.pt` (5.6 MB, repo root) is gitignored and required at runtime for
  live detection. A fresh clone will not have it — see `app/backend/README.md`.
```

- [ ] **Step 3: Write `CLAUDE.md`**

```bash
printf '@AGENTS.md\n' > CLAUDE.md
```

- [ ] **Step 4: Verify `CLAUDE.md` is exactly one line**

Run: `wc -l CLAUDE.md && cat CLAUDE.md`
Expected: `1 CLAUDE.md` and the single line `@AGENTS.md`.

- [ ] **Step 5: Verify `AGENTS.md` length is within 20–40 content lines**

Run: `grep -cve '^\s*$' AGENTS.md`
Expected: a number between 20 and 40. If above 40, cut prose from the identity paragraph — never from Deltas or Open risks.

- [ ] **Step 6: Cut `docs/02_coding_standards.md` to deltas only**

Replace the entire file with the content below. The current ~160 lines restate the master, which master §13 explicitly forbids ("Never copy this file into a project").

```markdown
# 0. Coding Standards

The baseline for this repo is the master standard at
`~/Documents/GitHub/coding-standards/coding_standards.md`. This file records
**only** where this project deliberately differs. Anything not listed here
follows the master.

## Doc shape: A

Master §2 asks a repo that is both a modelling project and a product to pick a
centre of gravity and declare it. This repo picks **Shape A**. Six of the ten
numbered docs are modelling docs, and the README leads with the champion model
and its metrics. The FastAPI + React app in `app/` is the product surface, not
the centre of gravity.

## Root directories beyond master §1

Master §1 asks for a small root. Two additions are deliberate:

- `app/` — the product runtime (FastAPI backend, React frontend, artifacts).
- `kaggle/` — the Kaggle run records. Each directory is one training run's
  script plus its notebook, kept so a result traces to the code that produced it.

`data/`, `results/`, and `models/` remain untracked, per master §1 and §8.

## `E402` is intended in `kaggle/`

Scripts under `kaggle/` mirror notebook cell order, where imports follow the
configuration block so the config is visible at the top of the run log. This is
the intended Kaggle style, not debt. Enforced as a `per-file-ignores` entry in
`pyproject.toml` rather than being "fixed".

## Four archived notebooks retain outputs

Master §4 permits keeping notebook outputs when they are intentionally preserved
as evidence. These four are, and their outputs must not be cleared:

- `notebooks/05_confidence_decision_layer.ipynb`
- `notebooks/06_food_recognition_demo_inference.ipynb`
- `notebooks/08_detection_to_foodlens_pipeline.ipynb`
- `notebooks/archive/03_modern_backbone_comparison.ipynb`

Every other notebook follows the master rule: clear outputs when code changes.
```

- [ ] **Step 7: Confirm no master restatement leaked in**

Run: `wc -l docs/02_coding_standards.md`
Expected: under 60 lines. The previous version was ~160. If it is still long, the cut did not happen.

- [ ] **Step 8: Commit**

```bash
git add AGENTS.md CLAUDE.md docs/02_coding_standards.md
git commit -m "docs(standards): add AGENTS.md and cut coding standards to deltas

Master §13 requires a layer-2 AGENTS.md plus a one-line CLAUDE.md; this repo
had neither, so every session started with no knowledge of the standard.

docs/02_coding_standards.md was a ~160-line copy of the master, which §13
explicitly forbids. Cut to the four genuine deltas: Shape A declared despite
shipping a product app, app/ and kaggle/ as legitimate root dirs, E402 as
intended Kaggle style, and four notebooks retaining outputs as evidence.

AGENTS.md names ai-meal-planner as a confusion risk — it is also a food-domain
FastAPI + ML repo in 1. Study/ with notebooks and a frontend.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

## Task 2: Renumber docs to Shape A and add the agent log

**Files:**
- Rename: eight `docs/0N_*.md`, plus `docs/superpowers/foodlens-runtime-contract.md` → `docs/8_runtime_contract.md`
- Create: `docs/9_agent_log.md`
- Delete: `docs/superpowers/foodlens-completion-status.md`, `foodlens-subagent-evidence-bundle.md`, `foodlens-completion-runbook.md`
- Modify: `README.md`, `docs/README.md`, `docs/4_next_steps.md`, `docs/releases/github-release-v1.0.0-foodlens-completion.md`, `docs/superpowers/plans/2026-06-11-foodlens-subagent-execution-board.md`, four notebooks

**Interfaces:**
- Consumes: `AGENTS.md` from Task 1, which imports `@docs/0_coding_standards.md`.
- Produces: `docs/0_coding_standards.md` (resolves Task 1's import), `docs/3_model_results.md`, `docs/8_runtime_contract.md`, `docs/9_agent_log.md` — all four are referenced by `AGENTS.md`.

- [ ] **Step 1: Write the link checker first**

This is the test for this task. Create `scripts/check_doc_links.py`:

```python
"""Verify every relative markdown link in tracked .md files resolves.

Run from the repo root. Exits non-zero and lists offenders when a link
points at a file that does not exist.

Fenced code blocks are skipped. Spec and plan documents quote example
markdown inside fences, and those examples describe files that may not
exist yet -- treating them as live links produces false positives.
"""

import re
import subprocess
import sys
from pathlib import Path

LINK_PATTERN = re.compile(r"\[[^\]]*\]\(([^)]+)\)")
FENCE_PATTERN = re.compile(r"^\s*(`{3,}|~{3,})")


def tracked_markdown_files() -> list[Path]:
    """Return every git-tracked .md file in the repo."""
    output = subprocess.run(
        ["git", "ls-files", "*.md"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return [Path(line) for line in output.splitlines() if line]


def strip_code_fences(text: str) -> str:
    """Blank out fenced code blocks, preserving line numbering.

    A fence closes only on a marker at least as long as the one that
    opened it, so a ```bash block nested inside a ````markdown block does
    not close the outer fence.
    """
    lines = text.splitlines()
    kept: list[str] = []
    fence: str | None = None
    for line in lines:
        match = FENCE_PATTERN.match(line)
        if fence is None:
            if match:
                fence = match.group(1)
                kept.append("")
                continue
            kept.append(line)
        else:
            if match and match.group(1)[0] == fence[0] and len(match.group(1)) >= len(fence):
                fence = None
            kept.append("")
    return "\n".join(kept)


def broken_links(path: Path) -> list[str]:
    """Return relative links in `path` that do not resolve to a real file."""
    problems = []
    body = strip_code_fences(path.read_text(encoding="utf-8"))
    for target in LINK_PATTERN.findall(body):
        target = target.split("#", 1)[0].strip()
        if not target or target.startswith(("http://", "https://", "mailto:")):
            continue
        if not (path.parent / target).resolve().exists():
            problems.append(target)
    return problems


def main() -> int:
    failures = 0
    for path in tracked_markdown_files():
        for target in broken_links(path):
            print(f"{path}: broken link -> {target}")
            failures += 1
    if failures:
        print(f"\n{failures} broken link(s).")
        return 1
    print("All relative markdown links resolve.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Run it against the current tree to capture the baseline**

Run: `python3 scripts/check_doc_links.py; echo "exit=$?"`

Expected: **exactly 4 broken links, all in
`docs/superpowers/plans/2026-06-11-foodlens-subagent-execution-board.md`**, and
`exit=1`. These are pre-existing breakage from June, confirmed before this plan
was written:

```
docs/superpowers/plans/2026-06-11-foodlens-subagent-execution-board.md: broken link -> docs/superpowers/foodlens-runtime-contract.md
docs/superpowers/plans/2026-06-11-foodlens-subagent-execution-board.md: broken link -> ../05_next_steps.md
docs/superpowers/plans/2026-06-11-foodlens-subagent-execution-board.md: broken link -> foodlens-completion-runbook.md
docs/superpowers/plans/2026-06-11-foodlens-subagent-execution-board.md: broken link -> foodlens-completion-status.md
```

They are broken at the wrong *relative depth*, not the wrong filename: the file
lives in `docs/superpowers/plans/`, but its links are written as though it sat
in `docs/superpowers/` or at the repo root. Step 10 fixes them.

**If you see any broken link outside that one file, stop** — something else
regressed, and you cannot otherwise tell your own breakage from pre-existing
breakage.

- [ ] **Step 2a: Confirm the checker ignores fenced code blocks**

The spec and plan documents quote example markdown inside fences, referring to
files that do not exist yet. Those must not count as live links.

Run: `python3 scripts/check_doc_links.py | grep -c "2026-09-10-standards-alignment"`
Expected: `0`. If this reports a non-zero count, `strip_code_fences` is not
working — fix it before proceeding, or every later run will drown in false
positives from the plan's own example blocks.

- [ ] **Step 3: Rename the eight numbered docs**

```bash
git mv docs/02_coding_standards.md docs/0_coding_standards.md
git mv docs/01_project_instructions.md docs/1_instructions.md
git mv docs/03_modeling_approach.md docs/2_modeling_approach.md
git mv docs/04_model_results.md docs/3_model_results.md
git mv docs/05_next_steps.md docs/4_next_steps.md
git mv docs/06_foodlens_app_concept.md docs/5_app_concept.md
git mv docs/07_multi_food_detection_plan.md docs/6_multi_food_detection_plan.md
git mv docs/08_model_accuracy_improvement_plan.md docs/7_accuracy_improvement_plan.md
```

- [ ] **Step 4: Promote the runtime contract**

```bash
git mv docs/superpowers/foodlens-runtime-contract.md docs/8_runtime_contract.md
```

Then change its first line from `# FoodLens Runtime Contract` to `# 8. Runtime Contract` so it matches the numbered-doc heading style.

- [ ] **Step 5: Verify the renames are staged as renames, not delete+add**

Run: `git status --short docs/`
Expected: lines beginning with `R ` for all nine moves. A `D `/`A ` pair means history was lost — redo with `git mv`.

- [ ] **Step 6: Write the agent log, absorbing the three retired status docs**

Read all three before writing, so the entry captures what they actually recorded:

```bash
cat docs/superpowers/foodlens-completion-status.md \
    docs/superpowers/foodlens-subagent-evidence-bundle.md \
    docs/superpowers/foodlens-completion-runbook.md
```

Create `docs/9_agent_log.md`:

```markdown
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

**Changed:** added `AGENTS.md` + `CLAUDE.md`; cut `0_coding_standards.md` from a
~160-line master copy to four deltas; renumbered docs to Shape A; promoted the
runtime contract to `8_runtime_contract.md`; started this log; pinned Python
3.11.9; added `pyproject.toml`, consolidated requirements, and CI.

**Measured, not assumed:** ruff reported 191 findings repo-wide, but only 38 (26
auto-fixable) were real debt in `app/`, `tests/`, `scripts/`. Of the remainder,
56 `E402` hits are intended Kaggle style and became a per-file-ignore, and 49
`UP045` hits were `Optional[X]` annotations added by commit `8174d6f` to satisfy
a Python 3.9 interpreter — pinning 3.11 reverted them rather than fixing them.

**Deliberately left open:** S1 (`kaggle/` deduplication), S2 (`inference.py`
decomposition), S3 (frontend consolidation).
```

- [ ] **Step 7: Delete the three retired status docs**

```bash
git rm docs/superpowers/foodlens-completion-status.md \
       docs/superpowers/foodlens-subagent-evidence-bundle.md \
       docs/superpowers/foodlens-completion-runbook.md
```

- [ ] **Step 8: Rewrite `docs/README.md` as the new index**

Replace the table at the top (keep the notebook lists and the accuracy-plan prose below it unchanged) with:

```markdown
| File | Purpose |
| --- | --- |
| [`0_coding_standards.md`](0_coding_standards.md) | deltas from the master standard — this repo's deliberate differences only |
| [`1_instructions.md`](1_instructions.md) | project objective, dataset scope, evaluation contract, and artifact policy |
| [`2_modeling_approach.md`](2_modeling_approach.md) | notebook-by-notebook reasoning flow and modeling decisions |
| [`3_model_results.md`](3_model_results.md) | baseline, refinement, backbone, calibration, and inference results |
| [`4_next_steps.md`](4_next_steps.md) | recommended next work, demo validation, and decision-layer plan |
| [`5_app_concept.md`](5_app_concept.md) | FoodLens product concept, MVP scope, app architecture, and roadmap |
| [`6_multi_food_detection_plan.md`](6_multi_food_detection_plan.md) | detector-plus-classifier plan for multi-food image and video recognition |
| [`7_accuracy_improvement_plan.md`](7_accuracy_improvement_plan.md) | phased plan for improving Food-101 accuracy, calibration, and product-level model quality |
| [`8_runtime_contract.md`](8_runtime_contract.md) | `/runtime/status` and multi-food response field semantics |
| [`9_agent_log.md`](9_agent_log.md) | append-only session history |
```

Also update the first paragraph to mention that the folder follows Shape A numbering per `0_coding_standards.md`.

- [ ] **Step 9: Fix the 12 numbered-doc links and 6 superpowers links in `README.md`**

```bash
sed -i '' \
  -e 's|docs/01_project_instructions\.md|docs/1_instructions.md|g' \
  -e 's|docs/02_coding_standards\.md|docs/0_coding_standards.md|g' \
  -e 's|docs/03_modeling_approach\.md|docs/2_modeling_approach.md|g' \
  -e 's|docs/04_model_results\.md|docs/3_model_results.md|g' \
  -e 's|docs/05_next_steps\.md|docs/4_next_steps.md|g' \
  -e 's|docs/06_foodlens_app_concept\.md|docs/5_app_concept.md|g' \
  -e 's|docs/07_multi_food_detection_plan\.md|docs/6_multi_food_detection_plan.md|g' \
  -e 's|docs/08_model_accuracy_improvement_plan\.md|docs/7_accuracy_improvement_plan.md|g' \
  -e 's|docs/superpowers/foodlens-runtime-contract\.md|docs/8_runtime_contract.md|g' \
  README.md
```

Then hand-edit the "Multi-Agent Completion Drive" block at `README.md:82-85`. The evidence-bundle and runbook lines now point at deleted files; replace all four bullets with:

```markdown
- Board: [docs/superpowers/plans/2026-06-11-foodlens-subagent-execution-board.md](docs/superpowers/plans/2026-06-11-foodlens-subagent-execution-board.md)
- Session history: [docs/9_agent_log.md](docs/9_agent_log.md)
- Runtime contract: [docs/8_runtime_contract.md](docs/8_runtime_contract.md)
```

- [ ] **Step 10: Fix links in the three remaining markdown files**

`docs/4_next_steps.md` has 2 refs to `08_model_accuracy_improvement_plan.md`:

```bash
sed -i '' 's|08_model_accuracy_improvement_plan\.md|7_accuracy_improvement_plan.md|g' docs/4_next_steps.md
```

`docs/superpowers/plans/2026-06-11-foodlens-subagent-execution-board.md` (2 numbered + 6 superpowers refs) and `docs/releases/github-release-v1.0.0-foodlens-completion.md` (3 superpowers refs) are **historical records**. Fix only the paths — do not touch any claim, metric, or narrative.

First repair the four **pre-existing** broken links in the execution board that
Step 2 captured. These are wrong at the relative-depth level, so a filename
substitution alone will not fix them. The file sits in `docs/superpowers/plans/`,
so `../../` reaches the repo root and `../../docs/` reaches the docs folder.
Apply these four exact replacements to that file:

| Line | Current link target | Correct target |
| ---: | --- | --- |
| 34 | `(docs/superpowers/foodlens-runtime-contract.md)` | `(../../8_runtime_contract.md)` |
| 76 | `(../05_next_steps.md)` | `(../../4_next_steps.md)` |
| 77 | `(foodlens-completion-runbook.md)` | `(../../9_agent_log.md)` |
| 103 | `(foodlens-completion-status.md)` | `(../../9_agent_log.md)` |

Update each link's visible text to match its new target as well — for example
line 76's `[docs/05_next_steps.md](../05_next_steps.md)` becomes
`[docs/4_next_steps.md](../../4_next_steps.md)`. Line 75's
`[README.md](../../README.md)` is already correct; leave it alone.

Then apply the filename substitutions to both historical files:

```bash
sed -i '' \
  -e 's|docs/05_next_steps\.md|docs/4_next_steps.md|g' \
  -e 's|05_next_steps\.md|4_next_steps.md|g' \
  -e 's|docs/superpowers/foodlens-runtime-contract\.md|docs/8_runtime_contract.md|g' \
  -e 's|docs/superpowers/foodlens-completion-status\.md|docs/9_agent_log.md|g' \
  -e 's|docs/superpowers/foodlens-subagent-evidence-bundle\.md|docs/9_agent_log.md|g' \
  -e 's|docs/superpowers/foodlens-completion-runbook\.md|docs/9_agent_log.md|g' \
  docs/superpowers/plans/2026-06-11-foodlens-subagent-execution-board.md \
  docs/releases/github-release-v1.0.0-foodlens-completion.md
```

- [ ] **Step 11: Fix the four notebook markdown cells**

Each contains one reference to `08_model_accuracy_improvement_plan.md` in a markdown cell. Editing a markdown cell does not require clearing outputs (this repo's §4).

```bash
sed -i '' 's|08_model_accuracy_improvement_plan\.md|7_accuracy_improvement_plan.md|g' \
  notebooks/archive/09_food101_accuracy_phase1_a1_resnet50_ft_v3.ipynb \
  notebooks/archive/10_food101_accuracy_phase1_a3_convnext_tiny.ipynb \
  kaggle/accuracy_phase1/09_food101_accuracy_phase1_a1_resnet50_ft_v3.ipynb \
  kaggle/accuracy_phase1_a3/10_food101_accuracy_phase1_a3_convnext_tiny.ipynb
```

- [ ] **Step 12: Verify the notebooks are still valid JSON**

```bash
for f in notebooks/archive/09_food101_accuracy_phase1_a1_resnet50_ft_v3.ipynb \
         notebooks/archive/10_food101_accuracy_phase1_a3_convnext_tiny.ipynb \
         kaggle/accuracy_phase1/09_food101_accuracy_phase1_a1_resnet50_ft_v3.ipynb \
         kaggle/accuracy_phase1_a3/10_food101_accuracy_phase1_a3_convnext_tiny.ipynb; do
  python3 -c "import json,sys; json.load(open(sys.argv[1])); print('OK', sys.argv[1])" "$f"
done
```

Expected: four `OK` lines. A `JSONDecodeError` means `sed` corrupted the file — restore with `git checkout -- <file>` and edit the cell with a JSON-aware tool instead.

- [ ] **Step 13: Confirm the two kaggle mirrors still match their archive counterparts**

```bash
diff notebooks/archive/09_food101_accuracy_phase1_a1_resnet50_ft_v3.ipynb \
     kaggle/accuracy_phase1/09_food101_accuracy_phase1_a1_resnet50_ft_v3.ipynb && echo "09 mirrors match"
diff notebooks/archive/10_food101_accuracy_phase1_a3_convnext_tiny.ipynb \
     kaggle/accuracy_phase1_a3/10_food101_accuracy_phase1_a3_convnext_tiny.ipynb && echo "10 mirrors match"
```

Expected: both "mirrors match" lines. If they differed *before* this task, note it and move on — do not reconcile unrelated drift here.

- [ ] **Step 14: Run the link checker — this is the gate**

Run: `python3 scripts/check_doc_links.py; echo "exit=$?"`
Expected: `All relative markdown links resolve.` and `exit=0`.

If it fails, fix each reported link. Do not proceed with a non-zero exit.

- [ ] **Step 15: Confirm no stale references survive anywhere**

```bash
git grep -n "0[1-8]_[a-z_]*\.md\|foodlens-runtime-contract\|foodlens-completion-status\|foodlens-completion-runbook\|foodlens-subagent-evidence-bundle" -- '*.md' '*.ipynb' || echo "CLEAN"
```

Expected: `CLEAN`. The only acceptable non-clean result is `CHANGELOG.md` lines 6–7, which mention `docs/superpowers/` as a directory in prose (still true — `specs/` and `plans/` remain) and reference the docs by name rather than by link. Leave those alone.

- [ ] **Step 16: Commit**

```bash
git add -A
git status --short
git commit -m "docs(structure): renumber docs to Shape A and add agent log

Master §2 single-digit numbering. Shape A chosen because six of the ten numbered
docs are modelling docs and the README leads with the champion model; the
FastAPI + React app is the product surface, not the centre of gravity.
Renumbering an existing repo is permitted here under §2's 'repos being
substantially reworked anyway' clause.

Promotes the runtime contract out of docs/superpowers/ to 8_runtime_contract.md
— it is a durable API contract, not a session artifact.

Collapses three overlapping, stale, rewritable status docs into a single
append-only 9_agent_log.md entry per §13, including the caveat that the June
DONE markers predate any lint or CI gate.

Fixes 35 cross-references. Historical records (releases/, the execution board)
had paths repaired but no claim altered. Four notebooks had markdown-cell-only
edits, so outputs were legitimately kept per this repo's §4.

Adds scripts/check_doc_links.py, which gates this and every future doc move.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

## Task 3: Pin Python 3.11, add pyproject, consolidate requirements

**Files:**
- Create: `pyproject.toml`, `runtime.txt`, `requirements.txt`, `requirements-dev.txt`, `requirements-detector.txt`, `requirements-lock.txt`
- Delete: `app/backend/requirements.txt`, `app/backend/requirements-dev.txt`, `app/backend/requirements-detector.txt`
- Modify: `app/backend/README.md`, `docs/8_runtime_contract.md`

**Interfaces:**
- Consumes: `docs/8_runtime_contract.md` from Task 2 (Step 10 appends to it) and `scripts/check_doc_links.py` from Task 2 (Step 12 runs it).
- Produces: `pyproject.toml` carrying the ruff and pytest config that Task 4 relies on, and root-level requirements files that Task 5's CI installs by path.

- [ ] **Step 1: Install the pinned interpreter**

```bash
uv python install 3.11.9
uv python list | grep 3.11.9
```

Expected: a line showing `cpython-3.11.9-macos-aarch64-none` with a local path, not `<download available>`.

- [ ] **Step 2: Write `runtime.txt`**

```bash
printf 'python-3.11.9\n' > runtime.txt
```

- [ ] **Step 3: Write `pyproject.toml`**

```toml
[project]
name = "foodlens-calibrated-food-recognition"
version = "1.0.0"
description = "Calibrated Food-101 recognition with a confidence decision layer and a multi-food FastAPI + React workbench."
requires-python = ">=3.11,<3.13"
dependencies = [
  "fastapi",
  "Pillow",
  "python-multipart",
  "torch",
  "torchvision",
  "uvicorn",
  "yt-dlp",
]

[project.optional-dependencies]
detector = ["ultralytics"]
dev = ["httpx", "pytest", "ruff"]

[tool.ruff]
line-length = 100
target-version = "py311"

[tool.ruff.lint]
select = ["E", "F", "I", "B", "UP"]

[tool.ruff.lint.per-file-ignores]
# Kaggle scripts and notebooks mirror notebook cell order: imports follow the
# configuration block so the config is visible at the top of the run log. This
# is the intended style for that directory, not debt. See docs/0_coding_standards.md.
"kaggle/**" = ["E402", "E501"]

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
python_functions = ["test_*"]
addopts = "-q"
```

- [ ] **Step 4: Create the root requirements files**

The current three files under `app/backend/` use `-r` chaining. Preserve that shape at the root.

`requirements.txt`:

```
fastapi
python-multipart
uvicorn
Pillow
torch
torchvision
yt-dlp
```

`requirements-dev.txt`:

```
-r requirements.txt
pytest
httpx
ruff
```

`requirements-detector.txt`:

```
-r requirements.txt
ultralytics
```

- [ ] **Step 5: Delete the old backend requirements files**

```bash
git rm app/backend/requirements.txt app/backend/requirements-dev.txt app/backend/requirements-detector.txt
```

- [ ] **Step 6: Rebuild the virtualenv on 3.11.9**

```bash
rm -rf .venv
uv venv --python 3.11.9
.venv/bin/python --version
```

Expected: `Python 3.11.9`.

- [ ] **Step 7: Install dev dependencies**

```bash
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements-dev.txt
```

Expected: completes without a resolution error. **If torch/torchvision conflict**, pin the pair to the latest versions that resolve together and record the pins in `requirements.txt` — that is what the lock file in Step 8 exists to capture.

- [ ] **Step 8: Freeze the lock file**

```bash
.venv/bin/python -m pip freeze > requirements-lock.txt
head -5 requirements-lock.txt
wc -l requirements-lock.txt
```

Expected: a non-empty file listing pinned `==` versions.

- [ ] **Step 9: Verify the existing tests still pass — this proves the move broke no imports**

Run: `.venv/bin/python -m pytest -q`
Expected: all tests in `tests/backend/` pass. This is the acceptance gate for the packaging change: the three existing test files must pass **unmodified**.

- [ ] **Step 10: Document the `yolo11n.pt` dependency in `docs/8_runtime_contract.md`**

The contract already describes `weights_path`, `weights_found`, and
`weights_source` under `/runtime/status` → `detector`, but never says what the
file is or how to obtain it. Append this section to `docs/8_runtime_contract.md`:

```markdown
## Detector weights

Live detection needs `yolo11n.pt` — the YOLO11-nano checkpoint from
Ultralytics, 5.6 MB. It is gitignored by the `*.pt` rule, so a fresh clone
does not have it, and `weights_found` reports `false` until it is present.

Resolution order, reflected in `weights_source`:

1. `"environment"` — the `FOODLENS_DETECTOR_WEIGHTS` environment variable.
2. `"auto_discovered"` — `yolo11n.pt` found at the repo root.
3. `"ultralytics_default"` — the package default path. Ultralytics downloads
   the checkpoint on first use when it resolves here.

With no weights and no `ultralytics` install, `/predict/multi-food/*` serves
the deterministic demo fallback (`detector_status: "fallback_demo"`). That is
the expected state in CI, which does not install the detector extra.
```

- [ ] **Step 11: Update `app/backend/README.md` install paths**

The three `pip install -r app/backend/requirements*.txt` commands now point at deleted files. Replace that block with:

````markdown
Install runtime dependencies:

```bash
pip install -r requirements.txt
```

Install development dependencies before running tests:

```bash
pip install -r requirements-dev.txt
```

Install the optional detector dependency for live multi-food analysis:

```bash
pip install -r requirements-detector.txt
```

Live detection also needs the YOLO weights file `yolo11n.pt` in the repo root.
It is gitignored (5.6 MB) so a fresh clone will not have it. Ultralytics
downloads it automatically on first use, or set `FOODLENS_DETECTOR_WEIGHTS` to
an existing path. Without it the backend serves the deterministic demo
fallback, which is the expected behaviour in CI.
````

- [ ] **Step 12: Run the link checker again**

Run: `python3 scripts/check_doc_links.py; echo "exit=$?"`
Expected: `exit=0`.

- [ ] **Step 13: Commit**

```bash
git add -A
git status --short
git commit -m "build(python): pin 3.11.9, add pyproject, consolidate requirements

README claimed 3.11 while the local interpreter was 3.9, and commit 8174d6f had
added Optional[X] annotations to satisfy it. Pinning 3.11.9 removes the cause;
Task 4 removes the symptom.

pyproject.toml follows the ai-meal-planner precedent: ruff at 100 cols targeting
py311 with E/F/I/B/UP, plus pytest config. kaggle/** carries a per-file-ignore
for E402 and E501 because those scripts mirror notebook cell order by design.

Three scattered app/backend/requirements*.txt files consolidate to the repo
root, with requirements-lock.txt capturing resolved versions per the
unsw-ma-hackathon-2026 precedent.

Verified: .venv reports 3.11.9 and the three existing backend test files pass
unmodified, proving the move broke no imports.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

## Task 4: Clear lint findings and restore PEP 604 annotations

**Files:**
- Modify: `app/backend/*.py`, `tests/backend/*.py`, `scripts/*.py`, and `kaggle/**/*.py` (annotations only)

**Interfaces:**
- Consumes: `pyproject.toml` ruff config from Task 3. All ruff invocations below rely on it — **do not pass `--select` or `--line-length` on the command line**, or you are testing a different config than CI will.
- Produces: a clean `ruff check .`, which Task 5's CI enforces.

- [ ] **Step 1: Capture the baseline from the real config**

Run: `.venv/bin/python -m ruff check . --statistics`
Expected: roughly 135 findings — lower than the raw 191, because `pyproject.toml`'s `per-file-ignores` now suppresses the 56 intended `E402` hits and the `kaggle/**` `E501` hits. Record the exact number before changing anything.

- [ ] **Step 2: Apply the safe automatic fixes**

```bash
.venv/bin/python -m ruff check . --fix
.venv/bin/python -m ruff check . --statistics
```

Expected: `I001` (unsorted imports), `F401` (unused import), `UP035` (deprecated import), and most `UP045` findings disappear. `E501`, `B905`, and `B008` remain — they need judgement.

- [ ] **Step 3: Verify tests still pass after the automatic fixes**

Run: `.venv/bin/python -m pytest -q`
Expected: all pass. Import reordering can break code that depends on import side effects — this catches it immediately.

- [ ] **Step 4: Commit the automatic fixes separately**

Keeping the mechanical sweep in its own commit makes the hand-edited fixes in the next steps reviewable.

```bash
git add -A
git commit -m "style(lint): apply ruff automatic fixes

Import sorting, one unused import, and PEP 604 annotation rewrites. The 49
Optional[X] annotations came from commit 8174d6f to satisfy Python 3.9; now
that 3.11.9 is pinned they revert to X | None. Mechanical only — no behaviour
change. Tests pass unchanged.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

- [ ] **Step 5: Fix remaining `B905` — `zip()` without `strict=`**

Run `.venv/bin/python -m ruff check . --select B905` to list them. Each is a calibration-bin loop of the form:

```python
for lower, upper in zip(bin_boundaries[:-1], bin_boundaries[1:]):
```

These pair adjacent elements of one array, so the lengths are guaranteed equal. Add `strict=True` to make that guarantee explicit and turn any future mismatch into an error rather than silent truncation:

```python
for lower, upper in zip(bin_boundaries[:-1], bin_boundaries[1:], strict=True):
```

- [ ] **Step 6: Fix remaining `B008` — function call in default argument**

Run `.venv/bin/python -m ruff check . --select B008`. All three are in `app/backend/api.py` and take the form:

```python
async def predict_image(file: UploadFile = File(...)) -> PredictionResponse:
```

**This is correct FastAPI usage** — `File(...)`, `Form(...)`, and `Depends(...)` in defaults are how FastAPI declares parameters. Do not "fix" them. Add a per-file-ignore to `pyproject.toml` instead:

```toml
[tool.ruff.lint.per-file-ignores]
# Kaggle scripts and notebooks mirror notebook cell order: imports follow the
# configuration block so the config is visible at the top of the run log. This
# is the intended style for that directory, not debt. See docs/0_coding_standards.md.
"kaggle/**" = ["E402", "E501"]
# FastAPI declares request parameters as callables in argument defaults
# (File(...), Form(...), Depends(...)). B008 flags the pattern the framework requires.
"app/backend/api.py" = ["B008"]
```

- [ ] **Step 7: Fix remaining `E501` — lines over 100 characters**

Run `.venv/bin/python -m ruff check . --select E501` to list them (8 in `app/`, `tests/`, `scripts/`; `kaggle/**` is already ignored). Wrap each by hand. For a long function signature:

```python
def build_multi_food_classifier_fallback_response(
    regions: list[dict[str, Any]],
    fallback_reason: str,
) -> MultiFoodPredictionResponse:
```

For a long string, use implicit concatenation rather than a backslash:

```python
message = (
    "Detector proposals are available but classifier artifacts are missing; "
    "returning detector-only regions."
)
```

Never reformat surrounding code while fixing a line-length finding — keep the diff to the offending lines.

- [ ] **Step 8: Verify ruff is clean**

Run: `.venv/bin/python -m ruff check .`
Expected: `All checks passed!`

- [ ] **Step 9: Verify tests still pass**

Run: `.venv/bin/python -m pytest -q`
Expected: all pass.

- [ ] **Step 10: Verify everything still compiles**

Run: `.venv/bin/python -m compileall -q app scripts tests && echo "compile OK"`
Expected: `compile OK`.

- [ ] **Step 11: Commit**

```bash
git add -A
git status --short
git commit -m "style(lint): clear remaining ruff findings by hand

zip() calls over adjacent slices of one array take strict=True — the lengths
are guaranteed equal, and making that explicit turns a future mismatch into an
error instead of silent truncation.

B008 in app/backend/api.py is not a defect: File(...) in an argument default is
how FastAPI declares request parameters. Added a per-file-ignore with the
reason, rather than breaking the framework contract.

Eight over-length lines wrapped without reformatting surrounding code.

Verified: ruff check . passes, pytest -q passes, compileall clean.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

## Task 5: Add CI and finish repo hygiene

**Files:**
- Create: `.github/workflows/ci.yml`
- Modify: nothing else in git; the worktree cleanup touches untracked state

**Interfaces:**
- Consumes: `requirements.txt` / `requirements-dev.txt` from Task 3; a clean `ruff check .` from Task 4; `scripts/check_doc_links.py` from Task 2.
- Produces: the green CI run that is acceptance criterion 8.

- [ ] **Step 1: Write the workflow**

Create `.github/workflows/ci.yml`:

```yaml
name: CI

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  backend:
    runs-on: ubuntu-latest
    steps:
      - name: Check out repository
        uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.11"
          cache: pip

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements-dev.txt

      - name: Lint
        run: ruff check .

      - name: Compile
        run: python -m compileall -q app scripts tests

      - name: Check documentation links
        run: python scripts/check_doc_links.py

      - name: Run tests
        run: pytest -q

  frontend:
    runs-on: ubuntu-latest
    defaults:
      run:
        working-directory: app/frontend
    steps:
      - name: Check out repository
        uses: actions/checkout@v4

      - name: Set up Node
        uses: actions/setup-node@v4
        with:
          node-version: "24"
          cache: npm
          cache-dependency-path: app/frontend/package-lock.json

      - name: Install dependencies
        run: npm ci

      - name: Typecheck
        run: npm run typecheck

      - name: Build
        run: npm run build

      - name: Test
        run: npm test
```

Note: `requirements-detector.txt` (ultralytics) is deliberately not installed. The backend tests exercise the demo-fallback path, which needs no detector weights; installing it would add a large download to every run.

- [ ] **Step 2: Verify the backend job's commands locally, in order**

```bash
.venv/bin/python -m ruff check . && \
.venv/bin/python -m compileall -q app scripts tests && \
.venv/bin/python scripts/check_doc_links.py && \
.venv/bin/python -m pytest -q && echo "BACKEND JOB WOULD PASS"
```

Expected: `BACKEND JOB WOULD PASS`.

- [ ] **Step 3: Verify the frontend job's commands locally, in order**

```bash
cd app/frontend && npm ci && npm run typecheck && npm run build && npm test
cd ../..
```

Expected: all four succeed. **If `npm ci` fails** because `package-lock.json` is out of sync with `package.json`, run `npm install`, commit the updated lock file, and note it in the commit body — CI cannot use `npm ci` without a valid lock.

- [ ] **Step 4: Prune the stale worktree registration**

The registration points at `/Users/tuanm.nguyen/Documents/multi-class-food-recognition/.worktrees/...` — a different username and repo name, left from a path migration. Git marks it `prunable`.

```bash
git worktree list
git worktree prune
git worktree list
```

Expected: after pruning, only the main worktree is listed. `git worktree remove` would fail here, because the registered path does not exist.

- [ ] **Step 5: Confirm the branch is fully merged before deleting anything**

```bash
git log --oneline main..foodlens-react-vite-refinement
git branch --merged main | grep foodlens-react-vite-refinement
```

Expected: the first command prints **nothing** (no unique commits), the second prints the branch name (it is merged). **If the first command prints any commit, stop** — there is unmerged work, and deletion would lose it.

- [ ] **Step 6: Delete the orphaned directory and the merged branch**

```bash
rm -rf .worktrees/foodlens-react-vite-refinement
rmdir .worktrees 2>/dev/null || true
git branch -d foodlens-react-vite-refinement
```

`git branch -d` (lowercase) refuses to delete an unmerged branch, which is the safety net. Do not use `-D`.

- [ ] **Step 7: Confirm the working tree is clean**

```bash
git status --short
```

Expected: only `.github/workflows/ci.yml` as untracked (plus `package-lock.json` if Step 3 required it). `.worktrees/` was gitignored, so its removal produces no diff.

- [ ] **Step 8: Commit**

```bash
git add .github/workflows/ci.yml
git status --short
git commit -m "ci(github): add backend and frontend workflow

Two independent jobs so a frontend failure does not mask a backend one.

Backend: Python 3.11, ruff, compileall, doc-link check, pytest. Frontend:
Node 24, npm ci, typecheck, build, vitest.

requirements-detector.txt (ultralytics) is deliberately not installed — the
backend tests exercise the demo-fallback path and need no detector weights,
and installing it would add a large download to every run.

Also prunes a stale worktree registration left from a path migration (it
pointed at a different username and repo name) and deletes the merged
foodlens-react-vite-refinement branch after confirming it had no unique
commits.

Verified: both job command sequences run green locally before pushing.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

- [ ] **Step 9: Push on a branch and confirm CI goes green**

CI cannot be verified locally. Push to a branch and open a PR so both jobs actually run.

All six commits already sit on `chore/s0-standards-alignment`, created before Task 1. Do **not** create a branch here — just push it.

```bash
git branch --show-current   # must print chore/s0-standards-alignment
git push -u origin chore/s0-standards-alignment
gh pr create --title "S0: standards alignment and scaffolding" \
  --body "Implements docs/superpowers/specs/2026-09-10-standards-alignment-design.md

Aligns the repo with the master coding standard: AGENTS.md/CLAUDE.md agent
layer, Shape A doc renumbering with an append-only agent log, Python 3.11.9
pin with pyproject.toml and consolidated requirements, ruff clean, and CI
covering both stacks.

S1 (kaggle/ dedup), S2 (inference.py decomposition) and S3 (frontend
consolidation) are explicit non-goals with their own cycles.

🤖 Generated with [Claude Code](https://claude.com/claude-code)"
gh pr checks --watch
```

Expected: both `backend` and `frontend` report success. **Do not report this task complete until `gh pr checks` shows green** — a workflow file that has never run is not verified.

---

## Final Verification

Run every acceptance criterion from spec §5 and record the actual output. None may be claimed without running it.

- [ ] `.venv/bin/python -m ruff check .` → `All checks passed!`
- [ ] `.venv/bin/python -m pytest -q` → all pass, three backend test files unmodified
- [ ] `cd app/frontend && npm run typecheck && npm run build && npm test` → all pass
- [ ] `.venv/bin/python --version` → `Python 3.11.9`
- [ ] `python3 scripts/check_doc_links.py` → exit 0
- [ ] `wc -l CLAUDE.md` → `1`, containing `@AGENTS.md`
- [ ] `wc -l docs/0_coding_standards.md` → under 60 lines, four deltas only
- [ ] `gh pr checks` → both jobs green
- [ ] `git status --short` → clean

Then use `superpowers:verification-before-completion` before claiming the task done, and `superpowers:finishing-a-development-branch` to decide how the branch integrates.
