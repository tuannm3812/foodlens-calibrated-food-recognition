# S1 — Kaggle Deduplication (Design)

Date: 2026-09-11
Status: Approved by decomposition; design revised after investigation

## 1. Purpose

Remove genuine duplication from `kaggle/` without damaging the run records or
breaking Kaggle execution.

## 2. What the investigation changed

The S0 decomposition promised: *"4 × ~700-line near-identical scripts → one
parameterised runner + per-run configs; 3 byte-identical copies of
`recalibrate_decision_layer.py` → one module."*

**The first half of that is wrong, and this design drops it.** Two facts found
by inspection contradict it:

1. **Kaggle runs the notebook, self-contained.** Every
   `kaggle/*/kernel-metadata.json` sets `"code_file"` to the `.ipynb` and
   `"kernel_sources": []`. No utility kernel is attached, so a notebook cannot
   `import` a shared local module. Master §4 states the same rule directly:
   "Prefer readable, self-contained notebook code over imports from local
   project modules. Kaggle should be able to run the notebook after attaching
   the required dataset."

2. **The training scripts are run records, not live source.** Each one is the
   code that produced a specific published result — A1 78.28%, A3 83.41%,
   A3b 83.90%, A4. Merging them into a parameterised runner would destroy the
   property the repo is built on: that any claim in `docs/3_model_results.md`
   traces to the code that produced it. A "deduplicated" A3 script is no longer
   the script A3 ran.

So the 46-282 changed lines between run scripts are **not** duplication to be
removed. They are the record of what differed between experiments.

## 3. The duplication that is real

| # | Duplication | Lines | Verdict |
| --- | --- | ---: | --- |
| 1 | `recalibrate_decision_layer.py` in three directories, byte-identical (md5 `17b5133111a33b21d7b62d0a65ba3c89`) | 1,872 total, 1,248 redundant | **Remove.** Not a run record and not a Kaggle `code_file`. |
| 2 | Each `.py` training script duplicates its own `.ipynb` code cells at 97.8-98.1% similarity | ~2,900 | **Keep, but expose.** See §5. |
| 3 | Run scripts differ from each other by 46-282 lines | — | **Not duplication.** Immutable run records. |

### On #1

`recalibrate_decision_layer.py` is not named as `code_file` in any
`kernel-metadata.json`. It is a local CLI:
`scripts/watch_kaggle_kernel_and_recalibrate.py:43` hardcodes the
`accuracy_phase1` copy as its default, and `docs/4_next_steps.md:238` invokes
that same path. The copies under `accuracy_phase1_a3b/` and
`accuracy_phase1_a4/` are referenced by nothing.

Nothing about it is specific to the A1 run, so living under
`kaggle/accuracy_phase1/` is itself misleading. It moves to `scripts/`.

`docs/4_next_steps.md:290` shows it being run *on* Kaggle as
`!python /kaggle/working/recalibrate_decision_layer.py` — that is a manual
upload into the working directory and does not depend on the repo path, so the
move does not break it. The doc reference is updated regardless.

### On #2 — the mirrors were never in sync

Measured similarity between each `.py` and its own notebook's code cells:

| Run | Before S0 (`86f9275`) | After S0 (`HEAD`) |
| --- | ---: | ---: |
| a1 | 0.9862 | 0.9811 |
| a3 | 0.9863 | 0.9798 |
| a3b | 0.9839 | 0.9775 |
| a4 | 0.9918 | 0.9790 |

Two conclusions, and the second is a finding against S0:

- **The drift predates S0.** No `.py` was ever an exact export of its notebook,
  so the mirror never carried a guarantee worth trusting.
- **S0 widened it.** The ruff sweep rewrote `Optional[X]` → `X | None` and
  reordered imports in the `.py` files, while `extend-exclude = ["*.ipynb"]`
  left the notebooks untouched. That was a deliberate S0 decision with an
  unanticipated consequence: it modernised a *record of what ran* so that it no
  longer matches what ran. The notebooks — the actual `code_file` — are
  unchanged, so no published result is invalidated.

This design does **not** reconcile the drift. Regenerating the `.py` from the
notebooks would revert S0's lint fixes and put `kaggle/**` back into permanent
conflict with the lint gate; deleting the mirrors would discard a form that is
far easier to diff and review than notebook JSON. Both are real options, and
both are the user's call, not a change to make while the tradeoff is
undiscussed. Instead the drift is made **visible** (§5) so the decision can be
taken with numbers in front of it.

## 4. Scope

### In scope

1. Collapse the three `recalibrate_decision_layer.py` copies to one at
   `scripts/recalibrate_decision_layer.py`; update every reference.
2. Add `scripts/check_kaggle_mirrors.py`, reporting per-run `.py`/`.ipynb`
   similarity and the differing lines.
3. Record in `docs/0_coding_standards.md` that `kaggle/*/` training scripts and
   notebooks are immutable run records, why they are not deduplicated, and what
   the mirror relationship is.
4. Update `AGENTS.md`'s open-risk bullet to reflect what S1 actually resolved.

### Explicitly not in scope

- Merging run scripts into a parameterised runner — see §2.
- Reconciling or deleting the `.py` mirrors — see §3, this is the user's call.
- Editing any notebook. They are the `code_file` and the record of what ran.
- Re-running any training job.

## 5. The mirror checker

`scripts/check_kaggle_mirrors.py` pairs each `kaggle/*/foodlens_*.py` with the
`.ipynb` in the same directory, extracts the notebook's code cells, and reports
similarity plus a unified diff.

It is **report-only and not wired into CI.** The mirrors are already out of sync,
so a blocking check would fail on arrival and teach everyone to ignore it. Its
job is to make an existing, invisible condition measurable. Exit code is 0 unless
a pairing cannot be resolved at all, which would indicate a genuinely broken
directory rather than drift.

## 6. Acceptance criteria

1. Exactly one `recalibrate_decision_layer.py` exists in the repo, under
   `scripts/`, and it is byte-identical to the three it replaces.
2. `grep -rn "kaggle/.*/recalibrate_decision_layer" --include='*.py' --include='*.md'`
   returns nothing.
3. `scripts/watch_kaggle_kernel_and_recalibrate.py` resolves its default to the
   new path, verified by running it with `--help`.
4. `scripts/check_kaggle_mirrors.py` reports all four runs and exits 0.
5. `ruff check .` clean, `pytest` green, `check_doc_links.py` exits 0.
6. No file under `kaggle/` is modified except the two deleted copies.

## 7. Risks

- **The watcher's default path is the only programmatic consumer.** If its
  resolution is wrong, recalibration silently targets a missing file. Mitigated
  by acceptance criterion 3, which executes it rather than reading it.
- **`docs/4_next_steps.md` documents an operational runbook.** Its commands are
  what the user actually types. Path edits there must be exact; the file's
  narrative must not otherwise change.
