# 10. Decisions

The short record of decisions that shape this repository: what was chosen, what
it ruled out, and where the evidence lives. It is the index; `9_agent_log.md` is
the trail. Read this first, and follow an entry's evidence link when the
reasoning matters.

Rules for this file:

- One entry per decision. Keep each to what a newcomer needs to avoid undoing it.
- A decision is changed by adding a new entry that **supersedes** the old one.
  Mark the old entry `Superseded by D-NNN`; never delete it.
- Open decisions are listed first, with an owner. Resolving one means moving it
  to an accepted entry.
- Every entry links its evidence: a section, an agent-log entry, or a commit.

## Open decisions

| ID | Question | Owner | Context |
| --- | --- | --- | --- |
| O-1 | Promote A3b ConvNeXt-Tiny over ResNet50 FT-V2? | User | A3b's recalibration is complete and accepted as evidence. It leads on accuracy and every decision band except review-band top-5 containment, but its calibration is about half as tight (test ECE 0.0556 vs 0.0265). The trade is accuracy against how literally a shown confidence can be read. Evidence: `3_model_results.md` §16. |
| O-3 | Reconcile, delete, or keep the `.py`/`.ipynb` mirror pairs? | User | They have never been in sync (A1, A3, A3b 97.8-98.1%; A4 90.2% since its record was restored). `scripts/check_kaggle_mirrors.py` measures the drift and names each run's authoritative file. |
| O-4 | Keep or delete `app/frontend-static/`? | User | A deliberately archived prototype (2026-05-31 plan), 1,967 unmaintained lines that git would preserve if deleted. |
| O-5 | Run the Codex GitHub reviewer automatically or on demand? | User | Automatic review on every PR has exhausted the account's quota since 2026-09-10 (PRs #7–#11, including log-only #11). On-demand review (`@codex review`) would spend it where it matters. This is an account setting, not a repo file. |

## Accepted decisions

### D-001 — Doc shape A, despite shipping a product app (2026-09-10)

**Decision.** The repository follows the master standard's Shape A numbering.
**Why.** Most numbered docs are modelling docs and the README leads with the
champion model. **Ruled out.** Shape B. **Evidence.** `0_coding_standards.md`;
`superpowers/specs/2026-09-10-standards-alignment-design.md`.

### D-002 — `kaggle/*/` directories are immutable run records (2026-09-11)

**Decision.** Training scripts stay duplicated; no parameterised runner. New
experiments get a new directory. The record of a run is the file its
`kernel-metadata.json` names as `code_file`. **Why.** Kaggle executes that file
self-contained, and every published figure must trace to the code that produced
it. **Ruled out.** Merging the four training scripts (the original S1 plan).
**Evidence.** `0_coding_standards.md`; `superpowers/specs/2026-09-11-kaggle-dedup-design.md`.
**Amended 2026-10-10:** the earlier wording said the notebook is always the
record; A4 is a `script` kernel (Codex GitHub review on PR #4).

### D-003 — Lint scope (2026-09-10)

**Decision.** Ruff excludes notebooks; `E402` and `E501` are ignored in
`kaggle/`; `B008` is ignored in `app/backend/api.py`. **Why.** Notebook cells
inherently trip import and name rules; Kaggle scripts mirror cell order; FastAPI
declares parameters as defaults. **Ruled out.** Editing run records or notebooks
to satisfy the linter. **Evidence.** `0_coding_standards.md`; `pyproject.toml`.

### D-004 — `requirements-lock.txt` is a local snapshot, not a CI constraint (2026-09-11)

**Decision.** CI installs the unpinned requirement files. **Why.** The lock was
frozen on macOS/arm64; wiring it into Linux CI could resolve different wheels,
and unpinned CI surfaces upstream breakage early. **Ruled out.** Pinning CI to
the lock without separate validation. **Evidence.** `app/backend/README.md`;
agent log, 2026-09-11 response to the Codex S0 review.

### D-005 — Decision-layer methodology (2026-09-14)

**Decision.** Thresholds, hard classes and confusion pairs are fitted on
validation; the frozen policy is scored once on test. Offline recalibration and
production share one `route_decision`, whose signature has no actual-label
parameter. **Why.** The previous method used the true label during routing and
selected thresholds on test, so its band metrics described a system that cannot
run. **Ruled out.** Same-split fitting (kept only as a deprecated, warning
alias). **Evidence.** `superpowers/specs/2026-09-14-decision-layer-methodology-design.md`.

### D-006 — Derived inputs come only from fit predictions, with provenance (2026-10-05)

**Decision.** Hard classes and confusion pairs are derived from the fit-split
predictions unless a file is named explicitly; ties at both cutoffs break by
label order; every run writes `derivation_provenance.json`, including the
recalibration script's hash. **Why.** Three successive defects came from the
script silently changing its inputs depending on which optional files sat in
the run directory. **Ruled out.** Implicit sidecar discovery; refusing to run
when a sidecar exists. **Evidence.** `3_model_results.md` §16; agent log,
2026-09-27 to 2026-10-10.

### D-007 — Withdraw the pre-repair decision-layer figures (2026-10-09)

**Decision.** The 58.02% auto-accept at 96.47% and the 100% suggestion-band
containment are withdrawn everywhere they were presented as current. **Why.**
They came from the leaking method; the 100% figure was true by construction.
**Evidence.** `3_model_results.md` §11 (marked superseded) and §16.

### D-008 — ResNet50 FT-V2 stays champion, with the corrected policy (2026-10-09, user)

**Decision.** Keep ResNet50 FT-V2 as the model and deploy its corrected decision
policy (0.70 / 0.35 / 0.05; 11 hard classes; 40 pairs). **Why.** The legacy
policy, measured correctly, auto-accepted 59.02% at 94.83% and sent 11.52% to
review — not the 58.02% / 96.47% it was credited with. **Ruled out.** Keeping the
legacy runtime; promoting A3b in the same step (see O-1). **Evidence.**
`8_runtime_contract.md`, "Deployed decision policy".

### D-009 — Policies deploy only through `scripts/deploy_decision_policy.py` (2026-10-10)

**Decision.** Deploy with the script, restart every API process, then run
`--verify-live`. **Why.** A verbatim copy of a recalibration output makes every
request fall back to demo answers while status still reports `ready`; a running
process keeps its cached policy until restarted. **Ruled out.** Hand-copying
artifacts; treating a fresh file read as proof the live service changed.
**Evidence.** `8_runtime_contract.md`; agent log, 2026-10-10.

### D-010 — Merging stacked PRs (2026-10-10)

**Decision.** Use merge commits. Before merging, replay the full merge order
locally and run the full suite on the result. After each merge, retarget child
PRs to `main` explicitly, confirm nothing still targets the merged branch, then
delete it. **Why.** Per-PR CI never tests stacks together: a clean-merging stack
failed 24 tests until fixed. Deleting a base branch through the API **closed**
its child PRs instead of retargeting them. **Ruled out.** Squash or rebase on
published stacks; relying on delete-to-retarget. **Evidence.** Agent log,
2026-10-10, "stack merged to main".

### D-011 — Read both review channels (2026-10-10)

**Decision.** Codex reviews arrive in two places: entries in `9_agent_log.md` and
the Codex GitHub bot's reviews on pull requests. Check both before treating a PR
as reviewed. **Why.** The bot reviewed #3–#6 on 2026-09-10 with four findings
that went unread for a month; two were still live on `main` (fixed 2026-10-10).
**Evidence.** Agent log, 2026-10-10, "Codex GitHub bot findings".

### D-012 — Restore A4's run record to what ran (2026-10-10, user)

**Decision.** `kaggle/accuracy_phase1_a4/foodlens_accuracy_phase1_a4.py` — A4's
`code_file` — is restored to commit `151986b`, which matches the source pulled
from the Kaggle kernel byte for byte, and is excluded from ruff. **Why.** The
repo copy had drifted 154 lines from what ran, including edits from S0's lint
sweep, contrary to D-002. **Ruled out.** Keeping the drifted copy as a working
version. The June edits (`45ef2b7`, `e227ef4`) remain in git history; a future A4
re-run belongs in a new directory, per D-002. **Evidence.** Agent log,
2026-10-10, "Codex GitHub bot findings". Resolves O-2.
