# Decision-Layer Methodology Repair (Design)

Date: 2026-09-14
Status: Approved — responds to the Codex review of 2026-09-14

## 1. Why

The 2026-09-11 A3b recalibration produced numbers that were reported as a
promotion case. They do not describe production behaviour. The Codex review
found why, and independent verification confirmed every finding.

Until these are fixed, **no A3b promotion decision should be made on the
recalibrated numbers**, and the 2026-09-11 band table should be treated as
withdrawn.

## 2. The findings, verified

### 2.1 Offline routing consumes ground truth (P1)

`scripts/recalibrate_decision_layer.py::assign_decision_band` and
`app/backend/decision.py::build_decision` implement different decision
functions. The offline one uses the true label in three places production
cannot see:

| Rule | Offline | Production |
| --- | --- | --- |
| review | `is_frequent_confusion_pair`, built from the exact `(actual, predicted)` pair | `predicted_label` appearing in *any* confusion pair, **and** `margin < margin_threshold` |
| hard case | `is_hard_case` — true if **actual or** predicted class is hard | `predicted_label in hard_classes` only |
| suggest | `top_1_confidence >= suggest_confidence` **and `top_5_contains_actual`** | `top_1_confidence >= suggest_confidence` alone |

Production also gates `review` on the margin; the offline version does not.

Measured impact on the A3b test run:

- **251 rows (2.49%)** are classed hard by the actual label alone.
- **Suggest's "100.00% top-5 containment" is a tautology.** Membership in the
  band requires `top_5_contains_actual`, so the metric can only ever be 1.0. It
  was reported on 2026-09-11 as evidence of model quality. It is evidence of
  nothing. A metric landing on an exact 100.0000% should have been challenged.

The grid search therefore optimises a policy that cannot be executed at
inference time, and the band metrics describe an oracle-assisted system.

### 2.2 The re-scored artifact is not consumable by the documented command (P1)

`rescore_predictions.py` writes `<split>_predictions_rescored.csv`;
`recalibrate_decision_layer.py` unconditionally reads
`<split>_predictions.csv`. The command in `kaggle/a3b_rescore/README.md`
therefore reads the **old, incompatible** file and fails schema validation.

The 2026-09-11 run only succeeded because a sibling run directory was staged by
hand — a step that exists in no documentation and that the README's own
suggested workaround (copy or symlink over the conventional name) contradicts,
since it conflicts with the immutable-run-record rule.

### 2.3 Thresholds are selected on the test set and reported on it (P1)

`docs/4_next_steps.md` and the re-score README both direct `--split test`.
`run_analysis()` searches the threshold grid, derives confusion pairs and hard
classes, and computes final band metrics **all from that one dataframe**. Test
labels leak into policy selection, so the promotion metrics are optimistic.
This violates the master standard's leakage rule directly.

### 2.4 A failed self-check leaves a complete-looking artifact (P2)

`rescore_predictions.py` writes the CSV *before* comparing achieved against
recorded accuracy. On mismatch it exits non-zero but the file remains, despite
the README claiming it exits rather than writing confidences belonging to the
wrong model.

### 2.5 `resolve_temperature` accepts non-finite and non-positive values

Zero, negative, NaN and infinity are all accepted and divided into the logits.
Not the current A3b failure — its temperature is 0.884 — but a silent corruption
path.

## 3. Design

### 3.1 One routing function, no ground truth

Extract the band decision into `app/backend/decision_rules.py`: a pure,
dependency-free function

```python
def route_decision(
    top_1_confidence: float,
    margin: float,
    predicted_label: str,
    *,
    policy: dict[str, float],
    hard_classes: set[str],
    confusion_pairs: set[tuple[str, str]],
    mode: str = "image",
) -> str
```

returning `"auto_accept" | "suggest" | "confirm" | "review"`. Its parameters are
exactly what an inference-time caller has. There is no actual-label parameter,
which makes the leak structurally impossible rather than merely discouraged.

`app/backend/decision.py::build_decision` calls it and maps the band to its
title and recommended action, preserving today's HTTP responses exactly.
`scripts/recalibrate_decision_layer.py` calls the same function.

**A differential test asserts the two agree** across a grid of synthetic cases
spanning every branch — confidence either side of both thresholds, margin either
side, predicted label hard or not, in a confusion pair or not, video mode. This
test is the guarantee that the offline policy remains executable.

Actual labels are used **only after routing**, to score each band.

### 3.2 Fit and evaluation splits separated

`recalibrate_decision_layer.py` gains `--fit-split` (default `val`) and
`--eval-split` (default `test`). The grid search, hard classes and confusion
pairs are derived from the fit split alone; the selected policy is then frozen
and evaluated **once** on the eval split. Both split metrics are written, clearly
labelled, so the gap between them is visible rather than hidden.

The existing `--split` is kept as a deprecated alias meaning "fit and evaluate
on the same split", and emits a warning naming the leakage. Removing it outright
would silently change the meaning of runbook commands already in circulation.

This requires `val_predictions.csv` in the contract schema, so the val split
must also be re-scored.

### 3.3 Explicit predictions input

`--predictions-file` overrides the `<split>_predictions.csv` convention, so a
re-scored artifact is consumed directly with no renaming, copying or symlinking
and no mutation of a run record. The README command is corrected to use it.

### 3.4 Write only what passed

`rescore_predictions.py` writes to a temporary file, runs the accuracy
self-check, and only then atomically renames it into place. On mismatch no final
artifact exists. A regression test asserts this.

### 3.5 Temperature validation

`resolve_temperature` rejects non-finite and non-positive values with a clear
message, whether they came from `calibration.json` or `--temperature`.

## 4. Acceptance criteria

1. A differential test proves `route_decision` and `build_decision` agree on a
   grid covering every branch.
2. `grep -n "actual" scripts/recalibrate_decision_layer.py` shows no actual-label
   use inside routing — only in post-routing scoring.
3. The full pipeline runs end to end with **no manual staging**: re-score val and
   test, fit on val, evaluate on test, using `--predictions-file`.
4. Fit-split and eval-split metrics are both reported and differ (if identical,
   the split separation is not working).
5. The rescore script refuses to start if its destination output file
   already exists, unless `--overwrite` is passed; a deliberately mismatched
   self-check then leaves the destination exactly as it was before the run
   started (nothing, on a fresh path; the pre-existing file, if `--overwrite`
   was passed into a run that then failed its self-check) -- never a new,
   complete-looking file that belongs to the wrong model.
6. `pytest`, `ruff check .`, and both doc gates pass.
7. `app/backend/api.py` and the three original backend test files unmodified;
   HTTP response shapes unchanged.

## 5. What this does not do

It does not promote A3b, and it does not re-issue a promotion recommendation.
It produces numbers that *could* support one. The comparison against the
champion remains uncontrolled until ResNet50 FT-V2 goes through the identical
pipeline — its published band metrics came from the same flawed offline routing
and are equally oracle-assisted.
