# A3b Re-score

## Why this directory exists

A3b (`kaggle/accuracy_phase1_a3b/`) is the accuracy-phase leader -- 83.90%
test top-1, 95.78% test top-5 -- but it cannot be promoted yet because the
decision layer (`scripts/recalibrate_decision_layer.py`) must be recalibrated
against it first, and recalibration refuses A3b's predictions:

```
missing columns: top_5_confidence
columns found: path, actual, predicted, confidence, is_correct, top_5
```

A3b's `test_predictions.csv` predates the predictions-CSV contract documented
in `docs/4_next_steps.md`: the accuracy-phase training script recorded
`top_5` as pipe-separated class *labels* only, never the per-class
confidences the top1-top2 margin needs. Those confidences cannot be
recovered from the existing CSV, so the test set has to be re-scored from the
trained checkpoint.

This lives in its own directory rather than as an edit to
`kaggle/accuracy_phase1_a3b/` because `docs/0_coding_standards.md` records
that directory as an immutable run record ("Do not edit a run record to
satisfy a linter or a refactor. New experiments get a new directory, never an
edit to an existing one."). `rescore_predictions.py` trains nothing -- it
loads the existing `convnext_tiny_continued_best.pth` checkpoint, reconstructs
the exact A3b model and eval preprocessing, and re-scores. It writes a new
`test_predictions_rescored.csv` and never touches the original
`test_predictions.csv`.

## Files

- `rescore_predictions.py` -- the re-scoring script. See its module docstring
  for full behavior, including the self-check it runs against the run's
  recorded accuracy.
- `kernel-metadata.json` -- Kaggle kernel metadata for `tuannm3823`. Mounts
  the A3b run's own kernel (`foodlens-a3b-convnext-tiny-continued`) as a
  kernel source so its checkpoint and manifests are available under
  `/kaggle/input/` on Kaggle.

## Running locally

Food-101 (~5GB) is not in this repo. Point `--data-dir` at wherever you have
it extracted (a directory of `<class_name>/<id>.jpg` files), or export
`FOODLENS_DATA_DIR` once instead of passing the flag every time:

```bash
.venv/bin/python kaggle/a3b_rescore/rescore_predictions.py \
  --results-dir results/accuracy_phase1/a3b_convnext_tiny_continued_224 \
  --data-dir /path/to/food-101 \
  --split test
```

This writes
`results/accuracy_phase1/a3b_convnext_tiny_continued_224/test_predictions_rescored.csv`
and prints the achieved top-1/top-5 accuracy plus a self-check against
`test_metrics.csv` from the original run. A self-check mismatch (more than
0.05 percentage points off) means the preprocessing or class ordering is
wrong -- the script exits non-zero rather than writing confidences that
belong to a different model than the one A3b's champion-comparison numbers
describe.

Run `--help` for the full flag list (`--checkpoint`, `--batch-size`,
`--num-workers`, `--device`, `--output`).

## Pushing to Kaggle

```bash
source .venv/bin/activate
KAGGLE_CONFIG_DIR=/path/to/your/kaggle-credentials kaggle kernels push -p kaggle/a3b_rescore
```

See `kaggle/accuracy_phase1_a4/README.md` for how to set up a durable
`KAGGLE_CONFIG_DIR` and why it must be set explicitly on this machine.

## What consumes the output

`test_predictions_rescored.csv` is written in the schema
`scripts/recalibrate_decision_layer.py` requires (see "Required
`*_predictions.csv` schema" in `docs/4_next_steps.md`). Once it exists,
recalibrate against it directly:

```bash
python3 scripts/recalibrate_decision_layer.py \
  --results-dir results/accuracy_phase1/a3b_convnext_tiny_continued_224 \
  --split test
```

(`recalibrate_decision_layer.py` reads `<split>_predictions.csv` by
convention; point it at the rescored file by copying/symlinking it to that
name, or by adjusting `--results-dir`/an equivalent override once one exists
-- the immutable-run-record rule above is why this script does not overwrite
the original in place.)
