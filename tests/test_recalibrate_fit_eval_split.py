"""Regression tests for fit/eval split separation in
scripts/recalibrate_decision_layer.py.

Background: `run_analysis()` used to search the threshold grid, derive hard
classes/confusion pairs, and compute final band metrics all from one
dataframe, with the runbook directing `--split test`. That leaks test labels
into policy selection, so reported band metrics are optimistic (see
docs/superpowers/specs/2026-09-14-decision-layer-methodology-design.md,
section 3.2). `--fit-split` (default `val`) and `--eval-split` (default
`test`) fix this: the grid search, hard classes and confusion pairs come
from the fit split alone; the frozen policy is evaluated exactly once on the
eval split. `--split` remains as a deprecated alias meaning "fit and
evaluate on the same split", emitting a warning.

The module lives in scripts/, not a package, so it is loaded by file path
rather than imported normally (see tests/test_check_doc_links.py).
"""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path

import pandas as pd
import pytest

SCRIPT_PATH = (
    Path(__file__).resolve().parent.parent / "scripts" / "recalibrate_decision_layer.py"
)

_spec = importlib.util.spec_from_file_location("recalibrate_decision_layer", SCRIPT_PATH)
assert _spec is not None and _spec.loader is not None
recalibrate_decision_layer = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(recalibrate_decision_layer)

run_analysis = recalibrate_decision_layer.run_analysis
main = recalibrate_decision_layer.main
parse_args = recalibrate_decision_layer.parse_args
resolve_splits = recalibrate_decision_layer.resolve_splits
ConflictingSplitArgumentsError = recalibrate_decision_layer.ConflictingSplitArgumentsError
DEPRECATED_SPLIT_WARNING = recalibrate_decision_layer.DEPRECATED_SPLIT_WARNING

LABELS = ["miso_soup", "pho", "ramen", "sushi", "tacos"]


def _write_predictions(path: Path, n: int, correct_fraction: float) -> None:
    """Write `n` fake prediction rows, roughly `correct_fraction` correct.

    Confidence is set high (0.9) for correct rows and spread out for
    incorrect ones, so a split that is mostly correct looks meaningfully
    different -- in both accuracy and confidence distribution -- from one
    that is mostly wrong. That is what lets a test assert the fit and eval
    band metrics actually differ when fit and eval draw from different data.
    """
    rows = []
    for i in range(n):
        actual = LABELS[i % len(LABELS)]
        is_correct = (i % 100) < round(correct_fraction * 100)
        predicted = actual if is_correct else LABELS[(i + 1) % len(LABELS)]
        confidences = {label: 0.02 for label in LABELS}
        confidences[predicted] = 0.9 if is_correct else 0.55
        ranked = sorted(LABELS, key=lambda label: -confidences[label])
        top_5 = "|".join(ranked)
        top_5_confidence = "|".join(f"{confidences[label]:.4f}" for label in ranked)
        rows.append(
            {
                "path": f"/data/{actual}/{i}.jpg",
                "actual": actual,
                "predicted": predicted,
                "is_correct": is_correct,
                "top_5": top_5,
                "top_5_confidence": top_5_confidence,
            }
        )
    pd.DataFrame(rows).to_csv(path, index=False)


# --------------------------------------------------------------------------
# Acceptance criterion 2: no actual-label use inside routing.
# --------------------------------------------------------------------------


def test_no_actual_label_use_inside_routing_function() -> None:
    """`assign_decision_band` (the routing call site) must never *access* the
    ground-truth label -- only `route_decision`'s inference-time inputs
    (`row["predicted"]`, `top_1_confidence`, margin). Its docstring is
    allowed to *talk about* the actual label (to explain why it's absent);
    what must never appear is code that reads a `row["actual"]`-shaped
    column. `actual` may still be used elsewhere in the script (feature
    building, post-routing scoring)."""
    source = SCRIPT_PATH.read_text(encoding="utf-8")
    start = source.index("def assign_decision_band")
    end = source.index("\ndef ", start + 1)
    body = source[start:end]
    docstring_end = body.index('"""', body.index('"""') + 3) + 3
    code_only = body[docstring_end:]
    assert "actual" not in code_only, code_only


# --------------------------------------------------------------------------
# run_analysis: fit split alone drives the grid search / hard classes /
# confusion pairs; eval split is scored once with the frozen policy.
# --------------------------------------------------------------------------


def test_fit_and_eval_metrics_differ_when_splits_differ(tmp_path: Path) -> None:
    results_dir = tmp_path / "run"
    results_dir.mkdir()

    val_path = results_dir / "val_predictions.csv"
    test_path = results_dir / "test_predictions.csv"
    # Deliberately very different accuracy profiles between the two splits.
    _write_predictions(val_path, n=200, correct_fraction=0.95)
    _write_predictions(test_path, n=200, correct_fraction=0.40)

    output_dir = tmp_path / "out"
    band_metrics_fit, band_metrics_eval = run_analysis(
        results_dir=results_dir,
        fit_split="val",
        eval_split="test",
        hard_classes_file=None,
        confusion_pairs_file=None,
        class_report_file=None,
        output_dir=output_dir,
        max_confusion_pairs=10,
        skip_zip=True,
    )

    assert (output_dir / "decision_band_metrics_fit.csv").exists()
    assert (output_dir / "decision_band_metrics_eval.csv").exists()
    assert (output_dir / "decision_policy.json").exists()
    # Exactly one frozen policy -- not one per split.
    assert not (output_dir / "decision_band_metrics.csv").exists()

    # The generalisation gap must be visible, not hidden: fit and eval band
    # metrics must not be identical when the underlying data differs this
    # much (acceptance criterion 4).
    assert not band_metrics_fit.equals(band_metrics_eval)

    # The same frozen policy produces very different coverage on data with a
    # different confidence/accuracy profile -- the generalisation gap.
    fit_auto_coverage = band_metrics_fit.set_index("decision_band").loc[
        "auto_accept", "coverage"
    ]
    eval_auto_coverage = band_metrics_eval.set_index("decision_band").loc[
        "auto_accept", "coverage"
    ]
    assert fit_auto_coverage != pytest.approx(eval_auto_coverage)


def test_hard_classes_and_confusion_pairs_derived_from_fit_split_only(tmp_path: Path) -> None:
    """Confusion pairs (and, transitively, which rows are hard/risky) must
    come only from the fit split's own errors -- eval split errors that
    don't occur in the fit split must not show up as confusion pairs."""
    results_dir = tmp_path / "run"
    results_dir.mkdir()

    # Fit split: no errors at all -> no confusion pairs derived.
    val_path = results_dir / "val_predictions.csv"
    _write_predictions(val_path, n=100, correct_fraction=1.0)

    # Eval split: lots of errors -- these must NOT leak into confusion_pairs.
    test_path = results_dir / "test_predictions.csv"
    _write_predictions(test_path, n=100, correct_fraction=0.2)

    output_dir = tmp_path / "out"
    run_analysis(
        results_dir=results_dir,
        fit_split="val",
        eval_split="test",
        hard_classes_file=None,
        confusion_pairs_file=None,
        class_report_file=None,
        output_dir=output_dir,
        max_confusion_pairs=10,
        skip_zip=True,
    )

    # No errors on the fit split -> no confusion pairs file (matches the
    # existing "only written if non-empty" behavior).
    assert not (output_dir / "top_confusion_pairs.csv").exists()
    import json

    confusion_pairs = json.loads((output_dir / "confusion_pairs.json").read_text())
    assert confusion_pairs == []


def test_default_fit_and_eval_splits_are_val_and_test() -> None:
    args = parse_args(["--results-dir", "/tmp/does-not-matter"])
    fit_split, eval_split, fit_pf, eval_pf, deprecated = resolve_splits(args)

    assert fit_split == "val"
    assert eval_split == "test"
    assert fit_pf is None
    assert eval_pf is None
    assert deprecated is False


# --------------------------------------------------------------------------
# Deprecated --split alias
# --------------------------------------------------------------------------


def test_deprecated_split_fits_and_evaluates_on_the_same_split() -> None:
    args = parse_args(["--results-dir", "/tmp/does-not-matter", "--split", "test"])
    fit_split, eval_split, fit_pf, eval_pf, deprecated = resolve_splits(args)

    assert fit_split == eval_split == "test"
    assert deprecated is True


def test_deprecated_split_prints_warning_naming_leakage_and_new_flags(
    capsys: pytest.CaptureFixture[str],
) -> None:
    args = parse_args(["--results-dir", "/tmp/does-not-matter", "--split", "val"])
    resolve_splits(args)

    captured = capsys.readouterr()
    assert "deprecated" in captured.err.lower()
    assert "leak" in captured.err.lower()
    assert "--fit-split" in captured.err
    assert "--eval-split" in captured.err


def test_deprecated_split_end_to_end_still_works(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The full CLI path (main()) with the deprecated --split flag must
    still work end to end -- runbook commands already using it must not
    break -- and must still print the leakage warning."""
    results_dir = tmp_path / "run"
    results_dir.mkdir()
    _write_predictions(results_dir / "test_predictions.csv", n=100, correct_fraction=0.8)

    output_dir = tmp_path / "out"
    main(
        [
            "--results-dir",
            str(results_dir),
            "--split",
            "test",
            "--output-dir",
            str(output_dir),
            "--no-zip",
        ]
    )

    captured = capsys.readouterr()
    assert re.search(DEPRECATED_SPLIT_WARNING[:30], captured.err) or "deprecated" in captured.err
    assert (output_dir / "decision_policy.json").exists()
    assert (output_dir / "decision_band_metrics_fit.csv").exists()
    assert (output_dir / "decision_band_metrics_eval.csv").exists()

    # Deprecated mode: fit split == eval split == same data, so the two
    # metrics tables must be identical (this is the leakage the warning
    # names, not a bug in the new code path).
    fit_metrics = pd.read_csv(output_dir / "decision_band_metrics_fit.csv")
    eval_metrics = pd.read_csv(output_dir / "decision_band_metrics_eval.csv")
    pd.testing.assert_frame_equal(fit_metrics, eval_metrics)


def test_predictions_file_only_valid_with_deprecated_split() -> None:
    args = parse_args(
        [
            "--results-dir",
            "/tmp/does-not-matter",
            "--predictions-file",
            "/tmp/some_predictions.csv",
        ]
    )
    with pytest.raises(ConflictingSplitArgumentsError, match="only valid together"):
        resolve_splits(args)


@pytest.mark.parametrize(
    "extra_args",
    [
        ["--fit-split", "val"],
        ["--eval-split", "test"],
        ["--fit-predictions-file", "/tmp/fit.csv"],
        ["--eval-predictions-file", "/tmp/eval.csv"],
    ],
)
def test_split_conflicts_with_new_flags(extra_args: list[str]) -> None:
    args = parse_args(
        ["--results-dir", "/tmp/does-not-matter", "--split", "test", *extra_args]
    )
    with pytest.raises(ConflictingSplitArgumentsError, match="cannot be combined"):
        resolve_splits(args)


def test_main_exits_cleanly_with_conflicting_arguments(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(
            [
                "--results-dir",
                str(tmp_path),
                "--split",
                "test",
                "--fit-split",
                "val",
            ]
        )

    assert excinfo.value.code == 1
    captured = capsys.readouterr()
    assert "error:" in captured.err
