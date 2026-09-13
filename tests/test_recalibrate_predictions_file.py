"""Regression tests for `--predictions-file` in scripts/recalibrate_decision_layer.py.

Background: `kaggle/a3b_rescore/rescore_predictions.py` deliberately writes
`<split>_predictions_rescored.csv` rather than `<split>_predictions.csv`, so it
never clobbers the immutable run record (see
`docs/superpowers/specs/2026-09-14-decision-layer-methodology-design.md`,
section 3.3, and `docs/0_coding_standards.md`). But
`recalibrate_decision_layer.py` used to read `<split>_predictions.csv`
unconditionally, so the documented handoff command failed schema validation
against the old, incompatible file -- the only reason the 2026-09-11 run
worked was a sibling run directory staged by hand, a step in no
documentation. `--predictions-file` fixes the handoff: it overrides the
`<split>_predictions.csv` convention while `--results-dir` still supplies the
other artifacts (class report, confusion pairs, output location), with no
copying, symlinking or renaming required.

The module lives in scripts/, not a package, so it is loaded by file path
rather than imported normally (see tests/test_check_doc_links.py).
"""

from __future__ import annotations

import importlib.util
import json
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


def _write_fake_predictions(path: Path, n: int = 20) -> None:
    labels = ["miso_soup", "pho", "ramen", "sushi"]
    rows = []
    for i in range(n):
        actual = labels[i % len(labels)]
        predicted = actual if i % 3 != 0 else labels[(i + 1) % len(labels)]
        top_5 = "|".join(labels)
        top_5_confidence = "|".join(
            f"{max(0.05, 0.9 - 0.2 * abs(j - labels.index(predicted))):.4f}"
            for j in range(len(labels))
        )
        rows.append(
            {
                "path": f"/data/{actual}/{i}.jpg",
                "actual": actual,
                "predicted": predicted,
                "is_correct": actual == predicted,
                "top_5": top_5,
                "top_5_confidence": top_5_confidence,
            }
        )
    pd.DataFrame(rows).to_csv(path, index=False)


def test_predictions_file_overrides_conventional_name(tmp_path: Path) -> None:
    """`--predictions-file` is read instead of `<split>_predictions.csv`, and
    no `<split>_predictions.csv` needs to exist at all."""
    results_dir = tmp_path / "run"
    results_dir.mkdir()

    rescored_path = results_dir / "test_predictions_rescored.csv"
    _write_fake_predictions(rescored_path)

    # Deliberately absent: this must not be required or read.
    conventional_path = results_dir / "test_predictions.csv"
    assert not conventional_path.exists()

    output_dir = tmp_path / "out"
    run_analysis(
        results_dir=results_dir,
        split="test",
        hard_classes_file=None,
        confusion_pairs_file=None,
        class_report_file=None,
        output_dir=output_dir,
        max_confusion_pairs=10,
        skip_zip=True,
        predictions_file=str(rescored_path),
    )

    assert not conventional_path.exists()
    assert (output_dir / "decision_policy.json").exists()
    policy = json.loads((output_dir / "decision_policy.json").read_text())
    assert policy  # a policy was actually selected


def test_predictions_file_relative_path_resolves_against_cwd_not_results_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`--predictions-file` is a normal CLI path (relative to cwd), not
    relative to --results-dir -- unlike --hard-classes-file and friends. This
    matters because the documented handoff command
    (kaggle/a3b_rescore/README.md) passes a path like
    `results/.../test_predictions_rescored.csv` relative to the repo root,
    which is *not* relative to --results-dir (it repeats the results-dir
    prefix)."""
    results_dir = tmp_path / "run"
    results_dir.mkdir()
    rescored_path = results_dir / "test_predictions_rescored.csv"
    _write_fake_predictions(rescored_path)

    monkeypatch.chdir(tmp_path)

    output_dir = tmp_path / "out"
    run_analysis(
        results_dir=results_dir,
        split="test",
        hard_classes_file=None,
        confusion_pairs_file=None,
        class_report_file=None,
        output_dir=output_dir,
        max_confusion_pairs=10,
        skip_zip=True,
        predictions_file="run/test_predictions_rescored.csv",
    )

    assert (output_dir / "decision_policy.json").exists()


def test_without_predictions_file_still_uses_conventional_name(tmp_path: Path) -> None:
    """Unchanged default behavior: no --predictions-file means the
    `<split>_predictions.csv` convention still applies."""
    results_dir = tmp_path / "run"
    results_dir.mkdir()
    conventional_path = results_dir / "test_predictions.csv"
    _write_fake_predictions(conventional_path)

    output_dir = tmp_path / "out"
    run_analysis(
        results_dir=results_dir,
        split="test",
        hard_classes_file=None,
        confusion_pairs_file=None,
        class_report_file=None,
        output_dir=output_dir,
        max_confusion_pairs=10,
        skip_zip=True,
        predictions_file=None,
    )

    assert (output_dir / "decision_policy.json").exists()


def test_missing_predictions_file_fails_with_clear_error(tmp_path: Path) -> None:
    results_dir = tmp_path / "run"
    results_dir.mkdir()

    output_dir = tmp_path / "out"
    with pytest.raises(FileNotFoundError, match="does_not_exist.csv"):
        run_analysis(
            results_dir=results_dir,
            split="test",
            hard_classes_file=None,
            confusion_pairs_file=None,
            class_report_file=None,
            output_dir=output_dir,
            max_confusion_pairs=10,
            skip_zip=True,
            predictions_file="does_not_exist.csv",
        )
