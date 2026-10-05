"""Tests for how scripts/recalibrate_decision_layer.py resolves its derived inputs.

Background: the controlled A3b-vs-champion comparison was blocked three times
by one recurring defect -- the script silently changed its inputs depending on
which optional files happened to sit in the run directory (see
docs/9_agent_log.md, the 2026-09-22 Codex review and the entries after it).
These tests pin the agreed contract:

- hard classes and confusion pairs are derived from the fit-split predictions
  unless an override file is named; an unnamed sidecar changes nothing;
- a named override that is missing or malformed fails through `main()`'s
  handled-error path (one `error:` line, exit 1, no traceback).

The module lives in scripts/, not a package, so it is loaded by file path
rather than imported normally (see tests/test_check_doc_links.py).
"""

from __future__ import annotations

import importlib.util
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

LABELS = [f"class_{i:02d}" for i in range(20)]


def _prediction_rows(errors_per_class: dict[str, int], per_class: int = 20) -> list[dict]:
    """Rows over `LABELS`; `errors_per_class[c]` rows of class `c` are wrong.

    A wrong row of class `LABELS[i]` is predicted as `LABELS[i + 1]`, so each
    class contributes exactly one confusion pair whose count is its error
    count. Equal error counts therefore produce tied pairs and, because each
    class then also has equal recall, tied F1 scores.
    """
    rows = []
    for index, label in enumerate(LABELS):
        wrong_target = LABELS[(index + 1) % len(LABELS)]
        n_wrong = errors_per_class.get(label, 0)
        for i in range(per_class):
            is_correct = i >= n_wrong
            predicted = label if is_correct else wrong_target
            confidence = 0.9 if is_correct else 0.55
            others = [name for name in LABELS if name != predicted][:4]
            ranked = [predicted, *others]
            confidences = [confidence, *([0.02] * len(others))]
            rows.append(
                {
                    "path": f"/data/{label}/{i}.jpg",
                    "actual": label,
                    "predicted": predicted,
                    "is_correct": is_correct,
                    "top_5": "|".join(ranked),
                    "top_5_confidence": "|".join(f"{c:.4f}" for c in confidences),
                }
            )
    return rows


DEFAULT_ERRORS = {label: (i % 4) + 1 for i, label in enumerate(LABELS)}


def _make_run_dir(path: Path, errors: dict[str, int] | None = None) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    rows = _prediction_rows(errors or DEFAULT_ERRORS)
    pd.DataFrame(rows).to_csv(path / "val_predictions.csv", index=False)
    pd.DataFrame(rows).to_csv(path / "test_predictions.csv", index=False)
    return path


def _run_main(results_dir: Path, *extra: str) -> None:
    main(
        [
            "--results-dir",
            str(results_dir),
            "--output-dir",
            str(results_dir / "out"),
            "--no-zip",
            *extra,
        ]
    )


def _assert_handled_failure(
    capsys: pytest.CaptureFixture[str], results_dir: Path, *extra: str, needle: str
) -> str:
    with pytest.raises(SystemExit) as excinfo:
        _run_main(results_dir, *extra)
    assert excinfo.value.code == 1
    err = capsys.readouterr().err
    assert err.startswith("error: "), err
    assert "Traceback" not in err
    assert needle in err, err
    return err


# --------------------------------------------------------------------------
# A. No implicit sidecars.
# --------------------------------------------------------------------------


def test_unnamed_confusion_pair_sidecar_does_not_change_any_output(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    results_dir = _make_run_dir(tmp_path / "run")

    def run(output_name: str) -> dict[str, bytes]:
        output_dir = results_dir / output_name
        run_analysis(
            results_dir=results_dir,
            fit_split="val",
            eval_split="test",
            hard_classes_file=None,
            confusion_pairs_file=None,
            class_report_file=None,
            output_dir=output_dir,
            max_confusion_pairs=5,
            skip_zip=True,
        )
        return {p.name: p.read_bytes() for p in sorted(output_dir.iterdir())}

    without_sidecar = run("out_without")
    # A sidecar naming pairs the fit predictions never produce: if it were
    # read, confusion_pairs.json (and routing) would change.
    pd.DataFrame(
        {
            "true_label": ["class_00", "class_05"],
            "pred_label": ["class_19", "class_11"],
            "count": [99, 98],
        }
    ).to_csv(results_dir / "val_confusion_pairs.csv", index=False)
    capsys.readouterr()
    with_sidecar = run("out_with")

    assert with_sidecar == without_sidecar
    out = capsys.readouterr().out
    assert "val_confusion_pairs.csv is present but ignored" in out


def test_named_confusion_pair_file_is_used(tmp_path: Path) -> None:
    import json

    results_dir = _make_run_dir(tmp_path / "run")
    pd.DataFrame(
        {"true_label": ["class_00"], "pred_label": ["class_19"], "count": [3]}
    ).to_csv(results_dir / "val_confusion_pairs.csv", index=False)
    _run_main(results_dir, "--confusion-pairs-file", "val_confusion_pairs.csv")
    pairs = json.loads((results_dir / "out" / "confusion_pairs.json").read_text())
    assert pairs == [{"actual": "class_00", "predicted": "class_19"}]


def test_missing_named_confusion_pair_file_fails_through_handled_path(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    results_dir = _make_run_dir(tmp_path / "run")
    _assert_handled_failure(
        capsys,
        results_dir,
        "--confusion-pairs-file",
        "nope.csv",
        needle="does not exist",
    )
    assert not (results_dir / "out" / "confusion_pairs.json").exists()


@pytest.mark.parametrize(
    ("content", "needle"),
    [
        ("label_a,label_b\nx,y\n", "needs actual and predicted label columns"),
        ("true_label,pred_label,count\n", "no pair rows"),
        ("true_label,pred_label,count\nclass_00,class_01,many\n", "finite numbers"),
        ("true_label,pred_label\nclass_00,\n", "empty actual or predicted"),
        (
            "true_label,pred_label\nclass_00,class_01\nclass_00 ,class_01\n",
            "more than once",
        ),
        ("", "not a readable CSV"),
    ],
    ids=["no-label-columns", "header-only", "non-numeric-count", "empty-label",
         "duplicate-pair", "empty-file"],
)
def test_malformed_named_confusion_pair_file_fails_through_handled_path(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], content: str, needle: str
) -> None:
    results_dir = _make_run_dir(tmp_path / "run")
    (results_dir / "pairs.csv").write_text(content)
    _assert_handled_failure(
        capsys, results_dir, "--confusion-pairs-file", "pairs.csv", needle=needle
    )


# --------------------------------------------------------------------------
# B. Validated hard-class overrides.
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("content", "needle"),
    [
        ('"steak"', "must be a JSON list"),
        ('{"steak": true}', "must be a JSON list"),
        ('["steak", ', "is not valid JSON"),
        ("[]", "empty list"),
        ('["steak", ""]', "empty or whitespace-only"),
        ('["steak", "   "]', "empty or whitespace-only"),
        ('["steak", " steak"]', "more than once"),
        ('["steak", 3]', "must all be strings"),
    ],
    ids=["json-string", "json-object", "malformed-json", "empty-list", "empty-name",
         "whitespace-name", "duplicate-normalised", "non-string"],
)
def test_malformed_hard_classes_file_fails_through_handled_path(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], content: str, needle: str
) -> None:
    results_dir = _make_run_dir(tmp_path / "run")
    (results_dir / "hard.json").write_text(content)
    _assert_handled_failure(
        capsys, results_dir, "--hard-classes-file", "hard.json", needle=needle
    )


def test_valid_hard_classes_file_is_used_after_normalisation(tmp_path: Path) -> None:
    import json

    results_dir = _make_run_dir(tmp_path / "run")
    (results_dir / "hard.json").write_text('[" class_03 ", "class_07"]')
    _run_main(results_dir, "--hard-classes-file", "hard.json")
    hard = json.loads((results_dir / "out" / "hard_classes.json").read_text())
    assert hard == ["class_03", "class_07"]


@pytest.mark.parametrize(
    ("content", "needle"),
    [
        ("class_name,f1-score\n", "no class rows"),
        ("class_name,f1-score\nsteak,high\nramen,0.5\n", "finite numbers"),
        ("class_name,f1-score\nsteak,inf\nramen,0.5\n", "finite numbers"),
        ("class_name,f1-score\nsteak,\nramen,0.5\n", "finite numbers"),
        ("class_name,f1-score\nsteak,0.4\n steak,0.5\n", "duplicate class names"),
        ("class_name,f1-score\n,0.4\nramen,0.5\n", "empty `class_name`"),
        ("class_name,precision\nsteak,0.4\n", "`f1-score` columns"),
        ("", "not a readable CSV"),
    ],
    ids=["header-only", "non-numeric-f1", "infinite-f1", "missing-f1",
         "duplicate-normalised", "empty-name", "no-f1-column", "empty-file"],
)
def test_malformed_class_report_fails_through_handled_path(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], content: str, needle: str
) -> None:
    results_dir = _make_run_dir(tmp_path / "run")
    (results_dir / "report.csv").write_text(content)
    _assert_handled_failure(
        capsys, results_dir, "--class-report-file", "report.csv", needle=needle
    )


def test_missing_named_override_files_fail_through_handled_path(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    results_dir = _make_run_dir(tmp_path / "run")
    _assert_handled_failure(
        capsys, results_dir, "--hard-classes-file", "nope.json", needle="does not exist"
    )
    _assert_handled_failure(
        capsys, results_dir, "--class-report-file", "nope.csv", needle="does not exist"
    )
