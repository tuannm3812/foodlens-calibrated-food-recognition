"""Regression tests for kaggle/a3b_rescore/rescore_predictions.py.

The A3b run's `test_predictions.csv` predates the predictions-CSV contract in
`docs/4_next_steps.md` (see `tests/test_recalibration_contract.py`): it has no
per-class top-5 confidences, so `scripts/recalibrate_decision_layer.py`
refuses it. `rescore_predictions.py` re-scores the split using the already
trained A3b checkpoint to add them.

Food-101 is ~5GB and not available in this environment, so these tests only
exercise the pure helpers that do not need a model or real images: manifest
path remapping, top-5 row formatting, the class-ordering rule, temperature
resolution/application, and the accuracy self-check. The
model-construction/checkpoint-loading and full scoring-loop paths are
exercised separately, by hand, against the real checkpoint (see the task
report) rather than in this suite.

The module lives in kaggle/, not a package, so it is loaded by file path
rather than imported normally (see tests/test_check_doc_links.py).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest
import torch

SCRIPT_PATH = (
    Path(__file__).resolve().parent.parent / "kaggle" / "a3b_rescore" / "rescore_predictions.py"
)

_spec = importlib.util.spec_from_file_location("rescore_predictions", SCRIPT_PATH)
assert _spec is not None and _spec.loader is not None
rescore_predictions = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rescore_predictions)

remap_manifest_path = rescore_predictions.remap_manifest_path
find_missing_paths = rescore_predictions.find_missing_paths
ensure_paths_exist = rescore_predictions.ensure_paths_exist
load_class_names = rescore_predictions.load_class_names
resolve_temperature = rescore_predictions.resolve_temperature
format_topk_columns = rescore_predictions.format_topk_columns
build_prediction_row = rescore_predictions.build_prediction_row
accuracy_from_rows = rescore_predictions.accuracy_from_rows
load_recorded_metrics = rescore_predictions.load_recorded_metrics
check_accuracy_matches_recorded = rescore_predictions.check_accuracy_matches_recorded
resolve_manifest_paths = rescore_predictions.resolve_manifest_paths
resolve_data_dir = rescore_predictions.resolve_data_dir
REQUIRED_OUTPUT_COLUMNS = rescore_predictions.REQUIRED_OUTPUT_COLUMNS
AccuracyMismatchError = rescore_predictions.AccuracyMismatchError
MissingTemperatureError = rescore_predictions.MissingTemperatureError


# --------------------------------------------------------------------------
# Manifest path remapping
# --------------------------------------------------------------------------


def test_remap_manifest_path_takes_last_two_segments(tmp_path: Path) -> None:
    kaggle_path = "/kaggle/input/datasets/kmader/food41/images/miso_soup/1014272.jpg"

    result = remap_manifest_path(kaggle_path, tmp_path)

    assert result == tmp_path / "miso_soup" / "1014272.jpg"


def test_remap_manifest_path_rejects_too_short_path(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="fewer than two segments"):
        remap_manifest_path("1014272.jpg", tmp_path)


def test_resolve_manifest_paths_without_data_dir_uses_raw_paths() -> None:
    manifest = pd.DataFrame({"path": ["/kaggle/input/images/pho/1.jpg"], "label": ["pho"]})

    result = resolve_manifest_paths(manifest, data_dir=None)

    assert result == [Path("/kaggle/input/images/pho/1.jpg")]


def test_resolve_manifest_paths_with_data_dir_remaps_all_rows(tmp_path: Path) -> None:
    manifest = pd.DataFrame(
        {
            "path": [
                "/kaggle/input/images/pho/1.jpg",
                "/kaggle/input/images/ramen/2.jpg",
            ],
            "label": ["pho", "ramen"],
        }
    )

    result = resolve_manifest_paths(manifest, data_dir=tmp_path)

    assert result == [tmp_path / "pho" / "1.jpg", tmp_path / "ramen" / "2.jpg"]


def test_resolve_data_dir_prefers_cli_over_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FOODLENS_DATA_DIR", "/env/data")

    result = resolve_data_dir("/cli/data")

    assert result == Path("/cli/data").resolve()


def test_resolve_data_dir_falls_back_to_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FOODLENS_DATA_DIR", "/env/data")

    result = resolve_data_dir(None)

    assert result == Path("/env/data").resolve()


def test_resolve_data_dir_none_when_neither_set(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FOODLENS_DATA_DIR", raising=False)

    assert resolve_data_dir(None) is None


# --------------------------------------------------------------------------
# Missing-file failure (the "fail early with a clear message" requirement)
# --------------------------------------------------------------------------


def test_find_missing_paths_reports_only_absent_files(tmp_path: Path) -> None:
    present = tmp_path / "present.jpg"
    present.write_bytes(b"x")
    absent = tmp_path / "absent.jpg"

    assert find_missing_paths([present, absent]) == [absent]


def test_ensure_paths_exist_passes_when_all_present(tmp_path: Path) -> None:
    present = tmp_path / "present.jpg"
    present.write_bytes(b"x")

    ensure_paths_exist([present])  # must not raise


def test_ensure_paths_exist_raises_with_count_and_example(tmp_path: Path) -> None:
    present = tmp_path / "present.jpg"
    present.write_bytes(b"x")
    absent = tmp_path / "absent.jpg"

    with pytest.raises(FileNotFoundError) as excinfo:
        ensure_paths_exist([present, absent])

    message = str(excinfo.value)
    assert "1 of 2" in message
    assert str(absent) in message


# --------------------------------------------------------------------------
# Class ordering rule
# --------------------------------------------------------------------------


def test_load_class_names_reads_json_in_file_order(tmp_path: Path) -> None:
    (tmp_path / "class_names.json").write_text('["zebra_food", "apple_pie"]', encoding="utf-8")

    result = load_class_names(tmp_path)

    assert result == ["zebra_food", "apple_pie"]


def test_load_class_names_missing_file_fails_clearly(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="class_names.json"):
        load_class_names(tmp_path)


# --------------------------------------------------------------------------
# Temperature resolution
#
# `app/backend/inference.py:426` serves `softmax(logits / temperature, dim=1)`
# with temperature read from calibration.json, and
# `scripts/recalibrate_decision_layer.py` never applies temperature itself --
# it fits thresholds directly on whatever confidences this script emits. So
# this script must default to the same calibration.json temperature, accept
# an explicit override, and never silently fall back to 1.0.
# --------------------------------------------------------------------------


def test_resolve_temperature_reads_calibration_json(tmp_path: Path) -> None:
    (tmp_path / "calibration.json").write_text(
        '{"temperature": 0.88435298204422}', encoding="utf-8"
    )

    temperature, source = resolve_temperature(tmp_path, explicit=None)

    assert temperature == pytest.approx(0.88435298204422)
    assert str(tmp_path / "calibration.json") == source


def test_resolve_temperature_explicit_overrides_calibration_json(tmp_path: Path) -> None:
    (tmp_path / "calibration.json").write_text('{"temperature": 0.5}', encoding="utf-8")

    temperature, source = resolve_temperature(tmp_path, explicit=1.0)

    assert temperature == 1.0
    assert source == "--temperature flag"


def test_resolve_temperature_missing_file_fails_clearly_not_silently(tmp_path: Path) -> None:
    with pytest.raises(MissingTemperatureError, match="calibration.json"):
        resolve_temperature(tmp_path, explicit=None)


def test_resolve_temperature_missing_key_fails_clearly_not_silently(tmp_path: Path) -> None:
    (tmp_path / "calibration.json").write_text('{"other_key": 1.0}', encoding="utf-8")

    with pytest.raises(MissingTemperatureError, match="temperature"):
        resolve_temperature(tmp_path, explicit=None)


def test_resolve_temperature_explicit_one_point_zero_bypasses_missing_file(
    tmp_path: Path,
) -> None:
    """`--temperature 1.0` must work even with no calibration.json, as a
    deliberate opt-out of scaling -- distinct from the silent-default bug."""
    temperature, source = resolve_temperature(tmp_path, explicit=1.0)

    assert temperature == 1.0
    assert source == "--temperature flag"


# --------------------------------------------------------------------------
# Temperature scaling direction
#
# Softmax(logits / T) must be applied in the same direction as
# app/backend/inference.py: T < 1 sharpens (higher top-1 confidence than
# raw), T > 1 flattens (lower), T == 1 is a no-op.
# --------------------------------------------------------------------------


def _top1_confidence(logits: list[float], temperature: float) -> float:
    tensor = torch.tensor([logits])
    probabilities = torch.softmax(tensor / temperature, dim=1)
    return float(probabilities.max().item())


def test_temperature_below_one_sharpens_top1_confidence_above_raw() -> None:
    logits = [4.0, 1.0, 0.5, 0.1, -1.0]

    raw = _top1_confidence(logits, temperature=1.0)
    scaled = _top1_confidence(logits, temperature=0.88435298204422)

    assert scaled > raw


def test_temperature_above_one_flattens_top1_confidence_below_raw() -> None:
    logits = [4.0, 1.0, 0.5, 0.1, -1.0]

    raw = _top1_confidence(logits, temperature=1.0)
    scaled = _top1_confidence(logits, temperature=2.0)

    assert scaled < raw


def test_temperature_of_one_leaves_top1_confidence_unchanged() -> None:
    logits = [4.0, 1.0, 0.5, 0.1, -1.0]

    raw = _top1_confidence(logits, temperature=1.0)
    scaled = _top1_confidence(logits, temperature=1.0)

    assert scaled == pytest.approx(raw)


# --------------------------------------------------------------------------
# Top-5 row formatting
# --------------------------------------------------------------------------


def test_format_topk_columns_is_pipe_separated_and_rank_aligned() -> None:
    top_5, top_5_confidence = format_topk_columns(
        ["miso_soup", "pho", "ramen"], [0.9, 0.05, 0.02]
    )

    assert top_5 == "miso_soup|pho|ramen"
    assert top_5_confidence == "0.90000000|0.05000000|0.02000000"


def test_format_topk_columns_rejects_length_mismatch() -> None:
    with pytest.raises(ValueError, match="same length"):
        format_topk_columns(["a", "b"], [0.5])


def test_build_prediction_row_emits_exactly_the_documented_columns() -> None:
    row = build_prediction_row(
        path="/kaggle/input/images/pho/1.jpg",
        true_label="pho",
        top_labels=["pho", "ramen", "miso_soup", "sushi", "tacos"],
        top_scores=[0.7, 0.1, 0.08, 0.07, 0.05],
        temperature=0.88435298204422,
    )

    assert set(row.keys()) == set(REQUIRED_OUTPUT_COLUMNS)
    assert row["pred_label"] == "pho"
    assert row["confidence"] == pytest.approx(0.7)
    assert row["is_correct"] is True
    assert row["top_5"] == "pho|ramen|miso_soup|sushi|tacos"
    assert row["temperature"] == pytest.approx(0.88435298204422)


def test_build_prediction_row_marks_incorrect_when_top1_differs() -> None:
    row = build_prediction_row(
        path="/kaggle/input/images/pho/1.jpg",
        true_label="pho",
        top_labels=["ramen", "pho", "miso_soup", "sushi", "tacos"],
        top_scores=[0.6, 0.2, 0.1, 0.06, 0.04],
    )

    assert row["is_correct"] is False
    assert row["pred_label"] == "ramen"


def test_accuracy_from_rows_computes_top1_and_top5_percentages() -> None:
    rows = [
        build_prediction_row(
            "p1", "pho", ["pho", "a", "b", "c", "d"], [0.9, 0.05, 0.02, 0.02, 0.01]
        ),
        build_prediction_row(
            "p2", "sushi", ["a", "sushi", "b", "c", "d"], [0.5, 0.3, 0.1, 0.05, 0.05]
        ),
        build_prediction_row(
            "p3", "tacos", ["a", "b", "c", "d", "e"], [0.5, 0.3, 0.1, 0.05, 0.05]
        ),
    ]

    top1_pct, top5_pct = accuracy_from_rows(rows)

    assert top1_pct == pytest.approx(100.0 / 3.0)
    assert top5_pct == pytest.approx(200.0 / 3.0)


# --------------------------------------------------------------------------
# Accuracy self-check
# --------------------------------------------------------------------------


def test_load_recorded_metrics_reads_top1_and_top5(tmp_path: Path) -> None:
    metrics_path = tmp_path / "test_metrics.csv"
    pd.DataFrame([{"split": "test", "top_1_accuracy": 83.9, "top_5_accuracy": 95.78}]).to_csv(
        metrics_path, index=False
    )

    result = load_recorded_metrics(metrics_path)

    assert result == pytest.approx((83.9, 95.78))


def test_load_recorded_metrics_none_when_file_absent(tmp_path: Path) -> None:
    assert load_recorded_metrics(tmp_path / "missing_metrics.csv") is None


def test_self_check_passes_within_tolerance() -> None:
    check_accuracy_matches_recorded(83.90, 95.78, recorded=(83.9001, 95.7799))  # must not raise


def test_self_check_skips_when_no_recorded_metrics() -> None:
    check_accuracy_matches_recorded(83.90, 95.78, recorded=None)  # must not raise


def test_self_check_fails_on_mismatch_with_explanatory_message() -> None:
    with pytest.raises(AccuracyMismatchError) as excinfo:
        check_accuracy_matches_recorded(70.0, 80.0, recorded=(83.9, 95.78))

    message = str(excinfo.value)
    assert "preprocessing or class ordering is wrong" in message
    assert "70.0" in message or "70.0000" in message
