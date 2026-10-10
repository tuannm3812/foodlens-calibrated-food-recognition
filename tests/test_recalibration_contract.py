"""Regression tests for the predictions-schema gate in
scripts/recalibrate_decision_layer.py.

Running the recalibration script against a real accuracy-phase run (which
emits `top_5` as labels only, with no per-class confidences) used to die with
a bare `KeyError: 'top_5_confidence'` raised from deep inside pandas, after
hard-class and confusion-pair analysis had already run. These tests pin down
the replacement: an explicit, actionable schema check that runs before any
analysis work.

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

validate_predictions_schema = recalibrate_decision_layer.validate_predictions_schema
PredictionSchemaError = recalibrate_decision_layer.PredictionSchemaError
REQUIRED_PREDICTION_COLUMNS = recalibrate_decision_layer.REQUIRED_PREDICTION_COLUMNS


COMPLETE_COLUMNS = {
    "actual": ["miso_soup", "frozen_yogurt"],
    "predicted": ["miso_soup", "frozen_yogurt"],
    "is_correct": [True, True],
    "top_5": ["miso_soup|pho|ramen", "frozen_yogurt|ice_cream|donuts"],
    "top_5_confidence": ["0.9|0.05|0.02", "0.8|0.1|0.05"],
}


def test_missing_top_5_confidence_raises_schema_error_naming_column_and_file(
    tmp_path: Path,
) -> None:
    frame = pd.DataFrame(
        {key: value for key, value in COMPLETE_COLUMNS.items() if key != "top_5_confidence"}
    )
    predictions_path = tmp_path / "test_predictions.csv"
    frame.to_csv(predictions_path, index=False)

    with pytest.raises(PredictionSchemaError) as excinfo:
        validate_predictions_schema(frame, predictions_path)

    message = str(excinfo.value)
    assert "top_5_confidence" in message
    assert str(predictions_path) in message


def test_missing_several_columns_names_all_of_them(tmp_path: Path) -> None:
    frame = pd.DataFrame({"actual": ["miso_soup"], "predicted": ["miso_soup"]})
    predictions_path = tmp_path / "test_predictions.csv"
    frame.to_csv(predictions_path, index=False)

    with pytest.raises(PredictionSchemaError) as excinfo:
        validate_predictions_schema(frame, predictions_path)

    message = str(excinfo.value)
    for missing_col in ("is_correct", "top_5", "top_5_confidence"):
        assert missing_col in message
    # The columns that ARE present should not be reported as missing.
    assert "missing columns: actual" not in message
    assert "missing columns: predicted" not in message


def test_frame_with_every_required_column_passes_validation(tmp_path: Path) -> None:
    frame = pd.DataFrame(COMPLETE_COLUMNS)
    predictions_path = tmp_path / "test_predictions.csv"
    frame.to_csv(predictions_path, index=False)

    # Should not raise.
    validate_predictions_schema(frame, predictions_path)

    # And every column build_features() needs really is accounted for.
    assert set(REQUIRED_PREDICTION_COLUMNS).issubset(frame.columns)


def test_schema_error_is_not_a_bare_key_error(tmp_path: Path) -> None:
    frame = pd.DataFrame(
        {key: value for key, value in COMPLETE_COLUMNS.items() if key != "top_5_confidence"}
    )
    predictions_path = tmp_path / "test_predictions.csv"
    frame.to_csv(predictions_path, index=False)

    with pytest.raises(PredictionSchemaError) as excinfo:
        validate_predictions_schema(frame, predictions_path)

    assert not isinstance(excinfo.value, KeyError)
    assert isinstance(excinfo.value, ValueError)
    # A KeyError's str() would be the quoted key alone (e.g. "'top_5_confidence'");
    # the real message is a multi-line, multi-sentence explanation.
    assert len(str(excinfo.value).splitlines()) > 1
