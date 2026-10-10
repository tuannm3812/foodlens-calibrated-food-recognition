"""Producer evidence in scripts/recalibrate_decision_layer.py and compare_provenance.py.

Recalibration verifies each predictions CSV's evidence sidecar against the
bytes it read, rejects mixed or inconsistent temperatures, requires the fit and
eval predictions to come from one model, and records the evidence in
``derivation_provenance.json`` (schema 3) for the deploy script to bind to.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest
from evidence_fixtures import prediction_evidence, rescore, sha256, write_scored_split

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


recalibrate = _load("recalibrate_decision_layer")
compare = _load("compare_provenance")

LABELS = [f"class_{index:02d}" for index in range(20)]


def rows(temperature: float) -> list[dict]:
    """Twenty rows per class; class i's first (i % 4) + 1 rows are predicted as class i+1."""
    built = []
    for index, label in enumerate(LABELS):
        wrong = LABELS[(index + 1) % len(LABELS)]
        for row in range(20):
            predicted = wrong if row <= index % 4 else label
            others = [name for name in LABELS if name != predicted][:4]
            built.append(
                rescore.build_prediction_row(
                    path=f"/kaggle/input/{label}/{row}.jpg",
                    true_label=label,
                    top_labels=[predicted, *others],
                    top_scores=[0.9 if predicted == label else 0.55, 0.02, 0.02, 0.02, 0.02],
                    temperature=temperature,
                )
            )
    return built


def model_run(
    path: Path,
    *,
    checkpoint_bytes: bytes = b"weights",
    architecture: str = "convnext_tiny",
    temperature: float = 0.884,
    eval_temperature: float | None = None,
) -> Path:
    """A run with re-scored val and test CSVs and their evidence sidecars."""
    path.mkdir(parents=True)
    checkpoint = path / "model_best.pth"
    checkpoint.write_bytes(checkpoint_bytes)
    for split, value in (("val", temperature), ("test", eval_temperature or temperature)):
        write_scored_split(
            path / f"{split}_predictions_rescored.csv",
            split=split,
            checkpoint=checkpoint,
            architecture=architecture,
            class_names=LABELS,
            temperature=value,
            rows=rows(value),
        )
    return path


def recalibrate_run(run: Path) -> dict:
    recalibrate.main(
        [
            "--results-dir", str(run),
            "--fit-predictions-file", str(run / "val_predictions_rescored.csv"),
            "--eval-predictions-file", str(run / "test_predictions_rescored.csv"),
            "--output-dir", str(run / "out"),
            "--no-zip",
        ]
    )
    return json.loads((run / "out" / "derivation_provenance.json").read_text())


def rejected(run: Path, capsys, fragment: str) -> None:
    with pytest.raises(SystemExit) as info:
        recalibrate_run(run)
    assert info.value.code == 1
    err = capsys.readouterr().err
    assert err.startswith("error: ") and fragment in err, err
    assert not (run / "out" / "derivation_provenance.json").exists()


# --- Evidence is verified and recorded ---------------------------------------------


def test_provenance_records_each_splits_verified_evidence(tmp_path) -> None:
    run = model_run(tmp_path / "run")
    provenance = recalibrate_run(run)

    assert provenance["schema_version"] == 3
    for split, name in (("fit", "val"), ("eval", "test")):
        entry = provenance["predictions"][split]
        sidecar = run / f"{name}_predictions_rescored.csv.evidence.json"
        assert entry["evidence"]["path"] == str(sidecar)
        assert entry["evidence"]["sha256"] == sha256(sidecar)
        assert entry["evidence"]["record"] == json.loads(sidecar.read_text())
        assert entry["evidence"]["record"]["predictions"]["sha256"] == entry["sha256"]
    binding = prediction_evidence.policy_evidence(provenance, Path("provenance"))
    assert binding["checkpoint_sha256"] == sha256(run / "model_best.pth")
    assert binding["temperature"] == 0.884


def test_a_mixed_temperature_column_is_rejected(tmp_path, capsys) -> None:
    run = model_run(tmp_path / "run")
    path = run / "val_predictions_rescored.csv"
    frame = pd.read_csv(path)
    frame.loc[frame.index[:5], "temperature"] = 2.0
    frame.to_csv(path, index=False)
    prediction_evidence.sidecar_path(path).unlink()  # even without evidence
    prediction_evidence.sidecar_path(run / "test_predictions_rescored.csv").unlink()
    rejected(run, capsys, "mixes 2 temperature values")


def test_a_sidecar_whose_csv_hash_does_not_match_is_rejected(tmp_path, capsys) -> None:
    run = model_run(tmp_path / "run")
    path = run / "test_predictions_rescored.csv"
    frame = pd.read_csv(path)
    frame.loc[0, "confidence"] = 0.99
    frame.to_csv(path, index=False)
    rejected(run, capsys, "belongs to other bytes")


def test_a_temperature_column_unlike_the_sidecar_is_rejected(tmp_path, capsys) -> None:
    run = model_run(tmp_path / "run")
    path = run / "val_predictions_rescored.csv"
    sidecar = prediction_evidence.sidecar_path(path)
    record = json.loads(sidecar.read_text())
    record["temperature"]["value"] = 2.0
    sidecar.write_text(json.dumps(record))
    rejected(run, capsys, "its evidence")


def test_fit_and_eval_from_different_checkpoints_are_rejected(tmp_path, capsys) -> None:
    run = model_run(tmp_path / "run")
    other = model_run(tmp_path / "other", checkpoint_bytes=b"other weights")
    for suffix in ("", ".evidence.json"):
        name = f"test_predictions_rescored.csv{suffix}"
        (run / name).write_bytes((other / name).read_bytes())
    rejected(run, capsys, "checkpoint_sha256")


def test_fit_and_eval_with_different_temperatures_are_rejected(tmp_path, capsys) -> None:
    run = model_run(tmp_path / "run", eval_temperature=2.0)
    rejected(run, capsys, "temperature: fit 0.884, eval 2.0")


def test_without_sidecars_differing_temperature_columns_are_rejected(tmp_path, capsys) -> None:
    run = model_run(tmp_path / "run", eval_temperature=2.0)
    for name in ("val", "test"):
        prediction_evidence.sidecar_path(run / f"{name}_predictions_rescored.csv").unlink()
    rejected(run, capsys, "share one calibration")


def test_evidence_on_one_split_only_is_rejected(tmp_path, capsys) -> None:
    run = model_run(tmp_path / "run")
    prediction_evidence.sidecar_path(run / "test_predictions_rescored.csv").unlink()
    rejected(run, capsys, "has producer evidence but")


def test_without_any_sidecar_recalibration_runs_and_records_no_evidence(tmp_path) -> None:
    run = model_run(tmp_path / "run")
    for name in ("val", "test"):
        prediction_evidence.sidecar_path(run / f"{name}_predictions_rescored.csv").unlink()
    provenance = recalibrate_run(run)
    assert "evidence" not in provenance["predictions"]["fit"]
    with pytest.raises(prediction_evidence.EvidenceError, match="Regenerate evidence"):
        prediction_evidence.policy_evidence(provenance, Path("provenance"))


# --- compare_provenance ----------------------------------------------------------


def test_two_models_evidence_differs_only_in_model_specific_fields(tmp_path) -> None:
    first = recalibrate_run(model_run(tmp_path / "a"))
    second = recalibrate_run(
        model_run(
            tmp_path / "b",
            checkpoint_bytes=b"resnet weights",
            architecture="resnet50",
            temperature=0.958,
        )
    )
    incompatible, model_specific = compare.compare_provenance(first, second)
    assert incompatible == []
    for field in (
        "predictions.fit.evidence.sha256",
        "predictions.fit.evidence.record.checkpoint.sha256",
        "predictions.fit.evidence.record.architecture",
        "predictions.eval.evidence.record.temperature.value",
    ):
        assert field in model_specific


def test_different_preprocessing_or_producer_is_incompatible(tmp_path) -> None:
    first = recalibrate_run(model_run(tmp_path / "a"))
    second = json.loads(json.dumps(first))
    record = second["predictions"]["fit"]["evidence"]["record"]
    record["preprocessing"]["resize"] = [256, 256]
    record["producer"]["sha256"] = "0" * 64
    incompatible, _ = compare.compare_provenance(first, second)
    assert "predictions.fit.evidence.record.preprocessing.resize" in incompatible
    assert "predictions.fit.evidence.record.producer.sha256" in incompatible


def test_evidence_on_one_side_only_is_incompatible(tmp_path) -> None:
    first = recalibrate_run(model_run(tmp_path / "a"))
    second = json.loads(json.dumps(first))
    del second["predictions"]["fit"]["evidence"]
    incompatible, _ = compare.compare_provenance(first, second)
    assert "predictions.fit.evidence.sha256" in incompatible
