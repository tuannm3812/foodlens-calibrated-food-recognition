"""The evidence sidecar the rescorer writes, and scripts/prediction_evidence.py.

The rescorer (a self-contained Kaggle ``code_file``) and the consumer module
each define the sidecar's schema, class-name hash and preprocessing record;
these tests pin them to each other and to what the backend serves.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from evidence_fixtures import prediction_evidence, rescore, sha256, write_scored_split

from app.backend import artifacts

CLASSES = [f"class_{index:03d}" for index in range(101)]


@pytest.fixture
def checkpoint(tmp_path: Path) -> Path:
    path = tmp_path / "model_best.pth"
    path.write_bytes(b"weights")
    return path


def scored(tmp_path: Path, checkpoint: Path, split: str = "val", temperature=0.884) -> Path:
    return write_scored_split(
        tmp_path / f"{split}_predictions_rescored.csv",
        split=split,
        checkpoint=checkpoint,
        architecture="convnext_tiny",
        class_names=CLASSES,
        temperature=temperature,
    )


# --- The producer and the consumer agree ---------------------------------------


def test_producer_and_consumer_share_schema_hash_and_preprocessing() -> None:
    assert rescore.EVIDENCE_SUFFIX == prediction_evidence.SIDECAR_SUFFIX
    assert rescore.EVIDENCE_SCHEMA == prediction_evidence.EVIDENCE_SCHEMA
    assert rescore.EVIDENCE_SCHEMA_VERSION == prediction_evidence.EVIDENCE_SCHEMA_VERSION
    assert rescore.PREPROCESSING == prediction_evidence.SERVED_PREPROCESSING
    for names in (CLASSES, CLASSES[::-1], ["crème_brûlée", "pho"]):
        assert rescore.class_names_sha256(names) == artifacts.class_names_sha256(names)
    assert artifacts.class_names_sha256(CLASSES) != artifacts.class_names_sha256(CLASSES[::-1])


def test_preprocessing_record_matches_the_rescorer_transform() -> None:
    resize, _, normalize = rescore.EVAL_TRANSFORMS.transforms
    assert list(resize.size) == prediction_evidence.SERVED_PREPROCESSING["resize"]
    assert resize.interpolation.value == prediction_evidence.SERVED_PREPROCESSING["interpolation"]
    assert list(normalize.mean) == prediction_evidence.SERVED_PREPROCESSING["normalize_mean"]
    assert list(normalize.std) == prediction_evidence.SERVED_PREPROCESSING["normalize_std"]


def test_preprocessing_record_matches_the_backend_transform(monkeypatch, tmp_path) -> None:
    """load_runtime()'s transform is the one SERVED_PREPROCESSING describes."""
    import torch
    from torch import nn

    from app.backend import inference

    monkeypatch.setattr(inference, "_RUNTIME", None)
    monkeypatch.setenv("FOODLENS_ARTIFACT_DIR", str(tmp_path))
    (tmp_path / "class_names.json").write_text(json.dumps(CLASSES))
    tiny = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(3, 101))
    torch.save(tiny.state_dict(), tmp_path / "resnet50_ft_v2_best.pth")
    monkeypatch.setattr(inference, "build_classifier_model", lambda *args: tiny)

    resize, _, normalize = inference.load_runtime()["transform"].transforms
    expected = prediction_evidence.SERVED_PREPROCESSING
    assert list(resize.size) == expected["resize"]
    assert resize.interpolation.value == expected["interpolation"]
    assert list(normalize.mean) == expected["normalize_mean"]
    assert list(normalize.std) == expected["normalize_std"]


# --- The sidecar the rescorer writes ---------------------------------------------


def test_sidecar_is_bound_to_the_csv_it_describes(tmp_path, checkpoint) -> None:
    csv_path = scored(tmp_path, checkpoint)
    record = prediction_evidence.read_sidecar(csv_path)

    assert prediction_evidence.sidecar_path(csv_path).name == (
        "val_predictions_rescored.csv.evidence.json"
    )
    assert record["predictions"]["sha256"] == sha256(csv_path)
    assert record["predictions"]["rows"] == 3
    assert record["checkpoint"]["sha256"] == sha256(checkpoint)
    assert record["architecture"] == "convnext_tiny"
    assert record["class_names"]["sha256"] == artifacts.class_names_sha256(CLASSES)
    assert record["temperature"] == {"value": 0.884, "source": "test fixture"}
    assert record["producer"]["sha256"] == sha256(Path(rescore.__file__))
    assert record["self_check"]["status"] == "skipped"


def _evidence(tmp_path: Path) -> dict:
    return rescore.build_evidence_record(
        split="val",
        output_path=tmp_path / "out.csv",
        rows=1,
        checkpoint_path=tmp_path / "out.csv",
        checkpoint_sha256="0" * 64,
        arch="resnet50",
        class_names=CLASSES,
        class_names_source=tmp_path / "class_names.json",
        temperature=1.0,
        temperature_source="--temperature flag",
        achieved=(70.0, 80.0),
        recorded=(83.9, 95.78),
        recorded_source="val_metrics.csv",
    )


def test_failed_self_check_writes_neither_csv_nor_sidecar(tmp_path) -> None:
    frame = pd.DataFrame([{"pred_label": "pho"}])
    with pytest.raises(rescore.AccuracyMismatchError):
        rescore.write_predictions_if_accuracy_matches(
            frame, tmp_path / "out.csv", 70.0, 80.0, (83.9, 95.78), evidence=_evidence(tmp_path)
        )
    assert list(tmp_path.iterdir()) == []


def test_passed_self_check_writes_both_and_records_it(tmp_path) -> None:
    frame = pd.DataFrame([{"pred_label": "pho"}])
    rescore.write_predictions_if_accuracy_matches(
        frame, tmp_path / "out.csv", 83.9, 95.78, (83.9, 95.78), evidence=_evidence(tmp_path)
    )
    assert sorted(path.name for path in tmp_path.iterdir()) == ["out.csv", "out.csv.evidence.json"]
    record = json.loads((tmp_path / "out.csv.evidence.json").read_text())
    assert record["self_check"]["status"] == "passed"
    assert record["predictions"]["sha256"] == sha256(tmp_path / "out.csv")


def test_an_existing_sidecar_is_refused_without_overwrite(tmp_path) -> None:
    (tmp_path / "out.csv.evidence.json").write_text("{}")
    with pytest.raises(rescore.OutputAlreadyExistsError, match="evidence.json"):
        rescore.write_predictions_if_accuracy_matches(
            pd.DataFrame([{"a": 1}]), tmp_path / "out.csv", 1.0, 1.0, None,
            evidence=_evidence(tmp_path),
        )
    assert (tmp_path / "out.csv.evidence.json").read_text() == "{}"
    assert not (tmp_path / "out.csv").exists()


# --- Consumer checks ---------------------------------------------------------------


def test_a_csv_without_a_sidecar_has_no_evidence(tmp_path, checkpoint) -> None:
    csv_path = scored(tmp_path, checkpoint)
    prediction_evidence.sidecar_path(csv_path).unlink()
    assert prediction_evidence.predictions_evidence(csv_path, sha256(csv_path)) is None


def test_a_sidecar_for_other_csv_bytes_is_rejected(tmp_path, checkpoint) -> None:
    csv_path = scored(tmp_path, checkpoint)
    csv_path.write_text(csv_path.read_text() + "\n")
    with pytest.raises(prediction_evidence.EvidenceError, match="belongs to other bytes"):
        prediction_evidence.predictions_evidence(csv_path, sha256(csv_path))


def test_mixed_temperatures_are_rejected(tmp_path, checkpoint) -> None:
    csv_path = scored(tmp_path, checkpoint)
    frame = pd.read_csv(csv_path)
    frame.loc[0, "temperature"] = 1.0
    frame.to_csv(csv_path, index=False)
    with pytest.raises(prediction_evidence.EvidenceError, match="mixes 2 temperature values"):
        prediction_evidence.single_temperature(csv_path)


def test_a_malformed_sidecar_is_rejected(tmp_path, checkpoint) -> None:
    csv_path = scored(tmp_path, checkpoint)
    path = prediction_evidence.sidecar_path(csv_path)
    record = json.loads(path.read_text())
    del record["checkpoint"]
    path.write_text(json.dumps(record))
    with pytest.raises(prediction_evidence.EvidenceError, match="checkpoint.sha256"):
        prediction_evidence.read_sidecar(csv_path)


def test_binding_mismatches_name_each_field(tmp_path, checkpoint) -> None:
    record = prediction_evidence.read_sidecar(scored(tmp_path, checkpoint))
    evidence = prediction_evidence.binding(record)
    model = prediction_evidence.model_binding(
        checkpoint_sha256=sha256(checkpoint),
        architecture="convnext_tiny",
        class_names=CLASSES,
        temperature=0.884,
    )
    assert prediction_evidence.binding_mismatches(evidence, model) == []
    other = {**model, "temperature": 0.884 + 1e-12, "architecture": "resnet50"}
    problems = prediction_evidence.binding_mismatches(evidence, other)
    assert [problem.split(":")[0] for problem in problems] == ["architecture", "temperature"]
