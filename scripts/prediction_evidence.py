"""Producer evidence for re-scored predictions, and the checks that bind a policy to it.

A decision policy is fitted to one model's temperature-scaled confidences. The
prediction CSVs it was fitted on are only meaningful together with what
produced them: the checkpoint's exact bytes, the architecture, the ordered class
names that map output indices to labels, the effective temperature and the eval
preprocessing. ``kaggle/a3b_rescore/rescore_predictions.py`` records those in a
sidecar, ``<predictions CSV>.evidence.json``, bound to the CSV by its SHA-256.

This module is the consumer side:

- ``scripts/recalibrate_decision_layer.py`` calls ``fit_eval_evidence()`` to
  verify each sidecar against the CSV actually read, check the CSV's
  ``temperature`` column, require the fit and eval evidence to describe the same
  model, and carry the evidence into ``derivation_provenance.json``.
- ``scripts/deploy_decision_policy.py`` calls ``policy_evidence()`` and
  ``binding_mismatches()`` to require that evidence to match the model being
  installed (promotion) or already served (policy-only deploy).

The rescorer writes its sidecar itself rather than importing this module: it is
a Kaggle ``code_file`` and must stay self-contained (see
``docs/0_coding_standards.md``). ``tests/test_prediction_evidence.py`` checks
that the two agree on the class-name hash and the preprocessing record.

Nothing here fabricates evidence. A CSV without a sidecar has none, and a policy
fitted on such CSVs cannot be deployed; ``REGENERATE_HINT`` says how to produce
evidence by re-scoring.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.backend.artifacts import class_names_sha256  # noqa: E402

SIDECAR_SUFFIX = ".evidence.json"
EVIDENCE_SCHEMA = "foodlens.prediction_evidence"
EVIDENCE_SCHEMA_VERSION = 1

# The eval transform both the rescorer and app/backend/inference.py's
# load_runtime() apply: Resize((224, 224)) with torchvision's default bilinear
# interpolation, ToTensor, then ImageNet normalisation. The rescorer records
# exactly this dict; tests/test_prediction_evidence.py checks it against both.
SERVED_PREPROCESSING: dict[str, Any] = {
    "id": "resize_224x224_bilinear+to_tensor+normalize_imagenet",
    "resize": [224, 224],
    "interpolation": "bilinear",
    "to_tensor": "RGB, float in [0, 1], channels first",
    "normalize_mean": [0.485, 0.456, 0.406],
    "normalize_std": [0.229, 0.224, 0.225],
}

# Temperatures are compared exactly. JSON round-trips a Python float exactly,
# and the rescorer reads calibration.json's value with float() like the
# deploy script does, so a correctly bound policy matches to the last bit; any
# difference means a different calibration. The documented tolerance is zero.
TEMPERATURE_TOLERANCE = 0.0

# What a sidecar binds, as (label, path into the record).
BINDING_FIELDS = (
    ("checkpoint_sha256", ("checkpoint", "sha256")),
    ("architecture", ("architecture",)),
    ("class_names_sha256", ("class_names", "sha256")),
    ("temperature", ("temperature", "value")),
    ("preprocessing", ("preprocessing",)),
)

REGENERATE_HINT = (
    "Regenerate evidence by re-scoring rather than writing it by hand: run "
    "kaggle/a3b_rescore/rescore_predictions.py --results-dir <model run> --arch "
    "<architecture> --split val (and again with --split test) with a new --output "
    "name, which writes <output>.evidence.json beside each CSV once its self-check "
    "passes; then recalibrate from the new CSVs with scripts/recalibrate_decision_layer.py "
    "--fit-predictions-file <val CSV> --eval-predictions-file <test CSV> --output-dir "
    "<new dir>, and deploy that run. See docs/8_runtime_contract.md, "
    "'Regenerating evidence for older runs'."
)


class EvidenceError(ValueError):
    """Prediction evidence is missing, malformed or inconsistent."""


def sha256_file(path: Path) -> str:
    """SHA-256 hex digest of a file's bytes, read in chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sidecar_path(predictions_csv: Path) -> Path:
    """Where the evidence for a predictions CSV lives: ``<csv>.evidence.json``."""
    return predictions_csv.with_name(predictions_csv.name + SIDECAR_SUFFIX)


def _field(record: Any, path: tuple[str, ...]) -> Any:
    value = record
    for key in path:
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    return value


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(char in "0123456789abcdef" for char in value)
    )


def validate_record(record: Any, where: str) -> dict[str, Any]:
    """Require a decoded sidecar to be a complete evidence record of a known schema.

    Raises:
        EvidenceError: Naming the first missing or malformed field.
    """
    if not isinstance(record, dict):
        raise EvidenceError(f"{where} must hold an evidence object.")
    if record.get("schema") != EVIDENCE_SCHEMA:
        raise EvidenceError(f"{where} is not a {EVIDENCE_SCHEMA} record.")
    if record.get("schema_version") != EVIDENCE_SCHEMA_VERSION:
        raise EvidenceError(
            f"{where} has schema_version {record.get('schema_version')!r}; "
            f"expected {EVIDENCE_SCHEMA_VERSION}."
        )
    for label, path in (
        ("predictions.sha256", ("predictions", "sha256")),
        ("checkpoint.sha256", ("checkpoint", "sha256")),
        ("class_names.sha256", ("class_names", "sha256")),
        ("producer.sha256", ("producer", "sha256")),
    ):
        if not _is_sha256(_field(record, path)):
            raise EvidenceError(f"{where}: {label} must be a SHA-256 hex digest.")
    architecture = record.get("architecture")
    if not isinstance(architecture, str) or not architecture:
        raise EvidenceError(f"{where}: architecture must be a non-empty string.")
    temperature = _field(record, ("temperature", "value"))
    if (
        isinstance(temperature, bool)
        or not isinstance(temperature, (int, float))
        or not math.isfinite(temperature)
        or temperature <= 0
    ):
        raise EvidenceError(f"{where}: temperature.value must be a finite positive number.")
    preprocessing = record.get("preprocessing")
    if not isinstance(preprocessing, dict) or not isinstance(preprocessing.get("id"), str):
        raise EvidenceError(f"{where}: preprocessing must be an object with an id.")
    return record


def read_sidecar(predictions_csv: Path) -> dict[str, Any] | None:
    """The validated evidence beside a predictions CSV, or None when there is none."""
    path = sidecar_path(predictions_csv)
    if not path.exists():
        return None
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise EvidenceError(f"{path} is not valid JSON: {exc}") from exc
    return validate_record(record, str(path))


def csv_temperatures(predictions_csv: Path) -> list[float] | None:
    """The distinct values of a CSV's ``temperature`` column, or None without one.

    Values are parsed with Python's ``float()``, which round-trips the repr the
    rescorer wrote exactly; pandas' default parser can be off by one ulp.
    """
    with predictions_csv.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or "temperature" not in reader.fieldnames:
            return None
        values: set[float] = set()
        for row_number, row in enumerate(reader, start=2):
            raw = row["temperature"]
            try:
                values.add(float(raw))
            except (TypeError, ValueError) as exc:
                raise EvidenceError(
                    f"{predictions_csv}: line {row_number} has a non-numeric temperature {raw!r}."
                ) from exc
    return sorted(values)


def single_temperature(predictions_csv: Path) -> float | None:
    """The CSV's one temperature value, or None without a ``temperature`` column.

    Raises:
        EvidenceError: If the column holds more than one value -- rows scaled by
            different temperatures are not one model's calibrated confidences.
    """
    values = csv_temperatures(predictions_csv)
    if values is None:
        return None
    if len(values) != 1:
        shown = ", ".join(repr(value) for value in values[:5])
        raise EvidenceError(
            f"{predictions_csv} mixes {len(values)} temperature values ({shown}); one "
            "predictions file must be scaled by one temperature. Re-score it."
        )
    return values[0]


def predictions_evidence(predictions_csv: Path, csv_sha256: str) -> dict[str, Any] | None:
    """Verify one predictions CSV's evidence; return its provenance entry or None.

    Args:
        predictions_csv: The CSV recalibration read.
        csv_sha256: SHA-256 of the bytes recalibration read.

    Returns:
        ``{"path", "sha256", "record"}`` for the sidecar, or None when the CSV
        has no sidecar. Either way a ``temperature`` column, when present, must
        be single-valued, and equal to the sidecar's temperature when there is
        one.

    Raises:
        EvidenceError: On a malformed sidecar, a sidecar bound to different CSV
            bytes, or an inconsistent ``temperature`` column.
    """
    column_temperature = single_temperature(predictions_csv)
    record = read_sidecar(predictions_csv)
    if record is None:
        return None
    path = sidecar_path(predictions_csv)
    if record["predictions"]["sha256"] != csv_sha256:
        raise EvidenceError(
            f"{path} records predictions sha256 {record['predictions']['sha256']}, but "
            f"{predictions_csv} hashes to {csv_sha256}: the evidence belongs to other bytes. "
            + REGENERATE_HINT
        )
    temperature = record["temperature"]["value"]
    if column_temperature is not None and column_temperature != temperature:
        raise EvidenceError(
            f"{predictions_csv} was scaled by temperature {column_temperature!r}, but its "
            f"evidence {path} records {temperature!r}."
        )
    return {"path": str(path), "sha256": sha256_file(path), "record": record}


def binding(record: dict[str, Any]) -> dict[str, Any]:
    """The fields a sidecar binds: checkpoint, architecture, labels, temperature, preprocessing."""
    return {label: _field(record, path) for label, path in BINDING_FIELDS}


def _differences(first: dict[str, Any], second: dict[str, Any]) -> list[str]:
    return [label for label, _ in BINDING_FIELDS if first[label] != second[label]]


def fit_eval_evidence(
    fit_csv: Path, fit_sha256: str, eval_csv: Path, eval_sha256: str
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """Verify both splits' evidence and require it to describe one model.

    Returns:
        ``(fit_entry, eval_entry)`` for the provenance record; both None when
        neither CSV has a sidecar.

    Raises:
        EvidenceError: If one split has evidence and the other does not, if the
            two disagree on checkpoint, architecture, class order, temperature or
            preprocessing, or (without sidecars) if their temperature columns
            disagree. A policy fitted on one model's val predictions and scored
            on another's test predictions describes neither.
    """
    fit_entry = predictions_evidence(fit_csv, fit_sha256)
    eval_entry = predictions_evidence(eval_csv, eval_sha256)
    if (fit_entry is None) != (eval_entry is None):
        missing, present = (fit_csv, eval_csv) if fit_entry is None else (eval_csv, fit_csv)
        raise EvidenceError(
            f"{present} has producer evidence but {missing} does not, so nothing shows the "
            "two splits came from the same model. " + REGENERATE_HINT
        )
    if fit_entry is not None and eval_entry is not None:
        fit_binding, eval_binding = binding(fit_entry["record"]), binding(eval_entry["record"])
        differing = _differences(fit_binding, eval_binding)
        if differing:
            detail = "; ".join(
                f"{label}: fit {fit_binding[label]!r}, eval {eval_binding[label]!r}"
                for label in differing
            )
            raise EvidenceError(
                "The fit and eval predictions were produced by different models or "
                f"calibrations ({detail}). A policy must be fitted and scored on one "
                "model's predictions."
            )
        return fit_entry, eval_entry
    fit_temperature = single_temperature(fit_csv)
    eval_temperature = single_temperature(eval_csv)
    if None not in (fit_temperature, eval_temperature) and fit_temperature != eval_temperature:
        raise EvidenceError(
            f"{fit_csv} was scaled by temperature {fit_temperature!r} but {eval_csv} by "
            f"{eval_temperature!r}; the fit and eval predictions must share one calibration."
        )
    return None, None


def policy_evidence(provenance: Any, provenance_path: Path) -> dict[str, Any]:
    """The producer evidence a policy run recorded, as one binding.

    Re-checks what recalibration checked, against the provenance as it is now:
    each split's evidence is present and complete, is bound to the prediction
    hash the provenance records, and the two splits agree.

    Raises:
        EvidenceError: If the policy run records no evidence (every run before
            evidence existed), with instructions for regenerating it.
    """
    predictions = provenance.get("predictions") if isinstance(provenance, dict) else None
    bindings = {}
    for split in ("fit", "eval"):
        entry = predictions.get(split) if isinstance(predictions, dict) else None
        evidence = entry.get("evidence") if isinstance(entry, dict) else None
        if not isinstance(evidence, dict):
            raise EvidenceError(
                f"{provenance_path} records no producer evidence for its {split} predictions, "
                "so nothing shows which checkpoint, class order and temperature produced the "
                "confidences this policy was fitted to (policies recalibrated before "
                "2026-10-11 have none). Directory membership is not evidence. "
                + REGENERATE_HINT
            )
        record = validate_record(evidence.get("record"), f"{provenance_path} ({split} evidence)")
        if record["predictions"]["sha256"] != entry.get("sha256"):
            raise EvidenceError(
                f"{provenance_path}: the {split} evidence is bound to predictions sha256 "
                f"{record['predictions']['sha256']}, not the recorded {entry.get('sha256')}."
            )
        bindings[split] = binding(record)
    differing = _differences(bindings["fit"], bindings["eval"])
    if differing:
        raise EvidenceError(
            f"{provenance_path}: the fit and eval evidence disagree on {', '.join(differing)}."
        )
    return bindings["fit"]


def model_binding(
    *,
    checkpoint_sha256: str,
    architecture: str,
    class_names: list[str],
    temperature: float,
) -> dict[str, Any]:
    """The binding a model being installed or served would need its policy's evidence to have."""
    return {
        "checkpoint_sha256": checkpoint_sha256,
        "architecture": architecture,
        "class_names_sha256": class_names_sha256(class_names),
        "temperature": float(temperature),
        "preprocessing": SERVED_PREPROCESSING,
    }


def binding_mismatches(evidence: dict[str, Any], expected: dict[str, Any]) -> list[str]:
    """Each bound field where a policy's evidence differs from a model, with both values."""
    problems = []
    for label, _ in BINDING_FIELDS:
        produced, wanted = evidence[label], expected[label]
        if label == "temperature":
            same = (
                isinstance(produced, (int, float))
                and abs(float(produced) - float(wanted)) <= TEMPERATURE_TOLERANCE
            )
        else:
            same = produced == wanted
        if not same:
            problems.append(f"{label}: policy evidence {produced!r}, model {wanted!r}")
    return problems
