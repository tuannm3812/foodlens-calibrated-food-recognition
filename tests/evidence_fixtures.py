"""Shared helpers: re-scored prediction CSVs with real evidence sidecars.

The sidecars are written by the rescorer's own writer
(``kaggle/a3b_rescore/rescore_predictions.py``), not assembled by hand, so the
deploy and recalibration tests consume exactly the format the producer writes.
Not collected by pytest (no ``test_`` prefix); test modules in this directory
import it directly.
"""

from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import prediction_evidence  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "rescore_predictions_for_fixtures", ROOT / "kaggle" / "a3b_rescore" / "rescore_predictions.py"
)
rescore = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rescore)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_scored_split(
    path: Path,
    *,
    split: str,
    checkpoint: Path,
    architecture: str,
    class_names: list[str],
    temperature: float,
    marker: str = "",
) -> Path:
    """Write a small re-scored CSV and its evidence sidecar through the rescorer.

    ``checkpoint`` is hashed as it is now, as the rescorer hashes the bytes it
    loads. ``marker`` varies the rows so different runs get different CSVs.
    """
    rows = [
        rescore.build_prediction_row(
            path=f"/kaggle/input/{marker}{split}/{index}.jpg",
            true_label=class_names[index],
            top_labels=[class_names[(index + offset) % len(class_names)] for offset in range(5)],
            top_scores=[0.6, 0.2, 0.1, 0.05, 0.05],
            temperature=temperature,
        )
        for index in range(3)
    ]
    frame = pd.DataFrame(rows, columns=rescore.REQUIRED_OUTPUT_COLUMNS)
    names_file = path.parent / "class_names.json"
    evidence = rescore.build_evidence_record(
        split=split,
        output_path=path,
        rows=len(frame),
        checkpoint_path=checkpoint,
        checkpoint_sha256=sha256(checkpoint),
        arch=architecture,
        class_names=class_names,
        class_names_source=names_file,
        temperature=temperature,
        temperature_source="test fixture",
        achieved=(100.0, 100.0),
        recorded=None,
        recorded_source=None,
    )
    rescore.write_predictions_if_accuracy_matches(
        frame, path, 100.0, 100.0, None, overwrite=True, evidence=evidence
    )
    return path


def provenance_with_evidence(fit_csv: Path, eval_csv: Path) -> dict:
    """The ``predictions`` part of a provenance record, as recalibration writes it."""
    predictions = {}
    for split, csv_path in (("fit", fit_csv), ("eval", eval_csv)):
        digest = sha256(csv_path)
        entry = {"path": str(csv_path), "sha256": digest}
        evidence = prediction_evidence.predictions_evidence(csv_path, digest)
        if evidence is not None:
            entry["evidence"] = evidence
        predictions[split] = entry
    return {"schema_version": 3, "predictions": predictions}
