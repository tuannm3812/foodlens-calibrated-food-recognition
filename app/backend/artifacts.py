"""Artifact file loading and validation for the FoodLens classifier.

Reads calibration, decision-policy, and class-list artifacts from a resolved
artifact directory that callers pass in. Must not import inference:
artifact_dir_path() (which resolves that directory) stays in inference.py
because it derives the repo root from Path(__file__), and a test patches
inference.__file__ to exercise it.
"""

import json
import logging
from pathlib import Path
from typing import Any

from .decision import DEFAULT_HARD_CLASSES, DEFAULT_POLICY

logger = logging.getLogger(__name__)

TEMPERATURE = 0.958111
REQUIRED_CLASSIFIER_ARTIFACTS = (
    "resnet50_ft_v2_best.pth",
    "class_names.json",
)


def read_json(path: Path, default: Any, *, tolerate_invalid: bool = False) -> Any:
    """Read a JSON artifact when available.

    A missing file always returns ``default``. A present-but-unreadable file
    (malformed JSON, undecodable bytes, or an OS-level read failure) raises
    by default -- callers with no safe default should see that failure
    immediately rather than silently degrading. Pass ``tolerate_invalid=True``
    to instead log a warning naming the file and the error, and return
    ``default``.
    """
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, UnicodeDecodeError, OSError) as error:
        if not tolerate_invalid:
            raise
        logger.warning(
            "Ignoring unreadable artifact %s (%s: %s); using default value.",
            path,
            type(error).__name__,
            error,
        )
        return default


def read_temperature(artifact_dir: Path) -> float:
    """Read the calibrated temperature artifact when available."""
    calibration = read_json(artifact_dir / "calibration.json", {}, tolerate_invalid=True)
    return float(calibration.get("temperature", TEMPERATURE))


def read_policy(artifact_dir: Path) -> dict[str, float]:
    """Read decision thresholds when available."""
    policy = read_json(
        artifact_dir / "decision_policy.json", DEFAULT_POLICY, tolerate_invalid=True
    )
    return {
        "auto_confidence": float(policy.get("auto_confidence", DEFAULT_POLICY["auto_confidence"])),
        "suggest_confidence": float(
            policy.get("suggest_confidence", DEFAULT_POLICY["suggest_confidence"])
        ),
        "margin_threshold": float(
            policy.get("margin_threshold", DEFAULT_POLICY["margin_threshold"])
        ),
    }


def read_hard_classes(artifact_dir: Path) -> set[str]:
    """Read hard classes when available."""
    hard_classes = read_json(
        artifact_dir / "hard_classes.json", list(DEFAULT_HARD_CLASSES), tolerate_invalid=True
    )
    return set(hard_classes)


def read_confusion_pairs(artifact_dir: Path) -> set[tuple[str, str]]:
    """Read known confusion pairs when available."""
    raw_pairs = read_json(artifact_dir / "confusion_pairs.json", [], tolerate_invalid=True)
    pairs = set()
    for pair in raw_pairs:
        if isinstance(pair, dict) and {"actual", "predicted"}.issubset(pair):
            pairs.add((str(pair["actual"]), str(pair["predicted"])))
        elif isinstance(pair, (list, tuple)) and len(pair) >= 2:
            pairs.add((str(pair[0]), str(pair[1])))
    return pairs


def artifact_file_status(path: Path) -> dict[str, Any]:
    """Return status details for one artifact file."""
    return {
        "path": str(path),
        "exists": path.exists(),
        "size_bytes": path.stat().st_size if path.exists() else 0,
    }


def classifier_artifacts_ready(artifact_dir: Path) -> bool:
    """Return whether the required classifier artifacts exist in a directory."""
    return all(
        (artifact_dir / artifact_name).exists() for artifact_name in REQUIRED_CLASSIFIER_ARTIFACTS
    )
