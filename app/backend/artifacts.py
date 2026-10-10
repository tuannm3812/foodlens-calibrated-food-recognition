"""Artifact file loading and validation for the FoodLens classifier.

Reads the model manifest, calibration, decision-policy, and class-list
artifacts from a resolved artifact directory that callers pass in. Must not import inference:
artifact_dir_path() (which resolves that directory) stays in inference.py
because it derives the repo root from Path(__file__), and a test patches
inference.__file__ to exercise it.
"""

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

from .decision import DEFAULT_HARD_CLASSES, DEFAULT_POLICY

logger = logging.getLogger(__name__)

TEMPERATURE = 0.958111
# The required files when no model.json manifest is present -- the legacy,
# ResNet50-only deployment. With a manifest, the checkpoint it names replaces
# the first entry; see required_classifier_artifacts().
REQUIRED_CLASSIFIER_ARTIFACTS = (
    "resnet50_ft_v2_best.pth",
    "class_names.json",
)

MODEL_MANIFEST = "model.json"
SUPPORTED_ARCHITECTURES = ("resnet50", "convnext_tiny")
# What an artifact directory without model.json serves. Exactly the
# pre-manifest behaviour: ResNet50 FT-V2 from resnet50_ft_v2_best.pth.
LEGACY_MODEL = {
    "architecture": "resnet50",
    "checkpoint": "resnet50_ft_v2_best.pth",
    "model_name": "resnet50_ft_v2",
}
MANIFEST_REQUIRED_KEYS = ("architecture", "checkpoint", "model_name")
# Informational: the run directory the checkpoint came from. The deploy script
# writes it on promotion and uses it to keep a later policy-only deploy from
# pairing this model with another model's policy.
MANIFEST_OPTIONAL_KEYS = ("model_run",)


class ModelManifestError(ValueError):
    """model.json exists but cannot be read or names no servable model.

    Raised, never defaulted: a bad manifest must not silently serve ResNet50.
    Inside load_runtime() it becomes the classifier_load_error fallback, like
    an unreadable class_names.json.
    """


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


def validate_model_manifest(raw: Any, source: Path) -> dict[str, str]:
    """Validate a decoded model.json; return its fields or raise ModelManifestError."""
    if not isinstance(raw, dict):
        raise ModelManifestError(f"{source} must hold a JSON object, found {type(raw).__name__}.")
    missing = [key for key in MANIFEST_REQUIRED_KEYS if key not in raw]
    if missing:
        raise ModelManifestError(f"{source} is missing keys: {', '.join(missing)}.")
    unknown = sorted(set(raw) - set(MANIFEST_REQUIRED_KEYS) - set(MANIFEST_OPTIONAL_KEYS))
    if unknown:
        raise ModelManifestError(f"{source} has unknown keys: {', '.join(unknown)}.")
    for key, value in raw.items():
        if not isinstance(value, str) or not value.strip():
            raise ModelManifestError(f"{source}: {key} must be a non-empty string.")
    if raw["architecture"] not in SUPPORTED_ARCHITECTURES:
        raise ModelManifestError(
            f"{source}: unknown architecture {raw['architecture']!r}; expected one of "
            f"{', '.join(SUPPORTED_ARCHITECTURES)}."
        )
    checkpoint = raw["checkpoint"]
    if Path(checkpoint).name != checkpoint or checkpoint in {".", ".."}:
        raise ModelManifestError(
            f"{source}: checkpoint must be a file name inside the artifact directory, "
            f"found {checkpoint!r}."
        )
    return dict(raw)


def read_model_manifest(artifact_dir: Path) -> dict[str, str]:
    """Return the model an artifact directory serves.

    Without ``model.json`` this is ``LEGACY_MODEL``, exactly the pre-manifest
    behaviour. A present manifest is strict: unreadable JSON, a missing or
    unknown key, or an unsupported architecture raises ``ModelManifestError``
    rather than falling back to ResNet50.
    """
    path = artifact_dir / MODEL_MANIFEST
    if not path.exists():
        return dict(LEGACY_MODEL)
    try:
        raw = read_json(path, None)
    except (json.JSONDecodeError, UnicodeDecodeError, OSError) as error:
        raise ModelManifestError(
            f"{path} is unreadable ({type(error).__name__}: {error})."
        ) from error
    return validate_model_manifest(raw, path)


def manifest_source(artifact_dir: Path) -> str:
    """Name where the model identity comes from: the manifest or the legacy default."""
    return MODEL_MANIFEST if (artifact_dir / MODEL_MANIFEST).exists() else "legacy_default"


def classifier_checkpoint_name(artifact_dir: Path) -> str | None:
    """The checkpoint file the directory serves, or None if model.json is invalid."""
    try:
        return read_model_manifest(artifact_dir)["checkpoint"]
    except ModelManifestError:
        return None


def checkpoint_sha256(data: bytes) -> str:
    """SHA-256 of checkpoint bytes. load_runtime() calls it once, on the bytes it loads."""
    return hashlib.sha256(data).hexdigest()


def class_names_sha256(class_names: list[str]) -> str:
    """SHA-256 of an ordered class-name list, the order that maps output indices to labels.

    The canonical form is the list's compact JSON (``separators=(",", ":")``,
    ASCII-escaped) in UTF-8, so a reordered list hashes differently. The deploy
    script, the evidence sidecar the rescorer writes and ``load_runtime()`` all
    hash with this form.
    """
    canonical = json.dumps(list(class_names), ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def classifier_artifacts_ready(artifact_dir: Path) -> bool:
    """Return whether the required classifier artifacts exist in a directory.

    The checkpoint is the one ``model.json`` names, or ``resnet50_ft_v2_best.pth``
    without a manifest. A present but invalid manifest counts as present, so
    ``load_runtime()`` runs and fails into ``classifier_load_error`` -- reporting
    ``missing_artifacts`` instead would hide the bad manifest.
    """
    if not (artifact_dir / "class_names.json").exists():
        return False
    checkpoint = classifier_checkpoint_name(artifact_dir)
    return checkpoint is None or (artifact_dir / checkpoint).exists()


def model_status(runtime: dict[str, Any] | None, artifact_dir: Path) -> dict[str, Any]:
    """Describe the model a process serves, for the /runtime/status ``model`` block.

    Once the runtime has loaded, report the identity cached at load time,
    including the checkpoint SHA-256 hashed then -- never re-hashed here. Before
    that, report what the artifact files would load, without hashing the
    checkpoint.
    """
    if runtime is not None:
        identity = runtime.get("model_identity")
        if identity is None:
            return {"source": "loaded_runtime", "error": "the loaded runtime has no model identity"}
        return {"source": "loaded_runtime", **identity}
    source = manifest_source(artifact_dir)
    try:
        manifest = read_model_manifest(artifact_dir)
    except ModelManifestError as error:
        return {"source": "artifact_files", "manifest": source, "error": str(error)}
    return {
        "source": "artifact_files",
        "manifest": source,
        **manifest,
        "checkpoint_sha256": None,
        "note": "not loaded yet: the checkpoint is hashed once, when the runtime loads",
    }
