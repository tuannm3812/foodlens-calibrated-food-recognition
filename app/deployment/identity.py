"""What a model and a decision policy are, and the checks that bind them.

Manifest interpretation, checkpoint fit, class-order and temperature
validation, decision-layer policy validation, the adapters to the backend's own
readers, and producer-evidence comparison. Every operation uses these; none
re-implements them. This module checks; it never decides which model and which
policy belong together -- the operations do.

Operations call these functions through the module (``identity.<name>``) so a
test that patches ``verify_through_backend``, ``verify_model_files`` or
``build_model`` here reaches every caller.
"""

from __future__ import annotations

import math
import os
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .errors import DeployError
from .paths import MANIFEST_FILE, REPO_ROOT, display, from_display, read_json, sha256

# Producer evidence is read by scripts/prediction_evidence.py. It stays a
# script-level module: recalibration (scripts/recalibrate_decision_layer.py)
# uses it too, and tests/test_prediction_evidence.py pins the Kaggle rescorer's
# self-contained copy of the evidence schema to it.
_SCRIPTS = str(REPO_ROOT / "scripts")
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)
import prediction_evidence  # noqa: E402

POLICY_KEYS = ("auto_confidence", "suggest_confidence", "margin_threshold")
CLASS_COUNT = 101
# What identifies a served model, as /runtime/status reports it.
IDENTITY_KEYS = ("architecture", "model_name", "checkpoint_sha256")


# --- Decision-layer policy -----------------------------------------------------


def runtime_policy(raw: Any, source: Path) -> dict[str, float]:
    """Convert a recalibration policy record into the runtime threshold dict.

    Args:
        raw: Decoded ``decision_policy.json`` -- a dict, or a one-element list
            holding one, as the recalibration script writes it.
        source: Path the policy came from, for error messages.

    Returns:
        A dict with exactly the three thresholds the backend reads.

    Raises:
        DeployError: If the record is ambiguous, incomplete, or out of range.
    """
    if isinstance(raw, list):
        if len(raw) != 1:
            raise DeployError(f"{source} holds {len(raw)} policy records; expected exactly one.")
        raw = raw[0]
    if not isinstance(raw, dict):
        raise DeployError(f"{source} must hold a policy object, found {type(raw).__name__}.")
    missing = [key for key in POLICY_KEYS if key not in raw]
    if missing:
        raise DeployError(f"{source} is missing policy keys: {', '.join(missing)}.")

    policy: dict[str, float] = {}
    for key in POLICY_KEYS:
        value = raw[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise DeployError(f"{source}: {key} must be a number, found {value!r}.")
        value = float(value)
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise DeployError(f"{source}: {key} must be a finite value in [0, 1], found {value}.")
        policy[key] = value
    if policy["suggest_confidence"] > policy["auto_confidence"]:
        raise DeployError(
            f"{source}: suggest_confidence ({policy['suggest_confidence']}) exceeds "
            f"auto_confidence ({policy['auto_confidence']}), so no prediction could suggest."
        )
    return policy


def validated_hard_classes(raw: Any, source: Path) -> list[str]:
    """Validate a hard-class list: non-empty, unique, non-empty strings."""
    if not isinstance(raw, list) or not raw:
        raise DeployError(f"{source} must be a non-empty JSON list of class names.")
    names = []
    for item in raw:
        if not isinstance(item, str) or not item.strip():
            raise DeployError(f"{source} contains an invalid class name: {item!r}.")
        names.append(item.strip())
    if len(set(names)) != len(names):
        raise DeployError(f"{source} contains duplicate class names.")
    return names


def validated_confusion_pairs(raw: Any, source: Path) -> list[dict[str, str]]:
    """Validate confusion pairs as a non-empty list of {actual, predicted} records."""
    if not isinstance(raw, list) or not raw:
        raise DeployError(f"{source} must be a non-empty JSON list of pair records.")
    pairs = []
    for item in raw:
        if not isinstance(item, dict) or not {"actual", "predicted"} <= set(item):
            raise DeployError(f"{source} contains a malformed pair record: {item!r}.")
        actual, predicted = item["actual"], item["predicted"]
        if not all(isinstance(v, str) and v.strip() for v in (actual, predicted)):
            raise DeployError(f"{source} contains a pair with an empty label: {item!r}.")
        pairs.append({"actual": actual.strip(), "predicted": predicted.strip()})
    if len({(p["actual"], p["predicted"]) for p in pairs}) != len(pairs):
        raise DeployError(f"{source} contains duplicate confusion pairs.")
    return pairs


def read_policy_files(directory: Path) -> tuple[dict[str, float], list[str], list[dict[str, str]]]:
    """Read and validate a directory's three decision-layer files."""
    policy = runtime_policy(read_json(directory / "decision_policy.json"), directory)
    hard = validated_hard_classes(read_json(directory / "hard_classes.json"), directory)
    pairs = validated_confusion_pairs(read_json(directory / "confusion_pairs.json"), directory)
    return policy, hard, pairs


def decision_layer_labels(hard: list[str], pairs: list[dict[str, str]]) -> set[str]:
    """Every class label a decision layer names: hard classes and both sides of each pair."""
    return set(hard) | {p["actual"] for p in pairs} | {p["predicted"] for p in pairs}


def check_class_coverage(target: Path, hard: list[str], pairs: list[dict[str, str]]) -> None:
    """Reject labels the served model cannot predict -- they would never match."""
    names_path = target / "class_names.json"
    if not names_path.exists():
        raise DeployError(f"{names_path} is missing; cannot check labels against the model.")
    known = set(read_json(names_path))
    unknown = sorted(decision_layer_labels(hard, pairs) - known)
    if unknown:
        raise DeployError(
            f"Labels not in the served model's class_names.json: {', '.join(unknown[:5])}."
        )


# --- Class order and calibration -----------------------------------------------


def validated_class_names(raw: Any, source: Path) -> list[str]:
    """Validate an ordered class list as the backend needs it: 101 unique names."""
    if (
        not isinstance(raw, list)
        or len(raw) != CLASS_COUNT
        or not all(isinstance(name, str) and name.strip() for name in raw)
        or len(set(raw)) != len(raw)
    ):
        raise DeployError(f"{source} must list {CLASS_COUNT} unique, non-empty class names.")
    return raw


def validated_temperature(raw: Any, source: Path) -> float:
    """Validate a calibration record's temperature: a finite, positive number."""
    value = raw.get("temperature") if isinstance(raw, dict) else None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DeployError(f"{source} must hold a numeric 'temperature'.")
    if not math.isfinite(value) or value <= 0:
        raise DeployError(f"{source}: temperature must be finite and positive, found {value}.")
    return float(value)


def target_temperature(target: Path) -> float:
    """The temperature a target's validated ``calibration.json`` holds.

    The backend reads calibration tolerantly and silently serves a built-in
    default when the file is missing or unreadable, so a target without a
    valid one has no calibration a policy or a live service can be checked
    against.
    """
    path = target / "calibration.json"
    if not path.exists():
        raise DeployError(
            f"{display(path)} is missing: the backend would silently serve its built-in "
            "default temperature, so there is no calibration to bind or verify against."
        )
    return validated_temperature(read_json(path), path)


def target_class_names(target: Path) -> list[str]:
    """A target's ordered ``class_names.json``."""
    path = target / "class_names.json"
    names = read_json(path)
    if not isinstance(names, list) or not all(isinstance(name, str) for name in names):
        raise DeployError(f"{display(path)} must be a JSON list of class names.")
    return names


def check_class_names_match(target: Path, class_names: list[str]) -> str:
    """Require the model's class names to equal the target's, in content and order.

    The class order maps each output index to a label: a reordered list would
    silently mislabel every prediction, so it is refused, not reconciled.
    """
    path = target / "class_names.json"
    if not path.exists():
        raise DeployError(f"{display(path)} is missing; cannot compare the model's class names.")
    current = read_json(path)
    if current == class_names:
        return "identical in content and order to the target's"
    same_set = isinstance(current, list) and sorted(map(str, current)) == sorted(class_names)
    detail = (
        "the same classes in a different order, which would mislabel every prediction"
        if same_set
        else "a different set of classes"
    )
    raise DeployError(
        f"The model run's class_names.json does not match {display(path)}: {detail}. "
        "Refusing to deploy."
    )


def target_calibration(target: Path) -> dict[str, Any]:
    """The calibration a target holds: its validated temperature and class-order hash."""
    return {
        "temperature": target_temperature(target),
        "class_names_sha256": prediction_evidence.class_names_sha256(target_class_names(target)),
    }


# --- Adapters to the backend's own readers -------------------------------------


def backend_modules() -> tuple[Any, Any]:
    """Import the backend inference module and the fingerprint function."""
    from app.backend import inference
    from app.backend.policy_fingerprint import decision_layer_fingerprint

    return inference, decision_layer_fingerprint


def backend_artifacts() -> Any:
    """Import the backend's artifact readers (manifest, temperature)."""
    backend_modules()
    from app.backend import artifacts

    return artifacts


def read_manifest(directory: Path) -> dict[str, str]:
    """The model a directory serves, through the backend's own strict reader."""
    artifacts = backend_artifacts()
    try:
        return artifacts.read_model_manifest(directory)
    except artifacts.ModelManifestError as exc:
        raise DeployError(str(exc)) from exc


def validate_manifest(raw: Any, source: Path) -> dict[str, str]:
    """Validate decoded ``model.json`` fields with the backend's own validator."""
    artifacts = backend_artifacts()
    try:
        return artifacts.validate_model_manifest(raw, source)
    except artifacts.ModelManifestError as exc:
        raise DeployError(str(exc)) from exc


def legacy_manifest() -> dict[str, str]:
    """The model a directory without ``model.json`` serves: the legacy ResNet50 default."""
    return dict(backend_artifacts().LEGACY_MODEL)


def decision_layer_fingerprint(
    policy: dict[str, float], hard: list[str], pairs: list[dict[str, str]]
) -> str:
    """The backend's fingerprint of a decision layer."""
    _, fingerprint = backend_modules()
    return fingerprint(policy, hard, pairs)


def read_through_backend(directory: Path) -> tuple[dict[str, float], set[str], set[Any], str]:
    """Read a directory's decision layer through the backend's own readers.

    Returns:
        The policy, hard classes and confusion pairs exactly as ``load_runtime()``
        would cache them, plus their decision-layer fingerprint.

    Raises:
        DeployError: If the backend resolves a different artifact directory.
    """
    inference, fingerprint = backend_modules()
    previous = os.environ.get("FOODLENS_ARTIFACT_DIR")
    os.environ["FOODLENS_ARTIFACT_DIR"] = str(directory)
    try:
        policy = inference.read_policy()
        hard = inference.read_hard_classes()
        pairs = inference.read_confusion_pairs()
        resolved = inference.artifact_dir_path()
    finally:
        if previous is None:
            os.environ.pop("FOODLENS_ARTIFACT_DIR", None)
        else:
            os.environ["FOODLENS_ARTIFACT_DIR"] = previous
    if resolved.resolve() != directory.resolve():
        raise DeployError(f"Backend resolved artifacts to {resolved}, not {directory}.")
    return policy, hard, pairs, fingerprint(policy, hard, pairs)


def verify_through_backend(
    target: Path, policy: dict[str, float], hard: list[str], pairs: list[dict[str, str]]
) -> str:
    """Load the deployed files through the backend's own readers and compare.

    This is the check that would have caught the list-vs-dict mismatch: it does
    not trust the files, it trusts what the runtime reads from them. It reads
    them freshly in *this* process, so it says nothing about what an
    already-running API process has cached -- that is ``--verify-live``'s job.

    Returns:
        The decision-layer fingerprint of what the backend read.
    """
    observed_policy, observed_hard, observed_pairs, fingerprint = read_through_backend(target)
    if observed_policy != policy:
        raise DeployError(f"Backend read policy {observed_policy}, expected {policy}.")
    if observed_hard != set(hard):
        raise DeployError("Backend read a different hard-class set than was deployed.")
    expected_pairs = {(p["actual"], p["predicted"]) for p in pairs}
    if observed_pairs != expected_pairs:
        raise DeployError("Backend read a different confusion-pair set than was deployed.")
    return fingerprint


def verify_model_files(
    directory: Path, manifest: Mapping[str, str], temperature: float, class_names: list[str]
) -> None:
    """Read a directory's model files back as the backend would, and compare.

    The backend reads calibration tolerantly (an unreadable file silently means
    the default temperature), so the temperature it reads is compared with the
    one deployed rather than trusted.
    """
    observed = read_manifest(directory)
    if observed != dict(manifest):
        raise DeployError(f"Backend read model manifest {observed}, expected {dict(manifest)}.")
    if not (directory / manifest["checkpoint"]).is_file():
        raise DeployError(f"{display(directory / manifest['checkpoint'])} is missing.")
    observed_temperature = backend_artifacts().read_temperature(directory)
    if observed_temperature != temperature:
        raise DeployError(
            f"Backend read temperature {observed_temperature}, expected {temperature}."
        )
    if read_json(directory / "class_names.json") != class_names:
        raise DeployError("Backend would read different class names than were deployed.")


# --- Checkpoints ---------------------------------------------------------------


def build_model(architecture: str) -> Any:
    """Build the backend's untrained model for an architecture (tests replace this)."""
    try:
        from torch import nn
        from torchvision import models
    except ImportError as exc:
        raise DeployError("Checking a checkpoint needs torch and torchvision.") from exc
    backend_modules()
    from app.backend.classifier import build_classifier_model

    try:
        return build_classifier_model(models, nn, architecture)
    except ValueError as exc:
        raise DeployError(str(exc)) from exc


def check_checkpoint_fits(path: Path, architecture: str, label: str | None = None) -> str:
    """Require a checkpoint to load into an architecture with no missing or extra keys.

    Args:
        path: The checkpoint file to load.
        architecture: The architecture it must fit.
        label: How to name the file in errors (default: its path).

    Returns:
        A one-line statement of what was checked, for the deployment record.

    Raises:
        DeployError: If the file is not a state dict, or any key is missing,
            unexpected or of the wrong shape.
    """
    try:
        import torch
    except ImportError as exc:
        raise DeployError("Checking a checkpoint needs torch.") from exc
    name = label or display(path)
    model = build_model(architecture)
    try:
        state = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise DeployError(f"{name} is not a loadable checkpoint: {exc}") from exc
    if not isinstance(state, Mapping):
        raise DeployError(f"{name} holds {type(state).__name__}, not a state dict.")
    try:
        result = model.load_state_dict(state, strict=False)
    except RuntimeError as exc:
        detail = " ".join(str(exc).split())[:300]
        raise DeployError(f"{name} does not fit {architecture}: {detail}") from exc
    missing, unexpected = list(result.missing_keys), list(result.unexpected_keys)
    if missing or unexpected:
        raise DeployError(
            f"{name} does not fit {architecture}: {len(missing)} missing keys "
            f"(e.g. {missing[:3]}) and {len(unexpected)} unexpected keys (e.g. {unexpected[:3]})."
        )
    return f"loads into {architecture} with 0 missing and 0 unexpected keys"


# --- The served model ----------------------------------------------------------


def served_model_run(target: Path) -> Path | None:
    """The model run a target's ``model.json`` names, or None without a manifest.

    Raises:
        DeployError: If a manifest exists but is invalid or names no model run,
            because then no policy can be shown to belong to the served model.
    """
    if not (target / MANIFEST_FILE).exists():
        return None
    manifest = read_manifest(target)
    if "model_run" not in manifest:
        raise DeployError(
            f"{display(target / MANIFEST_FILE)} names no model_run, so no policy can be shown "
            "to belong to the served model. Promote the model with --model-run instead."
        )
    return from_display(manifest["model_run"])


def served_model(target: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """The model a target serves -- ``model.json`` or the legacy ResNet50 default.

    Returns:
        ``(identity, binding)``: the served model for the record, with the
        actual checkpoint bytes' SHA-256, and its
        ``prediction_evidence.model_binding()``.
    """
    manifest = read_manifest(target)
    checkpoint = target / manifest["checkpoint"]
    if not checkpoint.is_file():
        raise DeployError(
            f"{display(checkpoint)}, the checkpoint the target serves, is missing; no policy "
            "can be shown to belong to it."
        )
    class_names = target_class_names(target)
    binding = prediction_evidence.model_binding(
        checkpoint_sha256=sha256(checkpoint),
        architecture=manifest["architecture"],
        class_names=class_names,
        temperature=target_temperature(target),
    )
    identity = {
        "manifest": MANIFEST_FILE if (target / MANIFEST_FILE).exists() else "legacy_default",
        **manifest,
        "checkpoint_sha256": binding["checkpoint_sha256"],
        "temperature": binding["temperature"],
        "class_names_sha256": binding["class_names_sha256"],
    }
    return identity, binding


def current_model(target: Path, prior: Mapping[str, str | None]) -> dict[str, Any]:
    """What the target served before a promotion or restore, for the record."""
    try:
        manifest = read_manifest(target)
    except DeployError as exc:
        return {"error": str(exc)}
    identity: dict[str, Any] = {
        "manifest": MANIFEST_FILE if (target / MANIFEST_FILE).exists() else "legacy_default",
        **manifest,
    }
    if manifest["checkpoint"] in prior:
        identity["checkpoint_sha256"] = prior[manifest["checkpoint"]]
    return identity


def target_model_identity(target: Path) -> dict[str, str]:
    """The model a target directory holds: its manifest fields and checkpoint SHA-256."""
    manifest = read_manifest(target)
    checkpoint = target / manifest["checkpoint"]
    if not checkpoint.is_file():
        raise DeployError(
            f"{display(target)} has no {manifest['checkpoint']}, the checkpoint its model "
            "manifest names, so there is no model to verify."
        )
    return {
        "architecture": manifest["architecture"],
        "checkpoint": manifest["checkpoint"],
        "model_name": manifest["model_name"],
        "checkpoint_sha256": sha256(checkpoint),
    }


# --- Binding a policy to a model -----------------------------------------------


def check_policy_belongs_to_model(source: Path, provenance: Any, model_run: Path) -> str:
    """Require the policy to have been fitted and scored on ``model_run``'s predictions.

    A policy's thresholds, hard classes and confusion pairs are fitted to one
    model's confidences; served with another model they route on numbers they
    were never fitted to. The policy run's ``derivation_provenance.json`` names
    the prediction files it used; both must live inside ``model_run`` and still
    hash to what the provenance recorded.

    Returns:
        A one-line statement of what was checked, for the deployment record.
    """
    provenance_path = source / "derivation_provenance.json"
    predictions = provenance.get("predictions") if isinstance(provenance, dict) else None
    if not isinstance(predictions, dict):
        raise DeployError(
            f"{display(provenance_path)} records no prediction files, so nothing shows the "
            f"policy was fitted on {display(model_run)}'s predictions."
        )
    run = model_run.resolve()
    for split in ("fit", "eval"):
        entry = predictions.get(split)
        if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
            raise DeployError(f"{display(provenance_path)} names no {split} predictions file.")
        path = from_display(entry["path"])
        if not path.is_relative_to(run):
            raise DeployError(
                f"The policy in {display(source)} was fitted on {split} predictions from "
                f"{display(path)}, which is not under the model run {display(model_run)}. "
                "A policy is fitted to one model's confidences: deploy a model only with a "
                "policy recalibrated on that model's predictions."
            )
        if not path.is_file():
            raise DeployError(f"The policy's {split} predictions {display(path)} no longer exist.")
        recorded = entry.get("sha256")
        if not isinstance(recorded, str):
            raise DeployError(
                f"{display(provenance_path)} records no hash for its {split} predictions."
            )
        if sha256(path) != recorded:
            raise DeployError(
                f"{display(path)} changed after the policy was fitted (recorded sha256 "
                f"{recorded}); recalibrate before deploying."
            )
    return f"fit and eval predictions are inside {display(model_run)} with their recorded hashes"


def policy_binding(provenance: Any, provenance_path: Path) -> dict[str, Any]:
    """What the policy's producer evidence binds it to; refuse a policy without evidence.

    Directory membership does not show which checkpoint, class order and
    temperature produced the confidences a policy was fitted to; only the
    evidence the rescorer recorded does (see ``scripts/prediction_evidence.py``).
    """
    try:
        return prediction_evidence.policy_evidence(provenance, provenance_path)
    except prediction_evidence.EvidenceError as exc:
        raise DeployError(str(exc)) from exc


def model_binding(
    *, checkpoint_sha256: str, architecture: str, class_names: list[str], temperature: float
) -> dict[str, Any]:
    """The binding a model's policy evidence must match (``prediction_evidence.model_binding``)."""
    return prediction_evidence.model_binding(
        checkpoint_sha256=checkpoint_sha256,
        architecture=architecture,
        class_names=class_names,
        temperature=temperature,
    )


def check_evidence_matches(
    evidence: dict[str, Any], model: dict[str, Any], source: Path, what: str
) -> str:
    """Require the policy's producer evidence to match a model field by field.

    Args:
        evidence: ``policy_binding()`` of the policy run.
        model: ``model_binding()`` of the model being installed or served.
        source: The policy run, for the message.
        what: Which model, for the message (e.g. "the model being promoted").

    Returns:
        A one-line statement of what was checked, for the deployment record.
    """
    problems = prediction_evidence.binding_mismatches(evidence, model)
    if problems:
        raise DeployError(
            f"The policy in {display(source)} was fitted on predictions that {what} did not "
            f"produce ({'; '.join(problems)}). Its thresholds describe another model's or "
            "another calibration's confidences: recalibrate on this model's own re-scored "
            "predictions before deploying."
        )
    return (
        "checkpoint_sha256, architecture, class_names_sha256, temperature (exact) and "
        f"preprocessing match the policy's producer evidence for {what}"
    )
