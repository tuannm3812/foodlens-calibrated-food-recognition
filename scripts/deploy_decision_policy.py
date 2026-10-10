"""Deploy a recalibrated decision-layer policy, or promote a model with its policy.

A recalibration run (``scripts/recalibrate_decision_layer.py``) writes its
outputs in the analysis format, which is not the format the backend reads:

- ``decision_policy.json`` is a one-element *list* of policy records carrying
  search metadata, while ``app.backend`` reads a single *dict* of thresholds.
  Copying it verbatim makes ``read_policy()`` raise inside ``load_runtime()``,
  whose ``except Exception`` turns every request into a demo fallback -- the app
  degrades silently instead of failing.
- ``hard_classes.json`` and ``confusion_pairs.json`` already match the runtime
  format and are validated, not converted.

This script converts and validates the three files, then:

1. stages them in a hidden directory inside the target and reads them back
   through the backend's own readers -- the target is untouched until this
   passes;
2. backs up whatever it is about to replace, including any existing
   ``deployment_provenance.json`` (``app/artifacts/`` is gitignored, so an
   overwritten file is otherwise unrecoverable), in a collision-free
   ``replaced_<UTC stamp>_<suffix>/`` directory, with a ``backup_record.json``
   naming each file's prior SHA-256 (or that it did not exist);
3. installs each file with an atomic ``os.replace`` and writes
   ``deployment_provenance.json`` recording the source run, its derivation
   provenance and the hash of every deployed file;
4. reads the installed target back through the backend readers.

If anything fails after installation starts, every original file is restored,
files that did not exist before are removed, the new backup directory is
deleted, and the command fails saying the deployment was rolled back. A failed
invocation therefore leaves the target exactly as it was.

**The file check is not proof of the live service.** Step 4 reads the files
freshly in the deploy process. A running API process caches the model and the
decision layer in ``inference._RUNTIME`` on first use and keeps serving them
until it restarts, so the record says ``artifact_files_verified`` and carries a
``live_service`` note, never a claim about the service. After a deploy,
stop/restart every API process, then run ``--verify-live URL``: it sends one
synthetic image to ``/predict/image`` (so a restarted process loads its runtime,
and a demo fallback fails the check), then requires ``/runtime/status`` to
report, as ``loaded_runtime``, the same decision-layer fingerprint as the target
files (and ``deployment_provenance.json``, when it records one), the same
model identity -- architecture, model name and checkpoint SHA-256 -- as the
target's ``model.json`` and checkpoint, and the same calibration: the probe's
temperature, the status block's cached temperature and the target's
``calibration.json`` must agree exactly, and the cached class-order hash must
match the target's. Model, routing and calibration are reported separately.

**Policy-only mode** (``--source`` alone) touches only the three decision-layer
files and the provenance record. The model checkpoint, class names, calibration
and ``model.json`` are never modified, so deploying a policy can never change
which model is served. The policy's producer evidence (recorded by the
rescorer, carried in its ``derivation_provenance.json``; see
``scripts/prediction_evidence.py``) must match the served model: the actual
checkpoint bytes' SHA-256, the architecture ``model.json`` names (the legacy
ResNet50 default without one), the target's class order and its
``calibration.json`` temperature. When the target carries a ``model.json``
with a ``model_run``, the predictions must also live in that run.

**Promotion mode** (``--source`` with ``--model-run``) installs the checkpoint,
``calibration.json``, ``class_names.json``, ``model.json`` and the three policy
files as one unit through the same steps. A policy is fitted to one model's
confidences, so a model and its policy never deploy apart. Before the target is
touched it requires that the policy run's ``derivation_provenance.json`` names
fit and eval predictions inside ``--model-run`` (with their recorded hashes),
that ``class_names.json`` equals the target's in content and order, that the
staged checkpoint loads into ``--architecture`` with zero missing and zero
unexpected keys, and that the policy's producer evidence matches what is being
installed -- the checkpoint's SHA-256, ``--architecture``, the class order and
the ``calibration.json`` temperature, compared exactly. Directory membership
alone is not evidence: a policy without it is refused, with instructions for
regenerating it by re-scoring. The backup also holds the previously served
checkpoint.

**Restore mode** (``--restore BACKUP_DIR``) puts back the state a backup
recorded -- restoring each backed-up file and removing files that did not exist
then -- through the same stage, verify, back up, install, post-check and
rollback steps, and writes a provenance record of the restore. It is the
rollback path for a promotion. Restart and ``--verify-live`` afterwards.

Usage:
    python scripts/deploy_decision_policy.py \\
        --source results/accuracy_phase1/champion_resnet50_ft_v2/decision_layer_closure_2026-10-10
    RUN=results/accuracy_phase1/a3b_convnext_tiny_continued_224
    python scripts/deploy_decision_policy.py \\
        --source $RUN/decision_layer_closure_2026-10-10 --model-run $RUN \\
        --architecture convnext_tiny --model-name a3b_convnext_tiny
    python scripts/deploy_decision_policy.py --restore app/artifacts/replaced_<stamp>_<suffix>
    # after any of them, stop/restart every API process, then:
    python scripts/deploy_decision_policy.py --verify-live http://127.0.0.1:8000
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import shutil
import sys
import tempfile
import urllib.error
import urllib.parse
import urllib.request
import uuid
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# Producer evidence lives in a sibling module, so this script stays orchestration.
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
import prediction_evidence  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TARGET = REPO_ROOT / "app" / "artifacts"
POLICY_KEYS = ("auto_confidence", "suggest_confidence", "margin_threshold")
DEPLOYED_FILES = ("decision_policy.json", "hard_classes.json", "confusion_pairs.json")
MANIFEST_FILE = "model.json"
# Promotion installs these with the checkpoint, whose name varies by model.
MODEL_FILES = ("calibration.json", "class_names.json", MANIFEST_FILE)
PROVENANCE_FILE = "deployment_provenance.json"
BACKUP_RECORD_FILE = "backup_record.json"
RESERVED_NAMES = frozenset((*DEPLOYED_FILES, *MODEL_FILES, PROVENANCE_FILE, BACKUP_RECORD_FILE))
CLASS_COUNT = 101
IDENTITY_KEYS = ("architecture", "model_name", "checkpoint_sha256")
# What --verify-live checks, each reported on its own.
CHECK_NAMES = ("model", "routing", "calibration")
HTTP_TIMEOUT_SECONDS = 180  # the first prediction after a restart loads the model


class DeployError(Exception):
    """A predictable deployment failure, reported without a traceback."""


def display(path: Path) -> str:
    """Repo-relative path when inside the repo, absolute otherwise."""
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT))
    except ValueError:
        return str(resolved)


def from_display(value: str) -> Path:
    """Invert ``display()``: a relative path is relative to the repo root."""
    path = Path(value)
    return (path if path.is_absolute() else REPO_ROOT / path).resolve()


def sha256(path: Path) -> str:
    """Return the SHA-256 hex digest of a file's bytes, read in chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def now_utc() -> str:
    """The current UTC time as an ISO-8601 string, to the second."""
    return datetime.now(UTC).isoformat(timespec="seconds")


def read_json(path: Path) -> Any:
    """Read JSON, converting a missing or malformed file into a DeployError."""
    if not path.exists():
        raise DeployError(f"{path} does not exist.")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise DeployError(f"{path} is not valid JSON: {exc}") from exc


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


def check_class_coverage(target: Path, hard: list[str], pairs: list[dict[str, str]]) -> None:
    """Reject labels the served model cannot predict -- they would never match."""
    names_path = target / "class_names.json"
    if not names_path.exists():
        raise DeployError(f"{names_path} is missing; cannot check labels against the model.")
    known = set(read_json(names_path))
    labels = set(hard) | {p["actual"] for p in pairs} | {p["predicted"] for p in pairs}
    unknown = sorted(labels - known)
    if unknown:
        raise DeployError(
            f"Labels not in the served model's class_names.json: {', '.join(unknown[:5])}."
        )


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


def backend_modules() -> tuple[Any, Any]:
    """Import the backend inference module and the fingerprint function."""
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
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


def check_evidence_matches(
    evidence: dict[str, Any], model: dict[str, Any], source: Path, what: str
) -> str:
    """Require the policy's producer evidence to match a model field by field.

    Args:
        evidence: ``policy_binding()`` of the policy run.
        model: ``prediction_evidence.model_binding()`` of the model being
            installed or served.
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


def live_service_note(target: Path) -> str:
    """The record's statement of what the file check does not prove."""
    return (
        "NOT verified by this deploy. artifact_files_verified means the files were "
        "read back freshly in the deploy process; a running API process keeps the "
        "model and decision layer it cached at load time until it is restarted. Stop/restart "
        "every API process, then run: python scripts/deploy_decision_policy.py "
        f"--target {display(target)} --verify-live <API base URL>"
    )


def write_json_file(path: Path, value: Any) -> None:
    """Write indented JSON with a trailing newline."""
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def prior_state(target: Path, names: tuple[str, ...]) -> dict[str, str | None]:
    """Each file's SHA-256 in the target now, or None if it does not exist."""
    return {
        name: sha256(target / name) if (target / name).exists() else None
        for name in dict.fromkeys(names)
    }


def make_backup(
    target: Path, prior: dict[str, str | None], operation: str
) -> Path | None:
    """Copy every existing file in ``prior`` to a fresh ``replaced_*`` directory.

    The directory also gets ``backup_record.json``: the full prior state,
    including which files did not exist, which is what ``--restore`` puts back.
    Returns None when no file existed, since there is then nothing to keep.
    """
    if not any(digest is not None for digest in prior.values()):
        return None
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    backup = Path(tempfile.mkdtemp(prefix=f"replaced_{stamp}_", dir=target))
    try:
        for name, digest in prior.items():
            if digest is not None:
                shutil.copy2(target / name, backup / name)
                if sha256(backup / name) != digest:
                    raise DeployError(f"backup copy of {name} does not match the original")
        write_json_file(
            backup / BACKUP_RECORD_FILE,
            {
                "schema_version": 1,
                "created_at": now_utc(),
                "operation": operation,
                "target": display(target),
                "files": prior,
                "restore_with": (
                    f"python scripts/deploy_decision_policy.py --target {display(target)} "
                    f"--restore {display(backup)}"
                ),
            },
        )
    except Exception:
        shutil.rmtree(backup, ignore_errors=True)
        raise
    return backup


def rollback(
    target: Path, staging: Path, backup: Path | None, prior: dict[str, str | None]
) -> None:
    """Restore every installed file to its pre-deploy bytes, then drop the backup.

    Args:
        target: The runtime artifact directory.
        staging: The staging directory, used to stage restored copies so each
            restore is an atomic same-filesystem ``os.replace``.
        backup: Directory holding the pre-deploy copies, or ``None`` when no
            installed file existed before.
        prior: For each installed file name, the SHA-256 it had before the
            deploy, or ``None`` if it did not exist.

    Raises:
        DeployError: If any file cannot be restored. The backup directory is
            then left in place and named, because it holds the only copy.
    """
    try:
        for name, digest in prior.items():
            if digest is None:
                (target / name).unlink(missing_ok=True)
                continue
            if backup is None:
                raise DeployError(f"no backup holds the original {name}")
            if (target / name).exists() and sha256(target / name) == digest:
                continue
            restored = staging / f"restore-{name}"
            shutil.copy2(backup / name, restored)
            os.replace(restored, target / name)
            if sha256(target / name) != digest:
                raise DeployError(f"restored {name} does not match its pre-deploy hash")
    except Exception as exc:
        where = display(backup) if backup is not None else "(no backup was needed)"
        raise DeployError(
            f"ROLLBACK FAILED ({exc}). The target {display(target)} may hold a mixed "
            f"state. The pre-deploy files are preserved in {where}; restore them by hand."
        ) from exc
    if backup is not None:
        shutil.rmtree(backup)


def install(
    target: Path,
    staging: Path,
    record: dict[str, Any],
    installed: tuple[str, ...],
    prior: dict[str, str | None],
    post_check: Callable[[], None],
    *,
    removed: tuple[str, ...] = (),
    operation: str = "policy_deploy",
) -> None:
    """Back up, atomically install the staged files and provenance, and re-check.

    Args:
        target: The runtime artifact directory.
        staging: Directory inside ``target`` holding every name in ``installed``.
        record: The provenance record; backup details are added to it.
        installed: Staged file names to ``os.replace`` into the target.
        prior: Pre-deploy SHA-256 (or None) of every file to back up: the
            installed and removed names, the provenance file, and any file kept
            only for a later ``--restore`` (a promotion's previous checkpoint).
        post_check: Re-reads the installed target; raises on any mismatch.
        removed: Names to delete from the target (a restore of absent files).
        operation: What made the backup, stored in its ``backup_record.json``.

    If anything fails after the first file is replaced, every original file
    (including ``deployment_provenance.json``) is restored, files that did not
    exist before are removed, and the backup directory created here is deleted,
    so a failed deploy leaves the target exactly as it was.
    """
    touched = (*installed, *removed, PROVENANCE_FILE)
    backup = make_backup(target, prior, operation)
    if backup is not None:
        record["backup_dir"] = display(backup)
        record["replaced_files_sha256"] = {
            name: digest
            for name, digest in prior.items()
            if name != PROVENANCE_FILE and digest is not None
        }
        if prior[PROVENANCE_FILE] is not None:
            record["replaced_provenance_sha256"] = prior[PROVENANCE_FILE]

    try:
        for name in installed:
            os.replace(staging / name, target / name)
        for name in removed:
            (target / name).unlink(missing_ok=True)
        write_json_file(staging / PROVENANCE_FILE, record)
        os.replace(staging / PROVENANCE_FILE, target / PROVENANCE_FILE)

        # Post-install check: the target, not the staged copy, is what the
        # backend will read.
        post_check()
    except BaseException as exc:
        rollback(target, staging, backup, {name: prior[name] for name in touched})
        if not isinstance(exc, Exception):
            raise
        raise DeployError(
            f"Deployment rolled back; {display(target)} is unchanged. Cause: {exc}"
        ) from exc


def check_installed_hashes(target: Path, expected: Mapping[str, str]) -> None:
    """Require each installed file to hash to what was staged."""
    for name, digest in expected.items():
        if sha256(target / name) != digest:
            raise DeployError(f"installed {name} does not match the staged file")


def deploy(source: Path, target: Path, dry_run: bool) -> dict[str, Any]:
    """Validate, stage, verify, back up, install and re-verify; return the record.

    A failure at any step leaves ``target`` byte-for-byte as it was: staged files
    are verified before the target is touched, and a failure after installation
    starts is rolled back.
    """
    provenance_path = source / "derivation_provenance.json"
    # Refuse a source without derivation provenance: a deployed policy must trace
    # back to the run that produced it.
    provenance = read_json(provenance_path)
    if not isinstance(provenance, dict):
        raise DeployError(f"{provenance_path} must hold a provenance object.")

    policy, hard, pairs = read_policy_files(source)
    check_class_coverage(target, hard, pairs)
    # Only a promoted target has a manifest; the legacy deployment does not.
    model_run = served_model_run(target)
    model_check = (
        check_policy_belongs_to_model(source, provenance, model_run) if model_run else None
    )
    evidence = policy_binding(provenance, provenance_path)
    served_identity, served_binding = served_model(target)
    evidence_check = check_evidence_matches(
        evidence, served_binding, source, "the target's served model"
    )

    record: dict[str, Any] = {
        "deployed_at": now_utc(),
        "source_run": display(source),
        "source_provenance_sha256": sha256(provenance_path),
        "source_files_sha256": {
            name: sha256(source / name) for name in DEPLOYED_FILES
        },
        "policy": policy,
        "hard_class_count": len(hard),
        "confusion_pair_count": len(pairs),
        "untouched": ["model checkpoint", "class_names.json", "calibration.json"],
        "policy_evidence": evidence,
        "served_model": served_identity,
        "policy_evidence_matches_served_model": evidence_check,
        # So --verify-live can tell when the served model's files change later.
        "served_model_files_sha256": {
            served_identity["checkpoint"]: served_identity["checkpoint_sha256"],
            "calibration.json": sha256(target / "calibration.json"),
            "class_names.json": sha256(target / "class_names.json"),
        },
    }
    if model_check is not None:
        record["untouched"].append(MANIFEST_FILE)
        record["policy_belongs_to_served_model"] = model_check
    if dry_run:
        record["dry_run"] = True
        return record

    # Stage inside the target so every install step is a same-filesystem
    # os.replace, and verify the staged files before the target is touched.
    staging = Path(tempfile.mkdtemp(prefix=".deploy-staging-", dir=target))
    try:
        write_json_file(staging / "decision_policy.json", policy)
        write_json_file(staging / "hard_classes.json", hard)
        write_json_file(staging / "confusion_pairs.json", pairs)
        fingerprint = verify_through_backend(staging, policy, hard, pairs)
        record["deployed_files_sha256"] = {
            name: sha256(staging / name) for name in DEPLOYED_FILES
        }
        # A fresh read of the files in this process -- not proof that a running
        # API process serves them (see live_service).
        record["artifact_files_verified"] = True
        record["decision_layer_fingerprint"] = fingerprint
        record["live_service"] = live_service_note(target)

        def post_check() -> None:
            if verify_through_backend(target, policy, hard, pairs) != fingerprint:
                raise DeployError(
                    "installed decision layer fingerprint differs from the staged one"
                )
            check_installed_hashes(target, record["deployed_files_sha256"])

        prior = prior_state(target, (*DEPLOYED_FILES, PROVENANCE_FILE))
        install(target, staging, record, DEPLOYED_FILES, prior, post_check)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return record


def model_run_checkpoint(model_run: Path, checkpoint_name: str | None) -> Path:
    """The checkpoint to promote: ``--checkpoint``, or the run's only ``.pth`` file."""
    if not model_run.is_dir():
        raise DeployError(f"--model-run {model_run} is not a directory.")
    if checkpoint_name is None:
        candidates = sorted(path.name for path in model_run.glob("*.pth"))
        if len(candidates) != 1:
            raise DeployError(
                f"{display(model_run)} holds {len(candidates)} .pth files "
                f"({', '.join(candidates) or 'none'}); name one with --checkpoint."
            )
        checkpoint_name = candidates[0]
    if Path(checkpoint_name).name != checkpoint_name or checkpoint_name in RESERVED_NAMES:
        raise DeployError(
            f"--checkpoint must be a plain checkpoint file name, not {checkpoint_name!r}."
        )
    checkpoint = model_run / checkpoint_name
    if not checkpoint.is_file():
        raise DeployError(f"{display(checkpoint)} does not exist.")
    return checkpoint


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


def promote(
    source: Path,
    model_run: Path,
    architecture: str,
    model_name: str,
    target: Path,
    dry_run: bool,
    checkpoint_name: str | None = None,
) -> dict[str, Any]:
    """Install a model and the policy fitted on it as one atomic unit; return the record.

    The checkpoint, ``calibration.json``, ``class_names.json``, ``model.json``
    and the three policy files go through the same stage, verify, back up,
    install, post-check and rollback steps as a policy deploy. A failure at any
    step leaves ``target`` byte-for-byte as it was.
    """
    provenance_path = source / "derivation_provenance.json"
    provenance = read_json(provenance_path)
    if not isinstance(provenance, dict):
        raise DeployError(f"{provenance_path} must hold a provenance object.")
    policy, hard, pairs = read_policy_files(source)
    policy_check = check_policy_belongs_to_model(source, provenance, model_run)
    evidence = policy_binding(provenance, provenance_path)

    checkpoint = model_run_checkpoint(model_run, checkpoint_name)
    temperature = validated_temperature(
        read_json(model_run / "calibration.json"), model_run / "calibration.json"
    )
    class_names = validated_class_names(
        read_json(model_run / "class_names.json"), model_run / "class_names.json"
    )
    names_check = check_class_names_match(target, class_names)
    check_class_coverage(target, hard, pairs)

    manifest_fields = {
        "architecture": architecture,
        "checkpoint": checkpoint.name,
        "model_name": model_name,
        "model_run": display(model_run),
    }
    artifacts = backend_artifacts()
    try:
        manifest = artifacts.validate_model_manifest(manifest_fields, Path(MANIFEST_FILE))
    except artifacts.ModelManifestError as exc:
        raise DeployError(str(exc)) from exc

    checkpoint_sha = sha256(checkpoint)
    # What the policy's producer evidence must match: the checkpoint bytes, the
    # architecture, the class order and the calibration being installed. It is
    # checked right after the checkpoint is shown to fit the architecture.
    model = prediction_evidence.model_binding(
        checkpoint_sha256=checkpoint_sha,
        architecture=architecture,
        class_names=class_names,
        temperature=temperature,
    )
    record: dict[str, Any] = {
        "operation": "model_promotion",
        "deployed_at": now_utc(),
        "source_run": display(source),
        "source_provenance_sha256": sha256(provenance_path),
        "source_files_sha256": {name: sha256(source / name) for name in DEPLOYED_FILES},
        "policy": policy,
        "hard_class_count": len(hard),
        "confusion_pair_count": len(pairs),
        "model_run": display(model_run),
        "model": {
            "architecture": architecture,
            "checkpoint": checkpoint.name,
            "model_name": model_name,
            "checkpoint_sha256": checkpoint_sha,
        },
        "calibration_temperature": temperature,
        "model_source_files_sha256": {
            checkpoint.name: checkpoint_sha,
            "calibration.json": sha256(model_run / "calibration.json"),
            "class_names.json": sha256(model_run / "class_names.json"),
        },
        "checks": {"policy_belongs_to_model": policy_check, "class_names": names_check},
        "policy_evidence": evidence,
    }
    what = "the model being promoted"
    if dry_run:
        record["checks"]["checkpoint_fits_architecture"] = check_checkpoint_fits(
            checkpoint, architecture
        )
        record["checks"]["policy_evidence_matches_model"] = check_evidence_matches(
            evidence, model, source, what
        )
        record["dry_run"] = True
        return record

    installed = (checkpoint.name, *MODEL_FILES, *DEPLOYED_FILES)
    staging = Path(tempfile.mkdtemp(prefix=".deploy-staging-", dir=target))
    try:
        shutil.copy2(checkpoint, staging / checkpoint.name)
        if sha256(staging / checkpoint.name) != checkpoint_sha:
            raise DeployError(f"staged {checkpoint.name} does not match {display(checkpoint)}")
        shutil.copy2(model_run / "calibration.json", staging / "calibration.json")
        shutil.copy2(model_run / "class_names.json", staging / "class_names.json")
        write_json_file(staging / MANIFEST_FILE, manifest)
        write_json_file(staging / "decision_policy.json", policy)
        write_json_file(staging / "hard_classes.json", hard)
        write_json_file(staging / "confusion_pairs.json", pairs)

        # Every check runs on the staged bytes, before the target is touched.
        record["checks"]["checkpoint_fits_architecture"] = check_checkpoint_fits(
            staging / checkpoint.name, architecture, f"{display(checkpoint)} (staged copy)"
        )
        record["checks"]["policy_evidence_matches_model"] = check_evidence_matches(
            evidence, model, source, what
        )
        fingerprint = verify_through_backend(staging, policy, hard, pairs)
        verify_model_files(staging, manifest, temperature, class_names)
        record["deployed_files_sha256"] = {
            name: checkpoint_sha if name == checkpoint.name else sha256(staging / name)
            for name in installed
        }
        record["artifact_files_verified"] = True
        record["decision_layer_fingerprint"] = fingerprint
        record["live_service"] = live_service_note(target)

        # Keep the previously served checkpoint in the backup too, so --restore
        # does not depend on it surviving in the target.
        try:
            previous_checkpoint = (read_manifest(target)["checkpoint"],)
        except DeployError:
            previous_checkpoint = ()
        prior = prior_state(target, (*installed, *previous_checkpoint, PROVENANCE_FILE))
        record["previous_model"] = current_model(target, prior)

        def post_check() -> None:
            if verify_through_backend(target, policy, hard, pairs) != fingerprint:
                raise DeployError(
                    "installed decision layer fingerprint differs from the staged one"
                )
            verify_model_files(target, manifest, temperature, class_names)
            check_installed_hashes(target, record["deployed_files_sha256"])

        install(
            target, staging, record, installed, prior, post_check, operation="model_promotion"
        )
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return record


def read_backup_record(backup: Path, target: Path) -> dict[str, Any]:
    """Read and integrity-check a backup's ``backup_record.json``.

    Raises:
        DeployError: If the backup is not inside the target, has no record, or a
            backed-up file is missing or no longer matches its recorded hash.
    """
    if not backup.is_dir():
        raise DeployError(f"--restore {backup} is not a directory.")
    if backup.resolve().parent != target.resolve():
        raise DeployError(
            f"--restore takes a backup directory inside the target {display(target)}; "
            f"{display(backup)} is not one."
        )
    record_path = backup / BACKUP_RECORD_FILE
    if not record_path.exists():
        raise DeployError(
            f"{display(backup)} has no {BACKUP_RECORD_FILE}: it predates backup records, so "
            "the state before it (which files did not exist) is unknown. Restore it by hand."
        )
    backup_record = read_json(record_path)
    files = backup_record.get("files") if isinstance(backup_record, dict) else None
    if not isinstance(files, dict) or not files:
        raise DeployError(f"{display(record_path)} lists no files.")
    for name, digest in files.items():
        if Path(name).name != name or name == BACKUP_RECORD_FILE:
            raise DeployError(f"{display(record_path)} names an invalid file {name!r}.")
        if digest is None:
            continue
        if not isinstance(digest, str) or not (backup / name).is_file():
            raise DeployError(f"{display(backup)} is missing its copy of {name}.")
        if sha256(backup / name) != digest:
            raise DeployError(
                f"{display(backup / name)} no longer matches the hash recorded when it was "
                "backed up; refusing to restore a damaged backup."
            )
    return backup_record


def restore(backup: Path, target: Path, dry_run: bool) -> dict[str, Any]:
    """Put back the state a backup recorded; return the restore record.

    Each backed-up file is restored and each file the backup records as absent
    is removed, through the same stage, verify, back up, install, post-check and
    rollback steps as a deploy. The state being replaced is itself backed up, so
    a restore can be undone with another ``--restore``. The target's
    ``deployment_provenance.json`` becomes a record of the restore that embeds
    the restored provenance.
    """
    backup_record = read_backup_record(backup, target)
    files: dict[str, str | None] = backup_record["files"]
    names = tuple(name for name in files if name != PROVENANCE_FILE)
    installed = tuple(name for name in names if files[name] is not None)
    absent = tuple(name for name in names if files[name] is None)

    def resulting(name: str) -> Path | None:
        """Where the restored state's copy of a file is, or None if it has none."""
        if name in installed:
            return backup / name
        if name in absent or not (target / name).exists():
            return None
        return target / name

    artifacts = backend_artifacts()
    manifest_path = resulting(MANIFEST_FILE)
    try:
        manifest = (
            artifacts.validate_model_manifest(read_json(manifest_path), manifest_path)
            if manifest_path is not None
            else dict(artifacts.LEGACY_MODEL)
        )
    except artifacts.ModelManifestError as exc:
        raise DeployError(f"The restored state's model.json is invalid: {exc}") from exc
    checkpoint_path = resulting(manifest["checkpoint"])
    if checkpoint_path is None:
        raise DeployError(
            f"The restored state would have no {manifest['checkpoint']}, the checkpoint its "
            "model manifest names; refusing to restore a state that cannot serve."
        )
    names_path = resulting("class_names.json")
    if names_path is None:
        raise DeployError("The restored state would have no class_names.json.")
    class_names = validated_class_names(read_json(names_path), names_path)
    calibration_path = resulting("calibration.json")
    if calibration_path is not None:
        validated_temperature(read_json(calibration_path), calibration_path)
    policy_paths = {name: resulting(name) for name in DEPLOYED_FILES}
    missing_policy = [name for name, path in policy_paths.items() if path is None]
    if missing_policy:
        raise DeployError(f"The restored state would lack {', '.join(missing_policy)}.")
    policy = runtime_policy(read_json(policy_paths["decision_policy.json"]), backup)
    hard = validated_hard_classes(read_json(policy_paths["hard_classes.json"]), backup)
    pairs = validated_confusion_pairs(read_json(policy_paths["confusion_pairs.json"]), backup)
    labels = set(hard) | {p["actual"] for p in pairs} | {p["predicted"] for p in pairs}
    if labels - set(class_names):
        raise DeployError("The restored decision layer names classes the restored model lacks.")
    _, fingerprint_of = backend_modules()
    fingerprint = fingerprint_of(policy, hard, pairs)

    restored_provenance = (
        read_json(backup / PROVENANCE_FILE) if files.get(PROVENANCE_FILE) else None
    )
    record: dict[str, Any] = {
        "operation": "restore",
        "restored_at": now_utc(),
        "restored_from": display(backup),
        "backup_operation": backup_record.get("operation"),
        "backup_created_at": backup_record.get("created_at"),
        "restored_files_sha256": {name: files[name] for name in installed},
        "removed_files": list(absent),
        "model": {
            "architecture": manifest["architecture"],
            "checkpoint": manifest["checkpoint"],
            "model_name": manifest["model_name"],
            "checkpoint_sha256": files.get(manifest["checkpoint"]) or sha256(checkpoint_path),
        },
        "policy": policy,
        "hard_class_count": len(hard),
        "confusion_pair_count": len(pairs),
        "restored_deployment_provenance": restored_provenance,
    }
    model_changes = MANIFEST_FILE in files or manifest["checkpoint"] in files
    if dry_run:
        if model_changes:
            record["checkpoint_fits_architecture"] = check_checkpoint_fits(
                checkpoint_path, manifest["architecture"]
            )
        record["dry_run"] = True
        return record

    staging = Path(tempfile.mkdtemp(prefix=".deploy-staging-", dir=target))
    try:
        for name in installed:
            shutil.copy2(backup / name, staging / name)
            if sha256(staging / name) != files[name]:
                raise DeployError(f"staged {name} does not match the backup record")
        if model_changes:
            staged_checkpoint = (
                staging / manifest["checkpoint"]
                if manifest["checkpoint"] in installed
                else checkpoint_path
            )
            record["checkpoint_fits_architecture"] = check_checkpoint_fits(
                staged_checkpoint, manifest["architecture"], manifest["checkpoint"]
            )
        record["deployed_files_sha256"] = dict(record["restored_files_sha256"])
        record["artifact_files_verified"] = True
        record["decision_layer_fingerprint"] = fingerprint
        record["live_service"] = live_service_note(target)

        current_checkpoint: tuple[str, ...] = ()
        try:
            current_checkpoint = (read_manifest(target)["checkpoint"],)
        except DeployError:
            pass
        prior = prior_state(target, (*installed, *absent, *current_checkpoint, PROVENANCE_FILE))
        record["previous_model"] = current_model(target, prior)

        def post_check() -> None:
            if verify_through_backend(target, policy, hard, pairs) != fingerprint:
                raise DeployError("restored decision layer fingerprint differs from the backup's")
            if read_manifest(target) != manifest:
                raise DeployError("the restored model manifest differs from the backup's")
            check_installed_hashes(target, record["deployed_files_sha256"])
            leftover = [name for name in absent if (target / name).exists()]
            if leftover:
                raise DeployError(f"files absent in the backup remain: {', '.join(leftover)}")

        install(
            target,
            staging,
            record,
            installed,
            prior,
            post_check,
            removed=absent,
            operation="restore",
        )
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return record


Fetch = Callable[[str, str, bytes | None, dict[str, str]], Any]


def http_fetch(method: str, url: str, body: bytes | None, headers: dict[str, str]) -> Any:
    """Send one HTTP request with the standard library and decode its JSON body."""
    request = urllib.request.Request(url, data=body, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=HTTP_TIMEOUT_SECONDS) as response:
            payload = response.read()
    except urllib.error.HTTPError as exc:
        raise DeployError(f"{method} {url} returned HTTP {exc.code}.") from exc
    except OSError as exc:
        raise DeployError(f"{method} {url} failed: {exc}") from exc
    try:
        return json.loads(payload)
    except json.JSONDecodeError as exc:
        raise DeployError(f"{method} {url} did not return JSON: {exc}") from exc


def probe_image() -> bytes:
    """A small synthetic PNG, built in memory, for the warm-up prediction."""
    try:
        from PIL import Image
    except ImportError as exc:
        raise DeployError("--verify-live needs Pillow to build its probe image.") from exc
    buffer = io.BytesIO()
    Image.new("RGB", (64, 64), color=(200, 120, 60)).save(buffer, format="PNG")
    return buffer.getvalue()


def multipart_file(field: str, filename: str, content: bytes, content_type: str):
    """Encode one file as a multipart/form-data body; return (body, headers)."""
    boundary = uuid.uuid4().hex
    head = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="{field}"; filename="{filename}"\r\n'
        f"Content-Type: {content_type}\r\n\r\n"
    ).encode()
    body = head + content + f"\r\n--{boundary}--\r\n".encode()
    return body, {"Content-Type": f"multipart/form-data; boundary={boundary}"}


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


def live_model_problem(served: Any, expected: Mapping[str, str]) -> str | None:
    """Compare the status ``model`` block with the target's model; describe any mismatch."""
    if not isinstance(served, dict):
        return (
            "/runtime/status reports no model block; the service runs code that predates "
            "the model identity check. Restart it on the current code."
        )
    if served.get("source") != "loaded_runtime":
        return (
            f"The service reports model.source={served.get('source')!r}, not "
            "'loaded_runtime', so it has not loaded a model to compare. Restart it and retry."
        )
    mismatched = [
        f"{key}: service {served.get(key)!r}, target {expected[key]!r}"
        for key in IDENTITY_KEYS
        if served.get(key) != expected[key]
    ]
    if not mismatched:
        return None
    return (
        "The live service serves a different model than the target holds ("
        + "; ".join(mismatched)
        + "). It is still running the model it loaded before the files changed: "
        "stop/restart every API process, then rerun --verify-live."
    )


def target_calibration(target: Path) -> dict[str, Any]:
    """The calibration a target holds: its validated temperature and class-order hash."""
    return {
        "temperature": target_temperature(target),
        "class_names_sha256": prediction_evidence.class_names_sha256(target_class_names(target)),
    }


def live_calibration_problems(
    prediction: Mapping[str, Any], served: Any, expected: Mapping[str, Any]
) -> list[str]:
    """Compare the runtime's cached calibration with the target's; describe each mismatch.

    The temperature is cached by ``load_runtime()`` and is part of neither the
    model identity nor the routing fingerprint, so it is checked on its own:
    the probe response's ``temperature`` (what the runtime just scaled logits
    by) must equal the status ``model`` block's, and both must equal the
    target's validated ``calibration.json``, exactly. The class-order hash the
    runtime computed at load must equal the target's.
    """
    if not isinstance(served, dict) or served.get("source") != "loaded_runtime":
        return ["/runtime/status reports no loaded model block, so no cached calibration."]
    restart = "stop/restart every API process, then rerun --verify-live."
    problems = []
    status_temperature = served.get("temperature")
    probe_temperature = prediction.get("temperature")
    if status_temperature is None:
        problems.append(
            "/runtime/status reports no temperature in its model block; the service runs code "
            "that predates the calibration check. Restart it on the current code."
        )
    elif probe_temperature != status_temperature:
        problems.append(
            f"The probe response was scaled by temperature {probe_temperature!r}, but the "
            f"status model block reports {status_temperature!r}."
        )
    if status_temperature is not None and status_temperature != expected["temperature"]:
        problems.append(
            f"The live service scales logits by cached temperature {status_temperature!r}, but "
            f"the target's calibration.json holds {expected['temperature']!r}. It is still "
            f"running the calibration it loaded before the files changed: {restart}"
        )
    served_names = served.get("class_names_sha256")
    if served_names != expected["class_names_sha256"]:
        problems.append(
            f"class_names_sha256: service {served_names!r}, target "
            f"{expected['class_names_sha256']!r}. A different class order mislabels every "
            f"prediction: {restart}"
        )
    return problems


def recorded_drift(
    provenance: Mapping[str, Any],
    target: Path,
    files_fingerprint: str,
    expected_model: Mapping[str, str],
    expected_calibration: Mapping[str, Any],
) -> dict[str, list[str]]:
    """Each way the target's files changed after the deployment record was written.

    Returns:
        Problems by check (``model``, ``routing``, ``calibration``); a record
        that predates a field is not held against the target.
    """
    redeploy = "it changed after deployment. Redeploy before verifying the service."
    drift: dict[str, list[str]] = {name: [] for name in CHECK_NAMES}
    recorded = provenance.get("decision_layer_fingerprint")
    if recorded is not None and recorded != files_fingerprint:
        drift["routing"].append(
            f"The target files (fingerprint {files_fingerprint}) no longer match "
            f"{PROVENANCE_FILE} (fingerprint {recorded}); they changed after deployment. "
            "Redeploy before verifying the service."
        )
    recorded_model = provenance.get("model")
    if isinstance(recorded_model, dict) and recorded_model.get("checkpoint_sha256") not in (
        None,
        expected_model["checkpoint_sha256"],
    ):
        drift["model"].append(
            f"The target checkpoint (sha256 {expected_model['checkpoint_sha256']}) no longer "
            f"matches {PROVENANCE_FILE} (sha256 {recorded_model['checkpoint_sha256']}); "
            + redeploy
        )
    for key in ("deployed_files_sha256", "served_model_files_sha256"):
        hashes = provenance.get(key)
        if not isinstance(hashes, dict):
            continue
        for name in ("calibration.json", "class_names.json"):
            if name in hashes and hashes[name] != sha256(target / name):
                drift["calibration"].append(
                    f"The target's {name} (sha256 {sha256(target / name)}) no longer matches "
                    f"{PROVENANCE_FILE}'s {key} (sha256 {hashes[name]}); " + redeploy
                )
    temperatures = [provenance.get("calibration_temperature")]
    if isinstance(provenance.get("served_model"), dict):
        temperatures.append(provenance["served_model"].get("temperature"))
    for value in temperatures:
        if value is not None and value != expected_calibration["temperature"]:
            drift["calibration"].append(
                f"The target's temperature {expected_calibration['temperature']!r} differs from "
                f"the {value!r} {PROVENANCE_FILE} recorded; " + redeploy
            )
    drift["calibration"] = list(dict.fromkeys(drift["calibration"]))
    return drift


def verification_failure(
    problems: Mapping[str, list[str]], stage: str, passed: str = "passed"
) -> DeployError:
    """One error naming every check's outcome, so a failure says which one broke."""
    lines = [f"--verify-live failed ({stage})."]
    for name in CHECK_NAMES:
        if problems[name]:
            lines.append(f"{name}: FAILED")
            lines.extend(f"  - {problem}" for problem in problems[name])
        else:
            lines.append(f"{name}: {passed}")
    return DeployError("\n".join(lines))


def verify_live(url: str, target: Path, fetch: Fetch = http_fetch) -> dict[str, Any]:
    """Prove that a running API process serves the target's model, routing and calibration.

    1. Check the target's files against ``deployment_provenance.json``, when it
       records them: the decision-layer fingerprint, the checkpoint hash, and
       the hashes of ``calibration.json`` and ``class_names.json`` (and the
       recorded temperature), so files edited after deployment are caught.
    2. POST a synthetic image to ``/predict/image`` so a freshly restarted
       process loads its runtime; any ``fallback_reason`` fails the check,
       because a silent demo fallback is exactly the hazard being ruled out.
    3. GET ``/runtime/status`` and check, independently:

       - **routing**: ``decision_layer.source == "loaded_runtime"`` with a
         fingerprint equal to the target files';
       - **model**: a ``loaded_runtime`` model block whose architecture, model
         name and checkpoint SHA-256 equal the target's;
       - **calibration**: the probe response's temperature equals the model
         block's cached temperature, both equal the target's validated
         ``calibration.json`` exactly, and the cached class-order hash equals
         the target's.

    Args:
        url: Base URL of the running API, e.g. ``http://127.0.0.1:8000``.
        target: The artifact directory the service is meant to serve.
        fetch: HTTP transport, injectable so tests need no network.

    Returns:
        The verification record, with a ``checks`` entry per check.

    Raises:
        DeployError: On any mismatch, fallback or transport failure, naming
            each check as passed or FAILED with every problem found.
    """
    base = url.rstrip("/")
    if urllib.parse.urlparse(base).scheme not in {"http", "https"}:
        raise DeployError(f"--verify-live needs an http(s) URL, got {url!r}.")
    missing = [name for name in DEPLOYED_FILES if not (target / name).exists()]
    if missing:
        raise DeployError(
            f"{display(target)} is missing {', '.join(missing)}; the backend would "
            "serve built-in defaults, so there is no deployed policy to verify."
        )
    _, _, _, files_fingerprint = read_through_backend(target)
    expected_model = target_model_identity(target)
    expected_calibration = target_calibration(target)

    result: dict[str, Any] = {"url": base, "target": display(target)}
    provenance_path = target / PROVENANCE_FILE
    provenance: dict[str, Any] = {}
    if provenance_path.exists():
        provenance = read_json(provenance_path)
        if not isinstance(provenance, dict):
            raise DeployError(f"{provenance_path} must hold a provenance object.")
    drift = recorded_drift(
        provenance, target, files_fingerprint, expected_model, expected_calibration
    )
    if any(drift.values()):
        raise verification_failure(
            drift,
            "the target's files changed after deployment; the service was not probed",
            passed="target files match the deployment record",
        )
    if provenance.get("decision_layer_fingerprint") is None:
        result["provenance_check"] = (
            f"{PROVENANCE_FILE} records no decision_layer_fingerprint (deployments before "
            "2026-10-10 did not); compared against the target files alone."
        )
    else:
        result["provenance_check"] = "target files match deployment_provenance.json"

    body, headers = multipart_file("file", "verify-live-probe.png", probe_image(), "image/png")
    prediction = fetch("POST", f"{base}/predict/image", body, headers)
    if not isinstance(prediction, dict):
        kind = type(prediction).__name__
        raise DeployError(f"{base}/predict/image returned {kind}, not an object.")
    if prediction.get("fallback_reason") or prediction.get("artifact_status") != "ready":
        raise DeployError(
            f"The probe prediction fell back (fallback_reason="
            f"{prediction.get('fallback_reason')!r}, artifact_status="
            f"{prediction.get('artifact_status')!r}): the service is serving demo output, "
            "not the model. Check its artifacts and logs, then restart it."
        )

    status = fetch("GET", f"{base}/runtime/status", None, {})
    problems: dict[str, list[str]] = {name: [] for name in CHECK_NAMES}
    layer = status.get("decision_layer") if isinstance(status, dict) else None
    if not isinstance(layer, dict):
        problems["routing"].append(
            f"{base}/runtime/status reports no decision_layer; the service runs code that "
            "predates the live check. Restart it on the current code."
        )
    elif layer.get("source") != "loaded_runtime":
        problems["routing"].append(
            f"The service reports decision_layer.source={layer.get('source')!r}, not "
            "'loaded_runtime', so it has not loaded a runtime to compare. Restart it and retry."
        )
    elif layer.get("fingerprint") != files_fingerprint:
        problems["routing"].append(
            f"The live service routes with fingerprint {layer.get('fingerprint')}, but the "
            f"target files have {files_fingerprint}. The service is still serving a cached "
            "decision layer: stop/restart every API process, then rerun --verify-live."
        )
    served_model = status.get("model") if isinstance(status, dict) else None
    model_problem = live_model_problem(served_model, expected_model)
    if model_problem:
        problems["model"].append(model_problem)
    problems["calibration"] = live_calibration_problems(
        prediction, served_model, expected_calibration
    )
    if any(problems.values()):
        raise verification_failure(problems, "the live service differs from the target")

    result.update(
        {
            "live_service_verified": True,
            "checks": {
                "model": "passed: architecture, model_name and checkpoint_sha256 match",
                "routing": "passed: the loaded decision-layer fingerprint matches",
                "calibration": (
                    "passed: probe and status temperature equal calibration.json exactly; "
                    "class_names_sha256 matches"
                ),
            },
            "decision_layer_fingerprint": layer.get("fingerprint"),
            "served_policy": layer.get("policy"),
            "hard_class_count": layer.get("hard_class_count"),
            "confusion_pair_count": layer.get("confusion_pair_count"),
            "model": {key: served_model.get(key) for key in ("checkpoint", *IDENTITY_KEYS)},
            "calibration": {
                "temperature": served_model.get("temperature"),
                "probe_temperature": prediction.get("temperature"),
                "class_names_sha256": served_model.get("class_names_sha256"),
            },
        }
    )
    return result


def main(argv: list[str] | None = None, fetch: Fetch = http_fetch) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--source", type=Path, help="Recalibration run output dir to deploy")
    parser.add_argument("--target", type=Path, default=DEFAULT_TARGET, help="Runtime artifact dir")
    parser.add_argument("--dry-run", action="store_true", help="Validate only; write nothing")
    parser.add_argument(
        "--model-run",
        type=Path,
        help="Promote this training run's model together with the --source policy fitted on it",
    )
    parser.add_argument("--architecture", help="With --model-run: resnet50 or convnext_tiny")
    parser.add_argument("--model-name", help="With --model-run: the name responses report")
    parser.add_argument(
        "--checkpoint", help="With --model-run: checkpoint file name (default: its only .pth)"
    )
    parser.add_argument(
        "--restore",
        type=Path,
        metavar="BACKUP_DIR",
        help="Put back the state a replaced_* backup in --target recorded (run on its own)",
    )
    parser.add_argument(
        "--verify-live",
        metavar="URL",
        help="Check a restarted API at URL serves the target's model and policy (run on its own)",
    )
    args = parser.parse_args(argv)
    model_options = (args.architecture, args.model_name, args.checkpoint)
    if args.verify_live and (args.source or args.dry_run or args.model_run or args.restore):
        parser.error(
            "--verify-live runs on its own: deploy, restart every API process, then verify."
        )
    if args.restore and (args.source or args.model_run or any(model_options)):
        parser.error("--restore runs on its own (with --target and, optionally, --dry-run).")
    if args.model_run and not (args.source and args.architecture and args.model_name):
        parser.error(
            "--model-run needs --source (a policy run fitted on that model), "
            "--architecture and --model-name."
        )
    if any(model_options) and not args.model_run:
        parser.error("--architecture, --model-name and --checkpoint apply only with --model-run.")
    if not (args.verify_live or args.restore) and args.source is None:
        parser.error("one of --source, --restore or --verify-live is required")
    try:
        if args.verify_live:
            result = verify_live(args.verify_live, args.target, fetch)
        elif args.restore:
            result = restore(args.restore, args.target, args.dry_run)
        elif args.model_run:
            result = promote(
                args.source,
                args.model_run,
                args.architecture,
                args.model_name,
                args.target,
                args.dry_run,
                args.checkpoint,
            )
        else:
            result = deploy(args.source, args.target, args.dry_run)
    except (DeployError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2))
    if args.verify_live:
        print(
            "live service verified: it serves the target's model, decision layer and "
            "calibration.",
            file=sys.stderr,
        )
    elif not args.dry_run:
        print(
            "NOTE: only the artifact files were verified, not the live service. Running "
            "API processes keep their cached model and decision layer until restarted: restart "
            "every API process, then run with --verify-live <API base URL>.",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
