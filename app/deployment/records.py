"""The deployment record: building it from values, and reading historical ones.

``deployment_provenance.json`` is written by every operation -- policy deploy,
model promotion and restore -- and read by ``--verify-live``. This module is
the one definition of its shape. It holds data only: every hash, path and
identity is passed in by the operation that read it, and nothing here touches a
target directory or loads a model.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .paths import MANIFEST_FILE, PROVENANCE_FILE, display

# The model identity every deployment record carries under "model".
RECORDED_MODEL_KEYS = ("architecture", "checkpoint", "model_name", "checkpoint_sha256")


def recorded_model(identity: Mapping[str, Any]) -> dict[str, Any]:
    """The ``model`` entry of a record: the authoritative identity, in one shape."""
    return {key: identity[key] for key in RECORDED_MODEL_KEYS}


def live_service_note(target: Path) -> str:
    """The record's statement of what the file check does not prove."""
    return (
        "NOT verified by this deploy. artifact_files_verified means the files were "
        "read back freshly in the deploy process; a running API process keeps the "
        "model and decision layer it cached at load time until it is restarted. Stop/restart "
        "every API process, then run: python scripts/deploy_decision_policy.py "
        f"--target {display(target)} --verify-live <API base URL>"
    )


def policy_deploy_record(
    *,
    deployed_at: str,
    source_run: str,
    source_provenance_sha256: str,
    source_files_sha256: dict[str, str],
    policy: dict[str, float],
    hard: list[str],
    pairs: list[dict[str, str]],
    policy_evidence: dict[str, Any],
    served_model: dict[str, Any],
    evidence_check: str,
    served_model_files_sha256: dict[str, str],
    model_check: str | None,
) -> dict[str, Any]:
    """The record of a policy-only deploy, before staging."""
    record: dict[str, Any] = {
        "deployed_at": deployed_at,
        "source_run": source_run,
        "source_provenance_sha256": source_provenance_sha256,
        "source_files_sha256": source_files_sha256,
        "policy": policy,
        "hard_class_count": len(hard),
        "confusion_pair_count": len(pairs),
        "untouched": ["model checkpoint", "class_names.json", "calibration.json"],
        "policy_evidence": policy_evidence,
        "served_model": served_model,
        # The same authoritative identity promotion and restore record, so
        # --verify-live reads one representation whatever wrote the record.
        "model": recorded_model(served_model),
        "policy_evidence_matches_served_model": evidence_check,
        # So --verify-live can tell when the served model's files change later.
        "served_model_files_sha256": served_model_files_sha256,
    }
    if model_check is not None:
        record["untouched"].append(MANIFEST_FILE)
        record["policy_belongs_to_served_model"] = model_check
    return record


def promotion_record(
    *,
    deployed_at: str,
    source_run: str,
    source_provenance_sha256: str,
    source_files_sha256: dict[str, str],
    policy: dict[str, float],
    hard: list[str],
    pairs: list[dict[str, str]],
    model_run: str,
    model: Mapping[str, Any],
    calibration_temperature: float,
    model_source_files_sha256: dict[str, str],
    policy_check: str,
    names_check: str,
    policy_evidence: dict[str, Any],
) -> dict[str, Any]:
    """The record of a model promotion, before its checkpoint and evidence checks."""
    return {
        "operation": "model_promotion",
        "deployed_at": deployed_at,
        "source_run": source_run,
        "source_provenance_sha256": source_provenance_sha256,
        "source_files_sha256": source_files_sha256,
        "policy": policy,
        "hard_class_count": len(hard),
        "confusion_pair_count": len(pairs),
        "model_run": model_run,
        "model": recorded_model(model),
        "calibration_temperature": calibration_temperature,
        "model_source_files_sha256": model_source_files_sha256,
        "checks": {"policy_belongs_to_model": policy_check, "class_names": names_check},
        "policy_evidence": policy_evidence,
    }


def restore_record(
    *,
    restored_at: str,
    restored_from: str,
    backup_record: Mapping[str, Any],
    restored_files_sha256: dict[str, str | None],
    removed_files: list[str],
    model: Mapping[str, Any],
    policy: dict[str, float],
    hard: list[str],
    pairs: list[dict[str, str]],
    restored_deployment_provenance: Any,
) -> dict[str, Any]:
    """The record of a restore, embedding the provenance it puts back."""
    return {
        "operation": "restore",
        "restored_at": restored_at,
        "restored_from": restored_from,
        "backup_operation": backup_record.get("operation"),
        "backup_created_at": backup_record.get("created_at"),
        "restored_files_sha256": restored_files_sha256,
        "removed_files": removed_files,
        "model": recorded_model(model),
        "policy": policy,
        "hard_class_count": len(hard),
        "confusion_pair_count": len(pairs),
        "restored_deployment_provenance": restored_deployment_provenance,
    }


def mark_files_verified(
    record: dict[str, Any],
    deployed_files_sha256: dict[str, Any],
    fingerprint: str,
    target: Path,
) -> None:
    """Record that the staged files read back correctly -- and what that does not prove."""
    record["deployed_files_sha256"] = deployed_files_sha256
    # A fresh read of the files in this process -- not proof that a running
    # API process serves them (see live_service).
    record["artifact_files_verified"] = True
    record["decision_layer_fingerprint"] = fingerprint
    record["live_service"] = live_service_note(target)


def add_backup(
    record: dict[str, Any], backup: Path | None, prior: Mapping[str, str | None]
) -> dict[str, Any]:
    """Name the backup and the replaced files' prior hashes in the record; return it."""
    if backup is not None:
        record["backup_dir"] = display(backup)
        record["replaced_files_sha256"] = {
            name: digest
            for name, digest in prior.items()
            if name != PROVENANCE_FILE and digest is not None
        }
        if prior[PROVENANCE_FILE] is not None:
            record["replaced_provenance_sha256"] = prior[PROVENANCE_FILE]
    return record


def recorded_model_identity(provenance: Mapping[str, Any]) -> dict[str, Any] | None:
    """The model identity a deployment record says was deployed, or None.

    Promotion, restore and (from 2026-10-11) policy-only records carry it under
    ``model``. Policy-only records written before then carry it only under
    ``served_model``; read that as a fallback so those records are still
    checked rather than silently skipped.
    """
    for key in ("model", "served_model"):
        identity = provenance.get(key)
        if isinstance(identity, dict) and identity.get("checkpoint_sha256"):
            return identity
    return None
