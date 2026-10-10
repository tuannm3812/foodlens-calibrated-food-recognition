"""Restore: put back the state a ``replaced_*`` backup recorded.

Each backed-up file is restored and each file the backup records as absent is
removed, through the same stage, verify, back up, install, post-check and
rollback steps as a deploy. It is the rollback path for a promotion.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from . import identity, records
from .errors import DeployError
from .install import check_installed_hashes, install, prior_state, staging_dir
from .paths import (
    BACKUP_RECORD_FILE,
    DEPLOYED_FILES,
    MANIFEST_FILE,
    PROVENANCE_FILE,
    display,
    now_utc,
    read_json,
    sha256,
)


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

    manifest_path = resulting(MANIFEST_FILE)
    if manifest_path is None:
        manifest = identity.legacy_manifest()
    else:
        raw_manifest = read_json(manifest_path)
        try:
            manifest = identity.validate_manifest(raw_manifest, manifest_path)
        except DeployError as exc:
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
    class_names = identity.validated_class_names(read_json(names_path), names_path)
    calibration_path = resulting("calibration.json")
    if calibration_path is None:
        raise DeployError(
            "The restored state would have no calibration.json: the backend would silently "
            "serve its built-in default temperature, and no record could name the calibration "
            "served."
        )
    temperature = identity.validated_temperature(read_json(calibration_path), calibration_path)
    policy_paths = {name: resulting(name) for name in DEPLOYED_FILES}
    missing_policy = [name for name, path in policy_paths.items() if path is None]
    if missing_policy:
        raise DeployError(f"The restored state would lack {', '.join(missing_policy)}.")
    policy = identity.runtime_policy(read_json(policy_paths["decision_policy.json"]), backup)
    hard = identity.validated_hard_classes(read_json(policy_paths["hard_classes.json"]), backup)
    pairs = identity.validated_confusion_pairs(
        read_json(policy_paths["confusion_pairs.json"]), backup
    )
    if identity.decision_layer_labels(hard, pairs) - set(class_names):
        raise DeployError("The restored decision layer names classes the restored model lacks.")
    fingerprint = identity.decision_layer_fingerprint(policy, hard, pairs)

    restored_provenance = (
        read_json(backup / PROVENANCE_FILE) if files.get(PROVENANCE_FILE) else None
    )
    record = records.restore_record(
        restored_at=now_utc(),
        restored_from=display(backup),
        backup_record=backup_record,
        restored_files_sha256={name: files[name] for name in installed},
        removed_files=list(absent),
        model={
            "architecture": manifest["architecture"],
            "checkpoint": manifest["checkpoint"],
            "model_name": manifest["model_name"],
            "checkpoint_sha256": files.get(manifest["checkpoint"]) or sha256(checkpoint_path),
        },
        calibration_temperature=temperature,
        calibration_files_sha256={
            "calibration.json": sha256(calibration_path),
            "class_names.json": sha256(names_path),
        },
        policy=policy,
        hard=hard,
        pairs=pairs,
        restored_deployment_provenance=restored_provenance,
    )
    model_changes = MANIFEST_FILE in files or manifest["checkpoint"] in files
    if dry_run:
        if model_changes:
            record["checkpoint_fits_architecture"] = identity.check_checkpoint_fits(
                checkpoint_path, manifest["architecture"]
            )
        record["dry_run"] = True
        return record

    with staging_dir(target) as staging:
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
            record["checkpoint_fits_architecture"] = identity.check_checkpoint_fits(
                staged_checkpoint, manifest["architecture"], manifest["checkpoint"]
            )
        records.mark_files_verified(
            record, dict(record["restored_files_sha256"]), fingerprint, target
        )

        current_checkpoint: tuple[str, ...] = ()
        try:
            current_checkpoint = (identity.read_manifest(target)["checkpoint"],)
        except DeployError:
            pass
        prior = prior_state(target, (*installed, *absent, *current_checkpoint, PROVENANCE_FILE))
        record["previous_model"] = identity.current_model(target, prior)

        def post_check() -> None:
            if identity.verify_through_backend(target, policy, hard, pairs) != fingerprint:
                raise DeployError("restored decision layer fingerprint differs from the backup's")
            if identity.read_manifest(target) != manifest:
                raise DeployError("the restored model manifest differs from the backup's")
            check_installed_hashes(target, record["deployed_files_sha256"])
            leftover = [name for name in absent if (target / name).exists()]
            if leftover:
                raise DeployError(f"files absent in the backup remain: {', '.join(leftover)}")

        # The complete record is checked before a backup exists, and again,
        # naming its backup, before any target file is replaced.
        records.validate_record(record, "base")
        install(
            target,
            staging,
            lambda backup_dir: records.finalised_record(record, backup_dir, prior),
            installed,
            prior,
            post_check,
            removed=absent,
            operation="restore",
        )
    return record
