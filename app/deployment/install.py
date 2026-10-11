"""Staging, backup, atomic install, post-install check and rollback.

The installer knows the staged plan -- which staged file names replace which
target files, and which target files to remove -- the backup state and a
verification callback. It never decides what is installed and does not know
what a model or a policy is; the operations do.

``os.replace``, ``shutil.copy2`` and ``tempfile.mkdtemp`` are called through
their modules, and the backup timestamp through this module's ``datetime``,
so tests can inject failures and a fixed clock.
"""

from __future__ import annotations

import os
import shutil
import tempfile
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .errors import DeployError
from .paths import BACKUP_RECORD_FILE, PROVENANCE_FILE, display, now_utc, sha256, write_json_file


@contextmanager
def staging_dir(target: Path) -> Iterator[Path]:
    """A fresh staging directory inside the target, removed afterwards.

    Staging inside the target makes every install step a same-filesystem
    ``os.replace``.
    """
    staging = Path(tempfile.mkdtemp(prefix=".deploy-staging-", dir=target))
    try:
        yield staging
    finally:
        shutil.rmtree(staging, ignore_errors=True)


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
    record_for_backup: Callable[[Path | None], Any],
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
        record_for_backup: Called with the backup directory just created (None
            when nothing existed to back up); returns the provenance record to
            install, which then names that backup. It may raise to refuse the
            record: nothing has been replaced then, and the backup is removed.
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
    try:
        record = record_for_backup(backup)
    except BaseException:
        # Nothing is replaced yet: drop the backup just made, so a refused
        # record leaves the target exactly as it was.
        if backup is not None:
            shutil.rmtree(backup, ignore_errors=True)
        raise

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
