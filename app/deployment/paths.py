"""Repository paths, deployment file names and small file helpers."""

from __future__ import annotations

import hashlib
import json
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .errors import DeployError

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TARGET = REPO_ROOT / "app" / "artifacts"
DEPLOYED_FILES = ("decision_policy.json", "hard_classes.json", "confusion_pairs.json")
MANIFEST_FILE = "model.json"
# Promotion installs these with the checkpoint, whose name varies by model.
MODEL_FILES = ("calibration.json", "class_names.json", MANIFEST_FILE)
PROVENANCE_FILE = "deployment_provenance.json"
BACKUP_RECORD_FILE = "backup_record.json"
RESERVED_NAMES = frozenset((*DEPLOYED_FILES, *MODEL_FILES, PROVENANCE_FILE, BACKUP_RECORD_FILE))


def is_plain_file_name(name: Any) -> bool:
    """True for a name that can only mean a file directly inside a directory.

    Records name files by bare name, and every reader joins that name to the
    target or a backup. This refuses each name that would leave the directory
    or fail to open: empty, ``.``, ``..``, a path separator, or NUL.
    """
    return (
        isinstance(name, str)
        and name.strip() != ""
        and name not in (".", "..")
        and not any(character in name for character in "/\\\x00")
    )


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
    """Read JSON, converting a missing or malformed file into a DeployError.

    ``ValueError`` covers more than ``JSONDecodeError``: bytes that are not
    UTF-8, and an integer longer than Python converts from text (4,300 digits).
    """
    if not path.exists():
        raise DeployError(f"{path} does not exist.")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except ValueError as exc:
        raise DeployError(f"{path} is not valid JSON: {brief(exc)}") from exc


def finite_float(value: Any) -> float | None:
    """``value`` as a finite float, or None if it is not a number or has no finite float.

    JSON integers are unbounded, and ``float()`` or ``math.isfinite()`` raise
    ``OverflowError`` on one such as 10**400 instead of returning False.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        number = float(value)
    except OverflowError:
        return None
    return number if math.isfinite(number) else None


def brief(value: Any, limit: int = 80) -> str:
    """``repr(value)``, cut to ``limit`` characters, for quoting a value in an error.

    A record can hold any JSON value, including a 4,000-digit integer, whose
    ``repr`` would also raise past Python's integer-to-text limit.
    """
    if isinstance(value, BaseException):
        text = str(value)
    else:
        try:
            text = repr(value)
        except ValueError:
            text = f"<{type(value).__name__} too large to show>"
    return text if len(text) <= limit else f"{text[: limit - 3]}..."


def write_json_file(path: Path, value: Any) -> None:
    """Write indented JSON with a trailing newline."""
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
