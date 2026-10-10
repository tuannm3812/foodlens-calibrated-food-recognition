"""Repository paths, deployment file names and small file helpers."""

from __future__ import annotations

import hashlib
import json
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


def write_json_file(path: Path, value: Any) -> None:
    """Write indented JSON with a trailing newline."""
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
