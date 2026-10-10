"""The deployment record: its schema, building it from values, and reading historical ones.

``deployment_provenance.json`` is written by every operation -- policy deploy,
model promotion and restore -- and read by ``--verify-live``. This module is
the one definition of its shape. It holds data only: every hash, path and
identity is passed in by the operation that read it, and nothing here touches a
target directory or loads a model.

**Writers use the strict schema.** A record carries ``schema_version`` and an
``operation`` discriminator; the fields every operation must record and the
fields each operation adds are declared below as data (``COMMON_FIELDS``,
``OPERATION_FIELDS``, ``BACKUP_FIELDS``), each with the check its value must
pass. ``validate_record`` applies them, at two stages: the ``base`` record
before any backup is made, and the ``final`` record, which names its backup,
before any target file is replaced.

**Readers use the normalised form.** ``normalise_record`` returns a record in
the schema's shape whatever wrote it: a current record exactly as validated, or
a record written before the schema mapped from its own fields, with explicit
``limits`` for what it never captured. It never fills a gap from the target.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Mapping
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from .errors import DeployError
from .paths import MANIFEST_FILE, PROVENANCE_FILE, display

SCHEMA_VERSION = 1
# The operation discriminator. "policy_deploy" is the name install.make_backup
# already gave a policy-only deploy in backup_record.json.
OPERATIONS = ("policy_deploy", "model_promotion", "restore")
# The model identity every deployment record carries under "model".
RECORDED_MODEL_KEYS = ("architecture", "checkpoint", "model_name", "checkpoint_sha256")
# The calibration every record names: the temperature and these files' hashes.
CALIBRATION_FILES = ("calibration.json", "class_names.json")

Stage = Literal["base", "final"]

# --- Value checks: each returns None, or what is wrong with the value ----------

Check = Callable[[Any], str | None]
_SHA256 = re.compile(r"[0-9a-f]{64}")


def _hex_sha256(value: Any) -> str | None:
    if isinstance(value, str) and _SHA256.fullmatch(value):
        return None
    return f"must be a 64-character lowercase hex SHA-256, found {value!r}"


def _text(value: Any) -> str | None:
    return None if isinstance(value, str) and value.strip() else (
        f"must be a non-empty string, found {value!r}"
    )


def _file_name(value: Any) -> str | None:
    if _text(value) is None and "/" not in value and "\\" not in value:
        return None
    return f"must be a plain file name, found {value!r}"


def _timestamp(value: Any) -> str | None:
    try:
        parsed = datetime.fromisoformat(value) if isinstance(value, str) else None
    except ValueError:
        parsed = None
    if parsed is None or parsed.tzinfo is None:
        return f"must be an ISO-8601 timestamp with a UTC offset, found {value!r}"
    return None


def _number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _finite_positive(value: Any) -> str | None:
    if _number(value) and math.isfinite(value) and value > 0:
        return None
    return f"must be a finite, positive number, found {value!r}"


def _count(value: Any) -> str | None:
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return None
    return f"must be a non-negative integer, found {value!r}"


def _object(value: Any) -> str | None:
    return None if isinstance(value, dict) else f"must be an object, found {value!r}"


def _object_or_null(value: Any) -> str | None:
    return None if value is None or isinstance(value, dict) else (
        f"must be an object or null, found {value!r}"
    )


def _string_list(value: Any) -> str | None:
    if isinstance(value, list) and all(_text(item) is None for item in value):
        return None
    return f"must be a list of non-empty strings, found {value!r}"


def _hash_map(value: Any) -> str | None:
    if not isinstance(value, dict):
        return f"must be an object of file name to SHA-256, found {value!r}"
    bad = {name: digest for name, digest in value.items() if _hex_sha256(digest)}
    return f"holds values that are not hex SHA-256 digests: {bad!r}" if bad else None


def _policy_values(value: Any) -> str | None:
    if (
        isinstance(value, dict)
        and value
        and all(_number(item) and math.isfinite(item) for item in value.values())
    ):
        return None
    return f"must be a non-empty object of finite thresholds, found {value!r}"


def _exactly(expected: Any) -> Check:
    def check(value: Any) -> str | None:
        if value is expected or (type(value) is type(expected) and value == expected):
            return None
        return f"must be {expected!r}, found {value!r}"

    return check


def _one_of(allowed: tuple[str, ...]) -> Check:
    def check(value: Any) -> str | None:
        return None if value in allowed else f"must be one of {list(allowed)}, found {value!r}"

    return check


# A nested object: these keys, each passing its check. Problems are reported
# under the nested field's path, e.g. "model.checkpoint_sha256".
Shape = Mapping[str, Check]


# --- The schema ------------------------------------------------------------------

MODEL_SHAPE: Shape = {
    "architecture": _text,
    "checkpoint": _file_name,
    "model_name": _text,
    "checkpoint_sha256": _hex_sha256,
}
CALIBRATION_SHAPE: Mapping[str, Check | Shape] = {
    "temperature": _finite_positive,
    "files_sha256": {name: _hex_sha256 for name in CALIBRATION_FILES},
}

# Every operation records: what it is, the model identity, the calibration and
# the routing it leaves in place, and that the installed files read back.
COMMON_FIELDS: Mapping[str, Check | Shape] = {
    "schema_version": _exactly(SCHEMA_VERSION),
    "operation": _one_of(OPERATIONS),
    "model": MODEL_SHAPE,
    "calibration": CALIBRATION_SHAPE,
    "policy": _policy_values,
    "hard_class_count": _count,
    "confusion_pair_count": _count,
    "decision_layer_fingerprint": _hex_sha256,
    "deployed_files_sha256": _hash_map,
    "artifact_files_verified": _exactly(True),
    "live_service": _text,
}

# Policy deploys and promotions both record when, and the policy run and the
# producer evidence that bind the deployed policy to a model.
_EVIDENCE_FIELDS: Mapping[str, Check | Shape] = {
    "deployed_at": _timestamp,
    "source_run": _text,
    "source_provenance_sha256": _hex_sha256,
    "source_files_sha256": _hash_map,
    "policy_evidence": _object,
}

OPERATION_FIELDS: Mapping[str, Mapping[str, Check | Shape]] = {
    "policy_deploy": {
        **_EVIDENCE_FIELDS,
        "policy_evidence_matches_served_model": _text,
        "served_model": _object,
        "served_model_files_sha256": _hash_map,
        "untouched": _string_list,
    },
    "model_promotion": {
        **_EVIDENCE_FIELDS,
        "model_run": _text,
        "calibration_temperature": _finite_positive,
        "model_source_files_sha256": _hash_map,
        "checks": {
            "policy_belongs_to_model": _text,
            "class_names": _text,
            "checkpoint_fits_architecture": _text,
            "policy_evidence_matches_model": _text,
        },
        "previous_model": _object,
    },
    "restore": {
        "restored_at": _timestamp,
        "restored_from": _text,
        "backup_operation": _text,
        "backup_created_at": _timestamp,
        "restored_files_sha256": _hash_map,
        "removed_files": _string_list,
        "restored_deployment_provenance": _object_or_null,
        "previous_model": _object,
    },
}

# Checked when present; an operation writes them only in some cases.
OPTIONAL_FIELDS: Mapping[str, Mapping[str, Check]] = {
    "policy_deploy": {"policy_belongs_to_served_model": _text},
    "model_promotion": {},
    "restore": {"checkpoint_fits_architecture": _text},
}

# What the final record says about the backup it made.
BACKUP_FIELDS: Mapping[str, Check] = {"backup_dir": _text, "replaced_files_sha256": _hash_map}
OPTIONAL_BACKUP_FIELDS: Mapping[str, Check] = {"replaced_provenance_sha256": _hex_sha256}
# These always replace a file the target already has (its class names, at
# least), so their final record must name a backup. A policy-only deploy to a
# target without decision-layer files makes none.
BACKUP_REQUIRED = frozenset({"model_promotion", "restore"})


def _field_problems(
    record: Mapping[str, Any], fields: Mapping[str, Check | Shape], *, required: bool, prefix=""
) -> list[tuple[str, str | None]]:
    """``(field path, problem)`` for each field; a ``None`` problem means missing."""
    problems: list[tuple[str, str | None]] = []
    for name, check in fields.items():
        path = f"{prefix}{name}"
        if name not in record:
            if required:
                problems.append((path, None))
            continue
        value = record[name]
        if isinstance(check, Mapping):
            if not isinstance(value, dict):
                problems.append((path, f"must be an object, found {value!r}"))
            else:
                problems += _field_problems(value, check, required=True, prefix=f"{path}.")
            continue
        problem = check(value)
        if problem is not None:
            problems.append((path, problem))
    return problems


def schema_problems(record: Any, stage: Stage) -> list[tuple[str, str | None]]:
    """Every way ``record`` falls short of the schema at ``stage``.

    Returns:
        ``(field path, problem)`` pairs, where a ``None`` problem means the
        field is missing. Empty when the record is valid.
    """
    if not isinstance(record, Mapping):
        return [("record", f"must be an object, found {type(record).__name__}")]
    problems = _field_problems(record, COMMON_FIELDS, required=True)
    operation = record.get("operation")
    if operation not in OPERATION_FIELDS:
        return problems
    problems += _field_problems(record, OPERATION_FIELDS[operation], required=True)
    problems += _field_problems(record, OPTIONAL_FIELDS[operation], required=False)
    if stage == "final":
        names_backup = operation in BACKUP_REQUIRED or "backup_dir" in record
        problems += _field_problems(record, BACKUP_FIELDS, required=names_backup)
        problems += _field_problems(record, OPTIONAL_BACKUP_FIELDS, required=False)
    return problems


def describe(problems: list[tuple[str, str | None]]) -> str:
    """One line naming each missing or malformed field."""
    return "; ".join(f"{path}: {problem or 'missing'}" for path, problem in problems)


def validate_record(record: dict[str, Any], stage: Stage) -> dict[str, Any]:
    """Require a record about to be written to match the schema; return it.

    Args:
        record: The deployment record.
        stage: ``"base"`` -- the complete record before a backup exists -- or
            ``"final"`` -- the record naming its backup, before any target file
            is replaced.

    Raises:
        DeployError: Naming every missing or malformed field. The caller has
            not changed the target yet, so a refused record changes nothing.
    """
    problems = schema_problems(record, stage)
    if problems:
        raise DeployError(
            f"Refusing to write a {PROVENANCE_FILE} that does not match schema "
            f"{SCHEMA_VERSION} ({stage} record): {describe(problems)}. The target is unchanged."
        )
    return record


# --- Building records ------------------------------------------------------------


def recorded_model(identity: Mapping[str, Any]) -> dict[str, Any]:
    """The ``model`` entry of a record: the authoritative identity, in one shape."""
    return {key: identity[key] for key in RECORDED_MODEL_KEYS}


def recorded_calibration(
    temperature: Any, files_sha256: Mapping[str, Any]
) -> dict[str, Any]:
    """The ``calibration`` entry of a record: the temperature and its files' hashes."""
    return {
        "temperature": temperature,
        "files_sha256": {
            name: files_sha256[name] for name in CALIBRATION_FILES if name in files_sha256
        },
    }


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
        "schema_version": SCHEMA_VERSION,
        "operation": "policy_deploy",
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
        "calibration": recorded_calibration(
            served_model.get("temperature"), served_model_files_sha256
        ),
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
        "schema_version": SCHEMA_VERSION,
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
        "calibration": recorded_calibration(
            calibration_temperature, model_source_files_sha256
        ),
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
    calibration_temperature: float,
    calibration_files_sha256: Mapping[str, str],
    policy: dict[str, float],
    hard: list[str],
    pairs: list[dict[str, str]],
    restored_deployment_provenance: Any,
) -> dict[str, Any]:
    """The record of a restore, embedding the provenance it puts back."""
    return {
        "schema_version": SCHEMA_VERSION,
        "operation": "restore",
        "restored_at": restored_at,
        "restored_from": restored_from,
        "backup_operation": backup_record.get("operation"),
        "backup_created_at": backup_record.get("created_at"),
        "restored_files_sha256": restored_files_sha256,
        "removed_files": removed_files,
        "model": recorded_model(model),
        "calibration": recorded_calibration(calibration_temperature, calibration_files_sha256),
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


def finalised_record(
    record: dict[str, Any], backup: Path | None, prior: Mapping[str, str | None]
) -> dict[str, Any]:
    """Name the backup in the record and validate the result as the final record.

    The operations pass this to ``install.install`` as its ``record_for_backup``
    callback, so the final record is validated after the backup exists and
    before any target file is replaced; a refusal removes that backup.
    """
    return validate_record(add_backup(record, backup, prior), "final")


# --- Reading records: strict for the schema, normalised for history ------------

# Where a record written before the schema (no schema_version) kept data the
# schema now names, as key paths into that record, tried in order. Only the
# record's own values are read: a field it never captured stays absent and is
# listed in the normalised record's "limits", never filled from today's files.
HISTORICAL_SOURCES: Mapping[str, tuple[tuple[str, ...], ...]] = {
    # Promotion, restore and (from 2026-10-11) policy-only records wrote
    # "model"; policy-only records from 2026-10-10 wrote only "served_model".
    "model": (("model",), ("served_model",)),
    "calibration.temperature": (("calibration_temperature",), ("served_model", "temperature")),
    # Per file: promotion and restore hashed what they installed, policy-only
    # deploys hashed the served model's files they left in place.
    "calibration.files_sha256": (("deployed_files_sha256",), ("served_model_files_sha256",)),
}
# Before the discriminator, promotion and restore records already named their
# operation; only policy-only deploys wrote "untouched".
POLICY_ONLY_MARKER = "untouched"


def _at(record: Mapping[str, Any], path: tuple[str, ...]) -> Any:
    value: Any = record
    for key in path:
        value = value.get(key) if isinstance(value, Mapping) else None
    return value


def _normalise_historical(record: Mapping[str, Any]) -> dict[str, Any]:
    """Map a pre-schema record into the schema's shape, listing what it lacks."""
    normalised = dict(record)
    read_from: dict[str, str] = {}
    limits: list[str] = []

    operation = record.get("operation")
    if operation is None and POLICY_ONLY_MARKER in record:
        operation = "policy_deploy"
        read_from["operation"] = (
            f"inferred from '{POLICY_ONLY_MARKER}', which only policy-only deploys wrote"
        )
    normalised["operation"] = operation
    if operation is None:
        limits.append("no operation recorded")

    normalised.pop("model", None)
    for path in HISTORICAL_SOURCES["model"]:
        identity = _at(record, path)
        if isinstance(identity, Mapping) and identity.get("checkpoint_sha256"):
            normalised["model"] = {k: identity[k] for k in RECORDED_MODEL_KEYS if k in identity}
            if path != ("model",):
                read_from["model"] = ".".join(path)
            break
    else:
        if "model" in record or "served_model" in record:
            limits.append("model recorded without a checkpoint_sha256; not used")

    calibration: dict[str, Any] = {}
    for path in HISTORICAL_SOURCES["calibration.temperature"]:
        temperature = _at(record, path)
        if temperature is not None:
            calibration["temperature"] = temperature
            read_from["calibration.temperature"] = ".".join(path)
            break
    files: dict[str, Any] = {}
    for name in CALIBRATION_FILES:
        for path in HISTORICAL_SOURCES["calibration.files_sha256"]:
            hashes = _at(record, path)
            if isinstance(hashes, Mapping) and name in hashes:
                files[name] = hashes[name]
                read_from[f"calibration.files_sha256.{name}"] = ".".join(path)
                break
    if files:
        calibration["files_sha256"] = files
    if calibration:
        calibration.setdefault("files_sha256", {})
        normalised["calibration"] = calibration
    else:
        normalised.pop("calibration", None)

    # Everything else the schema asks for, judged as the final record of the
    # inferred operation; schema_version and operation are accounted for above.
    probe = {**normalised, "schema_version": SCHEMA_VERSION}
    for path, problem in schema_problems(probe, "final"):
        if path in ("schema_version", "operation"):
            continue
        if problem is None:
            limits.append(f"no {path} recorded")
        else:
            limits.append(f"{path} as recorded is unusable: {problem}")
    normalised.update({"historical": True, "read_from": read_from, "limits": limits})
    return normalised


def normalise_record(record: Any) -> dict[str, Any]:
    """The reader's view of a deployment record, whatever wrote it.

    - A record with ``schema_version`` must match the schema exactly, as the
      final record its operation validated before writing it.
    - A record without one predates the schema. It is mapped into the schema's
      shape from the record's own fields (``HISTORICAL_SOURCES``), and
      ``limits`` names every field it never captured, or captured unusably.
      Nothing is invented: an absent field stays absent.
    - ``None`` -- no record at all, as before deployment records existed --
      normalises to a record that holds only its limits.

    Every normalised record carries ``historical``, ``read_from`` (which old
    field each mapped value came from) and ``limits``.

    Raises:
        DeployError: If the record is not an object, declares a schema version
            this code does not know, or declares this one but does not match it.
    """
    if record is None:
        normalised = _normalise_historical({})
        normalised["limits"].insert(0, f"no {PROVENANCE_FILE}: nothing was recorded")
        return normalised
    if not isinstance(record, Mapping):
        raise DeployError(f"{PROVENANCE_FILE} must hold an object, not {type(record).__name__}.")
    if "schema_version" not in record:
        return _normalise_historical(record)
    if record["schema_version"] != SCHEMA_VERSION or isinstance(record["schema_version"], bool):
        raise DeployError(
            f"{PROVENANCE_FILE} declares schema_version {record['schema_version']!r}; this code "
            f"reads schema {SCHEMA_VERSION} and records written before it."
        )
    problems = schema_problems(record, "final")
    if problems:
        raise DeployError(
            f"{PROVENANCE_FILE} declares schema {SCHEMA_VERSION} but does not match it "
            f"({describe(problems)}); it changed after it was written. Redeploy."
        )
    return {**record, "historical": False, "read_from": {}, "limits": []}
