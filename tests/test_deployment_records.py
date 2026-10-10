"""The deployment record schema in app/deployment/records.py.

Writers use the strict schema: every record carries `schema_version` and an
`operation` discriminator, the common model identity, calibration and routing
fields, and its operation's evidence and backup fields. These tests build
records from the module's own builders, with made-up values, so they need no
torch, no artifacts and no target directory.
"""

from __future__ import annotations

import copy
import math
from pathlib import Path

import pytest

from app.deployment import records
from app.deployment.errors import DeployError

TARGET = Path("/tmp/does-not-matter/artifacts")
POLICY = {"auto_confidence": 0.7, "suggest_confidence": 0.35, "margin_threshold": 0.05}
HARD = ["steak"]
PAIRS = [{"actual": "steak", "predicted": "filet_mignon"}]
WHEN = "2026-10-11T09:00:00+00:00"


def digest(seed: str) -> str:
    return (seed * 64)[:64]


MODEL = {
    "architecture": "convnext_tiny",
    "checkpoint": "a3b_best.pth",
    "model_name": "a3b_convnext_tiny",
    "checkpoint_sha256": digest("a"),
}
MODEL_FILES = {
    "a3b_best.pth": digest("a"),
    "calibration.json": digest("b"),
    "class_names.json": digest("c"),
}
DEPLOYED = {"decision_policy.json": digest("d"), "hard_classes.json": digest("e")}
PRIOR = {"decision_policy.json": digest("1"), "deployment_provenance.json": digest("2")}


def finish(record: dict, backup: Path | None = TARGET / "replaced_x") -> dict:
    """Mark the record's files verified and name its backup, as an operation does."""
    records.mark_files_verified(record, dict(DEPLOYED), digest("f"), TARGET)
    return records.add_backup(record, backup, PRIOR)


def policy_record() -> dict:
    return records.policy_deploy_record(
        deployed_at=WHEN,
        source_run="results/run/closure",
        source_provenance_sha256=digest("3"),
        source_files_sha256=dict(DEPLOYED),
        policy=dict(POLICY),
        hard=HARD,
        pairs=PAIRS,
        policy_evidence={"architecture": "convnext_tiny"},
        served_model={"manifest": "model.json", **MODEL, "temperature": 0.884},
        evidence_check="match",
        served_model_files_sha256=dict(MODEL_FILES),
        model_check="fit and eval predictions are inside the run",
    )


def promotion_record() -> dict:
    record = records.promotion_record(
        deployed_at=WHEN,
        source_run="results/run/closure",
        source_provenance_sha256=digest("3"),
        source_files_sha256=dict(DEPLOYED),
        policy=dict(POLICY),
        hard=HARD,
        pairs=PAIRS,
        model_run="results/run",
        model=MODEL,
        calibration_temperature=0.884,
        model_source_files_sha256=dict(MODEL_FILES),
        policy_check="inside the run",
        names_check="identical",
        policy_evidence={"architecture": "convnext_tiny"},
    )
    record["checks"]["checkpoint_fits_architecture"] = "loads"
    record["checks"]["policy_evidence_matches_model"] = "match"
    record["previous_model"] = {"manifest": "legacy_default", "architecture": "resnet50"}
    return record


def restore_record() -> dict:
    record = records.restore_record(
        restored_at=WHEN,
        restored_from="app/artifacts/replaced_x",
        backup_record={"operation": "model_promotion", "created_at": WHEN},
        restored_files_sha256=dict(DEPLOYED),
        removed_files=["model.json"],
        model=MODEL,
        calibration_temperature=0.958,
        calibration_files_sha256={
            "calibration.json": digest("b"),
            "class_names.json": digest("c"),
        },
        policy=dict(POLICY),
        hard=HARD,
        pairs=PAIRS,
        restored_deployment_provenance={"deployed_at": "earlier"},
    )
    record["previous_model"] = {"manifest": "model.json", **MODEL}
    return record


BUILDERS = {
    "policy_deploy": policy_record,
    "model_promotion": promotion_record,
    "restore": restore_record,
}


@pytest.mark.parametrize("operation", sorted(BUILDERS))
def test_every_builder_writes_a_record_that_passes_both_stages(operation: str) -> None:
    record = BUILDERS[operation]()
    assert record["schema_version"] == records.SCHEMA_VERSION
    assert record["operation"] == operation
    records.mark_files_verified(record, dict(DEPLOYED), digest("f"), TARGET)
    assert records.validate_record(record, "base") is record
    assert records.validate_record(records.add_backup(record, TARGET / "b", PRIOR), "final")


def test_the_operation_values_are_the_ones_backups_already_use() -> None:
    # install.make_backup names a policy-only deploy "policy_deploy" in
    # backup_record.json; promotion and restore records already said so.
    assert records.OPERATIONS == ("policy_deploy", "model_promotion", "restore")


def test_the_calibration_entry_names_the_temperature_and_both_files() -> None:
    record = promotion_record()
    assert record["calibration"] == {
        "temperature": 0.884,
        "files_sha256": {"calibration.json": digest("b"), "class_names.json": digest("c")},
    }


def rejected(record: dict, stage: str = "final") -> str:
    with pytest.raises(DeployError) as info:
        records.validate_record(record, stage)
    return str(info.value)


@pytest.mark.parametrize("operation", sorted(BUILDERS))
def test_a_record_without_the_model_identity_is_refused(operation: str) -> None:
    record = finish(BUILDERS[operation]())
    del record["model"]
    message = rejected(record, "base")
    assert "model: missing" in message
    assert "base record" in message


@pytest.mark.parametrize(
    ("field", "value", "fragment"),
    [
        ("schema_version", 2, "schema_version: must be 1, found 2"),
        ("schema_version", "1", "schema_version: must be 1, found '1'"),
        ("schema_version", True, "schema_version: must be 1, found True"),
        ("operation", "policy", "operation: must be one of"),
        ("operation", None, "operation: must be one of"),
        ("decision_layer_fingerprint", "not-a-hash", "decision_layer_fingerprint: must be"),
        ("hard_class_count", -1, "hard_class_count: must be a non-negative integer"),
        ("artifact_files_verified", "yes", "artifact_files_verified: must be True"),
        ("policy", {"auto_confidence": math.nan}, "policy: must be a non-empty object"),
        ("deployed_at", "earlier", "deployed_at: must be an ISO-8601 timestamp"),
        ("deployed_at", "2026-10-11T09:00:00", "deployed_at: must be an ISO-8601 timestamp"),
    ],
)
def test_malformed_values_are_refused_by_field(field, value, fragment) -> None:
    record = finish(promotion_record())
    record[field] = value
    assert fragment in rejected(record)


@pytest.mark.parametrize(
    "value",
    [digest("A"), digest("a")[:63], digest("g"), 12345, None],
)
def test_a_checkpoint_hash_that_is_not_hex_sha256_is_refused(value) -> None:
    record = finish(policy_record())
    record["model"]["checkpoint_sha256"] = value
    assert "model.checkpoint_sha256: must be a 64-character lowercase hex SHA-256" in rejected(
        record
    )


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf, 0, -0.5, True, "0.884", None])
def test_a_temperature_that_is_not_finite_and_positive_is_refused(value) -> None:
    record = finish(restore_record())
    record["calibration"]["temperature"] = value
    assert "calibration.temperature: must be a finite, positive number" in rejected(record)


def test_a_missing_calibration_file_hash_is_named() -> None:
    record = finish(policy_record())
    del record["calibration"]["files_sha256"]["class_names.json"]
    assert "calibration.files_sha256.class_names.json: missing" in rejected(record)


def test_a_checkpoint_name_with_a_path_is_refused() -> None:
    record = finish(policy_record())
    record["model"]["checkpoint"] = "../elsewhere/a3b_best.pth"
    assert "model.checkpoint: must be a plain file name" in rejected(record)


@pytest.mark.parametrize(
    ("operation", "field"),
    [
        ("policy_deploy", "policy_evidence"),
        ("policy_deploy", "served_model_files_sha256"),
        ("model_promotion", "policy_evidence"),
        ("model_promotion", "previous_model"),
        ("restore", "restored_from"),
        ("restore", "removed_files"),
    ],
)
def test_each_operation_requires_its_own_fields(operation: str, field: str) -> None:
    record = finish(BUILDERS[operation]())
    del record[field]
    assert f"{field}: missing" in rejected(record)


def test_a_promotion_must_record_every_check() -> None:
    record = finish(promotion_record())
    del record["checks"]["policy_evidence_matches_model"]
    assert "checks.policy_evidence_matches_model: missing" in rejected(record)


@pytest.mark.parametrize("operation", ["model_promotion", "restore"])
def test_promotion_and_restore_must_name_a_backup_when_final(operation: str) -> None:
    record = BUILDERS[operation]()
    records.mark_files_verified(record, dict(DEPLOYED), digest("f"), TARGET)
    records.validate_record(record, "base")  # no backup yet: fine
    message = rejected(copy.deepcopy(record), "final")
    assert "backup_dir: missing" in message and "replaced_files_sha256: missing" in message


def test_a_policy_deploy_without_anything_to_back_up_is_valid() -> None:
    record = finish(policy_record(), backup=None)
    assert "backup_dir" not in record
    records.validate_record(record, "final")


def test_a_named_backup_must_carry_its_replaced_hashes() -> None:
    record = finish(policy_record())
    record["replaced_files_sha256"] = {"decision_policy.json": "corrupt"}
    assert "replaced_files_sha256: holds values that are not hex" in rejected(record)


def test_every_problem_is_reported_at_once() -> None:
    record = finish(policy_record())
    del record["model"]
    record["calibration"]["temperature"] = math.inf
    message = rejected(record)
    assert "model: missing" in message
    assert "calibration.temperature" in message


def test_records_imports_no_filesystem_module() -> None:
    # The record module validates values it is given; the boundary test pins its
    # sibling imports, this pins that validating needs no file access.
    source = Path(records.__file__).read_text(encoding="utf-8")
    for forbidden in ("import os", "import shutil", "import tempfile", "open("):
        assert forbidden not in source
