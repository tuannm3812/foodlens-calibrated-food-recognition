"""The deployment record schema in app/deployment/records.py.

Writers use the strict schema: every record carries `schema_version` and an
`operation` discriminator, the common model identity, calibration and routing
fields, and its operation's evidence and backup fields. These tests build
records from the module's own builders, with made-up values, so they need no
torch, no artifacts and no target directory.
"""

from __future__ import annotations

import copy
import json
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
        ("operation", [], "operation: must be one of"),
        ("operation", {}, "operation: must be one of"),
        ("operation", {"model_promotion": 1}, "operation: must be one of"),
        ("decision_layer_fingerprint", "not-a-hash", "decision_layer_fingerprint: must be"),
        ("hard_class_count", -1, "hard_class_count: must be a non-negative integer"),
        ("artifact_files_verified", "yes", "artifact_files_verified: must be True"),
        ("policy", {**POLICY, "auto_confidence": math.nan}, "policy: must hold finite"),
        ("policy", {**POLICY, "auto_confidence": 10**400}, "policy: must hold finite"),
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


@pytest.mark.parametrize(
    "value", [math.nan, math.inf, -math.inf, 0, -0.5, True, "0.884", None, 10**400]
)
def test_a_temperature_that_is_not_finite_and_positive_is_refused(value) -> None:
    record = finish(restore_record())
    record["calibration"]["temperature"] = value
    assert "calibration.temperature: must be a finite, positive number" in rejected(record)


# Valid JSON that Python cannot hash or convert: a list or object where a string
# belongs, and an integer past the float range. Each is a field problem, never
# a TypeError or OverflowError escaping the validator.


@pytest.mark.parametrize("stage", ["base", "final"])
@pytest.mark.parametrize("value", [[], {}, ["model_promotion"]])
def test_an_operation_that_is_not_a_string_is_a_field_problem(stage, value) -> None:
    record = finish(promotion_record()) if stage == "final" else promotion_record()
    record["operation"] = value
    assert f"operation: must be one of {list(records.OPERATIONS)}" in rejected(record, stage)


def test_a_promotion_calibration_temperature_past_the_float_range_is_refused() -> None:
    record = finish(promotion_record())
    record["calibration_temperature"] = 10**400
    assert "calibration_temperature: must be a finite, positive number" in rejected(record)


def test_an_oversized_value_is_shown_briefly() -> None:
    record = finish(restore_record())
    record["calibration"]["temperature"] = 10**4000
    message = rejected(record)
    assert "calibration.temperature: must be a finite, positive number, found 1000" in message
    assert len(message) < 1000


# The recorded policy is the runtime's: verify-live trusts it as a statement of
# what routes predictions, so it must be one the runtime could have loaded.


@pytest.mark.parametrize(
    ("policy", "fragment"),
    [
        ({"not_a_threshold": 2.0}, "must hold exactly the thresholds"),
        ({}, "must hold exactly the thresholds"),
        ({k: v for k, v in POLICY.items() if k != "margin_threshold"}, "must hold exactly"),
        ({**POLICY, "extra": 0.5}, "must hold exactly the thresholds"),
        ({**POLICY, "auto_confidence": 1.5}, "auto_confidence is not"),
        ({**POLICY, "margin_threshold": -0.1}, "margin_threshold is not"),
        ({**POLICY, "suggest_confidence": "0.35"}, "suggest_confidence is not"),
        ({**POLICY, "suggest_confidence": True}, "suggest_confidence is not"),
        ({**POLICY, "suggest_confidence": 0.8}, "suggest_confidence above auto_confidence"),
        ([0.7, 0.35, 0.05], "must hold exactly the thresholds"),
    ],
)
def test_a_policy_the_runtime_could_not_load_is_refused(policy, fragment) -> None:
    record = finish(policy_record())
    record["policy"] = policy
    message = rejected(record)
    assert "policy: " in message and fragment in message


def test_the_recorded_policy_rule_matches_the_deploy_input_rule() -> None:
    """A recorded policy is valid exactly when runtime_policy would have produced it."""
    from app.deployment import identity

    cases = [
        dict(POLICY),
        {**POLICY, "auto_confidence": 1.0, "suggest_confidence": 1.0},
        {**POLICY, "margin_threshold": 0},
        {**POLICY, "auto_confidence": 1.5},
        {**POLICY, "suggest_confidence": 0.8},
        {**POLICY, "margin_threshold": math.inf},
        {**POLICY, "margin_threshold": False},
    ]
    for policy in cases:
        try:
            identity.runtime_policy(policy, Path("policy.json"))
            accepted = True
        except DeployError:
            accepted = False
        assert (records.policy_problem(policy) is None) == accepted, policy


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


# --- Reading: strict for the schema, normalised for history --------------------
#
# The fixtures are copies of the real records' JSON (app/artifacts is
# gitignored, so CI does not have the originals): the 2026-10-10 promotion
# record, and the 2026-10-09 policy-only record kept in that promotion's
# backup. The 2026-10-09 deploy's own backup, replaced_20261009T224436Z/, holds
# no record at all, which normalises from None.

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "deployment_records"
NORMALISED_KEYS = {"historical", "read_from", "limits"}


def fixture(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def leaves(value, path=()):
    """Every scalar in a JSON value, by path."""
    if isinstance(value, dict):
        for key, item in value.items():
            yield from leaves(item, (*path, key))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from leaves(item, (*path, index))
    else:
        yield path, value


def assert_nothing_invented(original: dict, normalised: dict) -> None:
    """Every normalised value is the original's, or was read from one of its fields."""
    for key, value in normalised.items():
        if key in NORMALISED_KEYS:
            continue
        if key in original:
            if key not in ("model", "calibration", "operation"):
                assert value == original[key], key
            continue
        assert key in ("model", "calibration", "operation"), f"invented {key}"
    original_scalars = {value for _, value in leaves(original) if value is not None}
    for path, value in leaves({k: normalised.get(k) for k in ("model", "calibration")}):
        if value is not None:
            assert value in original_scalars, f"{path}={value!r} is not in the original record"


def test_the_2026_10_10_promotion_record_normalises_with_two_limits() -> None:
    original = fixture("2026-10-10_model_promotion.json")
    pristine = copy.deepcopy(original)
    normalised = records.normalise_record(original)

    assert original == pristine, "normalising never mutates the record"
    assert normalised["historical"] is True
    assert normalised["operation"] == "model_promotion"
    assert normalised["model"] == original["model"]
    assert normalised["calibration"] == {
        "temperature": original["calibration_temperature"],
        "files_sha256": {
            "calibration.json": original["deployed_files_sha256"]["calibration.json"],
            "class_names.json": original["deployed_files_sha256"]["class_names.json"],
        },
    }
    assert normalised["decision_layer_fingerprint"] == original["decision_layer_fingerprint"]
    assert normalised["read_from"] == {
        "calibration.temperature": "calibration_temperature",
        "calibration.files_sha256.calibration.json": "deployed_files_sha256",
        "calibration.files_sha256.class_names.json": "deployed_files_sha256",
    }
    # It predates producer evidence: those limits are stated, not filled.
    assert normalised["limits"] == [
        "no policy_evidence recorded",
        "no checks.policy_evidence_matches_model recorded",
    ]
    assert "policy_evidence" not in normalised
    assert_nothing_invented(original, normalised)


def test_the_2026_10_09_policy_only_record_normalises_with_its_limits() -> None:
    original = fixture("2026-10-09_policy_only.json")
    normalised = records.normalise_record(original)

    assert normalised["historical"] is True
    assert normalised["operation"] == "policy_deploy"
    assert "untouched" in normalised["read_from"]["operation"]
    for absent in ("model", "calibration", "decision_layer_fingerprint", "policy_evidence"):
        assert absent not in normalised, absent
    assert normalised["limits"] == [
        "no model recorded",
        "no calibration recorded",
        "no decision_layer_fingerprint recorded",
        "no artifact_files_verified recorded",
        "no live_service recorded",
        "no policy_evidence recorded",
        "no policy_evidence_matches_served_model recorded",
        "no served_model recorded",
        "no served_model_files_sha256 recorded",
    ]
    # backend_verified is that era's name, with a weaker meaning; it is kept
    # as recorded, not promoted to artifact_files_verified.
    assert normalised["backend_verified"] is True
    assert_nothing_invented(original, normalised)


def test_a_backup_without_any_record_normalises_to_its_limits() -> None:
    normalised = records.normalise_record(None)
    assert normalised["historical"] is True
    assert normalised["operation"] is None
    assert normalised["limits"][:3] == [
        "no deployment_provenance.json: nothing was recorded",
        "no operation recorded",
        "no model recorded",
    ]
    assert "no decision_layer_fingerprint recorded" in normalised["limits"]
    assert set(normalised) == {"operation", *NORMALISED_KEYS}


def served_model_only_record() -> dict:
    """A policy-only record from 2026-10-10: the identity under served_model alone."""
    record = finish(policy_record())
    for key in ("schema_version", "operation", "model", "calibration"):
        del record[key]
    return record


def test_a_served_model_only_record_reads_its_identity_from_served_model() -> None:
    original = served_model_only_record()
    normalised = records.normalise_record(original)
    assert normalised["operation"] == "policy_deploy"
    assert normalised["model"] == MODEL
    assert normalised["read_from"]["model"] == "served_model"
    assert normalised["calibration"] == {
        "temperature": 0.884,
        "files_sha256": {"calibration.json": digest("b"), "class_names.json": digest("c")},
    }
    assert normalised["read_from"]["calibration.temperature"] == "served_model.temperature"
    assert normalised["limits"] == []
    assert_nothing_invented(original, normalised)


def test_an_unusable_historical_identity_is_a_limit_not_a_guess() -> None:
    original = served_model_only_record()
    del original["served_model"]["checkpoint_sha256"]
    normalised = records.normalise_record(original)
    assert "model" not in normalised
    assert "model recorded without a checkpoint_sha256; not used" in normalised["limits"]


@pytest.mark.parametrize("operation", sorted(BUILDERS))
def test_a_current_record_round_trips_through_the_reader(operation: str) -> None:
    record = finish(BUILDERS[operation]())
    normalised = records.normalise_record(json.loads(json.dumps(record)))
    assert normalised["historical"] is False
    assert normalised["limits"] == [] and normalised["read_from"] == {}
    assert {k: v for k, v in normalised.items() if k not in NORMALISED_KEYS} == record


@pytest.mark.parametrize("version", [2, 0, "1", True])
def test_an_unknown_schema_version_is_refused_on_read(version) -> None:
    record = finish(policy_record())
    record["schema_version"] = version
    with pytest.raises(DeployError, match="schema"):
        records.normalise_record(record)


def test_a_current_record_edited_after_writing_is_refused_on_read() -> None:
    record = finish(promotion_record())
    del record["model"]
    with pytest.raises(DeployError, match="model: missing"):
        records.normalise_record(record)


@pytest.mark.parametrize("value", [[], {}])
def test_a_current_record_with_an_unhashable_operation_is_refused_on_read(value) -> None:
    record = finish(policy_record())
    record["operation"] = value
    with pytest.raises(DeployError, match="operation: must be one of"):
        records.normalise_record(record)


def test_a_current_record_with_an_oversized_temperature_is_refused_on_read() -> None:
    record = finish(policy_record())
    record["calibration"]["temperature"] = 10**400
    with pytest.raises(DeployError, match="calibration.temperature: must be a finite"):
        records.normalise_record(record)


@pytest.mark.parametrize("value", [[], {}])
def test_a_historical_record_with_an_unhashable_operation_names_it_as_a_limit(value) -> None:
    original = fixture("2026-10-10_model_promotion.json")
    original["operation"] = value
    normalised = records.normalise_record(original)
    assert normalised["historical"] is True
    assert normalised["operation"] == value, "kept as recorded, not replaced"
    assert any(
        limit.startswith("operation as recorded is unusable: must be one of")
        for limit in normalised["limits"]
    )


def test_a_historical_record_with_an_oversized_temperature_names_it_as_a_limit() -> None:
    original = fixture("2026-10-10_model_promotion.json")
    original["calibration_temperature"] = 10**400
    limits = records.normalise_record(original)["limits"]
    assert any(
        limit.startswith("calibration.temperature as recorded is unusable") for limit in limits
    )


def test_a_record_that_is_not_an_object_is_refused_on_read() -> None:
    with pytest.raises(DeployError, match="must hold an object"):
        records.normalise_record(["not", "a", "record"])
