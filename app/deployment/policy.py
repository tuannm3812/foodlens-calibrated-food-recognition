"""Policy-only deploy: install a recalibrated decision layer for the model already served.

Only the three decision-layer files and the provenance record are written; the
checkpoint, class names, calibration and ``model.json`` are never touched, so a
policy deploy can never change which model is served. The policy must belong to
that model: its producer evidence must match the served checkpoint,
architecture, class order and temperature, and on a promoted target its
predictions must live in the model run ``model.json`` names.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from . import identity, records
from .errors import DeployError
from .install import check_installed_hashes, install, prior_state, staging_dir
from .paths import (
    DEPLOYED_FILES,
    PROVENANCE_FILE,
    display,
    now_utc,
    read_json,
    sha256,
    write_json_file,
)


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

    policy, hard, pairs = identity.read_policy_files(source)
    identity.check_class_coverage(target, hard, pairs)
    # Only a promoted target has a manifest; the legacy deployment does not.
    model_run = identity.served_model_run(target)
    model_check = (
        identity.check_policy_belongs_to_model(source, provenance, model_run)
        if model_run
        else None
    )
    evidence = identity.policy_binding(provenance, provenance_path)
    served_identity, served_binding = identity.served_model(target)
    evidence_check = identity.check_evidence_matches(
        evidence, served_binding, source, "the target's served model"
    )

    record = records.policy_deploy_record(
        deployed_at=now_utc(),
        source_run=display(source),
        source_provenance_sha256=sha256(provenance_path),
        source_files_sha256={name: sha256(source / name) for name in DEPLOYED_FILES},
        policy=policy,
        hard=hard,
        pairs=pairs,
        policy_evidence=evidence,
        served_model=served_identity,
        evidence_check=evidence_check,
        served_model_files_sha256={
            served_identity["checkpoint"]: served_identity["checkpoint_sha256"],
            "calibration.json": sha256(target / "calibration.json"),
            "class_names.json": sha256(target / "class_names.json"),
        },
        model_check=model_check,
    )
    if dry_run:
        record["dry_run"] = True
        return record

    # Verify the staged files before the target is touched.
    with staging_dir(target) as staging:
        write_json_file(staging / "decision_policy.json", policy)
        write_json_file(staging / "hard_classes.json", hard)
        write_json_file(staging / "confusion_pairs.json", pairs)
        fingerprint = identity.verify_through_backend(staging, policy, hard, pairs)
        records.mark_files_verified(
            record,
            {name: sha256(staging / name) for name in DEPLOYED_FILES},
            fingerprint,
            target,
        )

        def post_check() -> None:
            if identity.verify_through_backend(target, policy, hard, pairs) != fingerprint:
                raise DeployError(
                    "installed decision layer fingerprint differs from the staged one"
                )
            check_installed_hashes(target, record["deployed_files_sha256"])

        prior = prior_state(target, (*DEPLOYED_FILES, PROVENANCE_FILE))
        install(
            target,
            staging,
            lambda backup: records.add_backup(record, backup, prior),
            DEPLOYED_FILES,
            prior,
            post_check,
        )
    return record
