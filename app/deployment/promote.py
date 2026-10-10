"""Model promotion: install a model and the policy fitted on it as one unit.

The checkpoint, ``calibration.json``, ``class_names.json``, ``model.json`` and
the three policy files go through one stage, verify, back up, install,
post-check and rollback sequence. A policy is fitted to one model's
confidences, so a model and its policy never deploy apart.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from . import identity, records
from .errors import DeployError
from .install import check_installed_hashes, install, prior_state, staging_dir
from .paths import (
    DEPLOYED_FILES,
    MANIFEST_FILE,
    MODEL_FILES,
    PROVENANCE_FILE,
    RESERVED_NAMES,
    display,
    now_utc,
    read_json,
    sha256,
    write_json_file,
)


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
    policy, hard, pairs = identity.read_policy_files(source)
    policy_check = identity.check_policy_belongs_to_model(source, provenance, model_run)
    evidence = identity.policy_binding(provenance, provenance_path)

    checkpoint = model_run_checkpoint(model_run, checkpoint_name)
    temperature = identity.validated_temperature(
        read_json(model_run / "calibration.json"), model_run / "calibration.json"
    )
    class_names = identity.validated_class_names(
        read_json(model_run / "class_names.json"), model_run / "class_names.json"
    )
    names_check = identity.check_class_names_match(target, class_names)
    identity.check_class_coverage(target, hard, pairs)

    manifest_fields = {
        "architecture": architecture,
        "checkpoint": checkpoint.name,
        "model_name": model_name,
        "model_run": display(model_run),
    }
    manifest = identity.validate_manifest(manifest_fields, Path(MANIFEST_FILE))

    checkpoint_sha = sha256(checkpoint)
    # What the policy's producer evidence must match: the checkpoint bytes, the
    # architecture, the class order and the calibration being installed. It is
    # checked right after the checkpoint is shown to fit the architecture.
    model = identity.model_binding(
        checkpoint_sha256=checkpoint_sha,
        architecture=architecture,
        class_names=class_names,
        temperature=temperature,
    )
    record = records.promotion_record(
        deployed_at=now_utc(),
        source_run=display(source),
        source_provenance_sha256=sha256(provenance_path),
        source_files_sha256={name: sha256(source / name) for name in DEPLOYED_FILES},
        policy=policy,
        hard=hard,
        pairs=pairs,
        model_run=display(model_run),
        model={
            "architecture": architecture,
            "checkpoint": checkpoint.name,
            "model_name": model_name,
            "checkpoint_sha256": checkpoint_sha,
        },
        calibration_temperature=temperature,
        model_source_files_sha256={
            checkpoint.name: checkpoint_sha,
            "calibration.json": sha256(model_run / "calibration.json"),
            "class_names.json": sha256(model_run / "class_names.json"),
        },
        policy_check=policy_check,
        names_check=names_check,
        policy_evidence=evidence,
    )
    what = "the model being promoted"
    if dry_run:
        record["checks"]["checkpoint_fits_architecture"] = identity.check_checkpoint_fits(
            checkpoint, architecture
        )
        record["checks"]["policy_evidence_matches_model"] = identity.check_evidence_matches(
            evidence, model, source, what
        )
        record["dry_run"] = True
        return record

    installed = (checkpoint.name, *MODEL_FILES, *DEPLOYED_FILES)
    with staging_dir(target) as staging:
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
        record["checks"]["checkpoint_fits_architecture"] = identity.check_checkpoint_fits(
            staging / checkpoint.name, architecture, f"{display(checkpoint)} (staged copy)"
        )
        record["checks"]["policy_evidence_matches_model"] = identity.check_evidence_matches(
            evidence, model, source, what
        )
        fingerprint = identity.verify_through_backend(staging, policy, hard, pairs)
        identity.verify_model_files(staging, manifest, temperature, class_names)
        records.mark_files_verified(
            record,
            {
                name: checkpoint_sha if name == checkpoint.name else sha256(staging / name)
                for name in installed
            },
            fingerprint,
            target,
        )

        # Keep the previously served checkpoint in the backup too, so --restore
        # does not depend on it surviving in the target.
        try:
            previous_checkpoint = (identity.read_manifest(target)["checkpoint"],)
        except DeployError:
            previous_checkpoint = ()
        prior = prior_state(target, (*installed, *previous_checkpoint, PROVENANCE_FILE))
        record["previous_model"] = identity.current_model(target, prior)

        def post_check() -> None:
            if identity.verify_through_backend(target, policy, hard, pairs) != fingerprint:
                raise DeployError(
                    "installed decision layer fingerprint differs from the staged one"
                )
            identity.verify_model_files(target, manifest, temperature, class_names)
            check_installed_hashes(target, record["deployed_files_sha256"])

        install(
            target,
            staging,
            lambda backup: records.add_backup(record, backup, prior),
            installed,
            prior,
            post_check,
            operation="model_promotion",
        )
    return record
