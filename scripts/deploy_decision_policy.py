"""Deploy a recalibrated decision-layer policy, or promote a model with its policy.

A recalibration run (``scripts/recalibrate_decision_layer.py``) writes its
outputs in the analysis format, which is not the format the backend reads:

- ``decision_policy.json`` is a one-element *list* of policy records carrying
  search metadata, while ``app.backend`` reads a single *dict* of thresholds.
  Copying it verbatim makes ``read_policy()`` raise inside ``load_runtime()``,
  whose ``except Exception`` turns every request into a demo fallback -- the app
  degrades silently instead of failing.
- ``hard_classes.json`` and ``confusion_pairs.json`` already match the runtime
  format and are validated, not converted.

This script converts and validates the three files, then:

1. stages them in a hidden directory inside the target and reads them back
   through the backend's own readers -- the target is untouched until this
   passes;
2. backs up whatever it is about to replace, including any existing
   ``deployment_provenance.json`` (``app/artifacts/`` is gitignored, so an
   overwritten file is otherwise unrecoverable), in a collision-free
   ``replaced_<UTC stamp>_<suffix>/`` directory, with a ``backup_record.json``
   naming each file's prior SHA-256 (or that it did not exist);
3. installs each file with an atomic ``os.replace`` and writes
   ``deployment_provenance.json`` recording the source run, its derivation
   provenance and the hash of every deployed file;
4. reads the installed target back through the backend readers.

If anything fails after installation starts, every original file is restored,
files that did not exist before are removed, the new backup directory is
deleted, and the command fails saying the deployment was rolled back. A failed
invocation therefore leaves the target exactly as it was.

**The file check is not proof of the live service.** Step 4 reads the files
freshly in the deploy process. A running API process caches the model and the
decision layer in ``inference._RUNTIME`` on first use and keeps serving them
until it restarts, so the record says ``artifact_files_verified`` and carries a
``live_service`` note, never a claim about the service. After a deploy,
stop/restart every API process, then run ``--verify-live URL``: it sends one
synthetic image to ``/predict/image`` (so a restarted process loads its runtime,
and a demo fallback fails the check), then requires ``/runtime/status`` to
report, as ``loaded_runtime``, the same decision-layer fingerprint as the target
files (and ``deployment_provenance.json``, when it records one), the same
model identity -- architecture, model name and checkpoint SHA-256 -- as the
target's ``model.json`` and checkpoint, and the same calibration: the probe's
temperature, the status block's cached temperature and the target's
``calibration.json`` must agree exactly, and the cached class-order hash must
match the target's. Model, routing and calibration are reported separately.

**Policy-only mode** (``--source`` alone) touches only the three decision-layer
files and the provenance record. The model checkpoint, class names, calibration
and ``model.json`` are never modified, so deploying a policy can never change
which model is served. The policy's producer evidence (recorded by the
rescorer, carried in its ``derivation_provenance.json``; see
``scripts/prediction_evidence.py``) must match the served model: the actual
checkpoint bytes' SHA-256, the architecture ``model.json`` names (the legacy
ResNet50 default without one), the target's class order and its
``calibration.json`` temperature. When the target carries a ``model.json``
with a ``model_run``, the predictions must also live in that run.

**Promotion mode** (``--source`` with ``--model-run``) installs the checkpoint,
``calibration.json``, ``class_names.json``, ``model.json`` and the three policy
files as one unit through the same steps. A policy is fitted to one model's
confidences, so a model and its policy never deploy apart. Before the target is
touched it requires that the policy run's ``derivation_provenance.json`` names
fit and eval predictions inside ``--model-run`` (with their recorded hashes),
that ``class_names.json`` equals the target's in content and order, that the
staged checkpoint loads into ``--architecture`` with zero missing and zero
unexpected keys, and that the policy's producer evidence matches what is being
installed -- the checkpoint's SHA-256, ``--architecture``, the class order and
the ``calibration.json`` temperature, compared exactly. Directory membership
alone is not evidence: a policy without it is refused, with instructions for
regenerating it by re-scoring. The backup also holds the previously served
checkpoint.

**Restore mode** (``--restore BACKUP_DIR``) puts back the state a backup
recorded -- restoring each backed-up file and removing files that did not exist
then -- through the same stage, verify, back up, install, post-check and
rollback steps, and writes a provenance record of the restore. It is the
rollback path for a promotion. Restart and ``--verify-live`` afterwards.

The logic lives in the ``app/deployment/`` package (its ``__init__`` maps the
modules); this script parses arguments, dispatches and prints.

Usage:
    python scripts/deploy_decision_policy.py \\
        --source results/accuracy_phase1/champion_resnet50_ft_v2/decision_layer_closure_2026-10-10
    RUN=results/accuracy_phase1/a3b_convnext_tiny_continued_224
    python scripts/deploy_decision_policy.py \\
        --source $RUN/decision_layer_closure_2026-10-10 --model-run $RUN \\
        --architecture convnext_tiny --model-name a3b_convnext_tiny
    python scripts/deploy_decision_policy.py --restore app/artifacts/replaced_<stamp>_<suffix>
    # after any of them, stop/restart every API process, then:
    python scripts/deploy_decision_policy.py --verify-live http://127.0.0.1:8000
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.deployment.errors import DeployError  # noqa: E402
from app.deployment.paths import DEFAULT_TARGET  # noqa: E402
from app.deployment.policy import deploy  # noqa: E402
from app.deployment.promote import promote  # noqa: E402
from app.deployment.restore import restore  # noqa: E402
from app.deployment.verify_live import Fetch, http_fetch, verify_live  # noqa: E402


def main(argv: list[str] | None = None, fetch: Fetch = http_fetch) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--source", type=Path, help="Recalibration run output dir to deploy")
    parser.add_argument("--target", type=Path, default=DEFAULT_TARGET, help="Runtime artifact dir")
    parser.add_argument("--dry-run", action="store_true", help="Validate only; write nothing")
    parser.add_argument(
        "--model-run",
        type=Path,
        help="Promote this training run's model together with the --source policy fitted on it",
    )
    parser.add_argument("--architecture", help="With --model-run: resnet50 or convnext_tiny")
    parser.add_argument("--model-name", help="With --model-run: the name responses report")
    parser.add_argument(
        "--checkpoint", help="With --model-run: checkpoint file name (default: its only .pth)"
    )
    parser.add_argument(
        "--restore",
        type=Path,
        metavar="BACKUP_DIR",
        help="Put back the state a replaced_* backup in --target recorded (run on its own)",
    )
    parser.add_argument(
        "--verify-live",
        metavar="URL",
        help="Check a restarted API at URL serves the target's model and policy (run on its own)",
    )
    args = parser.parse_args(argv)
    model_options = (args.architecture, args.model_name, args.checkpoint)
    if args.verify_live and (args.source or args.dry_run or args.model_run or args.restore):
        parser.error(
            "--verify-live runs on its own: deploy, restart every API process, then verify."
        )
    if args.restore and (args.source or args.model_run or any(model_options)):
        parser.error("--restore runs on its own (with --target and, optionally, --dry-run).")
    if args.model_run and not (args.source and args.architecture and args.model_name):
        parser.error(
            "--model-run needs --source (a policy run fitted on that model), "
            "--architecture and --model-name."
        )
    if any(model_options) and not args.model_run:
        parser.error("--architecture, --model-name and --checkpoint apply only with --model-run.")
    if not (args.verify_live or args.restore) and args.source is None:
        parser.error("one of --source, --restore or --verify-live is required")
    try:
        if args.verify_live:
            result = verify_live(args.verify_live, args.target, fetch)
        elif args.restore:
            result = restore(args.restore, args.target, args.dry_run)
        elif args.model_run:
            result = promote(
                args.source,
                args.model_run,
                args.architecture,
                args.model_name,
                args.target,
                args.dry_run,
                args.checkpoint,
            )
        else:
            result = deploy(args.source, args.target, args.dry_run)
    except (DeployError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2))
    if args.verify_live:
        print(
            "live service verified: it serves the target's model, decision layer and "
            "calibration.",
            file=sys.stderr,
        )
    elif not args.dry_run:
        print(
            "NOTE: only the artifact files were verified, not the live service. Running "
            "API processes keep their cached model and decision layer until restarted: restart "
            "every API process, then run with --verify-live <API base URL>.",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
