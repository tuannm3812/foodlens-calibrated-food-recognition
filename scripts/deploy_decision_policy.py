"""Deploy a recalibrated decision-layer policy into the runtime artifact directory.

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
   ``replaced_<UTC stamp>_<suffix>/`` directory;
3. installs each file with an atomic ``os.replace`` and writes
   ``deployment_provenance.json`` recording the source run, its derivation
   provenance and the hash of every deployed file;
4. reads the installed target back through the backend readers.

If anything fails after installation starts, every original file is restored,
files that did not exist before are removed, the new backup directory is
deleted, and the command fails saying the deployment was rolled back. A failed
invocation therefore leaves the target exactly as it was.

**The file check is not proof of the live service.** Step 4 reads the files
freshly in the deploy process. A running API process caches the policy, hard
classes and confusion pairs in ``inference._RUNTIME`` on first use and keeps
serving them until it restarts, so the record says ``artifact_files_verified``
and carries a ``live_service`` note, never a claim about the service. After a
deploy, stop/restart every API process, then run ``--verify-live URL``: it
sends one synthetic image to ``/predict/image`` (so a restarted process loads
its runtime, and a demo fallback fails the check), then requires
``/runtime/status`` to report ``decision_layer.source == "loaded_runtime"`` with
the same decision-layer fingerprint as the target files and, when it records
one, ``deployment_provenance.json``.

Only decision-layer artifacts are touched. The model checkpoint, class names and
calibration are never modified, so deploying a policy can never change which
model is served.

Usage:
    python scripts/deploy_decision_policy.py \\
        --source results/accuracy_phase1/champion_resnet50_ft_v2/decision_layer_closure_2026-10-10
    # stop/restart every API process, then:
    python scripts/deploy_decision_policy.py --verify-live http://127.0.0.1:8000
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import shutil
import sys
import tempfile
import urllib.error
import urllib.parse
import urllib.request
import uuid
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TARGET = REPO_ROOT / "app" / "artifacts"
POLICY_KEYS = ("auto_confidence", "suggest_confidence", "margin_threshold")
DEPLOYED_FILES = ("decision_policy.json", "hard_classes.json", "confusion_pairs.json")
PROVENANCE_FILE = "deployment_provenance.json"
HTTP_TIMEOUT_SECONDS = 180  # the first prediction after a restart loads the model


class DeployError(Exception):
    """A predictable deployment failure, reported without a traceback."""


def display(path: Path) -> str:
    """Repo-relative path when inside the repo, absolute otherwise."""
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT))
    except ValueError:
        return str(resolved)


def sha256(path: Path) -> str:
    """Return the SHA-256 hex digest of a file's bytes."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> Any:
    """Read JSON, converting a missing or malformed file into a DeployError."""
    if not path.exists():
        raise DeployError(f"{path} does not exist.")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise DeployError(f"{path} is not valid JSON: {exc}") from exc


def runtime_policy(raw: Any, source: Path) -> dict[str, float]:
    """Convert a recalibration policy record into the runtime threshold dict.

    Args:
        raw: Decoded ``decision_policy.json`` -- a dict, or a one-element list
            holding one, as the recalibration script writes it.
        source: Path the policy came from, for error messages.

    Returns:
        A dict with exactly the three thresholds the backend reads.

    Raises:
        DeployError: If the record is ambiguous, incomplete, or out of range.
    """
    if isinstance(raw, list):
        if len(raw) != 1:
            raise DeployError(f"{source} holds {len(raw)} policy records; expected exactly one.")
        raw = raw[0]
    if not isinstance(raw, dict):
        raise DeployError(f"{source} must hold a policy object, found {type(raw).__name__}.")
    missing = [key for key in POLICY_KEYS if key not in raw]
    if missing:
        raise DeployError(f"{source} is missing policy keys: {', '.join(missing)}.")

    policy: dict[str, float] = {}
    for key in POLICY_KEYS:
        value = raw[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise DeployError(f"{source}: {key} must be a number, found {value!r}.")
        value = float(value)
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise DeployError(f"{source}: {key} must be a finite value in [0, 1], found {value}.")
        policy[key] = value
    if policy["suggest_confidence"] > policy["auto_confidence"]:
        raise DeployError(
            f"{source}: suggest_confidence ({policy['suggest_confidence']}) exceeds "
            f"auto_confidence ({policy['auto_confidence']}), so no prediction could suggest."
        )
    return policy


def validated_hard_classes(raw: Any, source: Path) -> list[str]:
    """Validate a hard-class list: non-empty, unique, non-empty strings."""
    if not isinstance(raw, list) or not raw:
        raise DeployError(f"{source} must be a non-empty JSON list of class names.")
    names = []
    for item in raw:
        if not isinstance(item, str) or not item.strip():
            raise DeployError(f"{source} contains an invalid class name: {item!r}.")
        names.append(item.strip())
    if len(set(names)) != len(names):
        raise DeployError(f"{source} contains duplicate class names.")
    return names


def validated_confusion_pairs(raw: Any, source: Path) -> list[dict[str, str]]:
    """Validate confusion pairs as a non-empty list of {actual, predicted} records."""
    if not isinstance(raw, list) or not raw:
        raise DeployError(f"{source} must be a non-empty JSON list of pair records.")
    pairs = []
    for item in raw:
        if not isinstance(item, dict) or not {"actual", "predicted"} <= set(item):
            raise DeployError(f"{source} contains a malformed pair record: {item!r}.")
        actual, predicted = item["actual"], item["predicted"]
        if not all(isinstance(v, str) and v.strip() for v in (actual, predicted)):
            raise DeployError(f"{source} contains a pair with an empty label: {item!r}.")
        pairs.append({"actual": actual.strip(), "predicted": predicted.strip()})
    if len({(p["actual"], p["predicted"]) for p in pairs}) != len(pairs):
        raise DeployError(f"{source} contains duplicate confusion pairs.")
    return pairs


def check_class_coverage(target: Path, hard: list[str], pairs: list[dict[str, str]]) -> None:
    """Reject labels the served model cannot predict -- they would never match."""
    names_path = target / "class_names.json"
    if not names_path.exists():
        raise DeployError(f"{names_path} is missing; cannot check labels against the model.")
    known = set(read_json(names_path))
    labels = set(hard) | {p["actual"] for p in pairs} | {p["predicted"] for p in pairs}
    unknown = sorted(labels - known)
    if unknown:
        raise DeployError(
            f"Labels not in the served model's class_names.json: {', '.join(unknown[:5])}."
        )


def backend_modules() -> tuple[Any, Any]:
    """Import the backend inference module and the fingerprint function."""
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from app.backend import inference
    from app.backend.policy_fingerprint import decision_layer_fingerprint

    return inference, decision_layer_fingerprint


def read_through_backend(directory: Path) -> tuple[dict[str, float], set[str], set[Any], str]:
    """Read a directory's decision layer through the backend's own readers.

    Returns:
        The policy, hard classes and confusion pairs exactly as ``load_runtime()``
        would cache them, plus their decision-layer fingerprint.

    Raises:
        DeployError: If the backend resolves a different artifact directory.
    """
    inference, fingerprint = backend_modules()
    previous = os.environ.get("FOODLENS_ARTIFACT_DIR")
    os.environ["FOODLENS_ARTIFACT_DIR"] = str(directory)
    try:
        policy = inference.read_policy()
        hard = inference.read_hard_classes()
        pairs = inference.read_confusion_pairs()
        resolved = inference.artifact_dir_path()
    finally:
        if previous is None:
            os.environ.pop("FOODLENS_ARTIFACT_DIR", None)
        else:
            os.environ["FOODLENS_ARTIFACT_DIR"] = previous
    if resolved.resolve() != directory.resolve():
        raise DeployError(f"Backend resolved artifacts to {resolved}, not {directory}.")
    return policy, hard, pairs, fingerprint(policy, hard, pairs)


def verify_through_backend(
    target: Path, policy: dict[str, float], hard: list[str], pairs: list[dict[str, str]]
) -> str:
    """Load the deployed files through the backend's own readers and compare.

    This is the check that would have caught the list-vs-dict mismatch: it does
    not trust the files, it trusts what the runtime reads from them. It reads
    them freshly in *this* process, so it says nothing about what an
    already-running API process has cached -- that is ``--verify-live``'s job.

    Returns:
        The decision-layer fingerprint of what the backend read.
    """
    observed_policy, observed_hard, observed_pairs, fingerprint = read_through_backend(target)
    if observed_policy != policy:
        raise DeployError(f"Backend read policy {observed_policy}, expected {policy}.")
    if observed_hard != set(hard):
        raise DeployError("Backend read a different hard-class set than was deployed.")
    expected_pairs = {(p["actual"], p["predicted"]) for p in pairs}
    if observed_pairs != expected_pairs:
        raise DeployError("Backend read a different confusion-pair set than was deployed.")
    return fingerprint


def live_service_note(target: Path) -> str:
    """The record's statement of what the file check does not prove."""
    return (
        "NOT verified by this deploy. artifact_files_verified means the files were "
        "read back freshly in the deploy process; a running API process keeps the "
        "decision layer it cached at load time until it is restarted. Stop/restart "
        "every API process, then run: python scripts/deploy_decision_policy.py "
        f"--target {display(target)} --verify-live <API base URL>"
    )


def write_json_file(path: Path, value: Any) -> None:
    """Write indented JSON with a trailing newline."""
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


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
    record: dict[str, Any],
    policy: dict[str, float],
    hard: list[str],
    pairs: list[dict[str, str]],
) -> None:
    """Back up, atomically install the staged files and provenance, and re-check.

    If anything fails after the first file is replaced, every original file
    (including ``deployment_provenance.json``) is restored, files that did not
    exist before are removed, and the backup directory created here is deleted,
    so a failed deploy leaves the target exactly as it was.
    """
    installed = (*DEPLOYED_FILES, PROVENANCE_FILE)
    prior = {
        name: sha256(target / name) if (target / name).exists() else None
        for name in installed
    }
    backup: Path | None = None
    if any(digest is not None for digest in prior.values()):
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        backup = Path(tempfile.mkdtemp(prefix=f"replaced_{stamp}_", dir=target))
        try:
            for name, digest in prior.items():
                if digest is not None:
                    shutil.copy2(target / name, backup / name)
        except Exception:
            shutil.rmtree(backup, ignore_errors=True)
            raise
        record["backup_dir"] = display(backup)
        record["replaced_files_sha256"] = {
            name: prior[name] for name in DEPLOYED_FILES if prior[name] is not None
        }
        if prior[PROVENANCE_FILE] is not None:
            record["replaced_provenance_sha256"] = prior[PROVENANCE_FILE]

    try:
        for name in DEPLOYED_FILES:
            os.replace(staging / name, target / name)
        write_json_file(staging / PROVENANCE_FILE, record)
        os.replace(staging / PROVENANCE_FILE, target / PROVENANCE_FILE)

        # Post-install check: the target, not the staged copy, is what the
        # backend will read.
        installed_fingerprint = verify_through_backend(target, policy, hard, pairs)
        if installed_fingerprint != record["decision_layer_fingerprint"]:
            raise DeployError("installed decision layer fingerprint differs from the staged one")
        for name, digest in record["deployed_files_sha256"].items():
            if sha256(target / name) != digest:
                raise DeployError(f"installed {name} does not match the staged file")
    except BaseException as exc:
        rollback(target, staging, backup, prior)
        if not isinstance(exc, Exception):
            raise
        raise DeployError(
            f"Deployment rolled back; {display(target)} is unchanged. Cause: {exc}"
        ) from exc


def deploy(source: Path, target: Path, dry_run: bool) -> dict[str, Any]:
    """Validate, stage, verify, back up, install and re-verify; return the record.

    A failure at any step leaves ``target`` byte-for-byte as it was: staged files
    are verified before the target is touched, and a failure after installation
    starts is rolled back.
    """
    provenance_path = source / "derivation_provenance.json"
    # Refuse a source without derivation provenance: a deployed policy must trace
    # back to the run that produced it.
    if not isinstance(read_json(provenance_path), dict):
        raise DeployError(f"{provenance_path} must hold a provenance object.")

    policy = runtime_policy(read_json(source / "decision_policy.json"), source)
    hard = validated_hard_classes(read_json(source / "hard_classes.json"), source)
    pairs = validated_confusion_pairs(read_json(source / "confusion_pairs.json"), source)
    check_class_coverage(target, hard, pairs)

    record: dict[str, Any] = {
        "deployed_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "source_run": display(source),
        "source_provenance_sha256": sha256(provenance_path),
        "source_files_sha256": {
            name: sha256(source / name) for name in DEPLOYED_FILES
        },
        "policy": policy,
        "hard_class_count": len(hard),
        "confusion_pair_count": len(pairs),
        "untouched": ["model checkpoint", "class_names.json", "calibration.json"],
    }
    if dry_run:
        record["dry_run"] = True
        return record

    # Stage inside the target so every install step is a same-filesystem
    # os.replace, and verify the staged files before the target is touched.
    staging = Path(tempfile.mkdtemp(prefix=".deploy-staging-", dir=target))
    try:
        write_json_file(staging / "decision_policy.json", policy)
        write_json_file(staging / "hard_classes.json", hard)
        write_json_file(staging / "confusion_pairs.json", pairs)
        fingerprint = verify_through_backend(staging, policy, hard, pairs)
        record["deployed_files_sha256"] = {
            name: sha256(staging / name) for name in DEPLOYED_FILES
        }
        # A fresh read of the files in this process -- not proof that a running
        # API process serves them (see live_service).
        record["artifact_files_verified"] = True
        record["decision_layer_fingerprint"] = fingerprint
        record["live_service"] = live_service_note(target)
        install(target, staging, record, policy, hard, pairs)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return record


Fetch = Callable[[str, str, bytes | None, dict[str, str]], Any]


def http_fetch(method: str, url: str, body: bytes | None, headers: dict[str, str]) -> Any:
    """Send one HTTP request with the standard library and decode its JSON body."""
    request = urllib.request.Request(url, data=body, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=HTTP_TIMEOUT_SECONDS) as response:
            payload = response.read()
    except urllib.error.HTTPError as exc:
        raise DeployError(f"{method} {url} returned HTTP {exc.code}.") from exc
    except OSError as exc:
        raise DeployError(f"{method} {url} failed: {exc}") from exc
    try:
        return json.loads(payload)
    except json.JSONDecodeError as exc:
        raise DeployError(f"{method} {url} did not return JSON: {exc}") from exc


def probe_image() -> bytes:
    """A small synthetic PNG, built in memory, for the warm-up prediction."""
    try:
        from PIL import Image
    except ImportError as exc:
        raise DeployError("--verify-live needs Pillow to build its probe image.") from exc
    buffer = io.BytesIO()
    Image.new("RGB", (64, 64), color=(200, 120, 60)).save(buffer, format="PNG")
    return buffer.getvalue()


def multipart_file(field: str, filename: str, content: bytes, content_type: str):
    """Encode one file as a multipart/form-data body; return (body, headers)."""
    boundary = uuid.uuid4().hex
    head = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="{field}"; filename="{filename}"\r\n'
        f"Content-Type: {content_type}\r\n\r\n"
    ).encode()
    body = head + content + f"\r\n--{boundary}--\r\n".encode()
    return body, {"Content-Type": f"multipart/form-data; boundary={boundary}"}


def verify_live(url: str, target: Path, fetch: Fetch = http_fetch) -> dict[str, Any]:
    """Prove that a running API process routes with the target's decision layer.

    1. POST a synthetic image to ``/predict/image`` so a freshly restarted
       process loads its runtime; any ``fallback_reason`` fails the check,
       because a silent demo fallback is exactly the hazard being ruled out.
    2. GET ``/runtime/status`` and require ``decision_layer.source ==
       "loaded_runtime"`` with a fingerprint equal to the target files' and to
       ``deployment_provenance.json``'s, when that records one.

    Args:
        url: Base URL of the running API, e.g. ``http://127.0.0.1:8000``.
        target: The artifact directory the service is meant to serve.
        fetch: HTTP transport, injectable so tests need no network.

    Returns:
        The verification record.

    Raises:
        DeployError: On any mismatch, fallback or transport failure.
    """
    base = url.rstrip("/")
    if urllib.parse.urlparse(base).scheme not in {"http", "https"}:
        raise DeployError(f"--verify-live needs an http(s) URL, got {url!r}.")
    missing = [name for name in DEPLOYED_FILES if not (target / name).exists()]
    if missing:
        raise DeployError(
            f"{display(target)} is missing {', '.join(missing)}; the backend would "
            "serve built-in defaults, so there is no deployed policy to verify."
        )
    _, _, _, files_fingerprint = read_through_backend(target)

    result: dict[str, Any] = {"url": base, "target": display(target)}
    provenance_path = target / PROVENANCE_FILE
    recorded = None
    if provenance_path.exists():
        provenance = read_json(provenance_path)
        if not isinstance(provenance, dict):
            raise DeployError(f"{provenance_path} must hold a provenance object.")
        recorded = provenance.get("decision_layer_fingerprint")
    if recorded is None:
        result["provenance_check"] = (
            f"{PROVENANCE_FILE} records no decision_layer_fingerprint (deployments before "
            "2026-10-10 did not); compared against the target files alone."
        )
    elif recorded != files_fingerprint:
        raise DeployError(
            f"The target files (fingerprint {files_fingerprint}) no longer match "
            f"{PROVENANCE_FILE} (fingerprint {recorded}); they changed after deployment. "
            "Redeploy before verifying the service."
        )
    else:
        result["provenance_check"] = "target files match deployment_provenance.json"

    body, headers = multipart_file("file", "verify-live-probe.png", probe_image(), "image/png")
    prediction = fetch("POST", f"{base}/predict/image", body, headers)
    if not isinstance(prediction, dict):
        kind = type(prediction).__name__
        raise DeployError(f"{base}/predict/image returned {kind}, not an object.")
    if prediction.get("fallback_reason") or prediction.get("artifact_status") != "ready":
        raise DeployError(
            f"The probe prediction fell back (fallback_reason="
            f"{prediction.get('fallback_reason')!r}, artifact_status="
            f"{prediction.get('artifact_status')!r}): the service is serving demo output, "
            "not the model. Check its artifacts and logs, then restart it."
        )

    status = fetch("GET", f"{base}/runtime/status", None, {})
    layer = status.get("decision_layer") if isinstance(status, dict) else None
    if not isinstance(layer, dict):
        raise DeployError(
            f"{base}/runtime/status reports no decision_layer; the service runs code that "
            "predates the live check. Restart it on the current code."
        )
    if layer.get("source") != "loaded_runtime":
        raise DeployError(
            f"The service reports decision_layer.source={layer.get('source')!r}, not "
            "'loaded_runtime', so it has not loaded a runtime to compare. Restart it and retry."
        )
    served = layer.get("fingerprint")
    if served != files_fingerprint:
        raise DeployError(
            f"The live service routes with fingerprint {served}, but the target files have "
            f"{files_fingerprint}. The service is still serving a cached decision layer: "
            "stop/restart every API process, then rerun --verify-live."
        )
    result.update(
        {
            "live_service_verified": True,
            "decision_layer_fingerprint": served,
            "served_policy": layer.get("policy"),
            "hard_class_count": layer.get("hard_class_count"),
            "confusion_pair_count": layer.get("confusion_pair_count"),
        }
    )
    return result


def main(argv: list[str] | None = None, fetch: Fetch = http_fetch) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--source", type=Path, help="Recalibration run output dir to deploy")
    parser.add_argument("--target", type=Path, default=DEFAULT_TARGET, help="Runtime artifact dir")
    parser.add_argument("--dry-run", action="store_true", help="Validate only; write nothing")
    parser.add_argument(
        "--verify-live",
        metavar="URL",
        help="Check a restarted API at URL serves the target's policy (run on its own)",
    )
    args = parser.parse_args(argv)
    if args.verify_live and (args.source or args.dry_run):
        parser.error(
            "--verify-live runs on its own: deploy, restart every API process, then verify."
        )
    if not args.verify_live and args.source is None:
        parser.error("one of --source or --verify-live is required")
    try:
        if args.verify_live:
            result = verify_live(args.verify_live, args.target, fetch)
        else:
            result = deploy(args.source, args.target, args.dry_run)
    except (DeployError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2))
    if args.verify_live:
        print("live service verified: it routes with the target's decision layer.", file=sys.stderr)
    elif not args.dry_run:
        print(
            "NOTE: only the artifact files were verified, not the live service. Running "
            "API processes keep their cached decision layer until restarted: restart every "
            "API process, then run with --verify-live <API base URL>.",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
