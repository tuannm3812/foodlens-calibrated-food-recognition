"""``--verify-live``: prove a running API serves the target's model, routing and calibration.

Reads the target through ``identity`` and the deployment record through
``records``; it imports no write operation (``policy``, ``promote``,
``restore``, ``install``), so verifying can never change a target.
"""

from __future__ import annotations

import io
import json
import urllib.error
import urllib.parse
import urllib.request
import uuid
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from . import identity
from .errors import DeployError
from .paths import DEPLOYED_FILES, PROVENANCE_FILE, display, read_json, sha256
from .records import recorded_model_identity

# What --verify-live checks, each reported on its own.
CHECK_NAMES = ("model", "routing", "calibration")
HTTP_TIMEOUT_SECONDS = 180  # the first prediction after a restart loads the model

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


def live_model_problem(served: Any, expected: Mapping[str, str]) -> str | None:
    """Compare the status ``model`` block with the target's model; describe any mismatch."""
    if not isinstance(served, dict):
        return (
            "/runtime/status reports no model block; the service runs code that predates "
            "the model identity check. Restart it on the current code."
        )
    if served.get("source") != "loaded_runtime":
        return (
            f"The service reports model.source={served.get('source')!r}, not "
            "'loaded_runtime', so it has not loaded a model to compare. Restart it and retry."
        )
    mismatched = [
        f"{key}: service {served.get(key)!r}, target {expected[key]!r}"
        for key in identity.IDENTITY_KEYS
        if served.get(key) != expected[key]
    ]
    if not mismatched:
        return None
    return (
        "The live service serves a different model than the target holds ("
        + "; ".join(mismatched)
        + "). It is still running the model it loaded before the files changed: "
        "stop/restart every API process, then rerun --verify-live."
    )


def live_calibration_problems(
    prediction: Mapping[str, Any], served: Any, expected: Mapping[str, Any]
) -> list[str]:
    """Compare the runtime's cached calibration with the target's; describe each mismatch.

    The temperature is cached by ``load_runtime()`` and is part of neither the
    model identity nor the routing fingerprint, so it is checked on its own:
    the probe response's ``temperature`` (what the runtime just scaled logits
    by) must equal the status ``model`` block's, and both must equal the
    target's validated ``calibration.json``, exactly. The class-order hash the
    runtime computed at load must equal the target's.
    """
    if not isinstance(served, dict) or served.get("source") != "loaded_runtime":
        return ["/runtime/status reports no loaded model block, so no cached calibration."]
    restart = "stop/restart every API process, then rerun --verify-live."
    problems = []
    status_temperature = served.get("temperature")
    probe_temperature = prediction.get("temperature")
    if status_temperature is None:
        problems.append(
            "/runtime/status reports no temperature in its model block; the service runs code "
            "that predates the calibration check. Restart it on the current code."
        )
    elif probe_temperature != status_temperature:
        problems.append(
            f"The probe response was scaled by temperature {probe_temperature!r}, but the "
            f"status model block reports {status_temperature!r}."
        )
    if status_temperature is not None and status_temperature != expected["temperature"]:
        problems.append(
            f"The live service scales logits by cached temperature {status_temperature!r}, but "
            f"the target's calibration.json holds {expected['temperature']!r}. It is still "
            f"running the calibration it loaded before the files changed: {restart}"
        )
    served_names = served.get("class_names_sha256")
    if served_names != expected["class_names_sha256"]:
        problems.append(
            f"class_names_sha256: service {served_names!r}, target "
            f"{expected['class_names_sha256']!r}. A different class order mislabels every "
            f"prediction: {restart}"
        )
    return problems


def recorded_drift(
    provenance: Mapping[str, Any],
    target: Path,
    files_fingerprint: str,
    expected_model: Mapping[str, str],
    expected_calibration: Mapping[str, Any],
) -> dict[str, list[str]]:
    """Each way the target's files changed after the deployment record was written.

    Returns:
        Problems by check (``model``, ``routing``, ``calibration``); a record
        that predates a field is not held against the target.
    """
    redeploy = "it changed after deployment. Redeploy before verifying the service."
    drift: dict[str, list[str]] = {name: [] for name in CHECK_NAMES}
    recorded = provenance.get("decision_layer_fingerprint")
    if recorded is not None and recorded != files_fingerprint:
        drift["routing"].append(
            f"The target files (fingerprint {files_fingerprint}) no longer match "
            f"{PROVENANCE_FILE} (fingerprint {recorded}); they changed after deployment. "
            "Redeploy before verifying the service."
        )
    recorded_model = recorded_model_identity(provenance)
    if recorded_model is not None:
        for key in ("checkpoint_sha256", "architecture", "checkpoint"):
            recorded_value = recorded_model.get(key)
            if recorded_value is not None and recorded_value != expected_model[key]:
                drift["model"].append(
                    f"The target's {key} ({expected_model[key]}) no longer matches "
                    f"{PROVENANCE_FILE} ({recorded_value}); " + redeploy
                )
    for key in ("deployed_files_sha256", "served_model_files_sha256"):
        hashes = provenance.get(key)
        if not isinstance(hashes, dict):
            continue
        for name, recorded_hash in hashes.items():
            if not name.endswith(".pth") or not (target / name).is_file():
                continue
            if recorded_hash != sha256(target / name):
                problem = (
                    f"The target's {name} (sha256 {sha256(target / name)}) no longer matches "
                    f"{PROVENANCE_FILE}'s {key} (sha256 {recorded_hash}); " + redeploy
                )
                if problem not in drift["model"]:
                    drift["model"].append(problem)
        for name in ("calibration.json", "class_names.json"):
            if name in hashes and hashes[name] != sha256(target / name):
                drift["calibration"].append(
                    f"The target's {name} (sha256 {sha256(target / name)}) no longer matches "
                    f"{PROVENANCE_FILE}'s {key} (sha256 {hashes[name]}); " + redeploy
                )
    temperatures = [provenance.get("calibration_temperature")]
    if isinstance(provenance.get("served_model"), dict):
        temperatures.append(provenance["served_model"].get("temperature"))
    for value in temperatures:
        if value is not None and value != expected_calibration["temperature"]:
            drift["calibration"].append(
                f"The target's temperature {expected_calibration['temperature']!r} differs from "
                f"the {value!r} {PROVENANCE_FILE} recorded; " + redeploy
            )
    drift["calibration"] = list(dict.fromkeys(drift["calibration"]))
    return drift


def verification_failure(
    problems: Mapping[str, list[str]], stage: str, passed: str = "passed"
) -> DeployError:
    """One error naming every check's outcome, so a failure says which one broke."""
    lines = [f"--verify-live failed ({stage})."]
    for name in CHECK_NAMES:
        if problems[name]:
            lines.append(f"{name}: FAILED")
            lines.extend(f"  - {problem}" for problem in problems[name])
        else:
            lines.append(f"{name}: {passed}")
    return DeployError("\n".join(lines))


def verify_live(url: str, target: Path, fetch: Fetch = http_fetch) -> dict[str, Any]:
    """Prove that a running API process serves the target's model, routing and calibration.

    1. Check the target's files against ``deployment_provenance.json``, when it
       records them: the decision-layer fingerprint, the checkpoint hash, and
       the hashes of ``calibration.json`` and ``class_names.json`` (and the
       recorded temperature), so files edited after deployment are caught.
    2. POST a synthetic image to ``/predict/image`` so a freshly restarted
       process loads its runtime; any ``fallback_reason`` fails the check,
       because a silent demo fallback is exactly the hazard being ruled out.
    3. GET ``/runtime/status`` and check, independently:

       - **routing**: ``decision_layer.source == "loaded_runtime"`` with a
         fingerprint equal to the target files';
       - **model**: a ``loaded_runtime`` model block whose architecture, model
         name and checkpoint SHA-256 equal the target's;
       - **calibration**: the probe response's temperature equals the model
         block's cached temperature, both equal the target's validated
         ``calibration.json`` exactly, and the cached class-order hash equals
         the target's.

    Args:
        url: Base URL of the running API, e.g. ``http://127.0.0.1:8000``.
        target: The artifact directory the service is meant to serve.
        fetch: HTTP transport, injectable so tests need no network.

    Returns:
        The verification record, with a ``checks`` entry per check.

    Raises:
        DeployError: On any mismatch, fallback or transport failure, naming
            each check as passed or FAILED with every problem found.
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
    _, _, _, files_fingerprint = identity.read_through_backend(target)
    expected_model = identity.target_model_identity(target)
    expected_calibration = identity.target_calibration(target)

    result: dict[str, Any] = {"url": base, "target": display(target)}
    provenance_path = target / PROVENANCE_FILE
    provenance: dict[str, Any] = {}
    if provenance_path.exists():
        provenance = read_json(provenance_path)
        if not isinstance(provenance, dict):
            raise DeployError(f"{provenance_path} must hold a provenance object.")
    drift = recorded_drift(
        provenance, target, files_fingerprint, expected_model, expected_calibration
    )
    if any(drift.values()):
        raise verification_failure(
            drift,
            "the target's files changed after deployment; the service was not probed",
            passed="target files match the deployment record",
        )
    if provenance.get("decision_layer_fingerprint") is None:
        result["provenance_check"] = (
            f"{PROVENANCE_FILE} records no decision_layer_fingerprint (deployments before "
            "2026-10-10 did not); compared against the target files alone."
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
    problems: dict[str, list[str]] = {name: [] for name in CHECK_NAMES}
    layer = status.get("decision_layer") if isinstance(status, dict) else None
    if not isinstance(layer, dict):
        problems["routing"].append(
            f"{base}/runtime/status reports no decision_layer; the service runs code that "
            "predates the live check. Restart it on the current code."
        )
    elif layer.get("source") != "loaded_runtime":
        problems["routing"].append(
            f"The service reports decision_layer.source={layer.get('source')!r}, not "
            "'loaded_runtime', so it has not loaded a runtime to compare. Restart it and retry."
        )
    elif layer.get("fingerprint") != files_fingerprint:
        problems["routing"].append(
            f"The live service routes with fingerprint {layer.get('fingerprint')}, but the "
            f"target files have {files_fingerprint}. The service is still serving a cached "
            "decision layer: stop/restart every API process, then rerun --verify-live."
        )
    served_model = status.get("model") if isinstance(status, dict) else None
    model_problem = live_model_problem(served_model, expected_model)
    if model_problem:
        problems["model"].append(model_problem)
    problems["calibration"] = live_calibration_problems(
        prediction, served_model, expected_calibration
    )
    if any(problems.values()):
        raise verification_failure(problems, "the live service differs from the target")

    result.update(
        {
            "live_service_verified": True,
            "checks": {
                "model": "passed: architecture, model_name and checkpoint_sha256 match",
                "routing": "passed: the loaded decision-layer fingerprint matches",
                "calibration": (
                    "passed: probe and status temperature equal calibration.json exactly; "
                    "class_names_sha256 matches"
                ),
            },
            "decision_layer_fingerprint": layer.get("fingerprint"),
            "served_policy": layer.get("policy"),
            "hard_class_count": layer.get("hard_class_count"),
            "confusion_pair_count": layer.get("confusion_pair_count"),
            "model": {
                key: served_model.get(key) for key in ("checkpoint", *identity.IDENTITY_KEYS)
            },
            "calibration": {
                "temperature": served_model.get("temperature"),
                "probe_temperature": prediction.get("temperature"),
                "class_names_sha256": served_model.get("class_names_sha256"),
            },
        }
    )
    return result
