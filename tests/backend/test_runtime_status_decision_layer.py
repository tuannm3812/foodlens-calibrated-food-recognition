"""The /runtime/status decision_layer block and the runtime-cache boundary.

``load_runtime()`` caches the policy, hard classes and confusion pairs in
``inference._RUNTIME``; a running process keeps them until it restarts. The
status block must report what the process has cached, not what the files on
disk now say, or a stale service would look freshly deployed.
"""

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import app.backend.inference as inference
from app.backend.api import app
from app.backend.policy_fingerprint import decision_layer_fingerprint

client = TestClient(app)

LEGACY_POLICY = {"auto_confidence": 0.7, "suggest_confidence": 0.35, "margin_threshold": 0.4}
NEW_POLICY = {**LEGACY_POLICY, "margin_threshold": 0.05}


def write_decision_layer(artifact_dir: Path, policy: dict, hard: list, pairs: list) -> None:
    (artifact_dir / "decision_policy.json").write_text(json.dumps(policy))
    (artifact_dir / "hard_classes.json").write_text(json.dumps(hard))
    (artifact_dir / "confusion_pairs.json").write_text(
        json.dumps([{"actual": a, "predicted": p} for a, p in pairs])
    )


def fake_load_runtime() -> dict:
    """The metadata half of load_runtime(): caches exactly what it would cache."""
    inference._RUNTIME = {
        "policy": inference.read_policy(),
        "hard_classes": inference.read_hard_classes(),
        "confusion_pairs": inference.read_confusion_pairs(),
    }
    return inference._RUNTIME


@pytest.fixture
def artifacts(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """An artifact dir with a legacy decision layer and no runtime loaded."""
    monkeypatch.setattr(inference, "ARTIFACT_DIR", tmp_path)
    monkeypatch.setenv("FOODLENS_ARTIFACT_DIR", str(tmp_path))
    # monkeypatch restores _RUNTIME to its original value after the test.
    monkeypatch.setattr(inference, "_RUNTIME", None)
    monkeypatch.setattr(inference, "load_runtime", fake_load_runtime)
    write_decision_layer(tmp_path, LEGACY_POLICY, ["pizza"], [("pizza", "sushi")])
    return tmp_path


def test_before_load_status_reports_artifact_files(artifacts: Path) -> None:
    layer = client.get("/runtime/status").json()["decision_layer"]
    assert layer["source"] == "artifact_files"
    assert layer["policy"] == LEGACY_POLICY
    assert layer["hard_class_count"] == 1
    assert layer["confusion_pair_count"] == 1
    assert layer["fingerprint"] == decision_layer_fingerprint(
        LEGACY_POLICY, ["pizza"], [("pizza", "sushi")]
    )


def test_status_never_loads_the_model(
    artifacts: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def forbidden() -> dict:
        raise AssertionError("runtime_status() must not load the runtime")

    monkeypatch.setattr(inference, "load_runtime", forbidden)
    assert inference.runtime_status()["decision_layer"]["source"] == "artifact_files"
    assert inference._RUNTIME is None


def test_after_load_status_reports_loaded_runtime(artifacts: Path) -> None:
    inference.load_runtime()
    layer = inference.runtime_status()["decision_layer"]
    assert layer["source"] == "loaded_runtime"
    assert layer["policy"] == LEGACY_POLICY


def test_cached_runtime_is_reported_after_the_files_change(artifacts: Path) -> None:
    inference.load_runtime()
    legacy_fingerprint = inference.runtime_status()["decision_layer"]["fingerprint"]

    # A deploy rewrites the files under a running process.
    write_decision_layer(
        artifacts, NEW_POLICY, ["steak", "pizza"], [("steak", "filet_mignon")]
    )
    new_fingerprint = decision_layer_fingerprint(
        NEW_POLICY, ["steak", "pizza"], [("steak", "filet_mignon")]
    )

    layer = client.get("/runtime/status").json()["decision_layer"]
    assert layer["source"] == "loaded_runtime"
    assert layer["policy"] == LEGACY_POLICY
    assert layer["fingerprint"] == legacy_fingerprint
    assert layer["fingerprint"] != new_fingerprint


def test_unreadable_policy_is_reported_not_raised(artifacts: Path) -> None:
    # The recalibration output format the backend cannot read.
    (artifacts / "decision_policy.json").write_text(json.dumps([LEGACY_POLICY]))
    response = client.get("/runtime/status")
    assert response.status_code == 200
    layer = response.json()["decision_layer"]
    assert layer["source"] == "artifact_files"
    assert "error" in layer


def test_fingerprint_ignores_order_of_classes_and_pairs() -> None:
    first = decision_layer_fingerprint(
        LEGACY_POLICY, ["a", "b", "c"], [("a", "b"), ("c", "d")]
    )
    second = decision_layer_fingerprint(
        dict(reversed(LEGACY_POLICY.items())),
        {"c", "a", "b"},
        [{"actual": "c", "predicted": "d"}, {"actual": "a", "predicted": "b"}],
    )
    assert first == second


@pytest.mark.parametrize("key", sorted(LEGACY_POLICY))
def test_fingerprint_changes_with_every_threshold(key: str) -> None:
    base = decision_layer_fingerprint(LEGACY_POLICY, ["a"], [("a", "b")])
    changed = decision_layer_fingerprint({**LEGACY_POLICY, key: 0.123}, ["a"], [("a", "b")])
    assert base != changed


def test_fingerprint_distinguishes_pair_direction() -> None:
    forward = decision_layer_fingerprint(LEGACY_POLICY, [], [("a", "b")])
    backward = decision_layer_fingerprint(LEGACY_POLICY, [], [("b", "a")])
    assert forward != backward
