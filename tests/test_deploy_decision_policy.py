"""Tests for scripts/deploy_decision_policy.py.

The central regression: a recalibration run writes decision_policy.json as a
one-element list, the backend reads a dict, and copying the file verbatim makes
every request fall back to demo output while artifact_status still reports
"ready". The deploy step must convert, validate, back up and verify instead.

The logic lives in app/deployment/. Failure injection patches the module that
owns each call -- identity.verify_through_backend, install's os/shutil/tempfile
and its datetime -- and each injection test asserts its fault was reached.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest
from evidence_fixtures import provenance_with_evidence, write_scored_split

from app.deployment import identity, install, paths

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "deploy_decision_policy.py"
_spec = importlib.util.spec_from_file_location("deploy_decision_policy", SCRIPT)
deploy_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(deploy_mod)

CLASSES = ["apple_pie", "bread_pudding", "pizza", "steak", "filet_mignon", "sushi"]
POLICY = {"auto_confidence": 0.7, "suggest_confidence": 0.35, "margin_threshold": 0.05}
# The legacy target's checkpoint bytes and calibration, which the source policy's
# producer evidence must match.
CHECKPOINT = b"checkpoint"
TEMPERATURE = 0.958


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value))


@pytest.fixture
def source(tmp_path: Path) -> Path:
    """A recalibration run directory in the analysis format."""
    run = tmp_path / "run"
    run.mkdir()
    write_json(
        run / "decision_policy.json", [{**POLICY, "fit_split": "val", "eval_split": "test"}]
    )
    write_json(run / "hard_classes.json", ["bread_pudding", "steak"])
    write_json(
        run / "confusion_pairs.json",
        [
            {"actual": "steak", "predicted": "filet_mignon"},
            {"actual": "apple_pie", "predicted": "bread_pudding"},
        ],
    )
    # Fitted on predictions the target's model produced, with their evidence.
    model_run = tmp_path / "model_run"
    model_run.mkdir()
    (model_run / "resnet50_ft_v2_best.pth").write_bytes(CHECKPOINT)
    scored = [
        write_scored_split(
            model_run / f"{split}_predictions_rescored.csv",
            split=split,
            checkpoint=model_run / "resnet50_ft_v2_best.pth",
            architecture="resnet50",
            class_names=CLASSES,
            temperature=TEMPERATURE,
        )
        for split in ("val", "test")
    ]
    write_json(
        run / "derivation_provenance.json",
        {"generator": "test", **provenance_with_evidence(*scored)},
    )
    return run


@pytest.fixture
def target(tmp_path: Path) -> Path:
    """A runtime artifact directory holding a legacy policy."""
    art = tmp_path / "artifacts"
    art.mkdir()
    write_json(art / "class_names.json", CLASSES)
    write_json(art / "decision_policy.json", {**POLICY, "margin_threshold": 0.4})
    write_json(art / "hard_classes.json", ["pizza"])
    write_json(art / "confusion_pairs.json", [{"actual": "pizza", "predicted": "sushi"}])
    # The checkpoint --verify-live compares with the service's model identity.
    (art / "resnet50_ft_v2_best.pth").write_bytes(CHECKPOINT)
    write_json(art / "calibration.json", {"temperature": TEMPERATURE})
    return art


def test_deploy_converts_list_policy_to_the_dict_the_backend_reads(source, target):
    assert deploy_mod.main(["--source", str(source), "--target", str(target)]) == 0
    assert json.loads((target / "decision_policy.json").read_text()) == POLICY


def test_deploy_verifies_values_through_the_backend_readers(source, target):
    record = deploy_mod.deploy(source, target, dry_run=False)
    assert record["artifact_files_verified"] is True
    assert record["hard_class_count"] == 2
    assert record["confusion_pair_count"] == 2


def test_deploy_backs_up_replaced_files_byte_for_byte(source, target):
    before = {n: (target / n).read_bytes() for n in paths.DEPLOYED_FILES}
    record = deploy_mod.deploy(source, target, dry_run=False)
    backup = Path(record["backup_dir"])
    for name, content in before.items():
        assert (backup / name).read_bytes() == content


def test_deploy_never_touches_model_or_calibration_files(source, target):
    write_json(target / "calibration.json", {"temperature": 0.958})
    (target / "resnet50_ft_v2_best.pth").write_bytes(b"checkpoint")
    deploy_mod.deploy(source, target, dry_run=False)
    assert json.loads((target / "calibration.json").read_text()) == {"temperature": 0.958}
    assert (target / "resnet50_ft_v2_best.pth").read_bytes() == b"checkpoint"
    assert json.loads((target / "class_names.json").read_text()) == CLASSES


def test_deploy_writes_provenance_with_hashes(source, target):
    deploy_mod.deploy(source, target, dry_run=False)
    record = json.loads((target / "deployment_provenance.json").read_text())
    assert set(record["deployed_files_sha256"]) == set(paths.DEPLOYED_FILES)
    expected = paths.sha256(source / "derivation_provenance.json")
    assert record["source_provenance_sha256"] == expected


def test_dry_run_writes_nothing(source, target):
    before = {p.name: p.read_bytes() for p in target.iterdir()}
    assert deploy_mod.main(["--source", str(source), "--target", str(target), "--dry-run"]) == 0
    assert {p.name: p.read_bytes() for p in target.iterdir()} == before


@pytest.mark.parametrize(
    ("policy", "fragment"),
    [
        ([POLICY, POLICY], "expected exactly one"),
        ([{"auto_confidence": 0.7, "suggest_confidence": 0.35}], "missing policy keys"),
        ([{**POLICY, "margin_threshold": "0.05"}], "must be a number"),
        ([{**POLICY, "auto_confidence": 1.5}], "in [0, 1]"),
        ([{**POLICY, "auto_confidence": 0.3, "suggest_confidence": 0.5}], "exceeds"),
        ("policy", "must hold a policy object"),
    ],
)
def test_invalid_policy_is_rejected_without_writing(source, target, capsys, policy, fragment):
    write_json(source / "decision_policy.json", policy)
    before = (target / "decision_policy.json").read_bytes()
    assert deploy_mod.main(["--source", str(source), "--target", str(target)]) == 1
    assert fragment in capsys.readouterr().err
    assert (target / "decision_policy.json").read_bytes() == before
    assert not any(p.name.startswith("replaced_") for p in target.iterdir())


@pytest.mark.parametrize(
    ("hard", "fragment"),
    [
        ([], "non-empty"),
        ("steak", "non-empty JSON list"),
        (["steak", "steak"], "duplicate"),
        ([""], "invalid"),
    ],
)
def test_invalid_hard_classes_are_rejected(source, target, capsys, hard, fragment):
    write_json(source / "hard_classes.json", hard)
    assert deploy_mod.main(["--source", str(source), "--target", str(target)]) == 1
    assert fragment in capsys.readouterr().err


def test_labels_unknown_to_the_served_model_are_rejected(source, target, capsys):
    write_json(source / "hard_classes.json", ["not_a_food"])
    assert deploy_mod.main(["--source", str(source), "--target", str(target)]) == 1
    assert "not_a_food" in capsys.readouterr().err


def test_malformed_json_is_a_handled_error(source, target, capsys):
    (source / "confusion_pairs.json").write_text("[{")
    assert deploy_mod.main(["--source", str(source), "--target", str(target)]) == 1
    assert "not valid JSON" in capsys.readouterr().err


# --- The policy must be bound to the served model by producer evidence -------


def test_a_policy_without_producer_evidence_is_rejected(source, target, capsys):
    write_json(source / "derivation_provenance.json", {"generator": "test"})
    before = {p.name: p.read_bytes() for p in target.iterdir()}
    assert deploy_mod.main(["--source", str(source), "--target", str(target)]) == 1
    err = capsys.readouterr().err
    assert "records no producer evidence" in err
    assert "Regenerate evidence by re-scoring" in err
    assert {p.name: p.read_bytes() for p in target.iterdir()} == before


@pytest.mark.parametrize(
    ("change", "fragment"),
    [
        (lambda art: (art / "resnet50_ft_v2_best.pth").write_bytes(b"other"), "checkpoint_sha256"),
        (lambda art: write_json(art / "calibration.json", {"temperature": 0.9581}), "temperature"),
        (lambda art: write_json(art / "class_names.json", CLASSES[::-1]), "class_names_sha256"),
    ],
)
def test_evidence_unlike_the_served_model_is_rejected(source, target, capsys, change, fragment):
    change(target)
    before = {p.name: p.read_bytes() for p in target.iterdir()}
    assert deploy_mod.main(["--source", str(source), "--target", str(target)]) == 1
    assert f"{fragment}: policy evidence" in capsys.readouterr().err
    assert {p.name: p.read_bytes() for p in target.iterdir()} == before


def test_a_target_without_calibration_is_rejected(source, target, capsys):
    (target / "calibration.json").unlink()
    assert deploy_mod.main(["--source", str(source), "--target", str(target)]) == 1
    assert "built-in default temperature" in capsys.readouterr().err


def test_record_names_the_served_model_and_its_files(source, target):
    record = deploy_mod.deploy(source, target, dry_run=True)
    assert record["served_model"]["manifest"] == "legacy_default"
    assert record["served_model"]["temperature"] == TEMPERATURE
    assert record["policy_evidence"]["architecture"] == "resnet50"
    assert set(record["served_model_files_sha256"]) == {
        "resnet50_ft_v2_best.pth", "calibration.json", "class_names.json",
    }


# --- Failure safety: a failed deploy leaves the target exactly as it was ------


def snapshot(directory: Path) -> dict[str, bytes | None]:
    """Every entry in a directory: file bytes, or None for a subdirectory."""
    return {
        p.name: (None if p.is_dir() else p.read_bytes()) for p in sorted(directory.iterdir())
    }


@pytest.fixture
def deployed_target(target: Path) -> Path:
    """A target that already carries provenance from an earlier deployment."""
    write_json(target / "deployment_provenance.json", {"deployed_at": "earlier"})
    return target


def fail_when_reading(directory: Path):
    """A verify_through_backend stand-in that fails only for one directory."""
    real = identity.verify_through_backend

    def verify(where, *args):
        if Path(where).resolve() == directory.resolve():
            raise OSError("simulated backend read failure")
        return real(where, *args)

    return verify


def test_post_install_verification_failure_rolls_back(
    source, deployed_target, monkeypatch, capsys
):
    before = snapshot(deployed_target)
    monkeypatch.setattr(
        identity, "verify_through_backend", fail_when_reading(deployed_target)
    )
    assert deploy_mod.main(["--source", str(source), "--target", str(deployed_target)]) == 1
    err = capsys.readouterr().err
    assert "rolled back" in err
    assert "simulated backend read failure" in err
    assert snapshot(deployed_target) == before


def test_failure_on_second_replace_rolls_back_mixed_state(source, deployed_target, monkeypatch):
    before = snapshot(deployed_target)
    real_replace = install.os.replace
    calls = {"n": 0}

    def flaky_replace(src, dst):
        calls["n"] += 1
        if calls["n"] == 2:
            raise OSError("simulated replace failure")
        return real_replace(src, dst)

    monkeypatch.setattr(install.os, "replace", flaky_replace)
    with pytest.raises(deploy_mod.DeployError, match="rolled back"):
        deploy_mod.deploy(source, deployed_target, dry_run=False)
    assert calls["n"] > 2, "the first file was installed, so rollback must have replaced it"
    assert snapshot(deployed_target) == before


def test_staging_verification_failure_leaves_target_untouched(source, target, monkeypatch):
    before = snapshot(target)
    staged: list[Path] = []

    def fail_on_staging(where, *args):
        staged.append(Path(where))
        raise deploy_mod.DeployError("staged files read back wrong")

    monkeypatch.setattr(identity, "verify_through_backend", fail_on_staging)
    with pytest.raises(deploy_mod.DeployError, match="read back wrong"):
        deploy_mod.deploy(source, target, dry_run=False)
    assert staged and staged[0].name.startswith(".deploy-staging-")
    assert snapshot(target) == before


def test_prior_provenance_is_restored_on_rollback(source, deployed_target, monkeypatch):
    prior = (deployed_target / "deployment_provenance.json").read_bytes()
    monkeypatch.setattr(
        identity, "verify_through_backend", fail_when_reading(deployed_target)
    )
    with pytest.raises(deploy_mod.DeployError, match="simulated backend read failure"):
        deploy_mod.deploy(source, deployed_target, dry_run=False)
    assert (deployed_target / "deployment_provenance.json").read_bytes() == prior


def test_rollback_removes_provenance_that_did_not_exist(source, target, monkeypatch):
    monkeypatch.setattr(identity, "verify_through_backend", fail_when_reading(target))
    with pytest.raises(deploy_mod.DeployError, match="simulated backend read failure"):
        deploy_mod.deploy(source, target, dry_run=False)
    assert not (target / "deployment_provenance.json").exists()


def test_failed_rollback_names_the_preserved_backup(source, target, monkeypatch):
    monkeypatch.setattr(identity, "verify_through_backend", fail_when_reading(target))
    monkeypatch.setattr(
        install.shutil,
        "copy2",
        _copy_then_fail_on_restore(install.shutil.copy2),
    )
    with pytest.raises(deploy_mod.DeployError, match="ROLLBACK FAILED") as info:
        deploy_mod.deploy(source, target, dry_run=False)
    backups = [p for p in target.iterdir() if p.name.startswith("replaced_")]
    assert len(backups) == 1
    assert backups[0].name in str(info.value)
    assert "simulated restore failure" in str(info.value)
    assert (backups[0] / "decision_policy.json").exists()


def _copy_then_fail_on_restore(real_copy):
    def copy(src, dst, *args, **kwargs):
        if Path(dst).name.startswith("restore-"):
            raise OSError("simulated restore failure")
        return real_copy(src, dst, *args, **kwargs)

    return copy


def test_back_to_back_deploys_get_distinct_backups(source, target, monkeypatch):
    class FrozenDatetime(install.datetime):
        @classmethod
        def now(cls, tz=None):
            return install.datetime(2026, 10, 10, 12, 0, 0, tzinfo=tz)

    monkeypatch.setattr(install, "datetime", FrozenDatetime)
    first = deploy_mod.deploy(source, target, dry_run=False)
    second = deploy_mod.deploy(source, target, dry_run=False)
    assert first["backup_dir"] != second["backup_dir"]
    backups = sorted(p.name for p in target.iterdir() if p.name.startswith("replaced_"))
    assert len(backups) == 2
    assert all(name.startswith("replaced_20261010T120000Z_") for name in backups)


def test_an_existing_backup_name_does_not_collide(source, target, monkeypatch):
    class FrozenDatetime(install.datetime):
        @classmethod
        def now(cls, tz=None):
            return install.datetime(2026, 10, 10, 12, 0, 0, tzinfo=tz)

    monkeypatch.setattr(install, "datetime", FrozenDatetime)
    (target / "replaced_20261010T120000Z").mkdir()
    assert deploy_mod.main(["--source", str(source), "--target", str(target)]) == 0
    # The frozen clock was used: the new backup carries its stamp beside the old name.
    backups = sorted(p.name for p in target.iterdir() if p.name.startswith("replaced_"))
    assert backups[0] == "replaced_20261010T120000Z"
    assert len(backups) == 2 and backups[1].startswith("replaced_20261010T120000Z_")


def test_second_backup_preserves_first_deployments_provenance(source, target):
    deploy_mod.deploy(source, target, dry_run=False)
    first_provenance = (target / "deployment_provenance.json").read_bytes()
    record = deploy_mod.deploy(source, target, dry_run=False)
    backup = Path(record["backup_dir"])
    assert (backup / "deployment_provenance.json").read_bytes() == first_provenance
    assert record["replaced_provenance_sha256"] == paths.sha256(
        backup / "deployment_provenance.json"
    )


def test_successful_deploy_leaves_no_staging_directory(source, target):
    deploy_mod.deploy(source, target, dry_run=False)
    assert not any(p.name.startswith(".deploy-staging-") for p in target.iterdir())


def test_unexpected_os_error_is_a_handled_error(source, target, monkeypatch, capsys):
    def no_space(*args, **kwargs):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(install.tempfile, "mkdtemp", no_space)
    assert deploy_mod.main(["--source", str(source), "--target", str(target)]) == 1
    assert "No space left on device" in capsys.readouterr().err


# --- The record names what it verified, and what it did not ------------------


def test_record_claims_file_verification_not_live_service(source, target, capsys):
    assert deploy_mod.main(["--source", str(source), "--target", str(target)]) == 0
    out, err = capsys.readouterr()
    written = json.loads((target / "deployment_provenance.json").read_text())
    for record in (json.loads(out), written):
        assert "backend_verified" not in record
        assert record["artifact_files_verified"] is True
        assert record["decision_layer_fingerprint"] == files_fingerprint(target)
        assert "restart" in record["live_service"]
        assert "--verify-live" in record["live_service"]
    assert "not the live service" in err


# --- --verify-live, driven through an injected fetch -------------------------


def files_fingerprint(directory: Path) -> str:
    return identity.read_through_backend(directory)[3]


def served_model(target: Path) -> dict:
    """The status model block of a service that loaded the target's model and calibration."""
    return {
        "source": "loaded_runtime",
        **identity.target_model_identity(target),
        **identity.target_calibration(target),
    }


def fake_service(fingerprint: str, source: str = "loaded_runtime", fallback=None, model=None):
    """An injectable fetch standing in for a running API process."""
    calls: list[tuple[str, str]] = []

    def fetch(method, url, body, headers):
        calls.append((method, url))
        if url.endswith("/predict/image"):
            assert method == "POST"
            assert headers["Content-Type"].startswith("multipart/form-data; boundary=")
            assert b"\x89PNG" in body
            return {
                "artifact_status": "mock" if fallback else "ready",
                "fallback_reason": fallback,
                # A real runtime reports the temperature it cached at load.
                "temperature": (model or {}).get("temperature"),
            }
        if url.endswith("/runtime/status"):
            status = {"decision_layer": {"source": source, "fingerprint": fingerprint}}
            if model is not None:
                status["model"] = model
            return status
        raise AssertionError(url)

    fetch.calls = calls
    return fetch


def verify_live(target, fetch):
    return deploy_mod.main(
        ["--target", str(target), "--verify-live", "http://api.test:8000/"], fetch=fetch
    )


def test_verify_live_passes_when_the_service_serves_the_deployed_layer(
    source, target, capsys
):
    deploy_mod.deploy(source, target, dry_run=False)
    capsys.readouterr()
    fetch = fake_service(files_fingerprint(target), model=served_model(target))
    assert verify_live(target, fetch) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["live_service_verified"] is True
    assert result["provenance_check"] == "target files match deployment_provenance.json"
    assert fetch.calls == [
        ("POST", "http://api.test:8000/predict/image"),
        ("GET", "http://api.test:8000/runtime/status"),
    ]


def test_verify_live_fails_on_a_stale_cached_fingerprint(source, target, capsys):
    legacy = files_fingerprint(target)
    deploy_mod.deploy(source, target, dry_run=False)
    capsys.readouterr()
    assert verify_live(target, fake_service(legacy, model=served_model(target))) == 1
    err = capsys.readouterr().err
    assert legacy in err
    assert files_fingerprint(target) in err
    assert "restart" in err


def test_verify_live_fails_when_no_runtime_is_loaded(source, target, capsys):
    deploy_mod.deploy(source, target, dry_run=False)
    capsys.readouterr()
    fetch = fake_service(
        files_fingerprint(target), source="artifact_files", model=served_model(target)
    )
    assert verify_live(target, fetch) == 1
    assert "'artifact_files'" in capsys.readouterr().err


def test_verify_live_fails_on_a_fallback_probe(source, target, capsys):
    deploy_mod.deploy(source, target, dry_run=False)
    capsys.readouterr()
    fetch = fake_service(files_fingerprint(target), fallback="classifier_load_error")
    assert verify_live(target, fetch) == 1
    assert "classifier_load_error" in capsys.readouterr().err
    assert fetch.calls == [("POST", "http://api.test:8000/predict/image")]


def test_verify_live_without_a_recorded_fingerprint_compares_files_alone(target, capsys):
    # Provenance as the 2026-10-09 deployment wrote it: no fingerprint.
    write_json(target / "deployment_provenance.json", {"backend_verified": True})
    fetch = fake_service(files_fingerprint(target), model=served_model(target))
    assert verify_live(target, fetch) == 0
    result = json.loads(capsys.readouterr().out)
    assert "compared against the target files alone" in result["provenance_check"]


def test_verify_live_fails_when_files_drift_from_provenance(source, target, capsys):
    deploy_mod.deploy(source, target, dry_run=False)
    write_json(target / "hard_classes.json", ["pizza"])
    capsys.readouterr()
    fetch = fake_service(files_fingerprint(target), model=served_model(target))
    assert verify_live(target, fetch) == 1
    assert "changed after deployment" in capsys.readouterr().err


def test_verify_live_transport_failure_is_a_handled_error(target, capsys):
    def unreachable(method, url, body, headers):
        raise deploy_mod.DeployError(f"{method} {url} failed: connection refused")

    assert verify_live(target, unreachable) == 1
    assert "connection refused" in capsys.readouterr().err


@pytest.mark.parametrize(
    "argv",
    [
        [],
        ["--verify-live", "http://x", "--source", "run"],
        ["--verify-live", "http://x", "--dry-run"],
    ],
)
def test_verify_live_runs_on_its_own(argv):
    with pytest.raises(SystemExit) as info:
        deploy_mod.main(argv)
    assert info.value.code == 2


def test_verify_live_rejects_non_http_urls(target, capsys):
    assert deploy_mod.main(["--target", str(target), "--verify-live", "file:///etc/passwd"]) == 1
    assert "http(s)" in capsys.readouterr().err


# --- --verify-live calibration (policy-only deploys) ---------------------------


def test_verify_live_reports_model_routing_and_calibration_separately(source, target, capsys):
    deploy_mod.deploy(source, target, dry_run=False)
    capsys.readouterr()
    fetch = fake_service(files_fingerprint(target), model=served_model(target))
    assert verify_live(target, fetch) == 0
    result = json.loads(capsys.readouterr().out)
    assert list(result["checks"]) == ["model", "routing", "calibration"]
    assert result["calibration"]["temperature"] == TEMPERATURE


def test_verify_live_catches_calibration_drift_after_a_policy_deploy(source, target, capsys):
    deploy_mod.deploy(source, target, dry_run=False)
    write_json(target / "calibration.json", {"temperature": 2.0})
    capsys.readouterr()
    fetch = fake_service(files_fingerprint(target), model=served_model(target))
    assert verify_live(target, fetch) == 1
    err = capsys.readouterr().err
    assert "calibration: FAILED" in err and "served_model_files_sha256" in err
    assert fetch.calls == []


def test_verify_live_rejects_a_stale_temperature_with_matching_model_and_routing(
    source, target, capsys
):
    deploy_mod.deploy(source, target, dry_run=False)
    capsys.readouterr()
    stale = {**served_model(target), "temperature": 0.884}
    assert verify_live(target, fake_service(files_fingerprint(target), model=stale)) == 1
    err = capsys.readouterr().err
    assert "model: passed" in err and "routing: passed" in err
    assert "cached temperature 0.884" in err
