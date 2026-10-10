"""Model promotion, --restore and the model-identity half of --verify-live.

A model and the policy fitted on its confidences deploy together, as one atomic
unit, through scripts/deploy_decision_policy.py. These tests use tiny stand-in
checkpoints: the model builder is replaced, so no 100MB checkpoint and no
network access are needed.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest
import torch
from evidence_fixtures import provenance_with_evidence, write_scored_split
from torch import nn

from app.backend import artifacts

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "deploy_decision_policy.py"
_spec = importlib.util.spec_from_file_location("deploy_decision_policy_promote", SCRIPT)
deploy_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(deploy_mod)

CLASSES = [f"class_{index:03d}" for index in range(101)]
OLD_POLICY = {"auto_confidence": 0.7, "suggest_confidence": 0.35, "margin_threshold": 0.05}
NEW_POLICY = {"auto_confidence": 0.75, "suggest_confidence": 0.4, "margin_threshold": 0.1}


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value))


def tiny(architecture: str) -> nn.Module:
    """Stand-ins whose state-dict keys differ by architecture, like the real ones."""
    torch.manual_seed(0)
    if architecture == "convnext_tiny":
        return nn.Sequential(nn.Linear(4, 3))
    if architecture == "resnet50":
        return nn.Sequential(nn.Identity(), nn.Linear(4, 3))
    raise deploy_mod.DeployError(f"Unsupported classifier architecture: {architecture!r}.")


@pytest.fixture(autouse=True)
def fake_builder(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(deploy_mod, "build_model", tiny)


def make_run(root: Path, name: str, architecture: str, policy: dict) -> tuple[Path, Path]:
    """A training run with a checkpoint, and a closure policy fitted on its predictions.

    The predictions are re-scored CSVs with evidence sidecars written by the
    rescorer's own writer, bound to this run's checkpoint, class order and
    temperature, and the provenance carries that evidence as recalibration does.
    """
    run = root / name
    run.mkdir()
    torch.save(tiny(architecture).state_dict(), run / f"{name}_best.pth")
    write_json(run / "class_names.json", CLASSES)
    write_json(run / "calibration.json", {"temperature": 0.884})
    scored = {
        split: write_scored_split(
            run / file_name,
            split=split,
            checkpoint=run / f"{name}_best.pth",
            architecture=architecture,
            class_names=CLASSES,
            temperature=0.884,
            marker=name,
        )
        for split, file_name in (("val", "val_predictions.csv"), ("test", "test_predictions.csv"))
    }
    closure = run / "closure"
    closure.mkdir()
    write_json(closure / "decision_policy.json", [{**policy, "fit_split": "val"}])
    write_json(closure / "hard_classes.json", ["class_001", "class_002"])
    write_json(
        closure / "confusion_pairs.json", [{"actual": "class_003", "predicted": "class_004"}]
    )
    write_json(
        closure / "derivation_provenance.json",
        provenance_with_evidence(scored["val"], scored["test"]),
    )
    return run, closure


@pytest.fixture
def runs(tmp_path: Path) -> dict[str, Path]:
    root = tmp_path / "results"
    root.mkdir()
    a3b, a3b_policy = make_run(root, "a3b", "convnext_tiny", NEW_POLICY)
    champion, champion_policy = make_run(root, "champion", "resnet50", OLD_POLICY)
    return {
        "a3b": a3b,
        "a3b_policy": a3b_policy,
        "champion": champion,
        "champion_policy": champion_policy,
    }


@pytest.fixture
def target(tmp_path: Path, runs: dict[str, Path]) -> Path:
    """A legacy ResNet deployment: no model.json, the champion's policy."""
    art = tmp_path / "artifacts"
    art.mkdir()
    (art / "resnet50_ft_v2_best.pth").write_bytes(
        (runs["champion"] / "champion_best.pth").read_bytes()
    )
    (art / "class_names.json").write_text(json.dumps(CLASSES, indent=1))
    write_json(art / "calibration.json", {"temperature": 0.958111})
    write_json(art / "decision_policy.json", OLD_POLICY)
    write_json(art / "hard_classes.json", ["class_005"])
    write_json(art / "confusion_pairs.json", [{"actual": "class_006", "predicted": "class_007"}])
    write_json(art / "deployment_provenance.json", {"deployed_at": "earlier"})
    (art / "demo_predictions.csv").write_text("kept\n")
    return art


def tree(directory: Path, *, bookkeeping: bool = True) -> dict[str, bytes]:
    """Every file under a directory, by relative path.

    With ``bookkeeping=False``, leave out what a deploy or restore writes about
    itself: ``deployment_provenance.json`` and the ``replaced_*`` backups.
    """
    files = {}
    for path in sorted(directory.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(directory)
        if not bookkeeping and (
            relative.name == "deployment_provenance.json" and len(relative.parts) == 1
            or relative.parts[0].startswith("replaced_")
        ):
            continue
        files[str(relative)] = path.read_bytes()
    return files


def promote_argv(runs: dict[str, Path], target: Path, **overrides: str) -> list[str]:
    options = {
        "--source": str(runs["a3b_policy"]),
        "--model-run": str(runs["a3b"]),
        "--architecture": "convnext_tiny",
        "--model-name": "a3b_convnext_tiny",
        "--target": str(target),
        **overrides,
    }
    return [item for pair in options.items() for item in pair]


def backup_path(record: dict) -> Path:
    return deploy_mod.from_display(record["backup_dir"])


def promote(runs, target):
    return deploy_mod.promote(
        runs["a3b_policy"], runs["a3b"], "convnext_tiny", "a3b_convnext_tiny", target, False
    )


# --- A successful promotion ----------------------------------------------------


def test_promotion_installs_model_and_policy_together(runs, target, capsys):
    assert deploy_mod.main(promote_argv(runs, target)) == 0
    record = json.loads(capsys.readouterr().out)

    assert artifacts.read_model_manifest(target) == {
        "architecture": "convnext_tiny",
        "checkpoint": "a3b_best.pth",
        "model_name": "a3b_convnext_tiny",
        "model_run": str(runs["a3b"].resolve()),
    }
    assert (target / "a3b_best.pth").read_bytes() == (runs["a3b"] / "a3b_best.pth").read_bytes()
    assert json.loads((target / "calibration.json").read_text()) == {"temperature": 0.884}
    assert json.loads((target / "decision_policy.json").read_text()) == NEW_POLICY
    assert record["operation"] == "model_promotion"
    assert record["model"]["checkpoint_sha256"] == deploy_mod.sha256(target / "a3b_best.pth")
    assert record["checks"]["checkpoint_fits_architecture"].startswith("loads into convnext_tiny")
    assert record["previous_model"]["architecture"] == "resnet50"
    assert record["previous_model"]["manifest"] == "legacy_default"
    assert set(record["deployed_files_sha256"]) == {
        "a3b_best.pth", "calibration.json", "class_names.json", "model.json",
        "decision_policy.json", "hard_classes.json", "confusion_pairs.json",
    }
    assert json.loads((target / "deployment_provenance.json").read_text()) == record
    assert not any(p.name.startswith(".deploy-staging-") for p in target.iterdir())


def test_promotion_backup_holds_the_full_previous_state(runs, target):
    before = tree(target)
    backup = backup_path(promote(runs, target))

    for name in ("resnet50_ft_v2_best.pth", "calibration.json", "class_names.json",
                 "decision_policy.json", "hard_classes.json", "confusion_pairs.json",
                 "deployment_provenance.json"):
        assert (backup / name).read_bytes() == before[name]
    files = json.loads((backup / "backup_record.json").read_text())["files"]
    assert files["model.json"] is None
    assert files["a3b_best.pth"] is None
    assert files["resnet50_ft_v2_best.pth"] == deploy_mod.sha256(backup / "resnet50_ft_v2_best.pth")


def test_promotion_dry_run_checks_the_checkpoint_and_writes_nothing(runs, target, capsys):
    before = tree(target)
    assert deploy_mod.main([*promote_argv(runs, target), "--dry-run"]) == 0
    record = json.loads(capsys.readouterr().out)
    assert record["dry_run"] is True
    assert "0 missing" in record["checks"]["checkpoint_fits_architecture"]
    assert tree(target) == before


# --- Rejections leave the target untouched -------------------------------------


def assert_rejected(runs, target, capsys, fragment, **overrides):
    before = tree(target)
    assert deploy_mod.main(promote_argv(runs, target, **overrides)) == 1
    assert fragment in capsys.readouterr().err
    assert tree(target) == before
    assert {p.name for p in target.iterdir()} == {Path(name).parts[0] for name in before}


def test_rejects_another_models_policy(runs, target, capsys):
    champion_policy = {"--source": str(runs["champion_policy"])}
    assert_rejected(runs, target, capsys, "not under the model run", **champion_policy)


def test_rejects_reordered_class_names(runs, target, capsys):
    write_json(target / "class_names.json", list(reversed(CLASSES)))
    assert_rejected(runs, target, capsys, "different order")


def test_rejects_different_class_names(runs, target, capsys):
    write_json(target / "class_names.json", [*CLASSES[:-1], "something_else"])
    assert_rejected(runs, target, capsys, "different set of classes")


def test_rejects_a_checkpoint_that_does_not_fit_the_architecture(runs, target, capsys):
    assert_rejected(
        runs, target, capsys, "does not fit resnet50", **{"--architecture": "resnet50"}
    )


def test_rejects_an_unknown_architecture(runs, target, capsys):
    assert_rejected(runs, target, capsys, "unknown architecture", **{"--architecture": "vit"})


def test_rejects_predictions_changed_after_fitting(runs, target, capsys):
    (runs["a3b"] / "test_predictions.csv").write_text("rescored again\n")
    assert_rejected(runs, target, capsys, "changed after the policy was fitted")


# --- Evidence binding: the policy's predictions must come from this model -----
#
# Codex's 2026-10-10 review, finding 1: directory membership plus CSV hashes
# let a promotion keep the fitted CSVs while swapping in different weights or a
# different calibration. These must fail before installation.


def different_weights(architecture: str) -> dict:
    """Valid weights for the architecture that differ from the fixture's."""
    torch.manual_seed(1234)
    model = tiny(architecture)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(1.0)
    return model.state_dict()


def test_rejects_different_valid_weights_with_the_fitted_predictions(runs, target, capsys):
    torch.save(different_weights("convnext_tiny"), runs["a3b"] / "a3b_best.pth")
    # The weights still fit the architecture: only the evidence can tell.
    deploy_mod.check_checkpoint_fits(runs["a3b"] / "a3b_best.pth", "convnext_tiny")
    assert_rejected(runs, target, capsys, "checkpoint_sha256: policy evidence")


def test_rejects_a_changed_calibration_temperature(runs, target, capsys):
    write_json(runs["a3b"] / "calibration.json", {"temperature": 2.0})
    assert_rejected(runs, target, capsys, "temperature: policy evidence 0.884, model 2.0")


def test_rejects_a_policy_without_producer_evidence(runs, target, capsys):
    provenance_path = runs["a3b_policy"] / "derivation_provenance.json"
    provenance = json.loads(provenance_path.read_text())
    for split in ("fit", "eval"):
        del provenance["predictions"][split]["evidence"]
    write_json(provenance_path, provenance)
    assert_rejected(runs, target, capsys, "Regenerate evidence by re-scoring")


def test_dry_run_records_the_evidence_binding(runs, target, capsys):
    assert deploy_mod.main([*promote_argv(runs, target), "--dry-run"]) == 0
    record = json.loads(capsys.readouterr().out)
    assert record["checks"]["policy_evidence_matches_model"].startswith("checkpoint_sha256")
    assert record["policy_evidence"]["temperature"] == 0.884
    assert record["policy_evidence"]["architecture"] == "convnext_tiny"


def test_rejects_a_run_with_several_checkpoints_unless_one_is_named(runs, target, capsys):
    # A byte copy, so its evidence (the fitted predictions' checkpoint) matches.
    (runs["a3b"] / "a3b_last.pth").write_bytes((runs["a3b"] / "a3b_best.pth").read_bytes())
    assert_rejected(runs, target, capsys, "name one with --checkpoint")
    assert deploy_mod.main([*promote_argv(runs, target), "--checkpoint", "a3b_last.pth"]) == 0


@pytest.mark.parametrize(
    "argv",
    [
        ["--model-run", "run", "--source", "policy"],
        ["--architecture", "convnext_tiny", "--source", "policy"],
        ["--restore", "backup", "--source", "policy"],
        ["--verify-live", "http://x", "--model-run", "run"],
        ["--verify-live", "http://x", "--restore", "backup"],
    ],
)
def test_incompatible_options_are_usage_errors(argv):
    with pytest.raises(SystemExit) as info:
        deploy_mod.main(argv)
    assert info.value.code == 2


# --- A failed promotion rolls back to the identical target ---------------------


def test_post_install_failure_rolls_the_promotion_back(runs, target, monkeypatch, capsys):
    before = tree(target)
    real = deploy_mod.verify_model_files

    def fail_on_target(directory, *args):
        if Path(directory).resolve() == target.resolve():
            raise deploy_mod.DeployError("simulated post-install read failure")
        return real(directory, *args)

    monkeypatch.setattr(deploy_mod, "verify_model_files", fail_on_target)
    assert deploy_mod.main(promote_argv(runs, target)) == 1
    err = capsys.readouterr().err
    assert "rolled back" in err and "simulated post-install read failure" in err
    assert tree(target) == before


def test_replace_failure_midway_rolls_the_promotion_back(runs, target, monkeypatch):
    before = tree(target)
    real_replace = deploy_mod.os.replace
    calls = {"n": 0}

    def flaky_replace(src, dst):
        calls["n"] += 1
        if calls["n"] == 5:
            raise OSError("simulated replace failure")
        return real_replace(src, dst)

    monkeypatch.setattr(deploy_mod.os, "replace", flaky_replace)
    with pytest.raises(deploy_mod.DeployError, match="rolled back"):
        promote(runs, target)
    assert tree(target) == before


# --- --restore -----------------------------------------------------------------


def test_restore_round_trip_is_byte_identical(runs, target, capsys):
    before = tree(target, bookkeeping=False)
    original_provenance = json.loads((target / "deployment_provenance.json").read_text())
    promoted = promote(runs, target)

    assert deploy_mod.main(
        ["--target", str(target), "--restore", str(backup_path(promoted))]
    ) == 0
    record = json.loads(capsys.readouterr().out)

    assert tree(target, bookkeeping=False) == before
    assert not (target / "model.json").exists()
    assert not (target / "a3b_best.pth").exists()
    assert artifacts.read_model_manifest(target)["architecture"] == "resnet50"
    written = json.loads((target / "deployment_provenance.json").read_text())
    assert written == record
    assert record["operation"] == "restore"
    assert sorted(record["removed_files"]) == ["a3b_best.pth", "model.json"]
    assert record["restored_deployment_provenance"] == original_provenance
    assert record["model"]["architecture"] == "resnet50"
    assert record["checkpoint_fits_architecture"].startswith("loads into resnet50")
    assert record["decision_layer_fingerprint"] == deploy_mod.read_through_backend(target)[3]


def test_a_restore_can_itself_be_undone(runs, target):
    promote(runs, target)
    promoted = tree(target, bookkeeping=False)
    restore_record = deploy_mod.restore(
        backup_path(json.loads((target / "deployment_provenance.json").read_text())),
        target,
        False,
    )
    deploy_mod.restore(backup_path(restore_record), target, False)
    assert tree(target, bookkeeping=False) == promoted
    assert artifacts.read_model_manifest(target)["architecture"] == "convnext_tiny"


def test_failed_restore_rolls_back(runs, target, monkeypatch):
    record = promote(runs, target)
    promoted = tree(target)
    monkeypatch.setattr(
        deploy_mod,
        "verify_through_backend",
        lambda *args: (_ for _ in ()).throw(deploy_mod.DeployError("simulated read failure")),
    )
    with pytest.raises(deploy_mod.DeployError, match="rolled back"):
        deploy_mod.restore(backup_path(record), target, False)
    assert tree(target) == promoted


def test_restore_dry_run_writes_nothing(runs, target):
    record = promote(runs, target)
    promoted = tree(target)
    result = deploy_mod.restore(backup_path(record), target, True)
    assert result["dry_run"] is True
    assert tree(target) == promoted


def test_restore_refuses_a_damaged_backup(runs, target, capsys):
    record = promote(runs, target)
    (backup_path(record) / "calibration.json").write_text('{"temperature": 2.0}')
    promoted = tree(target)
    assert deploy_mod.main(["--target", str(target), "--restore", str(backup_path(record))]) == 1
    assert "damaged backup" in capsys.readouterr().err
    assert tree(target) == promoted


def test_restore_refuses_a_backup_without_a_record(target, capsys):
    legacy = target / "replaced_20261009T224436Z"
    legacy.mkdir()
    write_json(legacy / "decision_policy.json", OLD_POLICY)
    assert deploy_mod.main(["--target", str(target), "--restore", str(legacy)]) == 1
    assert "backup_record.json" in capsys.readouterr().err


def test_restore_refuses_a_backup_outside_the_target(runs, target, tmp_path, capsys):
    record = promote(runs, target)
    elsewhere = tmp_path / "elsewhere"
    backup_path(record).rename(elsewhere)
    assert deploy_mod.main(["--target", str(target), "--restore", str(elsewhere)]) == 1
    assert "inside the target" in capsys.readouterr().err


# --- Policy-only deploys after a promotion -------------------------------------


def test_policy_deploy_cannot_pair_a_promoted_model_with_another_models_policy(
    runs, target, capsys
):
    promote(runs, target)
    before = tree(target)
    argv = ["--source", str(runs["champion_policy"]), "--target", str(target)]
    assert deploy_mod.main(argv) == 1
    assert "not under the model run" in capsys.readouterr().err
    assert tree(target) == before


def test_policy_deploy_rejects_evidence_for_another_served_checkpoint(runs, target, capsys):
    promote(runs, target)
    torch.save(different_weights("convnext_tiny"), target / "a3b_best.pth")
    before = tree(target)
    argv = ["--source", str(runs["a3b_policy"]), "--target", str(target)]
    assert deploy_mod.main(argv) == 1
    assert "checkpoint_sha256: policy evidence" in capsys.readouterr().err
    assert tree(target) == before


def test_policy_deploy_rejects_evidence_for_another_served_temperature(runs, target, capsys):
    promote(runs, target)
    write_json(target / "calibration.json", {"temperature": 2.0})
    before = tree(target)
    argv = ["--source", str(runs["a3b_policy"]), "--target", str(target)]
    assert deploy_mod.main(argv) == 1
    assert "temperature: policy evidence 0.884, model 2.0" in capsys.readouterr().err
    assert tree(target) == before


def test_policy_deploy_to_a_legacy_target_binds_to_the_legacy_resnet(runs, target, capsys):
    # No model.json: the served model is the legacy ResNet50 default, whose
    # checkpoint bytes are the champion's. Only the temperature differs.
    argv = ["--source", str(runs["champion_policy"]), "--target", str(target), "--dry-run"]
    assert deploy_mod.main(argv) == 1
    assert "temperature: policy evidence 0.884, model 0.958111" in capsys.readouterr().err
    write_json(target / "calibration.json", {"temperature": 0.884})
    assert deploy_mod.main(argv) == 0
    record = json.loads(capsys.readouterr().out)
    assert record["served_model"]["manifest"] == "legacy_default"
    assert record["served_model"]["architecture"] == "resnet50"


def test_policy_deploy_for_the_promoted_model_leaves_model_files_alone(runs, target):
    promote(runs, target)
    model_files = {
        name: (target / name).read_bytes()
        for name in ("a3b_best.pth", "model.json", "calibration.json", "class_names.json")
    }
    record = deploy_mod.deploy(runs["a3b_policy"], target, dry_run=False)
    assert "model.json" in record["untouched"]
    assert record["policy_belongs_to_served_model"].startswith("fit and eval predictions")
    for name, content in model_files.items():
        assert (target / name).read_bytes() == content


# --- --verify-live model identity ----------------------------------------------


def service(target: Path, model: dict | None, probe_temperature: float | None = None):
    """A fake API that serves the target's decision layer and the given model block.

    The probe response reports ``probe_temperature``, or by default the model
    block's temperature, as a real runtime reports the one it cached.
    """
    fingerprint = deploy_mod.read_through_backend(target)[3]
    if probe_temperature is None and model is not None:
        probe_temperature = model.get("temperature")

    def fetch(method, url, body, headers):
        if url.endswith("/predict/image"):
            return {
                "artifact_status": "ready",
                "fallback_reason": None,
                "temperature": probe_temperature,
            }
        status = {"decision_layer": {"source": "loaded_runtime", "fingerprint": fingerprint}}
        if model is not None:
            status["model"] = model
        return status

    return fetch


def verify(target: Path, fetch) -> int:
    return deploy_mod.main(["--target", str(target), "--verify-live", "http://api.test"], fetch)


def runtime_calibration(target: Path) -> dict:
    """The temperature and class-names hash a runtime that loaded the target reports."""
    return {
        "temperature": json.loads((target / "calibration.json").read_text())["temperature"],
        "class_names_sha256": artifacts.class_names_sha256(
            json.loads((target / "class_names.json").read_text())
        ),
    }


def loaded(target: Path) -> dict:
    """The status model block of a runtime freshly loaded from the target's files."""
    return {
        "source": "loaded_runtime",
        **deploy_mod.target_model_identity(target),
        **runtime_calibration(target),
    }


def test_verify_live_passes_and_reports_the_promoted_model(runs, target, capsys):
    promote(runs, target)
    capsys.readouterr()
    assert verify(target, service(target, loaded(target))) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["model"]["architecture"] == "convnext_tiny"
    assert result["model"]["model_name"] == "a3b_convnext_tiny"


def test_verify_live_fails_when_the_service_kept_the_old_model(runs, target, capsys):
    old_model = loaded(target)
    promote(runs, target)
    capsys.readouterr()
    # Same decision layer, old model: only the identity check can catch this.
    assert verify(target, service(target, old_model)) == 1
    err = capsys.readouterr().err
    assert "different model" in err
    assert "architecture: service 'resnet50', target 'convnext_tiny'" in err
    assert "restart" in err


@pytest.mark.parametrize(
    ("model", "fragment"),
    [
        (None, "no model block"),
        ({"source": "artifact_files"}, "model.source='artifact_files'"),
    ],
)
def test_verify_live_fails_without_a_loaded_model_identity(runs, target, capsys, model, fragment):
    promote(runs, target)
    capsys.readouterr()
    assert verify(target, service(target, model)) == 1
    assert fragment in capsys.readouterr().err


def test_verify_live_fails_when_the_checkpoint_changed_after_deployment(runs, target, capsys):
    promote(runs, target)
    (target / "a3b_best.pth").write_bytes((target / "a3b_best.pth").read_bytes() + b"x")
    capsys.readouterr()
    assert verify(target, service(target, loaded(target))) == 1
    assert "changed after deployment" in capsys.readouterr().err

