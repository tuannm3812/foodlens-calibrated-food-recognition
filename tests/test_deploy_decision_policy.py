"""Tests for scripts/deploy_decision_policy.py.

The central regression: a recalibration run writes decision_policy.json as a
one-element list, the backend reads a dict, and copying the file verbatim makes
every request fall back to demo output while artifact_status still reports
"ready". The deploy step must convert, validate, back up and verify instead.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "deploy_decision_policy.py"
_spec = importlib.util.spec_from_file_location("deploy_decision_policy", SCRIPT)
deploy_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(deploy_mod)

CLASSES = ["apple_pie", "bread_pudding", "pizza", "steak", "filet_mignon", "sushi"]
POLICY = {"auto_confidence": 0.7, "suggest_confidence": 0.35, "margin_threshold": 0.05}


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
    write_json(run / "derivation_provenance.json", {"generator": "test"})
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
    return art


def test_deploy_converts_list_policy_to_the_dict_the_backend_reads(source, target):
    assert deploy_mod.main(["--source", str(source), "--target", str(target)]) == 0
    assert json.loads((target / "decision_policy.json").read_text()) == POLICY


def test_deploy_verifies_values_through_the_backend_readers(source, target):
    record = deploy_mod.deploy(source, target, dry_run=False)
    assert record["backend_verified"] is True
    assert record["hard_class_count"] == 2
    assert record["confusion_pair_count"] == 2


def test_deploy_backs_up_replaced_files_byte_for_byte(source, target):
    before = {n: (target / n).read_bytes() for n in deploy_mod.DEPLOYED_FILES}
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
    assert set(record["deployed_files_sha256"]) == set(deploy_mod.DEPLOYED_FILES)
    expected = deploy_mod.sha256(source / "derivation_provenance.json")
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
