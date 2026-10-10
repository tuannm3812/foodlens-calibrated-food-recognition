"""Tests for scripts/check_kaggle_mirrors.py's choice of authoritative file.

The checker used to assume the notebook was always what Kaggle executed. A4's
kernel metadata names its `.py` as `code_file` (a `script` kernel), so for A4
the script is the run record and the notebook is the mirror. Flagged by the
Codex GitHub reviewer on PR #4.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "check_kaggle_mirrors.py"
_spec = importlib.util.spec_from_file_location("check_kaggle_mirrors", SCRIPT)
mirrors = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mirrors)


@pytest.fixture
def run_dir(tmp_path: Path) -> Path:
    run = tmp_path / "accuracy_run"
    run.mkdir()
    (run / "foodlens_run.py").write_text("print('x')\n")
    (run / "09_run.ipynb").write_text("{}")
    return run


def write_metadata(run: Path, code_file: str) -> None:
    (run / "kernel-metadata.json").write_text(json.dumps({"code_file": code_file}))


def test_notebook_code_file_marks_the_notebook(run_dir: Path) -> None:
    write_metadata(run_dir, "09_run.ipynb")
    side = mirrors.authoritative_side(run_dir / "foodlens_run.py", run_dir / "09_run.ipynb")
    assert side == "notebook"


def test_script_code_file_marks_the_script(run_dir: Path) -> None:
    write_metadata(run_dir, "foodlens_run.py")
    side = mirrors.authoritative_side(run_dir / "foodlens_run.py", run_dir / "09_run.ipynb")
    assert side == "script"


def test_code_file_naming_neither_is_reported(run_dir: Path) -> None:
    write_metadata(run_dir, "other.py")
    side = mirrors.authoritative_side(run_dir / "foodlens_run.py", run_dir / "09_run.ipynb")
    assert side.startswith("unknown") and "other.py" in side


def test_missing_metadata_is_reported_not_assumed(run_dir: Path) -> None:
    side = mirrors.authoritative_side(run_dir / "foodlens_run.py", run_dir / "09_run.ipynb")
    assert side.startswith("unknown")


def test_real_a4_run_is_a_script_kernel() -> None:
    a4 = SCRIPT.parents[1] / "kaggle" / "accuracy_phase1_a4"
    script = a4 / "foodlens_accuracy_phase1_a4.py"
    notebook = a4 / "foodlens_accuracy_phase1_a4.ipynb"
    assert mirrors.authoritative_side(script, notebook) == "script"
