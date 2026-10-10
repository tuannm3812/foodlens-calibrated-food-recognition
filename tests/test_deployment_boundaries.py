"""Enforce the dependency rules of `app/deployment/`.

The deploy script's last four correctness findings sat at seams between
operations that each built or read records their own way. The package split
(docs/superpowers/specs/2026-10-11-deployment-package-design.md) draws
boundaries to stop that recurring; this test keeps them from eroding quietly.
It reads imports statically, so it needs no torch and no artifacts.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "app" / "deployment"
OPERATIONS = {"policy", "promote", "restore"}


def sibling_imports(module: str) -> set[str]:
    """The `app.deployment` modules a module imports, by short name."""
    tree = ast.parse((PACKAGE / f"{module}.py").read_text(encoding="utf-8"))
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.level == 1 and node.module is None:
                found.update(alias.name for alias in node.names)
            elif node.level == 1:
                found.add(node.module.split(".")[0])
            elif node.module and node.module.startswith("app.deployment."):
                found.add(node.module.split(".")[2])
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("app.deployment."):
                    found.add(alias.name.split(".")[2])
    return found


def all_imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            found.add(node.module)
        elif isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
    return found


def test_the_package_has_the_designed_modules() -> None:
    modules = {path.stem for path in PACKAGE.glob("*.py")} - {"__init__"}
    assert modules == {
        "errors", "paths", "identity", "records", "install",
        "policy", "promote", "restore", "verify_live",
    }


def test_live_verification_never_imports_a_write_operation() -> None:
    assert sibling_imports("verify_live") & (OPERATIONS | {"install"}) == set()


def test_records_hold_data_only() -> None:
    assert sibling_imports("records") <= {"paths", "errors"}
    assert not all_imports(PACKAGE / "records.py") & {"torch", "torchvision", "shutil", "tempfile"}


def test_the_installer_knows_nothing_of_models_or_policies() -> None:
    assert sibling_imports("install") <= {"paths", "errors"}


def test_shared_layers_never_import_operations() -> None:
    for module in ("paths", "errors", "identity", "records", "install"):
        assert sibling_imports(module) & OPERATIONS == set(), module


@pytest.mark.parametrize("operation", sorted(OPERATIONS))
def test_operations_do_not_import_each_other(operation: str) -> None:
    assert sibling_imports(operation) & (OPERATIONS - {operation}) == set()


def test_the_backend_never_imports_deployment() -> None:
    offenders = [
        path.name
        for path in (ROOT / "app" / "backend").glob("*.py")
        if any(name.startswith("app.deployment") for name in all_imports(path))
        or "from ..deployment" in path.read_text(encoding="utf-8")
    ]
    assert offenders == []


def test_the_cli_stays_thin() -> None:
    lines = (ROOT / "scripts" / "deploy_decision_policy.py").read_text().count("\n")
    assert lines < 300, f"the deploy CLI has grown to {lines} lines; move logic into app/deployment"
