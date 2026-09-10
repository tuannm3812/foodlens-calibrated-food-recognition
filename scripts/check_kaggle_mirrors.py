"""Report how far each kaggle/*/ training script has drifted from its notebook.

Every `kaggle/<run>/` directory that holds a `foodlens_*.py` training script
also holds a `.ipynb` that Kaggle actually executes (`kernel-metadata.json`
names the notebook as `code_file`). The `.py` is meant to mirror the
notebook's code cells, but the two have never been kept in sync -- not before
this script existed and not after. This script makes that drift measurable
instead of leaving it invisible.

This check is report-only by design and always exits 0 for drift, no matter
how large. It does not fail the build. The mirrors are already out of sync
today, so a blocking check would fail the moment it landed and everyone would
learn to ignore or bypass it rather than fix anything -- a check nobody heeds
is worse than no check. The only failure this script reports is a pairing it
cannot even resolve: a run directory with no notebook, or more than one,
which is a broken directory rather than ordinary drift.

Run directly, or with `--diff` to also print a unified diff per pair.
"""

# Run with whatever `python3` is on PATH, not necessarily the project venv;
# see scripts/check_doc_links.py for why annotations must stay deferred.
from __future__ import annotations

import argparse
import difflib
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
KAGGLE_ROOT = REPO_ROOT / "kaggle"


def notebook_code_lines(notebook_path: Path) -> list[str]:
    """Return the lines of a notebook's code cells, concatenated in order.

    Each code cell's `source` is joined into one string; a cell whose source
    does not already end in a newline gets one appended so it cannot run
    together with the next cell's first line. The cells are then concatenated
    in notebook order and split into lines, mirroring how the cells would
    read if exported top to bottom.

    Args:
        notebook_path: Path to a `.ipynb` file.

    Returns:
        The notebook's code-cell source, one entry per line.
    """
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    parts: list[str] = []
    for cell in notebook.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        source = cell.get("source", "")
        if isinstance(source, list):
            source = "".join(source)
        if source and not source.endswith("\n"):
            source += "\n"
        parts.append(source)
    return "".join(parts).splitlines()


def find_pair(script_path: Path) -> Path | None:
    """Return the single `.ipynb` in `script_path`'s directory, if resolvable.

    Args:
        script_path: Path to a `kaggle/<run>/foodlens_*.py` training script.

    Returns:
        The matching notebook path, or `None` if the directory holds zero or
        more than one `.ipynb` -- a pairing that cannot be resolved.
    """
    notebooks = sorted(script_path.parent.glob("*.ipynb"))
    if len(notebooks) != 1:
        return None
    return notebooks[0]


def report_pair(script_path: Path, notebook_path: Path, show_diff: bool) -> float:
    """Print the line-count and similarity report for one script/notebook pair.

    Args:
        script_path: The `.py` training script.
        notebook_path: The `.ipynb` it mirrors.
        show_diff: Whether to also print a unified diff of the two line lists.

    Returns:
        The `difflib.SequenceMatcher` ratio between the script and the
        notebook's code cells, in the range [0.0, 1.0].
    """
    run_name = script_path.parent.name
    py_lines = script_path.read_text(encoding="utf-8").splitlines()
    nb_lines = notebook_code_lines(notebook_path)
    ratio = difflib.SequenceMatcher(None, py_lines, nb_lines).ratio()

    print(f"{run_name}:")
    print(f"  script:   {script_path.relative_to(REPO_ROOT)} ({len(py_lines)} lines)")
    print(f"  notebook: {notebook_path.relative_to(REPO_ROOT)} ({len(nb_lines)} lines)")
    print(f"  similarity ratio: {ratio:.4f}")

    if show_diff:
        diff = difflib.unified_diff(
            py_lines,
            nb_lines,
            fromfile=str(script_path.relative_to(REPO_ROOT)),
            tofile=str(notebook_path.relative_to(REPO_ROOT)),
            lineterm="",
        )
        diff_text = "\n".join(diff)
        if diff_text:
            print(diff_text)
        else:
            print("  (no line-level differences)")

    return ratio


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Argument list to parse; defaults to `sys.argv[1:]`.

    Returns:
        The parsed namespace, exposing `diff: bool`.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--diff",
        action="store_true",
        help="Also print a unified diff for each script/notebook pair.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Report mirror drift for every kaggle run, returning the process exit code.

    Args:
        argv: Argument list to parse; defaults to `sys.argv[1:]`.

    Returns:
        0 if every `foodlens_*.py` paired with exactly one notebook,
        regardless of how much the pair has drifted; 1 if any pairing could
        not be resolved.
    """
    args = parse_args(argv)
    scripts = sorted(KAGGLE_ROOT.glob("*/foodlens_*.py"))

    unresolved: list[Path] = []
    for script_path in scripts:
        notebook_path = find_pair(script_path)
        if notebook_path is None:
            unresolved.append(script_path)
            continue
        report_pair(script_path, notebook_path, args.diff)

    if unresolved:
        print("\nCould not resolve a notebook pairing for:", file=sys.stderr)
        for script_path in unresolved:
            candidates = sorted(script_path.parent.glob("*.ipynb"))
            print(
                f"  {script_path.relative_to(REPO_ROOT)}: found {len(candidates)} .ipynb file(s)",
                file=sys.stderr,
            )
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
