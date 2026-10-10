"""Verify the repo's documentation obeys the structure rules in docs/0_coding_standards.md.

Run from the repo root. Exits non-zero and lists offenders when a rule is
broken; exits zero when the docs are clean.

S0 introduced these rules but nothing enforced them, and that gap already
cost real defects: a doc renumbering renamed files without updating their H1
headings, and AGENTS.md carried line counts a later commit had made false.
Both were caught by hand in a final review. This script catches them instead.

Checks:

1. CLAUDE.md is exactly one line containing "@AGENTS.md" (Master Sec.13: one
   source of truth, not two that drift).
2. AGENTS.md is between 20 and 40 non-blank lines (it loads on every
   session, so it earns its length).
3. Every docs/<N>_*.md file's first H1 begins "# <N>. ", matching its
   filename.
4. No two numbered docs share a number.
5. No gaps in the numbering sequence from 0 to the highest present.
6. docs/README.md indexes every numbered doc, and every doc it indexes
   exists -- checked in both directions.
7. docs/0_coding_standards.md has not re-grown into a copy of the master
   standard (Master Sec.13 forbids duplicating it). This is heuristic, not
   proof, and the output says so.
"""

# This script is run directly from a shell, sometimes by whatever `python3`
# is on PATH rather than the project venv. Deferring annotation evaluation
# keeps it importable on interpreters older than the 3.11 the project
# targets, so the doc gate never fails for a reason unrelated to the docs.
from __future__ import annotations

import re
import sys
from pathlib import Path

NUMBERED_DOC_GLOB = "docs/[0-9]*_*.md"
NUMBERED_DOC_PATTERN = re.compile(r"^(\d+)_.+\.md$")
HEADING_NUMBER_PATTERN = re.compile(r"^#\s+(\d+)\.")
README_LINK_PATTERN = re.compile(r"\((\d+_[^)]+\.md)\)")

AGENTS_MIN_NON_BLANK_LINES = 20
AGENTS_MAX_NON_BLANK_LINES = 40

CODING_STANDARDS_LINE_LIMIT = 120
CODING_STANDARDS_MASTER_PHRASES = (
    "PEP 8",
    "Conventional Commits",
    "4-space indentation",
    "Google-style docstring",
    "snake_case",
    "Viridis",
)


def numbered_docs(root: Path) -> list[Path]:
    """Return every docs/<N>_*.md file under `root`, newest rules included.

    Discovered by glob rather than a hardcoded list, so the check does not
    go stale the moment a doc is added, renamed, or removed.
    """
    return sorted(root.glob(NUMBERED_DOC_GLOB), key=lambda path: path.name)


def doc_number(path: Path) -> int | None:
    """Return the leading number in `path`'s filename, or None if it has none."""
    match = NUMBERED_DOC_PATTERN.match(path.name)
    return int(match.group(1)) if match else None


def check_claude_md(root: Path) -> list[str]:
    """Check that CLAUDE.md is exactly one line containing "@AGENTS.md"."""
    path = root / "CLAUDE.md"
    if not path.exists():
        return [f"{path}: file does not exist"]
    lines = path.read_text(encoding="utf-8").splitlines()
    if lines != ["@AGENTS.md"]:
        return [
            f"{path}: must be exactly one line containing '@AGENTS.md' "
            f"(Master Sec.13 requires one source of truth, not two that "
            f"drift); found {len(lines)} line(s) instead"
        ]
    return []


def check_agents_md(root: Path) -> list[str]:
    """Check that AGENTS.md has between 20 and 40 non-blank lines."""
    path = root / "AGENTS.md"
    if not path.exists():
        return [f"{path}: file does not exist"]
    non_blank = [
        line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    count = len(non_blank)
    if not (AGENTS_MIN_NON_BLANK_LINES <= count <= AGENTS_MAX_NON_BLANK_LINES):
        return [
            f"{path}: has {count} non-blank line(s); must be between "
            f"{AGENTS_MIN_NON_BLANK_LINES} and {AGENTS_MAX_NON_BLANK_LINES} "
            f"(it loads on every session, so it earns its length)"
        ]
    return []


def check_heading_numbers(docs: list[Path]) -> list[str]:
    """Check that each numbered doc's first H1 begins with its own number."""
    problems = []
    for path in docs:
        expected = doc_number(path)
        if expected is None:
            continue
        heading_line = None
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.startswith("# "):
                heading_line = line
                break
        match = HEADING_NUMBER_PATTERN.match(heading_line) if heading_line else None
        if match is None:
            problems.append(
                f"{path}: filename implies heading number {expected}, but no H1 "
                f"of the form '# {expected}. ...' was found"
            )
        elif int(match.group(1)) != expected:
            problems.append(
                f"{path}: filename implies heading number {expected}, but the H1 "
                f"reads {match.group(1)!r} ({heading_line!r})"
            )
    return problems


def check_unique_numbers(docs: list[Path]) -> list[str]:
    """Check that no two numbered docs share the same leading number."""
    by_number: dict[int, list[Path]] = {}
    for path in docs:
        number = doc_number(path)
        if number is None:
            continue
        by_number.setdefault(number, []).append(path)
    problems = []
    for number, paths in sorted(by_number.items()):
        if len(paths) > 1:
            names = ", ".join(str(path) for path in paths)
            problems.append(f"number {number} is used by more than one doc: {names}")
    return problems


def check_no_gaps(docs: list[Path]) -> list[str]:
    """Check the numbering sequence has no gaps from 0 to the highest present."""
    numbers = sorted({doc_number(path) for path in docs if doc_number(path) is not None})
    if not numbers:
        return []
    missing = [n for n in range(0, numbers[-1] + 1) if n not in numbers]
    if missing:
        gaps = ", ".join(str(n) for n in missing)
        return [
            f"numbering has gap(s) at {gaps} (a missing number usually means a "
            f"file was deleted without renumbering, or added at the wrong index)"
        ]
    return []


def check_readme_index(root: Path, docs: list[Path]) -> list[str]:
    """Check docs/README.md indexes every numbered doc, and vice versa."""
    readme_path = root / "docs" / "README.md"
    if not readme_path.exists():
        return [f"{readme_path}: file does not exist"]
    text = readme_path.read_text(encoding="utf-8")
    indexed = set(README_LINK_PATTERN.findall(text))
    existing = {path.name for path in docs}
    problems = []
    for name in sorted(existing - indexed):
        problems.append(f"{readme_path}: does not index {name}")
    for name in sorted(indexed - existing):
        problems.append(f"{readme_path}: indexes {name}, but that file does not exist")
    return problems


def check_coding_standards_size(root: Path) -> list[str]:
    """Heuristically check that 0_coding_standards.md has not re-grown into a
    copy of the master standard.

    Two heuristics, neither proof: a line-count ceiling, and a search for
    phrases that belong to the shared master standard rather than to this
    project's deltas from it.
    """
    path = root / "docs" / "0_coding_standards.md"
    if not path.exists():
        return []
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    problems = []
    if len(lines) > CODING_STANDARDS_LINE_LIMIT:
        problems.append(
            f"{path}: {len(lines)} lines, over the heuristic "
            f"{CODING_STANDARDS_LINE_LIMIT}-line threshold -- this is a heuristic, "
            f"not proof, but it suggests the file may be restating the shared "
            f"master standard (Master Sec.13) instead of recording this "
            f"project's deltas from it"
        )
    lowered = text.lower()
    hits = [phrase for phrase in CODING_STANDARDS_MASTER_PHRASES if phrase.lower() in lowered]
    if hits:
        problems.append(
            f"{path}: contains master-standard phrase(s) {hits} -- this is a "
            f"heuristic, not proof, but a hit may mean the file is restating the "
            f"shared standard instead of recording a deviation from it"
        )
    return problems


def collect_problems(root: Path) -> list[str]:
    """Run every check against `root` and return all findings, unordered by severity."""
    docs = numbered_docs(root)
    problems: list[str] = []
    problems += check_claude_md(root)
    problems += check_agents_md(root)
    problems += check_heading_numbers(docs)
    problems += check_unique_numbers(docs)
    problems += check_no_gaps(docs)
    problems += check_readme_index(root, docs)
    problems += check_coding_standards_size(root)
    return problems


def main() -> int:
    root = Path.cwd()
    problems = collect_problems(root)
    for problem in problems:
        print(problem)
    if problems:
        print(f"\n{len(problems)} documentation structure problem(s).")
        return 1
    print("All documentation structure checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
