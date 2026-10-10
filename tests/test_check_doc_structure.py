"""Regression tests for scripts/check_doc_structure.py.

S0 introduced documentation-structure rules but nothing enforced them, and
that gap already cost real defects: a doc renumbering renamed files without
updating their H1 headings, and AGENTS.md carried line counts a later commit
had made false. These tests pin down each of the seven checks the script
implements, with a passing and a failing case apiece.

The module lives in scripts/, not a package, so it is loaded by file path
rather than imported normally.
"""

import importlib.util
from pathlib import Path

SCRIPT_PATH = Path(__file__).resolve().parent.parent / "scripts" / "check_doc_structure.py"

_spec = importlib.util.spec_from_file_location("check_doc_structure", SCRIPT_PATH)
assert _spec is not None and _spec.loader is not None
check_doc_structure = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(check_doc_structure)

check_claude_md = check_doc_structure.check_claude_md
check_agents_md = check_doc_structure.check_agents_md
check_heading_numbers = check_doc_structure.check_heading_numbers
check_unique_numbers = check_doc_structure.check_unique_numbers
check_no_gaps = check_doc_structure.check_no_gaps
check_readme_index = check_doc_structure.check_readme_index
check_coding_standards_size = check_doc_structure.check_coding_standards_size
numbered_docs = check_doc_structure.numbered_docs
collect_problems = check_doc_structure.collect_problems


def _write_valid_docs_tree(root: Path) -> None:
    """Build a minimal, fully-passing docs/ tree plus root CLAUDE.md/AGENTS.md."""
    docs = root / "docs"
    docs.mkdir(parents=True, exist_ok=True)
    (docs / "0_coding_standards.md").write_text(
        "# 0. Coding Standards\n\nDeltas from the master only.\n", encoding="utf-8"
    )
    (docs / "1_instructions.md").write_text(
        "# 1. Project Instructions\n\nBody text.\n", encoding="utf-8"
    )
    (docs / "README.md").write_text(
        "# Documentation Index\n\n"
        "| File | Purpose |\n| --- | --- |\n"
        "[`0_coding_standards.md`](0_coding_standards.md)\n"
        "[`1_instructions.md`](1_instructions.md)\n",
        encoding="utf-8",
    )
    (root / "CLAUDE.md").write_text("@AGENTS.md\n", encoding="utf-8")
    (root / "AGENTS.md").write_text(
        "\n".join(f"Line {i} of the project agent guide." for i in range(25)) + "\n",
        encoding="utf-8",
    )


# --- Check 1: CLAUDE.md is exactly one line containing "@AGENTS.md" ---


def test_claude_md_with_exactly_one_line_passes(tmp_path: Path) -> None:
    (tmp_path / "CLAUDE.md").write_text("@AGENTS.md\n", encoding="utf-8")

    assert check_claude_md(tmp_path) == []


def test_claude_md_with_extra_line_is_reported(tmp_path: Path) -> None:
    (tmp_path / "CLAUDE.md").write_text("@AGENTS.md\nSome extra note.\n", encoding="utf-8")

    problems = check_claude_md(tmp_path)

    assert len(problems) == 1
    assert "must be exactly one line containing '@AGENTS.md'" in problems[0]
    assert "found 2 line(s)" in problems[0]


# --- Check 2: AGENTS.md is between 20 and 40 non-blank lines ---


def test_agents_md_within_range_passes(tmp_path: Path) -> None:
    (tmp_path / "AGENTS.md").write_text("\n".join(f"line {i}" for i in range(25)), encoding="utf-8")

    assert check_agents_md(tmp_path) == []


def test_agents_md_too_short_is_reported(tmp_path: Path) -> None:
    (tmp_path / "AGENTS.md").write_text("\n".join(f"line {i}" for i in range(5)), encoding="utf-8")

    problems = check_agents_md(tmp_path)

    assert len(problems) == 1
    assert "has 5 non-blank line(s)" in problems[0]
    assert "must be between 20 and 40" in problems[0]


# --- Check 3: numbered heading matches filename ---


def test_heading_matching_filename_number_passes(tmp_path: Path) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    doc = docs / "3_model_results.md"
    doc.write_text("# 3. Model Results\n\nBody.\n", encoding="utf-8")

    assert check_heading_numbers([doc]) == []


def test_heading_mismatched_with_filename_number_is_reported(tmp_path: Path) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    doc = docs / "3_model_results.md"
    doc.write_text("# 8. Model Results\n\nBody.\n", encoding="utf-8")

    problems = check_heading_numbers([doc])

    assert len(problems) == 1
    assert "filename implies heading number 3" in problems[0]
    assert "reads '8'" in problems[0]


# --- Check 4: no two numbered docs share a number ---


def test_unique_numbers_with_no_duplicates_passes(tmp_path: Path) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    a = docs / "1_a.md"
    b = docs / "2_b.md"
    a.write_text("# 1. A\n", encoding="utf-8")
    b.write_text("# 2. B\n", encoding="utf-8")

    assert check_unique_numbers([a, b]) == []


def test_duplicate_numbers_are_reported(tmp_path: Path) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    a = docs / "8_runtime_contract.md"
    b = docs / "8_agent_log.md"
    a.write_text("# 8. Runtime Contract\n", encoding="utf-8")
    b.write_text("# 8. Agent Log\n", encoding="utf-8")

    problems = check_unique_numbers([a, b])

    assert len(problems) == 1
    assert "number 8 is used by more than one doc" in problems[0]
    assert "8_runtime_contract.md" in problems[0]
    assert "8_agent_log.md" in problems[0]


# --- Check 5: no gaps in the numbering sequence from 0 to the highest present ---


def test_contiguous_numbering_passes(tmp_path: Path) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    paths = [docs / f"{n}_doc.md" for n in (0, 1, 2)]
    for path in paths:
        path.write_text("placeholder\n", encoding="utf-8")

    assert check_no_gaps(paths) == []


def test_gap_in_numbering_is_reported(tmp_path: Path) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    paths = [docs / f"{n}_doc.md" for n in (0, 1, 3)]
    for path in paths:
        path.write_text("placeholder\n", encoding="utf-8")

    problems = check_no_gaps(paths)

    assert len(problems) == 1
    assert "numbering has gap(s) at 2" in problems[0]


# --- Check 6: docs/README.md indexes every numbered doc, and vice versa ---


def test_readme_indexing_every_doc_passes(tmp_path: Path) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    doc = docs / "1_a.md"
    doc.write_text("# 1. A\n", encoding="utf-8")
    (docs / "README.md").write_text("[`1_a.md`](1_a.md)\n", encoding="utf-8")

    assert check_readme_index(tmp_path, [doc]) == []


def test_doc_missing_from_readme_index_is_reported(tmp_path: Path) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    doc = docs / "1_a.md"
    doc.write_text("# 1. A\n", encoding="utf-8")
    (docs / "README.md").write_text("No links here.\n", encoding="utf-8")

    problems = check_readme_index(tmp_path, [doc])

    assert len(problems) == 1
    assert "does not index 1_a.md" in problems[0]


def test_readme_index_entry_with_no_matching_doc_is_reported(tmp_path: Path) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "README.md").write_text("[`2_gone.md`](2_gone.md)\n", encoding="utf-8")

    problems = check_readme_index(tmp_path, [])

    assert len(problems) == 1
    assert "indexes 2_gone.md, but that file does not exist" in problems[0]


# --- Check 7: 0_coding_standards.md has not re-grown into a copy of the master ---


def test_short_coding_standards_without_master_phrases_passes(tmp_path: Path) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "0_coding_standards.md").write_text(
        "# 0. Coding Standards\n\nThis repo differs from the master by using Shape A.\n",
        encoding="utf-8",
    )

    assert check_coding_standards_size(tmp_path) == []


def test_oversized_coding_standards_is_reported_as_heuristic(tmp_path: Path) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    body = "\n".join(f"Line {i} of padding text." for i in range(150))
    (docs / "0_coding_standards.md").write_text(
        f"# 0. Coding Standards\n\n{body}\n", encoding="utf-8"
    )

    problems = check_coding_standards_size(tmp_path)

    assert len(problems) == 1
    assert "over the heuristic 120-line threshold" in problems[0]
    assert "heuristic, not proof" in problems[0]


def test_coding_standards_with_master_phrase_is_reported_as_heuristic(tmp_path: Path) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "0_coding_standards.md").write_text(
        "# 0. Coding Standards\n\nFollow PEP 8 and use snake_case everywhere.\n",
        encoding="utf-8",
    )

    problems = check_coding_standards_size(tmp_path)

    assert len(problems) == 1
    assert "master-standard phrase(s)" in problems[0]
    assert "PEP 8" in problems[0]
    assert "snake_case" in problems[0]
    assert "heuristic, not proof" in problems[0]


# --- numbered_docs discovery and end-to-end wiring ---


def test_numbered_docs_discovers_by_glob_not_hardcoded_list(tmp_path: Path) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "0_a.md").write_text("# 0. A\n", encoding="utf-8")
    (docs / "1_b.md").write_text("# 1. B\n", encoding="utf-8")
    (docs / "not_numbered.md").write_text("# Not numbered\n", encoding="utf-8")

    found = {path.name for path in numbered_docs(tmp_path)}

    assert found == {"0_a.md", "1_b.md"}


def test_collect_problems_on_fully_valid_tree_is_empty(tmp_path: Path) -> None:
    _write_valid_docs_tree(tmp_path)

    assert collect_problems(tmp_path) == []


def test_collect_problems_reports_every_broken_check(tmp_path: Path) -> None:
    _write_valid_docs_tree(tmp_path)
    (tmp_path / "CLAUDE.md").write_text("@AGENTS.md\nextra line\n", encoding="utf-8")

    problems = collect_problems(tmp_path)

    assert any("CLAUDE.md" in problem for problem in problems)


def test_real_repo_passes_all_checks() -> None:
    repo_root = Path(__file__).resolve().parent.parent

    assert collect_problems(repo_root) == []
