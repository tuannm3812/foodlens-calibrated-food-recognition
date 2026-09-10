"""Regression tests for scripts/check_doc_links.py.

Codex's independent review of the S0 branch (docs/9_agent_log.md,
"2026-09-11 -- Codex review of Claude's S0 implementation") found the link
checker misread two standard Markdown forms: it read an optional link title
as part of the filename (false positive on valid CommonMark), and it never
checked reference-style links at all (false negative on broken ones). These
tests pin both fixes down, plus the existing behaviours the fix must not
disturb.

The module lives in scripts/, not a package, so it is loaded by file path
rather than imported normally.
"""

import importlib.util
from pathlib import Path

SCRIPT_PATH = Path(__file__).resolve().parent.parent / "scripts" / "check_doc_links.py"

_spec = importlib.util.spec_from_file_location("check_doc_links", SCRIPT_PATH)
assert _spec is not None and _spec.loader is not None
check_doc_links = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(check_doc_links)

broken_links = check_doc_links.broken_links


def test_titled_inline_link_to_existing_file_is_not_reported(tmp_path: Path) -> None:
    (tmp_path / "present.md").write_text("target\n", encoding="utf-8")
    doc = tmp_path / "doc.md"
    doc.write_text('[valid](present.md "Title")\n', encoding="utf-8")

    assert broken_links(doc) == []


def test_angle_bracket_destination_to_existing_file_is_not_reported(tmp_path: Path) -> None:
    (tmp_path / "present.md").write_text("target\n", encoding="utf-8")
    doc = tmp_path / "doc.md"
    doc.write_text("[a](<present.md>)\n", encoding="utf-8")

    assert broken_links(doc) == []


def test_reference_style_link_to_missing_file_is_reported(tmp_path: Path) -> None:
    doc = tmp_path / "doc.md"
    doc.write_text("[a][g]\n\n[g]: absent-xyz.md\n", encoding="utf-8")

    problems = broken_links(doc)

    assert len(problems) == 1
    assert "absent-xyz.md" in problems[0]


def test_reference_style_link_to_existing_file_is_not_reported(tmp_path: Path) -> None:
    (tmp_path / "present.md").write_text("target\n", encoding="utf-8")
    doc = tmp_path / "doc.md"
    doc.write_text("[a][g]\n\n[g]: present.md\n", encoding="utf-8")

    assert broken_links(doc) == []


def test_link_inside_fenced_block_is_not_reported(tmp_path: Path) -> None:
    doc = tmp_path / "doc.md"
    doc.write_text(
        "```markdown\n[a](absent-xyz.md)\n```\n",
        encoding="utf-8",
    )

    assert broken_links(doc) == []


def test_link_inside_inline_code_span_is_not_reported(tmp_path: Path) -> None:
    doc = tmp_path / "doc.md"
    doc.write_text("the link used to read `[a](absent-xyz.md)`\n", encoding="utf-8")

    assert broken_links(doc) == []


def test_unclosed_fence_fails_open_and_still_reports_broken_link(tmp_path: Path) -> None:
    doc = tmp_path / "doc.md"
    doc.write_text(
        "```markdown\n[a](absent-xyz.md)\n",
        encoding="utf-8",
    )

    assert broken_links(doc) == ["absent-xyz.md"]


def test_plain_broken_inline_link_is_reported(tmp_path: Path) -> None:
    doc = tmp_path / "doc.md"
    doc.write_text("[a](absent-xyz.md)\n", encoding="utf-8")

    assert broken_links(doc) == ["absent-xyz.md"]


def test_external_https_url_is_not_reported(tmp_path: Path) -> None:
    doc = tmp_path / "doc.md"
    doc.write_text("[a](https://example.com/absent-xyz)\n", encoding="utf-8")

    assert broken_links(doc) == []
