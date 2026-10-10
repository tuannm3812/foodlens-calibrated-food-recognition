"""Verify every relative markdown link in tracked .md files resolves.

Run from the repo root. Exits non-zero and lists offenders when a link
points at a file that does not exist.

Fenced code blocks are skipped. Spec and plan documents quote example
markdown inside fences, and those examples describe files that may not
exist yet -- treating them as live links produces false positives.

Inline-code stripping fails open: it only ever removes a balanced span
that opens and closes on the same line, and any line with an unbalanced
or unclosed backtick run is left intact rather than guessed at. This
script is the CI gate for every future doc move, so a missed broken
link (false negative) is worse than a spurious one (false positive) --
when in doubt, the link gets checked.
"""

# This script is run directly from a shell, sometimes by whatever `python3`
# is on PATH rather than the project venv. Deferring annotation evaluation keeps
# it importable on interpreters older than the 3.11 the project targets, so the
# doc gate never fails for a reason unrelated to the docs.
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

LINK_PATTERN = re.compile(r"\[[^\]]*\]\(([^)]+)\)")
REFERENCE_DEFINITION_START_PATTERN = re.compile(r"^[ \t]{0,3}\[([^\]]+)\]:\s*(.+)$")
FENCE_PATTERN = re.compile(r"^\s*(`{3,}|~{3,})")
BACKTICK_RUN_PATTERN = re.compile(r"`+")
TITLE_SUFFIX_PATTERN = re.compile(r"^(\S+)\s+(?:\"[^\"]*\"|'[^']*'|\([^)]*\))\s*$")


def _parse_destination(raw: str) -> str:
    """Strip an optional link title from a raw inline-link destination.

    Handles three CommonMark destination forms: a bare path (`path`), an
    angle-bracket path (`<path>`), and a titled path where the title is a
    whitespace-separated suffix in double quotes, single quotes, or
    parentheses (`path "Title"`, `path 'Title'`, `path (Title)`).
    """
    raw = raw.strip()
    if raw.startswith("<"):
        end = raw.find(">")
        if end != -1:
            return raw[1:end].strip()
        return raw[1:].strip()
    match = TITLE_SUFFIX_PATTERN.match(raw)
    if match:
        return match.group(1)
    return raw


def _parse_reference_definition(line: str) -> tuple[str, str] | None:
    """Return (label, destination) if `line` is a link reference definition.

    A link reference definition is `[label]: destination` optionally followed
    by a title (`"Title"`, `'Title'`, or `(Title)`). Unlike an inline link
    destination, an unbracketed reference destination cannot contain spaces --
    so a prose line that merely starts with `[word]: ...` but whose remainder
    isn't a bare destination plus a validly-delimited title is not a
    definition at all, and must not be misread as one.
    """
    match = REFERENCE_DEFINITION_START_PATTERN.match(line)
    if not match:
        return None
    label, rest = match.group(1), match.group(2).strip()
    if not rest:
        return None
    if rest.startswith("<"):
        end = rest.find(">")
        if end == -1:
            return None
        destination = rest[1:end].strip()
        remainder = rest[end + 1 :].strip()
    else:
        parts = rest.split(None, 1)
        destination = parts[0]
        remainder = parts[1].strip() if len(parts) > 1 else ""
    if remainder and not (
        (remainder.startswith('"') and remainder.endswith('"'))
        or (remainder.startswith("'") and remainder.endswith("'"))
        or (remainder.startswith("(") and remainder.endswith(")"))
    ):
        return None
    return label, destination


def tracked_markdown_files() -> list[Path]:
    """Return every git-tracked .md file in the repo."""
    output = subprocess.run(
        ["git", "ls-files", "*.md"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return [Path(line) for line in output.splitlines() if line]


def strip_code_fences(text: str, label: str = "<text>") -> str:
    """Blank out fenced code blocks, preserving line numbering.

    A fence closes only on a marker at least as long as the one that
    opened it, so a ```bash block nested inside a ````markdown block does
    not close the outer fence.

    A fence that is never closed is a real false-negative risk: naively
    blanking to end-of-file would silently disable link checking for the
    rest of the document. To stay fail-open, an unclosed trailing fence is
    treated as NOT a fence at all -- the lines from its opening marker to
    end of file are returned intact so their links still get checked, and
    a warning naming `label` is printed to stderr so the malformed fence is
    visible rather than silently swallowed.
    """
    lines = text.splitlines()
    kept: list[str] = []
    fence: str | None = None
    open_index: int | None = None
    for index, line in enumerate(lines):
        match = FENCE_PATTERN.match(line)
        if fence is None:
            if match:
                fence = match.group(1)
                open_index = index
                kept.append("")
                continue
            kept.append(line)
        else:
            if match and match.group(1)[0] == fence[0] and len(match.group(1)) >= len(fence):
                fence = None
                open_index = None
            kept.append("")
    if fence is not None and open_index is not None:
        print(
            f"warning: {label}: unclosed code fence at line {open_index + 1}; "
            "checking its links anyway",
            file=sys.stderr,
        )
        kept[open_index:] = lines[open_index:]
    return "\n".join(kept)


def _strip_inline_code_from_line(line: str) -> str:
    """Blank out balanced inline-code spans on a single line.

    A link shown inside backticks, like `[label](target)`, is displayed to
    the reader as literal text, not asserted as a real target -- e.g. a
    plan document quoting "the link used to read `[a](b.md)`". Treating it
    as live produces false positives, so it must be stripped just like a
    fenced block. An opening run of N backticks closes at the next run of
    exactly N backticks on the same line, so multi-backtick delimiters
    (e.g. ``a `b` c``) are handled correctly.

    This never spans a newline, and it only strips a span once every
    opening run on this line has found a matching closing run. If any
    backtick run on the line can't be paired, the line is fail-open: it is
    returned unchanged so its links still get checked, rather than risking
    a guess that swallows real content.
    """
    runs = list(BACKTICK_RUN_PATTERN.finditer(line))
    if not runs:
        return line

    spans: list[tuple[int, int]] = []
    i = 0
    while i < len(runs):
        opener = runs[i]
        length = len(opener.group(0))
        closer_index = None
        for j in range(i + 1, len(runs)):
            if len(runs[j].group(0)) == length:
                closer_index = j
                break
        if closer_index is None:
            # Unbalanced/unclosed backtick run: fail open for this whole
            # line rather than guess which text was "meant" as code.
            return line
        spans.append((opener.start(), runs[closer_index].end()))
        i = closer_index + 1

    result = line
    for start, end in sorted(spans, reverse=True):
        result = result[:start] + result[end:]
    return result


def strip_inline_code(text: str) -> str:
    """Blank out balanced inline code spans, line by line.

    Operates after fence-stripping. Each line is handled independently so a
    span can never cross a newline; see `_strip_inline_code_from_line` for
    the fail-open pairing rule.
    """
    return "\n".join(_strip_inline_code_from_line(line) for line in text.splitlines())


def _is_checkable(target: str) -> bool:
    """Whether `target` (already anchor-stripped) should be resolved on disk."""
    return bool(target) and not target.startswith(("http://", "https://", "mailto:"))


def broken_links(path: Path) -> list[str]:
    """Return relative links in `path` that do not resolve to a real file.

    Covers both inline links (`[a](path)`, including the angle-bracket and
    titled destination forms) and reference-style links, whose destinations
    live on separate `[label]: destination` definition lines rather than at
    the link site itself.
    """
    problems = []
    body = strip_code_fences(path.read_text(encoding="utf-8"), label=str(path))
    body = strip_inline_code(body)

    for raw_target in LINK_PATTERN.findall(body):
        target = _parse_destination(raw_target).split("#", 1)[0].strip()
        if not _is_checkable(target):
            continue
        if not (path.parent / target).resolve().exists():
            problems.append(target)

    for line in body.splitlines():
        definition = _parse_reference_definition(line)
        if definition is None:
            continue
        label, raw_target = definition
        target = raw_target.split("#", 1)[0].strip()
        if not _is_checkable(target):
            continue
        if not (path.parent / target).resolve().exists():
            problems.append(f"{target} (reference [{label}])")

    return problems


def main() -> int:
    failures = 0
    for path in tracked_markdown_files():
        for target in broken_links(path):
            print(f"{path}: broken link -> {target}")
            failures += 1
    if failures:
        print(f"\n{failures} broken link(s).")
        return 1
    print("All relative markdown links resolve.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
