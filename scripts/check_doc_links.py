"""Verify every relative markdown link in tracked .md files resolves.

Run from the repo root. Exits non-zero and lists offenders when a link
points at a file that does not exist.

Fenced code blocks are skipped. Spec and plan documents quote example
markdown inside fences, and those examples describe files that may not
exist yet -- treating them as live links produces false positives.
"""

import re
import subprocess
import sys
from pathlib import Path

LINK_PATTERN = re.compile(r"\[[^\]]*\]\(([^)]+)\)")
FENCE_PATTERN = re.compile(r"^\s*(`{3,}|~{3,})")


def tracked_markdown_files() -> list[Path]:
    """Return every git-tracked .md file in the repo."""
    output = subprocess.run(
        ["git", "ls-files", "*.md"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return [Path(line) for line in output.splitlines() if line]


def strip_code_fences(text: str) -> str:
    """Blank out fenced code blocks, preserving line numbering.

    A fence closes only on a marker at least as long as the one that
    opened it, so a ```bash block nested inside a ````markdown block does
    not close the outer fence.
    """
    lines = text.splitlines()
    kept: list[str] = []
    fence: str | None = None
    for line in lines:
        match = FENCE_PATTERN.match(line)
        if fence is None:
            if match:
                fence = match.group(1)
                kept.append("")
                continue
            kept.append(line)
        else:
            if match and match.group(1)[0] == fence[0] and len(match.group(1)) >= len(fence):
                fence = None
            kept.append("")
    return "\n".join(kept)


def broken_links(path: Path) -> list[str]:
    """Return relative links in `path` that do not resolve to a real file."""
    problems = []
    body = strip_code_fences(path.read_text(encoding="utf-8"))
    for target in LINK_PATTERN.findall(body):
        target = target.split("#", 1)[0].strip()
        if not target or target.startswith(("http://", "https://", "mailto:")):
            continue
        if not (path.parent / target).resolve().exists():
            problems.append(target)
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
