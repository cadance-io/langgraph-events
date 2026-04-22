"""Compile every ```python fence in migrating.md and concepts.md.

A previous revision advertised ``NamespaceModel.mermaid(view="structure")`` in
three docs — the method never accepted a ``view`` argument, so users hit a
``TypeError`` on first call. Compilation alone catches syntax and
indentation errors; runtime mismatches still slip through, but this is a
cheap first net and keeps the cost linear in the number of fences.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
FENCE_RE = re.compile(r"```python\r?\n(.*?)```", re.DOTALL)


def _fenced_python(md_path: Path) -> list[tuple[int, str]]:
    """Return ``(line_number, source)`` for each python fence in *md_path*."""
    text = md_path.read_text()
    out: list[tuple[int, str]] = []
    for match in FENCE_RE.finditer(text):
        line = text.count("\n", 0, match.start()) + 1
        out.append((line, match.group(1)))
    return out


def describe_docs_code_fences():
    @pytest.mark.parametrize(
        "md_relpath",
        ["docs/migrating.md", "docs/concepts.md"],
    )
    def it_compiles_every_python_fence(md_relpath: str) -> None:
        md_path = REPO_ROOT / md_relpath
        fences = _fenced_python(md_path)
        assert fences, f"no python fences found in {md_relpath}"
        for line, source in fences:
            compile(source, f"{md_relpath}:{line}", "exec")
