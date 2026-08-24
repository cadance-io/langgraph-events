"""Compile every ```python fence in the published docs.

A previous revision advertised ``NamespaceModel.mermaid(view="structure")`` in
three docs — the method never accepted a ``view`` argument, so users hit a
``TypeError`` on first call. Compilation alone catches syntax and
indentation errors; runtime mismatches still slip through, but this is a
cheap first net and keeps the cost linear in the number of fences.

Files are discovered by glob rather than listed by hand. An explicit list let
``docs/control-flow.md`` ship 8 unchecked fences for as long as nobody
remembered to extend it (issue #129); a glob means a new doc is covered the
moment it lands.

The bar is ``compile()``, not ``exec()``. Prose examples are deliberately
elliptical — ``...`` bodies and undefined names like ``upstream_rate_limited()``
are correct for a doc and must keep passing. Two accommodations follow from
treating fences as prose rather than modules:

* Fences nested in a MkDocs admonition (``!!! tip``) are indented by the
  Markdown, so the source is dedented before compiling. ``textwrap.dedent``
  strips only the *common* prefix, so a genuinely misindented fence still fails.
* Snippets are written REPL-style, with ``await`` / ``async for`` at top level
  rather than wrapped in a throwaway ``async def main()``. Compiling with
  ``PyCF_ALLOW_TOP_LEVEL_AWAIT`` applies the same bar an async REPL does, and
  narrows nothing else.
"""

from __future__ import annotations

import ast
import re
import textwrap
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
DOCS_ROOT = REPO_ROOT / "docs"
FENCE_RE = re.compile(r"```python\r?\n(.*?)```", re.DOTALL)

# Kept out of the built site by ``exclude_docs`` in mkdocs.yml. These are dated
# design specs — a historical record of a decision, not documentation users can
# follow — so the published site defines this test's scope. Freezing a snapshot
# dated 2026-07-27 into something a future compiler must keep accepting would
# falsify the record; "it no longer compiles" is the wrong reason to edit one.
EXCLUDED_DIRS = frozenset({"superpowers"})

# Floors, not exact counts: doc churn must not require touching this test, but a
# broken glob or a mangled FENCE_RE would drive both to zero and quietly turn
# every parametrised case into a no-op. That vacuity is the one failure mode the
# hand-written list at least made visible, so it is asserted explicitly below.
# Measured at the time of writing: 12 files, 71 fences.
MIN_DOC_FILES = 10
MIN_TOTAL_FENCES = 60


def _doc_files() -> list[str]:
    """Return repo-relative paths of every published Markdown doc."""
    return [
        path.relative_to(REPO_ROOT).as_posix()
        for path in sorted(DOCS_ROOT.rglob("*.md"))
        if EXCLUDED_DIRS.isdisjoint(path.relative_to(DOCS_ROOT).parts[:-1])
    ]


DOC_FILES = _doc_files()


def _fenced_python(md_path: Path) -> list[tuple[int, str]]:
    """Return ``(line_number, source)`` for each python fence in *md_path*."""
    text = md_path.read_text()
    out: list[tuple[int, str]] = []
    for match in FENCE_RE.finditer(text):
        line = text.count("\n", 0, match.start()) + 1
        out.append((line, textwrap.dedent(match.group(1))))
    return out


def describe_docs_code_fences():
    @pytest.mark.parametrize("md_relpath", DOC_FILES)
    def it_compiles_every_python_fence(md_relpath: str) -> None:
        # A doc with no python fences (patterns.md, checkpointer-evolution.md —
        # their autogen blocks are mermaid and text tabs) passes vacuously here;
        # the corpus guard below is what proves the suite is not vacuous overall.
        for line, source in _fenced_python(REPO_ROOT / md_relpath):
            compile(
                source,
                f"{md_relpath}:{line}",
                "exec",
                flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT,
            )

    def it_discovers_a_substantial_corpus_to_check():
        total = sum(len(_fenced_python(REPO_ROOT / rel)) for rel in DOC_FILES)
        assert len(DOC_FILES) >= MIN_DOC_FILES, (
            f"discovered only {len(DOC_FILES)} docs under {DOCS_ROOT}; "
            "the glob is broken and the fence check is running on almost nothing"
        )
        assert total >= MIN_TOTAL_FENCES, (
            f"found only {total} python fences across {len(DOC_FILES)} docs; "
            "FENCE_RE has likely stopped matching and every case above is a no-op"
        )
