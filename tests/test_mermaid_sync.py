"""Verify generated mermaid artifacts stay in sync with the examples, and
that the one hand-written diagram — the legend — numbers its own edges right.

The legend in ``scripts/generate_mermaid.py`` is the only mermaid in the
repo whose ``linkStyle`` indices are typed by hand rather than counted by
``MermaidFlowchart.render``. It is also the diagram readers consult to
decode every other one, so a mis-numbered directive silently teaches the
wrong colour vocabulary (#133).
"""

import importlib.util
import re
import subprocess
import sys
from pathlib import Path

from langgraph_events._namespace._mermaid import (
    _LINKSTYLE_CHAIN,
    _LINKSTYLE_INVARIANT,
    _LINKSTYLE_ORCHESTRATE,
    _LINKSTYLE_OWNS,
    _LINKSTYLE_RAISES,
    _LINKSTYLE_RETRY,
    _LINKSTYLE_SCATTER,
)

REPO_ROOT = Path(__file__).resolve().parents[1]

# Longest-first alternation: ``-.->`` must win over ``-.-``. The optional
# trailing group captures a ``|label|`` when the edge carries one.
_ARROW_RE = re.compile(r'(==>|-\.->|-\.-|-->)(?:\|("[^"]*"|[^|]*)\|)?')
_LINKSTYLE_RE = re.compile(r"^linkStyle ([\d,]+) (.+)$")
_NON_EDGE_PREFIXES = ("graph ", "classDef ", "subgraph ", "direction ", "end", "%%")

# What colour each legend edge claims, keyed by its label — sourced from the
# renderer's own palette so the legend can't drift from the diagrams it
# explains. Unlabelled edges fall back to their arrow: ``-.-`` is the
# ownership fill, ``==>`` the entry seed and ``-->`` a plain declared
# return, and those last two are deliberately unstyled.
_EXPECTED_BY_LABEL: dict[str, str] = {
    "(raises)": _LINKSTYLE_RAISES,
    "(retry)": _LINKSTYLE_RETRY,
    "scatter": _LINKSTYLE_SCATTER,
    "invariant": _LINKSTYLE_INVARIANT,
    "reactor": _LINKSTYLE_INVARIANT,
    "reactor [orchestrate]": _LINKSTYLE_ORCHESTRATE,
    "[chain]": _LINKSTYLE_CHAIN,
}
_EXPECTED_BY_ARROW: dict[str, str | None] = {"-.-": _LINKSTYLE_OWNS}


def _legend_mermaid_body() -> str:
    """The legend's mermaid source, imported from the generator script."""
    path = REPO_ROOT / "scripts" / "generate_mermaid.py"
    spec = importlib.util.spec_from_file_location("_generate_mermaid", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module._LEGEND_BLOCK.split("```mermaid")[1].split("```")[0]


def _declared_edges(body: str) -> list[tuple[str, str | None]]:
    """``(arrow, label)`` per edge, in the order mermaid numbers them.

    Chained declarations (``A -.->|x| B -.->|y| C``) contribute one edge
    each, which is exactly the counting subtlety the hand-numbering got
    wrong.
    """
    edges: list[tuple[str, str | None]] = []
    for line in body.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith(_NON_EDGE_PREFIXES):
            continue
        if _LINKSTYLE_RE.match(stripped):
            continue
        for match in _ARROW_RE.finditer(stripped):
            label = match.group(2)
            edges.append((match.group(1), label.strip('"') if label else None))
    return edges


def _applied_styles(body: str) -> dict[int, str]:
    """Style string per edge index, from the ``linkStyle`` directives."""
    applied: dict[int, str] = {}
    for line in body.splitlines():
        match = _LINKSTYLE_RE.match(line.strip())
        if match is None:
            continue
        for index in match.group(1).split(","):
            applied[int(index)] = match.group(2)
    return applied


def describe_mermaid_sync():
    def it_keeps_graph_files_current():
        result = subprocess.run(  # noqa: S603
            [sys.executable, "scripts/generate_mermaid.py", "--check"],
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, (
            f"Mermaid graph files are stale. "
            f"Run 'uv run python scripts/generate_mermaid.py' to update.\n"
            f"{result.stdout}{result.stderr}"
        )


def describe_legend_diagram():
    def it_styles_every_edge_the_colour_its_label_claims():
        body = _legend_mermaid_body()
        edges = _declared_edges(body)
        applied = _applied_styles(body)

        assert max(applied) < len(edges), (
            f"linkStyle {max(applied)} points past the last edge "
            f"(the legend declares {len(edges)}, indices 0-{len(edges) - 1})"
        )

        for index, (arrow, label) in enumerate(edges):
            if label is not None:
                assert label in _EXPECTED_BY_LABEL, (
                    f"legend edge {index} carries unknown label {label!r} — "
                    f"add it to _EXPECTED_BY_LABEL with its intended style"
                )
                expected = _EXPECTED_BY_LABEL[label]
            else:
                expected = _EXPECTED_BY_ARROW.get(arrow)
            assert applied.get(index) == expected, (
                f"legend edge {index} ({arrow} {label!r}) should be styled "
                f"{expected!r} but carries {applied.get(index)!r}"
            )
