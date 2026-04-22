"""Internal mermaid flowchart builder.

Thin typed API for composing ``graph LR/TB`` diagrams with named shapes,
titled subgraphs, tagged edges, and ``linkStyle`` resolution by tag.
Nothing public — kept inside the package so ``_namespace/_model.py`` can
focus on the DDD namespace model and leave mermaid syntax concerns here.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from collections.abc import Iterator

Direction = Literal["LR", "RL", "TB", "BT", "TD"]

# Flowchart node shapes; keys are stable internal names, values are the
# (open, close) bracket pair mermaid expects around the label.
Shape = Literal[
    "rect",
    "rounded",
    "stadium",
    "subroutine",
    "cylinder",
    "circle",
    "diamond",
    "hex",
    "parallelogram",
]

_SHAPE_WRAP: dict[Shape, tuple[str, str]] = {
    "rect": ("[", "]"),
    "rounded": ("(", ")"),
    "stadium": ("([", "])"),
    "subroutine": ("[[", "]]"),
    "cylinder": ("[(", ")]"),
    "circle": ("((", "))"),
    "diamond": ("{", "}"),
    "hex": ("{{", "}}"),
    "parallelogram": ("[/", "/]"),
}

Arrow = Literal["-->", "-.->", "-.-", "==>", "---"]


def _quote(label: str) -> str:
    """Wrap *label* in double quotes when it contains mermaid-special chars.

    Parens, pipes, quotes, and ``#`` all trip the pipe-label parser; quoting
    sidesteps the issue. Plain labels are left alone so the common case
    stays readable.
    """
    if any(c in label for c in '()|"#'):
        return f'"{label}"'
    return label


# ---------------------------------------------------------------------------
# Statements — internal order-preserving record of what the caller added.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Classdef:
    name: str
    style: str


@dataclass(frozen=True)
class _Node:
    node_id: str
    shape: Shape
    cls: str | None
    label: str | None


@dataclass(frozen=True)
class _Edge:
    src: str
    tgt: str
    arrow: Arrow
    label: str | None
    tag: str | None


@dataclass(frozen=True)
class _SubgraphOpen:
    name: str
    title: str | None
    direction: Direction | None


@dataclass(frozen=True)
class _SubgraphClose:
    pass


@dataclass(frozen=True)
class _Comment:
    text: str


@dataclass(frozen=True)
class _LinkStyle:
    tag: str
    style: str


_Statement = _Classdef | _Node | _Edge | _SubgraphOpen | _SubgraphClose | _Comment


class MermaidFlowchart:
    """Fluent builder for ``graph LR/TB`` mermaid diagrams.

    Statements are appended in call order and rendered verbatim. ``linkStyle``
    lines are resolved at ``render()`` time by collecting every edge whose
    ``tag`` matches a registered style and emitting one ``linkStyle`` line
    per non-empty tag.
    """

    def __init__(self, direction: Direction = "LR") -> None:
        self._direction: Direction = direction
        self._classdefs: list[_Classdef] = []
        self._statements: list[_Statement] = []
        self._linkstyles: list[_LinkStyle] = []
        self._entry_count = 0

    # ---- declarations ------------------------------------------------------

    def classdef(self, name: str, style: str) -> MermaidFlowchart:
        """Register a ``classDef`` line. Emitted before any statement."""
        self._classdefs.append(_Classdef(name, style))
        return self

    def node(
        self,
        node_id: str,
        shape: Shape = "rect",
        *,
        cls: str | None = None,
        label: str | None = None,
    ) -> MermaidFlowchart:
        """Declare a node with a shape + optional class assignment.

        ``label`` defaults to ``node_id`` when omitted.
        """
        self._statements.append(_Node(node_id, shape, cls, label))
        return self

    def edge(
        self,
        src: str,
        tgt: str,
        *,
        arrow: Arrow = "-->",
        label: str | None = None,
        tag: str | None = None,
    ) -> MermaidFlowchart:
        """Emit an edge. ``tag`` is how ``link_style`` finds its targets."""
        self._statements.append(_Edge(src, tgt, arrow, label, tag))
        return self

    def comment(self, text: str) -> MermaidFlowchart:
        """Emit a ``%% text`` comment at 0 indent (mermaid's line syntax)."""
        self._statements.append(_Comment(text))
        return self

    # ---- higher-level affordances -----------------------------------------

    @contextmanager
    def subgraph(
        self,
        name: str,
        *,
        title: str | None = None,
        direction: Direction | None = None,
    ) -> Iterator[MermaidFlowchart]:
        """Open a subgraph block. ``title`` becomes ``name["title"]`` syntax;
        ``direction`` emits a ``direction DIR`` line as the first body line."""
        self._statements.append(_SubgraphOpen(name, title, direction))
        try:
            yield self
        finally:
            self._statements.append(_SubgraphClose())

    def entry_seed(self, target: str, *, cls: str = "entry") -> MermaidFlowchart:
        """Invisible ``_eN_`` seed node with a thick ``==>`` edge to *target*.

        ``N`` auto-increments. The emitted edge counts in the global edge
        index so ``linkStyle`` for tagged edges resolves correctly.
        """
        src = f"_e{self._entry_count}_[ ]:::{cls}"
        self._entry_count += 1
        return self.edge(src, target, arrow="==>")

    # ---- linkStyle registration -------------------------------------------

    def link_style(self, tag: str, style: str) -> MermaidFlowchart:
        """Register a ``linkStyle`` applied to every edge with this ``tag``.

        Resolved at ``render()`` time. If no edges carry the tag, the
        ``linkStyle`` line is omitted entirely.
        """
        self._linkstyles.append(_LinkStyle(tag, style))
        return self

    # ---- output -----------------------------------------------------------

    def render(self) -> str:
        lines: list[str] = [f"graph {self._direction}"]

        for cd in self._classdefs:
            lines.append(f"    classDef {cd.name} {cd.style}")

        edge_index = 0
        edges_by_tag: dict[str, list[int]] = {}
        depth = 0

        for stmt in self._statements:
            indent = "    " * (depth + 1)
            if isinstance(stmt, _SubgraphOpen):
                title = f'["{stmt.title}"]' if stmt.title is not None else ""
                lines.append(f"{indent}subgraph {stmt.name}{title}")
                depth += 1
                if stmt.direction is not None:
                    lines.append(f"{'    ' * (depth + 1)}direction {stmt.direction}")
            elif isinstance(stmt, _SubgraphClose):
                depth -= 1
                lines.append(f"{'    ' * (depth + 1)}end")
            elif isinstance(stmt, _Node):
                lines.append(_render_node(stmt, indent))
            elif isinstance(stmt, _Edge):
                lines.append(_render_edge(stmt, indent))
                if stmt.tag is not None:
                    edges_by_tag.setdefault(stmt.tag, []).append(edge_index)
                edge_index += 1
            elif isinstance(stmt, _Comment):
                lines.append(f"%% {stmt.text}")

        for ls in self._linkstyles:
            ids = edges_by_tag.get(ls.tag)
            if not ids:
                continue
            lines.append(f"    linkStyle {','.join(str(i) for i in ids)} {ls.style}")

        return "\n".join(lines)


def _render_node(n: _Node, indent: str) -> str:
    open_s, close_s = _SHAPE_WRAP[n.shape]
    label = n.label if n.label is not None else n.node_id
    cls_suffix = f":::{n.cls}" if n.cls else ""
    return f"{indent}{n.node_id}{open_s}{label}{close_s}{cls_suffix}"


def _render_edge(e: _Edge, indent: str) -> str:
    if e.label is None:
        return f"{indent}{e.src} {e.arrow} {e.tgt}"
    return f"{indent}{e.src} {e.arrow}|{_quote(e.label)}| {e.tgt}"
