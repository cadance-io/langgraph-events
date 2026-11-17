"""The part of a :class:`NamespaceModel` that one mermaid diagram draws."""

import difflib
from dataclasses import dataclass
from typing import TYPE_CHECKING

from langgraph_events._namespace._model import NamespaceModel, _reducer_node_id

if TYPE_CHECKING:
    from collections.abc import Iterable

    from langgraph_events._namespace._mermaid import _FlowEdge


def _check_focus_names(d: NamespaceModel, focus: NamespaceModel.Focus) -> None:
    """Raise ``ValueError`` for a focus name that the model does not hold."""
    known_by_kind = {
        "namespaces": set(d.namespaces),
        "reactions": {r.name for r in d.reactions},
        "reducers": {r.name for r in d.reducers} | {dv.name for dv in d.derived},
    }
    for kind, known in known_by_kind.items():
        for name in getattr(focus, kind):
            if name in known:
                continue
            near = difflib.get_close_matches(name, known, n=1)
            hint = f" Did you mean {near[0]!r}?" if near else ""
            raise ValueError(
                f"Focus names an unknown {kind[:-1]} {name!r}.{hint} "
                f"Valid {kind}: {', '.join(sorted(known)) or 'none'}."
            )


@dataclass(frozen=True)
class _View:
    """The node IDs that one diagram draws. ``drawn=None`` draws every node.

    Under a focus, ``selected`` holds the IDs the focus names, and every
    other drawn node is context.
    """

    drawn: frozenset[str] | None
    selected: frozenset[str]
    focus: NamespaceModel.Focus | None
    selected_namespaces: frozenset[str] = frozenset()

    def shows(self, node: str) -> bool:
        return self.drawn is None or node == "?" or node in self.drawn

    def is_context(self, node: str) -> bool:
        return self.focus is not None and node not in self.selected

    def title(self, namespace: str) -> str:
        if (
            self.focus is None
            or namespace in self.focus.namespaces
            or namespace in self.selected_namespaces
        ):
            return f"{namespace} namespace"
        return f"{namespace} namespace (context)"

    def lists(self, reaction: str, subscribes: Iterable[type]) -> bool:
        """Whether the side-effect footer lists *reaction*."""
        if self.focus is None or reaction in self.focus.reactions:
            return True
        return any(
            getattr(t, "__namespace__", None) in self.focus.namespaces
            for t in subscribes
        )


def _make_view(
    focus: NamespaceModel.Focus | None,
    show_raises: bool,
    edges: list[_FlowEdge],
    namespace_of: dict[str, str],
    standalone: set[str],
) -> _View:
    """Choose the drawn nodes: the ends of the kept edges, plus selected nodes.

    Under a focus, an edge is kept when one end is selected or when the
    focus names its reaction. Without a focus, only hidden ``raises``
    edges remove nodes, and ``standalone`` nodes stay drawn.
    """
    if focus is None:
        if show_raises:
            return _View(None, frozenset(), None)
        ends = {n for e in edges for n in (e.src, e.tgt)}
        return _View(frozenset(ends | standalone), frozenset(), None)
    selected = {n for n, ns in namespace_of.items() if ns in focus.namespaces}
    selected |= {_reducer_node_id(name) for name in focus.reducers}
    kept = [
        e
        for e in edges
        if e.src in selected or e.tgt in selected or e.via in focus.reactions
    ]
    ends = {n for e in kept for n in (e.src, e.tgt)}
    holding = frozenset(namespace_of[n] for n in selected if n in namespace_of)
    return _View(frozenset(ends | selected), frozenset(selected), focus, holding)
