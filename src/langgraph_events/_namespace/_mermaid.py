"""Mermaid diagram rendering of a :class:`NamespaceModel`."""

from __future__ import annotations

import difflib
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

from langgraph_events._mermaid import MermaidFlowchart, Shape
from langgraph_events._namespace._model import (
    NamespaceModel,
    _build_node_id_map,
    _event_label,
    _node_class,
)

if TYPE_CHECKING:
    from collections.abc import Iterable

    from langgraph_events._event import Event
    from langgraph_events._event import (
        Invariant as InvariantBase,
    )

# Shape per classDef key — used by the mermaid renderer to dispatch through
# the ``MermaidFlowchart`` builder. ``halt`` uses the stadium shape too;
# the dashed thick outline comes from its ``classDef`` stroke-width.
_NODE_SHAPE_BY_CLASS: dict[str, Shape] = {
    "cmd": "hex",  # imperative intent
    "devt": "rounded",  # domain fact
    "intg": "parallelogram",  # crosses a boundary
    "syst": "stadium",  # framework emitted
    "halt": "stadium",  # same shape as syst, dashed via classDef
}


def _add_node(
    flow: MermaidFlowchart,
    cls: type,
    node_id: dict[type, str],
    *,
    context: bool = False,
) -> None:
    """Declare an event class on the flowchart with its shape + class.

    ``node_id`` maps each class to its mermaid-safe ID; the rendered
    label stays as the short leaf name so the diagram stays readable
    regardless of whether the ID escalated to qualname form. A context
    node keeps its shape and takes the dimmed ``:::ctx`` class.
    """
    cls_key = _node_class(cls)
    flow.node(
        node_id[cls],
        _NODE_SHAPE_BY_CLASS[cls_key],
        cls="ctx" if context else cls_key,
        label=_event_label(cls),
    )


def _add_invariant_node(
    flow: MermaidFlowchart, inv_cls: type, node_id: dict[type, str]
) -> None:
    """Declare an Invariant class as a diamond gate node styled ``:::inv``."""
    flow.node(node_id[inv_cls], "diamond", cls="inv", label=inv_cls.__name__)


def _add_hub_node(flow: MermaidFlowchart, hub_id: str, handler_name: str) -> None:
    """Declare a reactor hub: small circle, ``:::hub`` styling.

    The handler name lives on the hub label rather than repeated on every
    fanout edge — see ``reactor_hub_min`` on ``NamespaceModel.mermaid``.
    """
    flow.node(hub_id, "circle", cls="hub", label=handler_name)


_HUB_CLASSDEF_STYLE = "fill:#f1f5f9,stroke:#64748b,color:#334155,stroke-dasharray:3 2"
_REDUCER_CLASSDEF_STYLE = "fill:#e0f2fe,stroke:#0369a1,color:#0c4a6e"


def _reducer_node_id(name: str) -> str:
    """Mermaid-safe node ID of a reducer, apart from every event node ID."""
    return "_reducer_" + re.sub(r"\W", "_", name)


def _add_reducer_node(
    flow: MermaidFlowchart, name: str, *, context: bool = False
) -> None:
    """Declare a reducer: cylinder, ``:::reducer`` styling."""
    flow.node(
        _reducer_node_id(name),
        "cylinder",
        cls="ctx" if context else "reducer",
        label=name,
    )


_CONTEXT_CLASSDEF_STYLE = "fill:none,stroke:#9ca3af,color:#6b7280,stroke-dasharray:3 3"


def _check_focus_names(d: NamespaceModel, focus: NamespaceModel.Focus) -> None:
    """Raise ``ValueError`` for a focus name that the model does not hold."""
    known_by_kind = {
        "namespaces": set(d.namespaces),
        "reactions": {r.name for r in d.reactions},
        "reducers": {r.name for r in d.reducers},
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

    def shows(self, node: str) -> bool:
        return self.drawn is None or node == "?" or node in self.drawn

    def is_context(self, node: str) -> bool:
        return self.focus is not None and node not in self.selected

    def title(self, namespace: str) -> str:
        if self.focus is None or namespace in self.focus.namespaces:
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
    return _View(frozenset(ends | selected), frozenset(selected), focus)


# Classdef palette used by both choreography and structure renderers.
# Order matters: `render()` preserves registration order, and we want a
# stable output for the drift detector.
_CLASSDEF_STYLES: dict[str, str] = {
    "entry": "fill:none,stroke:none,color:none",
    "cmd": "fill:#dbeafe,stroke:#1d4ed8,color:#1e3a8a",
    "devt": "fill:#dcfce7,stroke:#15803d,color:#14532d",
    "intg": "fill:#ede9fe,stroke:#6d28d9,color:#4c1d95",
    "syst": "fill:#fef3c7,stroke:#b45309,color:#78350f",
    "halt": (
        "fill:#fef3c7,stroke:#b45309,color:#78350f,"
        "stroke-width:3px,stroke-dasharray:4 2"
    ),
    "inv": "fill:#ffedd5,stroke:#c2410c,color:#7c2d12",
}

_LINKSTYLE_RAISES = "stroke:#6b7280,stroke-dasharray:3 3"
# Retry is a *transient* failure absorbed in place — cyan and finely
# dotted so it reads apart from the grey ``raises`` escalation arrow
# that only fires once the retry budget is spent.
_LINKSTYLE_RETRY = "stroke:#0891b2,stroke-dasharray:2 4"
_LINKSTYLE_SCATTER = "stroke:#7c3aed,stroke-width:2.5px,stroke-dasharray:8 3"
_LINKSTYLE_OWNS = "stroke:#9ca3af,stroke-dasharray:3 3"
_LINKSTYLE_INVARIANT = "stroke:#c2410c,stroke-dasharray:4 2"
# Orchestrate (reactor → Command, a saga move) is visually loud; chain
# (Command → Command) is deliberately awkward to discourage the pattern.
_LINKSTYLE_ORCHESTRATE = "stroke:#0369a1,stroke-width:3px"
_LINKSTYLE_CHAIN = "stroke:#b91c1c,stroke-width:2px,stroke-dasharray:5 3"
_LINKSTYLE_FOLDS = "stroke:#0369a1,stroke-dasharray:2 2"


def _causation_override(e: NamespaceModel.Edge) -> tuple[str | None, str]:
    """``(tag, label_suffix)`` for a notable causation; ``(None, "")`` else.

    Only ``orchestrate`` / ``chain`` deviate from the default solid/scatter
    style — ``intent`` / ``react`` are the healthy defaults and stay
    visually unchanged (zero snapshot churn).
    """
    if e.causation in ("orchestrate", "chain"):
        return e.causation, f" [{e.causation}]"
    return None, ""


def _apply_classdefs(flow: MermaidFlowchart) -> None:
    for name, style in _CLASSDEF_STYLES.items():
        flow.classdef(name, style)


@dataclass(frozen=True)
class _FlowEdge:
    """Internal record of one choreography edge to render."""

    src: str
    tgt: str
    arrow: str
    label: str | None
    tag: str | None
    via: str | None = None


def _reaction_subscribes(r: Any) -> tuple[type[Event], ...]:
    """Return the subscribed events for a CommandHandler or Policy."""
    if isinstance(r, NamespaceModel.CommandHandler):
        return r.commands
    return r.subscribes  # Policy


def _order_namespaces_by_affinity(
    namespaces: list[str],
    affinity: dict[frozenset[str], int],
) -> list[str]:
    """Greedy nearest-neighbor ordering of namespaces by inter-namespace edges.

    Picks the namespace with the highest total cross-traffic as the head,
    then repeatedly appends the unplaced namespace with highest affinity
    to the current tail. Ties break alphabetically — so disconnected
    namespaces (zero affinity to the tail) land alphabetically at the end.
    """
    if not namespaces:
        return []

    def aff(a: str, b: str) -> int:
        return affinity.get(frozenset([a, b]), 0)

    remaining = sorted(namespaces)

    def total(n: str) -> int:
        return sum(aff(n, m) for m in remaining if m != n)

    head = sorted(remaining, key=lambda n: (-total(n), n))[0]
    ordered = [head]
    pool = set(remaining) - {head}

    while pool:
        last = ordered[-1]
        nxt = sorted(pool, key=lambda n: (-aff(last, n), n))[0]
        ordered.append(nxt)
        pool.remove(nxt)

    return ordered


def render_mermaid_choreography(  # noqa: PLR0912, PLR0915
    d: NamespaceModel,
    *,
    namespace_order: Literal["affinity", "alphabetical"] = "affinity",
    reactor_hub_min: int | None = None,
    focus: NamespaceModel.Focus | None = None,
    show_raises: bool = True,
) -> str:
    """Emit a semantic ``graph LR`` flowchart of the event choreography.

    Visual vocabulary:
    - Commands render as hex ``{{…}}``, blue
    - DomainEvents render as rounded ``(…)``, green
    - IntegrationEvents render as parallelogram ``[/…/]``, violet
    - SystemEvents (Interrupted/Resumed/HandlerRaised) render as stadium
      ``([…])``, amber
    - Halted subtypes render as stadium, amber, with a dashed thick outline
    - Namespace-owned nodes sit inside a ``subgraph`` titled "<Name> namespace"
    - Solid ``-->`` arrows carry declared returns; ``raises=`` edges are
      thin dashed grey; ``retry=`` edges are finely dotted cyan;
      ``Scatter[X]`` edges are thick dashed purple
    - Invariants render as diamond ``:::inv`` gate nodes.  When a pinned
      reactor (``@on(InvariantViolated, invariant=Cls)``) exists, its
      output is routed *through* the Invariant diamond:
      ``Command -.->|invariant| Invariant -.->|reactor| Target``.  The
      ``InvariantViolated`` system-event node is hidden when every
      reactor is pinned (no catch-all ``@on(InvariantViolated)``).
    - Seed events (no incoming edges) keep the thick ``==>`` entry arrow
    """
    if focus is not None:
        _check_focus_names(d, focus)
    node_id = _build_node_id_map(d)
    edges: list[_FlowEdge] = []
    side_effect_entries: list[tuple[str, str, tuple[type[Event], ...]]] = []
    referenced: set[type[Event]] = set()
    all_sources: set[str] = set()
    all_targets: set[str] = set()

    reactions: list[tuple[str, Any]] = [
        *((r.name, r) for r in d.command_handlers),
        *((r.name, r) for r in d.policies),
    ]

    edges_by_reaction: dict[str, list[NamespaceModel.Edge]] = {}
    for e in d.edges:
        if e.kind == "framework":
            continue
        edges_by_reaction.setdefault(e.via, []).append(e)

    # Pinned-reactor routing: a reactor with @on(InvariantViolated,
    # invariant=Cls) has its output edge rerouted from InvariantViolated
    # → Target to Invariant(Cls) → Target.  The InvariantViolated node
    # then disappears from the diagram when every reactor is pinned.
    pinned_reactor_invariant: dict[str, type[InvariantBase]] = {}
    for inv in d.invariants:
        for reactor_name in inv.reactors:
            pinned_reactor_invariant[reactor_name] = inv.cls

    # Reactor-hub pre-computation.  When ``reactor_hub_min`` is set, any
    # ``(source, handler)`` pair producing ``≥ reactor_hub_min`` solid +
    # scatter targets gets a hub node — the handler name moves from being
    # repeated on every fanout edge to a single label on the hub, and the
    # source dispatches once into the hub before fanning out.  Invariant-
    # gated reactors are skipped (the invariant chain already concentrates).
    hubs: dict[tuple[type[Event], str], str] = {}
    hub_in_namespace: dict[str, list[tuple[str, str]]] = defaultdict(list)
    if reactor_hub_min is not None:
        pair_count: dict[tuple[type[Event], str], int] = defaultdict(int)
        for e in d.edges:
            if e.kind not in ("solid", "scatter"):
                continue
            if pinned_reactor_invariant.get(e.via) is not None:
                continue
            pair_count[(e.source, e.via)] += 1
        inline_via = {ch.name for ch in d.command_handlers if ch.inline}
        for (source_cls, handler_name), count in pair_count.items():
            if count < reactor_hub_min:
                continue
            src_id = node_id[source_cls]
            # An inline command's hub still earns its concentration (one rope
            # out of the command, then the fanout), but the handler name is
            # redundant noise and its positional suffix churns the node id.
            # Key the id on the command alone and render an anonymous dot;
            # reactor hubs keep their meaningful function-name label. See #108.
            if handler_name in inline_via:
                new_hub_id = f"_hub_{src_id}"
                hub_label = " "
            else:
                new_hub_id = f"_hub_{src_id}_{handler_name}"
                hub_label = handler_name
            hubs[(source_cls, handler_name)] = new_hub_id
            ns = getattr(source_cls, "__namespace__", None)
            if ns is not None:
                hub_in_namespace[ns].append((new_hub_id, hub_label))

    # Tracks hubs whose ``Source → Hub`` connector has already been emitted
    # — we emit one such edge per hub regardless of how many targets it has.
    # The connector itself is intentionally tag-less (no linkStyle): it's
    # structural, the visual identity of the dispatch lives on the hub label.
    # Per-target ``Hub → Target`` edges keep their original solid/scatter tag
    # so existing linkStyle rules continue to apply.
    emitted_hub_inbound: set[str] = set()

    def _record(src_type: type[Event], tgt_type: type[Event] | None) -> tuple[str, str]:
        src_id = node_id[src_type]
        referenced.add(src_type)
        if tgt_type is None:
            tgt_id = "?"
        else:
            tgt_id = node_id[tgt_type]
            referenced.add(tgt_type)
        all_sources.add(src_id)
        all_targets.add(tgt_id)
        return src_id, tgt_id

    # Edges routed via an Invariant gate instead of InvariantViolated.
    # Keyed by reactor name; appended to `edges` after the main reaction
    # loop so they group visually with the other invariant-tagged edges.
    rerouted_pinned_edges: list[_FlowEdge] = []

    # Outcomes reached via an Invariant → reactor chain.  Used by the
    # ownership-gap fill to suppress redundant `Command -.- Outcome`
    # arrows when the invariant chain already connects them.
    reached_via_invariant: set[tuple[type[Event], type[Event]]] = set()

    for name, r in reactions:
        subs = _reaction_subscribes(r)
        re_edges = edges_by_reaction.get(name, [])
        # Inline command handlers (Policies have no ``inline`` field): the
        # edge's source node already IS the command, so the handler display
        # name (``handle``, positional ``handle_2``, …) is elided from edge
        # labels — only causation tags remain (#107). External @on(Command)
        # handlers and reactors keep their meaningful function names.
        inline = getattr(r, "inline", False)

        # Framework-signal edges first, so a side-effect handler declaring
        # nothing but ``raises=`` or ``retry=`` still contributes a real edge
        # to the graph. Both draw the same way — a dashed arrow into the
        # framework event, tagged with its kind so link_style can colour it —
        # and both fan out per subscribed source in the model, all pointing at
        # one target with one label. So dedupe per (src, tgt) within a kind:
        # ``raises=(E1, E2)`` is one visual edge, not a stack of identical
        # arrows, and a retry policy with ``observe="emit"`` gives
        # ``HandlerRetried`` exactly one visible producer (#132).
        for kind in ("raises", "retry"):
            seen: set[tuple[str, str]] = set()
            tag = f"({kind})"
            for e in (x for x in re_edges if x.kind == kind):
                src, tgt = _record(e.source, e.target)
                if (src, tgt) in seen or (kind == "raises" and not show_raises):
                    continue
                seen.add((src, tgt))
                label: str | None = tag if inline else f"{name} {tag}"
                edges.append(_FlowEdge(src, tgt, "-.->", label, kind, name))

        solid_edges = [e for e in re_edges if e.kind == "solid"]
        scatter_edges = [e for e in re_edges if e.kind == "scatter"]

        has_annotation = getattr(r, "has_annotation", True)
        side_effect = getattr(r, "side_effect", False)

        if not solid_edges and not scatter_edges:
            if has_annotation and side_effect:
                subs_label = ", ".join(_event_label(t) for t in subs)
                # An inline side-effect handler's name is redundant with its
                # command (which is exactly its ``subs``) — list the command
                # alone; external reactors keep ``name (subscribed events)``.
                entry = subs_label if inline else f"{name} ({subs_label})"
                side_effect_entries.append((entry, name, subs))
                continue
            if not has_annotation:
                # Unannotated handler with no known target → show "?" target.
                for src_type in subs:
                    src, _tgt = _record(src_type, None)
                    label = None if inline else name
                    edges.append(_FlowEdge(src, "?", "-->", label, "solid", name))
                continue

        inv_cls = pinned_reactor_invariant.get(name)

        for e in solid_edges:
            if inv_cls is not None:
                # Reroute: drop the InvariantViolated → target edge; emit
                # Invariant → target instead.  Record the (command, target)
                # pair so ownership-gap fill doesn't draw a redundant arrow.
                referenced.add(e.target)
                tgt_id = node_id[e.target]
                all_targets.add(tgt_id)
                rerouted_pinned_edges.append(
                    _FlowEdge(node_id[inv_cls], tgt_id, "-.->", name, "invariant", name)
                )
                for inv in d.invariants:
                    if inv.cls is inv_cls:
                        for cmd_cls in inv.commands:
                            reached_via_invariant.add((cmd_cls, e.target))
                continue
            hub_id: str | None = hubs.get((e.source, name))
            if hub_id is not None:
                referenced.add(e.source)
                referenced.add(e.target)
                src_id = node_id[e.source]
                tgt_id = node_id[e.target]
                all_sources.add(src_id)
                all_targets.add(tgt_id)
                if hub_id not in emitted_hub_inbound:
                    edges.append(_FlowEdge(src_id, hub_id, "-->", None, None, name))
                    emitted_hub_inbound.add(hub_id)
                edges.append(_FlowEdge(hub_id, tgt_id, "-->", None, "solid", name))
                continue
            src, tgt = _record(e.source, e.target)
            ctag, csuf = _causation_override(e)
            label = (csuf.lstrip() or None) if inline else f"{name}{csuf}"
            edges.append(_FlowEdge(src, tgt, "-->", label, ctag or "solid", name))
        for e in scatter_edges:
            if inv_cls is not None:
                referenced.add(e.target)
                tgt_id = node_id[e.target]
                all_targets.add(tgt_id)
                rerouted_pinned_edges.append(
                    _FlowEdge(node_id[inv_cls], tgt_id, "-.->", name, "invariant", name)
                )
                for inv in d.invariants:
                    if inv.cls is inv_cls:
                        for cmd_cls in inv.commands:
                            reached_via_invariant.add((cmd_cls, e.target))
                continue
            hub_id = hubs.get((e.source, name))
            if hub_id is not None:
                referenced.add(e.source)
                referenced.add(e.target)
                src_id = node_id[e.source]
                tgt_id = node_id[e.target]
                all_sources.add(src_id)
                all_targets.add(tgt_id)
                if hub_id not in emitted_hub_inbound:
                    edges.append(_FlowEdge(src_id, hub_id, "-->", None, None, name))
                    emitted_hub_inbound.add(hub_id)
                edges.append(_FlowEdge(hub_id, tgt_id, "-.->", None, "scatter", name))
                continue
            src, tgt = _record(e.source, e.target)
            ctag, csuf = _causation_override(e)
            label = (csuf.lstrip() or None) if inline else f"{name}{csuf}"
            edges.append(_FlowEdge(src, tgt, "-.->", label, ctag or "scatter", name))

    # Framework Interrupted → Resumed edge.
    for e in (x for x in d.edges if x.kind == "framework"):
        src, tgt = _record(e.source, e.target)
        edges.append(_FlowEdge(src, tgt, "-.->", None, "framework"))
        # Framework edge's source should not be treated as a seed.
        all_targets.add(src)

    # Ownership-gap fill: for every (command → declared outcome) pair
    # without a direct flow edge, emit a dashed "owns" arrow. Makes
    # declared outcomes always visibly connected to their command.
    # Skip pairs already reached via an invariant chain — the pinned
    # reactor edge Command -> Invariant -> outcome covers it.
    flow_pairs: set[tuple[type[Event], type[Event]]] = {
        (e.source, e.target) for e in d.edges if e.kind in ("solid", "scatter")
    } | reached_via_invariant
    for dom in d.namespaces.values():
        for cmd in dom.commands.values():
            for outcome in cmd.outcomes:
                if (cmd.cls, outcome) in flow_pairs:
                    continue
                src, tgt = _record(cmd.cls, outcome)
                edges.append(_FlowEdge(src, tgt, "-.-", None, "ownership"))

    # A reducer reads events and emits none, so each subscribed event gets a
    # dotted ``folds`` arrow into the reducer node.
    for red in d.reducers:
        red_id = _reducer_node_id(red.name)
        all_targets.add(red_id)
        for cls in red.subscribes:
            referenced.add(cls)
            all_sources.add(node_id[cls])
            edges.append(_FlowEdge(node_id[cls], red_id, "-.->", "folds", "folds"))

    # Append rerouted pinned-reactor edges to the flow edge list.  These
    # use the "invariant" tag so they get the same dashed-orange style as
    # the Command -> Invariant gate edges.
    edges.extend(rerouted_pinned_edges)

    # Group referenced nodes by domain for subgraph wrapping.
    domain_members: dict[str, list[type[Event]]] = {}
    loose_nodes: list[type[Event]] = []
    for cls in referenced:
        namespace_name = getattr(cls, "__namespace__", None)
        if namespace_name is not None:
            domain_members.setdefault(namespace_name, []).append(cls)
        else:
            loose_nodes.append(cls)
    for members in domain_members.values():
        members.sort(key=lambda c: node_id[c])
    loose_nodes.sort(key=lambda c: node_id[c])

    # Place invariant gate nodes under the domain(s) of their commands.
    # If an invariant spans multiple domains, it stays loose (top-level).
    namespace_invariants: dict[str, list[type[InvariantBase]]] = {}
    loose_invariants: list[type[InvariantBase]] = []
    invariant_edges: list[_FlowEdge] = []
    for inv in d.invariants:
        owning_domains = {getattr(c, "__namespace__", None) for c in inv.commands}
        owning_domains.discard(None)
        if len(owning_domains) == 1:
            ns = next(iter(owning_domains))
            namespace_invariants.setdefault(ns, []).append(inv.cls)  # type: ignore[arg-type]
        else:
            loose_invariants.append(inv.cls)
        for cmd_cls in inv.commands:
            invariant_edges.append(
                _FlowEdge(
                    node_id[cmd_cls],
                    node_id[inv.cls],
                    "-.->",
                    "invariant",
                    "invariant",
                )
            )
    for group in namespace_invariants.values():
        group.sort(key=lambda c: c.__name__)
    loose_invariants.sort(key=lambda c: c.__name__)

    namespace_reducers: dict[str, list[str]] = defaultdict(list)
    loose_reducers: list[str] = []
    for red in d.reducers:
        if red.namespace is None:
            loose_reducers.append(red.name)
        else:
            namespace_reducers[red.namespace].append(red.name)

    namespace_of: dict[str, str] = {
        node_id[cls]: ns for ns, members in domain_members.items() for cls in members
    }
    namespace_of |= {
        node_id[inv]: ns for ns, invs in namespace_invariants.items() for inv in invs
    }
    namespace_of |= {
        hub: ns for ns, hubs_here in hub_in_namespace.items() for hub, _ in hubs_here
    }
    namespace_of |= {
        _reducer_node_id(name): ns
        for ns, names in namespace_reducers.items()
        for name in names
    }
    view = _make_view(
        focus,
        show_raises,
        [*edges, *invariant_edges],
        namespace_of,
        {node_id[inv.cls] for inv in d.invariants}
        | {_reducer_node_id(r.name) for r in d.reducers},
    )
    edges = [e for e in edges if view.shows(e.src) and view.shows(e.tgt)]
    invariant_edges = [
        e for e in invariant_edges if view.shows(e.src) and view.shows(e.tgt)
    ]
    for members in domain_members.values():
        members[:] = [m for m in members if view.shows(node_id[m])]
    loose_nodes = [n for n in loose_nodes if view.shows(node_id[n])]
    for invs in namespace_invariants.values():
        invs[:] = [i for i in invs if view.shows(node_id[i])]
    loose_invariants = [i for i in loose_invariants if view.shows(node_id[i])]
    for hubs_here in hub_in_namespace.values():
        hubs_here[:] = [h for h in hubs_here if view.shows(h[0])]
    for names in namespace_reducers.values():
        names[:] = [n for n in names if view.shows(_reducer_node_id(n))]
    loose_reducers = [n for n in loose_reducers if view.shows(_reducer_node_id(n))]

    flow = MermaidFlowchart("LR")
    _apply_classdefs(flow)
    if hubs:
        flow.classdef("hub", _HUB_CLASSDEF_STYLE)
    if d.reducers:
        flow.classdef("reducer", _REDUCER_CLASSDEF_STYLE)
    if focus is not None:
        flow.classdef("ctx", _CONTEXT_CLASSDEF_STYLE)

    all_domain_names = sorted(
        ns
        for ns in set(domain_members)
        | set(namespace_invariants)
        | set(namespace_reducers)
        if domain_members.get(ns)
        or namespace_invariants.get(ns)
        or namespace_reducers.get(ns)
    )
    if namespace_order == "affinity":
        # Affinity counts: solid + scatter reaction edges plus invariant
        # chain edges (Command → Invariant), and nothing else. Ownership-fill
        # ``-.- `` arrows and framework-signal (``raises`` / ``retry``) edges
        # are rendering scaffolding rather than real flow, so they don't
        # contribute — an allowlist keeps a future ``kind`` out by default.
        # The framework ``Interrupted → Resumed`` edge carries
        # ``__namespace__ = None`` on both endpoints and is filtered by the
        # cross-namespace guard below.
        affinity: dict[frozenset[str], int] = defaultdict(int)

        def _bump(a: type, b: type) -> None:
            ns_a = getattr(a, "__namespace__", None)
            ns_b = getattr(b, "__namespace__", None)
            if ns_a is None or ns_b is None or ns_a == ns_b:
                return
            if not (view.shows(node_id[a]) and view.shows(node_id[b])):
                return
            affinity[frozenset([ns_a, ns_b])] += 1

        for e in d.edges:
            if e.kind not in ("solid", "scatter"):
                continue
            _bump(e.source, e.target)
        for inv in d.invariants:
            for cmd_cls in inv.commands:
                _bump(cmd_cls, inv.cls)

        all_domain_names = _order_namespaces_by_affinity(all_domain_names, affinity)

    for namespace_name in all_domain_names:
        title = view.title(namespace_name)
        with flow.subgraph(namespace_name, title=title, direction="LR"):
            for member in domain_members.get(namespace_name, []):
                context = view.is_context(node_id[member])
                _add_node(flow, member, node_id, context=context)
            for inv_cls in namespace_invariants.get(namespace_name, []):
                _add_invariant_node(flow, inv_cls, node_id)
            for hub_id, handler_name in hub_in_namespace.get(namespace_name, []):
                _add_hub_node(flow, hub_id, handler_name)
            for reducer_name in namespace_reducers.get(namespace_name, []):
                context = view.is_context(_reducer_node_id(reducer_name))
                _add_reducer_node(flow, reducer_name, context=context)

    for node in loose_nodes:
        _add_node(flow, node, node_id, context=view.is_context(node_id[node]))
    for inv_cls in loose_invariants:
        _add_invariant_node(flow, inv_cls, node_id)
    for reducer_name in loose_reducers:
        context = view.is_context(_reducer_node_id(reducer_name))
        _add_reducer_node(flow, reducer_name, context=context)

    # Seeds come from every edge, drawn or hidden: an event that a hidden
    # edge reaches is not an entry point of the system.
    for seed in sorted(all_sources - all_targets):
        if view.shows(seed):
            flow.entry_seed(seed)

    for ed in edges:
        flow.edge(ed.src, ed.tgt, arrow=ed.arrow, label=ed.label, tag=ed.tag)  # type: ignore[arg-type]
    for ed in invariant_edges:
        flow.edge(ed.src, ed.tgt, arrow=ed.arrow, label=ed.label, tag=ed.tag)  # type: ignore[arg-type]

    flow.link_style("scatter", _LINKSTYLE_SCATTER)
    flow.link_style("raises", _LINKSTYLE_RAISES)
    flow.link_style("retry", _LINKSTYLE_RETRY)
    flow.link_style("ownership", _LINKSTYLE_OWNS)
    flow.link_style("invariant", _LINKSTYLE_INVARIANT)
    flow.link_style("orchestrate", _LINKSTYLE_ORCHESTRATE)
    flow.link_style("chain", _LINKSTYLE_CHAIN)
    flow.link_style("folds", _LINKSTYLE_FOLDS)

    listed = [
        entry for entry, name, subs in side_effect_entries if view.lists(name, subs)
    ]
    if listed:
        flow.comment(f"Side-effect handlers: {', '.join(listed)}")

    return flow.render()
