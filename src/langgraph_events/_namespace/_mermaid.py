"""Mermaid diagram rendering of a :class:`NamespaceModel`."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from langgraph_events._mermaid import MermaidFlowchart, Shape
from langgraph_events._namespace._model import (
    NamespaceModel,
    _build_node_id_map,
    _event_label,
    _node_class,
)

if TYPE_CHECKING:
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


def _add_node(flow: MermaidFlowchart, cls: type, node_id: dict[type, str]) -> None:
    """Declare an event class on the flowchart with its shape + class.

    ``node_id`` maps each class to its mermaid-safe ID; the rendered
    label stays as the short leaf name so the diagram stays readable
    regardless of whether the ID escalated to qualname form.
    """
    cls_key = _node_class(cls)
    flow.node(
        node_id[cls],
        _NODE_SHAPE_BY_CLASS[cls_key],
        cls=cls_key,
        label=_event_label(cls),
    )


def _add_invariant_node(
    flow: MermaidFlowchart, inv_cls: type, node_id: dict[type, str]
) -> None:
    """Declare an Invariant class as a diamond gate node styled ``:::inv``."""
    flow.node(node_id[inv_cls], "diamond", cls="inv", label=inv_cls.__name__)


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
_LINKSTYLE_SCATTER = "stroke:#7c3aed,stroke-width:2.5px,stroke-dasharray:8 3"
_LINKSTYLE_OWNS = "stroke:#9ca3af,stroke-dasharray:3 3"
_LINKSTYLE_INVARIANT = "stroke:#c2410c,stroke-dasharray:4 2"


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


def _reaction_subscribes(r: Any) -> tuple[type[Event], ...]:
    """Return the subscribed events for a CommandHandler or Policy."""
    if isinstance(r, NamespaceModel.CommandHandler):
        return r.commands
    return r.subscribes  # Policy


def render_mermaid_choreography(d: NamespaceModel) -> str:  # noqa: PLR0912, PLR0915
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
      thin dashed grey; ``Scatter[X]`` edges are thick dashed purple
    - Invariants render as diamond ``:::inv`` gate nodes.  When a pinned
      reactor (``@on(InvariantViolated, invariant=Cls)``) exists, its
      output is routed *through* the Invariant diamond:
      ``Command -.->|invariant| Invariant -.->|reactor| Target``.  The
      ``InvariantViolated`` system-event node is hidden when every
      reactor is pinned (no catch-all ``@on(InvariantViolated)``).
    - Seed events (no incoming edges) keep the thick ``==>`` entry arrow
    """
    node_id = _build_node_id_map(d)
    edges: list[_FlowEdge] = []
    side_effect_entries: list[str] = []
    scatter_entries: list[str] = []
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

        # Raises edges first so a side-effect handler with only a raises
        # declaration still contributes a real edge to the graph.
        for e in (x for x in re_edges if x.kind == "raises"):
            src, tgt = _record(e.source, e.target)
            edges.append(_FlowEdge(src, tgt, "-.->", f"{name} (raises)", "raises"))

        solid_edges = [e for e in re_edges if e.kind == "solid"]
        scatter_edges = [e for e in re_edges if e.kind == "scatter"]

        has_annotation = getattr(r, "has_annotation", True)
        has_untyped_scatter = getattr(r, "has_untyped_scatter", False)
        side_effect = getattr(r, "side_effect", False)

        if not solid_edges and not scatter_edges:
            if has_annotation and not has_untyped_scatter and side_effect:
                subs_label = ", ".join(_event_label(t) for t in subs)
                side_effect_entries.append(f"{name} ({subs_label})")
                continue
            if has_untyped_scatter:
                subs_label = ", ".join(_event_label(t) for t in subs)
                scatter_entries.append(f"{name} ({subs_label})")
                continue
            if not has_annotation:
                # Unannotated handler with no known target → show "?" target.
                for src_type in subs:
                    src, _tgt = _record(src_type, None)
                    edges.append(_FlowEdge(src, "?", "-->", name, "solid"))
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
                    _FlowEdge(node_id[inv_cls], tgt_id, "-.->", name, "invariant")
                )
                for inv in d.invariants:
                    if inv.cls is inv_cls:
                        for cmd_cls in inv.commands:
                            reached_via_invariant.add((cmd_cls, e.target))
                continue
            src, tgt = _record(e.source, e.target)
            edges.append(_FlowEdge(src, tgt, "-->", name, "solid"))
        for e in scatter_edges:
            if inv_cls is not None:
                referenced.add(e.target)
                tgt_id = node_id[e.target]
                all_targets.add(tgt_id)
                rerouted_pinned_edges.append(
                    _FlowEdge(node_id[inv_cls], tgt_id, "-.->", name, "invariant")
                )
                for inv in d.invariants:
                    if inv.cls is inv_cls:
                        for cmd_cls in inv.commands:
                            reached_via_invariant.add((cmd_cls, e.target))
                continue
            src, tgt = _record(e.source, e.target)
            edges.append(_FlowEdge(src, tgt, "-.->", name, "scatter"))

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

    flow = MermaidFlowchart("LR")
    _apply_classdefs(flow)

    all_domain_names = sorted(set(domain_members) | set(namespace_invariants))
    for namespace_name in all_domain_names:
        title = f"{namespace_name} namespace"
        with flow.subgraph(namespace_name, title=title, direction="LR"):
            for member in domain_members.get(namespace_name, []):
                _add_node(flow, member, node_id)
            for inv_cls in namespace_invariants.get(namespace_name, []):
                _add_invariant_node(flow, inv_cls, node_id)

    for node in loose_nodes:
        _add_node(flow, node, node_id)
    for inv_cls in loose_invariants:
        _add_invariant_node(flow, inv_cls, node_id)

    for seed in sorted(all_sources - all_targets):
        flow.entry_seed(seed)

    for ed in edges:
        flow.edge(ed.src, ed.tgt, arrow=ed.arrow, label=ed.label, tag=ed.tag)  # type: ignore[arg-type]
    for ed in invariant_edges:
        flow.edge(ed.src, ed.tgt, arrow=ed.arrow, label=ed.label, tag=ed.tag)  # type: ignore[arg-type]

    flow.link_style("scatter", _LINKSTYLE_SCATTER)
    flow.link_style("raises", _LINKSTYLE_RAISES)
    flow.link_style("ownership", _LINKSTYLE_OWNS)
    flow.link_style("invariant", _LINKSTYLE_INVARIANT)

    if scatter_entries:
        flow.comment(f"Scatter handlers: {', '.join(scatter_entries)}")
    if side_effect_entries:
        flow.comment(f"Side-effect handlers: {', '.join(side_effect_entries)}")

    return flow.render()
