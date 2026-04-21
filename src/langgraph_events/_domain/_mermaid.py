"""Mermaid diagram rendering of a :class:`DomainModel`."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from langgraph_events._domain._model import (
    DomainModel,
    _event_label,
    _node_class,
)
from langgraph_events._event import (
    Command as CommandBase,
)
from langgraph_events._event import (
    DomainEvent,
    Event,
    Halted,
    HandlerRaised,
    IntegrationEvent,
    Interrupted,
    Resumed,
)
from langgraph_events._event import (
    Invariant as InvariantBase,
)
from langgraph_events._mermaid import MermaidFlowchart, Shape

# Order matters: most specific first (Command/DomainEvent before the
# SystemEvent subclasses, which share a common base).
_STEREOTYPE_BASES: tuple[tuple[type, str], ...] = (
    (CommandBase, "Command"),
    (DomainEvent, "DomainEvent"),
    (IntegrationEvent, "IntegrationEvent"),
    (Halted, "Halted"),
    (Interrupted, "Interrupted"),
    (Resumed, "Resumed"),
    (HandlerRaised, "HandlerRaised"),
)


def _event_stereotype(cls: type) -> str:
    """Short stereotype label for the structure ``classDiagram``.

    Aggregate membership doesn't affect the label — the stereotype reflects
    the event's category in the taxonomy (``DomainEvent``, ``Halted``, …),
    which matters for readers scanning the diagram.
    """
    for base, label in _STEREOTYPE_BASES:
        if issubclass(cls, base):
            return label
    return "SystemEvent"


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


def _add_node(flow: MermaidFlowchart, cls: type) -> None:
    """Declare an event class on the flowchart with its shape + class."""
    cls_key = _node_class(cls)
    flow.node(_event_label(cls), _NODE_SHAPE_BY_CLASS[cls_key], cls=cls_key)


def _add_invariant_node(flow: MermaidFlowchart, inv_cls: type) -> None:
    """Declare an Invariant class as a diamond gate node styled ``:::inv``."""
    flow.node(inv_cls.__name__, "diamond", cls="inv")


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
    if isinstance(r, DomainModel.CommandHandler):
        return r.commands
    return r.subscribes  # Policy


def render_mermaid_choreography(d: DomainModel) -> str:  # noqa: PLR0912, PLR0915
    """Emit a semantic ``graph LR`` flowchart of the event choreography.

    Visual vocabulary:
    - Commands render as hex ``{{…}}``, blue
    - DomainEvents render as rounded ``(…)``, green
    - IntegrationEvents render as parallelogram ``[/…/]``, violet
    - SystemEvents (Interrupted/Resumed/HandlerRaised) render as stadium
      ``([…])``, amber
    - Halted subtypes render as stadium, amber, with a dashed thick outline
    - Aggregate-owned nodes sit inside a ``subgraph`` titled "<Name> aggregate"
    - Solid ``-->`` arrows carry declared returns; ``raises=`` edges are
      thin dashed grey; ``Scatter[X]`` edges are thick dashed purple
    - Seed events (no incoming edges) keep the thick ``==>`` entry arrow
    """
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

    edges_by_reaction: dict[str, list[DomainModel.Edge]] = {}
    for e in d.edges:
        if e.kind == "framework":
            continue
        edges_by_reaction.setdefault(e.via, []).append(e)

    def _record(src_type: type[Event], tgt_type: type[Event] | None) -> tuple[str, str]:
        src_label = _event_label(src_type)
        referenced.add(src_type)
        if tgt_type is None:
            tgt_label = "?"
        else:
            tgt_label = _event_label(tgt_type)
            referenced.add(tgt_type)
        all_sources.add(src_label)
        all_targets.add(tgt_label)
        return src_label, tgt_label

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

        for e in solid_edges:
            src, tgt = _record(e.source, e.target)
            edges.append(_FlowEdge(src, tgt, "-->", name, "solid"))
        for e in scatter_edges:
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
    # declared outcomes always visibly connected to their command — even
    # when produced indirectly via a policy (e.g. Place owns Rejected,
    # which is only reached via InvariantViolated → explain_rejection).
    flow_pairs: set[tuple[type[Event], type[Event]]] = {
        (e.source, e.target) for e in d.edges if e.kind in ("solid", "scatter")
    }
    for agg in d.aggregates.values():
        for cmd in agg.commands.values():
            for outcome in cmd.outcomes:
                if (cmd.cls, outcome) in flow_pairs:
                    continue
                src, tgt = _record(cmd.cls, outcome)
                edges.append(_FlowEdge(src, tgt, "-.-", None, "ownership"))

    # Group referenced nodes by aggregate for subgraph wrapping.
    agg_members: dict[str, list[type[Event]]] = {}
    loose_nodes: list[type[Event]] = []
    for cls in referenced:
        agg_name = getattr(cls, "__aggregate__", None)
        if agg_name is not None:
            agg_members.setdefault(agg_name, []).append(cls)
        else:
            loose_nodes.append(cls)
    for members in agg_members.values():
        members.sort(key=_event_label)
    loose_nodes.sort(key=_event_label)

    # Place invariant gate nodes under the aggregate(s) of their commands.
    # If an invariant spans multiple aggregates, it stays loose (top-level).
    agg_invariants: dict[str, list[type[InvariantBase]]] = {}
    loose_invariants: list[type[InvariantBase]] = []
    invariant_edges: list[_FlowEdge] = []
    for inv in d.invariants:
        owning_aggs = {getattr(c, "__aggregate__", None) for c in inv.commands}
        owning_aggs.discard(None)
        if len(owning_aggs) == 1:
            agg_invariants.setdefault(next(iter(owning_aggs)), []).append(inv.cls)  # type: ignore[arg-type]
        else:
            loose_invariants.append(inv.cls)
        for cmd_cls in inv.commands:
            invariant_edges.append(
                _FlowEdge(
                    _event_label(cmd_cls),
                    inv.cls.__name__,
                    "-.->",
                    "invariant",
                    "invariant",
                )
            )
    for group in agg_invariants.values():
        group.sort(key=lambda c: c.__name__)
    loose_invariants.sort(key=lambda c: c.__name__)

    flow = MermaidFlowchart("LR")
    _apply_classdefs(flow)

    all_agg_names = sorted(set(agg_members) | set(agg_invariants))
    for agg_name in all_agg_names:
        with flow.subgraph(agg_name, title=f"{agg_name} aggregate", direction="LR"):
            for member in agg_members.get(agg_name, []):
                _add_node(flow, member)
            for inv_cls in agg_invariants.get(agg_name, []):
                _add_invariant_node(flow, inv_cls)

    for node in loose_nodes:
        _add_node(flow, node)
    for inv_cls in loose_invariants:
        _add_invariant_node(flow, inv_cls)

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
