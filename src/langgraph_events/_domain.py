"""Domain model introspection — a code-derived DDD snapshot of an EventGraph.

``EventGraph.domain()`` returns a :class:`DomainModel` instance. Everything
else — human-readable text, Mermaid diagrams, JSON export — hangs off that
single object. See the module docstring on :class:`DomainModel` for the full
surface.
"""

from __future__ import annotations

import json as _json
import typing
from dataclasses import dataclass
from dataclasses import fields as dc_fields
from typing import TYPE_CHECKING, Any, Literal

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
    InvariantViolated,
    Resumed,
    SystemEvent,
)
from langgraph_events._event import (
    Invariant as InvariantBase,
)
from langgraph_events._mermaid import MermaidFlowchart, Shape

if TYPE_CHECKING:
    from langgraph_events._graph import ReturnInfo
    from langgraph_events._handler import HandlerMeta


View = Literal["structure", "choreography"]


def _event_label(cls: type) -> str:
    """Short display name for event types in text and choreography diagrams.

    Uses ``cls.__name__`` to match the current mermaid() / describe() output
    and to stay readable in flow diagrams where node identity is already
    anchored by the edge label (the handler name).
    """
    return cls.__name__


def _event_node_id(cls: type) -> str:
    """Mermaid-safe node identifier — qualname-based for structure diagrams
    so nested events like ``Order.Place.Placed`` don't collide with a sibling
    aggregate's similarly-named class.
    """
    return cls.__qualname__.replace("<locals>.", "").replace(".", "_")


def _matcher_repr(matcher: Any, is_type: bool) -> str:
    if is_type and isinstance(matcher, type):
        return matcher.__name__
    return repr(matcher)


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


# Short classDef key for the choreography flowchart. Drives both the node
# shape and the fill/stroke palette. Halted nested under an aggregate gets
# the same amber fill as SystemEvent but a dashed double-outline stroke so
# terminals are visually distinct without stealing red from error flow.
_NODE_CLASS_BASES: tuple[tuple[type, str], ...] = (
    (CommandBase, "cmd"),
    (DomainEvent, "devt"),
    (IntegrationEvent, "intg"),
    (Halted, "halt"),
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


def _node_class(cls: type) -> str:
    """Return the classDef key (``cmd``/``devt``/``intg``/``syst``/``halt``)."""
    for base, cls_key in _NODE_CLASS_BASES:
        if issubclass(cls, base):
            return cls_key
    return "syst"


def _add_node(flow: MermaidFlowchart, cls: type) -> None:
    """Declare an event class on the flowchart with its shape + class."""
    cls_key = _node_class(cls)
    flow.node(_event_label(cls), _NODE_SHAPE_BY_CLASS[cls_key], cls=cls_key)


def _add_invariant_node(flow: MermaidFlowchart, inv_cls: type) -> None:
    """Declare an Invariant class as a diamond gate node styled ``:::inv``."""
    flow.node(inv_cls.__name__, "diamond", cls="inv")


@dataclass(frozen=True)
class DomainModel:
    """Code-derived snapshot of an ``EventGraph``'s domain.

    Two lenses:

    - ``view="structure"`` — the **taxonomy** (aggregates → commands → outcomes,
      free-standing DomainEvents, integration and system events). No handlers,
      no edges.
    - ``view="choreography"`` — the **full picture**: taxonomy + command
      handlers + policies + event-to-event edges + seed events. Default.

    "Choreography" is the DDD/event-storming term for event-driven flow
    without a central orchestrator — exactly what this library models.

    Rendering::

        d = graph.domain()
        d.text()                      # human-readable tree
        d.mermaid()                   # default diagram
        d.mermaid(view="structure")   # classDiagram
        d.json()                      # serializable snapshot

    Data access::

        d.aggregates           # dict[str, DomainModel.Aggregate]
        d.command_handlers     # tuple[DomainModel.CommandHandler, ...]
        d.policies             # tuple[DomainModel.Policy, ...]
        d.reactions            # command_handlers + policies
        d.edges                # tuple[DomainModel.Edge, ...]
        d.seeds                # tuple[type[Event], ...]
        d.integration_events   # tuple[type[IntegrationEvent], ...]
        d.system_events        # tuple[type[SystemEvent], ...]
    """

    # ---- nested types (class-level, not fields) ----

    @dataclass(frozen=True)
    class Aggregate:
        """An aggregate root — the grouping for its nested commands and events.

        ``events`` holds any ``Event`` nested in the aggregate that is not an
        outcome of one of its commands — typically free-standing DomainEvents,
        but also ``Halted`` subtypes nested in the aggregate for locality.
        """

        name: str
        commands: dict[str, DomainModel.Command]
        events: tuple[type[Event], ...]

    @dataclass(frozen=True)
    class Command:
        """A command class, plus its outcomes and the handlers that execute it."""

        cls: type[CommandBase]
        outcomes: tuple[type[DomainEvent], ...]
        handlers: tuple[str, ...]

    @dataclass(frozen=True)
    class CommandHandler:
        """A handler that executes a command.

        A handler is classified as a ``CommandHandler`` if any of its
        subscribed event types is a ``Command`` subclass. See the module
        docstring for mixed-subscription semantics.
        """

        name: str
        commands: tuple[type[CommandBase], ...]
        produces: tuple[type[Event], ...]
        scatters: tuple[type[Event], ...]
        raises: tuple[type[Exception], ...]
        invariants: tuple[type[InvariantBase], ...]
        field_matchers: tuple[tuple[str, str], ...]
        inline: bool
        side_effect: bool
        has_annotation: bool
        has_untyped_scatter: bool

    @dataclass(frozen=True)
    class Policy:
        """A handler that reacts to a domain/integration/system event."""

        name: str
        subscribes: tuple[type[Event], ...]
        produces: tuple[type[Event], ...]
        scatters: tuple[type[Event], ...]
        raises: tuple[type[Exception], ...]
        invariants: tuple[type[InvariantBase], ...]
        field_matchers: tuple[tuple[str, str], ...]
        side_effect: bool
        has_annotation: bool
        has_untyped_scatter: bool

    @dataclass(frozen=True)
    class Edge:
        """A directed causal edge: ``source`` → (via reaction) → ``target``."""

        source: type[Event]
        via: str
        target: type[Event]
        kind: Literal["solid", "scatter", "raises", "framework"]

    @dataclass(frozen=True)
    class Invariant:
        """An invariant declared by at least one handler.

        Aggregated across the graph: ``commands`` lists every command whose
        handler declares this invariant; ``declared_by`` and ``reactors`` are
        handler names. A reactor is a handler subscribed to
        ``InvariantViolated`` with a pinned ``invariant=`` field matcher.
        """

        cls: type[InvariantBase]
        commands: tuple[type[CommandBase], ...]
        declared_by: tuple[str, ...]
        reactors: tuple[str, ...]

    # Runtime union — type(DomainModel.Reaction) is types.UnionType.
    Reaction = CommandHandler | Policy

    # ---- fields ----

    aggregates: dict[str, DomainModel.Aggregate]
    integration_events: tuple[type[IntegrationEvent], ...]
    system_events: tuple[type[SystemEvent], ...]
    command_handlers: tuple[DomainModel.CommandHandler, ...]
    policies: tuple[DomainModel.Policy, ...]
    edges: tuple[DomainModel.Edge, ...]
    seeds: tuple[type[Event], ...]
    invariants: tuple[DomainModel.Invariant, ...]

    # ---- derived accessors ----

    @property
    def reactions(self) -> tuple[Any, ...]:
        """``command_handlers`` + ``policies``, in registration order.

        Return type is ``tuple[DomainModel.Reaction, ...]``; typed as
        ``tuple[Any, ...]`` at runtime to keep frozen-dataclass machinery happy.
        """
        return (*self.command_handlers, *self.policies)

    # ---- builder (internal) ----

    @classmethod
    def _build(
        cls,
        handler_metas: list[HandlerMeta],
        return_info: dict[str, ReturnInfo],
    ) -> DomainModel:
        return _build_domain_model(handler_metas, return_info)

    # ---- renderers ----

    def text(self, view: View = "choreography") -> str:
        """Render a human-readable tree."""
        if view == "structure":
            return _render_text_structure(self)
        if view == "choreography":
            return _render_text_choreography(self)
        raise ValueError(
            f"Unknown view {view!r}; expected 'structure' or 'choreography'"
        )

    def mermaid(self) -> str:
        """Render the unified choreography mermaid diagram.

        Shows handler-driven flow edges plus dashed ownership arrows for
        any command→outcome pair that isn't already linked by flow — so
        declared outcomes always appear connected to their command, even
        when produced indirectly via a policy reacting to an intermediate
        event.
        """
        return _render_mermaid_choreography(self)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dict representation.

        Event/Exception/Command classes are encoded as their qualnames.
        """
        return _encode_model(self)

    def json(self, *, indent: int | None = 2) -> str:
        """Return a JSON string of the model."""
        return _json.dumps(self.to_dict(), indent=indent)


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------


def _classify_event_bucket(  # noqa: PLR0911, PLR0912
    event_type: type[Event],
    aggregates: dict[str, dict[str, Any]],
    integration: list[type[IntegrationEvent]],
    system: list[type[SystemEvent]],
    seen: set[type[Event]],
) -> None:
    """Sort *event_type* into aggregate / integration / system buckets.

    Rule: if ``__aggregate__`` is set, aggregate membership wins regardless
    of whether the event inherits ``DomainEvent``, ``Halted``, or any other
    branch. This lets users nest e.g. ``class Blocked(Halted)`` inside an
    aggregate for locality without having it drift into the system bucket.

    ``aggregates`` is a mutable intermediate dict keyed by aggregate name;
    each value is ``{"commands": {cmd_name: {"type": cls, "outcomes": [...]}},
    "events": [Event, ...]}``.
    """
    if event_type in seen:
        return
    seen.add(event_type)
    if not isinstance(event_type, type):
        return

    if issubclass(event_type, CommandBase):
        agg_name = getattr(event_type, "__aggregate__", None)
        if agg_name is None:
            return
        entry = aggregates.setdefault(agg_name, {"commands": {}, "events": []})
        cmd_entry = entry["commands"].setdefault(
            event_type.__name__, {"type": event_type, "outcomes": []}
        )
        cmd_entry["type"] = event_type
        outcomes_union = getattr(event_type, "Outcomes", None)
        if outcomes_union is not None:
            args = typing.get_args(outcomes_union) or (outcomes_union,)
            for t in args:
                if (
                    isinstance(t, type)
                    and issubclass(t, Event)
                    and t not in cmd_entry["outcomes"]
                ):
                    cmd_entry["outcomes"].append(t)
        return

    if issubclass(event_type, DomainEvent):
        agg_name = getattr(event_type, "__aggregate__", None)
        if agg_name is None:
            return
        entry = aggregates.setdefault(agg_name, {"commands": {}, "events": []})
        cmd = getattr(event_type, "__command__", None)
        if cmd is not None:
            cmd_entry = entry["commands"].setdefault(
                cmd.__name__, {"type": cmd, "outcomes": []}
            )
            if event_type not in cmd_entry["outcomes"]:
                cmd_entry["outcomes"].append(event_type)
        elif event_type not in entry["events"]:
            entry["events"].append(event_type)
        return

    # Non-DomainEvent event nested in an Aggregate (e.g. Halted subtype):
    # aggregate membership beats IntegrationEvent/SystemEvent classification.
    agg_name = getattr(event_type, "__aggregate__", None)
    if agg_name is not None:
        entry = aggregates.setdefault(agg_name, {"commands": {}, "events": []})
        if event_type not in entry["events"]:
            entry["events"].append(event_type)
        return

    if issubclass(event_type, IntegrationEvent):
        if event_type not in integration:
            integration.append(event_type)
        return

    if issubclass(event_type, SystemEvent):
        if event_type not in system:
            system.append(event_type)


def _build_domain_model(  # noqa: PLR0912
    handler_metas: list[HandlerMeta],
    return_info: dict[str, ReturnInfo],
) -> DomainModel:
    agg_raw: dict[str, dict[str, Any]] = {}
    integration: list[type[IntegrationEvent]] = []
    system: list[type[SystemEvent]] = []
    seen: set[type[Event]] = set()

    command_handlers: list[DomainModel.CommandHandler] = []
    policies: list[DomainModel.Policy] = []
    edges: list[DomainModel.Edge] = []

    command_to_handler_names: dict[type[CommandBase], list[str]] = {}

    any_produces_interrupted = False
    any_subscribes_resumed = False

    for meta in handler_metas:
        info = return_info[meta.name]

        # Classify all subscribed events into taxonomy buckets.
        for et in meta.event_types:
            _classify_event_bucket(et, agg_raw, integration, system, seen)
        for et in info.event_types:
            _classify_event_bucket(et, agg_raw, integration, system, seen)
        for et in info.scatter_types:
            _classify_event_bucket(et, agg_raw, integration, system, seen)

        if info.has_interrupted:
            any_produces_interrupted = True
        if any(issubclass(t, Resumed) for t in meta.event_types):
            any_subscribes_resumed = True

        is_command_handler = any(
            isinstance(t, type) and issubclass(t, CommandBase) for t in meta.event_types
        )
        side_effect = (
            not info.event_types and not info.scatter_types and not info.has_scatter
        )
        invariants = tuple(cls for cls, _fn in meta.invariants)
        field_matchers = tuple(
            (fname, _matcher_repr(matcher, is_type))
            for fname, matcher, is_type in meta.field_matchers
        )

        if is_command_handler:
            cmds = tuple(
                t
                for t in meta.event_types
                if isinstance(t, type) and issubclass(t, CommandBase)
            )
            ch = DomainModel.CommandHandler(
                name=meta.name,
                commands=cmds,
                produces=tuple(info.event_types),
                scatters=tuple(info.scatter_types),
                raises=tuple(meta.raises),
                invariants=invariants,
                field_matchers=field_matchers,
                inline=getattr(meta.fn, "_inline_command", None) is not None,
                side_effect=side_effect,
                has_annotation=info.has_annotation,
                has_untyped_scatter=info.has_scatter and not info.scatter_types,
            )
            command_handlers.append(ch)
            for cmd in cmds:
                command_to_handler_names.setdefault(cmd, []).append(meta.name)
        else:
            policy = DomainModel.Policy(
                name=meta.name,
                subscribes=tuple(meta.event_types),
                produces=tuple(info.event_types),
                scatters=tuple(info.scatter_types),
                raises=tuple(meta.raises),
                invariants=invariants,
                field_matchers=field_matchers,
                side_effect=side_effect,
                has_annotation=info.has_annotation,
                has_untyped_scatter=info.has_scatter and not info.scatter_types,
            )
            policies.append(policy)

        # Emit edges — solid for declared returns, scatter for Scatter[X],
        # raises edges to HandlerRaised, mirroring the current mermaid() logic.
        for src_type in meta.event_types:
            for tgt in info.event_types:
                edges.append(
                    DomainModel.Edge(
                        source=src_type, via=meta.name, target=tgt, kind="solid"
                    )
                )
            for tgt in info.scatter_types:
                edges.append(
                    DomainModel.Edge(
                        source=src_type, via=meta.name, target=tgt, kind="scatter"
                    )
                )
            for _exc in meta.raises:
                edges.append(
                    DomainModel.Edge(
                        source=src_type,
                        via=meta.name,
                        target=HandlerRaised,
                        kind="raises",
                    )
                )

    # Framework Interrupted → Resumed edge (exists iff both halves appear).
    if any_produces_interrupted and any_subscribes_resumed:
        edges.append(
            DomainModel.Edge(
                source=Interrupted, via="framework", target=Resumed, kind="framework"
            )
        )

    # Freeze aggregates into nested dataclass form, attaching handler names.
    aggregates: dict[str, DomainModel.Aggregate] = {}
    for agg_name, raw in agg_raw.items():
        commands: dict[str, DomainModel.Command] = {}
        for cmd_name, cmd_raw in raw["commands"].items():
            cmd_type = cmd_raw["type"]
            commands[cmd_name] = DomainModel.Command(
                cls=cmd_type,
                outcomes=tuple(cmd_raw["outcomes"]),
                handlers=tuple(command_to_handler_names.get(cmd_type, [])),
            )
        aggregates[agg_name] = DomainModel.Aggregate(
            name=agg_name,
            commands=commands,
            events=tuple(raw["events"]),
        )

    # Seed events: sources that never appear as targets.
    sources = {e.source for e in edges}
    targets = {e.target for e in edges}
    seeds = tuple(sorted(sources - targets, key=_event_label))

    invariants_agg = _aggregate_invariants(command_handlers, policies)

    return DomainModel(
        aggregates=aggregates,
        integration_events=tuple(integration),
        system_events=tuple(system),
        command_handlers=tuple(command_handlers),
        policies=tuple(policies),
        edges=tuple(edges),
        seeds=seeds,
        invariants=invariants_agg,
    )


def _aggregate_invariants(
    command_handlers: list[DomainModel.CommandHandler],
    policies: list[DomainModel.Policy],
) -> tuple[DomainModel.Invariant, ...]:
    """Roll per-handler invariant declarations into first-class domain nodes.

    One entry per unique ``Invariant`` subclass. Declaration order follows
    the first handler that declared each class. Reactors are policies
    subscribed to ``InvariantViolated`` with a pinned ``invariant=`` type
    matcher; the matcher string was normalised by ``_matcher_repr`` to the
    matcher class's ``__name__`` — we match on that.
    """
    commands_by_cls: dict[type[InvariantBase], list[type[CommandBase]]] = {}
    declared_by_cls: dict[type[InvariantBase], list[str]] = {}
    order: list[type[InvariantBase]] = []

    for ch in command_handlers:
        for inv_cls in ch.invariants:
            if inv_cls not in declared_by_cls:
                order.append(inv_cls)
                declared_by_cls[inv_cls] = []
                commands_by_cls[inv_cls] = []
            if ch.name not in declared_by_cls[inv_cls]:
                declared_by_cls[inv_cls].append(ch.name)
            for cmd in ch.commands:
                if cmd not in commands_by_cls[inv_cls]:
                    commands_by_cls[inv_cls].append(cmd)

    reactors_by_name: dict[str, list[str]] = {}
    for p in policies:
        if not any(
            isinstance(sub, type) and issubclass(sub, InvariantViolated)
            for sub in p.subscribes
        ):
            continue
        for fname, repr_ in p.field_matchers:
            if fname == "invariant":
                reactors_by_name.setdefault(repr_, []).append(p.name)

    return tuple(
        DomainModel.Invariant(
            cls=inv_cls,
            commands=tuple(commands_by_cls[inv_cls]),
            declared_by=tuple(declared_by_cls[inv_cls]),
            reactors=tuple(reactors_by_name.get(inv_cls.__name__, ())),
        )
        for inv_cls in order
    )


# ---------------------------------------------------------------------------
# Text renderers
# ---------------------------------------------------------------------------


def _command_annotations(
    d: DomainModel, cmd_cls: type
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    """Collect dedup'd raises + invariant names + scatter targets from every
    command handler subscribed to *cmd_cls*."""
    raises: list[str] = []
    invariants: list[str] = []
    scatters: list[str] = []
    for ch in d.command_handlers:
        if cmd_cls not in ch.commands:
            continue
        for exc in ch.raises:
            if exc.__name__ not in raises:
                raises.append(exc.__name__)
        for inv in ch.invariants:
            label = inv.__name__
            if label not in invariants:
                invariants.append(label)
        for tgt in ch.scatters:
            label = f"Scatter[{_event_label(tgt)}]"
            if label not in scatters:
                scatters.append(label)
        if ch.has_untyped_scatter and "Scatter" not in scatters:
            scatters.append("Scatter")
    return tuple(raises), tuple(invariants), tuple(scatters)


def _policy_targets(p: DomainModel.Policy) -> str:
    """Comma-joined produces + scatter annotations, or empty string."""
    parts = [_event_label(t) for t in p.produces]
    for tgt in p.scatters:
        parts.append(f"Scatter[{_event_label(tgt)}]")
    if p.has_untyped_scatter:
        parts.append("Scatter")
    return ", ".join(parts)


def _render_taxonomy_lines(  # noqa: PLR0912
    d: DomainModel, *, include_handlers: bool
) -> list[str]:
    """Shared aggregate/integration/system block used by both text views.

    ``include_handlers`` adds ``  (handlers: ...)`` suffixes + ``raises`` +
    ``[invariant: …]`` annotations on commands in the choreography view;
    structure view omits those flow concerns but still marks free-standing
    ``Halted`` events with a ``[Halted]`` tag.
    """
    lines: list[str] = []
    if d.aggregates:
        lines.append("Aggregates:")
        for agg_name, agg in d.aggregates.items():
            lines.append(f"  {agg_name}")
            for cmd_name, cmd in agg.commands.items():
                suffix_parts: list[str] = []
                if include_handlers:
                    if cmd.handlers:
                        suffix_parts.append(f"handlers: {', '.join(cmd.handlers)}")
                    raises, invariants, scatters = _command_annotations(d, cmd.cls)
                    if scatters:
                        suffix_parts.append(f"scatters {', '.join(scatters)}")
                    if raises:
                        suffix_parts.append(f"raises {', '.join(raises)}")
                    if invariants:
                        suffix_parts.append(f"invariant: {', '.join(invariants)}")
                suffix = f"  ({'; '.join(suffix_parts)})" if suffix_parts else ""
                lines.append(f"    Command: {cmd_name}{suffix}")
                for outcome in cmd.outcomes:
                    lines.append(f"      → {_event_label(outcome)}")
            for event in agg.events:
                halt_tag = "  [Halted]" if _node_class(event) == "halt" else ""
                lines.append(f"    Event: {_event_label(event)}{halt_tag}")
    if d.integration_events:
        lines.append("Integration events:")
        for int_ev in d.integration_events:
            lines.append(f"  {_event_label(int_ev)}")
    if d.system_events:
        lines.append("System events:")
        for sys_ev in d.system_events:
            lines.append(f"  {_event_label(sys_ev)}")
    if d.invariants:
        lines.append("Invariants:")
        for inv in d.invariants:
            on_cmds = ", ".join(_event_label(c) for c in inv.commands)
            annotations = [f"on {on_cmds}"] if on_cmds else []
            if include_handlers and inv.reactors:
                annotations.append(f"reacted by: {', '.join(inv.reactors)}")
            suffix = f"  ({'; '.join(annotations)})" if annotations else ""
            lines.append(f"  {inv.cls.__name__}{suffix}")
    return lines


def _render_text_structure(d: DomainModel) -> str:
    return "\n".join(_render_taxonomy_lines(d, include_handlers=False))


def _render_text_choreography(d: DomainModel) -> str:
    lines = _render_taxonomy_lines(d, include_handlers=True)
    if d.policies:
        lines.append("Policies:")
        for p in d.policies:
            subs = ", ".join(_event_label(t) for t in p.subscribes)
            targets = _policy_targets(p)
            flow = f"{subs} → {targets}" if targets else subs
            annotations: list[str] = []
            if p.raises:
                annotations.append(f"raises {', '.join(e.__name__ for e in p.raises)}")
            if p.invariants:
                annotations.append(
                    f"invariant: {', '.join(i.__name__ for i in p.invariants)}"
                )
            if p.side_effect:
                annotations.append("side-effect")
            tail = f"  [{'; '.join(annotations)}]" if annotations else ""
            lines.append(f"  {p.name}  ({flow}){tail}")
    framework = [e for e in d.edges if e.kind == "framework"]
    if framework:
        lines.append("Framework:")
        for e in framework:
            lines.append(f"  {_event_label(e.source)} → {_event_label(e.target)}")
    if d.seeds:
        lines.append("Seed events:")
        for s in d.seeds:
            lines.append(f"  {_event_label(s)}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Mermaid renderers
# ---------------------------------------------------------------------------


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


def _render_mermaid_choreography(d: DomainModel) -> str:  # noqa: PLR0912, PLR0915
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


def _reaction_subscribes(r: Any) -> tuple[type[Event], ...]:
    """Return the subscribed events for a CommandHandler or Policy."""
    if isinstance(r, DomainModel.CommandHandler):
        return r.commands
    return r.subscribes  # Policy


# ---------------------------------------------------------------------------
# JSON encoding
# ---------------------------------------------------------------------------


def _qn(cls: type) -> str:
    return cls.__qualname__


def _encode_reaction(r: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "name": r.name,
        "produces": [_qn(t) for t in r.produces],
        "scatters": [_qn(t) for t in r.scatters],
        "raises": [_qn(t) for t in r.raises],
        "invariants": [_qn(t) for t in r.invariants],
        "field_matchers": [list(fm) for fm in r.field_matchers],
        "side_effect": r.side_effect,
        "has_annotation": r.has_annotation,
        "has_untyped_scatter": r.has_untyped_scatter,
    }
    if isinstance(r, DomainModel.CommandHandler):
        base["kind"] = "command_handler"
        base["commands"] = [_qn(t) for t in r.commands]
        base["inline"] = r.inline
    else:
        base["kind"] = "policy"
        base["subscribes"] = [_qn(t) for t in r.subscribes]
    return base


def _encode_model(d: DomainModel) -> dict[str, Any]:
    aggregates: dict[str, Any] = {}
    for agg_name, agg in d.aggregates.items():
        aggregates[agg_name] = {
            "name": agg.name,
            "commands": {
                cmd_name: {
                    "type": _qn(cmd.cls),
                    "outcomes": [_qn(t) for t in cmd.outcomes],
                    "handlers": list(cmd.handlers),
                }
                for cmd_name, cmd in agg.commands.items()
            },
            "events": [_qn(t) for t in agg.events],
        }
    return {
        "aggregates": aggregates,
        "integration_events": [_qn(t) for t in d.integration_events],
        "system_events": [_qn(t) for t in d.system_events],
        "command_handlers": [_encode_reaction(r) for r in d.command_handlers],
        "policies": [_encode_reaction(r) for r in d.policies],
        "edges": [
            {
                "source": _qn(e.source),
                "via": e.via,
                "target": _qn(e.target),
                "kind": e.kind,
            }
            for e in d.edges
        ],
        "seeds": [_qn(t) for t in d.seeds],
        "invariants": [
            {
                "cls": _qn(inv.cls),
                "commands": [_qn(c) for c in inv.commands],
                "declared_by": list(inv.declared_by),
                "reactors": list(inv.reactors),
            }
            for inv in d.invariants
        ],
    }


# Silence unused-import warnings for types referenced only in string annotations.
_ = dc_fields
