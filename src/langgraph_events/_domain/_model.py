"""`DomainModel` dataclasses, builder, and shared rendering helpers."""

from __future__ import annotations

import json as _json
import typing
from dataclasses import dataclass
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
    domain's similarly-named class.
    """
    return cls.__qualname__.replace("<locals>.", "").replace(".", "_")


def _matcher_repr(matcher: Any, is_type: bool) -> str:
    if is_type and isinstance(matcher, type):
        return matcher.__name__
    return repr(matcher)


# Short classDef key for the choreography flowchart. Drives both the node
# shape and the fill/stroke palette. Halted nested under a domain gets the
# same amber fill as SystemEvent but a dashed double-outline stroke so
# terminals are visually distinct without stealing red from error flow.
_NODE_CLASS_BASES: tuple[tuple[type, str], ...] = (
    (CommandBase, "cmd"),
    (DomainEvent, "devt"),
    (IntegrationEvent, "intg"),
    (Halted, "halt"),
)


def _node_class(cls: type) -> str:
    """Return the classDef key (``cmd``/``devt``/``intg``/``syst``/``halt``)."""
    for base, cls_key in _NODE_CLASS_BASES:
        if issubclass(cls, base):
            return cls_key
    return "syst"


@dataclass(frozen=True)
class DomainModel:
    """Code-derived snapshot of an ``EventGraph``'s domain model.

    Two lenses:

    - ``view="structure"`` — the **taxonomy** (domains → commands → outcomes,
      free-standing DomainEvents, integration and system events). No handlers,
      no edges.
    - ``view="choreography"`` — the **full picture**: taxonomy + command
      handlers + policies + event-to-event edges + seed events. Default.

    "Choreography" is the event-storming term for event-driven flow without
    a central orchestrator — exactly what this library models.

    Rendering::

        d = graph.domain()
        d.text()                      # human-readable tree
        d.text(view="structure")      # taxonomy only — no handlers
        d.mermaid()                   # graph LR choreography
        d.json()                      # serializable snapshot

    Data access::

        d.domains              # dict[str, DomainModel.Domain]
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
    class Domain:
        """A domain — the grouping for its nested commands and events.

        ``events`` holds any ``Event`` nested in the domain that is not an
        outcome of one of its commands — typically free-standing DomainEvents,
        but also ``Halted`` subtypes nested in the domain for locality.
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

        Rolled up across the graph: ``commands`` lists every command whose
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

    domains: dict[str, DomainModel.Domain]
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
        from langgraph_events._domain._text import (  # noqa: PLC0415
            render_text_choreography,
            render_text_structure,
        )

        if view == "structure":
            return render_text_structure(self)
        if view == "choreography":
            return render_text_choreography(self)
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
        from langgraph_events._domain._mermaid import (  # noqa: PLC0415
            render_mermaid_choreography,
        )

        return render_mermaid_choreography(self)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dict representation.

        Event/Exception/Command classes are encoded as their qualnames.
        """
        from langgraph_events._domain._json import encode_model  # noqa: PLC0415

        return encode_model(self)

    def json(self, *, indent: int | None = 2) -> str:
        """Return a JSON string of the model."""
        return _json.dumps(self.to_dict(), indent=indent)


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------


def _classify_event_bucket(  # noqa: PLR0911, PLR0912
    event_type: type[Event],
    domains: dict[str, dict[str, Any]],
    integration: list[type[IntegrationEvent]],
    system: list[type[SystemEvent]],
    seen: set[type[Event]],
) -> None:
    """Sort *event_type* into domain / integration / system buckets.

    Rule: if ``__domain__`` is set, domain membership wins regardless of
    whether the event inherits ``DomainEvent``, ``Halted``, or any other
    branch. This lets users nest e.g. ``class Blocked(Halted)`` inside a
    domain for locality without having it drift into the system bucket.

    ``domains`` is a mutable intermediate dict keyed by domain name; each
    value is ``{"commands": {cmd_name: {"type": cls, "outcomes": [...]}},
    "events": [Event, ...]}``.
    """
    if event_type in seen:
        return
    seen.add(event_type)
    if not isinstance(event_type, type):
        return

    if issubclass(event_type, CommandBase):
        domain_name = getattr(event_type, "__domain__", None)
        if domain_name is None:
            return
        entry = domains.setdefault(domain_name, {"commands": {}, "events": []})
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
        domain_name = getattr(event_type, "__domain__", None)
        if domain_name is None:
            return
        entry = domains.setdefault(domain_name, {"commands": {}, "events": []})
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

    # Non-DomainEvent event nested in a Domain (e.g. Halted subtype): domain
    # membership beats IntegrationEvent/SystemEvent classification.
    domain_name = getattr(event_type, "__domain__", None)
    if domain_name is not None:
        entry = domains.setdefault(domain_name, {"commands": {}, "events": []})
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
    domain_raw: dict[str, dict[str, Any]] = {}
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
            _classify_event_bucket(et, domain_raw, integration, system, seen)
        for et in info.event_types:
            _classify_event_bucket(et, domain_raw, integration, system, seen)
        for et in info.scatter_types:
            _classify_event_bucket(et, domain_raw, integration, system, seen)

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

    # Freeze domains into nested dataclass form, attaching handler names.
    domains: dict[str, DomainModel.Domain] = {}
    for domain_name, raw in domain_raw.items():
        commands: dict[str, DomainModel.Command] = {}
        for cmd_name, cmd_raw in raw["commands"].items():
            cmd_type = cmd_raw["type"]
            commands[cmd_name] = DomainModel.Command(
                cls=cmd_type,
                outcomes=tuple(cmd_raw["outcomes"]),
                handlers=tuple(command_to_handler_names.get(cmd_type, [])),
            )
        domains[domain_name] = DomainModel.Domain(
            name=domain_name,
            commands=commands,
            events=tuple(raw["events"]),
        )

    # Seed events: sources that never appear as targets.
    sources = {e.source for e in edges}
    targets = {e.target for e in edges}
    seeds = tuple(sorted(sources - targets, key=_event_label))

    invariants_rolled = _rollup_invariants(command_handlers, policies)

    return DomainModel(
        domains=domains,
        integration_events=tuple(integration),
        system_events=tuple(system),
        command_handlers=tuple(command_handlers),
        policies=tuple(policies),
        edges=tuple(edges),
        seeds=seeds,
        invariants=invariants_rolled,
    )


def _rollup_invariants(
    command_handlers: list[DomainModel.CommandHandler],
    policies: list[DomainModel.Policy],
) -> tuple[DomainModel.Invariant, ...]:
    """Roll per-handler invariant declarations into first-class nodes.

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
