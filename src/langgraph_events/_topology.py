"""EventTopology: the handlers, reducers and namespaces of a graph."""

from __future__ import annotations

import dataclasses
import types
import typing
from typing import TYPE_CHECKING, Any, NamedTuple, TypedDict, cast

from langgraph.graph import END, START, StateGraph

from langgraph_events._causes import CauseEntry
from langgraph_events._event import (
    OUTCOMES_ATTR,
    Command,
    DomainEvent,
    Event,
    Halted,
    HandlerRaised,
    Interrupted,
    Invariant,
    InvariantViolated,
    Namespace,
    Scatter,
    _iter_nested_outcomes,
    is_abstract_event,
)
from langgraph_events._handler import (
    BoundHandler,
    HandlerMeta,
    ServiceRequirement,
    _resolve_hints_and_errors,
    normalize_previous_names,
    on,
)
from langgraph_events._identity import command_identity
from langgraph_events._internal import (
    _BASE_FIELDS,
    _InputState,
    _leaf_node,
    _OutputState,
    build_state_schema,
    make_dispatch,
    make_handler_node,
    make_router_node,
    make_seed_node,
)
from langgraph_events._labels import distinct_labels, escalating_labels
from langgraph_events._namespace import NamespaceModel
from langgraph_events._namespace._command_privacy import enforce_command_privacy
from langgraph_events._services import NamedServices
from langgraph_events._warn import warn_user

if TYPE_CHECKING:
    from collections.abc import (
        Callable,
        Iterable,
        Mapping,
        Sequence,
    )

    from langchain_core.runnables import RunnableConfig, RunnableLambda
    from langgraph.graph.state import CompiledStateGraph
    from langgraph.store.base import BaseStore
    from langgraph.types import Checkpointer

    from langgraph_events._reducer import BaseReducer
    from langgraph_events._services import Services
    from langgraph_events._types import StateDict
    from langgraph_events.serde._jsonplus import (
        NamespaceAwareSerde,
    )
    from langgraph_events.serde.migrations._core import Migration


class OrphanedEventWarning(UserWarning):
    """Issued when handler return types have no matching subscriber."""


def _any_catcher_covers(
    catchers: list[tuple[HandlerMeta, type[Exception] | None]],
    exc_type: type[Exception],
) -> bool:
    """Return True if any catcher would handle *exc_type*."""
    for _meta, exc_filter in catchers:
        if exc_filter is None or issubclass(exc_type, exc_filter):
            return True
    return False


class ReturnInfo(NamedTuple):
    """Parsed handler return-type annotation."""

    event_types: list[type[Event]]
    scatter_types: list[type[Event]]
    has_interrupted: bool
    has_annotation: bool


class ReturnContract(NamedTuple):
    """Runtime contract for what a handler may legally return.

    Computed at ``EventGraph`` construction; enforced in ``_collect_result``.
    ``None`` → no enforcement (legacy shape-only check).
    """

    types: tuple[type, ...]
    """Allowed concrete event types for non-Scatter returns."""

    scatter_types: tuple[type, ...] | None
    """Allowed element types for Scatter returns. None = no check."""

    source: str
    """Human-readable origin of the contract, used in error messages."""


def _scatter_event_types(scatter_alias: object) -> list[type[Event]]:
    """Extract Event types from a parameterized ``Scatter[X]`` alias.

    Unpacks Union/UnionType members so ``Scatter[A | B]`` behaves like
    ``Scatter[A] | Scatter[B]`` for topology parsing. Non-Event members
    (e.g. ``None`` from ``Scatter[A | None]``) are silently dropped — only
    Event subclasses participate in topology.
    """
    out: list[type[Event]] = []
    for arg in typing.get_args(scatter_alias):
        if typing.get_origin(arg) in (typing.Union, types.UnionType):
            members = typing.get_args(arg)
        else:
            members = (arg,)
        for member in members:
            if isinstance(member, type) and issubclass(member, Event):
                out.append(member)
    return out


def _format_scatter_repr(scatter_alias: object) -> str:
    """Render a Scatter annotation back into a readable form for error text."""
    args = typing.get_args(scatter_alias)
    if not args:
        return "Scatter"
    parts: list[str] = []
    for arg in args:
        if typing.get_origin(arg) in (typing.Union, types.UnionType):
            inner = " | ".join(
                getattr(m, "__name__", repr(m)) for m in typing.get_args(arg)
            )
            parts.append(inner)
        else:
            parts.append(getattr(arg, "__name__", repr(arg)))
    return f"Scatter[{' | '.join(parts)}]"


def _raise_empty_scatter(meta: HandlerMeta, offending: str) -> None:
    """Reject a Scatter annotation that contributes no concrete event types.

    Bare ``Scatter``, ``Scatter[Any]``, ``Scatter[TypeVar]``, and
    ``Scatter[<abstract event>]`` (see ``is_abstract_event``, for example
    DomainEvent or RunPaused) all bypass build-time privacy enforcement
    because the resolved target set is empty or non-discriminating. Raise at
    build time so users can't paper over a privacy violation by widening the
    annotation.
    """
    cmd: type[Command] | None = getattr(meta.fn, "_inline_command", None)
    nested = _iter_nested_outcomes(cmd) if cmd is not None else ()
    suggested = " | ".join(o.__name__ for o in nested) if nested else "EventA | EventB"
    raise TypeError(
        f"{meta.blame} returns {offending}, which declares no concrete event "
        f"types and bypasses build-time privacy enforcement. "
        f"Use `Scatter[{suggested}]` to enumerate the events you scatter. "
        f"(Do not work around this by demoting your Command to a "
        f"DomainEvent/IntegrationEvent — that loses privacy and outcome "
        f"guarantees.)"
    )


def _parse_return_types(meta: HandlerMeta) -> ReturnInfo:
    """Parse handler return annotation into a ``ReturnInfo``.

    Raises ``TypeError`` for Scatter shapes that contribute no concrete event
    types — see :func:`_raise_empty_scatter`.
    """
    hints, hint_errors = _resolve_hints_and_errors(meta.fn)
    if "return" in hint_errors:
        # A handler constructs the events it returns, so those classes are
        # already importable at run time. An unresolvable return annotation is
        # therefore always a defect, and always a one-line fix. Raise rather
        # than fall back: the fallback drew the same "?" topology edge as a
        # genuinely unannotated handler, so the defect stayed invisible in a
        # committed mermaid baseline. See issue #183.
        raise TypeError(
            f"{meta.blame} has a return annotation that did "
            f"not resolve ({hint_errors['return']}). The framework reads the "
            f"return annotation to build the topology, so it cannot be left "
            f"unresolvable. Make every name in the annotation importable at "
            f"run time."
        )

    return_hint = hints.get("return")
    if return_hint is None:
        return ReturnInfo([], [], False, False)

    # If the top-level hint is Scatter[X], wrap it so the loop sees Scatter[X]
    # as a single candidate (otherwise get_args returns the type params).
    origin = typing.get_origin(return_hint)
    if origin is Scatter:
        candidates = (return_hint,)
    else:
        args = typing.get_args(return_hint)
        candidates = args if args else (return_hint,)

    event_types: list[type[Event]] = []
    scatter_types: list[type[Event]] = []
    has_interrupted = False
    for arg in candidates:
        if arg is type(None):
            continue
        if arg is Scatter:
            _raise_empty_scatter(meta, "bare `Scatter`")
        elif typing.get_origin(arg) is Scatter:
            members = _scatter_event_types(arg)
            if not members or any(is_abstract_event(m) for m in members):
                _raise_empty_scatter(meta, f"`{_format_scatter_repr(arg)}`")
            scatter_types.extend(members)
        elif isinstance(arg, type) and issubclass(arg, Event):
            if issubclass(arg, Interrupted):
                has_interrupted = True
            event_types.append(arg)
        else:
            raise TypeError(
                f"{meta.blame} has a return annotation that "
                f"holds {arg!r}, which is not an Event subclass. The framework "
                f"reads the return annotation to route the events, so each "
                f"member must be an Event subclass, Scatter[...] or None. For "
                f"an event mixin, name the Event classes that compose it."
            )

    return ReturnInfo(event_types, scatter_types, has_interrupted, True)


def _compute_return_contract(
    meta: HandlerMeta, info: ReturnInfo
) -> ReturnContract | None:
    """Derive the runtime return-contract for a handler.

    Priority:
    1. Explicit return annotation — authoritative.
    2. ``Command.Outcomes`` inferred from subscribed event types.
    3. No contract (legacy shape-only check in ``_collect_result``).
    """
    if info.has_annotation:
        union_desc = " | ".join(t.__name__ for t in info.event_types) or "None"
        return ReturnContract(
            types=tuple(info.event_types),
            scatter_types=tuple(info.scatter_types) if info.scatter_types else None,
            source=f"declared return type `{union_desc}`",
        )

    outcomes: list[type] = []
    command_names: list[str] = []
    for et in meta.event_types:
        if not (isinstance(et, type) and issubclass(et, Command)):
            continue
        outs = getattr(et, OUTCOMES_ATTR, None)
        if outs is None:
            continue
        args = typing.get_args(outs) or (outs,)
        for t in args:
            if isinstance(t, type) and t not in outcomes:
                outcomes.append(t)
        command_names.append(et.__qualname__)

    if not outcomes:
        return None

    sources = ", ".join(command_names)
    return ReturnContract(
        types=tuple(outcomes),
        scatter_types=None,
        source=f"outcomes of {sources}",
    )


def _verify_inline_outcome_coverage(meta: HandlerMeta, info: ReturnInfo) -> None:
    """For inline handlers with an explicit return annotation, require the
    annotation to cover every nested ``DomainEvent`` of the owning ``Command``.

    The inline handler is the only producer of a Command's nested outcomes
    (see :class:`CommandPrivacyError`); dropping an outcome from its
    annotation is almost always a mistake.

    No annotation → skipped; the contract falls back to ``Command.Outcomes``.
    """
    cmd = getattr(meta.fn, "_inline_command", None)
    if cmd is None or not info.has_annotation:
        return
    nested_outcomes = _iter_nested_outcomes(cmd)
    if not nested_outcomes:
        return
    covered = tuple(info.event_types) + tuple(info.scatter_types)
    missing = [o for o in nested_outcomes if not any(issubclass(c, o) for c in covered)]
    if not missing:
        return
    # Coverage is decided by identity (issubclass), so a declared class and a
    # missing outcome can share a __name__ and still not match. Rendering both
    # by name then reads as a tautology — "declares `Placed` but does not
    # cover Placed" — and sends the reader to edit an annotation that is
    # already correct. Qualify the names when they collide (#151).
    collision = bool({o.__name__ for o in missing} & {c.__name__ for c in covered})

    # ``escalating_labels`` over the whole cast, not ``distinct_labels``:
    # this renders a list against one shared verdict rather than telling a
    # single pair apart. Same escalation rule, so two lifetimes of one module
    # — identical module *and* qualname — still separate.
    labels = escalating_labels((*covered, *missing, *nested_outcomes))

    def label(t: type) -> str:
        return labels[t]

    declared = " | ".join(label(t) for t in covered) or "(no types)"
    missing_names = ", ".join(label(o) for o in missing)
    hint = (
        f"Add them to the annotation (e.g. `-> "
        f"{' | '.join(label(o) for o in nested_outcomes)}`) or drop "
        f"the annotation to let Outcomes drive the contract."
    )
    if collision:
        local = [t for t in (*covered, *missing) if "<locals>" in t.__qualname__]
        # Appended, not substituted: a collision on one name says nothing
        # about the other outcomes, which may be genuinely uncovered and
        # still need the annotation edit.
        hint += (
            " Note that some names above collide: those are different classes "
            "that happen to share a name, so the annotation already names "
            "something else."
        ) + (
            " A `<locals>` qualname means a class defined inside a function, "
            "whose string annotation resolved to a different object — declare "
            "event classes at module level."
            if local
            else ""
        )
    raise TypeError(
        f"{meta.blame} declares "
        f"return type `{declared}` but does not cover outcome(s): "
        f"{missing_names}. {hint}"
    )


def _register_graph_namespaces(
    event_types: Iterable[type], found: dict[str, type[Namespace]]
) -> None:
    """Record each event's owning namespace into *found*, rejecting collisions.

    Two *distinct* classes sharing a name is rejected: reducer discovery and
    reflection both group by name, so within a single graph the name must
    identify one class. Across graphs it need not, and does not — that is what
    lets several independent engine lifetimes coexist in one process (#148).

    Called once for subscribed types and again for produced ones. A handler
    subscribing to one lifetime and returning another's class would otherwise
    merge the two silently, and reflection would answer from whichever arrived
    first.
    """
    for et in event_types:
        namespace_cls = getattr(et, "__namespace_cls__", None)
        if namespace_cls is None:
            continue
        existing = found.setdefault(namespace_cls.__name__, namespace_cls)
        if existing is namespace_cls:
            continue
        here, there = distinct_labels(existing, namespace_cls)
        raise TypeError(
            f"Two different namespaces named {namespace_cls.__name__!r} "
            f"reached this graph: {here} and {there}. Namespace names must "
            f"be unique within a graph."
        )


def _collect_graph_namespaces(
    handlers: list[Callable[..., Any]],
) -> dict[str, type[Namespace]]:
    """Name → Namespace class for the namespaces this graph's handlers subscribe to.

    The graph's own registry, seeded from the ``__namespace_cls__`` stamps
    rather than from process-global state. Produced types are folded in later
    in ``_hold_handlers``, once return-type introspection has run.
    """
    found: dict[str, type[Namespace]] = {}
    for fn in handlers:
        _register_graph_namespaces(getattr(fn, "_event_types", ()), found)
    return found


def _discover_namespace_reducers(
    namespaces: dict[str, type[Namespace]],
    explicit_reducers: dict[str, BaseReducer],
) -> None:
    """Auto-register reducers declared on the graph's namespaces.

    Unions each namespace's ``__reducers__`` into *explicit_reducers*.
    *namespaces* is the graph's own registry from
    ``_collect_graph_namespaces``, so discovery never reaches a same-named
    namespace belonging to another graph.

    Collisions:
    - Two different discovered reducers with the same name → ``TypeError``.
    - Explicit reducer (already in the dict) shares a name with a discovered
      one → explicit wins; discovered one is skipped silently.
    """
    discovered: dict[str, BaseReducer] = {}
    for namespace_cls in namespaces.values():
        for r in namespace_cls.__reducers__:
            existing = discovered.get(r.name)
            if existing is None:
                discovered[r.name] = r
            elif existing is not r:
                raise TypeError(
                    f"Reducer name {r.name!r} collides between domains: "
                    f"{existing!r} and {r!r}"
                )
    # Merge into explicit; explicit wins on name conflict.
    for name, r in discovered.items():
        explicit_reducers.setdefault(name, r)


_RESERVED_NODE_NAMES = frozenset({"__seed__", "__router__", "__start__", "__end__"})
"""Graph-node names no handler or ``previously=`` alias may claim: the
framework's own pregel nodes (``__seed__``/``__router__``, registered in the
graph build) and LangGraph's reserved endpoints (``__start__``/``__end__``).
Without this check the collision surfaces only at first compile/invoke —
as an opaque "node already present" for the framework pair, and as a
claimant-less "reserved" error for LangGraph's — never naming the
declaration that smuggled the name in."""


def _dedup_handler_name(meta: HandlerMeta, count: int) -> HandlerMeta:
    """Suffix a colliding display name positionally (``handle`` → ``handle_2``).

    The stable ``node_name`` (graph / checkpoint identity) follows the
    deduplicated name for ordinary handlers, but an inline ``Command.handle()``
    handler keeps its command-qualname ``node_name`` untouched — so reordering
    ``handlers=[...]`` never remaps which command a paused checkpoint resumes
    into. See issue #97.
    """
    deduped = f"{meta.name}_{count}"
    # An inline command (or an @on(node_name=...) pin) owns an identity distinct
    # from its display name — keep it; an ordinary handler's identity *is* its
    # name, so it follows the dedup suffix.
    has_stable_identity = meta.node_name != meta.name
    node_name = meta.node_name if has_stable_identity else deduped
    return dataclasses.replace(meta, name=deduped, node_name=node_name)


def _expand_command_handlers(
    handlers: list[Any],
) -> list[Callable[..., Any]]:
    """Replace ``Command`` subclasses with their inline handler functions.

    Each substituted function is stamped via ``on(cls, raises=...,
    invariants=..., previously=...)(fn)`` so that ``HandlerMeta.of``
    sees it like any other ``@on``-subscribed handler.
    ``raises``/``invariants``/``retry``/``previously`` are read from
    class-level attributes on the Command, since inline handlers have no
    decorator slot for them — and a concrete Command may not be subclassed,
    so every one of them is declared by the command class itself. Raises
    ``TypeError`` if a Command subclass has no inline handler.
    """
    expanded: list[Callable[..., Any]] = []
    for h in handlers:
        if isinstance(h, type) and issubclass(h, Command):
            fn = getattr(h, "__command_handler__", None)
            if fn is None:
                raise TypeError(
                    f"Command {h.__qualname__} has no inline handler. "
                    f"Define one public method inside the command class "
                    f"(any name — `handle`, `place`, `ship`, …)."
                )
            existing = getattr(fn, "_inline_command", None)
            if existing is not None and existing is not h:
                handler_name = getattr(fn, "__name__", "<handler>")
                raise TypeError(
                    f"Command {h.__qualname__}'s inline handler "
                    f"({handler_name!r}) is already bound to "
                    f"{existing.__qualname__}. This happens if you alias the "
                    f"same function across Command classes — define each "
                    f"Command's handler in its own class body."
                )
            cmd_raises = getattr(h, "raises", ())
            cmd_invariants = getattr(h, "invariants", None)
            cmd_retry = getattr(h, "retry", None)
            cmd_previously = normalize_previous_names(
                getattr(h, "previously", ()),
                owner=f"Command {command_identity(h)!r}:",
            )
            on(
                h,
                raises=cmd_raises,
                invariants=cmd_invariants,
                previously=cmd_previously,
                retry=cmd_retry,
            )(fn)
            fn._inline_command = h
            expanded.append(fn)
        else:
            expanded.append(h)
    return expanded


class EventTopology:
    """The handlers, reducers and namespaces of a graph, with no services.

    Runs every construction check of :class:`EventGraph` that needs no
    service and no checkpointer. Use it to render or list a graph, to feed
    the migration gate, or to maintain a store, without the services of a
    run::

        topology = EventTopology([classify, route, review])
        print(topology.namespaces().mermaid())
        graph = EventGraph.of(topology, services={"llm": llm})

    A handler parameter that only a service can fill is listed in
    :attr:`requirements`. :meth:`EventGraph.of` binds the services to them.
    """

    # Lazily built by namespaces(); class-level default keeps __init__ lean.
    _namespaces_cache: NamespaceModel | None = None

    def __init__(
        self,
        handlers: list[Callable[..., Any]],
        *,
        reducers: list[BaseReducer] | None = None,
    ) -> None:
        if not handlers:
            raise ValueError("EventTopology requires at least one handler")
        self._hold_handlers(self._hold_reducers(handlers, reducers))

    def _hold_reducers(
        self, handlers: list[Callable[..., Any]], reducers: list[BaseReducer] | None
    ) -> list[Callable[..., Any]]:
        """Expand the commands, collect the namespaces and the reducers.

        Return the expanded handlers. :class:`EventGraph` parses its
        ``services=`` between this step and :meth:`_hold_handlers`.
        """
        handlers = _expand_command_handlers(handlers)
        self._reducers: dict[str, BaseReducer] = {r.name: r for r in (reducers or [])}
        self._namespaces = _collect_graph_namespaces(handlers)
        _discover_namespace_reducers(self._namespaces, self._reducers)
        conflicts = set(self._reducers.keys()) & set(_BASE_FIELDS.keys())
        if conflicts:
            raise ValueError(
                f"Reducer name(s) {conflicts} conflict with reserved state fields"
            )
        return handlers

    def _hold_handlers(self, handlers: list[Callable[..., Any]]) -> None:
        """Classify the handlers and run every check that needs no service."""
        self._handler_metas: list[HandlerMeta] = []

        reducer_names = self.reducer_names
        seen_names: dict[str, int] = {}
        for fn in handlers:
            meta = HandlerMeta.of(fn, reducer_names=reducer_names)
            # Deduplicate colliding display names positionally; see
            # _dedup_handler_name for how the stable node identity is preserved.
            if meta.name in seen_names:
                seen_names[meta.name] += 1
                meta = _dedup_handler_name(meta, seen_names[meta.name])
            else:
                seen_names[meta.name] = 1
            self._handler_metas.append(meta)

        self._validate_node_names()
        self._validate_handler_aliases()

        self._return_info: dict[str, ReturnInfo] = {}
        self._return_contracts: dict[str, ReturnContract | None] = {}
        for meta in self._handler_metas:
            info = _parse_return_types(meta)
            self._return_info[meta.name] = info
            if info.has_annotation and any(t is Event for t in info.event_types):
                raise ValueError(
                    f"{meta.blame} return type includes base 'Event'. "
                    f"Use specific types (e.g., TypeA | TypeB)."
                )
            self._return_contracts[meta.name] = _compute_return_contract(meta, info)
            _verify_inline_outcome_coverage(meta, info)

        self._loose_events = self._register_produced_types()

        enforce_command_privacy(self._handler_metas, self._return_info)
        self._verify_error_handling()

    def _serde_scope(
        self,
    ) -> tuple[tuple[type[Namespace], ...], tuple[type[Event], ...]]:
        """The ``namespaces=`` and ``events=`` of a serde that reads every
        event class this graph can store: its namespaces, and the events
        outside every namespace that its handlers subscribe to or return."""
        return tuple(self._namespaces.values()), self._loose_events

    def serde(
        self, migrations: Sequence[Migration] = (), *, legacy_write: bool = False
    ) -> NamespaceAwareSerde:
        """Build a ``NamespaceAwareSerde`` that reads every event class of
        the topology.

        The serde reads the namespaces of the topology, and each event
        outside every namespace that a handler subscribes to or returns. It
        collects each ``@migrate_from`` and ``@backfill`` on those classes.
        ``EventGraph.from_namespaces(..., checkpointer=...)`` wires the same
        serde. *migrations* and *legacy_write* go to the serde unchanged::

            saver.serde = topology.serde()
            store = EventStore(topology, saver)
        """
        from langgraph_events.serde import NamespaceAwareSerde  # noqa: PLC0415

        namespaces, events = self._serde_scope()
        return NamespaceAwareSerde(
            migrations,
            namespaces=namespaces,
            events=events,
            legacy_write=legacy_write,
        )

    def namespaces(self) -> NamespaceModel:
        """Return a :class:`NamespaceModel` — the code-derived snapshot.

        One artifact covers the full picture: domains, commands, outcomes,
        command handlers, policies, event-to-event edges, and seed events.
        Render it via :meth:`NamespaceModel.text`, :meth:`NamespaceModel.mermaid`
        (with ``view="structure"`` or ``view="choreography"``),
        :meth:`NamespaceModel.json`, or read the data attributes directly.

        Built once and cached — handler metadata is immutable after
        construction, and per-dispatch ``Reflection`` injection reads it.
        """
        if self._namespaces_cache is None:  # class-level default; set per instance
            self._namespaces_cache = NamespaceModel._build(
                self._handler_metas, self._return_info, self._reducers.values()
            )
        return self._namespaces_cache

    @property
    def reducers(self) -> Mapping[str, BaseReducer]:
        """Each registered reducer by its name. Read-only."""
        return types.MappingProxyType(self._reducers)

    @property
    def reducer_names(self) -> frozenset[str]:
        """The names of all registered reducers."""
        return frozenset(self._reducers.keys())

    @property
    def handler_names(self) -> frozenset[str]:
        """Canonical graph-node name of every registered handler.

        These are the names an interrupted checkpoint can pause at;
        ``@on(previously=...)`` aliases are excluded (use them to *cover* a
        renamed name, not to enumerate live handlers). For inline
        ``Command.handle()`` handlers this is the command's ``__qualname__``
        (stable, order-independent), not the method name.
        """
        return frozenset(meta.node_name for meta in self._handler_metas)

    @property
    def resumable_node_names(self) -> frozenset[str]:
        """Every node name a paused checkpoint can resume at: each name in
        :attr:`handler_names` and each ``@on(previously=...)`` alias."""
        return self.handler_names | {
            alias for meta in self._handler_metas for alias in meta.previous_names
        }

    @property
    def requirements(self) -> tuple[ServiceRequirement, ...]:
        """Each handler parameter that only a service can fill, in handler
        and signature order. A variadic parameter is not listed."""
        return tuple(r for meta in self._handler_metas for r in meta.requirements)

    def _bind(self, services: Services) -> tuple[BoundHandler, ...]:
        """Bind *services* to each handler, in handler order.

        Raises ``TypeError`` for a parameter that no source fills, and for a
        name-keyed service that does not satisfy the parameter annotation.
        """
        bound_handlers = []
        for meta in self._handler_metas:
            bound = meta.bind(services)
            bound.require_claimed(self.reducer_names)
            if isinstance(services, NamedServices):
                services.verify(bound)
            bound_handlers.append(bound)
        return tuple(bound_handlers)

    def _update_reducer_state(
        self,
        state: dict[str, Any],
        event: Event,
        reducer_names: list[str],
    ) -> frozenset[str]:
        """Incrementally update reducer state with a single new event."""
        changed: set[str] = set()
        for name in reducer_names:
            r = self._reducers[name]
            contribution = r.collect([event])
            if r.has_contributions(contribution):
                reducer_fn = getattr(r, "reducer", None)
                if callable(reducer_fn):
                    state[name] = reducer_fn(state[name], contribution)
                else:
                    state[name] = contribution
                changed.add(name)
        return frozenset(changed)

    def _resolve_reducer_names(self, include_reducers: bool | list[str]) -> list[str]:
        """Return reducer names to include, or empty list for disabled."""
        if include_reducers is True:
            return list(self._reducers.keys())
        if include_reducers:  # non-empty list
            unknown = set(include_reducers) - set(self._reducers.keys())
            if unknown:
                warn_user(
                    f"Unknown reducer name(s) {unknown} in include_reducers; "
                    f"available: {set(self._reducers.keys())}",
                )
            return [n for n in include_reducers if n in self._reducers]
        return []

    def _handler_node(self, bound: BoundHandler) -> RunnableLambda:
        """The node that runs *bound* with its services."""
        return make_handler_node(
            bound,
            reducers=self._reducers,
            return_contract=self._return_contracts.get(bound.meta.name),
            model_provider=self.namespaces,
        )

    def _compile(
        self,
        nodes: Sequence[tuple[HandlerMeta, RunnableLambda]],
        *,
        max_rounds: int,
        recursion_limit: int | None,
        checkpointer: Checkpointer,
        store: BaseStore | None,
    ) -> CompiledStateGraph:
        """Compile the topology into a LangGraph ``CompiledStateGraph``.

        *nodes* pairs each handler with its node, in handler order.
        :class:`EventGraph` passes the nodes of its bound handlers.
        :class:`EventStore` passes nodes that refuse to run, because a store
        has no services.
        """
        reducers = self._reducers
        handler_metas = self._handler_metas
        # Dynamic state schema with per-reducer channels
        state_schema = build_state_schema(reducers)

        # Always include reducer channels — filtering is an output concern
        out_schema: Any = _OutputState
        if reducers:
            reducer_fields: dict[str, Any] = {
                "events": list[Event],
                "causes": list[CauseEntry],
            }
            for name, r in reducers.items():
                reducer_fields[name] = r.output_type()
            _OutputWithReducers = TypedDict("_OutputWithReducers", reducer_fields)  # type: ignore[misc]
            out_schema = _OutputWithReducers

        graph: StateGraph[Any] = StateGraph(
            state_schema,
            input_schema=_InputState,  # type: ignore[arg-type]
            output_schema=out_schema,
        )

        # --- nodes ---
        seed_node = make_seed_node(reducers=reducers)
        router_node = make_router_node(max_rounds)
        dispatch_fn = make_dispatch(handler_metas)

        async def aseed(state: StateDict) -> StateDict:
            return seed_node(state)

        async def arouter(state: StateDict, config: RunnableConfig) -> StateDict:
            return router_node(state, config)

        graph.add_node(
            "__seed__", cast("Any", _leaf_node(seed_node, aseed, "__seed__"))
        )
        graph.add_node(
            "__router__", cast("Any", _leaf_node(router_node, arouter, "__router__"))
        )

        handler_names: list[str] = []
        for meta, handler_node in nodes:
            graph.add_node(meta.node_name, cast("Any", handler_node))
            handler_names.append(meta.node_name)
            # Register an alias node per historic name (@on(previously=...)) so an
            # interrupted checkpoint paused at the old node re-enters the same
            # handler on resume. The dispatcher only ever returns canonical
            # names, so aliases never fire for new work — they exist purely to
            # catch resumes of in-flight checkpoints.
            for alias in meta.previous_names:
                graph.add_node(alias, cast("Any", handler_node))
                handler_names.append(alias)

        # --- edges ---
        graph.add_edge(START, "__seed__")

        # dispatch from seed and from router
        destinations = [*handler_names, END]
        graph.add_conditional_edges("__seed__", dispatch_fn, destinations)
        graph.add_conditional_edges("__router__", dispatch_fn, destinations)

        # all handlers fan-in back to router
        for name in handler_names:
            graph.add_edge(name, "__router__")

        compile_kwargs: dict[str, Any] = {}
        if checkpointer is not None:
            compile_kwargs["checkpointer"] = checkpointer
        if store is not None:
            compile_kwargs["store"] = store
        compiled = graph.compile(**compile_kwargs)

        # Resolve recursion_limit: explicit kwarg wins; otherwise auto-size
        # so LangGraph's limit never trips before our max_rounds does.
        # Each round is at most 1 router + all handlers.
        if recursion_limit is not None:
            limit = recursion_limit
        else:
            n_handlers = len(handler_metas) + sum(
                len(meta.previous_names) for meta in handler_metas
            )
            needed = max_rounds * (n_handlers + 1) + 1
            existing = (compiled.config or {}).get("recursion_limit", 25)
            limit = max(needed, existing)
        compiled.config = {
            **(compiled.config or {}),
            "recursion_limit": limit,
        }

        return compiled

    def _register_produced_types(self) -> tuple[type[Event], ...]:
        """Fold handlers' return types into the graph's namespace registry, warn
        about events produced but never consumed, and return the event types this
        graph touches that belong to no namespace.

        Those loose types — module-level ``IntegrationEvent``s, framework
        ``SystemEvent``s — are what a serde needs in ``events=`` to keep them out
        of import-resolution. Computed here because the subscribed/produced split
        is already in hand; building a ``NamespaceModel`` to recover them would
        fire its design-smell warnings from inside the library and populate the
        model cache, so a later ``graph.namespaces()`` would never warn at all.

        Both jobs need the same subscribed/produced split, and both are only
        possible once return-type introspection has run. Registering here completes
        the registry: a namespace reachable only through a handler's return type is
        as much part of this graph as a subscribed one, and just as able to collide.
        """
        subscribed: set[type[Event]] = set()
        for meta in self._handler_metas:
            subscribed.update(meta.event_types)
        produced: set[type[Event]] = set()
        for info in self._return_info.values():
            produced.update(info.event_types)
            produced.update(info.scatter_types)

        _register_graph_namespaces(produced, self._namespaces)

        orphaned = {
            t
            for t in produced
            if not issubclass(t, (Halted, Interrupted))
            # A DomainEvent owned by a Namespace or Command — as an outcome
            # (__command__ set) or as a free-standing domain fact
            # (__namespace__ set) — is a terminal by design; having no
            # subscriber is idiomatic, not an orphan.
            and not (
                issubclass(t, DomainEvent)
                and (
                    getattr(t, "__command__", None) is not None
                    or getattr(t, "__namespace__", None) is not None
                )
            )
            # A catch-all ``@on(Event)`` observer does not process an event.
            and not any(issubclass(t, s) for s in subscribed if s is not Event)
        }
        if orphaned:
            names = ", ".join(sorted(t.__name__ for t in orphaned))
            warn_user(
                f"Event type(s) {names} are returned by handlers but no handler "
                f"subscribes to them. These events will be produced but never "
                f"processed.",
                OrphanedEventWarning,
            )

        # ``subscribed``/``produced`` are sets, so their iteration order varies
        # per process. It reaches migration collection, and from there decides
        # which class a collision diagnostic names first — keep it reproducible.
        return tuple(
            dict.fromkeys(
                t
                for t in sorted(
                    (*subscribed, *produced),
                    key=lambda c: (c.__module__, c.__qualname__),
                )
                # A mixin that a handler subscribes to is not a stored class.
                if issubclass(t, Event)
                and getattr(t, "__namespace_cls__", None) is None
            )
        )

    def _validate_node_names(self) -> None:
        """Raise if two handlers resolve to the same graph-node identity.

        ``node_name`` uniqueness is the invariant interrupted checkpoints depend on:
        a paused thread resumes into the node of that name, so two handlers sharing
        one is a silent mis-dispatch. Display-name collisions are deduplicated
        positionally, but ``node_name`` is not (an inline command's qualname or an
        ``@on(node_name=...)`` pin is taken as-is) — so guard it explicitly and fail
        at build time with a clear message rather than an opaque LangGraph
        ``add_node`` error at compile. The usual trigger is the same command (or two
        handlers pinned to the same name) registered twice in ``handlers=[...]``.
        """
        seen: dict[str, HandlerMeta] = {}
        for meta in self._handler_metas:
            if meta.node_name in _RESERVED_NODE_NAMES:
                # Reachable via an explicit @on(node_name=...) pin or a function
                # literally named after the reserved node (node names come from
                # fn.__name__).
                raise ValueError(
                    f"{meta.blame} resolves to a graph node name that is "
                    f"reserved for framework graph nodes; choose another name."
                )
            prior = seen.get(meta.node_name)
            if prior is not None:
                raise ValueError(
                    f"{prior.blame} and {meta.blame} resolve to one graph node; "
                    f"node identity must be unique. "
                    f"Register each command/handler once, or give one a distinct "
                    f"@on(node_name=...)."
                )
            seen[meta.node_name] = meta

    def _validate_handler_aliases(self) -> None:
        """Raise if any ``@on(previously=...)`` alias is unusable.

        Each historic node name becomes a real graph node, so it must be unique
        and must not shadow a live handler node — surfaced at build time.
        """
        live_names = {meta.node_name for meta in self._handler_metas}
        claimed: dict[str, HandlerMeta] = {}
        for meta in self._handler_metas:
            for alias in meta.previous_names:
                if alias in _RESERVED_NODE_NAMES:
                    raise ValueError(
                        f"{meta.blame} declares "
                        f"previously={alias!r}, which is reserved for the "
                        f"framework's own graph nodes and can never have been "
                        f"a historic handler name; remove it."
                    )
                if alias in live_names:
                    raise ValueError(
                        f"{meta.blame} declares "
                        f"previously={alias!r}, which collides with a live "
                        f"handler node of the same name; choose a distinct "
                        f"historic name."
                    )
                first = claimed.get(alias)
                if first is not None:
                    raise ValueError(
                        f"Historic node name {alias!r} is declared as previously= "
                        f"by both {first.blame} and {meta.blame}; each historic "
                        f"node name may map to only one handler."
                    )
                claimed[alias] = meta

    def _verify_error_handling(self) -> None:
        """Run the construction-time error-handling gates.

        Grouped to keep ``__init__`` readable. The three checks read disjoint
        state, so the order does not change whether a graph is accepted — it
        only decides which error surfaces first when a handler trips more
        than one gate.
        """
        self._verify_raises_coverage()
        self._verify_retry_policies()
        self._verify_invariants_coverage()

    def _verify_raises_coverage(self) -> None:
        """Ensure every declared ``raises=`` entry has a matching catcher.

        A catcher is a handler subscribed to ``HandlerRaised``. Its coverage:
        - no field matchers at all → covers any raise
        - ``exception=X`` as the only field matcher → covers any ``exc_type``
          with ``issubclass(exc_type, X)``

        Catchers with any non-``exception`` field matcher (e.g.
        ``source_event=SomeType``) are conservatively ignored for coverage,
        because such a matcher can silently exclude legitimate raises at
        runtime. The user must either drop the extra filter or add a broader
        catcher.

        Raises ``TypeError`` at compile time (first ``invoke()`` / ``compile()``)
        if any declared exception is uncovered.
        """
        catchers: list[tuple[HandlerMeta, type[Exception] | None]] = []
        for meta in self._handler_metas:
            if HandlerRaised not in meta.event_types:
                continue
            # A catcher counts only if its sole field matcher is
            # exception=<ExceptionType>. Any other field matcher, or a
            # non-type (str) matcher on exception=, is conservatively ignored.
            type_exception_match = None
            has_other = False
            for fname, matcher, is_type in meta.field_matchers:
                if fname == "exception" and is_type:
                    type_exception_match = cast("type[Exception]", matcher)
                else:
                    has_other = True
            if has_other:
                continue
            catchers.append((meta, type_exception_match))

        for meta in self._handler_metas:
            for exc_type in meta.raises:
                if not _any_catcher_covers(catchers, exc_type):
                    declared = ", ".join(t.__name__ for t in meta.raises)
                    raise TypeError(
                        f"{meta.blame} declares raises=({declared}), "
                        f"but no handler subscribes to catch {exc_type.__name__}. "
                        f"Add a handler decorated with "
                        f"@on(HandlerRaised, exception={exc_type.__name__}) "
                        f"(or @on(HandlerRaised) to catch all), or remove the "
                        f"type from raises=. Note: catchers with non-exception "
                        f"field matchers (e.g. source_event=SomeType) do not "
                        f"count toward coverage."
                    )

    def _verify_retry_policies(self) -> None:
        """Ensure every declared ``retry=`` policy can actually fire.

        A ``RetryPolicy`` only ever sees exceptions the framework catches, and
        the framework only catches what ``raises=`` declares. So two shapes are
        dead on arrival and are rejected at construction rather than silently
        doing nothing at runtime:

        - ``retry=`` with an empty ``raises=`` — nothing is ever caught.
        - an ``on=`` entry *disjoint from* ``raises=`` — that exception is
          never caught, so the policy is never consulted for it. Overlap in
          either direction is live; see the comment on the check itself.

        Raises ``TypeError``.
        """
        for meta in self._handler_metas:
            policy = meta.retry
            if policy is None:
                continue
            if not meta.raises:
                raise TypeError(
                    f"{meta.blame} declares retry= but no raises=. A "
                    f"retry policy only sees exceptions the framework catches; "
                    f"add raises=(YourError,) or drop the policy."
                )
            for exc_type in policy.on:
                # Scope is decided at runtime by ``isinstance(exc, policy.on)``,
                # so an ``on=`` entry overlapping ``raises=`` in *either*
                # direction is live: on=(OSError,) genuinely retries a declared
                # ConnectionResetError. Only a disjoint entry is dead.
                overlaps = issubclass(exc_type, meta.raises) or any(
                    issubclass(declared, exc_type) for declared in meta.raises
                )
                if not overlaps:
                    declared = ", ".join(t.__name__ for t in meta.raises)
                    raise TypeError(
                        f"{meta.blame} declares "
                        f"retry=RetryPolicy(on={exc_type.__name__}), but "
                        f"{exc_type.__name__} is not covered by "
                        f"raises=({declared}). An exception unrelated to "
                        f"raises= is never caught, so the policy can never "
                        f"retry it — add it to raises= or remove it from on=."
                    )

    def _verify_invariants_coverage(self) -> None:
        """Ensure every ``@on(InvariantViolated, invariant=X)`` matcher
        references an ``X`` declared by some handler's ``invariants=``.

        Unlike ``raises=``, a missing reactor is fine — unhandled violations
        just land in the log. The failure mode this check prevents is the
        reverse: a reactor pinned to an invariant class that no one declares
        is dead code (same silent no-op the old string API had).

        Raises ``TypeError`` at compile time if any pinned reactor references
        an undeclared invariant class.
        """
        declared: set[type[Invariant]] = set()
        for meta in self._handler_metas:
            for inv_cls, _pred in meta.invariants:
                declared.add(inv_cls)

        for meta in self._handler_metas:
            if InvariantViolated not in meta.event_types:
                continue
            for fname, matcher, is_type in meta.field_matchers:
                if fname != "invariant" or not is_type:
                    continue
                inv_cls = cast("type[Invariant]", matcher)
                if inv_cls not in declared:
                    raise TypeError(
                        f"{meta.blame} subscribes to "
                        f"@on(InvariantViolated, invariant={inv_cls.__name__}), "
                        f"but no handler declares {inv_cls.__name__} in "
                        f"invariants=. The reactor would never fire — add the "
                        f"invariant to some handler's invariants= dict, or "
                        f"remove the matcher."
                    )
