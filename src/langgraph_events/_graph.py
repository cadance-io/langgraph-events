"""EventGraph — the main entry point for building event-driven graphs."""

from __future__ import annotations

import contextlib
import dataclasses
import inspect
import types
import typing
from collections.abc import Mapping as _Mapping
from typing import TYPE_CHECKING, Any, Literal, NamedTuple, TypedDict, cast

from langgraph.graph import END, START, StateGraph
from langgraph.types import Command as LGCommand
from langgraph.types import StateUpdate

from langgraph_events._custom_event import STATE_SNAPSHOT_EVENT_NAME
from langgraph_events._event import (
    OUTCOMES_ATTR,
    Abandoned,
    Command,
    DomainEvent,
    Event,
    Halted,
    HandlerRaised,
    IntegrationEvent,
    Interrupted,
    Invariant,
    InvariantViolated,
    Namespace,
    Scatter,
    SystemEvent,
    Unresumable,
    _iter_nested_outcomes,
)
from langgraph_events._event_log import EventLog
from langgraph_events._handler import (
    HandlerMeta,
    _resolve_type_hints,
    extract_handler_meta,
    normalize_previous_names,
    on,
)
from langgraph_events._identity import command_identity
from langgraph_events._internal import (
    _BASE_FIELDS,
    _inject_deadline_keys,
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
from langgraph_events._warn import warn_user

if TYPE_CHECKING:
    from collections.abc import (
        AsyncIterator,
        Callable,
        Iterable,
        Iterator,
        Mapping,
        Sequence,
    )

    from langchain_core.runnables import RunnableConfig
    from langgraph.checkpoint.base import CheckpointTuple
    from langgraph.graph.state import CompiledStateGraph
    from langgraph.store.base import BaseStore
    from langgraph.types import StateSnapshot

    from langgraph_events._reducer import BaseReducer
    from langgraph_events._reflection import Reflection
    from langgraph_events._types import StateDict


class OrphanedEventWarning(UserWarning):
    """Issued when handler return types have no matching subscriber."""


class UnresumableError(RuntimeError):
    """Raised by ``resume()`` when the thread is not awaiting input.

    Default behavior of ``EventGraph(on_unresumable="raise")``: the paused
    handler was renamed/removed, the thread already completed, or it was
    resumed twice — in every case the resume would have been a silent no-op.
    Declare ``@on(previously=...)`` to recover a renamed handler, or set
    ``on_unresumable="halt"``/``"warn"`` to handle it non-fatally.
    """


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


_ABSTRACT_EVENT_BASES: tuple[type, ...] = (
    Event,
    DomainEvent,
    IntegrationEvent,
    SystemEvent,
)


def _scatter_event_types(scatter_alias: Any) -> list[type[Event]]:
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


def _format_scatter_repr(scatter_alias: Any) -> str:
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


def _raise_empty_scatter(fn: Callable[..., Any], offending: str) -> None:
    """Reject a Scatter annotation that contributes no concrete event types.

    Bare ``Scatter``, ``Scatter[Any]``, ``Scatter[TypeVar]``, and
    ``Scatter[<abstract base>]`` (Event/DomainEvent/IntegrationEvent/
    SystemEvent) all bypass build-time privacy enforcement because the
    resolved target set is empty or non-discriminating. Raise at build time
    so users can't paper over a privacy violation by widening the annotation.
    """
    handler_name = getattr(fn, "__name__", "handler")
    cmd: type[Command] | None = getattr(fn, "_inline_command", None)
    if cmd is not None:
        nested = _iter_nested_outcomes(cmd)
        suggested = (
            " | ".join(o.__name__ for o in nested) if nested else "EventA | EventB"
        )
        location = f"Inline handler {handler_name!r} on {cmd.__qualname__}"
    else:
        suggested = "EventA | EventB"
        location = f"Handler {handler_name!r}"
    raise TypeError(
        f"{location} returns {offending}, which declares no concrete event "
        f"types and bypasses build-time privacy enforcement. "
        f"Use `Scatter[{suggested}]` to enumerate the events you scatter. "
        f"(Do not work around this by demoting your Command to a "
        f"DomainEvent/IntegrationEvent — that loses privacy and outcome "
        f"guarantees.)"
    )


def _parse_return_types(fn: Callable[..., Any]) -> ReturnInfo:
    """Parse handler return annotation into a ``ReturnInfo``.

    Raises ``TypeError`` for Scatter shapes that contribute no concrete event
    types — see :func:`_raise_empty_scatter`.
    """
    try:
        hints = _resolve_type_hints(fn)
    except Exception as exc:
        warn_user(
            f"Failed to resolve return type hints for handler {fn.__qualname__!r}; "
            f"treating as unannotated for topology parsing. ({exc})",
        )
        hints = {}

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
            _raise_empty_scatter(fn, "bare `Scatter`")
        elif typing.get_origin(arg) is Scatter:
            members = _scatter_event_types(arg)
            if not members or any(m in _ABSTRACT_EVENT_BASES for m in members):
                _raise_empty_scatter(fn, f"`{_format_scatter_repr(arg)}`")
            scatter_types.extend(members)
        elif isinstance(arg, type) and issubclass(arg, Event):
            if issubclass(arg, Interrupted):
                has_interrupted = True
            event_types.append(arg)

    return ReturnInfo(event_types, scatter_types, has_interrupted, True)


class GraphState(NamedTuple):
    """Event-focused snapshot of a checkpointed thread."""

    events: EventLog
    is_interrupted: bool
    interrupted: Interrupted | None


class _PendingInterrupts(NamedTuple):
    """Topology-independent read of a checkpoint's pending interrupt(s).
    See ``EventGraph._read_pending_interrupts``.

    ``has_interrupt`` is ``True`` for any pending ``__interrupt__``
    write, including a non-``Event`` payload (e.g. a raw
    ``langgraph.types.interrupt("...")`` call) or an unrevivable one:
    what ``require_interrupt`` and ``event_type=None`` discovery gate
    on. ``events`` narrows that set to successfully-revived ``Event``
    payloads, deduped by qualname: what a class filter matches against.
    ``unresolved_names`` holds the recorded qualname of every interrupt
    whose class no longer imports, deduped in discovery order: no class
    to check, so a filter can never match one, but ``abandon()`` still
    folds them into ``discarded``.
    """

    has_interrupt: bool
    events: list[Event]
    unresolved_names: list[str]


class StreamFrame(NamedTuple):
    """A yielded frame from ``stream_events()`` when ``include_reducers`` is enabled.

    Contains the event and a snapshot of all requested reducer values at
    the point the event was produced.
    """

    event: Event
    reducers: dict[str, list[Any]]
    changed_reducers: frozenset[str] | None = None


class LLMToken(NamedTuple):
    """A text delta from an LLM invocation during handler execution."""

    run_id: str
    content: str


class LLMStreamEnd(NamedTuple):
    """Signals an LLM stream completed."""

    run_id: str
    message_id: str | None  # LangChain AIMessage.id for dedup


class LLMToolCallChunk(NamedTuple):
    """A tool-call chunk from an LLM stream.

    First chunk for a given ``call_index`` carries ``tool_call_id`` and
    ``name``; subsequent chunks may have those as empty strings and only
    carry ``args_delta`` (a partial JSON string). ``call_index`` is
    LangChain's ``tool_call_chunks[i].index`` — it groups chunks that
    belong to the same call when the LLM emits multiple in parallel.
    """

    run_id: str
    call_index: int
    tool_call_id: str
    name: str
    args_delta: str


class CustomEventFrame(NamedTuple):
    """A custom event frame emitted from LangGraph's v2 stream API."""

    name: str
    data: Any


class StateSnapshotFrame(NamedTuple):
    """A typed state snapshot custom frame emitted from v2 stream events."""

    data: dict[str, Any]


StreamItem = (
    Event
    | StreamFrame
    | LLMToken
    | LLMStreamEnd
    | LLMToolCallChunk
    | CustomEventFrame
    | StateSnapshotFrame
)


def _coerce_snapshot_data(data: Any) -> dict[str, Any]:
    if isinstance(data, dict):
        return cast("dict[str, Any]", data)
    return {}


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
    handler_name = getattr(meta.fn, "__name__", "handler")
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
        f"Inline handler {handler_name!r} on {cmd.__qualname__} declares "
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


def _register_produced_types(
    handler_metas: list[HandlerMeta],
    return_info: dict[str, ReturnInfo],
    namespaces: dict[str, type[Namespace]],
) -> tuple[type[Event], ...]:
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
    for meta in handler_metas:
        subscribed.update(meta.event_types)
    produced: set[type[Event]] = set()
    for info in return_info.values():
        produced.update(info.event_types)
        produced.update(info.scatter_types)

    _register_graph_namespaces(produced, namespaces)

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
        and not any(issubclass(t, s) for s in subscribed)
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
            if getattr(t, "__namespace_cls__", None) is None
        )
    )


def _collect_graph_namespaces(
    handlers: list[Callable[..., Any]],
) -> dict[str, type[Namespace]]:
    """Name → Namespace class for the namespaces this graph's handlers subscribe to.

    The graph's own registry, seeded from the ``__namespace_cls__`` stamps
    rather than from process-global state. Produced types are folded in later
    in ``EventGraph.__init__``, once return-type introspection has run.
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


def _build_service_registries(
    services: Sequence[Any] | _Mapping[str, Any] | None,
) -> tuple[dict[type, Any], dict[str, Any]]:
    """Split ``services=`` into type-keyed and name-keyed lookups.

    Mapping form populates name-keyed; sequence form populates type-keyed
    (rejecting same-type collisions). The two forms are mutually exclusive
    per ``EventGraph`` instance.
    """
    by_type: dict[type, Any] = {}
    by_name: dict[str, Any] = {}
    if isinstance(services, _Mapping):
        by_name = dict(services)
        return by_type, by_name
    if services is None:
        return by_type, by_name
    for s in services:
        t = type(s)
        if t in by_type:
            raise TypeError(
                f"EventGraph(services=...) has two instances of type "
                f"{t.__name__!r} — same-type collision. Register only "
                f"one instance, switch to the name-keyed mapping form "
                f"(services={{'primary': a, 'backup': b}}), or use a "
                f"subclass to distinguish them."
            )
        by_type[t] = s
    return by_type, by_name


def _validate_on_unresumable(value: str) -> None:
    """Reject an unknown ``on_unresumable`` policy at construction.

    The value set intentionally diverges from ``AGUIAdapter(on_unmapped=...)``:
    the default is ``"raise"`` (not ``"warn"``) because a non-resumable resume
    is a real bug worth failing on, and the third option is ``"halt"`` (emit a
    terminal event) rather than ``"ignore"`` (silently drop) — there is no
    silent option here on purpose.
    """
    if value not in ("raise", "halt", "warn"):
        raise ValueError(
            f"on_unresumable must be 'raise', 'halt', or 'warn', got {value!r}"
        )


_RESERVED_NODE_NAMES = frozenset({"__seed__", "__router__", "__start__", "__end__"})
"""Graph-node names no handler or ``previously=`` alias may claim: the
framework's own pregel nodes (``__seed__``/``__router__``, registered in the
graph build) and LangGraph's reserved endpoints (``__start__``/``__end__``).
Without this check the collision surfaces only at first compile/invoke —
as an opaque "node already present" for the framework pair, and as a
claimant-less "reserved" error for LangGraph's — never naming the
declaration that smuggled the name in."""


def _validate_handler_metas(metas: list[HandlerMeta]) -> None:
    """Run all build-time handler-identity checks (node names, then aliases)."""
    _validate_node_names(metas)
    _validate_handler_aliases(metas)


def _validate_node_names(metas: list[HandlerMeta]) -> None:
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
    for meta in metas:
        if meta.node_name in _RESERVED_NODE_NAMES:
            # Reachable via an explicit @on(node_name=...) pin or a function
            # literally named after the reserved node (node names come from
            # fn.__name__) — so name the claimant by its qualname; the node
            # name itself IS the reserved string and would point at nothing.
            raise ValueError(
                f"Handler {meta.fn.__qualname__!r} resolves to graph node "
                f"{meta.node_name!r}, which is reserved for framework graph "
                f"nodes; choose another name."
            )
        prior = seen.get(meta.node_name)
        if prior is not None:
            raise ValueError(
                f"Handlers {prior.name!r} and {meta.name!r} both resolve to "
                f"graph node {meta.node_name!r}; node identity must be unique. "
                f"Register each command/handler once, or give one a distinct "
                f"@on(node_name=...)."
            )
        seen[meta.node_name] = meta


def _validate_handler_aliases(metas: list[HandlerMeta]) -> None:
    """Raise if any ``@on(previously=...)`` alias is unusable.

    Each historic node name becomes a real graph node, so it must be unique
    and must not shadow a live handler node — surfaced at build time.
    """
    live_names = {meta.node_name for meta in metas}
    claimed: dict[str, str] = {}
    for meta in metas:
        for alias in meta.previous_names:
            if alias in _RESERVED_NODE_NAMES:
                raise ValueError(
                    f"Handler {meta.node_name!r} declares "
                    f"previously={alias!r}, which is reserved for the "
                    f"framework's own graph nodes and can never have been "
                    f"a historic handler name; remove it."
                )
            if alias in live_names:
                raise ValueError(
                    f"Handler {meta.node_name!r} declares "
                    f"previously={alias!r}, which collides with a live "
                    f"handler node of the same name; choose a distinct "
                    f"historic name."
                )
            first = claimed.get(alias)
            if first is not None:
                raise ValueError(
                    f"Historic node name {alias!r} is declared as previously= "
                    f"by both {first!r} and {meta.node_name!r}; each historic "
                    f"node name may map to only one handler."
                )
            claimed[alias] = meta.node_name


def _verify_no_unclaimed_params(meta: HandlerMeta) -> None:
    """Raise if a handler declares a param the framework cannot inject.

    Every parameter must be claimed by exactly one source: the event itself
    (the first positional param), ``EventLog``/``RunnableConfig``/``BaseStore``
    framework injectables, a registered reducer name, a field matcher, or a
    type-matched service. An unclaimed param means the user intended an
    injection the framework has no way to provide — surface it at graph
    build time so the failure is colocated with the misconfiguration.
    """
    sig = inspect.signature(meta.fn)
    first_param = next(iter(sig.parameters), None)
    claimed: set[str | None] = {first_param, "self"}
    claimed.update(meta.framework_params)
    claimed.update(meta.reducer_params)
    claimed.update(meta.field_inject_params)
    claimed.update(name for name, _ in meta.service_params)
    claimed.update(name for name, _ in meta.service_name_params)
    # Variadic params (``*args`` / ``**kwargs``) cannot be claimed by any
    # injection source; they are caller-controlled. Treat them as claimed
    # so generic catcher-style handlers (e.g. ``def react(event, *a, **kw)``)
    # build cleanly.
    variadic = {
        inspect.Parameter.VAR_POSITIONAL,
        inspect.Parameter.VAR_KEYWORD,
    }
    unclaimed = [
        name
        for name, p in sig.parameters.items()
        if name not in claimed and p.kind not in variadic
    ]
    if unclaimed:
        raise TypeError(
            f"Handler {meta.name!r} declares parameter(s) {unclaimed} that "
            f"the framework cannot inject. For service injection, register "
            f"a matching instance via EventGraph(services=[...]); for state, "
            f"register a Reducer; otherwise remove the parameter."
        )


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
    invariants=..., previously=...)(fn)`` so that ``extract_handler_meta``
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


class EventGraph:
    """Build and run an event-driven graph from ``@on`` handlers.

    Topology is auto-derived from handler subscriptions.  Internally builds
    a LangGraph ``StateGraph`` with a hub-and-spoke reactive loop.

    Accepts a single seed event or a list of seed events::

        graph = EventGraph([classify, route, review])

        # Single seed event
        log = graph.invoke(DocumentReceived(doc_id="1", content="..."))

        # Multiple seed events (e.g. system prompt + user message)
        log = graph.invoke([
            SystemPromptSet.from_str("You are helpful"),
            UserMessageReceived(message=HumanMessage(content="Hi")),
        ])
    """

    # Lazily built by namespaces(); class-level default keeps __init__ lean.
    _namespaces_cache: NamespaceModel | None = None

    def __init__(
        self,
        handlers: list[Callable[..., Any]],
        *,
        max_rounds: int = 100,
        reducers: list[BaseReducer] | None = None,
        services: Sequence[Any] | Mapping[str, Any] | None = None,
        checkpointer: Any = None,
        store: BaseStore | None = None,
        recursion_limit: int | None = None,
        on_unresumable: Literal["raise", "halt", "warn"] = "raise",
    ) -> None:
        if not handlers:
            raise ValueError("EventGraph requires at least one handler")
        _validate_on_unresumable(on_unresumable)

        handlers = _expand_command_handlers(handlers)

        self._max_rounds = max_rounds
        self._recursion_limit = recursion_limit
        self._on_unresumable = on_unresumable
        self._checkpointer = checkpointer
        self._store = store
        self._reducers: dict[str, BaseReducer] = {r.name: r for r in (reducers or [])}
        self._namespaces = _collect_graph_namespaces(handlers)
        _discover_namespace_reducers(self._namespaces, self._reducers)
        conflicts = set(self._reducers.keys()) & set(_BASE_FIELDS.keys())
        if conflicts:
            raise ValueError(
                f"Reducer name(s) {conflicts} conflict with reserved state fields"
            )

        # services= takes one of two shapes:
        #   • Mapping[str, Any] — name-keyed registry; resolution by handler
        #     param name. Allows multiple instances of the same type
        #     (e.g. primary + backup chat models).
        #   • Sequence[Any] — type-keyed registry; resolution by handler
        #     param annotation. Same-type collisions are rejected.
        # The two forms are mutually exclusive within a single EventGraph.
        self._services_by_type, self._services_by_name = _build_service_registries(
            services
        )

        self._handler_metas: list[HandlerMeta] = []
        self._compiled_graph: CompiledStateGraph | None = None

        reducer_names = frozenset(self._reducers.keys())
        service_types = frozenset(self._services_by_type.keys())
        service_names = frozenset(self._services_by_name.keys())
        seen_names: dict[str, int] = {}
        for fn in handlers:
            meta = extract_handler_meta(
                fn,
                reducer_names=reducer_names,
                service_types=service_types,
                service_names=service_names,
            )
            _verify_no_unclaimed_params(meta)
            # Deduplicate colliding display names positionally; see
            # _dedup_handler_name for how the stable node identity is preserved.
            if meta.name in seen_names:
                seen_names[meta.name] += 1
                meta = _dedup_handler_name(meta, seen_names[meta.name])
            else:
                seen_names[meta.name] = 1
            self._handler_metas.append(meta)

        _validate_handler_metas(self._handler_metas)

        self._return_info: dict[str, ReturnInfo] = {}
        self._return_contracts: dict[str, ReturnContract | None] = {}
        for meta in self._handler_metas:
            info = _parse_return_types(meta.fn)
            self._return_info[meta.name] = info
            if info.has_annotation and any(t is Event for t in info.event_types):
                raise ValueError(
                    f"Handler '{meta.name}' return type includes base 'Event'. "
                    f"Use specific types (e.g., TypeA | TypeB)."
                )
            self._return_contracts[meta.name] = _compute_return_contract(meta, info)
            _verify_inline_outcome_coverage(meta, info)

        self._loose_events = _register_produced_types(
            self._handler_metas, self._return_info, self._namespaces
        )

        enforce_command_privacy(self._handler_metas, self._return_info)
        self._verify_error_handling()

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
                self._handler_metas, self._return_info
            )
        return self._namespaces_cache

    def reflect(self, log: EventLog) -> Reflection:
        """Return a :class:`Reflection` — deterministic query surface over *log*.

        Bundles the log with this graph's namespace model and reducers so an
        agent (or code) can query facts about a run: listings, field dumps,
        static topology, reducer projections, and verdict-free evidence joins.
        """
        from langgraph_events._reflection import Reflection  # noqa: PLC0415

        return Reflection(log, model=self.namespaces(), reducers=self._reducers)

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
    def compiled(self) -> CompiledStateGraph:
        """The underlying LangGraph ``CompiledStateGraph``.

        This is the bridge to full LangGraph when you need features
        beyond the EventGraph API: subgraph composition, custom
        streaming modes, direct state access, or advanced checkpointer
        workflows.

        The instance is compiled lazily on first access and cached.
        """
        return self._compile()

    def _compile(self) -> CompiledStateGraph:
        """Compile into a LangGraph ``CompiledStateGraph`` (internal)."""
        if self._compiled_graph is not None:
            return self._compiled_graph

        # Dynamic state schema with per-reducer channels
        state_schema = build_state_schema(self._reducers)

        # Always include reducer channels — filtering is an output concern
        out_schema: Any = _OutputState
        if self._reducers:
            reducer_fields: dict[str, Any] = {"events": list[Event]}
            for name, r in self._reducers.items():
                reducer_fields[name] = r.output_type()
            _OutputWithReducers = TypedDict("_OutputWithReducers", reducer_fields)  # type: ignore[misc]
            out_schema = _OutputWithReducers

        graph: StateGraph[Any] = StateGraph(
            state_schema,
            input_schema=_InputState,  # type: ignore[arg-type]
            output_schema=out_schema,
        )

        # --- nodes ---
        seed_node = make_seed_node(reducers=self._reducers)
        router_node = make_router_node(self._max_rounds)
        dispatch_fn = make_dispatch(self._handler_metas)

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
        for meta in self._handler_metas:
            handler_node = make_handler_node(
                meta,
                reducers=self._reducers,
                return_contract=self._return_contracts.get(meta.name),
                services_by_type=self._services_by_type or None,
                services_by_name=self._services_by_name or None,
                model_provider=self.namespaces,
            )
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
        if self._checkpointer is not None:
            compile_kwargs["checkpointer"] = self._checkpointer
        if self._store is not None:
            compile_kwargs["store"] = self._store
        self._compiled_graph = graph.compile(**compile_kwargs)

        # Resolve recursion_limit: explicit kwarg wins; otherwise auto-size
        # so LangGraph's limit never trips before our max_rounds does.
        # Each round is at most 1 router + all handlers.
        if self._recursion_limit is not None:
            limit = self._recursion_limit
        else:
            n_handlers = len(self._handler_metas) + sum(
                len(meta.previous_names) for meta in self._handler_metas
            )
            needed = self._max_rounds * (n_handlers + 1) + 1
            existing = (self._compiled_graph.config or {}).get("recursion_limit", 25)
            limit = max(needed, existing)
        self._compiled_graph.config = {
            **(self._compiled_graph.config or {}),
            "recursion_limit": limit,
        }

        return self._compiled_graph

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
                        f"Handler {meta.claimant!r} declares raises=({declared}), "
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
                    f"Handler {meta.claimant!r} declares retry= but no raises=. A "
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
                        f"Handler {meta.claimant!r} declares "
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
                        f"Handler {meta.name!r} subscribes to "
                        f"@on(InvariantViolated, invariant={inv_cls.__name__}), "
                        f"but no handler declares {inv_cls.__name__} in "
                        f"invariants=. The reactor would never fire — add the "
                        f"invariant to some handler's invariants= dict, or "
                        f"remove the matcher."
                    )

    def _require_checkpointer(self, method: str) -> None:
        if self._checkpointer is None:
            raise ValueError(f"{method}() requires a checkpointer")

    @staticmethod
    def _prepare_input(seed: Event | list[Event]) -> dict[str, Any]:
        """Build the input dict from a seed event or list of events."""
        if isinstance(seed, list):
            return {"events": seed}
        return {"events": [seed]}

    @staticmethod
    def _apply_deadline_kwarg(kwargs: dict[str, Any]) -> dict[str, Any]:
        """Pop ``deadline`` from kwargs and inject it into the LangGraph config.

        Thin wrapper over :func:`_inject_deadline_keys` that pops the kwarg
        and threads it into a copied ``config`` dict, so callers can pass
        ``deadline=...`` through any entry point
        (invoke/ainvoke/resume/aresume/stream_events/...) and the router
        sees it via parameter injection.
        """
        deadline = kwargs.pop("deadline", None)
        if deadline is None:
            return kwargs
        config = dict(kwargs.get("config") or {})
        configurable = dict(config.get("configurable", {}))
        _inject_deadline_keys(configurable, deadline)
        config["configurable"] = configurable
        kwargs["config"] = config
        return kwargs

    def _run(self, inp: Any, **kwargs: Any) -> EventLog:
        kwargs = self._apply_deadline_kwarg(kwargs)
        compiled = self._compile()
        result = compiled.invoke(inp, **kwargs)
        return EventLog._from_owned(result["events"])

    async def _arun(self, inp: Any, **kwargs: Any) -> EventLog:
        kwargs = self._apply_deadline_kwarg(kwargs)
        compiled = self._compile()
        result = await compiled.ainvoke(inp, **kwargs)
        return EventLog._from_owned(result["events"])

    @classmethod
    def from_namespaces(
        cls,
        *domains: type[Namespace],
        handlers: list[Callable[..., Any]] | None = None,
        **kwargs: Any,
    ) -> EventGraph:
        """Build an ``EventGraph`` from domains' inline command handlers.

        Walks each domain's class namespace and registers every ``Command``
        that defines an inline handler. Commands without one are silently
        skipped — register those via the ``handlers=`` kwarg or
        ``EventGraph([...])`` directly, which errors on missing handlers.

        The ``handlers=`` kwarg is appended as-is — useful for reaction
        handlers subscribed to ``DomainEvent``s, ``HandlerRaised``,
        ``InvariantViolated``, etc.

        Example::

            graph = EventGraph.from_namespaces(Order, Customer,
                                            handlers=[react])
        """
        collected: list[Any] = []
        for dom in domains:
            if not (isinstance(dom, type) and issubclass(dom, Namespace)):
                raise TypeError(
                    f"from_namespaces expects Namespace subclasses, got {dom!r}"
                )
            for attr in dom.__dict__.values():
                if (
                    isinstance(attr, type)
                    and issubclass(attr, Command)
                    and getattr(attr, "__command_handler__", None) is not None
                ):
                    collected.append(attr)
        if handlers:
            collected.extend(handlers)

        # Auto-wire migration-aware deserialization: the user only writes
        # @migrate_from on the class. We have both the domains and the
        # checkpointer here, so scope a NamespaceAwareSerde to these
        # namespaces. A user-supplied NamespaceAwareSerde (possibly
        # carrying hand-authored migrations=) is left untouched — that is
        # the opt-out. The serde still owns coverage; we own construction.
        graph = cls(collected, **kwargs)

        # Wired after construction, not before: the events that live outside
        # every namespace — module-level IntegrationEvents, framework
        # SystemEvents — are only knowable once the graph has parsed its
        # handlers' subscriptions and return types. Without them those
        # identities resolve by import and are shared between engine
        # lifetimes of one module. ``compiled`` is lazy, so the serde is in
        # place well before anything reads it.
        checkpointer = kwargs.get("checkpointer")
        if checkpointer is not None:
            from langgraph_events.serde import NamespaceAwareSerde  # noqa: PLC0415

            serde = getattr(checkpointer, "serde", None)
            if serde is not None and not isinstance(serde, NamespaceAwareSerde):
                # Not ``domains``: a namespace can reach the graph through
                # ``handlers=`` alone, and its events would then keep
                # resolving by import — the bleed #155 exists to close.
                # ``graph._namespaces`` is the full set.
                checkpointer.serde = NamespaceAwareSerde(
                    namespaces=tuple(graph._namespaces.values()),
                    events=graph._loose_events,
                )

        return graph

    def invoke(self, seed: Event | list[Event], **kwargs: Any) -> EventLog:
        """Run the graph synchronously with one or more seed events.

        Args:
            seed: A single event or list of events to start the graph.

        Returns an ``EventLog`` containing all events produced during the run.
        """
        return self._run(self._prepare_input(seed), **kwargs)

    async def ainvoke(self, seed: Event | list[Event], **kwargs: Any) -> EventLog:
        """Run the graph asynchronously with one or more seed events."""
        return await self._arun(self._prepare_input(seed), **kwargs)

    def pre_seed(self, config: RunnableConfig, values: dict[str, Any]) -> None:
        """Inject external state into reducer channels before the first run.

        Use this to hydrate reducers from an external source (e.g. a database
        migration or test fixture) when modelling the data as seed events isn't
        practical.  Call it once before ``invoke``/``ainvoke``::

            graph.pre_seed(config, {"my_reducer": existing_value})
            graph.invoke(StartEvent(), config=config)

        Requires a checkpointer.
        """
        self._require_checkpointer("pre_seed")
        compiled = self._compile()
        compiled.update_state(config, values, as_node="__seed__")

    async def apre_seed(self, config: RunnableConfig, values: dict[str, Any]) -> None:
        """Async version of :meth:`pre_seed`."""
        self._require_checkpointer("apre_seed")
        compiled = self._compile()
        await compiled.aupdate_state(config, values, as_node="__seed__")

    def _resume_is_pending(self, kwargs: dict[str, Any]) -> bool:
        """Whether the thread has work to resume into.

        Keyed on the checkpoint's ``next`` (scheduled nodes), not the
        higher-level ``is_interrupted`` flag: a caller may ``pre_seed`` before
        resuming (e.g. ``AGUIAdapter`` commits a ``FrontendStateMutated``),
        which clears ``is_interrupted`` while the interrupt is still pending —
        but ``next`` stays non-empty. ``next == ()`` means nothing is scheduled,
        so ``resume()`` would be a silent no-op (paused handler gone, thread
        already finished, or resumed twice). A Phase-1 ``@on(previously=...)``
        alias keeps the node live, so a declared rename still reports ``True``.
        If no ``config`` was passed, assume pending and let the normal path
        surface any error.
        """
        config = kwargs.get("config")
        if config is None:
            return True
        return bool(self._compile().get_state(config).next)

    async def _aresume_is_pending(self, kwargs: dict[str, Any]) -> bool:
        """Async sibling of :meth:`_resume_is_pending`.

        Reads the checkpoint via ``aget_state`` so the async resume entry
        points (:meth:`aresume`/:meth:`astream_resume`) never drive an
        async-only checkpointer synchronously from the running event loop.
        """
        config = kwargs.get("config")
        if config is None:
            return True
        return bool((await self._compile().aget_state(config)).next)

    def _unresumable_message(
        self, log: EventLog | None = None, config: Any = None
    ) -> str:
        """Diagnostic for a ``resume()`` that would be a no-op.

        Keys the abandoned diagnosis on the *last* event, not
        ``log.latest(Halted)`` (a whole-log search): a thread reused
        after :meth:`abandon` must fall back to the generic message once
        ``Abandoned`` is no longer latest.
        """
        if log and isinstance(log[-1], Abandoned):
            abandoned = log[-1]
            thread_id = (config or {}).get("configurable", {}).get("thread_id")
            detail = ""
            if abandoned.reason:
                detail += f" reason={abandoned.reason!r}."
            if abandoned.discarded:
                detail += f" discarded={abandoned.discarded!r}."
            return (
                f"resume() called on thread {thread_id!r}, which was "
                f"abandoned via abandon()/aabandon().{detail} The thread was "
                "deliberately settled without an answer and is terminal; it "
                "cannot be resumed."
            )
        return (
            "resume() called on a thread that is not awaiting input. The paused "
            "handler may have been renamed/removed, or the thread already "
            "completed. Declare @on(previously=...) to recover a renamed "
            "handler, or set EventGraph(on_unresumable='halt'|'warn')."
        )

    def _unresumable_short_circuits(
        self, log: EventLog | None = None, config: Any = None
    ) -> bool:
        """Apply ``on_unresumable``'s ``raise``/``warn`` arm. ``False``
        means append a terminal event (``halt``). ``True`` means return
        the log unchanged (``warn``). ``raise`` raises.

        Caller supplies the state read. The async path must ``await
        aget_state``: an async-only checkpointer rejects a sync read
        from the running loop.
        """
        if self._on_unresumable == "raise":
            raise UnresumableError(self._unresumable_message(log, config))
        if self._on_unresumable == "warn":
            warn_user(self._unresumable_message(log, config))
            return True
        return False

    @staticmethod
    def _settle_supersteps(
        events: EventLog, terminal: Event
    ) -> list[list[StateUpdate]]:
        """Build the clear/append/clear supersteps that settle a thread onto
        *terminal* at rest.

        The leading clear (``values=None, as_node=END``, LangGraph's
        clear-all-tasks branch) skips ``ERROR``/``INTERRUPT`` pending
        writes, preserving an already-completed write from a fanned-out
        sibling.

        The suite pins the leading clear and the ``_cursor`` reset, each
        with its own failing test. The trailing clear and
        ``_pending: []`` are a second guard. Neither is pinned alone.
        """
        appended = [terminal]
        return [
            [StateUpdate(None, END)],
            [
                StateUpdate(
                    {
                        "events": appended,
                        "_cursor": len(events) + len(appended),
                        "_pending": [],
                    },
                    "__seed__",
                )
            ],
            [StateUpdate(None, END)],
        ]

    def _settle(self, config: Any, terminal: Event) -> EventLog:
        """Append *terminal* and settle the thread via :meth:`_settle_supersteps`."""
        # `_cursor` is correct only because this reads via `get_state`,
        # whose snapshot applies the same pending writes the leading
        # clear commits. A direct checkpoint read undercounts `_cursor`
        # without raising an error, and no test in this suite catches it.
        events = self.get_state(config).events
        self._compile().bulk_update_state(
            cast("RunnableConfig", config),
            self._settle_supersteps(events, terminal),
        )
        return self.get_state(config).events

    async def _asettle(self, config: Any, terminal: Event) -> EventLog:
        """Async sibling of :meth:`_settle`."""
        # Same `_cursor` reasoning as `_settle`, via `aget_state`.
        events = (await self.aget_state(config)).events
        await self._compile().abulk_update_state(
            cast("RunnableConfig", config),
            self._settle_supersteps(events, terminal),
        )
        return (await self.aget_state(config)).events

    def _apply_unresumable_policy(
        self, value: Event, kwargs: dict[str, Any]
    ) -> EventLog:
        """Sync ``on_unresumable`` for a resume that would be a no-op.

        Runs only when :meth:`_resume_is_pending` is ``False``, so no
        completed sibling write can be lost on this path.
        """
        config = kwargs.get("config")
        log = self.get_state(config).events
        if self._unresumable_short_circuits(log, config):
            return log
        return self._settle(config, self._unresumable_event(value))

    async def _aapply_unresumable_policy(
        self, value: Event, kwargs: dict[str, Any]
    ) -> EventLog:
        """Async sibling of :meth:`_apply_unresumable_policy`."""
        config = kwargs.get("config")
        log = (await self.aget_state(config)).events
        if self._unresumable_short_circuits(log, config):
            return log
        return await self._asettle(config, self._unresumable_event(value))

    @staticmethod
    def _unresumable_event(value: Event) -> Unresumable:
        return Unresumable(resume_value=type(value).__name__)

    def resume(self, value: Event, **kwargs: Any) -> EventLog:
        """Resume an interrupted graph with a domain event.

        The event is auto-dispatched (handlers subscribed to its type fire),
        then a ``Resumed`` event is created alongside it. If the thread is not
        awaiting input, the ``on_unresumable`` policy applies (default raise).
        """
        self._require_checkpointer("resume")
        if not self._resume_is_pending(kwargs):
            return self._apply_unresumable_policy(value, kwargs)
        return self._run(LGCommand(resume=value), **kwargs)

    async def aresume(self, value: Event, **kwargs: Any) -> EventLog:
        """Async version of resume().

        The event is auto-dispatched (handlers subscribed to its type fire),
        then a ``Resumed`` event is created alongside it. If the thread is not
        awaiting input, the ``on_unresumable`` policy applies (default raise).
        """
        self._require_checkpointer("aresume")
        if not await self._aresume_is_pending(kwargs):
            return await self._aapply_unresumable_policy(value, kwargs)
        return await self._arun(LGCommand(resume=value), **kwargs)

    @staticmethod
    def _is_interrupted(snapshot: StateSnapshot) -> bool:
        """Whether *snapshot* is currently paused at a real interrupt.

        ``snapshot.tasks[*].interrupts`` distinguishes a real interrupt
        from a cancelled or crashed graph, which also leaves
        ``snapshot.next`` set.

        Topology-dependent: ``snapshot.tasks`` only reports a task the
        graph that produced *snapshot* can still schedule, so a thread
        paused on a handler this graph no longer registers looks
        uninterrupted here. Used only by :meth:`_graph_state`, reached
        through a caller's own working graph. Everywhere else
        (discovery, ``abandon()``) reads the checkpoint's raw pending
        writes instead — see :meth:`_pending_interrupt_writes`.
        """
        has_interrupt = any(getattr(task, "interrupts", ()) for task in snapshot.tasks)
        return bool(snapshot.next) and has_interrupt

    @staticmethod
    def _pending_interrupt_events(snapshot: StateSnapshot) -> list[Event]:
        """Event instance(s) behind every pending interrupt, deduped by
        type name, in discovery order. Empty if none.

        Scans every task, not just ``tasks[0]``: a fanned-out dispatch
        can pause more than one. Skips a non-``Event`` payload, e.g. a
        raw ``langgraph.types.interrupt("...")`` call.

        Topology-dependent, same caveat as :meth:`_is_interrupted`: used
        only by :meth:`_graph_state`'s first-pause fallback.
        """
        seen: set[str] = set()
        values: list[Event] = []
        for task in snapshot.tasks:
            for i in getattr(task, "interrupts", ()):
                value = i.value
                if isinstance(value, Event) and type(value).__name__ not in seen:
                    seen.add(type(value).__name__)
                    values.append(value)
        return values

    @staticmethod
    def _pending_interrupt_writes(pending_writes: Iterable[Any]) -> _PendingInterrupts:
        """Read every pending ``__interrupt__`` checkpoint write into a
        :class:`_PendingInterrupts`.

        Reads a checkpoint's raw ``pending_writes``, topology-
        independent unlike :meth:`_pending_interrupt_events`. A thread
        paused on a handler the current graph no longer registers still
        carries this write: ``StateSnapshot.tasks`` would not show it.
        This is the mechanism behind discovery and ``abandon()``.

        A write's value is one interrupt or a sequence of them (a
        framework detail, handled either way). A payload degraded to
        ``UnrevivedIdentity`` (only possible when the read ran inside
        ``NamespaceAwareSerde.tolerate_unresolved``) contributes its
        stored qualname to ``unresolved_names`` instead of ``events``.
        """
        from langgraph_events.serde._jsonplus import (  # noqa: PLC0415
            UnrevivedIdentity,
        )

        has_interrupt = False
        seen: set[str] = set()
        unresolved_seen: set[str] = set()
        events: list[Event] = []
        unresolved_names: list[str] = []
        for _task_id, channel, value in pending_writes:
            if channel != "__interrupt__":
                continue
            entries = value if isinstance(value, (list, tuple)) else [value]
            for entry in entries:
                has_interrupt = True
                payload = getattr(entry, "value", entry)
                if (
                    isinstance(payload, Event)
                    and type(payload).__qualname__ not in seen
                ):
                    seen.add(type(payload).__qualname__)
                    events.append(payload)
                elif (
                    isinstance(payload, UnrevivedIdentity)
                    and payload.qualname not in unresolved_seen
                ):
                    unresolved_seen.add(payload.qualname)
                    unresolved_names.append(payload.qualname)
        return _PendingInterrupts(
            has_interrupt=has_interrupt,
            events=events,
            unresolved_names=unresolved_names,
        )

    def _tolerant_read(self) -> contextlib.AbstractContextManager[None]:
        """Context manager that degrades an unrevivable interrupt
        identity instead of raising, for the checkpointer's current
        serde: a no-op for any serde other than
        :class:`~langgraph_events.serde.NamespaceAwareSerde`, the only
        one this library ships that raises ``Cannot revive`` in the
        first place.

        Used by :meth:`_read_pending_interrupts`/
        :meth:`_aread_pending_interrupts` and by
        :meth:`abandon`/:meth:`aabandon`, which must also read the
        thread's event log and settle it despite the same unrevivable
        identity.
        """
        from langgraph_events.serde._jsonplus import (  # noqa: PLC0415
            NamespaceAwareSerde,
        )

        serde = getattr(self._checkpointer, "serde", None)
        if isinstance(serde, NamespaceAwareSerde):
            return serde.tolerate_unresolved()
        return contextlib.nullcontext()

    def _read_pending_interrupts(
        self, config: RunnableConfig, method: str
    ) -> _PendingInterrupts:
        """*config*'s latest checkpoint's pending interrupt(s), read
        straight from the checkpointer. See :meth:`_pending_interrupt_writes`.

        Read inside :meth:`_tolerant_read`: a stored write naming a class
        that no longer imports degrades to an ``UnrevivedIdentity``
        instead of raising, precisely on the thread a retirement caller
        is hunting.

        Any *other* checkpointer failure still propagates, re-raised
        naming *method* and the thread so the caller knows which one is
        unreadable. Never swallowed: skipping it here would make that
        thread silently invisible to every caller of this method,
        discovery included.
        """
        thread_id = config.get("configurable", {}).get("thread_id")
        try:
            with self._tolerant_read():
                tup = self._checkpointer.get_tuple(config)
        except Exception as exc:
            raise RuntimeError(
                f"{method}() could not read thread {thread_id!r}'s checkpoint: {exc}"
            ) from exc
        if tup is None:
            return _PendingInterrupts(
                has_interrupt=False, events=[], unresolved_names=[]
            )
        return self._pending_interrupt_writes(tup.pending_writes)

    async def _aread_pending_interrupts(
        self, config: RunnableConfig, method: str
    ) -> _PendingInterrupts:
        """Async sibling of :meth:`_read_pending_interrupts`, via
        ``aget_tuple``."""
        thread_id = config.get("configurable", {}).get("thread_id")
        try:
            with self._tolerant_read():
                tup = await self._checkpointer.aget_tuple(config)
        except Exception as exc:
            raise RuntimeError(
                f"{method}() could not read thread {thread_id!r}'s checkpoint: {exc}"
            ) from exc
        if tup is None:
            return _PendingInterrupts(
                has_interrupt=False, events=[], unresolved_names=[]
            )
        return self._pending_interrupt_writes(tup.pending_writes)

    @staticmethod
    def _require_settleable(
        method: str, snapshot: StateSnapshot, config: RunnableConfig
    ) -> None:
        """Raise ``ValueError`` if the thread has no events to settle.

        Checks the event log, not ``snapshot.created_at is None``: a
        ``pre_seed``ed-only thread has a checkpoint and would pass that
        check.
        """
        if snapshot.values.get("events"):
            return
        thread_id = config.get("configurable", {}).get("thread_id")
        raise ValueError(
            f"{method}() has no events to settle on thread {thread_id!r}: the "
            "thread was never run (or only pre_seed()ed)."
        )

    @staticmethod
    def _require_pending_interrupt(
        method: str, pending: _PendingInterrupts, config: RunnableConfig
    ) -> None:
        """Raise ``ValueError`` if *pending* (from
        :meth:`_read_pending_interrupts`) has no interrupt at all,
        including a non-``Event`` payload, which still genuinely pauses
        the thread even though ``discarded`` cannot name it.

        Guards the default ``require_interrupt=True``: without it,
        ``abandon()``/``aabandon()`` would silently settle an
        already-completed thread onto a terminal ``Abandoned``.
        """
        if pending.has_interrupt:
            return
        thread_id = config.get("configurable", {}).get("thread_id")
        raise ValueError(
            f"{method}() found no pending interrupt on thread {thread_id!r}. "
            "Pass require_interrupt=False to settle it anyway."
        )

    def abandon(
        self,
        config: RunnableConfig,
        *,
        reason: str = "",
        require_interrupt: bool = True,
    ) -> None:
        """Settle a thread onto a terminal ``Abandoned`` without
        dispatching whatever ``Interrupted`` it was paused on. Use this
        to retire an ``Interrupted`` subclass: resuming every paused
        thread first would instead append that identity to the log.

        If ``require_interrupt`` is ``True`` (the default) and the thread
        has no pending interrupt, raises ``ValueError`` naming the
        thread. Pass ``require_interrupt=False`` to settle such a thread
        anyway, recording ``Abandoned(discarded="")``.

        Settles a thread even when the pending interrupt names an
        ``Interrupted`` subclass already deleted from the codebase,
        recording the class's last-known qualname in ``discarded``
        instead of a live instance.

        Requires a checkpointer. Raises ``ValueError`` if the thread has
        no events to settle (never run, or only ``pre_seed``ed). Ignores
        ``on_unresumable``: that policy governs an accidental no-op
        ``resume()``, not a deliberate abandonment.

        Runs no graph, so returns no log: call
        ``graph.get_state(config).events`` for it.
        """
        self._require_checkpointer("abandon")
        with self._tolerant_read():
            snapshot = self._compile().get_state(config)
            self._require_settleable("abandon", snapshot, config)
            pending = self._read_pending_interrupts(config, "abandon")
            if require_interrupt:
                self._require_pending_interrupt("abandon", pending, config)
            discarded = ", ".join(
                (
                    *(type(v).__qualname__ for v in pending.events),
                    *pending.unresolved_names,
                )
            )
            self._settle(config, Abandoned(reason=reason, discarded=discarded))

    async def aabandon(
        self,
        config: RunnableConfig,
        *,
        reason: str = "",
        require_interrupt: bool = True,
    ) -> None:
        """Async version of :meth:`abandon`."""
        self._require_checkpointer("aabandon")
        with self._tolerant_read():
            snapshot = await self._compile().aget_state(config)
            self._require_settleable("aabandon", snapshot, config)
            pending = await self._aread_pending_interrupts(config, "aabandon")
            if require_interrupt:
                self._require_pending_interrupt("aabandon", pending, config)
            discarded = ", ".join(
                (
                    *(type(v).__qualname__ for v in pending.events),
                    *pending.unresolved_names,
                )
            )
            await self._asettle(config, Abandoned(reason=reason, discarded=discarded))

    def _list_thread_ids(self, method: str) -> list[str]:
        """Every distinct thread id the checkpointer holds, sorted.

        Raises ``ValueError`` naming *method* if the checkpointer's
        ``list()`` is unimplemented, instead of letting a bare
        ``NotImplementedError`` reach the caller.

        Runs inside :meth:`_tolerant_read`: ``list()`` deserializes
        every checkpoint it walks, including old versions of threads
        already settled, so a thread whose pending interrupt names a
        deleted class must not break enumeration for every other thread.
        """
        ids: set[str] = set()
        try:
            with self._tolerant_read():
                for tup in self._checkpointer.list(None):
                    tid = tup.config.get("configurable", {}).get("thread_id")
                    if tid is not None:
                        ids.add(tid)
        except NotImplementedError as exc:
            raise ValueError(
                f"{method}() needs checkpointer.list() support, which this "
                f"checkpointer does not implement. Enumerate thread ids "
                f"yourself and call get_state() on each."
            ) from exc
        return sorted(ids)

    async def _alist_thread_ids(self, method: str) -> list[str]:
        """Async sibling of :meth:`_list_thread_ids`, via ``alist()``."""
        ids: set[str] = set()
        try:
            with self._tolerant_read():
                async for tup in self._checkpointer.alist(None):
                    tid = tup.config.get("configurable", {}).get("thread_id")
                    if tid is not None:
                        ids.add(tid)
        except NotImplementedError as exc:
            raise ValueError(
                f"{method}() needs checkpointer.alist() support, which this "
                f"checkpointer does not implement. Enumerate thread ids "
                f"yourself and call aget_state() on each."
            ) from exc
        return sorted(ids)

    @staticmethod
    def _matches_event_type(
        pending: _PendingInterrupts, event_type: type[Interrupted] | None
    ) -> bool:
        """Whether *pending* (from :meth:`_read_pending_interrupts`) has
        an interrupt matching *event_type*: with ``None``, any pending
        interrupt matches, including a non-``Event`` payload."""
        if not pending.has_interrupt:
            return False
        if event_type is None:
            return True
        return any(isinstance(v, event_type) for v in pending.events)

    def threads_paused_on(
        self, event_type: type[Interrupted] | None = None
    ) -> list[RunnableConfig]:
        """Configs for every thread whose latest checkpoint has a pending
        interrupt. With *event_type*, keeps only threads paused on that
        class or a subclass. With ``None``, returns every paused thread.

        Reads each thread's raw checkpoint directly, not this graph's
        compiled topology. Two deletions this unlocks, with different
        outcomes:

        - The **handler** that produced the interrupt is already removed
          from this graph. The class still imports, so the interrupt
          revives normally and *event_type* filtering still works. This
          is the common order: retiring an ``Interrupted`` usually
          retires the handler that produced it first.
        - The **class** itself has been deleted and no longer imports.
          The interrupt cannot revive. With no filter
          (``event_type=None``), the thread is still returned. With a
          class filter, it matches nothing: a filter can never match an
          identity with no class. ``abandon()`` records the interrupt's
          last-known qualname in ``discarded`` instead of a live
          instance.

        WARNING: this reads every checkpoint the checkpointer holds and
        deserializes every row: cost is O(all checkpoints), not O(paused
        threads). A large deployment should filter thread ids server-side
        instead of calling this directly.

        Requires a checkpointer. Raises ``ValueError`` if the
        checkpointer's ``list()`` is unimplemented, or if a custom saver
        requires a ``thread_id`` filter: enumerate thread ids yourself
        and call :meth:`get_state` on each in that case.
        """
        self._require_checkpointer("threads_paused_on")
        configs: list[RunnableConfig] = []
        for tid in self._list_thread_ids("threads_paused_on"):
            cfg = cast("RunnableConfig", {"configurable": {"thread_id": tid}})
            pending = self._read_pending_interrupts(cfg, "threads_paused_on")
            if self._matches_event_type(pending, event_type):
                configs.append(cfg)
        return configs

    async def athreads_paused_on(
        self, event_type: type[Interrupted] | None = None
    ) -> list[RunnableConfig]:
        """Async version of :meth:`threads_paused_on`."""
        self._require_checkpointer("athreads_paused_on")
        configs: list[RunnableConfig] = []
        for tid in await self._alist_thread_ids("athreads_paused_on"):
            cfg = cast("RunnableConfig", {"configurable": {"thread_id": tid}})
            pending = await self._aread_pending_interrupts(cfg, "athreads_paused_on")
            if self._matches_event_type(pending, event_type):
                configs.append(cfg)
        return configs

    @staticmethod
    def _unrevived_names(tup: CheckpointTuple | None) -> list[str]:
        """Qualname of every ``UnrevivedIdentity`` in *tup*'s latest
        ``events`` channel and pending ``__interrupt__`` writes, deduped
        in discovery order. Empty when *tup* is ``None``.

        Only a read inside :meth:`_tolerant_read` can produce an
        ``UnrevivedIdentity``: outside it the same identity raises.
        """
        from langgraph_events.serde._jsonplus import (  # noqa: PLC0415
            UnrevivedIdentity,
        )

        if tup is None:
            return []
        names: list[str] = []
        stored = tup.checkpoint.get("channel_values", {}).get("events", [])
        for entry in stored:
            if isinstance(entry, UnrevivedIdentity) and entry.qualname not in names:
                names.append(entry.qualname)
        for name in EventGraph._pending_interrupt_writes(
            tup.pending_writes or ()
        ).unresolved_names:
            if name not in names:
                names.append(name)
        return names

    def unrevivable_threads(self) -> dict[str, list[str]]:
        """Every thread whose latest checkpoint holds an event identity
        this graph's serde can no longer revive, keyed by thread id and
        sorted, each with the qualnames it could not revive, deduped in
        discovery order. Empty when every thread revives.

        Walks the real store, not the baseline. The coverage gates
        compare the topology to a committed baseline and never read a
        checkpoint, so after ``write_baseline(..., allow_removed=True)``
        they stay green while a settled thread out there still raises
        ``Cannot revive``. This is the sweep that sees it.

        Reads both the settled ``events`` history and the pending
        interrupt of each thread's latest checkpoint. So it reports a
        thread that already *answered* an interrupt on a deleted class
        (the #159 case, which :meth:`threads_paused_on` never reaches)
        and a thread still paused on one. It does not walk older
        checkpoint versions: a thread's latest history is a superset of
        every earlier one.

        Degrades only through
        :class:`~langgraph_events.serde.NamespaceAwareSerde`. Any other
        serde never degrades an identity: its read revives or raises as
        it normally would, so nothing is reported.

        WARNING: cost is O(all checkpoints), like
        :meth:`threads_paused_on`.

        Requires a checkpointer. Raises ``ValueError`` if the
        checkpointer's ``list()`` is unimplemented.
        """
        self._require_checkpointer("unrevivable_threads")
        found: dict[str, list[str]] = {}
        for tid in self._list_thread_ids("unrevivable_threads"):
            cfg = cast("RunnableConfig", {"configurable": {"thread_id": tid}})
            with self._tolerant_read():
                tup = self._checkpointer.get_tuple(cfg)
            names = self._unrevived_names(tup)
            if names:
                found[tid] = names
        return found

    async def aunrevivable_threads(self) -> dict[str, list[str]]:
        """Async version of :meth:`unrevivable_threads`."""
        self._require_checkpointer("aunrevivable_threads")
        found: dict[str, list[str]] = {}
        for tid in await self._alist_thread_ids("aunrevivable_threads"):
            cfg = cast("RunnableConfig", {"configurable": {"thread_id": tid}})
            with self._tolerant_read():
                tup = await self._checkpointer.aget_tuple(cfg)
            names = self._unrevived_names(tup)
            if names:
                found[tid] = names
        return found

    def _graph_state(self, snapshot: StateSnapshot) -> GraphState:
        """Build a :class:`GraphState` from a checkpoint snapshot.

        Shared by the sync :meth:`get_state` and async :meth:`aget_state` so
        the snapshot-to-state logic stays in one place across both paths.

        Deliberately a pure function of *snapshot*: no checkpointer read
        happens here, which keeps ``is_interrupted``/``interrupted``
        topology-dependent (see :meth:`_is_interrupted`). A caller
        holding a ``GraphState`` reached it through its own working
        graph, unlike discovery (:meth:`threads_paused_on`) or
        ``abandon()``, which read the checkpoint directly.
        """
        all_events = snapshot.values.get("events", [])
        log = EventLog(all_events)
        is_interrupted = self._is_interrupted(snapshot)
        interrupted = log.latest(Interrupted) if is_interrupted else None
        if is_interrupted and interrupted is None:
            # An Interrupted joins the log only on resume. A first pause
            # has none there yet. Fall back to the snapshot's pending
            # interrupt payload.
            for value in self._pending_interrupt_events(snapshot):
                if isinstance(value, Interrupted):
                    interrupted = value
                    break
        return GraphState(
            events=log,
            is_interrupted=is_interrupted,
            interrupted=interrupted,
        )

    def get_state(self, config: Any) -> GraphState:
        """Get event-level state of a checkpointed thread."""
        self._require_checkpointer("get_state")
        return self._graph_state(self._compile().get_state(config))

    async def aget_state(self, config: Any) -> GraphState:
        """Async version of :meth:`get_state`.

        Reads the checkpoint via ``aget_state`` so async-only checkpointers
        (e.g. ``AsyncPostgresSaver``) aren't driven synchronously from the
        running event loop.
        """
        self._require_checkpointer("aget_state")
        return self._graph_state(await self._compile().aget_state(config))

    # --- LLM token helpers ---

    @staticmethod
    def _extract_text_content(chunk: Any) -> str:
        """Extract text from a LangChain chat model stream chunk."""
        content = getattr(chunk, "content", None)
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict) and item.get("type") == "text":
                    text = item.get("text")
                    if isinstance(text, str):
                        parts.append(text)
            return "".join(parts)
        return ""

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

    # --- High-level event streaming ---

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

    @staticmethod
    def _events_from_chunk(chunk: Any, seen: set[int]) -> list[Event]:
        """Extract unseen events from an updates-mode stream chunk."""
        events: list[Event] = []
        if isinstance(chunk, dict):
            for node_output in chunk.values():
                if isinstance(node_output, dict):
                    for event in node_output.get("events", []):
                        eid = id(event)
                        if eid not in seen:
                            seen.add(eid)
                            events.append(event)
        return events

    def _frames_from_values(
        self,
        state: dict[str, Any],
        prev_count: int,
        reducer_names: list[str],
    ) -> tuple[int, list[StreamFrame]]:
        """Extract new events and reducer snapshots from a values-mode state."""
        all_events: list[Event] = state.get("events", [])
        new_events = all_events[prev_count:]
        if not new_events:
            return prev_count, []
        reducers = {
            name: state.get(name, self._reducers[name].empty) for name in reducer_names
        }
        return len(all_events), [
            StreamFrame(event=e, reducers=reducers, changed_reducers=None)
            for e in new_events
        ]

    async def _astream_v2(  # noqa: PLR0912, PLR0915
        self,
        inp: Any,
        seeds: list[Event],
        *,
        reducer_names: list[str],
        include_llm_tokens: bool,
        include_custom_events: bool,
        **kwargs: Any,
    ) -> AsyncIterator[StreamItem]:
        """Stream events using LangGraph's v2 event API with LLM token support."""
        compiled = self._compile()
        is_resume = isinstance(inp, LGCommand)

        # Initialize incremental reducer state.  When a checkpoint exists
        # (subsequent run / resume), start from the checkpointed values so
        # the shadow state matches what the compiled graph already restored.
        # Fall back to r.seed() only on the true first run.
        # NOTE: this issues an extra aget_state round-trip when reducers are
        # present and a checkpointer is configured.
        reducer_state: dict[str, Any] = {}
        checkpoint_values: dict[str, Any] | None = None
        if reducer_names and self._checkpointer is not None:
            config = kwargs.get("config")
            if config is not None:
                snapshot = await compiled.aget_state(config)
                if snapshot.values:
                    checkpoint_values = snapshot.values

        for name in reducer_names:
            if checkpoint_values is not None:
                reducer_state[name] = checkpoint_values.get(
                    name, self._reducers[name].empty
                )
            else:
                reducer_state[name] = self._reducers[name].seed(seeds)

        # Merge seed contributions on top of checkpointed state.
        # - On astream_events second-run: seeds carry new events whose
        #   contributions need to layer on top of the restored checkpoint
        #   (LangGraph's seed_node will copy them into state["events"], but
        #   the in-adapter reducer_state shadow needs the contribution
        #   computed here so the StreamFrame yielded just below reflects it).
        # - On astream_resume (is_resume=True): skipped — the caller
        #   (e.g. AGUIAdapter) is responsible for pre-seeding contributions
        #   via apre_seed before calling astream_resume, because
        #   Command(resume=...) doesn't route seeds through the seed_node.
        #   Re-applying contributions here would double-count for
        #   accumulator reducers (operator.add channel reducers concatenate
        #   the contribution twice).  See
        #   `AGUIAdapter._resume_event_stream` for the canonical pattern.
        if checkpoint_values is not None and not is_resume:
            for s in seeds:
                self._update_reducer_state(reducer_state, s, reducer_names)

        # Yield seed events
        for s in seeds:
            if reducer_names:
                changed = frozenset(
                    name
                    for name in reducer_names
                    if self._reducers[name].has_contributions(
                        self._reducers[name].collect([s])
                    )
                )
                yield StreamFrame(
                    event=s,
                    reducers=dict(reducer_state),
                    changed_reducers=changed,
                )
            else:
                yield s

        seen: set[int] = set()

        async for raw in compiled.astream_events(
            inp,
            version="v2",
            stream_mode="updates",
            **kwargs,
        ):
            raw_event = raw.get("event")

            if raw_event == "on_chat_model_stream":
                if not include_llm_tokens:
                    continue
                run_id = raw.get("run_id", "")
                chunk = raw.get("data", {}).get("chunk")
                if chunk is None or not run_id:
                    continue
                delta = self._extract_text_content(chunk)
                if delta:
                    yield LLMToken(run_id=run_id, content=delta)
                for tc_chunk in getattr(chunk, "tool_call_chunks", None) or ():
                    index = tc_chunk.get("index")
                    if index is None:
                        raise ValueError(
                            f"LLM tool_call_chunk missing 'index' "
                            f"(run_id={run_id!r}, chunk={tc_chunk!r}). "
                            "LangChain normalizes this — a missing index "
                            "indicates a non-conformant provider or a bug "
                            "upstream."
                        )
                    yield LLMToolCallChunk(
                        run_id=run_id,
                        call_index=index,
                        tool_call_id=tc_chunk.get("id") or "",
                        name=tc_chunk.get("name") or "",
                        args_delta=tc_chunk.get("args") or "",
                    )
                continue

            if raw_event == "on_chat_model_end":
                if not include_llm_tokens:
                    continue
                run_id = raw.get("run_id", "")
                if run_id:
                    output = raw.get("data", {}).get("output")
                    message_id = getattr(output, "id", None)
                    yield LLMStreamEnd(run_id=run_id, message_id=message_id)
                continue

            if raw_event == "on_custom_event":
                if not include_custom_events:
                    continue
                if raw.get("name", "") == STATE_SNAPSHOT_EVENT_NAME:
                    yield StateSnapshotFrame(
                        data=_coerce_snapshot_data(raw.get("data")),
                    )
                else:
                    yield CustomEventFrame(
                        name=raw.get("name", ""),
                        data=raw.get("data"),
                    )
                continue

            if raw_event != "on_chain_stream" or raw.get("name") != "LangGraph":
                continue

            chunk = raw.get("data", {}).get("chunk")
            for event in self._events_from_chunk(chunk, seen):
                if reducer_names:
                    changed = self._update_reducer_state(
                        reducer_state,
                        event,
                        reducer_names,
                    )
                    yield StreamFrame(
                        event=event,
                        reducers=dict(reducer_state),
                        changed_reducers=changed,
                    )
                else:
                    yield event

    def _stream_sync(
        self,
        inp: Any,
        seeds: list[Event],
        reducer_names: list[str],
        **kwargs: Any,
    ) -> Iterator[Event | StreamFrame]:
        """Shared sync streaming core for stream_events/stream_resume."""
        kwargs = self._apply_deadline_kwarg(kwargs)
        compiled = self._compile()
        if not reducer_names:
            yield from seeds
            seen: set[int] = set()
            for chunk in compiled.stream(inp, stream_mode="updates", **kwargs):
                yield from self._events_from_chunk(chunk, seen)
        else:
            prev_count = 0
            first = True
            for state in compiled.stream(inp, stream_mode="values", **kwargs):
                if first:
                    first = False
                    continue
                prev_count, frames = self._frames_from_values(
                    state, prev_count, reducer_names
                )
                yield from frames

    async def _astream_core(
        self,
        inp: Any,
        seeds: list[Event],
        reducer_names: list[str],
        **kwargs: Any,
    ) -> AsyncIterator[Event | StreamFrame]:
        """Shared async streaming core for astream_events/astream_resume."""
        compiled = self._compile()
        if not reducer_names:
            for s in seeds:
                yield s
            seen: set[int] = set()
            async for chunk in compiled.astream(inp, stream_mode="updates", **kwargs):
                for event in self._events_from_chunk(chunk, seen):
                    yield event
        else:
            prev_count = 0
            first = True
            async for state in compiled.astream(inp, stream_mode="values", **kwargs):
                if first:
                    first = False
                    continue
                prev_count, frames = self._frames_from_values(
                    state, prev_count, reducer_names
                )
                for frame in frames:
                    yield frame

    def stream_events(
        self,
        seed: Event | list[Event],
        *,
        include_reducers: bool | list[str] = False,
        **kwargs: Any,
    ) -> Iterator[Event | StreamFrame]:
        """Yield individual events as they are produced during graph execution.

        Higher-level alternative to ``compiled.stream()`` — yields ``Event``
        objects directly instead of raw LangGraph state dicts.  Seed events
        are yielded first, followed by events produced by handlers.

        Args:
            seed: A single event or list of events to start the graph.
            include_reducers: When truthy, yields ``StreamFrame`` tuples
                instead of bare events.  Pass ``True`` for all reducers or
                a list of reducer names for selective inclusion.
        """
        inp = self._prepare_input(seed)
        kwargs.pop("stream_mode", None)
        reducer_names = self._resolve_reducer_names(include_reducers)
        yield from self._stream_sync(inp, inp["events"], reducer_names, **kwargs)

    def stream_resume(
        self,
        value: Event,
        *,
        seeds: list[Event] | None = None,
        include_reducers: bool | list[str] = False,
        **kwargs: Any,
    ) -> Iterator[Event | StreamFrame]:
        """Yield events produced when resuming an interrupted graph.

        Streaming equivalent of ``resume()`` — accepts a domain ``Event``,
        yields events as they are produced.  The ``Command(resume=value)``
        stays internal, exactly like ``resume()``.

        Args:
            value: The domain event to resume with.
            seeds: Optional events to dispatch alongside the resume in the
                same step.  Their reducer contributions layer onto the
                checkpointed state *before* the resume's downstream events
                run, so resume-handler outputs win for shared keys.  Used
                by ``AGUIAdapter`` to route ``FrontendStateMutated`` through
                the dispatch chain on resume.
            include_reducers: When truthy, yields ``StreamFrame`` tuples
                instead of bare events.  Pass ``True`` for all reducers or
                a list of reducer names for selective inclusion.
        """
        self._require_checkpointer("stream_resume")
        if not self._resume_is_pending(kwargs):
            log = self._apply_unresumable_policy(value, kwargs)  # raises for raise
            if self._on_unresumable == "halt":
                latest = log.latest(Unresumable)
                if latest is not None:
                    yield latest
            return
        kwargs.pop("stream_mode", None)
        reducer_names = self._resolve_reducer_names(include_reducers)
        yield from self._stream_sync(
            LGCommand(resume=value), seeds or [], reducer_names, **kwargs
        )

    async def _astream_entry(
        self,
        inp: Any,
        seeds: list[Event],
        *,
        include_reducers: bool | list[str],
        include_llm_tokens: bool,
        include_custom_events: bool,
        **kwargs: Any,
    ) -> AsyncIterator[StreamItem]:
        """Shared async-stream dispatcher — picks v2 vs core based on flags."""
        kwargs.pop("stream_mode", None)
        kwargs = self._apply_deadline_kwarg(kwargs)
        reducer_names = self._resolve_reducer_names(include_reducers)
        delegate = (
            self._astream_v2(
                inp,
                seeds,
                reducer_names=reducer_names,
                include_llm_tokens=include_llm_tokens,
                include_custom_events=include_custom_events,
                **kwargs,
            )
            if include_llm_tokens or include_custom_events
            else self._astream_core(inp, seeds, reducer_names, **kwargs)
        )
        async for item in delegate:
            yield item

    async def astream_resume(
        self,
        value: Event,
        *,
        seeds: list[Event] | None = None,
        include_reducers: bool | list[str] = False,
        include_llm_tokens: bool = False,
        include_custom_events: bool = False,
        **kwargs: Any,
    ) -> AsyncIterator[StreamItem]:
        """Async version of ``stream_resume()``.

        Streaming equivalent of ``aresume()`` — accepts a domain ``Event``,
        yields events as they are produced.

        Args:
            value: The domain event to resume with.
            seeds: Optional events to dispatch alongside the resume in the
                same step.  Their reducer contributions layer onto the
                checkpointed state *before* the resume's downstream events
                run, so resume-handler outputs win for shared keys.  Used
                by ``AGUIAdapter`` to route ``FrontendStateMutated`` through
                the dispatch chain on resume.
            include_reducers: When truthy, yields ``StreamFrame`` tuples
                instead of bare events.  Pass ``True`` for all reducers or
                a list of reducer names for selective inclusion.
            include_llm_tokens: When True, yields ``LLMToken`` and
                ``LLMStreamEnd`` frames for LLM token-level streaming.
            include_custom_events: When True, yields ``CustomEventFrame``
                and ``StateSnapshotFrame`` frames for ``on_custom_event`` payloads
                from LangGraph.
        """
        self._require_checkpointer("astream_resume")
        if not await self._aresume_is_pending(kwargs):
            # raises for the 'raise' policy; warn/halt return a log to adapt
            log = await self._aapply_unresumable_policy(value, kwargs)
            if self._on_unresumable == "halt":
                latest = log.latest(Unresumable)
                if latest is not None:
                    yield latest
            return
        async for item in self._astream_entry(
            LGCommand(resume=value),
            seeds or [],
            include_reducers=include_reducers,
            include_llm_tokens=include_llm_tokens,
            include_custom_events=include_custom_events,
            **kwargs,
        ):
            yield item

    async def astream_events(
        self,
        seed: Event | list[Event],
        *,
        include_reducers: bool | list[str] = False,
        include_llm_tokens: bool = False,
        include_custom_events: bool = False,
        **kwargs: Any,
    ) -> AsyncIterator[StreamItem]:
        """Async version of ``stream_events()``.

        Args:
            seed: A single event or list of events to start the graph.
            include_reducers: When truthy, yields ``StreamFrame`` tuples
                instead of bare events.  Pass ``True`` for all reducers or
                a list of reducer names for selective inclusion.
            include_llm_tokens: When True, yields ``LLMToken`` and
                ``LLMStreamEnd`` frames for LLM token-level streaming.
            include_custom_events: When True, yields ``CustomEventFrame``
                and ``StateSnapshotFrame`` frames for ``on_custom_event`` payloads
                from LangGraph.
        """
        inp = self._prepare_input(seed)
        async for item in self._astream_entry(
            inp,
            inp["events"],
            include_reducers=include_reducers,
            include_llm_tokens=include_llm_tokens,
            include_custom_events=include_custom_events,
            **kwargs,
        ):
            yield item
