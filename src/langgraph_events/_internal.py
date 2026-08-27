"""Internal LangGraph machinery — hidden from users.

Defines the internal state schema, seed/router/dispatch/handler node wrappers
that implement the hub-and-spoke reactive loop on top of LangGraph's StateGraph.
"""

from __future__ import annotations

import asyncio
import logging
import operator
import time
from collections.abc import Callable, Coroutine  # noqa: TC003
from typing import TYPE_CHECKING, Annotated, Any, TypedDict, cast

if TYPE_CHECKING:
    from contextvars import Token

    from langchain_core.runnables import RunnableConfig, RunnableLambda

    from langgraph_events._namespace import NamespaceModel
    from langgraph_events._reducer import BaseReducer

from langgraph.graph import END
from langgraph.types import Send  # noqa: TC002

from langgraph_events import _retry
from langgraph_events._custom_event import (
    _AsyncEmitter,
    _reset_custom_emitters,
    _set_custom_emitters,
    _SyncEmitter,
)
from langgraph_events._event import (
    Cancelled,
    Event,
    Halted,
    HandlerRaised,
    HandlerRetried,
    InvariantViolated,
    MaxRoundsExceeded,
    Resumed,
    RunPaused,
    Scatter,
)
from langgraph_events._event_log import EventLog
from langgraph_events._handler import HandlerMeta  # noqa: TC001
from langgraph_events._reducer import ReducerNotSetError
from langgraph_events._types import HandlerReturn, StateDict  # noqa: TC001

_logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Internal state — users never see this
# ---------------------------------------------------------------------------

# Base fields present on every graph (no reducers needed)
_BASE_FIELDS: dict[str, Any] = {
    "events": Annotated[list[Event], operator.add],
    "_cursor": int,
    "_pending": list[Event],
    "_round": int,
    # Router-side gate: one RunPaused per /run regardless of fan-ins (#88).
    "_run_paused_emitted": bool,
}

# Per-call deadline keys written into LangGraph ``configurable`` by the
# graph/adapter entry points and read by the router below. Names are
# centralised so a grep for the constant finds every writer and reader.
_DEADLINE_KEY = "__lge_deadline"
_DEADLINE_STARTED_AT_KEY = "__lge_deadline_started_at"


def _inject_deadline_keys(configurable: dict[str, Any], deadline: float) -> None:
    """Write the paired deadline keys into a ``configurable`` dict.

    Single writer for the two-key contract: every entry point that accepts
    a ``deadline=`` kwarg routes through here so the router can rely on
    both keys being present.
    """
    configurable[_DEADLINE_KEY] = deadline
    configurable[_DEADLINE_STARTED_AT_KEY] = time.monotonic()


class _InputState(TypedDict):
    events: list[Event]


class _OutputState(TypedDict):
    events: list[Event]


def build_state_schema(reducers: dict[str, BaseReducer]) -> type:
    """Create a dynamic TypedDict with per-reducer state channels.

    Each reducer gets its own channel (keyed by reducer name) with a type
    annotation determined by the reducer's ``state_annotation()`` method.
    """
    fields: dict[str, Any] = dict(_BASE_FIELDS)
    conflicts = set(reducers.keys()) & set(_BASE_FIELDS.keys())
    if conflicts:
        raise ValueError(
            f"Reducer name(s) {conflicts} conflict with reserved state fields"
        )
    for name, r in reducers.items():
        fields[name] = r.state_annotation()
    return TypedDict("_FullState", fields)  # type: ignore[operator]


# ---------------------------------------------------------------------------
# Node factories
# ---------------------------------------------------------------------------


def _leaf_node(func: Any, afunc: Any, name: str) -> RunnableLambda:
    """Wrap a node whose source LangGraph need not read.

    LangGraph reads a node function's source to find nested graphs. No
    EventGraph node holds one, so the read is pure cost, paid again by every
    new EventGraph over the same handlers. A node that declares no
    dependencies skips it.
    """
    from langchain_core.runnables import RunnableLambda  # noqa: PLC0415

    class _LeafNode(RunnableLambda):
        @property
        def deps(self) -> list[Any]:
            return []

    return _LeafNode(func=func, afunc=afunc, name=name)


def make_seed_node(
    reducers: dict[str, BaseReducer] | None = None,
) -> Callable[[StateDict], StateDict]:
    """Create the seed node that initialises cursor and pending from input."""
    reds = reducers or {}

    def seed(state: StateDict) -> StateDict:
        prev_cursor = state.get("_cursor", 0)
        all_events = state["events"]
        new_events = all_events[prev_cursor:]

        result: dict[str, Any] = {
            "_cursor": len(all_events),
            "_pending": new_events,
            "_round": 0,
            "_run_paused_emitted": False,
        }
        if reds:
            if prev_cursor == 0:
                for name, r in reds.items():
                    existing = state.get(name)
                    # Channel defaults: [] for list channels, None for
                    # scalar channels.  Anything else means pre-seeded
                    # via update_state / pre_seed().
                    if existing is not None and existing != []:
                        # Channel already has data — only apply seed
                        # contributions so the channel reducer merges
                        # them with the existing value.
                        collected = r.collect(new_events)
                        if r.has_contributions(collected):
                            result[name] = collected
                    else:
                        # True first run — initialize from default +
                        # seed events.
                        result[name] = r.seed(new_events)
            elif new_events:
                # Subsequent run (checkpointer) — only process new events
                for name, r in reds.items():
                    collected = r.collect(new_events)
                    if r.has_contributions(collected):
                        result[name] = collected
        return result

    return seed


def make_router_node(
    max_rounds: int,
) -> Callable[[StateDict, RunnableConfig], StateDict]:
    """Create the router node that collects new events and advances the cursor."""

    def router(state: StateDict, config: RunnableConfig) -> StateDict:
        new_events = state["events"][state["_cursor"] :]
        has_resume = any(isinstance(e, Resumed) for e in new_events)
        current_round = 1 if has_resume else state.get("_round", 0) + 1
        if current_round > max_rounds:
            halted = MaxRoundsExceeded(rounds=max_rounds)
            return {
                "_cursor": len(state["events"]),
                "_pending": [halted],
                "_round": current_round,
                "events": [halted],
            }
        configurable = (config or {}).get("configurable", {})
        deadline = configurable.get(_DEADLINE_KEY)
        if deadline is not None and time.monotonic() >= deadline:
            if state.get("_run_paused_emitted"):
                # Late fan-ins past the deadline drain without
                # re-emitting; in-flight events persist via
                # operator.add. See #88.
                return {
                    "_cursor": len(state["events"]),
                    "_pending": [],
                    "_round": current_round,
                }
            started_at = configurable[_DEADLINE_STARTED_AT_KEY]
            paused = RunPaused(
                elapsed_seconds=time.monotonic() - started_at,
            )
            return {
                # Advance cursor PAST the paused event so a fresh /run on
                # the same thread excludes it from new_events. Distinct
                # from MaxRoundsExceeded above which keeps cursor AT the
                # halted (terminal across runs).
                "_cursor": len(state["events"]) + 1,
                "_pending": [paused],
                "_round": current_round,
                "events": [paused],
                "_run_paused_emitted": True,
            }
        return {
            "_cursor": len(state["events"]),
            "_pending": new_events,
            "_round": current_round,
        }

    return router


def make_dispatch(
    handler_metas: list[HandlerMeta],
) -> Callable[[StateDict], list[str | Send] | str]:
    """Create the dispatch conditional edge function.

    Uses isinstance to match pending event types to handler subscriptions.
    ``handler.event_types`` is a tuple so ``isinstance(e, meta.event_types)``
    matches any subscribed type.
    Returns handler node names (list for parallel) or END.
    """

    def dispatch(state: StateDict) -> list[str | Send] | str:
        pending = state.get("_pending", [])
        if not pending:
            return END

        # Check for Halted
        if any(isinstance(e, Halted) for e in pending):
            return END

        # Find handlers whose event_types match any pending event. Route by
        # ``node_name`` (the registered graph node), which for inline command
        # handlers is the command qualname rather than the method name.
        matched: list[str] = []
        seen: set[str] = set()
        for meta in handler_metas:
            if meta.node_name not in seen and any(meta.matches(e) for e in pending):
                seen.add(meta.node_name)
                matched.append(meta.node_name)

        if not matched:
            return END

        return matched if len(matched) > 1 else matched[0]  # type: ignore[return-value]

    return dispatch


def _build_inject(  # noqa: PLR0912 — one branch per injectable kind
    meta: HandlerMeta,
    state: StateDict,
    reducers: dict[str, BaseReducer],
    config: RunnableConfig | None = None,
    services_by_type: dict[type, Any] | None = None,
    services_by_name: dict[str, Any] | None = None,
    *,
    model_provider: Callable[[], NamespaceModel],
) -> dict[str, Any]:
    """Build keyword arguments to inject into a handler call."""
    inject: dict[str, Any] = {}
    if meta.log_param or meta.reflection_param:
        log_view = EventLog(state["events"])
        if meta.log_param:
            inject[meta.log_param] = log_view
        if meta.reflection_param:
            from langgraph_events._reflection import Reflection  # noqa: PLC0415

            inject[meta.reflection_param] = Reflection(
                log_view, model=model_provider(), reducers=reducers
            )
    for param_name in meta.reducer_params:
        r = reducers.get(param_name)
        value = state.get(param_name, r.empty if r else [])
        if value is None and param_name in meta.required_reducer_params:
            raise _make_reducer_not_set_error(meta.name, param_name, r)
        inject[param_name] = value
    if meta.config_param and config is not None:
        inject[meta.config_param] = config
    if meta.store_param:
        if config is None:
            raise ValueError(
                f"Handler '{meta.name}' requested BaseStore injection, but runtime "
                "config is missing."
            )
        runtime = config.get("configurable", {}).get("__pregel_runtime")
        store = runtime.store if runtime is not None else None
        if store is None:
            raise ValueError(
                f"Handler '{meta.name}' requested BaseStore injection, but no store "
                "is configured. Pass store=... to EventGraph(...)."
            )
        inject[meta.store_param] = store
    if services_by_type and meta.service_params:
        for param_name, svc_type in meta.service_params:
            inject[param_name] = services_by_type[svc_type]
    if services_by_name and meta.service_name_params:
        for param_name, svc_name in meta.service_name_params:
            inject[param_name] = services_by_name[svc_name]
    return inject


def _make_reducer_not_set_error(
    handler_name: str, param_name: str, r: BaseReducer | None
) -> ReducerNotSetError:
    """Build a ``ReducerNotSetError`` with an actionable message.

    A catch-all reducer (``event_type=Event``) gives no useful hint, so drop
    the "Ensure an event of type Event…" sentence in that case and keep only
    the annotation suggestion.
    """
    if r is not None and r.event_type is not Event:
        type_name = getattr(r.event_type, "__name__", str(r.event_type))
        event_hint = (
            f"Ensure an event of type {type_name} has been "
            f"processed before this handler runs, or "
        )
    else:
        event_hint = "Set the reducer before this handler runs, or "
    return ReducerNotSetError(
        f"Handler {handler_name!r} requires reducer {param_name!r} to be "
        f"set, but its value is None. {event_hint}change the annotation to "
        f"permit None (e.g. '{param_name}: <type> | None')."
    )


def _apply_reducers(
    new_events: list[Event],
    reducers: dict[str, BaseReducer],
) -> dict[str, Any]:
    """Run reducer projections on newly produced events.

    Returns per-channel updates keyed by reducer name.
    """
    if not new_events:
        return {}
    updates: dict[str, Any] = {}
    for name, r in reducers.items():
        result = r.collect(new_events)
        if r.has_contributions(result):
            updates[name] = result
    return updates


def _inject_fields(
    meta: HandlerMeta,
    event: Event,
    inject: dict[str, Any],
) -> dict[str, Any]:
    """Add field-matcher values to the injection dict for a single event."""
    if not meta.field_inject_params:
        return inject
    merged = dict(inject)
    for field_name in meta.field_inject_params:
        merged[field_name] = getattr(event, field_name)
    return merged


def _make_handler_raised(
    meta: HandlerMeta,
    event: Event,
    exc: Exception,
    *,
    abandoned_for_deadline: bool = False,
) -> HandlerRaised:
    """Build a ``HandlerRaised`` event for a caught declared exception.

    The traceback is kept: this is the terminal failure, the one people debug
    from.  Its retry breadcrumbs drop theirs — see :func:`_detach_traceback`.
    """
    return HandlerRaised(
        handler=meta.name,
        source_event=event,
        exception=exc,
        abandoned_for_deadline=abandoned_for_deadline,
    )


def _detach_traceback(exc: BaseException) -> None:
    """Drop, in place, every frame *exc* keeps reachable.

    A stored exception's ``__traceback__`` pins the failing handler's frame —
    and every local on it — plus the dispatch frames above it, for as long as
    the event holding it lives.  Since the ``events`` channel is append-only,
    that is the whole run.

    The instance itself must survive: ``@on(HandlerRetried, exception=X)``
    isinstance-matches it and field injection hands it to the handler typed.
    So the frames go and the exception stays.

    The chain is walked because ``raise Wrapped(...) from upstream`` (and the
    implicit ``__context__`` an ``except`` block sets) leaves the *inner*
    exception's traceback pinning the very same handler frame — clearing only
    the outermost one would free nothing.  Group members are walked for the
    same reason, and are reachable through neither link: an
    ``asyncio.TaskGroup`` failure pins its frames through ``.exceptions``.
    """
    seen: set[int] = set()
    pending: list[BaseException] = [exc]
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        current.__traceback__ = None
        pending.extend(
            linked
            for linked in (current.__cause__, current.__context__)
            if linked is not None
        )
        if isinstance(current, BaseExceptionGroup):
            pending.extend(current.exceptions)


def _next_delay_or_give_up(
    meta: HandlerMeta,
    event: Event,
    exc: Exception,
    attempt: int,
    new_events: list[Event],
    deadline: float | None,
) -> float | None:
    """Seconds to wait before re-invoking *meta*, or ``None`` to give up.

    ``None`` means the retry loop is over and the terminal ``HandlerRaised``
    has already been appended to *new_events* — no policy, the budget is
    spent, this exception is out of the policy's ``on=`` scope, or the next
    backoff would cross *deadline*.  Otherwise the attempt is recorded per
    ``policy.observe`` (append a ``HandlerRetried``, log a warning, or stay
    quiet) and the delay returned.

    *deadline* is the run's soft-timeout instant on ``time.monotonic()``'s
    clock, or ``None`` when the caller passed no ``deadline=``.  A sleep that
    would land on or past it is refused outright rather than clamped: the
    router only checks the deadline *between* rounds, so an in-flight backoff
    cannot be preempted, and burning what is left of the budget on an attempt
    that almost certainly cannot finish either just delays the pause.  Giving
    up here lets the run reach the router and emit ``RunPaused`` promptly.
    The raise is tagged ``abandoned_for_deadline=True`` so it stays
    distinguishable from an exhausted attempt budget.

    Shared by both dispatch paths; only the sleep call itself differs between
    them.
    """
    policy = meta.retry
    if (
        policy is None
        or attempt >= policy.max_attempts
        or not policy.retries(exc, meta.raises)
    ):
        new_events.append(_make_handler_raised(meta, event, exc))
        return None
    delay = policy.delay_for(attempt, exc)
    if deadline is not None and time.monotonic() + delay >= deadline:
        new_events.append(
            _make_handler_raised(meta, event, exc, abandoned_for_deadline=True)
        )
        return None
    if policy.observe == "emit":
        # Why the breadcrumb must not carry the attempt's frames is in
        # _detach_traceback. Mutating the caller's exception is safe *here*
        # because the next attempt re-raises into a fresh traceback, so the
        # terminal HandlerRaised still gets one even when a handler re-raises
        # a single cached instance.
        _detach_traceback(exc)
        new_events.append(
            HandlerRetried(
                handler=meta.name,
                source_event=event,
                exception=exc,
                attempt=attempt,
                delay_seconds=delay,
            )
        )
    elif policy.observe == "log":
        _logger.warning(
            "Handler %r attempt %d raised %s: %s — retrying in %.3fs",
            meta.name,
            attempt,
            type(exc).__name__,
            exc,
            delay,
        )
    return delay


def _find_failing_invariant(meta: HandlerMeta, log: EventLog) -> type | None:
    """Return the first invariant whose predicate fails against *log*, else None.

    Predicates are sync-only (validated at decoration). Predicate exceptions
    propagate (do not become violations).
    """
    for inv_cls, predicate in meta.invariants:
        if not predicate(log):
            return inv_cls
    return None


def _check_invariants(
    meta: HandlerMeta, event: Event, state: StateDict
) -> InvariantViolated | None:
    """Pre-check — evaluate invariants against the current log.

    Returns the first violation, or None.  On false predicate, the handler
    is skipped and the returned ``InvariantViolated`` is committed in its
    place.  ``would_emit`` is empty because the handler never ran.
    """
    if not meta.invariants:
        return None
    inv_cls = _find_failing_invariant(meta, EventLog(state["events"]))
    if inv_cls is None:
        return None
    return InvariantViolated(
        invariant=inv_cls(),
        handler=meta.name,
        source_event=event,
    )


def _check_invariants_post(
    meta: HandlerMeta,
    event: Event,
    state: StateDict,
    new_events: list[Event],
    emitted: list[Event],
) -> InvariantViolated | None:
    """Post-check — evaluate invariants against the simulated log.

    Simulated log = ``state["events"]`` (pre-node committed state) plus
    everything the current node has buffered so far in *new_events* — which
    includes emissions from prior handler-loop iterations AND this call's
    *emitted* slice.  This gives per-command atomicity within a round:
    each handler call's invariant check sees the cumulative effect of
    earlier commands dispatched in the same node.

    Returns a violation if any predicate fails on the simulated state, else
    None.  On failure, the caller drops *emitted* and commits the returned
    ``InvariantViolated`` instead, with ``would_emit`` carrying the
    rolled-back events.
    """
    if not meta.invariants or not emitted:
        return None
    simulated = EventLog([*state["events"], *new_events])
    inv_cls = _find_failing_invariant(meta, simulated)
    if inv_cls is None:
        return None
    return InvariantViolated(
        invariant=inv_cls(),
        handler=meta.name,
        source_event=event,
        would_emit=tuple(emitted),
    )


def _check_sync_invocation_of_async(meta: HandlerMeta) -> None:
    """Raise if an async handler is invoked from within a running event loop.

    This check is framework-level (not a domain error) and must run *outside*
    the ``try/except meta.raises`` boundary, otherwise a user who declares
    ``raises=RuntimeError`` would silently swallow the diagnostic.
    """
    if not meta.is_async:
        return
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return
    raise RuntimeError(
        f"Handler {meta.name!r} is async but invoke() was called "
        "from within a running event loop (e.g. Jupyter, FastAPI). "
        "Use ainvoke() instead."
    )


def _invoke_sync_path(
    meta: HandlerMeta, event: Event, call_inject: dict[str, Any]
) -> HandlerReturn:
    """Invoke a handler from the sync dispatch path.

    The async-in-loop precondition is checked by
    :func:`_check_sync_invocation_of_async` at the top of the handler node.
    """
    if meta.is_async:
        coro = cast(
            "Coroutine[Any, Any, HandlerReturn]",
            meta.fn(event, **call_inject),
        )
        return asyncio.run(coro)
    return meta.fn(event, **call_inject)


async def _invoke_async_path(
    meta: HandlerMeta, event: Event, call_inject: dict[str, Any]
) -> HandlerReturn:
    """Invoke a handler from the async dispatch path."""
    if meta.is_async:
        coro = cast(
            "Coroutine[Any, Any, HandlerReturn]",
            meta.fn(event, **call_inject),
        )
        return await coro
    return meta.fn(event, **call_inject)


def _process_events_sync(
    meta: HandlerMeta,
    matching: list[Event],
    state: StateDict,
    inject: dict[str, Any],
    new_events: list[Event],
    lg_interrupt: Any,
    return_contract: Any = None,
    deadline: float | None = None,
) -> None:
    """Per-event invocation loop for the sync dispatch path."""
    for event in matching:
        violation = _check_invariants(meta, event, state)
        if violation is not None:
            new_events.append(violation)
            continue
        call_inject = _inject_fields(meta, event, inject)
        attempt = 1
        while True:
            try:
                result = _invoke_sync_path(meta, event, call_inject)
            except meta.raises as exc:
                delay = _next_delay_or_give_up(
                    meta, event, exc, attempt, new_events, deadline
                )
                if delay is None:
                    break
                _retry._sleep(delay)
                attempt += 1
                continue
            _collect_and_check(
                result, new_events, lg_interrupt, meta, state, event, return_contract
            )
            break


async def _process_events_async(
    meta: HandlerMeta,
    matching: list[Event],
    state: StateDict,
    inject: dict[str, Any],
    new_events: list[Event],
    lg_interrupt: Any,
    return_contract: Any = None,
    deadline: float | None = None,
) -> None:
    """Per-event invocation loop for the async dispatch path."""
    for event in matching:
        violation = _check_invariants(meta, event, state)
        if violation is not None:
            new_events.append(violation)
            continue
        call_inject = _inject_fields(meta, event, inject)
        attempt = 1
        while True:
            try:
                result = await _invoke_async_path(meta, event, call_inject)
            except meta.raises as exc:
                delay = _next_delay_or_give_up(
                    meta, event, exc, attempt, new_events, deadline
                )
                if delay is None:
                    break
                await _retry._asleep(delay)
                attempt += 1
                continue
            _collect_and_check(
                result, new_events, lg_interrupt, meta, state, event, return_contract
            )
            break


def make_handler_node(
    meta: HandlerMeta,
    reducers: dict[str, BaseReducer] | None = None,
    return_contract: Any = None,
    services_by_type: dict[type, Any] | None = None,
    services_by_name: dict[str, Any] | None = None,
    *,
    model_provider: Callable[[], NamespaceModel],
) -> RunnableLambda:
    """Wrap a user handler as a LangGraph node.

    Uses ``RunnableLambda`` with both sync and async implementations so
    the graph works with both ``invoke()`` and ``ainvoke()``.

    - Filters pending events by isinstance(e, handler.event_types)
    - Loops: calls handler once per matching event (strict event→event)
    - Normalises return: Event → [event], None → [], Scatter → list of events
    - Handles Interrupted: calls interrupt(), creates Resumed on resume
    - Applies reducer projections to new events
    """
    from langchain_core.callbacks.manager import (  # noqa: PLC0415
        adispatch_custom_event,
        dispatch_custom_event,
    )
    from langgraph.types import interrupt as lg_interrupt  # noqa: PLC0415

    reds = reducers or {}
    svcs_by_type = services_by_type
    svcs_by_name = services_by_name

    def _prepare(
        state: StateDict, config: RunnableConfig
    ) -> tuple[list[Event], dict[str, Any], float | None]:
        matching = [e for e in state["_pending"] if meta.matches(e)]
        inject = _build_inject(
            meta,
            state,
            reds,
            config,
            svcs_by_type,
            svcs_by_name,
            model_provider=model_provider,
        )
        # Read once per node call, not per event: the retry loop only needs
        # it on the failure path, and the no-deadline case stays a ``None``.
        deadline = (config or {}).get("configurable", {}).get(_DEADLINE_KEY)
        return matching, inject, deadline

    def _bind_custom_emitters(
        config: RunnableConfig,
    ) -> tuple[Token[_SyncEmitter | None], Token[_AsyncEmitter | None]]:
        """Bind this node call's custom-event emitters to *config*.

        Both dispatch paths bind the same pair; the sync and async node bodies
        diverge only at the ``_process_events_*`` call, not here.
        """
        return _set_custom_emitters(
            sync_emitter=lambda name, data: dispatch_custom_event(
                name,
                data,
                config=config,
            ),
            async_emitter=lambda name, data: adispatch_custom_event(
                name,
                data,
                config=config,
            ),
        )

    def _finalize(new_events: list[Event]) -> StateDict:
        output: StateDict = {"events": new_events}
        if reds:
            output.update(_apply_reducers(new_events, reds))
        return output

    def _run_handler_sync(state: StateDict, config: RunnableConfig) -> StateDict:
        # Precondition check — outside the raises= catch boundary so a user
        # with raises=RuntimeError can't swallow this framework diagnostic.
        _check_sync_invocation_of_async(meta)
        matching, inject, deadline = _prepare(state, config)
        new_events: list[Event] = []
        tokens = _bind_custom_emitters(config)
        try:
            _process_events_sync(
                meta,
                matching,
                state,
                inject,
                new_events,
                lg_interrupt,
                return_contract,
                deadline,
            )
        finally:
            _reset_custom_emitters(tokens)
        return _finalize(new_events)

    async def _run_handler_async(state: StateDict, config: RunnableConfig) -> StateDict:
        matching, inject, deadline = _prepare(state, config)
        new_events: list[Event] = []
        tokens = _bind_custom_emitters(config)
        try:
            await _process_events_async(
                meta,
                matching,
                state,
                inject,
                new_events,
                lg_interrupt,
                return_contract,
                deadline,
            )
        except asyncio.CancelledError:
            return _finalize([Cancelled()])
        finally:
            _reset_custom_emitters(tokens)
        return _finalize(new_events)

    return _leaf_node(_run_handler_sync, _run_handler_async, meta.name)


def _collect_result(
    result: HandlerReturn,
    new_events: list[Event],
    lg_interrupt: Callable[[Any], Any],
    meta: HandlerMeta | None = None,
    return_contract: Any = None,
) -> None:
    """Normalise handler return and handle Interrupted / Scatter."""
    if result is None:
        return

    if not isinstance(result, (Event, Scatter)):
        raise TypeError(
            f"Handler must return Event | None | Scatter, got {type(result).__name__}. "
            f"Handlers return a single event, None, or Scatter — never a list."
        )

    if return_contract is not None:
        _assert_return_matches(result, meta, return_contract)

    _assert_no_private_leak(result, meta)

    result._collect_into(new_events, lg_interrupt)


def _collect_and_check(
    result: HandlerReturn,
    new_events: list[Event],
    lg_interrupt: Callable[[Any], Any],
    meta: HandlerMeta,
    state: StateDict,
    event: Event,
    return_contract: Any = None,
) -> None:
    """Collect handler result then run the post-command invariant check.

    Snapshots the buffer length before ``_collect_result`` so the emitted
    delta can be isolated. If any invariant fails against ``log + emitted``,
    rolls back by truncating *new_events* back to the snapshot and appending
    a single ``InvariantViolated`` carrying ``would_emit``.
    """
    pre_len = len(new_events)
    _collect_result(result, new_events, lg_interrupt, meta, return_contract)
    emitted = new_events[pre_len:]
    violation = _check_invariants_post(meta, event, state, new_events, emitted)
    if violation is not None:
        del new_events[pre_len:]
        new_events.append(violation)


def _assert_no_private_leak(result: Event | Scatter, meta: HandlerMeta | None) -> None:
    """Defense-in-depth runtime check for Command-private leaks.

    Empty-typed ``Scatter`` annotations (bare, ``Scatter[Any]``,
    ``Scatter[Event]``, ``Scatter[DomainEvent]``, ``Scatter[TypeVar]``) are
    rejected at build time, so static analysis is the primary guard. This
    check still fires when a handler with a broad base-class annotation (e.g.
    ``-> DomainEvent``) constructs ``Scatter([Cmd.Private(...)])`` at runtime
    — Python's type system doesn't enforce annotations at the call site, so
    the runtime check catches what the static check structurally cannot see.
    """
    if meta is None:
        return
    if getattr(meta.fn, "_inline_command", None) is not None:
        return  # inline Cmd.handle — already constrained by static check
    from langgraph_events._event import DomainEvent  # noqa: PLC0415
    from langgraph_events._namespace._command_privacy import (  # noqa: PLC0415
        CommandPrivacyError,
    )

    events = result.events if isinstance(result, Scatter) else (result,)
    for ev in events:
        if not isinstance(ev, DomainEvent):
            continue
        owner_cmd = getattr(type(ev), "__command__", None)
        if owner_cmd is not None:
            raise CommandPrivacyError(
                f"Reactor {meta.name!r} emitted {type(ev).__qualname__}, "
                f"which is private to {owner_cmd.__qualname__}. Only "
                f"{owner_cmd.__qualname__}.handle() may emit it."
            )


def _assert_return_matches(
    result: Event | Scatter, meta: HandlerMeta | None, contract: Any
) -> None:
    """Enforce that *result* satisfies the handler's ``ReturnContract``."""
    handler_desc = f"Handler {meta.name!r}" if meta is not None else "Handler"
    if isinstance(result, Scatter):
        if contract.scatter_types:
            allowed = contract.scatter_types
            for ev in result.events:
                if not isinstance(ev, allowed):
                    allowed_names = " | ".join(t.__name__ for t in allowed)
                    raise TypeError(
                        f"{handler_desc} scattered a {type(ev).__name__}, "
                        f"but {contract.source} only permits {allowed_names}"
                    )
        return

    allowed = contract.types
    if not allowed:
        raise TypeError(
            f"{handler_desc} returned {type(result).__name__} but "
            f"{contract.source} permits only None. Remove the annotation or "
            f"widen it to include the event type(s) this handler returns."
        )
    if not isinstance(result, allowed):
        allowed_names = " | ".join(t.__name__ for t in allowed)
        raise TypeError(
            f"{handler_desc} must return one of {allowed_names} "
            f"({contract.source}), got {type(result).__name__}"
        )
