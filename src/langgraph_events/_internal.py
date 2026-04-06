"""Internal LangGraph machinery — hidden from users.

Defines the internal state schema, seed/router/dispatch/handler node wrappers
that implement the hub-and-spoke reactive loop on top of LangGraph's StateGraph.
"""

from __future__ import annotations

import asyncio
import operator
from collections.abc import Callable, Coroutine  # noqa: TC003
from typing import TYPE_CHECKING, Annotated, Any, TypedDict, cast

if TYPE_CHECKING:
    from langchain_core.runnables import RunnableConfig, RunnableLambda

from langgraph.graph import END
from langgraph.types import Send  # noqa: TC002

from langgraph_events._custom_event import _reset_custom_emitters, _set_custom_emitters
from langgraph_events._event import (
    Cancelled,
    Event,
    Halted,
    MaxRoundsExceeded,
    Resumed,
    Scatter,
)
from langgraph_events._event_log import EventLog
from langgraph_events._handler import HandlerMeta  # noqa: TC001
from langgraph_events._reducer import BaseReducer  # noqa: TC001
from langgraph_events._types import HandlerReturn, StateDict  # noqa: TC001

# ---------------------------------------------------------------------------
# Internal state — users never see this
# ---------------------------------------------------------------------------

# Base fields present on every graph (no reducers needed)
_BASE_FIELDS: dict[str, Any] = {
    "events": Annotated[list[Event], operator.add],
    "_cursor": int,
    "_pending": list[Event],
    "_round": int,
}


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
        }
        if reds:
            if prev_cursor == 0:
                # First run — initialize with default + seed events
                for name, r in reds.items():
                    result[name] = r.seed(new_events)
            elif new_events:
                # Subsequent run (checkpointer) — only process new events
                for name, r in reds.items():
                    collected = r.collect(new_events)
                    if r.has_contributions(collected):
                        result[name] = collected
        return result

    return seed


def make_router_node(max_rounds: int) -> Callable[[StateDict], StateDict]:
    """Create the router node that collects new events and advances the cursor."""

    def router(state: StateDict) -> StateDict:
        new_events = state["events"][state["_cursor"] :]
        has_resume = any(isinstance(e, Resumed) for e in new_events)
        current_round = 1 if has_resume else state.get("_round", 0) + 1
        if current_round > max_rounds:
            halted = MaxRoundsExceeded(rounds=max_rounds)  # type: ignore[call-arg]
            return {
                "_cursor": len(state["events"]),
                "_pending": [halted],
                "_round": current_round,
                "events": [halted],
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

        # Find handlers whose event_types match any pending event
        matched: list[str] = []
        seen: set[str] = set()
        for meta in handler_metas:
            if meta.name not in seen and any(meta.matches(e) for e in pending):
                seen.add(meta.name)
                matched.append(meta.name)

        if not matched:
            return END

        return matched if len(matched) > 1 else matched[0]  # type: ignore[return-value]

    return dispatch


def _build_inject(
    meta: HandlerMeta,
    state: StateDict,
    reducers: dict[str, BaseReducer],
    config: RunnableConfig | None = None,
) -> dict[str, Any]:
    """Build keyword arguments to inject into a handler call."""
    inject: dict[str, Any] = {}
    if meta.log_param:
        inject[meta.log_param] = EventLog(state["events"])
    for param_name in meta.reducer_params:
        r = reducers.get(param_name)
        inject[param_name] = state.get(param_name, r.empty if r else [])
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
    return inject


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


def make_handler_node(
    meta: HandlerMeta,
    reducers: dict[str, BaseReducer] | None = None,
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
    from langchain_core.runnables import RunnableLambda  # noqa: PLC0415
    from langgraph.types import interrupt as lg_interrupt  # noqa: PLC0415

    reds = reducers or {}

    def _prepare(
        state: StateDict, config: RunnableConfig
    ) -> tuple[list[Event], dict[str, Any]]:
        matching = [e for e in state["_pending"] if meta.matches(e)]
        inject = _build_inject(meta, state, reds, config)
        return matching, inject

    def _finalize(new_events: list[Event]) -> StateDict:
        output: StateDict = {"events": new_events}
        if reds:
            output.update(_apply_reducers(new_events, reds))
        return output

    def _run_handler_sync(state: StateDict, config: RunnableConfig) -> StateDict:
        matching, inject = _prepare(state, config)
        new_events: list[Event] = []
        tokens = _set_custom_emitters(
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
        try:
            for event in matching:
                call_inject = _inject_fields(meta, event, inject)
                if meta.is_async:
                    try:
                        asyncio.get_running_loop()
                    except RuntimeError:
                        # No running loop — safe to use asyncio.run().
                        coro = cast(
                            "Coroutine[Any, Any, HandlerReturn]",
                            meta.fn(event, **call_inject),
                        )
                        result = asyncio.run(coro)
                    else:
                        raise RuntimeError(
                            f"Handler {meta.name!r} is async but invoke() was called "
                            "from within a running event loop (e.g. Jupyter, "
                            "FastAPI). "
                            f"Use ainvoke() instead."
                        )
                else:
                    result = meta.fn(event, **call_inject)
                _collect_result(result, new_events, lg_interrupt)
        finally:
            _reset_custom_emitters(tokens)
        return _finalize(new_events)

    async def _run_handler_async(state: StateDict, config: RunnableConfig) -> StateDict:
        matching, inject = _prepare(state, config)
        new_events: list[Event] = []
        tokens = _set_custom_emitters(
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
        try:
            for event in matching:
                call_inject = _inject_fields(meta, event, inject)
                if meta.is_async:
                    coro = cast(
                        "Coroutine[Any, Any, HandlerReturn]",
                        meta.fn(event, **call_inject),
                    )
                    result = await coro
                else:
                    result = meta.fn(event, **call_inject)
                _collect_result(result, new_events, lg_interrupt)
        except asyncio.CancelledError:
            return _finalize([Cancelled()])
        finally:
            _reset_custom_emitters(tokens)
        return _finalize(new_events)

    return RunnableLambda(
        func=_run_handler_sync,
        afunc=_run_handler_async,
        name=meta.name,
    )


def _collect_result(
    result: HandlerReturn,
    new_events: list[Event],
    lg_interrupt: Callable[[Any], Any],
) -> None:
    """Normalise handler return and handle Interrupted / Scatter."""
    if result is None:
        return

    if not isinstance(result, (Event, Scatter)):
        raise TypeError(
            f"Handler must return Event | None | Scatter, got {type(result).__name__}. "
            f"Handlers return a single event, None, or Scatter — never a list."
        )

    result._collect_into(new_events, lg_interrupt)
