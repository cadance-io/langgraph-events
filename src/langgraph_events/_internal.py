"""Internal LangGraph machinery — hidden from users.

Defines the internal state schema, seed/router/dispatch/handler node wrappers
that implement the hub-and-spoke reactive loop on top of LangGraph's StateGraph.
"""

from __future__ import annotations

import asyncio
import operator
from collections.abc import Callable  # noqa: TC003
from typing import TYPE_CHECKING, Annotated, Any, TypedDict

if TYPE_CHECKING:
    from langchain_core.runnables import RunnableLambda

from langgraph.graph import END
from langgraph.types import Send  # noqa: TC002

from langgraph_events._event import Event, Halted, Resumed, Scatter
from langgraph_events._event_log import EventLog
from langgraph_events._handler import HandlerMeta  # noqa: TC001
from langgraph_events._reducer import Reducer  # noqa: TC001
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


def build_state_schema(reducers: dict[str, Reducer]) -> type:
    """Create a dynamic TypedDict with per-reducer state channels.

    Each reducer gets its own ``_r_<name>`` channel annotated with its
    ``reducer`` function, e.g. ``Annotated[list, add_messages]``.
    """
    fields: dict[str, Any] = dict(_BASE_FIELDS)
    for name, r in reducers.items():
        fields[f"_r_{name}"] = Annotated[list, r.reducer]
    return TypedDict("_FullState", fields)  # type: ignore[operator]


# ---------------------------------------------------------------------------
# Node factories
# ---------------------------------------------------------------------------


def _validate_and_collect(
    name: str,
    reducer: Reducer,
    events: list[Event],
) -> list[Any]:
    """Run a reducer fn over events with type validation."""
    contributions: list[Any] = []
    for event in events:
        contrib = reducer.fn(event)
        if contrib:
            if not isinstance(contrib, list):
                raise TypeError(
                    f"Reducer {name!r} fn must return a list, "
                    f"got {type(contrib).__name__}"
                )
            contributions.extend(contrib)
    return contributions


def make_seed_node(
    reducers: dict[str, Reducer] | None = None,
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
                    values = list(r.default)
                    values.extend(_validate_and_collect(name, r, new_events))
                    result[f"_r_{name}"] = values
            elif new_events:
                # Subsequent run (checkpointer) — only process new events
                for name, r in reds.items():
                    contribs = _validate_and_collect(name, r, new_events)
                    if contribs:
                        result[f"_r_{name}"] = contribs
        return result

    return seed


def make_router_node(max_rounds: int) -> Callable[[StateDict], StateDict]:
    """Create the router node that collects new events and advances the cursor."""

    def router(state: StateDict) -> StateDict:
        new_events = state["events"][state["_cursor"] :]
        has_resume = any(isinstance(e, Resumed) for e in new_events)
        current_round = 1 if has_resume else state.get("_round", 0) + 1
        if current_round > max_rounds:
            raise RuntimeError(
                f"EventGraph exceeded max_rounds={max_rounds}. "
                f"Possible infinite loop — check your event handlers for cycles."
            )
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
        pending = state["_pending"]
        if not pending:
            return END

        # Check for Halted
        if any(isinstance(e, Halted) for e in pending):
            return END

        # Find handlers whose event_types match any pending event
        matched: list[str] = []
        for meta in handler_metas:
            if any(isinstance(e, meta.event_types) for e in pending):
                if meta.name not in matched:
                    matched.append(meta.name)

        if not matched:
            return END

        return matched if len(matched) > 1 else matched[0]  # type: ignore[return-value]

    return dispatch


def _build_inject(
    meta: HandlerMeta,
    state: StateDict,
) -> dict[str, Any]:
    """Build keyword arguments to inject into a handler call."""
    inject: dict[str, Any] = {}
    if meta.log_param:
        inject[meta.log_param] = EventLog(state["events"])
    for param_name in meta.reducer_params:
        inject[param_name] = state.get(f"_r_{param_name}", [])
    return inject


def _apply_reducers(
    new_events: list[Event],
    reducers: dict[str, Reducer],
) -> dict[str, list[Any]]:
    """Run reducer projections on newly produced events.

    Returns per-channel updates keyed by ``_r_<name>``.
    """
    updates: dict[str, list[Any]] = {}
    for name, r in reducers.items():
        contributions = _validate_and_collect(name, r, new_events)
        if contributions:
            updates[f"_r_{name}"] = contributions
    return updates


def make_handler_node(
    meta: HandlerMeta,
    reducers: dict[str, Reducer] | None = None,
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
    from langchain_core.runnables import RunnableLambda  # noqa: PLC0415
    from langgraph.types import interrupt as lg_interrupt  # noqa: PLC0415

    reds = reducers or {}

    def _run_handler_sync(state: StateDict) -> StateDict:
        matching = [e for e in state["_pending"] if isinstance(e, meta.event_types)]
        inject = _build_inject(meta, state)

        new_events: list[Event] = []
        for event in matching:
            if meta.is_async:
                try:
                    asyncio.get_running_loop()
                    raise RuntimeError(
                        f"Handler {meta.name!r} is async but invoke() was called "
                        f"from within a running event loop (e.g. Jupyter, FastAPI). "
                        f"Use ainvoke() instead."
                    )
                except RuntimeError as exc:
                    if "invoke() was called" in str(exc):
                        raise
                    # No running loop — safe to use asyncio.run()
                    result: HandlerReturn = asyncio.run(meta.fn(event, **inject))  # type: ignore[arg-type]
            else:
                result = meta.fn(event, **inject)
            _collect_result(result, new_events, lg_interrupt)

        output: StateDict = {"events": new_events}
        if reds:
            output.update(_apply_reducers(new_events, reds))
        return output

    async def _run_handler_async(state: StateDict) -> StateDict:
        matching = [e for e in state["_pending"] if isinstance(e, meta.event_types)]
        inject = _build_inject(meta, state)

        new_events: list[Event] = []
        for event in matching:
            if meta.is_async:
                result = await meta.fn(event, **inject)  # type: ignore[misc]
            else:
                result = meta.fn(event, **inject)
            _collect_result(result, new_events, lg_interrupt)

        output: StateDict = {"events": new_events}
        if reds:
            output.update(_apply_reducers(new_events, reds))
        return output

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
