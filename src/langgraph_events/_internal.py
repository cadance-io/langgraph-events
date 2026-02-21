"""Internal LangGraph machinery — hidden from users.

Defines the internal state schema, seed/router/dispatch/handler node wrappers
that implement the hub-and-spoke reactive loop on top of LangGraph's StateGraph.
"""

from __future__ import annotations

import asyncio
import operator
from typing import Annotated, Any, TypedDict

from langgraph.graph import END
from langgraph.types import Send

from langgraph_events._event import Event, Halt, Interrupted, Resumed, Scatter
from langgraph_events._event_log import EventLog
from langgraph_events._handler import HandlerMeta


# ---------------------------------------------------------------------------
# Internal state — users never see this
# ---------------------------------------------------------------------------

class _FullState(TypedDict):
    events: Annotated[list, operator.add]  # append-only event log
    _cursor: int                            # how far router has processed
    _pending: list                          # current dispatch batch
    _round: int                             # round counter for max_rounds


class _InputState(TypedDict):
    events: list


class _OutputState(TypedDict):
    events: list


# ---------------------------------------------------------------------------
# Node factories
# ---------------------------------------------------------------------------

def make_seed_node():
    """Create the seed node that initialises cursor and pending from input."""

    def seed(state: _FullState) -> dict[str, Any]:
        return {
            "_cursor": len(state["events"]),
            "_pending": list(state["events"]),
            "_round": 0,
        }

    return seed


def make_router_node(max_rounds: int):
    """Create the router node that collects new events and advances the cursor."""

    def router(state: _FullState) -> dict[str, Any]:
        new_events = state["events"][state["_cursor"]:]
        current_round = state.get("_round", 0) + 1
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


def make_dispatch(handler_metas: list[HandlerMeta]):
    """Create the dispatch conditional edge function.

    Uses isinstance to match pending event types to handler subscriptions.
    ``handler.event_types`` is a tuple so ``isinstance(e, meta.event_types)``
    matches any subscribed type.
    Returns handler node names (list for parallel) or END.
    """
    def dispatch(state: _FullState) -> list[str | Send] | str:
        pending = state["_pending"]
        if not pending:
            return END

        # Check for Halt
        if any(isinstance(e, Halt) for e in pending):
            return END

        # Find handlers whose event_types match any pending event
        matched: list[str] = []
        for meta in handler_metas:
            if any(isinstance(e, meta.event_types) for e in pending):
                if meta.name not in matched:
                    matched.append(meta.name)

        if not matched:
            return END

        return matched if len(matched) > 1 else matched[0]

    return dispatch


def make_handler_node(meta: HandlerMeta):
    """Wrap a user handler as a LangGraph node.

    Uses ``RunnableLambda`` with both sync and async implementations so
    the graph works with both ``invoke()`` and ``ainvoke()``.

    - Filters pending events by isinstance(e, handler.event_types)
    - Loops: calls handler once per matching event (strict event→event)
    - Normalises return: Event → [event], None → [], Scatter → list of events
    - Handles Interrupted: calls interrupt(), creates Resumed on resume
    """
    from langchain_core.runnables import RunnableLambda
    from langgraph.types import interrupt as lg_interrupt

    def _run_handler_sync(state: _FullState) -> dict[str, Any]:
        matching = [
            e for e in state["_pending"]
            if isinstance(e, meta.event_types)
        ]
        log = EventLog(state["events"]) if meta.wants_log else None

        new_events: list[Event] = []
        for event in matching:
            args: list[Any] = [event]
            if meta.wants_log:
                args.append(log)

            if meta.is_async:
                result = asyncio.get_event_loop().run_until_complete(meta.fn(*args))
            else:
                result = meta.fn(*args)
            _collect_result(result, new_events, lg_interrupt)

        return {"events": new_events}

    async def _run_handler_async(state: _FullState) -> dict[str, Any]:
        matching = [
            e for e in state["_pending"]
            if isinstance(e, meta.event_types)
        ]
        log = EventLog(state["events"]) if meta.wants_log else None

        new_events: list[Event] = []
        for event in matching:
            args: list[Any] = [event]
            if meta.wants_log:
                args.append(log)

            if meta.is_async:
                result = await meta.fn(*args)
            else:
                result = meta.fn(*args)
            _collect_result(result, new_events, lg_interrupt)

        return {"events": new_events}

    return RunnableLambda(
        func=_run_handler_sync,
        afunc=_run_handler_async,
        name=meta.name,
    )


def _collect_result(
    result: Any,
    new_events: list[Event],
    lg_interrupt: Any,
) -> None:
    """Normalise handler return and handle Interrupted / Scatter."""
    if result is None:
        return

    if isinstance(result, Scatter):
        new_events.extend(result.events)
        return

    if not isinstance(result, Event):
        raise TypeError(
            f"Handler must return Event | None | Scatter, got {type(result).__name__}. "
            f"Handlers return a single event, None, or Scatter — never a list."
        )

    if isinstance(result, Interrupted):
        # Call LangGraph interrupt — pauses first time, returns value on resume
        resume_value = lg_interrupt(result)
        resumed = Resumed(value=resume_value, interrupted=result)
        new_events.append(result)
        new_events.append(resumed)
    else:
        new_events.append(result)
