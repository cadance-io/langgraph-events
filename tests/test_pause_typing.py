"""Static typing assertions for ``deadline=`` and ``pause_on=``.

These are checked by **mypy**, not pytest. The bodies are typing-only.
Run: ``uv run mypy tests/test_pause_typing.py``

Every ``EventGraph`` entry point declares both keywords. A type checker accepts
a valid call and rejects a bad value. Under strict ``warn_unused_ignores`` an
unused ignore is an error, so each ignore below proves that mypy rejects that
call.
"""

from __future__ import annotations

import asyncio
import threading

from langgraph_events import EventGraph, IntegrationEvent


class Ping(IntegrationEvent):
    pass


async def every_entry_point_accepts_a_deadline_and_a_signal(graph: EventGraph) -> None:
    signal = asyncio.Event()
    graph.invoke(Ping(), deadline=1.0, pause_on=threading.Event())
    await graph.ainvoke(Ping(), deadline=1.0, pause_on=signal)
    graph.resume(Ping(), deadline=1.0, pause_on=signal)
    await graph.aresume(Ping(), deadline=1.0, pause_on=signal)
    graph.stream_events(Ping(), deadline=1.0, pause_on=signal)
    graph.stream_resume(Ping(), deadline=1.0, pause_on=signal)
    graph.astream_events(Ping(), deadline=1.0, pause_on=signal)
    graph.astream_resume(Ping(), deadline=1.0, pause_on=signal)


async def every_entry_point_rejects_a_bool_signal(graph: EventGraph) -> None:
    graph.invoke(Ping(), pause_on=True)  # type: ignore[arg-type]
    await graph.ainvoke(Ping(), pause_on=True)  # type: ignore[arg-type]
    graph.resume(Ping(), pause_on=True)  # type: ignore[arg-type]
    await graph.aresume(Ping(), pause_on=True)  # type: ignore[arg-type]
    graph.stream_events(Ping(), pause_on=True)  # type: ignore[arg-type]
    graph.stream_resume(Ping(), pause_on=True)  # type: ignore[arg-type]
    graph.astream_events(Ping(), pause_on=True)  # type: ignore[arg-type]
    graph.astream_resume(Ping(), pause_on=True)  # type: ignore[arg-type]


def a_bound_is_set_is_rejected(graph: EventGraph) -> None:
    graph.invoke(Ping(), pause_on=threading.Event().is_set)  # type: ignore[arg-type]


def a_deadline_must_be_a_number(graph: EventGraph) -> None:
    graph.invoke(Ping(), deadline="soon")  # type: ignore[arg-type]


def the_event_class_is_rejected(graph: EventGraph) -> None:
    graph.invoke(Ping(), pause_on=threading.Event)  # type: ignore[arg-type]
