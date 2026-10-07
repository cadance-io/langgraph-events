"""Events and graphs for the event-store tests, shared with subprocesses.

A subprocess runs ``python -c "import _store_scenarios as s; s.main()" <fn>
<args>`` with ``tests/`` on ``PYTHONPATH``, so each scenario is a module
function.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from langgraph_events import Event, EventGraph, FoldReducer, IntegrationEvent, on
from langgraph_events.store import EventCodec, EventStream, JsonlEventStore


class Define(IntegrationEvent):
    name: str
    field: str


class Defined(IntegrationEvent):
    name: str
    field: str


def make_runtime_class(name: str, field: str) -> type[Event]:
    """The one factory a live define and a replay both call."""
    return type(
        name,
        (IntegrationEvent,),
        {
            "__module__": __name__,
            "__qualname__": f"Runtime.{name}",
            "__annotations__": {field: int},
        },
    )


def replay_defined(event: Event) -> list[type[Event]]:
    assert isinstance(event, Defined)
    return [make_runtime_class(event.name, event.field)]


def runtime_codec() -> EventCodec:
    return EventCodec(events=[Define, Defined], replay={Defined: replay_defined})


def write_runtime_class(path: str) -> None:
    widget = make_runtime_class("Widget", "size")
    events = [
        Define(name="Widget", field="size"),
        Defined(name="Widget", field="size"),
        widget(size=3),
    ]
    with JsonlEventStore(Path(path)) as store:
        store.append(runtime_codec().encode(events))


def read_runtime_class(path: str) -> None:
    with JsonlEventStore(Path(path)) as store:
        log = runtime_codec().decode(store.load())
    print(
        json.dumps(
            {
                "types": [type(event).__qualname__ for event in log],
                "size": getattr(log[-1], "size", None),
                "importable": hasattr(sys.modules[__name__], "Widget"),
            }
        )
    )


class Start(IntegrationEvent):
    marker: str


class Approved(IntegrationEvent):
    marker: str


class Done(IntegrationEvent):
    pass


@on(Start)
def approve(event: Start) -> Approved:
    return Approved(marker=event.marker)


@on(Approved)
def act_then_die(event: Approved) -> Done:
    Path(event.marker).write_text("side effect")
    os._exit(1)


def crash_after_side_effect(path: str, marker: str) -> None:
    with JsonlEventStore(Path(path)) as store:
        stream = EventStream(EventGraph([approve, act_then_die]), store, EventCodec())
        stream.invoke(Start(marker=marker))


class Tick(IntegrationEvent):
    pass


class Tock(IntegrationEvent):
    seen: int


FOLDS: list[int] = []
"""One entry per call of the ``ticks`` fold, so a test can count folds."""


def _count(state: int, event: Event) -> int:
    FOLDS.append(1)
    return state + 1


def ticks() -> FoldReducer[int]:
    return FoldReducer(name="ticks", event_type=Tick, default_factory=int, fold=_count)


@on(Tick)
def guard(event: Tick, ticks: int) -> Tock:
    return Tock(seen=ticks)


def main() -> None:
    getattr(sys.modules[__name__], sys.argv[1])(*sys.argv[2:])
