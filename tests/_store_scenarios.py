"""Events and graphs for the event-store tests, shared with subprocesses.

A subprocess runs ``python -c "import _store_scenarios as s; s.main()" <fn>
<args>`` with ``tests/`` on ``PYTHONPATH``, so each scenario is a module
function.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from langgraph_events import Event, IntegrationEvent
from langgraph_events.store import EventCodec, JsonlEventStore


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


def main() -> None:
    getattr(sys.modules[__name__], sys.argv[1])(*sys.argv[2:])
