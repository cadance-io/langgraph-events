"""Event base class and built-in special events."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Event:
    """Base class for all events.

    Subclass as a frozen dataclass to define domain events.
    Supports multiple inheritance for "one object, two interfaces" pattern.

    Example::

        @dataclass(frozen=True)
        class DocumentReceived(Event):
            doc_id: str
            content: str
    """


@dataclass(frozen=True)
class Halt(Event):
    """Special event that signals immediate graph termination."""

    reason: str = ""


@dataclass(frozen=True)
class Interrupted(Event):
    """Special event that pauses the graph for human input.

    When a handler returns an ``Interrupted`` event, the framework calls
    LangGraph's ``interrupt()`` and the graph pauses. Resume with
    ``Command(resume=value)`` to continue — the framework creates a
    ``Resumed`` event automatically.
    """

    prompt: str = ""
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Resumed(Event):
    """Created by the framework when a graph resumes from an interrupt.

    Contains the human's response and a reference to the original
    ``Interrupted`` event.
    """

    value: Any = None
    interrupted: Interrupted = field(default_factory=Interrupted)


class Scatter:
    """Special return type for map-reduce fan-out.

    When a handler returns ``Scatter([event1, event2, ...])``, the framework
    expands them into multiple pending events. Dispatch uses LangGraph's
    ``Send()`` to create truly parallel handler invocations.

    Example::

        @on(BatchReceived)
        def split(event: BatchReceived) -> Scatter:
            return Scatter([ItemToProcess(item=i) for i in event.items])
    """

    __slots__ = ("events",)

    def __init__(self, events: list[Event]) -> None:
        if not events:
            raise ValueError("Scatter requires at least one event")
        for e in events:
            if not isinstance(e, Event):
                raise TypeError(
                    f"Scatter events must be Event instances, got {type(e).__name__}"
                )
        self.events = list(events)
