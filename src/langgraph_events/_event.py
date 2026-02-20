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
