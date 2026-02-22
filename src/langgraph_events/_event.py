"""Event base class and built-in special events."""

from __future__ import annotations

from dataclasses import dataclass, field
from dataclasses import fields as dc_fields
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from langchain_core.messages import BaseMessage


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

    def as_messages(self) -> list[BaseMessage]:
        """Return LangChain messages represented by this event.

        Default returns an empty list.  Override in ``MessageEvent``
        subclasses to project events into the message stream.
        """
        return []

    def _collect_into(
        self,
        new_events: list[Event],
        interrupt_fn: Any,
    ) -> None:
        """Append this result to *new_events*.

        Framework-internal — overridden by ``Interrupted`` and ``Scatter``.
        """
        new_events.append(self)


@dataclass(frozen=True)
class MessageEvent(Event):
    """Mixin for events that wrap LangChain messages.

    Convention:
    - ``message`` field (single BaseMessage) → ``[self.message]``
    - ``messages`` field (tuple of BaseMessage) → ``list(self.messages)``
    - Override ``as_messages()`` for custom behavior.

    Example::

        @dataclass(frozen=True)
        class UserMessageReceived(MessageEvent, Auditable):
            message: HumanMessage

        @dataclass(frozen=True)
        class ToolsExecuted(MessageEvent, Auditable):
            messages: tuple[ToolMessage, ...]
    """

    def as_messages(self) -> list[BaseMessage]:
        msg = getattr(self, "message", None)
        if msg is not None:
            return [msg]
        msgs = getattr(self, "messages", None)
        if msgs is not None:
            return list(msgs)
        raise NotImplementedError(
            f"{type(self).__name__} must declare a 'message' or 'messages' field, "
            f"or override as_messages()"
        )


@dataclass(frozen=True)
class Auditable(Event):
    """Marker class for events that should be auto-logged.

    Inherit from this class to make events auditable. Use the ``trail()``
    method (or the ``@on(Auditable)`` subscription pattern) to log events
    as they flow through the graph.

    Example::

        @dataclass(frozen=True)
        class UserMessageReceived(Auditable):
            content: str = ""
    """

    def trail(self) -> str:
        """Return a compact, human-readable summary of this event."""
        name = type(self).__name__
        parts = []
        for f in dc_fields(self):
            val = getattr(self, f.name)
            if isinstance(val, str) and len(val) > 80:
                val = val[:80] + "..."
            elif isinstance(val, tuple) and len(val) > 3:
                val = f"({len(val)} items)"
            s = repr(val)
            if len(s) > 80:
                s = s[:77] + "..."
            parts.append(f"{f.name}={s}")
        return f"[{name}] {', '.join(parts)}"


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

    def _collect_into(
        self,
        new_events: list[Event],
        interrupt_fn: Any,
    ) -> None:
        """Record the interrupt, pause, and create a Resumed event."""
        new_events.append(self)
        resume_value = interrupt_fn(self)
        new_events.append(Resumed(value=resume_value, interrupted=self))


@dataclass(frozen=True)
class Resumed(Event):
    """Created by the framework when a graph resumes from an interrupt.

    Contains the human's response and a reference to the original
    ``Interrupted`` event.
    """

    value: Any = None
    interrupted: Interrupted | None = None


class Scatter:
    """Special return type for map-reduce fan-out.

    When a handler returns ``Scatter([event1, event2, ...])``, the framework
    expands them into multiple pending events for the next dispatch round.
    All matched handlers run before the router collects results.

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

    def _collect_into(
        self,
        new_events: list[Event],
        interrupt_fn: Any,
    ) -> None:
        """Expand scatter events into the collection list."""
        new_events.extend(self.events)
