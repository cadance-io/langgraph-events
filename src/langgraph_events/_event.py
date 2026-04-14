"""Event base class and built-in special events."""

from __future__ import annotations

import dataclasses
import types
from dataclasses import fields as dc_fields
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from langchain_core.messages import BaseMessage, SystemMessage


class Event:
    """Base class for all events.

    Subclasses are automatically made into frozen dataclasses, so you can
    simply write::

        class DocumentReceived(Event):
            doc_id: str
            content: str

    The ``@dataclass(frozen=True)`` decorator is no longer needed — it is
    applied automatically by ``__init_subclass__``.
    """

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        # Auto-apply @dataclass(frozen=True) to every Event subclass.
        # Check cls.__dict__ (not hasattr) because inherited
        # __dataclass_fields__ from parent dataclasses would give a false
        # positive with dataclasses.is_dataclass().
        if "__dataclass_fields__" not in cls.__dict__:
            dataclasses.dataclass(frozen=True)(cls)

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


class MessageEvent(Event):
    """Mixin for events that wrap LangChain messages.

    Convention:
    - ``message`` field (single BaseMessage) → ``[self.message]``
    - ``messages`` field (tuple of BaseMessage) → ``list(self.messages)``
    - Override ``as_messages()`` for custom behavior.

    Example::

        class UserMessageReceived(MessageEvent, Auditable):
            message: HumanMessage

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


class Auditable(Event):
    """Marker class for events that should be auto-logged.

    Inherit from this class to make events auditable. Use the ``trail()``
    method (or the ``@on(Auditable)`` subscription pattern) to log events
    as they flow through the graph.

    Example::

        class UserMessageReceived(Auditable):
            content: str = ""
    """

    def trail(self) -> str:
        """Return a compact, human-readable summary of this event."""
        name = type(self).__name__
        parts = []
        for f in dc_fields(self):  # type: ignore[arg-type]
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


class Halted(Event):
    """Special event that signals immediate graph termination.

    Subclass with domain-specific fields instead of generic payloads::

        class ContentBlocked(Halted):
            category: str

        class BudgetExceeded(Halted):
            spent: float
            limit: float
    """


class MaxRoundsExceeded(Halted):
    """Graph exceeded the configured ``max_rounds`` limit."""

    rounds: int = 0


class Cancelled(Halted):
    """Handler execution was cancelled (e.g. task timeout).

    When multiple handlers run in parallel, cancellation only discards
    the cancelled handler's partial events — sibling handler results
    that already committed to state will persist.
    """


class Interrupted(Event):
    """Special event that pauses the graph for human input.

    Subclass with domain-specific fields instead of generic payloads::

        class ApprovalRequested(Interrupted):
            draft: str
            revision: int

    When a handler returns an ``Interrupted`` subclass, the framework
    calls LangGraph's ``interrupt()`` and the graph pauses.  Resume with
    ``graph.resume(event)`` to continue — the event is auto-dispatched
    and a ``Resumed`` event is created alongside it.
    """

    def _collect_into(
        self,
        new_events: list[Event],
        interrupt_fn: Any,
    ) -> None:
        """Record the interrupt, pause, and create a Resumed event."""
        new_events.append(self)
        resume_value = interrupt_fn(self)
        if not isinstance(resume_value, Event):
            got = type(resume_value).__name__
            raise TypeError(f"resume() requires an Event instance, got {got}")
        new_events.append(resume_value)
        new_events.append(Resumed(value=resume_value, interrupted=self))  # type: ignore[call-arg]


class Resumed(Event):
    """Created by the framework when a graph resumes from an interrupt.

    Contains the resume event and a reference to the original
    ``Interrupted`` event.
    """

    value: Event | None = None
    interrupted: Interrupted | None = None


class HandlerRaised(Event):
    """Emitted when a handler raises an exception declared in its ``raises=`` clause.

    The raising handler's ``@on(..., raises=(MyError, ...))`` declares which
    exceptions the framework should catch.  Subscribe via the existing
    field-matcher mechanism::

        @on(HandlerRaised, exception=RateLimitError)
        def backoff(event: HandlerRaised, exception: RateLimitError):
            exception.retry_after  # typed via field injection

    ``@on(HandlerRaised)`` (no ``exception=`` matcher) catches every
    declared raise.
    """

    handler: str = ""
    event: Event | None = None
    exception: Exception = None  # type: ignore[assignment]


class SystemPromptSet(MessageEvent):
    """Built-in event for setting the system prompt as a first-class citizen.

    Wraps a ``SystemMessage`` so that the system prompt appears in the event
    log, is queryable via ``EventLog``, and participates in the ``Auditable``
    trail when mixed in.

    Pairs naturally with ``message_reducer()`` — the system message is
    automatically included in the accumulated message history via
    ``as_messages()``.

    Example::

        from langchain_core.messages import SystemMessage
        from langgraph_events import SystemPromptSet, EventGraph

        log = graph.invoke([
            SystemPromptSet(message=SystemMessage(content="You are helpful")),
            UserMessageReceived(message=HumanMessage(content="Hi")),
        ])

    Or as a convenience with a plain string::

        log = graph.invoke([
            SystemPromptSet.from_str("You are helpful"),
            UserMessageReceived(message=HumanMessage(content="Hi")),
        ])
    """

    message: SystemMessage = None  # type: ignore[assignment]

    @classmethod
    def from_str(cls, content: str) -> SystemPromptSet:
        """Create from a plain string, wrapping it in a ``SystemMessage``."""
        from langchain_core.messages import SystemMessage as SysMsg  # noqa: PLC0415

        return cls(message=SysMsg(content=content))  # type: ignore[call-arg]


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

    def __class_getitem__(cls, params: Any) -> types.GenericAlias:
        """Support ``Scatter[EventType]`` in type annotations."""
        if not isinstance(params, tuple):
            params = (params,)
        return types.GenericAlias(cls, params)

    def __init__(self, events: list[Event]) -> None:
        if not events:
            raise ValueError("Scatter requires at least one event")
        validated: list[Event] = []
        for e in events:
            if not isinstance(e, Event):
                raise TypeError(
                    f"Scatter events must be Event instances, got {type(e).__name__}"
                )
            validated.append(e)
        self.events = validated

    def _collect_into(
        self,
        new_events: list[Event],
        interrupt_fn: Any,
    ) -> None:
        """Expand scatter events into the collection list."""
        new_events.extend(self.events)
