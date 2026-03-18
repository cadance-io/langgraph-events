"""EventLog — query interface over the event list."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypeVar, overload

from langgraph_events._event import Event

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

T = TypeVar("T", bound=Event)


class EventLog:
    """Immutable, ordered container of events with query methods.

    Returned by ``EventGraph.invoke()`` / ``EventGraph.ainvoke()``.
    All queries use ``isinstance`` so subclass events match parent types.
    """

    __slots__ = ("_events",)

    def __init__(self, events: Iterable[Event]) -> None:
        self._events = tuple(events)

    @classmethod
    def _from_owned(cls, events: list[Any] | tuple[Any, ...]) -> EventLog:
        """Create an EventLog from an already-built events sequence."""
        obj = object.__new__(cls)
        obj._events = tuple(events)
        return obj

    def filter(self, event_type: type[T]) -> list[T]:
        """Return all events matching *event_type* (including subclasses)."""
        return [e for e in self._events if isinstance(e, event_type)]

    def latest(self, event_type: type[T]) -> T | None:
        """Return the most recent event of *event_type*, or ``None``."""
        for e in reversed(self._events):
            if isinstance(e, event_type):
                return e
        return None

    def has(self, event_type: type[Event]) -> bool:
        """Return ``True`` if any event of *event_type* exists."""
        return any(isinstance(e, event_type) for e in self._events)

    def first(self, event_type: type[T]) -> T | None:
        """Return the earliest event of *event_type*, or ``None``."""
        for e in self._events:
            if isinstance(e, event_type):
                return e
        return None

    def count(self, event_type: type[Event]) -> int:
        """Return the number of events matching *event_type*."""
        return sum(1 for e in self._events if isinstance(e, event_type))

    def after(self, event_type: type[Event]) -> EventLog:
        """Return an ``EventLog`` of events after the first *event_type*."""
        for i, e in enumerate(self._events):
            if isinstance(e, event_type):
                return EventLog._from_owned(self._events[i + 1 :])
        return EventLog._from_owned([])

    def before(self, event_type: type[Event]) -> EventLog:
        """Return an ``EventLog`` of events before the first *event_type*."""
        for i, e in enumerate(self._events):
            if isinstance(e, event_type):
                return EventLog._from_owned(self._events[:i])
        return EventLog._from_owned([])

    def select(self, event_type: type[T]) -> EventLog:
        """Like ``filter()`` but returns an ``EventLog`` for chaining."""
        filtered = [e for e in self._events if isinstance(e, event_type)]
        return EventLog._from_owned(filtered)

    @property
    def events(self) -> tuple[Event, ...]:
        """The events in this log as an immutable tuple."""
        return self._events

    # --- container protocol ---

    def __len__(self) -> int:
        return len(self._events)

    def __bool__(self) -> bool:
        return bool(self._events)

    def __iter__(self) -> Iterator[Event]:
        return iter(self._events)

    @overload
    def __getitem__(self, index: int) -> Event: ...

    @overload
    def __getitem__(self, index: slice) -> list[Event]: ...

    def __getitem__(self, index: int | slice) -> Event | list[Event]:
        if isinstance(index, slice):
            return list(self._events[index])
        return self._events[index]

    def __repr__(self) -> str:
        n = len(self._events)
        if n <= 5:
            return f"EventLog({self._events!r})"
        first = type(self._events[0]).__name__
        last = type(self._events[-1]).__name__
        return f"EventLog({n} events, {first} .. {last})"
