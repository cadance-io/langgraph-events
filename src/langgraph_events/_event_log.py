"""EventLog — query interface over the event list."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, TypeVar, overload

from langgraph_events._causes import resolve
from langgraph_events._event import Event

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

    from langgraph_events._causes import CauseEntry

T = TypeVar("T", bound=Event)


@dataclass(frozen=True)
class Cause:
    """The handler that produced an event, and the event it received.

    ``source`` is the event that the handler was called with. Event stores
    call it the causation ID. ``via`` is the handler's graph node name. For
    an inline command handler, ``via`` is the command qualname.
    """

    source: Event
    via: str


@dataclass(frozen=True, eq=False)
class _CauseTable:
    """The causes of one root log. Every log derived from it shares the table.

    ``entries[i]`` is ``(source index, via)`` for ``events[i]``, or ``None``
    for a seed. ``positions`` maps ``id(event)`` to its latest index. An
    event below ``known_from`` has an unknown cause: a checkpoint saved
    before causes existed did not record it.
    """

    events: tuple[Event, ...]
    entries: tuple[CauseEntry, ...]
    positions: dict[int, int]
    known_from: int = 0

    def locate(self, event: Event) -> int:
        """The root index of *event*: identity first, then the latest equal event.

        The same lookup as ``Reflection``. Raises ``ValueError`` when *event*
        is not in the root log.
        """
        position = self.positions.get(id(event))
        if position is not None:
            return position
        for i in range(len(self.events) - 1, -1, -1):
            candidate = self.events[i]
            if type(candidate) is type(event) and candidate == event:
                return i
        raise ValueError(f"event {type(event).__name__} is not in this log")


def _cause_at(table: _CauseTable, position: int) -> Cause | None:
    """The :class:`Cause` of the root event at *position*, or ``None``."""
    entry = table.entries[position]
    if entry is None:
        return None
    source, via = entry
    return Cause(source=table.events[source], via=via)


def _table_from_causes(
    events: tuple[Event, ...], causes: tuple[Cause | None, ...]
) -> _CauseTable:
    """Check *causes* against *events*, and store each source as an index."""
    if len(causes) != len(events):
        raise ValueError(
            f"causes has {len(causes)} entries, but the log has {len(events)} "
            f"events. Pass one entry per event: a Cause, or None for a seed."
        )
    positions: dict[int, int] = {}
    entries: list[CauseEntry] = []
    for i, (event, cause) in enumerate(zip(events, causes, strict=True)):
        if cause is None:
            entries.append(None)
        elif not isinstance(cause, Cause):
            raise TypeError(
                f"causes[{i}] must be a Cause or None, got "
                f"{type(cause).__name__}. Wrap the source event and the handler "
                f"name: Cause(source, via)."
            )
        else:
            source = positions.get(id(cause.source))
            if source is None:
                raise ValueError(
                    f"causes[{i}] names a {type(cause.source).__name__} source "
                    f"that is not an earlier event in this log. The source must "
                    f"be the same object as an earlier event."
                )
            entries.append((source, cause.via))
        positions[id(event)] = i
    return _CauseTable(events, tuple(entries), positions)


class EventLog:
    """Immutable, ordered container of events with query methods.

    Returned by ``EventGraph.invoke()`` / ``EventGraph.ainvoke()``.
    All queries use ``isinstance`` so subclass events match parent types.
    Pass ``causes`` to record which handler produced each event. See
    :class:`Cause`.
    """

    __slots__ = ("_events", "_table")

    def __init__(
        self,
        events: Iterable[Event],
        causes: Iterable[Cause | None] | None = None,
    ) -> None:
        self._events = tuple(events)
        self._table = (
            None if causes is None else _table_from_causes(self._events, tuple(causes))
        )

    @classmethod
    def _from_owned(
        cls,
        events: list[Any] | tuple[Any, ...],
        table: _CauseTable | None = None,
    ) -> EventLog:
        """Create an EventLog from an already-built events sequence."""
        obj = object.__new__(cls)
        obj._events = events if isinstance(events, tuple) else tuple(events)
        obj._table = table
        return obj

    @classmethod
    def _from_state(
        cls, events: list[Any] | tuple[Any, ...], causes: list[Any] | None
    ) -> EventLog:
        """Build the log of a run from its ``events`` and ``causes`` channels.

        :func:`~langgraph_events._causes.resolve` aligns the channels and
        raises ``RuntimeError`` when a writer drifted.
        """
        owned = tuple(events)
        entries, known_from = resolve(owned, causes)
        table = _CauseTable(
            owned,
            tuple(entries),
            {id(event): i for i, event in enumerate(owned)},
            known_from,
        )
        return cls._from_owned(owned, table)

    @property
    def causes(self) -> tuple[Cause | None, ...] | None:
        """The cause of each event, aligned with :attr:`events`.

        ``None`` when the log records no causes. Each entry is a
        :class:`Cause`, or ``None`` for an event without a recorded cause.
        ``EventLog(log.events, causes=log.causes)`` rebuilds a root log. In a
        log from ``after``, ``before`` or ``select``, a source can be an event
        outside that log. Building a log from its ``events`` and ``causes``
        then raises ``ValueError``.
        """
        table = self._table
        if table is None:
            return None
        return tuple(
            _cause_at(table, table.positions[id(event)]) for event in self._events
        )

    def cause(self, event: Event) -> Cause | None:
        """The handler that produced *event*, and the event it received.

        Returns ``None`` for a seed, and for an event whose cause was not
        recorded. Finds *event* by identity first, then as the latest equal
        event. A log from ``after``, ``before`` or ``select`` answers like
        its root log. Raises ``ValueError`` if the log records no causes, or
        if *event* is not in the root log.
        """
        table = self._require_table()
        return _cause_at(table, table.locate(event))

    def effects(self, event: Event) -> tuple[Event, ...]:
        """The events that *event* caused, in log order.

        Finds *event* and raises like :meth:`cause`.
        """
        table = self._require_table()
        position = table.locate(event)
        return tuple(
            table.events[i]
            for i, entry in enumerate(table.entries)
            if entry is not None and entry[0] == position
        )

    def flow(self, event: Event) -> tuple[Event, ...]:
        """The cause chain of *event*, from the root seed to *event*.

        The chain starts at the first event that has no recorded cause.
        Finds *event* and raises like :meth:`cause`.
        """
        table = self._require_table()
        chain = [table.locate(event)]
        while (entry := table.entries[chain[-1]]) is not None:
            chain.append(entry[0])
        return tuple(table.events[i] for i in reversed(chain))

    def _require_table(self) -> _CauseTable:
        if self._table is None:
            raise ValueError(
                "this log records no causes. A graph run records them. Rebuild "
                "a saved log with EventLog(events, causes=...)."
            )
        return self._table

    def _cause_is_known(self, event: Event) -> bool:
        """Whether the cause of *event* was recorded.

        This is the only private member of ``EventLog`` that ``Reflection``
        uses. ``cause()`` returns ``None`` both for a seed and for an older
        event of a checkpoint saved before causes existed. Reflection must
        show ``unknown`` for the second, and never a guessed ``seed``.
        Finds *event* and raises like :meth:`cause`.
        """
        table = self._require_table()
        return table.locate(event) >= table.known_from

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
                return EventLog._from_owned(self._events[i + 1 :], self._table)
        return EventLog._from_owned((), self._table)

    def before(self, event_type: type[Event]) -> EventLog:
        """Return an ``EventLog`` of events before the first *event_type*."""
        for i, e in enumerate(self._events):
            if isinstance(e, event_type):
                return EventLog._from_owned(self._events[:i], self._table)
        return EventLog._from_owned((), self._table)

    def select(self, event_type: type[T]) -> EventLog:
        """Like ``filter()`` but returns an ``EventLog`` for chaining."""
        filtered = [e for e in self._events if isinstance(e, event_type)]
        return EventLog._from_owned(filtered, self._table)

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
