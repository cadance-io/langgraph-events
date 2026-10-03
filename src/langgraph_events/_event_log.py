"""EventLog — query interface over the event list."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING, Any, TypeAlias, TypeVar, overload

from langgraph_events._causes import FRAMEWORK, NOT_RECORDED, dropped, resolve
from langgraph_events._event import Event

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator, Sequence

    from langgraph_events._causes import CauseEntry

T = TypeVar("T", bound=Event)


@dataclass(frozen=True)
class Cause:
    """The dispatch that wrote an event: the event a handler received, and
    the handler.

    Usually the handler returned the event. The framework also writes events
    for a dispatch: ``InvariantViolated`` when an invariant blocked it, and
    ``HandlerRaised`` or ``HandlerRetried`` when the handler raised. These
    carry the same ``Cause``. ``source`` is the event that the handler was
    called with. Event stores call it the causation ID. ``via`` is the
    handler's graph node name. For an inline command handler, ``via`` is the
    command qualname.
    """

    source: Event
    via: str

    def __post_init__(self) -> None:
        if not isinstance(self.via, str):
            raise TypeError(
                f"Cause.via must be a str, the handler's node name, got "
                f"{type(self.via).__name__}."
            )


class UnknownCause:
    """Base of every case where the cause of an event is not known.

    Each subclass names one case and keeps only the facts that are still
    true. ``reason`` states the case in one sentence. An unknown cause comes
    only from history that the framework did not record: a checkpoint saved
    before causes existed, or a source that ``rewrite_store(drop=...)``
    deleted.
    """

    __slots__ = ()

    @property
    def reason(self) -> str:
        raise NotImplementedError


@dataclass(frozen=True)
class NotRecorded(UnknownCause):
    """The event was written before this library recorded causes."""

    @property
    def reason(self) -> str:
        return "not recorded: the event was written before causes existed"


@dataclass(frozen=True)
class SourceDropped(UnknownCause):
    """A handler produced the event, but a store rewrite deleted its source.

    ``via`` is the handler. ``source_type`` is the qualname of the deleted
    event.
    """

    via: str
    source_type: str

    @property
    def reason(self) -> str:
        return f"source {self.source_type} dropped by rewrite_store, via {self.via}"


@dataclass(frozen=True)
class FrameworkEvent:
    """The framework wrote the event: ``RunPaused``, ``MaxRoundsExceeded``,
    ``Cancelled`` or ``Abandoned``. The event type names the mechanism."""


CauseValue: TypeAlias = "Cause | UnknownCause | FrameworkEvent | None"
"""What :meth:`EventLog.cause` returns. ``None`` is a seed."""


class _CauseTable:
    """The causes of one root log. Every log derived from it shares the table.

    The table reads the ``causes`` channel lazily: :func:`resolve` runs on the
    first query, not each time a handler receives the log. ``entries[i]`` is
    the absolute entry of ``events[i]`` in the format of
    :mod:`~langgraph_events._causes`. :func:`_decode` turns it into the public
    value.
    """

    def __init__(
        self,
        events: tuple[Event, ...],
        stored: list[Any] | None,
        entries: tuple[CauseEntry, ...] | None = None,
    ) -> None:
        self.events = events
        self._stored = stored
        if entries is not None:
            self.__dict__["entries"] = entries

    @cached_property
    def entries(self) -> tuple[CauseEntry, ...]:
        resolved, _known_from = resolve(self.events, self._stored)
        return tuple(resolved)

    @cached_property
    def positions(self) -> dict[int, int]:
        """``id(event)`` to its latest root index, for identity lookups only.

        Built on the first lookup, and left out of a pickle: an id is valid
        only for the objects of one process.
        """
        return {id(event): i for i, event in enumerate(self.events)}

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        state.pop("positions", None)
        return state

    def locate(self, event: Event) -> int:
        """The root index of *event*: identity first, then the one equal event.

        Raises ``ValueError`` when *event* is not in the root log, or when a
        copy matches more than one equal event: picking one would be a guess.
        """
        position = self.positions.get(id(event))
        if position is not None:
            return position
        equal = [
            i
            for i, candidate in enumerate(self.events)
            if type(candidate) is type(event) and candidate == event
        ]
        if len(equal) > 1:
            raise ValueError(
                f"event {type(event).__qualname__} is a copy that matches "
                f"{len(equal)} equal events in this log. Pass the logged object."
            )
        if not equal:
            raise ValueError(f"event {type(event).__qualname__} is not in this log")
        return equal[0]


def _cause_at(table: _CauseTable, position: int) -> CauseValue:
    """The cause of the root event at *position*, decoded from its entry."""
    return _decode(table.events, table.entries[position])


def _decode(events: tuple[Event, ...], entry: Any) -> CauseValue:
    if entry is None:
        return None
    tag = entry[0]
    if tag == "framework":
        return FrameworkEvent()
    if tag == "not_recorded":
        return NotRecorded()
    if tag == "dropped":
        return SourceDropped(via=entry[1], source_type=entry[2])
    return Cause(source=events[tag], via=entry[1])


def _is_handler_entry(entry: Any) -> bool:
    return entry is not None and isinstance(entry[0], int)


def _encode(i: int, cause: Any, positions: dict[int, int]) -> CauseEntry:
    if cause is None:
        return None
    if isinstance(cause, Cause):
        source = positions.get(id(cause.source))
        if source is None:
            raise ValueError(
                f"causes[{i}] names a {type(cause.source).__qualname__} source "
                f"that is not an earlier event in this log. The source must be "
                f"the same object as an earlier event: pass events[j], not a copy."
            )
        return (source, cause.via)
    if isinstance(cause, NotRecorded):
        return NOT_RECORDED
    if isinstance(cause, SourceDropped):
        return dropped(cause.via, cause.source_type)
    if isinstance(cause, FrameworkEvent):
        return FRAMEWORK
    raise TypeError(
        f"causes[{i}] must be a Cause, an UnknownCause, a FrameworkEvent or None, "
        f"got {type(cause).__name__}. Wrap a source event and a handler name: "
        f"Cause(source, via)."
    )


def _table_from_causes(
    events: tuple[Event, ...], causes: tuple[CauseValue, ...]
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
        entries.append(_encode(i, cause, positions))
        positions[id(event)] = i
    return _CauseTable(events, None, entries=tuple(entries))


class EventLog:
    """Immutable, ordered container of events with query methods.

    Returned by ``EventGraph.invoke()`` / ``EventGraph.ainvoke()``.
    All queries use ``isinstance`` so subclass events match parent types.
    Pass ``causes`` to record which handler produced each event. See
    :class:`Cause`.
    """

    __slots__ = ("_events", "_roots", "_table")

    def __init__(
        self,
        events: Iterable[Event],
        causes: Iterable[CauseValue] | None = None,
    ) -> None:
        self._events = tuple(events)
        self._table = (
            None if causes is None else _table_from_causes(self._events, tuple(causes))
        )
        self._roots: Sequence[int] | None = (
            None if causes is None else range(len(self._events))
        )

    @classmethod
    def _from_owned(
        cls,
        events: list[Any] | tuple[Any, ...],
        table: _CauseTable | None = None,
        roots: Sequence[int] | None = None,
    ) -> EventLog:
        """Create an EventLog from an already-built events sequence.

        *roots* holds the root index of each event, when *table* is given.
        """
        obj = object.__new__(cls)
        obj._events = events if isinstance(events, tuple) else tuple(events)
        obj._table = table
        obj._roots = roots
        return obj

    @classmethod
    def _from_state(
        cls,
        events: list[Any] | tuple[Any, ...],
        causes: list[Any] | None,
        *,
        check: bool = True,
    ) -> EventLog:
        """Build the log of a run from its ``events`` and ``causes`` channels.

        :func:`~langgraph_events._causes.resolve` aligns the channels and
        raises ``RuntimeError`` when a writer drifted. With ``check=False``
        it runs on the first query instead, so a handler that receives the
        log does not pay for it.
        """
        owned = tuple(events)
        table = _CauseTable(owned, causes)
        if check:
            table.entries  # noqa: B018
        return cls._from_owned(owned, table, range(len(owned)))

    def _derive(self, events: Sequence[Event], roots: Sequence[int] | None) -> EventLog:
        return EventLog._from_owned(tuple(events), self._table, roots)

    @property
    def causes(self) -> tuple[CauseValue, ...] | None:
        """The origin of each event, aligned with :attr:`events`.

        ``None`` when the log records no causes. Each entry is what
        :meth:`cause` returns for that event: a :class:`Cause`, an
        :class:`UnknownCause`, a :class:`FrameworkEvent`, or ``None`` for a
        seed. ``EventLog(log.events, causes=log.causes)`` rebuilds a root log. In a
        log from ``after``, ``before`` or ``select``, a source can be an event
        outside that log. Building a log from its ``events`` and ``causes``
        then raises ``ValueError``.
        """
        table, roots = self._table, self._roots
        if table is None or roots is None:
            return None
        return tuple(_cause_at(table, root) for root in roots)

    def cause(self, event: Event) -> CauseValue:
        """The origin of *event*, one case per type.

        - :class:`Cause`: a handler produced *event* from ``source``.
        - :class:`NotRecorded` or :class:`SourceDropped`: the cause is
          unknown, and the type says why. See :class:`UnknownCause`.
        - :class:`FrameworkEvent`: the framework wrote *event*.
        - ``None``: a seed. *event* came from outside.

        Finds *event* by identity first, then as its one equal event. A log
        from ``after``, ``before`` or ``select`` answers like its root log.
        Raises ``ValueError`` if the log records no causes, or if *event* is
        not in the root log.
        """
        table = self._require_table()
        return _cause_at(table, table.locate(event))

    def effects(self, event: Event) -> tuple[Event, ...]:
        """The events that a handler produced from *event*, in log order.

        Finds *event* and raises like :meth:`cause`.
        """
        table = self._require_table()
        position = table.locate(event)
        return tuple(
            table.events[i]
            for i, entry in enumerate(table.entries)
            if _is_handler_entry(entry) and entry[0] == position  # type: ignore[index]
        )

    def flow(self, event: Event) -> tuple[Event, ...]:
        """The chain of :class:`Cause` that ends at *event*, oldest first.

        The chain starts at the first event whose own cause is not a
        :class:`Cause`: a seed, a framework event, or an unknown cause.
        Finds *event* and raises like :meth:`cause`.
        """
        table = self._require_table()
        chain = [table.locate(event)]
        while True:
            entry = table.entries[chain[-1]]
            if entry is None or not isinstance(entry[0], int):
                break
            chain.append(entry[0])
        return tuple(table.events[i] for i in reversed(chain))

    def _require_table(self) -> _CauseTable:
        if self._table is None:
            raise ValueError(
                "this log records no causes. A graph run records them. Rebuild "
                "a saved log with EventLog(events, causes=...)."
            )
        return self._table

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
        roots = self._roots
        for i, e in enumerate(self._events):
            if isinstance(e, event_type):
                after = None if roots is None else roots[i + 1 :]
                return self._derive(self._events[i + 1 :], after)
        return self._derive((), None if roots is None else ())

    def before(self, event_type: type[Event]) -> EventLog:
        """Return an ``EventLog`` of events before the first *event_type*."""
        roots = self._roots
        for i, e in enumerate(self._events):
            if isinstance(e, event_type):
                return self._derive(
                    self._events[:i], None if roots is None else roots[:i]
                )
        return self._derive((), None if roots is None else ())

    def select(self, event_type: type[T]) -> EventLog:
        """Like ``filter()`` but returns an ``EventLog`` for chaining."""
        keep = [i for i, e in enumerate(self._events) if isinstance(e, event_type)]
        roots = self._roots
        return self._derive(
            [self._events[i] for i in keep],
            None if roots is None else [roots[i] for i in keep],
        )

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
