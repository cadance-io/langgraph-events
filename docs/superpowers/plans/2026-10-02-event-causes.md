# Event Causes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Record, for each event that a handler returns, the event that the handler received and the handler's graph node name, and expose the record as `EventLog.cause()`, `effects()`, `flow()`, `causes` and in `Reflection`.

**Architecture:** The graph state gets a `causes` channel next to `events`, written by every writer to `events` in the same order. One module, `_causes.py`, owns the storage format: the entry alias `CauseEntry` and `resolve()`, which applies the relative-source rule and the align-from-the-end rule. `EventLog` turns the resolved indices into a shared root table and answers by event, never by index. One helper, `pad_causes()`, pads every writer to `events` outside a handler call.

**Tech Stack:** Python 3.11, LangGraph state channels (`operator.add`), `MemorySaver` with `NamespaceAwareSerde`, pytest-describe, ruff, mypy strict, `uv`.

**Spec:** docs/superpowers/specs/2026-10-02-event-causes-design.md

## Global Constraints

- Run every tool through `uv run`. Do not call bare `python` or `pytest`.
- Line length is 88. Ruff lints and formats. mypy runs in strict mode on `src/`.
- Do not use `assert` in `src/`: ruff `S101` rejects it. An internal invariant raises `RuntimeError`.
- Tests use pytest-describe: `describe_` per API surface, `when_` per code branch, `it_` per assertion.
- A test, helper or handler name in `tests/` must not contain `_with_`, `_when_` or `_without_`. `scripts/validate_tests.py` rejects it.
- A `describe_` or `when_` block must not hold both `when_*` blocks and `it_*` functions.
- An event class used as a handler annotation must be defined at module level.
- A test asserts through the public API: `EventLog`, `EventGraph`, `Reflection`, `graph.compiled`. A test does not read a checkpoint's `channel_values`.
- `_causes.py` owns the storage format. Every module spells one entry as `CauseEntry`. Only `_causes.resolve` reads a relative source.
- The `causes` channel stores plain tuples `(source, via)` or `None`. It never stores a `Cause` object.
- `pad_causes()` in `_internal.py` is the one helper that pads a state update that writes `events` without causes.
- `via` is `HandlerMeta.node_name`. `via` equals `Edge.via` unless the handler has a stable identity: an inline command handler, or an `@on(node_name=...)` pin.
- The public API takes and returns events. No log index crosses from one `EventLog` to another.
- `Reflection` reads causes through the public `EventLog` API and one private method, `EventLog._cause_is_known`.
- Add release notes under `## [Unreleased]` in `CHANGELOG.md`. Do not edit a version string.
- jscpd fails above 5% duplication over `src/` and `tests/`. Reuse the test helpers that this plan defines.
- A test that passes at its first run is a pin. Its Step 2 expects PASS and states the reason in one line.
- Every commit message has a conventional prefix, is in STE, and ends with a blank line and `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

## Review Focus

1. Two equal events that are different objects, such as two `Tick()` seeds: each answers its own cause, and a constructor `Cause` whose source is only equal to an earlier event raises.
2. A client that rebuilds a log from its own storage and passes `(index, via)` tuples in `causes=`: the constructor raises `TypeError` that names `Cause`.
3. `pre_seed()` that writes events to a paused thread (the AG-UI resume path): the causes of the older events stay correct after the resume.
4. A thread that paused before the upgrade and resumes after it: the resume completes, and the events of the resumed handler have no cause.
5. `rewrite_store(drop=...)` that drops an event below the pending events of a paused thread: after the resume, the cause of the resumed handler's output is its real trigger.

Pinned by:

- Item 1: Task 1 `describe_EventLog_causes_argument.when_a_source_is_only_equal_to_an_earlier_event`, and Task 2 `describe_cause.when_equal_events_repeat`.
- Item 2: Task 1 `describe_EventLog_causes_argument.when_an_entry_is_not_a_cause`.
- Item 3: Task 6 `describe_pre_seed.when_events_are_written_to_a_paused_thread`.
- Item 4: Task 8 `describe_resume.when_the_thread_paused_before_causes_existed`.
- Item 5: Task 9 `describe_rewrite_store.when_a_dropped_event_sits_below_a_paused_handler`.

---

### Task 1: `Cause`, the `EventLog` causes argument and `EventLog.causes`

**Files:**
- Modify: `src/langgraph_events/_event_log.py:1-112`
- Modify: `src/langgraph_events/__init__.py:37`, `src/langgraph_events/__init__.py:98-113`
- Test: `tests/test_event_log.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `Cause(source: Event, via: str)`, a frozen dataclass, exported from `langgraph_events`.
  - `EventLog.__init__(self, events: Iterable[Event], causes: Iterable[Cause | None] | None = None)`.
  - `EventLog.causes -> tuple[Cause | None, ...] | None` (property).
  - `_CauseTable(events: tuple[Event, ...], entries: tuple[tuple[int, str] | None, ...], positions: dict[int, int])`.
  - `_cause_at(table: _CauseTable, position: int) -> Cause | None`.
  - `EventLog._from_owned(events, table: _CauseTable | None = None) -> EventLog`.
  - `EventLog._table: _CauseTable | None`.

- [ ] **Step 1: Write the failing test**

In `tests/test_event_log.py`, replace the import line `from langgraph_events import Event, EventLog, IntegrationEvent` with:

```python
from langgraph_events import Cause, Event, EventLog, IntegrationEvent
```

Append to the end of `tests/test_event_log.py`:

```python
def describe_EventLog_causes_argument():
    def when_a_source_comes_later():
        def it_raises_value_error():
            first = Alpha(v=1)
            later = Beta(v=2)

            with pytest.raises(ValueError, match="not an earlier event"):
                EventLog([first, later], causes=[Cause(later, "h"), None])

    def when_a_source_is_the_event_itself():
        def it_raises_value_error():
            seed = Alpha(v=1)

            with pytest.raises(ValueError, match="not an earlier event"):
                EventLog([seed], causes=[Cause(seed, "h")])

    def when_a_source_is_only_equal_to_an_earlier_event():
        def it_raises_value_error():
            seed = Alpha(v=1)

            with pytest.raises(ValueError, match="same object"):
                EventLog(
                    [seed, Beta(v=2)], causes=[None, Cause(Alpha(v=1), "h")]
                )

    def when_an_entry_is_not_a_cause():
        def it_raises_type_error():
            with pytest.raises(TypeError, match="must be a Cause or None"):
                EventLog([Alpha(v=1), Beta(v=2)], causes=[None, (0, "h")])

    def when_the_lengths_differ():
        def it_raises_value_error():
            with pytest.raises(ValueError, match="one entry per event"):
                EventLog([Alpha(v=1), Beta(v=2)], causes=[None])


def describe_causes():
    def when_causes_are_given():
        def it_returns_one_entry_per_event():
            seed = Alpha(v=1)

            log = EventLog([seed, Beta(v=2)], causes=[None, Cause(seed, "h")])

            assert log.causes == (None, Cause(source=seed, via="h"))

        def it_rebuilds_the_same_log():
            seed = Alpha(v=1)
            log = EventLog([seed, Beta(v=2)], causes=[None, Cause(seed, "h")])

            rebuilt = EventLog(log.events, causes=log.causes)

            assert rebuilt.causes == log.causes
            assert rebuilt.causes[1].source is seed

    def when_causes_are_omitted():
        def it_is_none():
            assert EventLog([Alpha(v=1)]).causes is None

    def when_the_log_derives_from_a_log_that_has_causes():
        def it_keeps_the_causes_of_its_events():
            seed = Alpha(v=1)
            log = EventLog([seed, Beta(v=2)], causes=[None, Cause(seed, "h")])

            assert log.select(Beta).causes == (Cause(source=seed, via="h"),)
            assert log.before(Beta).causes == (None,)

        def it_cannot_rebuild_a_log_whose_source_is_outside_it():
            seed = Alpha(v=1)
            log = EventLog([seed, Beta(v=2)], causes=[None, Cause(seed, "h")])
            derived = log.select(Beta)

            with pytest.raises(ValueError, match="not an earlier event"):
                EventLog(derived.events, causes=derived.causes)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_event_log.py -q -p no:cacheprovider`
Expected: FAIL with "ImportError: cannot import name 'Cause' from 'langgraph_events'".

- [ ] **Step 3: Write minimal implementation**

Replace the whole of `src/langgraph_events/_event_log.py` with:

```python
"""EventLog — query interface over the event list."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, TypeVar, overload

from langgraph_events._event import Event

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

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
    for a seed. ``positions`` maps ``id(event)`` to its latest index.
    """

    events: tuple[Event, ...]
    entries: tuple[tuple[int, str] | None, ...]
    positions: dict[int, int]


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
    entries: list[tuple[int, str] | None] = []
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
```

In `src/langgraph_events/__init__.py`, replace line 37 `from langgraph_events._event_log import EventLog` with:

```python
from langgraph_events._event_log import Cause, EventLog
```

In the `__all__` list of `src/langgraph_events/__init__.py`, add `"Cause",` directly after `"Cancelled",`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_event_log.py -q -p no:cacheprovider`
Expected: PASS. Then run the full suite `uv run pytest tests/ -q -p no:cacheprovider`, `uv run ruff check src/ tests/`, `uv run ruff format src/ tests/`, `uv run mypy src/`.

- [ ] **Step 5: Commit**

```bash
git add src/langgraph_events/_event_log.py src/langgraph_events/__init__.py tests/test_event_log.py
git commit -m "feat: add Cause, the causes argument and EventLog.causes

EventLog(events, causes=...) checks each cause. A source must be the
same object as an earlier event in the log. A derived log shares the
cause table of its root. The causes property is the inverse of the
constructor for a root log.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 2: `cause`, `effects` and `flow` on a log

**Files:**
- Modify: `src/langgraph_events/_event_log.py` (the `_CauseTable` class and the `EventLog` class from Task 1)
- Modify: `docs/api.md:68`, `docs/concepts.md:224-244`
- Test: `tests/test_event_log.py`

**Interfaces:**
- Consumes: `Cause`, `_CauseTable`, `_cause_at`, `EventLog._table` from Task 1.
- Produces:
  - `_CauseTable.locate(self, event: Event) -> int`. Identity first, then the latest equal event. Raises `ValueError` ("is not in this log").
  - `EventLog.cause(self, event: Event) -> Cause | None`.
  - `EventLog.effects(self, event: Event) -> tuple[Event, ...]`.
  - `EventLog.flow(self, event: Event) -> tuple[Event, ...]`.
  - `EventLog._require_table(self) -> _CauseTable`. Raises `ValueError` ("this log records no causes").

- [ ] **Step 1: Write the failing test**

Append to `tests/test_event_log.py`:

```python
def _caused_log():
    """seed causes reply and other. reply causes echo."""
    seed = Alpha(v=1)
    reply = Beta(v=2)
    echo = Alpha(v=3)
    other = Beta(v=4)
    log = EventLog(
        [seed, reply, echo, other],
        causes=[
            None,
            Cause(seed, "reply_h"),
            Cause(reply, "echo_h"),
            Cause(seed, "other_h"),
        ],
    )
    return log, seed, reply, echo, other


def _repeated_log():
    """Two equal Alpha(v=1) objects. The second one has a cause."""
    first = Alpha(v=1)
    reply = Beta(v=2)
    again = Alpha(v=1)
    log = EventLog(
        [first, reply, again],
        causes=[None, Cause(first, "h"), Cause(reply, "g")],
    )
    return log, first, reply, again


def describe_cause():
    def when_the_event_has_a_cause():
        def it_returns_the_source_event_and_the_handler():
            log, seed, reply, _echo, _other = _caused_log()

            cause = log.cause(reply)

            assert cause == Cause(source=Alpha(v=1), via="reply_h")
            assert cause.source is seed

    def when_the_event_is_a_seed():
        def it_returns_none():
            log, seed, _reply, _echo, _other = _caused_log()

            assert log.cause(seed) is None

    def when_equal_events_repeat():
        def it_finds_each_instance_by_identity():
            log, first, reply, again = _repeated_log()

            assert log.cause(first) is None
            assert log.cause(again) == Cause(source=reply, via="g")

        def it_falls_back_to_the_latest_equal_event():
            log, _first, reply, _again = _repeated_log()

            assert log.cause(Alpha(v=1)) == Cause(source=reply, via="g")

    def when_the_event_is_not_in_the_log():
        def it_raises_value_error():
            log, _seed, _reply, _echo, _other = _caused_log()

            with pytest.raises(ValueError, match="not in this log"):
                log.cause(Beta(v=99))

    def when_the_log_records_no_causes():
        def it_raises_value_error():
            seed = Alpha(v=1)

            with pytest.raises(ValueError, match="records no causes"):
                EventLog([seed]).cause(seed)

    def when_the_log_derives_from_a_log():
        def it_answers_like_the_root():
            log, seed, reply, echo, other = _caused_log()

            assert log.after(Beta).cause(echo).source is reply
            assert log.select(Beta).cause(other).source is seed


def describe_effects():
    def when_the_event_caused_events():
        def it_returns_them_in_log_order():
            log, seed, reply, _echo, other = _caused_log()

            assert log.effects(seed) == (reply, other)

    def when_the_event_caused_nothing():
        def it_returns_an_empty_tuple():
            log, _seed, _reply, _echo, other = _caused_log()

            assert log.effects(other) == ()

    def when_the_log_records_no_causes():
        def it_raises_value_error():
            seed = Alpha(v=1)

            with pytest.raises(ValueError, match="records no causes"):
                EventLog([seed]).effects(seed)


def describe_flow():
    def when_the_event_has_a_cause_chain():
        def it_returns_the_chain_from_the_root_seed():
            log, seed, reply, echo, _other = _caused_log()

            assert log.flow(echo) == (seed, reply, echo)

    def when_the_event_is_a_seed():
        def it_returns_the_seed_alone():
            log, seed, _reply, _echo, _other = _caused_log()

            assert log.flow(seed) == (seed,)

    def when_the_log_records_no_causes():
        def it_raises_value_error():
            seed = Alpha(v=1)

            with pytest.raises(ValueError, match="records no causes"):
                EventLog([seed]).flow(seed)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_event_log.py -q -p no:cacheprovider`
Expected: FAIL with "AttributeError: 'EventLog' object has no attribute 'cause'" (and the same for `effects` and `flow`).

- [ ] **Step 3: Write minimal implementation**

In `src/langgraph_events/_event_log.py`, replace the `_CauseTable` class with:

```python
@dataclass(frozen=True, eq=False)
class _CauseTable:
    """The causes of one root log. Every log derived from it shares the table.

    ``entries[i]`` is ``(source index, via)`` for ``events[i]``, or ``None``
    for a seed. ``positions`` maps ``id(event)`` to its latest index.
    """

    events: tuple[Event, ...]
    entries: tuple[tuple[int, str] | None, ...]
    positions: dict[int, int]

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
```

In the `EventLog` class, add these methods directly after the `causes` property:

```python
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
```

In `docs/api.md`, replace line 68:

```markdown
| `EventLog` | Class | Immutable query container (see [Concepts](concepts.md#eventlog)) |
```

with these two rows:

```markdown
| `EventLog` | Class | Immutable query container (see [Concepts](concepts.md#eventlog)). `cause(e)`, `effects(e)` and `flow(e)` read the causes that the log records. `causes` is the tuple of causes aligned with `events`, or `None` when the log records no causes. `EventLog(events, causes=...)` builds a log with causes, and `EventLog(log.events, causes=log.causes)` rebuilds a root log. A log from `after()`, `before()` or `select()` answers like its root |
| `Cause` | Frozen dataclass | `(source, via)`: the event that a handler received, and the handler's graph node name. Event stores call `source` the causation ID. `via` equals `Edge.via` unless the handler has a stable identity: an inline command handler, or an `@on(node_name=...)` pin. Returned by `EventLog.cause()` |
```

In `docs/concepts.md`, in the `## \`EventLog\`` table, add these rows directly after the row `| \`log.select(T)\` / \`log.after(T)\` / \`log.before(T)\` | chainable \`EventLog\` |`:

```markdown
| `log.cause(e)` | `Cause \| None`: the handler that produced `e`, and the event it received |
| `log.effects(e)` | `tuple[Event, ...]`: the events that `e` caused, in log order |
| `log.flow(e)` | `tuple[Event, ...]`: the cause chain, from the root seed to `e` |
| `log.causes` | `tuple[Cause \| None, ...] \| None`: one cause per event, or `None` when the log records no causes |
```

Then, in `docs/concepts.md`, insert this section directly before the line `## \`Namespace\` as a feature hub`:

````markdown
### Causes

A `Cause(source, via)` names the event that a handler received and the handler's graph node
name. `via` equals `Edge.via` unless the handler has a stable identity: an inline command
handler, or an `@on(node_name=...)` pin. `log.cause(e)` returns `None` for a seed. It raises
`ValueError` when the log records no causes, or when `e` is not in the log. The lookup uses
identity first, then the latest equal event. A log from `after()`, `before()` or `select()`
answers like the log it came from.

```python
fired = sum(1 for e in log if (c := log.cause(e)) and c.via == "hourly_wake_brief")
```

A client that saves events in its own format saves each cause with them. It rebuilds the log
with `EventLog(events, causes=...)`, one entry per event: a `Cause`, or `None` for a seed. Each
`source` must be the same object as an earlier event in `events`. `log.causes` gives the
entries back, so `EventLog(log.events, causes=log.causes)` rebuilds a root log. In a derived
log, a source can be outside that log, and the rebuild raises `ValueError`.

````

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_event_log.py tests/test_docs_code_fences.py -q -p no:cacheprovider`
Expected: PASS. Then run the full suite `uv run pytest tests/ -q -p no:cacheprovider`, `uv run ruff check src/ tests/`, `uv run mypy src/`.

- [ ] **Step 5: Commit**

```bash
git add src/langgraph_events/_event_log.py tests/test_event_log.py docs/api.md docs/concepts.md
git commit -m "feat: answer cause, effects and flow on an EventLog

cause() finds the event by identity first, then as the latest equal
event, and returns its Cause or None for a seed. effects() lists the
events that an event caused. flow() gives the chain from the root
seed. A log without causes raises ValueError.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 3: The `causes` channel, the storage owner and the seed gap fill

**Files:**
- Create: `src/langgraph_events/_causes.py`
- Modify: `src/langgraph_events/_event_log.py` (imports, the `_CauseTable` class, `_table_from_causes`, a new `EventLog._from_state` classmethod)
- Modify: `src/langgraph_events/_internal.py:46`, `:58-66`, `:90-91`, `:140-175`, `:284`
- Modify: `src/langgraph_events/_graph.py:38`, `:1250`, `:1510-1520`
- Create: `tests/test_event_causes.py`
- Test: `tests/test_event_causes.py`, `tests/test_event_graph.py:1479-1482`

**Interfaces:**
- Consumes: `_CauseTable`, `_cause_at`, `EventLog._from_owned`, `EventLog.causes` from Tasks 1 and 2.
- Produces:
  - `CauseEntry: TypeAlias = tuple[int, str] | None` in `_causes.py`.
  - `resolve(events: Sequence[Any], causes: Sequence[Any] | None) -> tuple[list[CauseEntry], int]` in `_causes.py`. Returns the absolute entries and `known_from`. Raises `RuntimeError` when `len(causes) > len(events)` or a source is not an earlier event.
  - `_absolute(position: int, entry: Sequence[Any]) -> tuple[int, str]` in `_causes.py`.
  - `_CauseTable.known_from: int = 0`.
  - `EventLog._from_state(cls, events: list[Any] | tuple[Any, ...], causes: list[Any] | None) -> EventLog`.
  - State channels `"causes": Annotated[list[CauseEntry], operator.add]` and `"_pending_base": int`.
  - Test helpers in `tests/test_event_causes.py`: handlers `step` (`Started -> Processed`) and `finish` (`Processed -> Ended`), `_config(thread_id: str) -> dict[str, Any]`, `_checkpointed(handlers: list[Any], thread_id: str) -> tuple[EventGraph, dict[str, Any], MemorySaver]`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_event_causes.py`:

```python
"""Graph runs record the cause of each event.

Spec: docs/superpowers/specs/2026-10-02-event-causes-design.md
"""

from typing import Any

import pytest
from conftest import Ended, Processed, Started
from langgraph.checkpoint.memory import MemorySaver

from langgraph_events import EventGraph, EventLog, Reducer, on
from langgraph_events.serde import NamespaceAwareSerde


@on(Started)
def step(event: Started) -> Processed:
    return Processed(data=event.data)


@on(Processed)
def finish(event: Processed) -> Ended:
    return Ended(result=event.data)


def _config(thread_id: str) -> dict[str, Any]:
    return {"configurable": {"thread_id": thread_id}}


def _checkpointed(
    handlers: list[Any], thread_id: str
) -> tuple[EventGraph, dict[str, Any], MemorySaver]:
    """A graph on a fresh MemorySaver, and the config of one thread."""
    saver = MemorySaver(serde=NamespaceAwareSerde())
    graph = EventGraph(handlers, checkpointer=saver)
    return graph, _config(thread_id), saver


def describe_invoke():
    def when_no_handler_reacts():
        def it_records_no_cause_for_the_seed():
            log = EventGraph([finish]).invoke(Started(data="x"))

            assert log.causes == (None,)

    def when_a_handler_reads_the_injected_log():
        def it_sees_the_causes_of_the_run():
            seen: list[Any] = []

            @on(Started)
            def read_log(event: Started, log: EventLog) -> None:
                seen.append(log.causes)

            EventGraph([read_log]).invoke(Started(data="x"))

            assert seen == [(None,)]

    def when_the_graph_has_reducers():
        def it_returns_the_causes():
            seen = Reducer(name="seen", event_type=Started, fn=lambda e: [e.data])

            log = EventGraph([finish], reducers=[seen]).invoke(Started(data="x"))

            assert log.causes == (None,)

    def when_the_causes_channel_holds_more_entries_than_events():
        def it_raises_runtime_error():
            graph, config, _saver = _checkpointed([finish], "longer")
            graph.compiled.update_state(
                config,
                {"events": [Started(data="a")], "causes": [None, None, None]},
                as_node="__seed__",
            )

            with pytest.raises(RuntimeError, match="3 entries for 2 events"):
                graph.invoke(Started(data="b"), config=config)

    def when_a_stored_source_is_not_an_earlier_event():
        def it_raises_runtime_error():
            graph, config, _saver = _checkpointed([finish], "bad-source")
            graph.compiled.update_state(
                config,
                {"events": [Started(data="a")], "causes": [(0, "h")]},
                as_node="__seed__",
            )

            with pytest.raises(RuntimeError, match="not an earlier event"):
                graph.invoke(Started(data="b"), config=config)


def describe_ainvoke():
    def when_no_handler_reacts():
        async def it_records_no_cause_for_the_seed():
            log = await EventGraph([finish]).ainvoke(Started(data="x"))

            assert log.causes == (None,)
```

In `tests/test_event_graph.py`, replace lines 1479-1482:

```python
            @pytest.mark.parametrize(
                "reserved_name",
                ["events", "_cursor", "_pending", "_round"],
            )
```

with:

```python
            @pytest.mark.parametrize(
                "reserved_name",
                ["events", "causes", "_cursor", "_pending", "_pending_base", "_round"],
            )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_event_causes.py tests/test_event_graph.py -q -p no:cacheprovider -k "causes or reserved"`
Expected: FAIL. The `causes` tests fail with "assert None == (None,)". The `RuntimeError` tests fail with "DID NOT RAISE". The reserved names `causes` and `_pending_base` fail with "DID NOT RAISE".

- [ ] **Step 3: Write minimal implementation**

Create `src/langgraph_events/_causes.py`:

```python
"""The storage format of the ``causes`` state channel.

``causes[i]`` describes ``events[i]``. This module owns the format: the
alias of one entry, and :func:`resolve`, which turns the stored entries into
absolute ones. ``EventLog`` and ``rewrite_store`` read the channel only
through :func:`resolve`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypeAlias

if TYPE_CHECKING:
    from collections.abc import Sequence

CauseEntry: TypeAlias = tuple[int, str] | None
"""One entry of the ``causes`` channel: ``(source, via)``, or ``None``.

A source is the log index of the event that the handler received.
"""


def resolve(
    events: Sequence[Any], causes: Sequence[Any] | None
) -> tuple[list[CauseEntry], int]:
    """The absolute entry of each event in *events*, and ``known_from``.

    The channels align from the end. A checkpoint saved before causes
    existed holds fewer causes than events. Each event below ``known_from``
    then has an unknown cause, and its entry is ``None``. Raises
    ``RuntimeError`` when a writer left the channels out of step: more
    causes than events, or a source that is not an earlier event.
    """
    stored = list(causes or ())
    known_from = len(events) - len(stored)
    if known_from < 0:
        raise RuntimeError(
            f"the causes channel holds {len(stored)} entries for {len(events)} "
            f"events. A writer to events left causes out of step. This is a "
            f"framework bug."
        )
    entries: list[CauseEntry] = [None] * known_from
    for position, entry in enumerate(stored, start=known_from):
        entries.append(None if entry is None else _absolute(position, entry))
    return entries, known_from


def _absolute(position: int, entry: Sequence[Any]) -> tuple[int, str]:
    source, via = entry
    if not 0 <= source < position:
        raise RuntimeError(
            f"the stored cause of event #{position} points to #{source}, which "
            f"is not an earlier event. A writer to events left causes out of "
            f"step. This is a framework bug."
        )
    return (source, via)
```

In `src/langgraph_events/_event_log.py`:

Replace the import lines from `from langgraph_events._event import Event` to the end of the `if TYPE_CHECKING:` block with:

```python
from langgraph_events._causes import resolve
from langgraph_events._event import Event

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

    from langgraph_events._causes import CauseEntry
```

Replace the `_CauseTable` class with:

```python
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
```

In `_table_from_causes`, replace the line `entries: list[tuple[int, str] | None] = []` with:

```python
    entries: list[CauseEntry] = []
```

In the `EventLog` class, add this classmethod directly after `_from_owned`:

```python
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
```

In `src/langgraph_events/_internal.py`:

Replace line 46 `from langgraph_events._event_log import EventLog` with:

```python
from langgraph_events._causes import CauseEntry
from langgraph_events._event_log import EventLog
```

Replace lines 58-66 (the comment and `_BASE_FIELDS`) with:

```python
# Base fields present on every graph (no reducers needed)
_BASE_FIELDS: dict[str, Any] = {
    "events": Annotated[list[Event], operator.add],
    # causes[i] describes events[i]. Every writer to events writes the same
    # number of entries to causes, in the same order. See _causes.py.
    "causes": Annotated[list[CauseEntry], operator.add],
    # The log index of _pending[0]. A handler adds k for the k-th pending event.
    "_pending_base": int,
    "_cursor": int,
    "_pending": list[Event],
    "_round": int,
    # Router-side gate: one RunPaused per /run regardless of fan-ins (#88).
    "_run_paused_emitted": bool,
}
```

Replace lines 90-91 (`_OutputState`) with:

```python
class _OutputState(TypedDict):
    events: list[Event]
    causes: list[CauseEntry]
```

Replace the `seed` function inside `make_seed_node` (lines 140-175) with:

```python
    def seed(state: StateDict) -> StateDict:
        prev_cursor = state.get("_cursor", 0)
        all_events = state["events"]
        new_events = all_events[prev_cursor:]
        recorded = len(state.get("causes") or [])
        # The input writes only events, so the seed fills the gap for it.
        gap: list[CauseEntry] = [None] * (len(all_events) - recorded)

        result: dict[str, Any] = {
            "causes": gap,
            "_cursor": len(all_events),
            "_pending": new_events,
            "_round": 0,
            "_run_paused_emitted": False,
        }
        if reds:
            if prev_cursor == 0:
                for name, r in reds.items():
                    existing = state.get(name)
                    # Channel defaults: [] for list channels, None for
                    # scalar channels.  Anything else means pre-seeded
                    # via update_state / pre_seed().
                    if existing is not None and existing != []:
                        # Channel already has data — only apply seed
                        # contributions so the channel reducer merges
                        # them with the existing value.
                        collected = r.collect(new_events)
                        if r.has_contributions(collected):
                            result[name] = collected
                    else:
                        # True first run — initialize from default +
                        # seed events.
                        result[name] = r.seed(new_events)
            elif new_events:
                # Subsequent run (checkpointer) — only process new events
                for name, r in reds.items():
                    collected = r.collect(new_events)
                    if r.has_contributions(collected):
                        result[name] = collected
        return result
```

In `_build_inject`, replace line 284 `log_view = EventLog(state["events"])` with:

```python
        log_view = EventLog._from_state(state["events"], state.get("causes"))
```

In `src/langgraph_events/_graph.py`:

Add this import directly above the `from langgraph_events._event_log import EventLog` line (line 38):

```python
from langgraph_events._causes import CauseEntry
```

Replace line 1250 `reducer_fields: dict[str, Any] = {"events": list[Event]}` with:

```python
            reducer_fields: dict[str, Any] = {
                "events": list[Event],
                "causes": list[CauseEntry],
            }
```

Replace `_run` and `_arun` (lines 1510-1520) with:

```python
    def _run(self, inp: Any, **kwargs: Any) -> EventLog:
        kwargs = self._apply_deadline_kwarg(kwargs)
        compiled = self._compile()
        result = compiled.invoke(inp, **kwargs)
        return EventLog._from_state(result["events"], result.get("causes"))

    async def _arun(self, inp: Any, **kwargs: Any) -> EventLog:
        kwargs = self._apply_deadline_kwarg(kwargs)
        compiled = self._compile()
        result = await compiled.ainvoke(inp, **kwargs)
        return EventLog._from_state(result["events"], result.get("causes"))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_event_causes.py tests/test_event_graph.py -q -p no:cacheprovider -k "causes or reserved"`
Expected: PASS. Then run the full suite `uv run pytest tests/ -q -p no:cacheprovider`, `uv run ruff check src/ tests/`, `uv run ruff format src/ tests/`, `uv run mypy src/`.

- [ ] **Step 5: Commit**

```bash
git add src/langgraph_events/_causes.py src/langgraph_events/_event_log.py src/langgraph_events/_internal.py src/langgraph_events/_graph.py tests/test_event_causes.py tests/test_event_graph.py
git commit -m "feat: add the causes channel and its storage owner

The graph state gets a causes channel next to events. _causes.py owns
its format and resolve(), which aligns the channels from the end and
raises RuntimeError when a writer drifts. The seed node fills the gap
for the input. invoke(), ainvoke() and the injected log return causes.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 4: The handler loop records the cause of each emission

**Files:**
- Modify: `src/langgraph_events/_internal.py` (`seed` in `make_seed_node`, the last `return` of `router` at `:224-228`, a new `pad_causes`, a new `_record_causes`, `_process_events_sync` and `_process_events_async` at `:643-712`, `make_handler_node` at `:745-833`)
- Modify: `CHANGELOG.md:8`, `docs/concepts.md` (the `### Causes` section from Task 2)
- Test: `tests/test_event_causes.py`

**Interfaces:**
- Consumes: `CauseEntry` (Task 3), the `causes` and `_pending_base` channels (Task 3), `EventLog._from_state` (Task 3), `EventLog.cause` (Task 2). Test helpers `step`, `finish` (Task 3).
- Produces:
  - `pad_causes(update: StateDict) -> StateDict` in `_internal.py`.
  - `_record_causes(new_causes: list[CauseEntry], new_events: list[Event], trigger: int, via: str) -> None`.
  - `_process_events_sync(meta, matching: list[tuple[int, Event]], state, inject, new_events, new_causes, lg_interrupt, return_contract=None, deadline=None) -> None`, and the same signature for `_process_events_async`.
  - The seed node and the router write `_pending_base`.

- [ ] **Step 1: Write the failing test**

In `tests/test_event_causes.py`, replace the import block with:

```python
import asyncio
from typing import Any

import pytest
from conftest import Ended, Order, Processed, Started
from langgraph.checkpoint.memory import MemorySaver

from langgraph_events import Cancelled, Cause, EventGraph, EventLog, Reducer, on
from langgraph_events.serde import NamespaceAwareSerde
```

Inside `describe_invoke`, add these `when_` blocks before `when_no_handler_reacts`:

```python
    def when_a_policy_reacts_to_a_seed():
        def it_records_the_trigger_and_the_handler():
            log = EventGraph([step, finish]).invoke(Started(data="x"))

            cause = log.cause(log.first(Processed))

            assert cause == Cause(source=Started(data="x"), via="step")
            assert cause.source is log.first(Started)

    def when_an_inline_command_handler_runs():
        def it_names_the_command_qualname():
            log = EventGraph([Order.Place]).invoke(Order.Place(customer_id="c1"))

            assert log.cause(log.first(Order.Place.Placed)) == Cause(
                source=Order.Place(customer_id="c1"), via="Order.Place"
            )

    def when_a_handler_pins_its_node_name():
        def it_names_the_pinned_node():
            @on(Processed, node_name="closer")
            def close(event: Processed) -> Ended:
                return Ended(result=event.data)

            log = EventGraph([step, close]).invoke(Started(data="x"))

            assert log.cause(log.first(Ended)).via == "closer"
```

Inside the existing `when_a_handler_reads_the_injected_log` block, add this test after `it_sees_the_causes_of_the_run`:

```python
        def it_finds_the_cause_of_its_trigger():
            seen: list[Cause | None] = []

            @on(Processed)
            def read_trigger(event: Processed, log: EventLog) -> None:
                seen.append(log.cause(event))

            EventGraph([step, read_trigger]).invoke(Started(data="x"))

            assert seen == [Cause(source=Started(data="x"), via="step")]
```

Inside `describe_ainvoke`, add these `when_` blocks after `when_no_handler_reacts`:

```python
    def when_a_policy_reacts_to_a_seed():
        async def it_records_the_trigger_and_the_handler():
            log = await EventGraph([step, finish]).ainvoke(Started(data="x"))

            assert log.cause(log.first(Ended)) == Cause(
                source=Processed(data="x"), via="finish"
            )

    def when_a_handler_is_cancelled():
        async def it_records_no_cause_for_the_cancellation():
            ready = asyncio.Event()

            @on(Processed)
            async def stall(event: Processed) -> Ended:
                ready.set()
                await asyncio.sleep(100)
                return Ended(result="never")

            task = asyncio.ensure_future(
                EventGraph([step, stall]).ainvoke(Started(data="x"))
            )
            await ready.wait()
            task.cancel()
            log = await task

            assert log.cause(log.first(Processed)) == Cause(
                source=log.first(Started), via="step"
            )
            assert log.cause(log.latest(Cancelled)) is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_event_causes.py -q -p no:cacheprovider`
Expected: FAIL. The policy, inline command, injected-trigger and cancellation tests fail with an `AssertionError` such as "assert None == Cause(source=Started(data='x'), via='step')": no handler writes a cause yet, so `resolve` reads the handler's events as unknown. The pinned-node test fails with "AttributeError: 'NoneType' object has no attribute 'via'".

- [ ] **Step 3: Write minimal implementation**

In `src/langgraph_events/_internal.py`:

In `seed`, replace the `result` dict head:

```python
        result: dict[str, Any] = {
            "causes": gap,
            "_cursor": len(all_events),
```

with:

```python
        result: dict[str, Any] = {
            "causes": gap,
            "_pending_base": prev_cursor,
            "_cursor": len(all_events),
```

Replace the last `return` of the `router` function (lines 224-228) with:

```python
        return {
            "_cursor": len(state["events"]),
            "_pending": new_events,
            "_pending_base": state["_cursor"],
            "_round": current_round,
        }
```

Add this function directly above `make_seed_node`:

```python
def pad_causes(update: StateDict) -> StateDict:
    """*update* plus one ``None`` cause for each event that it writes.

    Every writer to ``events`` outside a handler call goes through here: the
    router, the ``Cancelled`` path, ``abandon()`` and ``pre_seed()``. Without
    the padding, ``causes`` falls behind ``events``, and each older cause
    reads one position late.
    """
    events = update.get("events")
    if not events:
        return update
    return {**update, "causes": [None] * len(events)}
```

Add this function directly above `_process_events_sync` (line 643):

```python
def _record_causes(
    new_causes: list[CauseEntry],
    new_events: list[Event],
    trigger: int,
    via: str,
) -> None:
    """Record ``(trigger, via)`` for each event appended since the last call.

    Called after each handler call. An invariant rollback replaces the
    events of its call before this runs, so it leaves no cause behind.
    """
    new_causes.extend([(trigger, via)] * (len(new_events) - len(new_causes)))
```

Replace `_process_events_sync` and `_process_events_async` (lines 643-712) with:

```python
def _process_events_sync(
    meta: HandlerMeta,
    matching: list[tuple[int, Event]],
    state: StateDict,
    inject: dict[str, Any],
    new_events: list[Event],
    new_causes: list[CauseEntry],
    lg_interrupt: Any,
    return_contract: Any = None,
    deadline: float | None = None,
) -> None:
    """Per-event invocation loop for the sync dispatch path.

    Each *matching* entry pairs a pending event with its log index.
    """
    for trigger, event in matching:
        violation = _check_invariants(meta, event, state)
        if violation is not None:
            new_events.append(violation)
            _record_causes(new_causes, new_events, trigger, meta.node_name)
            continue
        call_inject = _inject_fields(meta, event, inject)
        attempt = 1
        while True:
            try:
                result = _invoke_sync_path(meta, event, call_inject)
            except meta.raises as exc:
                delay = _next_delay_or_give_up(
                    meta, event, exc, attempt, new_events, deadline
                )
                if delay is None:
                    break
                _retry._sleep(delay)
                attempt += 1
                continue
            _collect_and_check(
                result, new_events, lg_interrupt, meta, state, event, return_contract
            )
            break
        _record_causes(new_causes, new_events, trigger, meta.node_name)


async def _process_events_async(
    meta: HandlerMeta,
    matching: list[tuple[int, Event]],
    state: StateDict,
    inject: dict[str, Any],
    new_events: list[Event],
    new_causes: list[CauseEntry],
    lg_interrupt: Any,
    return_contract: Any = None,
    deadline: float | None = None,
) -> None:
    """Per-event invocation loop for the async dispatch path.

    Each *matching* entry pairs a pending event with its log index.
    """
    for trigger, event in matching:
        violation = _check_invariants(meta, event, state)
        if violation is not None:
            new_events.append(violation)
            _record_causes(new_causes, new_events, trigger, meta.node_name)
            continue
        call_inject = _inject_fields(meta, event, inject)
        attempt = 1
        while True:
            try:
                result = await _invoke_async_path(meta, event, call_inject)
            except meta.raises as exc:
                delay = _next_delay_or_give_up(
                    meta, event, exc, attempt, new_events, deadline
                )
                if delay is None:
                    break
                await _retry._asleep(delay)
                attempt += 1
                continue
            _collect_and_check(
                result, new_events, lg_interrupt, meta, state, event, return_contract
            )
            break
        _record_causes(new_causes, new_events, trigger, meta.node_name)
```

In `make_handler_node`, replace `_prepare` (lines 745-761) with:

```python
    def _prepare(
        state: StateDict, config: RunnableConfig
    ) -> tuple[list[tuple[int, Event]], dict[str, Any], float | None]:
        base = state["_pending_base"]
        matching = [
            (base + k, e) for k, e in enumerate(state["_pending"]) if meta.matches(e)
        ]
        inject = _build_inject(
            meta,
            state,
            reds,
            config,
            svcs_by_type,
            svcs_by_name,
            model_provider=model_provider,
        )
        # Read once per node call, not per event: the retry loop only needs
        # it on the failure path, and the no-deadline case stays a ``None``.
        deadline = (config or {}).get("configurable", {}).get(_DEADLINE_KEY)
        return matching, inject, deadline
```

Replace `_finalize`, `_run_handler_sync` and `_run_handler_async` (lines 784-831) with:

```python
    def _finalize(update: StateDict) -> StateDict:
        new_events = update["events"]
        if len(update["causes"]) != len(new_events):
            raise RuntimeError(
                f"Handler {meta.name!r} recorded {len(update['causes'])} causes "
                f"for {len(new_events)} events. The causes channel must stay "
                f"aligned with events. This is a framework bug."
            )
        output: StateDict = dict(update)
        if reds:
            output.update(_apply_reducers(new_events, reds))
        return output

    def _run_handler_sync(state: StateDict, config: RunnableConfig) -> StateDict:
        # Precondition check — outside the raises= catch boundary so a user
        # with raises=RuntimeError can't swallow this framework diagnostic.
        _check_sync_invocation_of_async(meta)
        matching, inject, deadline = _prepare(state, config)
        new_events: list[Event] = []
        new_causes: list[CauseEntry] = []
        tokens = _bind_custom_emitters(config)
        try:
            _process_events_sync(
                meta,
                matching,
                state,
                inject,
                new_events,
                new_causes,
                lg_interrupt,
                return_contract,
                deadline,
            )
        finally:
            _reset_custom_emitters(tokens)
        return _finalize({"events": new_events, "causes": new_causes})

    async def _run_handler_async(state: StateDict, config: RunnableConfig) -> StateDict:
        matching, inject, deadline = _prepare(state, config)
        new_events: list[Event] = []
        new_causes: list[CauseEntry] = []
        tokens = _bind_custom_emitters(config)
        try:
            await _process_events_async(
                meta,
                matching,
                state,
                inject,
                new_events,
                new_causes,
                lg_interrupt,
                return_contract,
                deadline,
            )
        except asyncio.CancelledError:
            return _finalize(pad_causes({"events": [Cancelled()]}))
        finally:
            _reset_custom_emitters(tokens)
        return _finalize({"events": new_events, "causes": new_causes})
```

In `docs/concepts.md`, replace the first sentence of the `### Causes` section, "A `Cause(source, via)` names the event that a handler received and the handler's graph node name.", with:

```markdown
A graph run records the cause of each event that a handler returns. A `Cause(source, via)`
names the event that the handler received and the handler's graph node name.
```

In `CHANGELOG.md`, replace the empty `## [Unreleased]` section (line 8 and the blank line after it) with:

```markdown
## [Unreleased]

### Added

- **A graph run records the cause of each event.** When a handler returns an event, the
  framework records the event that the handler received and the handler's graph node name.
  `EventLog.cause(event)` returns them as a `Cause(source, via)`, or `None` for a seed.
  `EventLog.effects(event)` lists the events that an event caused, in log order.
  `EventLog.flow(event)` gives the cause chain from the root seed. `EventLog.causes` gives
  one cause per event, or `None` when the log records no causes. `EventLog(events,
  causes=...)` rebuilds a log that a client saved in its own format. Each `source` must be
  the same object as an earlier event.

  `via` equals `Edge.via` unless the handler has a stable identity: an inline command
  handler, or an `@on(node_name=...)` pin. The graph state gets a `causes` channel next to
  `events`. Event classes, constructors and equality do not change. The names `causes` and
  `_pending_base` are now reserved state fields: a reducer with one of these names raises
  `ValueError` at graph build.

```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_event_causes.py -q -p no:cacheprovider`
Expected: PASS. Then run the full suite `uv run pytest tests/ -q -p no:cacheprovider`, `uv run ruff check src/ tests/`, `uv run ruff format src/ tests/`, `uv run mypy src/`.

- [ ] **Step 5: Commit**

```bash
git add src/langgraph_events/_internal.py tests/test_event_causes.py CHANGELOG.md docs/concepts.md
git commit -m "feat: record the trigger and the node name of each emission

The seed node and the router write _pending_base, the log index of the
first pending event. A handler records one (trigger index, node name)
entry per event after each call. The Cancelled path pads a None cause
through pad_causes.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 5: Alignment of the router writers, and pins for the handler cases

**Files:**
- Modify: `src/langgraph_events/_internal.py:189-223` (the `MaxRoundsExceeded` and `RunPaused` returns of `router`)
- Test: `tests/test_event_causes.py`

**Interfaces:**
- Consumes: `pad_causes`, `_record_causes` (Task 4). Test helpers `step` (Task 3).
- Produces: the router writes `[None]` to `causes` through `pad_causes`, and sets `_pending_base` to the index of the event it appends. Test classes `Noted`, `Tick`, `Tock` at module level in `tests/test_event_causes.py`.

- [ ] **Step 1: Write the failing test**

In `tests/test_event_causes.py`, replace the import block with:

```python
import asyncio
import time
from typing import Any

import pytest
from conftest import Ended, Order, Processed, Started
from langgraph.checkpoint.memory import MemorySaver

from langgraph_events import (
    Cancelled,
    Cause,
    EventGraph,
    EventLog,
    HandlerRaised,
    HandlerRetried,
    IntegrationEvent,
    Invariant,
    InvariantViolated,
    MaxRoundsExceeded,
    Reducer,
    RetryPolicy,
    RunPaused,
    Scatter,
    on,
)
from langgraph_events.serde import NamespaceAwareSerde
```

Insert these module-level definitions directly after the `finish` handler:

```python
class Batch(IntegrationEvent):
    n: int = 0


class Item(IntegrationEvent):
    n: int = 0


class Tally(IntegrationEvent):
    n: int = 0


class Echo(IntegrationEvent):
    n: int = 0


class Tick(IntegrationEvent):
    n: int = 0


class Tock(IntegrationEvent):
    n: int = 0


class Noted(IntegrationEvent):
    pass


class TockCap(Invariant):
    pass


class Closed(Invariant):
    pass


class FlakyError(Exception):
    pass


@on(Batch)
def split(event: Batch) -> Scatter[Item]:
    return Scatter([Item(n=1), Item(n=2)])


@on(Batch)
def count(event: Batch) -> Tally:
    return Tally(n=event.n)


@on(Batch)
def ignore(event: Batch) -> None:
    return None


@on(Tally)
def echo(event: Tally) -> Echo:
    return Echo(n=event.n)


@on(Tick)
def tock(event: Tick) -> Tock:
    return Tock(n=event.n)


@on(Tick, invariants={TockCap: lambda log: log.count(Tock) < 3})
def double(event: Tick) -> Scatter[Tock]:
    return Scatter([Tock(n=event.n), Tock(n=event.n + 10)])


@on(Tick, invariants={Closed: lambda log: False})
def blocked(event: Tick) -> Tock:
    return Tock(n=event.n)


@on(
    Tick,
    raises=FlakyError,
    retry=RetryPolicy(max_attempts=2, base_delay=0.0, jitter=False),
)
def flaky(event: Tick) -> Tock:
    raise FlakyError("down")


@on(HandlerRaised, exception=FlakyError)
def swallow(event: HandlerRaised) -> None:
    return None


@on(Tick)
def again(event: Tick) -> Tick:
    return Tick(n=event.n + 1)


@on(RunPaused)
def note_pause(event: RunPaused) -> Noted:
    return Noted()
```

Inside `describe_invoke`, add these `when_` blocks after `when_a_handler_pins_its_node_name`. The first five are pins: they pin cases that the Task 4 handler loop already covers.

```python
    def when_parallel_handlers_react_to_one_event():
        def it_records_each_emission_against_its_own_handler():
            log = EventGraph([split, count, ignore, echo]).invoke(Batch(n=3))
            batch = log.first(Batch)

            assert [log.cause(item) for item in log.filter(Item)] == [
                Cause(source=batch, via="split"),
                Cause(source=batch, via="split"),
            ]
            assert log.cause(log.first(Tally)) == Cause(source=batch, via="count")
            assert log.cause(log.first(Echo)) == Cause(
                source=log.first(Tally), via="echo"
            )

    def when_one_node_handles_several_triggers():
        def it_records_each_output_against_its_own_trigger():
            log = EventGraph([tock]).invoke([Tick(n=1), Noted(), Tick(n=2)])

            sources = [log.cause(t).source for t in log.filter(Tock)]

            assert sources == [Tick(n=1), Tick(n=2)]

    def when_an_invariant_rolls_back_an_emission():
        def it_records_the_violation_against_its_trigger():
            log = EventGraph([double]).invoke([Tick(n=1), Tick(n=2)])

            assert [log.cause(t).source for t in log.filter(Tock)] == [
                Tick(n=1),
                Tick(n=1),
            ]
            assert log.cause(log.first(InvariantViolated)) == Cause(
                source=Tick(n=2), via="double"
            )

    def when_an_invariant_blocks_the_handler():
        def it_records_the_violation_against_its_trigger():
            log = EventGraph([blocked]).invoke(Tick(n=1))

            assert log.cause(log.first(InvariantViolated)) == Cause(
                source=Tick(n=1), via="blocked"
            )

    def when_a_handler_raises_after_a_retry():
        def it_records_the_retry_and_the_raise_against_the_trigger():
            log = EventGraph([flaky, swallow]).invoke(Tick(n=1))
            expected = Cause(source=log.first(Tick), via="flaky")

            assert log.cause(log.first(HandlerRetried)) == expected
            assert log.cause(log.first(HandlerRaised)) == expected

    def when_max_rounds_is_exceeded():
        def it_records_no_cause_for_the_halt():
            log = EventGraph([again], max_rounds=2).invoke(Tick(n=0))
            ticks = log.filter(Tick)

            assert log.cause(ticks[0]) is None
            for earlier, later in zip(ticks, ticks[1:], strict=False):
                assert log.cause(later) == Cause(source=earlier, via="again")
            assert log.cause(log.latest(MaxRoundsExceeded)) is None

    def when_the_deadline_passes_after_a_round():
        def it_records_no_cause_for_the_pause():
            deadline = 0.0

            @on(Started)
            def slow(event: Started) -> Processed:
                time.sleep(max(0.0, deadline - time.monotonic()) + 0.01)
                return Processed(data=event.data)

            graph = EventGraph([slow, note_pause])
            assert graph.compiled is not None
            deadline = time.monotonic() + 0.2

            log = graph.invoke(Started(data="x"), deadline=deadline)
            paused = log.first(RunPaused)

            assert log.cause(log.first(Processed)) == Cause(
                source=log.first(Started), via="slow"
            )
            assert log.cause(paused) is None
            assert log.cause(log.first(Noted)) == Cause(
                source=paused, via="note_pause"
            )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_event_causes.py -q -p no:cacheprovider`
Expected:
- The five pins PASS: `when_parallel_handlers_react_to_one_event`, `when_one_node_handles_several_triggers`, `when_an_invariant_rolls_back_an_emission`, `when_an_invariant_blocks_the_handler` and `when_a_handler_raises_after_a_retry`. Reason: the Task 4 handler loop records a cause after each handler call.
- `when_max_rounds_is_exceeded` FAILS with "assert None == Cause(source=Tick(n=0), via='again')". `when_the_deadline_passes_after_a_round` FAILS with an `AssertionError` on the `Processed` cause. Reason: the router writes no cause, so each older cause reads one position late.

- [ ] **Step 3: Write minimal implementation**

In `src/langgraph_events/_internal.py`, inside `router`, replace the `MaxRoundsExceeded` return (lines 191-196) with:

```python
            return pad_causes(
                {
                    "_cursor": len(state["events"]),
                    "_pending": [halted],
                    "_pending_base": len(state["events"]),
                    "_round": current_round,
                    "events": [halted],
                }
            )
```

Replace the `RunPaused` return (lines 213-223) with:

```python
            return pad_causes(
                {
                    # Advance cursor PAST the paused event so a fresh /run on
                    # the same thread excludes it from new_events. Distinct
                    # from MaxRoundsExceeded above which keeps cursor AT the
                    # halted (terminal across runs).
                    "_cursor": len(state["events"]) + 1,
                    "_pending": [paused],
                    "_pending_base": len(state["events"]),
                    "_round": current_round,
                    "events": [paused],
                    "_run_paused_emitted": True,
                }
            )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_event_causes.py -q -p no:cacheprovider`
Expected: PASS. Then run the full suite `uv run pytest tests/ -q -p no:cacheprovider`, `uv run ruff check src/ tests/`, `uv run ruff format src/ tests/`, `uv run mypy src/`.

- [ ] **Step 5: Commit**

```bash
git add src/langgraph_events/_internal.py tests/test_event_causes.py
git commit -m "feat: keep causes aligned when the router appends an event

The router pads a None cause for MaxRoundsExceeded and RunPaused, and
points _pending_base at the event it appends. A handler on RunPaused
then records the pause as its source. Pins cover parallel handlers,
Scatter, several triggers, invariants and retries.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 6: Interrupt, resume, `get_state`, `abandon` and `pre_seed`

**Files:**
- Modify: `src/langgraph_events/_causes.py` (the `CauseEntry` docstring and `_absolute`)
- Modify: `src/langgraph_events/_internal.py` (`_record_causes`, a new `_closes_interrupt_block`)
- Modify: `src/langgraph_events/_graph.py:47-58`, `:1607-1627`, `:1728-1737`, `:2658-2659`
- Modify: `CHANGELOG.md`, `docs/api.md:69`, `docs/concepts.md` (the `### Causes` section)
- Test: `tests/test_event_causes.py`

**Interfaces:**
- Consumes: `resolve`, `_absolute` (Task 3), `pad_causes`, `_record_causes` (Task 4), `EventLog._from_state` (Task 3). Test helpers `step`, `_checkpointed` (Task 3), `Noted` (Task 5).
- Produces:
  - A stored source `< 0` counts back from its own event. `_absolute` returns the absolute index.
  - `_closes_interrupt_block(new_events: list[Event], j: int) -> bool` in `_internal.py`.
  - `EventGraph.get_state(config).events` has causes.
  - Test helpers in `tests/test_event_causes.py`: `Ask(Interrupted)`, `Approved(IntegrationEvent)`, handlers `ask` (`Processed -> Ask`) and `acknowledge` (`Resumed -> Ended`), `_paused(thread_id: str) -> tuple[EventGraph, dict[str, Any], MemorySaver]`.

- [ ] **Step 1: Write the failing test**

In `tests/test_event_causes.py`, in the `from langgraph_events import (...)` block, add `Abandoned,` before `Cancelled,`, add `Interrupted,` after `IntegrationEvent,`, and add `Resumed,` after `Reducer,`.

Insert these module-level definitions directly after the `note_pause` handler:

```python
class Ask(Interrupted):
    pass


class Approved(IntegrationEvent):
    pass


@on(Processed)
def ask(event: Processed) -> Ask:
    return Ask()


@on(Resumed)
def acknowledge(event: Resumed) -> Ended:
    return Ended(result="resumed")


def _paused(thread_id: str) -> tuple[EventGraph, dict[str, Any], MemorySaver]:
    """A checkpointed thread paused on ``Ask``: Started -> Processed -> Ask."""
    graph, config, saver = _checkpointed([step, ask, acknowledge], thread_id)
    graph.invoke(Started(data="x"), config=config)
    return graph, config, saver
```

Append to the end of `tests/test_event_causes.py`:

```python
def describe_resume():
    def when_a_human_answers_an_interrupt():
        def it_points_the_answer_and_the_resume_at_the_interrupt():
            graph, config, _saver = _paused("resume")

            log = graph.resume(Approved(), config=config)
            asked = log.first(Ask)

            assert log.cause(asked) == Cause(source=log.first(Processed), via="ask")
            assert log.cause(log.first(Approved)) == Cause(source=asked, via="ask")
            assert log.cause(log.first(Approved)).source is asked
            assert log.cause(log.first(Resumed)) == Cause(source=asked, via="ask")
            assert log.cause(log.first(Ended)) == Cause(
                source=log.first(Resumed), via="acknowledge"
            )


def describe_get_state():
    def when_the_thread_reloads_from_the_checkpoint():
        def it_answers_causes_by_identity_in_the_reloaded_log():
            graph, config, _saver = _paused("reload")
            graph.resume(Approved(), config=config)

            log = graph.get_state(config).events

            assert log.causes is not None
            assert log.cause(log.first(Approved)).source is log.first(Ask)
            assert log.cause(log.first(Processed)).source is log.first(Started)


def describe_abandon():
    def it_records_no_cause_for_the_abandoned_marker():
        graph, config, _saver = _paused("abandon")

        graph.abandon(config)
        log = graph.get_state(config).events

        assert log.cause(log.first(Processed)) == Cause(
            source=log.first(Started), via="step"
        )
        assert log.cause(log.latest(Abandoned)) is None


def describe_pre_seed():
    def when_events_are_written_to_a_paused_thread():
        def it_keeps_the_older_causes_aligned():
            graph, config, _saver = _paused("pre-seed")

            graph.pre_seed(config, {"events": [Noted()]})
            log = graph.resume(Approved(), config=config)

            assert log.cause(log.first(Noted)) is None
            assert log.cause(log.first(Processed)) == Cause(
                source=log.first(Started), via="step"
            )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_event_causes.py -q -p no:cacheprovider`
Expected: FAIL. `describe_resume` fails with "assert Cause(source=Processed(data='x'), via='ask') == Cause(source=Ask(), via='ask')". `describe_get_state` fails with "assert None is not None", and `describe_abandon` fails with "ValueError: this log records no causes": `get_state` builds a plain `EventLog`. `describe_pre_seed` fails with "assert None == Cause(source=Started(data='x'), via='step')".

- [ ] **Step 3: Write minimal implementation**

In `src/langgraph_events/_causes.py`, replace the `CauseEntry` docstring with:

```python
"""One entry of the ``causes`` channel: ``(source, via)``, or ``None``.

A source ``>= 0`` is a log index. A source ``< 0`` counts back from its own
event: ``-1`` is the event just before it. An ``Interrupted`` block uses it,
because its handler cannot know the absolute index of ``Interrupted``.
"""
```

Replace `_absolute` with:

```python
def _absolute(position: int, entry: Sequence[Any]) -> tuple[int, str]:
    source, via = entry
    absolute = position + source if source < 0 else source
    if not 0 <= absolute < position:
        raise RuntimeError(
            f"the stored cause of event #{position} points to #{absolute}, which "
            f"is not an earlier event. A writer to events left causes out of "
            f"step. This is a framework bug."
        )
    return (absolute, via)
```

In `src/langgraph_events/_internal.py`, replace `_record_causes` with these two functions:

```python
def _record_causes(
    new_causes: list[CauseEntry],
    new_events: list[Event],
    trigger: int,
    via: str,
) -> None:
    """Record a cause for each event appended since the last call.

    Called after each handler call. An invariant rollback replaces the
    events of its call before this runs, so it leaves no cause behind. An
    ``Interrupted`` appends ``[Interrupted, value, Resumed]`` as one block.
    The handler cannot know the absolute index of ``Interrupted``, because
    parallel tasks decide the final order. The value and the ``Resumed``
    therefore point back at it by a relative source: ``-1`` and ``-2``.
    """
    for j in range(len(new_causes), len(new_events)):
        if _closes_interrupt_block(new_events, j):
            new_causes[j - 1] = (-1, via)
            new_causes.append((-2, via))
        else:
            new_causes.append((trigger, via))


def _closes_interrupt_block(new_events: list[Event], j: int) -> bool:
    """Whether ``new_events[j]`` is the ``Resumed`` that ends an interrupt block."""
    event = new_events[j]
    return (
        isinstance(event, Resumed)
        and j >= 2
        and new_events[j - 2] is event.interrupted
        and new_events[j - 1] is event.value
    )
```

In `src/langgraph_events/_graph.py`, in the `from langgraph_events._internal import (...)` block (lines 47-58), add `pad_causes,` after `make_seed_node,`.

Replace `pre_seed` and `apre_seed` (lines 1607-1627) with:

```python
    def pre_seed(self, config: RunnableConfig, values: dict[str, Any]) -> None:
        """Inject external state into reducer channels before the first run.

        Use this to hydrate reducers from an external source (e.g. a database
        migration or test fixture) when modelling the data as seed events isn't
        practical.  Call it once before ``invoke``/``ainvoke``::

            graph.pre_seed(config, {"my_reducer": existing_value})
            graph.invoke(StartEvent(), config=config)

        Each event in ``values["events"]`` gets a ``None`` cause.

        Requires a checkpointer.
        """
        self._require_checkpointer("pre_seed")
        compiled = self._compile()
        compiled.update_state(config, pad_causes(values), as_node="__seed__")

    async def apre_seed(self, config: RunnableConfig, values: dict[str, Any]) -> None:
        """Async version of :meth:`pre_seed`."""
        self._require_checkpointer("apre_seed")
        compiled = self._compile()
        await compiled.aupdate_state(config, pad_causes(values), as_node="__seed__")
```

In `_settle_supersteps`, replace the middle superstep (lines 1728-1737):

```python
            [
                StateUpdate(
                    {
                        "events": appended,
                        "_cursor": len(events) + len(appended),
                        "_pending": [],
                    },
                    "__seed__",
                )
            ],
```

with:

```python
            [
                StateUpdate(
                    pad_causes(
                        {
                            "events": appended,
                            "_cursor": len(events) + len(appended),
                            "_pending": [],
                        }
                    ),
                    "__seed__",
                )
            ],
```

In `_graph_state`, replace lines 2658-2659:

```python
        all_events = snapshot.values.get("events", [])
        log = EventLog(all_events)
```

with:

```python
        all_events = snapshot.values.get("events", [])
        log = EventLog._from_state(all_events, snapshot.values.get("causes"))
```

In `docs/api.md`, replace line 69 (the `GraphState` row) with:

```markdown
| `GraphState` | NamedTuple | `(events, is_interrupted, interrupted)`. `events` carries the recorded causes of the thread |
```

In `docs/concepts.md`, append this paragraph to the end of the `### Causes` section:

```markdown
The value that answers an `Interrupted`, and the `Resumed` that the framework creates, have
that `Interrupted` as their source. `abandon()` and `pre_seed()` record no cause for the events
that they write.
```

In `CHANGELOG.md`, append this bullet to the end of the `### Added` list under `## [Unreleased]`:

```markdown
- **A resumed interrupt records its causes, and `get_state()` returns them.** The value that
  answers an `Interrupted`, and the `Resumed` that the framework creates, have that
  `Interrupted` as their source. `abandon()` and `pre_seed()` record no cause for the events
  that they write. `GraphState.events` carries the causes of the checkpointed thread.
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_event_causes.py -q -p no:cacheprovider`
Expected: PASS. Then run the full suite `uv run pytest tests/ -q -p no:cacheprovider`, `uv run ruff check src/ tests/`, `uv run ruff format src/ tests/`, `uv run mypy src/`.

- [ ] **Step 5: Commit**

```bash
git add src/langgraph_events/_causes.py src/langgraph_events/_internal.py src/langgraph_events/_graph.py tests/test_event_causes.py CHANGELOG.md docs/api.md docs/concepts.md
git commit -m "feat: record the causes of a resumed interrupt

The resume value and Resumed point at their Interrupted by a relative
source, because the handler cannot know its absolute index. resolve()
makes it absolute. get_state() returns the causes. abandon() and
pre_seed() pad a None cause for each event through pad_causes.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 7: Reflection shows the recorded cause

**Files:**
- Modify: `src/langgraph_events/_event_log.py` (a new `EventLog._cause_is_known`)
- Modify: `src/langgraph_events/_reflection/_text.py:152-169` and a new function `recorded_cause`
- Modify: `src/langgraph_events/_reflection/_evidence.py:14`, `:96-108`
- Modify: `src/langgraph_events/_reflection/_tool.py:23`, `:32-34`, `:126-142`, `:244-256`
- Modify: `tests/conftest.py`
- Modify: `docs/reflection.md`, `CHANGELOG.md`
- Test: `tests/test_reflection_queries.py`, `tests/test_reflection_evidence.py`, `tests/test_reflection_tool.py`

**Interfaces:**
- Consumes: `EventLog.cause`, `EventLog.causes`, `EventLog.events` (public, Tasks 1-2). `get_state` with causes (Task 6). `Reflection._resolve_index` and `Reflection.log` (existing).
- Produces:
  - `EventLog._cause_is_known(self, event: Event) -> bool`. Reflection's only private dependency on `EventLog`.
  - `recorded_cause(index: int, log: EventLog) -> str | None` in `_reflection/_text.py`.
  - `_cause_answer(reflection: Reflection, index: int) -> str` in `_reflection/_tool.py`.
  - The `query_log` op `cause`.
  - `strip_channels(saver: MemorySaver, config: dict[str, Any], *channels: str) -> None` in `tests/conftest.py`.

- [ ] **Step 1: Write the failing test**

In `tests/conftest.py`, replace the imports at the top of the file with:

```python
"""Shared fixtures and event classes for the test suite."""

import sys
from typing import Any

import pytest
from langgraph.checkpoint.memory import MemorySaver

from langgraph_events import (
    Command,
    DomainEvent,
    Event,
    EventGraph,
    IntegrationEvent,
    Namespace,
    ScalarReducer,
    on,
)
```

Append to the end of `tests/conftest.py`:

```python
def strip_channels(saver: MemorySaver, config: dict[str, Any], *channels: str) -> None:
    """Rewrite the latest checkpoint of the thread without *channels*.

    Simulates a checkpoint that a release saved before those channels
    existed. The checkpoint id stays the same, so a pending interrupt write
    still belongs to it.
    """
    tup = saver.get_tuple(config)
    assert tup is not None
    checkpoint = dict(tup.checkpoint)
    checkpoint["channel_values"] = {
        name: value
        for name, value in tup.checkpoint["channel_values"].items()
        if name not in channels
    }
    checkpoint["channel_versions"] = {
        name: version
        for name, version in tup.checkpoint["channel_versions"].items()
        if name not in channels
    }
    base = tup.parent_config or {
        "configurable": {
            "thread_id": config["configurable"]["thread_id"],
            "checkpoint_ns": "",
        }
    }
    saver.put(base, checkpoint, tup.metadata, {})
```

In `tests/test_reflection_queries.py`, replace `from conftest import Order, Started` with:

```python
from conftest import Order, Started, strip_channels
from langgraph.checkpoint.memory import MemorySaver
```

and add this import after the `from langgraph_events import (...)` block:

```python
from langgraph_events.serde import NamespaceAwareSerde
```

Inside `describe_event` in `tests/test_reflection_queries.py`, add these `when_` blocks after the last existing `when_` block:

```python
    def when_the_event_has_a_recorded_cause():
        def it_shows_the_source_index_and_the_handler():
            graph = EventGraph([Order.Place])
            reflection = graph.reflect(graph.invoke(Order.Place(customer_id="c1")))

            assert "cause: #0 via Order.Place" in reflection.event(1)

    def when_the_event_is_a_seed():
        def it_shows_no_cause_line():
            graph = EventGraph([Order.Place])
            reflection = graph.reflect(graph.invoke(Order.Place(customer_id="c1")))

            assert "cause:" not in reflection.event(0)

    def when_the_cause_was_not_recorded():
        def it_shows_an_unknown_cause():
            saver = MemorySaver(serde=NamespaceAwareSerde())
            graph = EventGraph([Order.Place], checkpointer=saver)
            config = {"configurable": {"thread_id": "legacy"}}
            graph.invoke(Order.Place(customer_id="c1"), config=config)
            strip_channels(saver, config, "causes", "_pending_base")

            reflection = graph.reflect(graph.get_state(config).events)

            assert "cause: unknown" in reflection.event(1)

    def when_the_source_is_outside_a_derived_log():
        def it_names_the_source_type():
            graph = EventGraph([Order.Place])
            log = graph.invoke(Order.Place(customer_id="c1"))

            reflection = graph.reflect(log.select(Order.Place.Placed))

            assert "cause: Place (outside this log) via Order.Place" in (
                reflection.event(0)
            )
```

Inside `describe_evidence` in `tests/test_reflection_evidence.py`, add these `when_` blocks after the last existing `when_` block:

```python
    def when_the_log_records_the_cause():
        def it_lists_the_recorded_cause_first():
            graph = EventGraph([Fulfillment.Ship, notify_customer])
            reflection = _reflect(graph, Fulfillment.Ship(order_id="o1"))

            lines = reflection.evidence(2).splitlines()

            assert lines[1] == "recorded cause: #1 via notify_customer"

        def it_lists_a_seed_as_a_seed():
            graph = EventGraph([Fulfillment.Ship, notify_customer])
            reflection = _reflect(graph, Fulfillment.Ship(order_id="o1"))

            assert reflection.evidence(0).splitlines()[1] == "recorded cause: seed"

    def when_the_log_records_no_causes():
        def it_lists_no_recorded_cause():
            graph = EventGraph([Fulfillment.Ship, notify_customer])
            reflection = graph.reflect(EventLog([Started(data="x")]))

            assert "recorded cause" not in reflection.evidence(0)
```

Inside `describe_tool` in `tests/test_reflection_tool.py`, add this `when_` block after `when_running_reflection_ops`:

```python
    def when_asking_for_a_cause():
        def it_answers_the_source_index_and_the_handler():
            tool, _ = _tool_and_reflection()

            assert tool.run(op="cause", index=1) == "#0 via Order.Place"

        def it_answers_seed_for_a_seed():
            tool, _ = _tool_and_reflection()

            assert tool.run(op="cause", index=0) == "seed"

        def it_says_so_for_a_log_that_records_no_causes():
            graph = EventGraph([Order.Place])
            log = EventLog([Order.Place(customer_id="c1")])

            tool = graph.reflect(log).tool()

            assert tool.run(op="cause", index=0) == "this log records no causes"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_reflection_queries.py tests/test_reflection_evidence.py tests/test_reflection_tool.py -q -p no:cacheprovider`
Expected:
- Two pins PASS: `it_shows_no_cause_line` and `it_lists_no_recorded_cause`. Reason: they assert an absence that holds before the change.
- The other `describe_event` tests FAIL with "AssertionError: assert 'cause: #0 via Order.Place' in ...". The other evidence tests FAIL with an `AssertionError` on `lines[1]`. The tool tests FAIL with "assert \"error: unknown op 'cause'...\" == '#0 via Order.Place'".

- [ ] **Step 3: Write minimal implementation**

In `src/langgraph_events/_event_log.py`, add this method to `EventLog` directly after `_require_table`:

```python
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
```

In `src/langgraph_events/_reflection/_text.py`, add this function directly above `render_event_detail`:

```python
def recorded_cause(index: int, log: EventLog) -> str | None:
    """The recorded cause of ``log[index]``: ``#N via <handler>``, ``seed``
    or ``unknown``. ``None`` when the log records no causes.

    *index* must be canonical. ``#N`` is a position in *log*. A source
    outside *log*, in a derived log, shows its type name instead.
    """
    if log.causes is None:
        return None
    event = log[index]
    if not log._cause_is_known(event):
        return "unknown"
    cause = log.cause(event)
    if cause is None:
        return "seed"
    position = next(
        (j for j, candidate in enumerate(log.events) if candidate is cause.source),
        None,
    )
    if position is None:
        return f"{type(cause.source).__name__} (outside this log) via {cause.via}"
    return f"#{position} via {cause.via}"
```

Replace `render_event_detail` with:

```python
def render_event_detail(index: int, log: EventLog) -> str:
    """The get op: one event, every field on its own line, plus taxonomy facts.

    *index* must be canonical (0-based, in range) — callers go through
    ``Reflection._resolve_index``.
    """
    event = log[index]
    lines = [f"#{index} {type(event).__name__}"]
    for f in dataclasses.fields(event):  # type: ignore[arg-type]
        lines.append(f"  {f.name}: {safe_repr(getattr(event, f.name))}")
    lines.append(f"  kind: {kind_of(event)}")
    namespace = getattr(type(event), "__namespace__", None)
    if namespace is not None:
        lines.append(f"  namespace: {namespace}")
    command = getattr(type(event), "__command__", None)
    if command is not None:
        lines.append(f"  command: {command.__name__}")
    cause = recorded_cause(index, log)
    if cause is not None and cause != "seed":
        lines.append(f"  cause: {cause}")
    return "\n".join(lines)
```

In `src/langgraph_events/_reflection/_evidence.py`, replace line 14:

```python
from langgraph_events._reflection._text import event_line
```

with:

```python
from langgraph_events._reflection._text import event_line, recorded_cause
```

In `render_evidence`, replace these two lines:

```python
    event = log[index]
    lines = [f"evidence for {event_line(index, event)}"]
```

with:

```python
    event = log[index]
    lines = [f"evidence for {event_line(index, event)}"]
    recorded = recorded_cause(index, log)
    if recorded is not None:
        lines.append(f"recorded cause: {recorded}")
```

In `src/langgraph_events/_reflection/_tool.py`:

Replace line 23 with:

```python
from langgraph_events._reflection._text import event_line, recorded_cause, safe_repr
```

Replace line 33 with:

```python
_INDEX_OPS = ("get", "evidence", "cause")
```

In `_DESCRIPTION_HEADER`, replace the two lines:

```text
  evidence(index) — all facts on how that event came to be: explicit links,
    owning command, static-edge candidates, forward face
```

with:

```text
  evidence(index) — all facts on how that event came to be: recorded cause,
    explicit links, owning command, static-edge candidates, forward face
  cause(index) — the recorded cause: #<index> via <handler>, seed, or unknown
```

Add this function directly above `build_tool`:

```python
def _cause_answer(reflection: Reflection, index: int) -> str:
    """The cause op: an index-preserving mirror of ``EventLog.cause``."""
    answer = recorded_cause(reflection._resolve_index(index), reflection.log)
    return "this log records no causes" if answer is None else answer
```

In `run`, replace the body of the `try:` block in the `_INDEX_OPS` branch:

```python
                if op == "get":
                    return reflection.event(index)
                return reflection.evidence(index)
```

with:

```python
                if op == "get":
                    return reflection.event(index)
                if op == "cause":
                    return _cause_answer(reflection, index)
                return reflection.evidence(index)
```

In `docs/reflection.md`, in the ops table, replace the `get` row with:

```markdown
| `get` | `index` | full field dump of one event + kind/namespace/command + recorded cause |
```

and add this row directly after the `evidence` row:

```markdown
| `cause` | `index` | the recorded cause: `#N via <handler>`, `seed`, or `unknown` |
```

In `docs/reflection.md`, in the section `### evidence — the join that replaces "why"`, replace the numbered list (items 1 to 4) with:

```markdown
1. **Recorded cause** — the handler that produced the event, and the event it received, as
   `#N via <handler>`. The framework records it at dispatch, so it is a fact, not a candidate.
   A seed shows `seed`. An older event of a checkpoint saved before causes existed shows
   `unknown`.
2. **Explicit links** — event-valued fields resolved to log positions
   (`HandlerRaised.source_event`, `Resumed.interrupted`), by identity with a
   labeled equality fallback.
3. **Owning command** — the outcome's command class and every preceding
   instance of it.
4. **Static edge candidates** — every model edge targeting this event's
   type, with its causation kind (`intent`/`react`/`orchestrate`/`chain`),
   handler, and preceding source instances.
5. **Forward face** — edges sourced at this type, with subsequent target
   instances.
```

In `docs/reflection.md`, in `## Design notes`, replace the bullet that starts with "**Richer events, richer facts.**" with:

```markdown
- **Recorded causes.** The framework records which handler produced each
  event, and from which event. `get`, `evidence` and `cause` show it. It is
  recorded at dispatch, not inferred, so the rule "deterministic only" holds.
  The recorded handler equals `Edge.via` unless the handler has a stable
  identity: an inline command handler, or an `@on(node_name=...)` pin.
```

In `CHANGELOG.md`, append this bullet to the end of the `### Added` list under `## [Unreleased]`:

```markdown
- **`Reflection` shows the recorded cause.** `event(i)` and the `get` op show
  `cause: #N via <handler>`, or `cause: unknown`. `evidence(i)` lists the recorded cause first.
  The `query_log` tool gets a `cause` op that answers `#N via <handler>`, `seed` or `unknown`.
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_reflection_queries.py tests/test_reflection_evidence.py tests/test_reflection_tool.py tests/test_docs_code_fences.py -q -p no:cacheprovider`
Expected: PASS. Then run the full suite `uv run pytest tests/ -q -p no:cacheprovider`, `uv run ruff check src/ tests/`, `uv run ruff format src/ tests/`, `uv run mypy src/`.

- [ ] **Step 5: Commit**

```bash
git add src/langgraph_events/_event_log.py src/langgraph_events/_reflection/_text.py src/langgraph_events/_reflection/_evidence.py src/langgraph_events/_reflection/_tool.py tests/conftest.py tests/test_reflection_queries.py tests/test_reflection_evidence.py tests/test_reflection_tool.py docs/reflection.md CHANGELOG.md
git commit -m "feat: show the recorded cause in Reflection

event() and the get op show the cause as #N via handler, or unknown.
evidence() lists the recorded cause first. The query_log tool gets a
cause op. Reflection reads causes through the public EventLog API and
one private method, _cause_is_known.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 8: Checkpoints saved before causes existed

**Files:**
- Modify: `src/langgraph_events/_internal.py` (`seed` in `make_seed_node`, a new `_seed_causes`, `_record_causes`, `_process_events_sync`, `_process_events_async`, `_prepare` in `make_handler_node`)
- Modify: `CHANGELOG.md`, `docs/concepts.md` (the `### Causes` section)
- Test: `tests/test_event_causes.py`

**Interfaces:**
- Consumes: `_record_causes`, `_closes_interrupt_block` (Task 6), `CauseEntry` (Task 3). `strip_channels` (Task 7). Test helpers `step`, `finish`, `_checkpointed` (Task 3), `_paused`, `Approved`, `acknowledge` (Task 6). `Reflection.event` with the cause line (Task 7).
- Produces:
  - `_seed_causes(all_events: list[Event], recorded: int, prev_cursor: int) -> list[CauseEntry]` in `_internal.py`.
  - `_record_causes(new_causes, new_events, trigger: int | None, via: str) -> None`.
  - `_process_events_sync` and `_process_events_async` take `matching: list[tuple[int | None, Event]]`.

- [ ] **Step 1: Write the failing test**

In `tests/test_event_causes.py`, replace `from conftest import Ended, Order, Processed, Started` with:

```python
from conftest import Ended, Order, Processed, Started, strip_channels
```

Inside `describe_invoke`, add this `when_` block after `when_the_deadline_passes_after_a_round`:

```python
    def when_the_thread_was_saved_before_causes_existed():
        def it_records_causes_for_the_new_run_only():
            graph, config, saver = _checkpointed([step, finish], "legacy")
            graph.invoke(Started(data="old"), config=config)
            strip_channels(saver, config, "causes", "_pending_base")

            log = graph.invoke(Started(data="new"), config=config)
            stored = graph.get_state(config).events

            assert log.cause(log.first(Processed)) is None
            assert log.cause(log.latest(Processed)).source is log.latest(Started)
            assert stored.causes[4] == Cause(source=Started(data="new"), via="step")
            assert "cause: unknown" in graph.reflect(log).event(1)
```

Inside `describe_resume`, add this `when_` block after `when_a_human_answers_an_interrupt`:

```python
    def when_the_thread_paused_before_causes_existed():
        def it_resumes_and_records_no_cause_for_the_answer():
            graph, config, saver = _paused("legacy-pause")
            strip_channels(saver, config, "causes", "_pending_base")

            log = graph.resume(Approved(), config=config)

            assert log.cause(log.first(Approved)) is None
            assert log.cause(log.first(Ended)) == Cause(
                source=log.first(Resumed), via="acknowledge"
            )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_event_causes.py -q -p no:cacheprovider -k "before_causes_existed"`
Expected: FAIL. `when_the_thread_was_saved_before_causes_existed` fails with "AssertionError: assert 'cause: unknown' in '#1 Processed...'": the seed pads a `None` for every older event, so Reflection reads them as seeds. `when_the_thread_paused_before_causes_existed` fails with "KeyError: '_pending_base'".

- [ ] **Step 3: Write minimal implementation**

In `src/langgraph_events/_internal.py`, add this function directly above `make_seed_node`:

```python
def _seed_causes(
    all_events: list[Event], recorded: int, prev_cursor: int
) -> list[CauseEntry]:
    """The ``None`` causes that the seed writes for the input events.

    The input writes only ``events``. On a thread that records causes, the
    input is the gap between ``events`` and ``causes``. A checkpoint saved
    before causes existed has no ``causes`` channel, so the gap covers the
    whole history. The seed then pads only the events after the cursor, and
    the older events keep an unknown cause.
    """
    total = len(all_events)
    return [None] * min(total - recorded, total - prev_cursor)
```

In `seed`, replace these three lines:

```python
        recorded = len(state.get("causes") or [])
        # The input writes only events, so the seed fills the gap for it.
        gap: list[CauseEntry] = [None] * (len(all_events) - recorded)
```

with:

```python
        recorded = len(state.get("causes") or [])
        gap = _seed_causes(all_events, recorded, prev_cursor)
```

Replace `_record_causes` with:

```python
def _record_causes(
    new_causes: list[CauseEntry],
    new_events: list[Event],
    trigger: int | None,
    via: str,
) -> None:
    """Record a cause for each event appended since the last call.

    Called after each handler call. An invariant rollback replaces the
    events of its call before this runs, so it leaves no cause behind. An
    ``Interrupted`` appends ``[Interrupted, value, Resumed]`` as one block.
    The handler cannot know the absolute index of ``Interrupted``, because
    parallel tasks decide the final order. The value and the ``Resumed``
    therefore point back at it by a relative source: ``-1`` and ``-2``.
    *trigger* is ``None`` on a thread that paused before causes existed.
    The cause is then unknown, and ``None`` keeps the channels aligned.
    """
    for j in range(len(new_causes), len(new_events)):
        if trigger is None:
            new_causes.append(None)
        elif _closes_interrupt_block(new_events, j):
            new_causes[j - 1] = (-1, via)
            new_causes.append((-2, via))
        else:
            new_causes.append((trigger, via))
```

In both `_process_events_sync` and `_process_events_async`, replace the parameter line:

```python
    matching: list[tuple[int, Event]],
```

with:

```python
    matching: list[tuple[int | None, Event]],
```

In both docstrings, replace the line "Each *matching* entry pairs a pending event with its log index." with:

```python
    Each *matching* entry pairs a pending event with its log index, or with
    ``None`` on a thread that paused before causes existed.
```

In `make_handler_node`, replace the first lines of `_prepare`:

```python
    def _prepare(
        state: StateDict, config: RunnableConfig
    ) -> tuple[list[tuple[int, Event]], dict[str, Any], float | None]:
        base = state["_pending_base"]
        matching = [
            (base + k, e) for k, e in enumerate(state["_pending"]) if meta.matches(e)
        ]
```

with:

```python
    def _prepare(
        state: StateDict, config: RunnableConfig
    ) -> tuple[list[tuple[int | None, Event]], dict[str, Any], float | None]:
        base = state.get("_pending_base")
        matching = [
            (None if base is None else base + k, e)
            for k, e in enumerate(state["_pending"])
            if meta.matches(e)
        ]
```

In `docs/concepts.md`, append this paragraph to the end of the `### Causes` section:

```markdown
A checkpoint saved before causes existed still loads. Its older events have an unknown cause,
and `log.cause(e)` returns `None` for them. A thread that paused before the upgrade resumes.
The events that the resumed handler returns have no recorded cause.
```

In `CHANGELOG.md`, append this bullet to the end of the `### Added` list under `## [Unreleased]`:

```markdown
- **A checkpoint saved before causes existed still loads.** Its older events have an unknown
  cause, and `EventLog.cause()` returns `None` for them. `Reflection` shows `cause: unknown`.
  A thread that paused before the upgrade resumes. The events that the resumed handler
  returns have no recorded cause.
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_event_causes.py -q -p no:cacheprovider`
Expected: PASS. Then run the full suite `uv run pytest tests/ -q -p no:cacheprovider`, `uv run ruff check src/ tests/`, `uv run ruff format src/ tests/`, `uv run mypy src/`.

- [ ] **Step 5: Commit**

```bash
git add src/langgraph_events/_internal.py tests/test_event_causes.py CHANGELOG.md docs/concepts.md
git commit -m "feat: load a checkpoint saved before causes existed

The seed pads only the events after the cursor, so older events keep an
unknown cause. A handler that resumes on a thread without _pending_base
records None, so the resume completes and the channels stay aligned.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 9: `rewrite_store(drop=...)` remaps the causes

**Files:**
- Modify: `src/langgraph_events/_rewrite.py:17-28`, `:36-37`, `:334-361`
- Modify: `CHANGELOG.md`, `docs/api.md:62`
- Test: `tests/test_rewrite_store.py`

**Interfaces:**
- Consumes: `resolve`, `CauseEntry` (Tasks 3 and 6), the `_pending_base` channel (Task 4), `EventLog.cause` (Task 2), `get_state` with causes (Task 6).
- Produces:
  - `_POSITION_CHANNELS = ("_cursor", "_pending_base")` in `_rewrite.py`.
  - `_shift_positions(values: dict[str, Any], events: list[Any], drop: tuple[type[Event], ...], new_values: dict[str, Any], changed: set[str]) -> None`.
  - `_remap_causes(causes: list[Any], events: list[Any], drop: tuple[type[Event], ...]) -> list[CauseEntry]`. Writes absolute entries only.

- [ ] **Step 1: Write the failing test**

In `tests/test_rewrite_store.py`, add `Cause,` to the `from langgraph_events import (...)` list, directly before `EventGraph,`.

Inside `describe_rewrite_store`, in the existing `when_drop_names_a_live_class` block, add this test after `it_dispatches_the_next_input_on_the_thread`:

```python
        def it_remaps_each_cause_to_the_kept_events():
            saver = MemorySaver()
            graph, cfg, retiring = _settled_drop_pair(saver, "t1")

            graph.rewrite_store(drop=(retiring,))
            log = graph.get_state(cfg).events

            assert log.cause(log.latest(Ended)) == Cause(source=_Go(), via="_go_ends")
            assert log.cause(log.latest(Ended)).source is log.first(_Go)
            assert log.cause(log.first(_Go)) is None
```

Inside `describe_rewrite_store`, add this `when_` block after `when_drop_names_a_base_class`:

```python
    def when_a_dropped_event_sits_below_a_paused_handler():
        def it_keeps_the_trigger_of_the_resumed_handler():
            saver = MemorySaver()

            class _Noise(IntegrationEvent):
                pass

            class _Gate(Interrupted):
                pass

            @on(_Noise)
            def promote(event: _Noise) -> Started:
                return Started(data="promoted")

            @on(Started)
            def wait(event: Started) -> _Gate:
                return _Gate()

            cfg = _cfg("t1")
            saver.serde = NamespaceAwareSerde(events=(Started, _Noise, _Gate))
            graph = EventGraph([promote, wait], checkpointer=saver)
            graph.invoke(_Noise(), config=cfg)
            graph.rewrite_store(drop=(_Noise,))

            log = graph.resume(_Go(), config=cfg)

            assert log.cause(log.first(_Gate)) == Cause(
                source=Started(data="promoted"), via="wait"
            )
            assert log.cause(log.first(_Gate)).source is log.first(Started)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_rewrite_store.py -q -p no:cacheprovider -k "remaps_each_cause or below_a_paused_handler"`
Expected: FAIL with "RuntimeError: the causes channel holds 5 entries for 4 events": the drop removes events but not their causes. After the remap alone, the paused case fails with "RuntimeError: the stored cause of event #1 points to #1, which is not an earlier event", because `_pending_base` still points one position late.

- [ ] **Step 3: Write minimal implementation**

In `src/langgraph_events/_rewrite.py`:

Replace the import lines 17-28 with:

```python
from langgraph_events._causes import resolve
from langgraph_events._event import Event, Resumed

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Sequence

    from langgraph.checkpoint.base import Checkpoint, CheckpointTuple

    from langgraph_events._causes import CauseEntry
    from langgraph_events.serde._jsonplus import (
        NamespaceAwareSerde,
        ReadRecord,
        UnrevivedIdentity,
    )
```

Replace lines 36-37:

```python
_LOG_CHANNELS = ("events", "_pending")
"""The two channels ``drop`` filters. ``_cursor`` indexes ``events``."""
```

with:

```python
_LOG_CHANNELS = ("events", "_pending")
"""The two channels ``drop`` filters. ``_POSITION_CHANNELS`` index ``events``."""

_POSITION_CHANNELS = ("_cursor", "_pending_base")
"""The channels that hold a position in ``events``."""
```

Replace `_drop_from_log` (lines 334-361) with these three functions:

```python
def _drop_from_log(
    values: dict[str, Any], drop: tuple[type[Event], ...]
) -> tuple[dict[str, Any], set[str]]:
    """Filter ``events`` and ``_pending``. Lower each position by the number
    of dropped ``events`` entries below it, so the next run still dispatches
    exactly the entries it would have dispatched. Remap ``causes``."""
    new_values = dict(values)
    changed: set[str] = set()
    if not drop:
        return new_values, changed
    for channel in _LOG_CHANNELS:
        entries = _log_entries(values, channel)
        if entries is None:
            continue
        kept = [
            _clear_resumed(entry, drop) for entry in entries if type(entry) not in drop
        ]
        if kept == entries:
            continue
        new_values[channel] = kept
        changed.add(channel)
    events = _log_entries(values, "events")
    if events is not None and "events" in changed:
        _shift_positions(values, events, drop, new_values, changed)
    return new_values, changed


def _shift_positions(
    values: dict[str, Any],
    events: list[Any],
    drop: tuple[type[Event], ...],
    new_values: dict[str, Any],
    changed: set[str],
) -> None:
    """Lower each position channel past the dropped events, and remap ``causes``."""
    for channel in _POSITION_CHANNELS:
        position = values.get(channel)
        if not isinstance(position, int):
            continue
        below = sum(1 for entry in events[:position] if type(entry) in drop)
        if below:
            new_values[channel] = position - below
            changed.add(channel)
    causes = _log_entries(values, "causes")
    if causes is None:
        return
    remapped = _remap_causes(causes, events, drop)
    if remapped != causes:
        new_values["causes"] = remapped
        changed.add("causes")


def _remap_causes(
    causes: list[Any], events: list[Any], drop: tuple[type[Event], ...]
) -> list[CauseEntry]:
    """Filter ``causes`` at the dropped positions and remap each source.

    :func:`~langgraph_events._causes.resolve` aligns the channels and makes
    each source absolute first, so this remaps absolute indices only and
    writes absolute entries back. An event below ``known_from`` keeps no
    entry, so the channels still align from the end. A cause whose source
    was dropped becomes ``None``.
    """
    entries, known_from = resolve(events, causes)
    new_position: dict[int, int] = {}
    for old, event in enumerate(events):
        if type(event) not in drop:
            new_position[old] = len(new_position)
    remapped: list[CauseEntry] = []
    for position in range(known_from, len(events)):
        if position not in new_position:
            continue
        entry = entries[position]
        if entry is None:
            remapped.append(None)
            continue
        source, via = entry
        target = new_position.get(source)
        remapped.append(None if target is None else (target, via))
    return remapped
```

In `docs/api.md`, in the `EventGraph.rewrite_store()` row (line 62), replace the text "and drops the stored instances of each `drop=` class from `events` and `_pending`." with:

```markdown
and drops the stored instances of each `drop=` class from `events` and `_pending`. It filters `causes` at the same positions and remaps each source. A cause whose source was dropped becomes `None`.
```

In `CHANGELOG.md`, append this bullet to the end of the `### Added` list under `## [Unreleased]`:

```markdown
- **`rewrite_store(drop=...)` keeps the causes aligned.** It filters `causes` at the same
  positions as `events` and remaps each source. A cause whose source was dropped becomes
  `None`. It also lowers `_pending_base`, so a paused handler resumes with its real trigger.
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_rewrite_store.py -q -p no:cacheprovider`
Expected: PASS. Then run the full suite `uv run pytest tests/ -q -p no:cacheprovider`, `uv run ruff check src/ tests/`, `uv run ruff format src/ tests/`, `uv run mypy src/`. Then run `uv run pre-commit run --all-files` once, because this is the last task.

- [ ] **Step 5: Commit**

```bash
git add src/langgraph_events/_rewrite.py tests/test_rewrite_store.py CHANGELOG.md docs/api.md
git commit -m "feat: remap the causes when rewrite_store drops events

The drop reads causes through resolve(), filters them at the same
positions as events and remaps each absolute source. A cause whose
source was dropped becomes None. _pending_base drops by the same count
as _cursor, so a paused handler resumes with its real trigger.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

## Self-Review

Spec coverage:

| Spec section | Task |
|---|---|
| Decisions 1-4: the cause next to the log, `Cause` holds the event, index-based storage | Tasks 1, 3 |
| Public API: `Cause`, `cause`, `effects`, `flow`, `causes`, the constructor and its checks | Tasks 1, 2 |
| Public API: `via` is `node_name`, and equals `Edge.via` unless the handler has a stable identity | Task 4 (`when_an_inline_command_handler_runs`, `when_a_handler_pins_its_node_name`), docs in Tasks 2, 4, 7 |
| Reflection: `get`, `evidence`, the `cause` op | Task 7 |
| Storage: the channel, plain tuples, `_pending_base`, `_causes.py` ownership | Tasks 3, 4 |
| Writers: handler | Task 4 |
| Writers: seed | Tasks 3, 4, 8 |
| Writers: router, `Cancelled`, abandon, `pre_seed` through `pad_causes` | Tasks 4, 5, 6 |
| Writers: `rewrite_store(drop=...)` | Task 9 |
| Fail fast on drift | Task 3 (`resolve` tests), Task 4 (`_finalize` guard) |
| Interrupt and resume, relative sources | Task 6 |
| Checkpoints saved before this feature | Task 3 (align from the end), Tasks 7 and 8 (`unknown`, the seed rule, the paused thread) |
| Clients that persist their own log | Task 1 (constructor, `causes` round trip, derived-log error) |
| TDD order 1-7 | Tasks 1, 2, 3-4, 5, 6 and 8, 9, 7 |

Checks:

- No step holds a placeholder. Every code step shows complete code or an exact replacement.
- Names stay the same across tasks: `CauseEntry`, `resolve`, `_absolute`, `pad_causes`, `_record_causes`, `_closes_interrupt_block`, `_seed_causes`, `_CauseTable.locate`, `_CauseTable.known_from`, `_cause_at`, `EventLog._from_state`, `EventLog._require_table`, `EventLog._cause_is_known`, `recorded_cause`, `_cause_answer`, `_shift_positions`, `_remap_causes`, `strip_channels`, `_config`, `_checkpointed`, `_paused`.
- Only `_causes.py` reads a relative source. `EventLog` and `_rewrite` receive absolute entries from `resolve`.
- Every writer to `events` outside a handler call uses `pad_causes`. The seed node keeps its own gap fill.
- Reflection uses `EventLog.causes`, `EventLog.cause`, `EventLog.events` and `EventLog._cause_is_known`. It uses no other private member of `EventLog` or `_causes`.
- No new test reads a checkpoint's `channel_values`. `strip_channels` changes a checkpoint to simulate an older release. The assertions go through `get_state`, `invoke`, `resume`, `graph.compiled` and `Reflection`.
- Each Review Focus item has a pinned test in the task that owns its code.
- Known gap: the `_finalize` length guard has no test, because no public input reaches it.
