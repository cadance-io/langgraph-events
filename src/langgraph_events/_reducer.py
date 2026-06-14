"""Reducer — generic LangGraph-native state channel for event-driven graphs."""

from __future__ import annotations

import operator
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import (
    TYPE_CHECKING,
    Annotated,
    Any,
    Generic,
    Protocol,
    TypeVar,
    cast,
    runtime_checkable,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from langchain_core.messages import BaseMessage

    from langgraph_events._event import Event, Namespace
    from langgraph_events._types import EventTypeSpec, ReducerFn


class ReducerNotSetError(ValueError):
    """Raised when a handler declares a non-``None`` reducer parameter but the
    channel value is ``None`` at injection time.

    A handler signature like ``def h(event, strategy: str)`` declares that
    ``strategy`` must be a ``str`` — ``None`` is not in the annotation, so the
    framework asserts the value before calling the handler. To opt out, widen
    the annotation to ``str | None`` / ``Optional[str]`` / ``Any``.
    """


class _SkipType:
    """Sentinel returned from ``ScalarReducer.fn`` to signal no contribution."""

    __slots__ = ()

    def __repr__(self) -> str:
        return "SKIP"


SKIP = _SkipType()
"""Return from a ``ScalarReducer.fn`` to signal no contribution.

When ``fn`` returns ``SKIP``, the reducer behaves as if no matching
event was found: the state channel keeps its current value.
Use this to distinguish "set to None" from "don't update".
"""


class _ResetType:
    """Sentinel returned from a ``FoldReducer.fold`` to clear the channel."""

    __slots__ = ()

    def __repr__(self) -> str:
        return "RESET"


RESET = _ResetType()
"""Return from a ``FoldReducer.fold`` to reset the channel.

When ``fold`` returns ``RESET``, the channel is cleared back to a fresh
``default_factory()`` value before any subsequent events fold onto it.
Use this so a reset event need not know the empty state's shape — and so
``None`` stays available as a legitimate stored state value.
"""


def _last_write_wins(existing: Any, new: Any) -> Any:
    """Binary operator that always takes the newer value."""
    return new


def _matches_namespace(event: Any, dom: type[Namespace] | None) -> bool:
    """Return True if *event* belongs to *dom* (or *dom* is None = no filter)."""
    if dom is None:
        return True
    return getattr(type(event), "__namespace__", None) == dom.__namespace_name__


class BaseReducer(ABC):
    """Abstract base for all reducer types.

    Subclasses declare ``event_type`` so the framework can filter events
    before calling ``fn``.  Use a ``@runtime_checkable Protocol`` for
    structural multi-type matching.

    When declared as a class attribute on a ``Namespace`` subclass, the
    reducer's ``name`` auto-fills from the attribute name and
    ``namespace`` auto-fills to the enclosing class — both via
    ``__set_name__``.
    """

    name: str
    event_type: EventTypeSpec
    namespace: type[Namespace] | None

    def __set_name__(self, owner: type, name: str) -> None:
        # __set_name__ runs before Namespace.__init_subclass__ stamps
        # __namespace_name__, so duck-typing via hasattr won't work here.
        # Runtime import avoids module-level circular dependency.
        from langgraph_events._event import Namespace  # noqa: PLC0415

        if isinstance(owner, type) and issubclass(owner, Namespace):
            if not self.name:
                self.name = name
            if self.namespace is None:
                self.namespace = owner

    @abstractmethod
    def state_annotation(self) -> Any:
        """LangGraph state channel type annotation."""

    @property
    @abstractmethod
    def empty(self) -> Any:
        """Default value when no contributions exist."""

    @abstractmethod
    def collect(self, events: list[Event]) -> Any:
        """Gather contributions from events."""

    @abstractmethod
    def has_contributions(self, result: Any) -> bool:
        """Whether collect() produced a meaningful update."""

    @abstractmethod
    def output_type(self) -> Any:
        """Output schema field type."""

    @abstractmethod
    def seed(self, events: list[Event]) -> Any:
        """Initialize with default + seed event contributions."""


@dataclass
class Reducer(BaseReducer):
    """Maps events to contributions for a named LangGraph state channel.

    The reducer filters events by ``event_type``, then calls ``fn`` on each
    matching event.  Its return value is merged into the state channel using
    the ``reducer`` function (defaults to ``operator.add`` for simple list
    concatenation).

    Any LangGraph-compatible reducer can be used — e.g., ``add_messages``
    from ``langchain_core.messages`` for smart message deduplication.

    Handlers receive the accumulated value by declaring a parameter whose
    name matches ``self.name``.

    Example::

        from langgraph_events import message_reducer

        messages = message_reducer([SystemMessage(content="You are helpful")])

        graph = EventGraph([call_llm], reducers=[messages])

        @on(UserMessageReceived)
        async def call_llm(event: Event, messages: list[BaseMessage]) -> LLMResponded:
            response = await llm.ainvoke(messages)
            ...
    """

    name: str = ""
    event_type: EventTypeSpec = field(kw_only=True)
    fn: Callable[[Any], list[Any]] = field(kw_only=True)
    reducer: ReducerFn = field(kw_only=True, default=operator.add)
    default: list[Any] = field(kw_only=True, default_factory=list)
    namespace: type[Namespace] | None = field(kw_only=True, default=None)

    def state_annotation(self) -> Any:
        return Annotated[list, self.reducer]

    @property
    def empty(self) -> Any:
        return list(self.default)

    def collect(self, events: list[Event]) -> Any:
        contributions: list[Any] = []
        for event in events:
            if not isinstance(event, self.event_type):
                continue
            if not _matches_namespace(event, self.namespace):
                continue
            contrib = self.fn(event)
            if contrib:
                if not isinstance(contrib, list):
                    raise TypeError(
                        f"Reducer {self.name!r} fn must return a list, "
                        f"got {type(contrib).__name__}"
                    )
                contributions.extend(contrib)
        return contributions

    def has_contributions(self, result: Any) -> bool:
        return bool(result)

    def output_type(self) -> Any:
        return list

    def seed(self, events: list[Event]) -> Any:
        values = list(self.default)
        values.extend(self.collect(events))
        return values


@dataclass
class ScalarReducer(BaseReducer):
    """Last-write-wins reducer that injects a bare value instead of a list.

    The reducer filters events by ``event_type``, then calls ``fn`` on the
    last matching event.  The return value — including ``None`` — is injected
    directly into the handler.  Return ``SKIP`` from ``fn`` to signal no
    contribution and keep the channel at its current value.

    Use a ``@runtime_checkable Protocol`` as ``event_type`` to match
    multiple event types structurally.

    A handler's parameter annotation declares whether the value may be
    ``None``. ``strategy: str`` rejects ``None`` and raises
    :class:`ReducerNotSetError` at injection if the channel is unset; widen
    to ``strategy: str | None`` (or ``Optional[str]`` / ``Any`` / ``object``,
    or omit the annotation) to allow ``None``.

    Example::

        strategy = ScalarReducer(
            name="strategy",
            event_type=StrategyChosen,
            fn=lambda e: e.strategy,
        )

        @on(TaskReceived)
        def handle(event: TaskReceived, strategy: str) -> Completed:
            # strategy is guaranteed non-None here; otherwise the framework
            # raises ReducerNotSetError before this body runs.
            ...
    """

    name: str = ""
    event_type: EventTypeSpec = field(kw_only=True)
    fn: Callable[[Any], Any] = field(kw_only=True)
    default: Any = field(kw_only=True, default=None)
    namespace: type[Namespace] | None = field(kw_only=True, default=None)

    def state_annotation(self) -> Any:
        return Annotated[Any, _last_write_wins]

    @property
    def empty(self) -> Any:
        return self.default

    def collect(self, events: list[Event]) -> Any:
        last: Any = SKIP
        for event in events:
            if not isinstance(event, self.event_type):
                continue
            if not _matches_namespace(event, self.namespace):
                continue
            last = event
        return self.fn(last) if last is not SKIP else SKIP

    def has_contributions(self, result: Any) -> bool:
        return result is not SKIP

    def output_type(self) -> Any:
        return Any

    def seed(self, events: list[Event]) -> Any:
        result = self.collect(events)
        return result if self.has_contributions(result) else self.default


S = TypeVar("S")
"""The accumulating state type a ``FoldReducer`` folds events into."""


@runtime_checkable
class Foldable(Protocol):
    """An event that knows how to fold itself into accumulating state.

    Structural, not nominal: an event satisfies ``Foldable`` simply by
    *having* a ``fold(self, state)`` method — it must **not** inherit from
    ``Foldable`` (that would clash with the event metaclass, and isn't
    needed). Used to type :class:`FoldReducer`'s default ``fold`` (which
    calls ``event.fold(state)``) without falling back to ``Any``.
    """

    def fold(self, state: Any) -> Any: ...


@dataclass(frozen=True)
class _Contributions:
    """Private wrapper for a ``FoldReducer``'s collected events.

    The fold-merge needs to tell a list of contribution events (normal
    execution / streaming) apart from a pre-folded state written directly
    through the channel via ``update_state`` / ``pre_seed``.  Wrapping the
    events in a nominal type the user can never produce makes that
    distinction unambiguous — so the fold state may be *anything*, including
    a ``list``.
    """

    events: list[Event]


@dataclass
class FoldReducer(BaseReducer, Generic[S]):
    """Folds each matching event into a single accumulating state object.

    Where :class:`Reducer` appends and :class:`ScalarReducer` takes the last
    write, ``FoldReducer`` computes the next value from the *prior* state and
    the event — a left-fold.  Use it for counters, merging dicts, re-derived
    cursors, or any channel whose update depends on what came before.

    Each event owns its transition through ``fold(self, state)`` (mirroring
    :meth:`MessageEvent.as_messages`), so callers usually supply only the
    channel name, the event type(s), and a ``default_factory``::

        counter = FoldReducer(
            name="counter",
            event_type=(Incremented, Reset),
            default_factory=lambda: {"n": 0},
        )
        # Incremented.fold(self, state) -> {"n": state["n"] + 1}
        # Reset.fold(self, state) -> RESET   # clears the channel

    A ``fold`` returning :data:`RESET` clears the channel back to
    ``default_factory()``; returning :data:`SKIP` leaves it unchanged; any
    other value — including ``None`` — becomes the new state.  Pass an
    explicit ``fold=`` to support events that don't carry a ``fold`` method.

    As with :class:`ScalarReducer`, a handler's parameter annotation declares
    whether the injected value may be ``None``; widen to ``... | None`` /
    ``Any`` to allow it, or :class:`ReducerNotSetError` is raised at injection
    when the channel is unset.
    """

    name: str = ""
    event_type: EventTypeSpec = field(kw_only=True)
    default_factory: Callable[[], S] = field(kw_only=True)
    fold: Callable[[S, Foldable], S | _ResetType | _SkipType] = field(
        kw_only=True, default=lambda state, event: event.fold(state)
    )
    namespace: type[Namespace] | None = field(kw_only=True, default=None)

    @property
    def reducer(self) -> Callable[[S, _Contributions | S], S]:
        # Exposed for the streaming shadow path, which folds per event via
        # ``getattr(r, "reducer")``.
        return self._merge

    def state_annotation(self) -> Any:
        # Stays ``Any`` on purpose: LangGraph's BinaryOperatorAggregate calls
        # ``Any()`` (which raises) so the channel starts MISSING and the first
        # write is set directly, matching ScalarReducer. A concrete ``S`` whose
        # ``S()`` succeeds (e.g. ``dict``) would break that first-write path.
        return Annotated[Any, self._merge]

    @property
    def empty(self) -> S:
        return self.default_factory()

    def collect(self, events: list[Event]) -> _Contributions:
        return _Contributions(
            [
                event
                for event in events
                if isinstance(event, self.event_type)
                and _matches_namespace(event, self.namespace)
            ]
        )

    def has_contributions(self, result: Any) -> bool:
        return bool(result.events)

    def output_type(self) -> Any:
        return Any

    def seed(self, events: list[Event]) -> S:
        return self._fold_events(self.default_factory(), self.collect(events).events)

    def _merge(self, current: S, update: _Contributions | S) -> S:
        if isinstance(update, _Contributions):
            return self._fold_events(current, update.events)
        # A pre-folded state written directly through the channel
        # (update_state / pre_seed) — replace.
        return update

    def _fold_events(self, state: S, events: list[Event]) -> S:
        for event in events:
            # ``collect``'s event_type filter guarantees every event here is
            # Foldable at runtime; the cast bridges the Event-typed internal
            # stream to ``fold``'s ``Foldable`` parameter.
            result = self.fold(state, cast("Foldable", event))
            # isinstance (not ``is``) so mypy narrows the result to ``S`` in
            # the else branch — SKIP/RESET are the sole instances of their
            # private types, so this is identity-equivalent in practice.
            if isinstance(result, _SkipType):
                continue
            state = self.default_factory() if isinstance(result, _ResetType) else result
        return state


def message_reducer(
    default: list[BaseMessage] | None = None,
    *,
    name: str = "messages",
) -> Reducer:
    """Built-in reducer for MessageEvent -> BaseMessage projection.

    Calls ``as_messages()`` on any ``MessageEvent`` and accumulates
    using ``langgraph.graph.message.add_messages`` for smart deduplication.

    Args:
        default: Optional list of initial messages (e.g. a SystemMessage).
        name: State channel name (default ``"messages"``).

    Example::

        # Using a SystemPromptSet seed event (preferred — prompt is in the event log):
        messages = message_reducer()
        graph = EventGraph([call_llm], reducers=[messages])
        log = graph.invoke([
            SystemPromptSet.from_str("You are helpful"),
            UserMessageReceived(message=HumanMessage(content="Hi")),
        ])

        # Using an explicit default list:
        messages = message_reducer([SystemMessage(content="You are helpful")])
    """
    from langgraph.graph.message import add_messages  # noqa: PLC0415

    from langgraph_events._event import MessageEvent  # noqa: PLC0415

    resolved_default = default or []

    def fn(event: MessageEvent) -> list[BaseMessage]:
        return event.as_messages()

    return Reducer(
        name=name,
        event_type=MessageEvent,
        fn=fn,
        reducer=add_messages,  # type: ignore[arg-type]
        default=resolved_default,
    )
