"""Reducer — generic LangGraph-native state channel for event-driven graphs."""

from __future__ import annotations

import inspect
import operator
import uuid
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
    from collections.abc import Callable, Collection, Mapping

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


T = TypeVar("T")


def _last_write_wins(existing: object, new: T) -> T:
    """Binary operator that always takes the newer value."""
    return new


def _matches_namespace(event: object, dom: type[Namespace] | None) -> bool:
    """Return True if *event* belongs to *dom* (or *dom* is None = no filter).

    Membership is the namespace *object*. Names are unique only within a
    graph, so matching on ``__namespace__`` would let one engine lifetime's
    reducer fold an event from another lifetime of the same module (#148).
    """
    if dom is None:
        return True
    return getattr(type(event), "__namespace_cls__", None) is dom


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
    def state_annotation(self) -> object:
        """LangGraph state channel type annotation."""

    @property
    @abstractmethod
    def empty(self) -> object:
        """Default value when no contributions exist."""

    @abstractmethod
    def collect(self, events: list[Event]) -> object:
        """Gather contributions from events."""

    @abstractmethod
    def has_contributions(self, result: Any) -> bool:  # noqa: ANN401 - a subclass narrows this to its own collect() result
        """Whether collect() produced a meaningful update."""

    @abstractmethod
    def output_type(self) -> object:
        """Output schema field type."""

    @abstractmethod
    def seed(self, events: list[Event]) -> object:
        """Initialize with default + seed event contributions."""

    @property
    def sources(self) -> tuple[BaseReducer, ...]:
        """The stored reducers that :meth:`read` reads: this reducer only."""
        return (self,)

    def read(self, state: Mapping[str, Any]) -> object:
        """The value of this channel in *state*, or :attr:`empty` if unset."""
        return state.get(self.name, self.empty)

    def _describe(self) -> str:
        return f"reducer {self.name!r}"

    def _not_set_hint(self) -> str:
        """The first fix that a ``ReducerNotSetError`` names.

        A catch-all reducer (``event_type=Event``) gives no useful hint.
        """
        from langgraph_events._event import Event  # noqa: PLC0415

        if self.event_type is Event:
            return "Set the reducer before this handler runs, or "
        type_name = getattr(self.event_type, "__name__", str(self.event_type))
        return (
            f"Ensure an event of type {type_name} has been "
            f"processed before this handler runs, or "
        )


class Readable(Protocol):
    """A value that a handler reads by parameter name.

    A stored reducer reads its own channel. A :class:`Derived`, once bound
    to its input reducers, computes its value from their channels.
    """

    @property
    def name(self) -> str: ...

    @property
    def sources(self) -> tuple[BaseReducer, ...]:
        """The stored reducers whose channels :meth:`read` reads."""
        ...

    def read(self, state: Mapping[str, Any]) -> Any:  # noqa: ANN401 - the value of a reducer
        """The value as of *state*, with an empty default for an unset channel."""
        ...

    def _describe(self) -> str: ...

    def _not_set_hint(self) -> str: ...


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

    def state_annotation(self) -> object:
        return Annotated[list, self.reducer]

    @property
    def empty(self) -> list[Any]:
        return list(self.default)

    def collect(self, events: list[Event]) -> list[Any]:
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

    def has_contributions(self, result: list[Any]) -> bool:
        return bool(result)

    def output_type(self) -> object:
        return list

    def seed(self, events: list[Event]) -> list[Any]:
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

    def state_annotation(self) -> object:
        return Annotated[Any, _last_write_wins]

    @property
    def empty(self) -> object:
        return self.default

    def collect(self, events: list[Event]) -> object:
        last: Any = SKIP
        for event in events:
            if not isinstance(event, self.event_type):
                continue
            if not _matches_namespace(event, self.namespace):
                continue
            last = event
        return self.fn(last) if last is not SKIP else SKIP

    def has_contributions(self, result: Any) -> bool:  # noqa: ANN401 - a subclass narrows this to its own collect() result
        return result is not SKIP

    def output_type(self) -> object:
        return Any

    def seed(self, events: list[Event]) -> object:
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

    def fold(self, state: Any) -> Any: ...  # noqa: ANN401 - matches the fold of any event; FoldReducer[S] holds the state type


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

    def state_annotation(self) -> object:
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

    def has_contributions(self, result: _Contributions) -> bool:
        return bool(result.events)

    def output_type(self) -> object:
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


@dataclass(frozen=True)
class Derived(Generic[T]):
    """A value that a pure function computes from other reducers. It is not stored.

    The parameter names of ``fn`` name its input reducers. Before a handler
    runs, the framework calls ``fn`` with the values of those reducers. A
    handler receives the result by parameter name, like a reducer value::

        stage = Derived(
            name="stage",
            fn=lambda placed, shipped: "shipped" if shipped else "placed",
        )

        @on(Checked)
        def check(event: Checked, stage: str) -> Reported: ...

    The value is never written to the checkpoint, so it cannot drift from
    its inputs. An input must be a stored reducer, not another ``Derived``.
    As on a :class:`ScalarReducer` parameter, a non-``None`` annotation opts
    into :class:`ReducerNotSetError`.

    When declared as a class attribute on a ``Namespace`` subclass, ``name``
    auto-fills from the attribute name, and ``namespace`` auto-fills to the
    enclosing class. The namespace only places the value in the structure
    views. An input can be any stored reducer of the graph.
    """

    name: str = ""
    fn: Callable[..., T] = field(kw_only=True)
    namespace: type[Namespace] | None = field(kw_only=True, default=None)

    def __set_name__(self, owner: type, name: str) -> None:
        from langgraph_events._event import Namespace  # noqa: PLC0415

        if isinstance(owner, type) and issubclass(owner, Namespace):
            # A frozen dataclass refuses attribute writes. The class body
            # fills in the declaration once, before any graph reads it.
            if not self.name:
                object.__setattr__(self, "name", name)
            if self.namespace is None:
                object.__setattr__(self, "namespace", owner)

    @property
    def inputs(self) -> tuple[str, ...]:
        """The names of the input reducers, in the order of ``fn``."""
        return tuple(p.name for p in self._parameters())

    def _parameters(self) -> tuple[inspect.Parameter, ...]:
        try:
            signature = inspect.signature(self.fn)
        except (TypeError, ValueError) as exc:
            raise TypeError(
                f"Derived {self.name!r} cannot read the parameters of "
                f"fn={self.fn!r}. Pass a function whose parameter names name "
                f"its input reducers."
            ) from exc
        return tuple(signature.parameters.values())

    def _bind(
        self, reducers: Mapping[str, BaseReducer], derived: Collection[str]
    ) -> _BoundDerived[T]:
        """Bind each input to the stored reducer of its name.

        Raises ``TypeError`` if an input does not name a stored reducer.
        *derived* holds the names of the other ``Derived`` values.
        """
        sources: list[BaseReducer] = []
        for param in self._parameters():
            if param.kind not in (param.POSITIONAL_OR_KEYWORD, param.KEYWORD_ONLY):
                raise TypeError(
                    f"Derived {self.name!r} has the variadic or positional-only "
                    f"parameter {param.name!r}. Each parameter must name one "
                    f"input reducer."
                )
            if param.name in derived:
                raise TypeError(
                    f"Derived {self.name!r} reads the derived {param.name!r}. "
                    f"An input must be a stored reducer. Read the inputs of "
                    f"{param.name!r} instead."
                )
            reducer = reducers.get(param.name)
            if reducer is None:
                default_hint = (
                    " A parameter with a default must still name a reducer. "
                    "Bind a constant in a closure."
                    if param.default is not param.empty
                    else ""
                )
                raise TypeError(
                    f"Derived {self.name!r} reads {param.name!r}, which is not "
                    f"a registered reducer.{default_hint} Available reducers: "
                    f"{sorted(reducers)}."
                )
            sources.append(reducer)
        return _BoundDerived(self, tuple(sources))


@dataclass(frozen=True)
class _BoundDerived(Generic[T]):
    """A :class:`Derived` of one graph, bound to its input reducers.

    One ``Derived`` can serve several graphs, so the binding is not stored
    on it.
    """

    derived: Derived[T]
    sources: tuple[BaseReducer, ...]

    @property
    def name(self) -> str:
        return self.derived.name

    def read(self, state: Mapping[str, Any]) -> T:
        """Call ``fn`` with each input from *state*, or its empty default."""
        values = {r.name: r.read(state) for r in self.sources}
        try:
            return self.derived.fn(**values)
        except Exception as exc:
            exc.add_note(f"Derived {self.name!r} raised while it computed its value.")
            raise

    def _describe(self) -> str:
        return f"Derived {self.name!r}"

    def _not_set_hint(self) -> str:
        return f"Make Derived {self.name!r} return a non-None value, or "


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
        # ``add_messages`` gives a message without an id its id in place, but
        # only when the step merges its writes. A paused step saves its write
        # before that merge, so each checkpoint read would give a new id.
        messages = event.as_messages()
        for message in messages:
            if message.id is None:
                message.id = str(uuid.uuid4())
        return messages

    return Reducer(
        name=name,
        event_type=MessageEvent,
        fn=fn,
        reducer=add_messages,  # type: ignore[arg-type]
        default=resolved_default,
    )
