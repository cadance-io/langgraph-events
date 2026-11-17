"""Reducer — generic LangGraph-native state channel for event-driven graphs."""

from __future__ import annotations

import inspect
import operator
import types
import typing
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

# Read at run time: ``message_reducer`` declares ``list[BaseMessage]``, and
# graph construction resolves the return annotation of its ``fn``.
from langchain_core.messages import BaseMessage

from langgraph_events._event_log import EventLog
from langgraph_events._hints import (
    UnsettledTypeError,
    _hint_label,
    _normal,
    _open_parameters,
    _resolve_parameter_hints,
    _same_type,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from langgraph_events._event import Event, MessageEvent, Namespace
    from langgraph_events._inputs import BoundDerived
    from langgraph_events._types import EventTypeSpec, ReducerFn


class ReducerNotSetError(ValueError):
    """Raised when a handler declares a non-``None`` reducer parameter but the
    channel value is ``None`` at injection time.

    A handler signature like ``def h(event, strategy: str)`` declares that
    ``strategy`` must be a ``str`` — ``None`` is not in the annotation, so the
    framework asserts the value before calling the handler. To opt out, widen
    the annotation to ``str | None`` / ``Optional[str]`` / ``Any``.
    """


class SkipType:
    """The type of :data:`SKIP`.

    Annotate a ``ScalarReducer.fn`` that can return ``SKIP`` as
    ``-> str | SkipType``. The stored type of the reducer leaves ``SkipType``
    out, because ``SKIP`` is never stored.
    """

    __slots__ = ()

    def __repr__(self) -> str:
        return "SKIP"


SKIP = SkipType()
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


E = TypeVar("E")
"""The event type that a reducer reads."""


def _union_args(hint: object) -> tuple[object, ...]:
    """The members of a union *hint*, or *hint* alone."""
    origin = typing.get_origin(hint)
    if origin is typing.Union or origin is types.UnionType:
        return typing.get_args(hint)
    return (hint,)


def _substituted(hint: object, bound: Mapping[object, object]) -> object:
    """*hint* with each type variable of *bound* replaced by its value."""
    if isinstance(hint, TypeVar):
        return bound.get(hint, hint)
    params = getattr(hint, "__parameters__", ())
    if not params:
        return hint
    return hint[tuple(bound.get(p, p) for p in params)]  # type: ignore[index]


def _base_arguments(
    klass: type, generic: type, bound: Mapping[object, object]
) -> tuple[object, ...]:
    """The type arguments that *klass* gives *generic* through its bases.
    *bound* maps the type parameters of *klass* to their values."""
    for base in klass.__dict__.get("__orig_bases__", ()):
        origin = typing.get_origin(base) or base
        if not isinstance(origin, type) or not issubclass(origin, generic):
            continue
        args = tuple(_substituted(a, bound) for a in typing.get_args(base))
        if origin is generic:
            return args
        inner = dict(zip(getattr(origin, "__parameters__", ()), args, strict=False))
        return _base_arguments(origin, generic, inner)
    return ()


def _type_name(spec: object) -> str:
    """A short name of an event type spec, for the fix of an error."""
    if isinstance(spec, tuple):
        return " | ".join(_type_name(t) for t in spec)
    return getattr(spec, "__name__", None) or _hint_label(spec)


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
    _declared: tuple[object, ...]

    def __setattr__(self, name: str, value: object) -> None:
        # ``Reducer[E, T](...)`` sets ``__orig_class__`` after ``__init__``.
        # Keep its type arguments in a field too, so ``dataclasses.replace``
        # carries them to the copy.
        if name == "__orig_class__":
            object.__setattr__(self, "_declared", typing.get_args(value))
        object.__setattr__(self, name, value)

    def _type_arguments(self, generic: type) -> tuple[object, ...]:
        """The type arguments that this reducer declares for *generic*.

        For *generic* itself, they are the arguments of the subscribed call.
        For a subclass, they come from its bases, with each type parameter
        of the subclass replaced by the argument of the call at its
        position. A parameter with no argument stays a type variable.
        """
        if type(self) is generic:
            return self._declared
        params = getattr(type(self), "__parameters__", ())
        bound = dict(zip(params, self._declared, strict=False))
        return _base_arguments(type(self), generic, bound)

    def _declared_argument(self, generic: type, example: str) -> object:
        """The last type argument that this reducer declares for *generic*.

        Raises ``TypeError`` that names the reducer and the fix when the
        argument is missing, or names no class yet. *example* is the
        subscription that the fix shows.
        """
        arguments = self._type_arguments(generic)
        label = f"{type(self).__name__} {self.name!r}"
        if not arguments:
            raise TypeError(
                f"{label} declares no stored type. Subscript it: {example}(...). "
                f"The framework checks each reader against the declared type, "
                f"and never guesses it."
            )
        declared = arguments[-1]
        try:
            _normal(declared)
        except UnsettledTypeError as exc:
            raise TypeError(
                f"{label} declares the stored type {declared!r}, but {exc}. "
                f"Subscript it with a class: {example}(...)."
            ) from None
        return declared

    def _require_fn_returns(
        self, fn: Callable[..., object], declared: object, *, skips: bool
    ) -> None:
        """Raise ``TypeError`` when the return annotation of *fn*, without
        ``SkipType``, is not *declared*, or does not resolve. A function with
        no return annotation, such as a lambda, is not checked. *skips* says
        whether the reducer supports ``SKIP``."""
        hints, errors = _resolve_parameter_hints(fn)
        label = f"{type(self).__name__} {self.name!r}"
        if "return" in errors:
            raise TypeError(
                f"{label} has a fn whose return annotation does not resolve "
                f"({errors['return']}). Make every name in it importable at run "
                f"time."
            )
        returns = hints.get("return", inspect.Parameter.empty)
        if returns is inspect.Parameter.empty:
            return
        if not skips and SkipType in _union_args(returns):
            raise TypeError(
                f"{label} has a fn that can return SkipType, but a "
                f"{type(self).__name__} does not support SKIP. Return an empty "
                f"list for no contribution, and remove SkipType from the "
                f"annotation."
            )
        kept = tuple(a for a in _union_args(returns) if a is not SkipType)
        if not kept:
            raise TypeError(
                f"{type(self).__name__} {self.name!r} has a fn that returns only "
                f"SkipType, so it never sets a value. Return the value type "
                f"too, for example -> str | SkipType, or remove the reducer."
            )
        returned = typing.Union[kept] if len(kept) > 1 else kept[0]  # noqa: UP007 - a runtime value
        if not _same_type(returned, declared, may_be_unset=False):
            merge = (
                " For a merge that stores another item type, use a FoldReducer "
                "or a BaseReducer subclass."
                if isinstance(self, Reducer)
                else ""
            )
            raise TypeError(
                f"{type(self).__name__} {self.name!r} declares "
                f"{_hint_label(declared)}, but its fn returns "
                f"{_hint_label(returned)}. Make the two the same type.{merge}"
            )

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

    @property
    def _reads_log(self) -> bool:
        """A stored reducer reads its own channel only, never the log."""
        return False

    def _read(self, read: StateRead) -> object:
        return self.read(read.state)

    @abstractmethod
    def stored_type(self) -> object:
        """The declared type of the value that :meth:`read` gives.

        Graph construction checks the annotation of each reader against it.
        Raises ``TypeError`` when the reducer declares no type.
        """

    def _value_type(self) -> object:
        return self.stored_type()

    def _describe(self) -> str:
        return f"reducer {self.name!r}"

    def _not_set_hint(self, reader: str) -> str:
        """The first fix that a ``ReducerNotSetError`` names. *reader* is
        the kind of reader, such as ``"this handler"``.

        A catch-all reducer (``event_type=Event``) gives no useful hint.
        """
        from langgraph_events._event import Event  # noqa: PLC0415

        if self.event_type is Event:
            return f"Set the reducer before {reader} runs, or "
        type_name = getattr(self.event_type, "__name__", str(self.event_type))
        return (
            f"Ensure an event of type {type_name} has been "
            f"processed before {reader} runs, or "
        )


class Readable(Protocol):
    """A value that a handler or a :class:`Derived` reads by parameter name.

    A stored reducer reads its own channel. A :class:`Derived`, once bound
    to its inputs, computes its value from them. The log input of a
    ``Derived`` reads the ``EventLog`` of the state.
    """

    @property
    def name(self) -> str: ...

    @property
    def sources(self) -> tuple[BaseReducer, ...]:
        """The stored reducers whose channels the value reads."""
        ...

    @property
    def _reads_log(self) -> bool:
        """Whether the value reads the ``EventLog`` of the state."""
        ...

    def _read(self, read: StateRead) -> Any:  # noqa: ANN401 - the value of a reducer
        """The value in *read*, with an empty default for an unset channel."""
        ...

    def _value_type(self) -> object:
        """The declared type of the value, for the check of a reader's
        annotation."""
        ...

    def _describe(self) -> str: ...

    def _not_set_hint(self, reader: str) -> str: ...


@dataclass
class Reducer(BaseReducer, Generic[E, T]):
    """Maps events to contributions for a named LangGraph state channel.

    The reducer filters events by ``event_type``, then calls ``fn`` on each
    matching event.  Its return value is merged into the state channel using
    the ``reducer`` function (defaults to ``operator.add`` for simple list
    concatenation).

    Any LangGraph-compatible reducer can be used — e.g., ``add_messages``
    from ``langchain_core.messages`` for smart message deduplication.

    Handlers receive the accumulated value by declaring a parameter whose
    name matches ``self.name``. ``Reducer[E, T]`` declares the event type
    ``E`` and the item type ``T``: the channel stores ``list[T]``. A reader
    annotates ``list[T]`` exactly.

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
    event_type: type[E] | tuple[type[E], ...] | types.UnionType = field(kw_only=True)
    fn: Callable[[E], list[T]] = field(kw_only=True)
    reducer: ReducerFn = field(kw_only=True, default=operator.add)
    default: list[T] = field(kw_only=True, default_factory=list)
    namespace: type[Namespace] | None = field(kw_only=True, default=None)
    _declared: tuple[object, ...] = field(
        kw_only=True, default=(), repr=False, compare=False
    )

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
            contrib = self.fn(cast("E", event))
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

    def stored_type(self) -> object:
        """``list[T]``, from ``Reducer[E, T]``. When ``fn`` has a return
        annotation, it must be ``list[T]`` too."""
        item = self._declared_argument(
            Reducer, f"Reducer[{_type_name(self.event_type)}, T]"
        )
        stored = list[item]  # type: ignore[valid-type]
        self._require_fn_returns(self.fn, stored, skips=False)
        return stored

    def seed(self, events: list[Event]) -> list[Any]:
        values = list(self.default)
        values.extend(self.collect(events))
        return values


@dataclass
class ScalarReducer(BaseReducer, Generic[E, T]):
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

    ``ScalarReducer[E, T]`` declares the event type ``E`` and the stored
    type ``T``. mypy checks ``fn`` against them, a lambda too. The source
    type of a reader is ``T``, joined with ``None`` when ``default`` is
    ``None``.

    Example::

        strategy = ScalarReducer[StrategyChosen, str](
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
    event_type: type[E] | tuple[type[E], ...] | types.UnionType = field(kw_only=True)
    fn: Callable[[E], T | SkipType] = field(kw_only=True)
    default: T | None = field(kw_only=True, default=None)
    namespace: type[Namespace] | None = field(kw_only=True, default=None)
    _declared: tuple[object, ...] = field(
        kw_only=True, default=(), repr=False, compare=False
    )

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
        return self.fn(cast("E", last)) if last is not SKIP else SKIP

    def has_contributions(self, result: Any) -> bool:  # noqa: ANN401 - a subclass narrows this to its own collect() result
        return result is not SKIP

    def output_type(self) -> object:
        return Any

    def stored_type(self) -> object:
        """``T``, from ``ScalarReducer[E, T]``, joined with ``None`` when
        ``default`` is ``None``. When ``fn`` has a return annotation, it must
        be ``T``, with or without ``SkipType``."""
        declared = self._declared_argument(
            ScalarReducer,
            f"ScalarReducer[{_type_name(self.event_type)}, {self._fn_example()}]",
        )
        self._require_fn_returns(self.fn, declared, skips=True)
        if self.default is None:
            return typing.Optional[declared]  # noqa: UP045 - a runtime value
        return declared

    def _fn_example(self) -> str:
        """The return type of ``fn`` without ``SkipType``, or ``T``, for the
        fix of an error."""
        returns = _resolve_parameter_hints(self.fn)[0].get("return")
        if returns is None:
            return "T"
        kept = [a for a in _union_args(returns) if a is not SkipType]
        return " | ".join(_hint_label(a) for a in kept) or "T"

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

    A checkpoint can store it as a pending write. ``NamespaceAwareSerde``
    encodes it under its own ext code, so a nested event inside it revives
    by qualname.
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

        counter = FoldReducer[dict[str, int]](
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
    fold: Callable[[S, Foldable], S | _ResetType | SkipType] = field(
        kw_only=True, default=lambda state, event: event.fold(state)
    )
    namespace: type[Namespace] | None = field(kw_only=True, default=None)
    _declared: tuple[object, ...] = field(
        kw_only=True, default=(), repr=False, compare=False
    )

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

    def stored_type(self) -> object:
        """``S``, from ``FoldReducer[S]``."""
        return self._declared_argument(FoldReducer, "FoldReducer[S]")

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
            if isinstance(result, SkipType):
                continue
            state = self.default_factory() if isinstance(result, _ResetType) else result
        return state


@dataclass(frozen=True)
class Derived(Generic[T]):
    """A value that a pure function computes from its inputs. It is not stored.

    Each parameter of ``fn`` is one input, and binds by the rule of a
    handler parameter. A parameter annotated with :class:`EventLog`
    receives the ``EventLog`` of the state. Each other parameter name names
    a stored reducer or another ``Derived``, and its annotation must name
    its declared type. A default is the value of an input that is ``None``. Before a
    handler runs, the framework calls ``fn`` with the values of the inputs.
    A handler receives the result by parameter name, like a reducer value::

        def stage_of(placed: bool | None, shipped: bool | None) -> str:
            return "shipped" if shipped else "placed"

        stage = Derived(name="stage", fn=stage_of)

        @on(Checked)
        def check(event: Checked, stage: str) -> Reported: ...

    The value is never written to the checkpoint, so it cannot drift from
    its inputs. One state read computes each ``Derived`` once. ``fn`` must
    have a return annotation, because the framework checks the annotation
    of each reader against it. ``Derived`` values must not read each other
    in a cycle. As on a handler parameter, a non-``None`` annotation with
    no default opts into :class:`ReducerNotSetError`, for the parameters of
    ``fn`` and for the handler that reads the value.

    When declared as a class attribute on a ``Namespace`` subclass, ``name``
    auto-fills from the attribute name, and ``namespace`` auto-fills to the
    enclosing class. The namespace only places the value in the structure
    views. An input can be any stored reducer or ``Derived`` of the graph.
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
        """The names of the input reducers and ``Derived`` values, in the
        order of ``fn``. A parameter annotated with ``EventLog`` is not in
        it, see :attr:`reads_log`."""
        log_params = self._log_params
        return tuple(p.name for p in self._parameters() if p.name not in log_params)

    @property
    def reads_log(self) -> bool:
        """Whether a parameter of ``fn`` is annotated with ``EventLog``."""
        return bool(self._log_params)

    @property
    def _log_params(self) -> frozenset[str]:
        hints, _ = _resolve_parameter_hints(self.fn)
        return frozenset(
            p.name for p in self._parameters() if hints.get(p.name) is EventLog
        )

    def _parameters(self) -> tuple[inspect.Parameter, ...]:
        """The parameters that a call of ``fn`` must fill. A keyword that a
        ``functools.partial`` binds is filled already."""
        try:
            return _open_parameters(self.fn)
        except (TypeError, ValueError) as exc:
            raise TypeError(
                f"Derived {self.name!r} cannot read the parameters of "
                f"fn={self.fn!r}. Pass a function whose parameters name "
                f"its inputs."
            ) from exc


class StateRead:
    """One read of one state, such as one handler injection or one frame.

    The read builds the ``EventLog`` of the state once, on first use. It
    computes each ``Derived`` once, keyed by name. It keeps nothing across
    reads, so a value cannot go stale.
    """

    def __init__(self, state: Mapping[str, Any], log: EventLog | None = None) -> None:
        """*log* is the ``EventLog`` of *state*, when the caller holds it."""
        self.state = state
        self._log = log
        self._computed: dict[str, object] = {}

    @property
    def log(self) -> EventLog:
        """The ``EventLog`` of the ``events`` and ``causes`` channels.

        A state with no ``events`` channel, such as a thread with no
        checkpoint, gives an empty log.
        """
        if self._log is None:
            # The log keeps its own tuple of events, but a reference to the
            # causes list. Copy it, so a value that keeps the log, such as a
            # view, does not see a later write that a stream appends.
            causes = self.state.get("causes")
            self._log = EventLog._from_state(
                self.state.get("events", ()),
                None if causes is None else list(causes),
                check=False,
            )
        return self._log

    def computed(self, derived: BoundDerived[T]) -> T:
        """The value of *derived* in this read, computed on first use."""
        if derived.name not in self._computed:
            self._computed[derived.name] = derived._compute(self)
        return cast("T", self._computed[derived.name])


def message_reducer(
    default: list[BaseMessage] | None = None,
    *,
    name: str = "messages",
) -> Reducer[MessageEvent, BaseMessage]:
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

    return Reducer[MessageEvent, BaseMessage](
        name=name,
        event_type=MessageEvent,
        fn=fn,
        reducer=add_messages,  # type: ignore[arg-type]
        default=resolved_default,
    )
