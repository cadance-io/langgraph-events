"""Event base class and built-in special events."""

from __future__ import annotations

import dataclasses
import functools
import operator
import types
import typing
import uuid
from dataclasses import field
from dataclasses import fields as dc_fields
from typing import TYPE_CHECKING, Any, ClassVar

if TYPE_CHECKING:
    from langchain_core.messages import BaseMessage, SystemMessage


class Event:
    """Internal base class for all events.

    Not intended for direct subclassing — use ``DomainEvent``,
    ``IntegrationEvent``, ``Command``, or compose with ``Auditable`` /
    ``MessageEvent``.  Direct subclassing raises ``TypeError``.

    Valid as a type annotation (``event: Event``) and as a reducer
    filter (``event_type=Event`` catches all events).
    """

    def __init_subclass__(cls, *, _event_base: bool = False, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if "__dataclass_fields__" not in cls.__dict__:
            dataclasses.dataclass(frozen=True)(cls)
        if Event in cls.__bases__ and not _event_base:
            raise TypeError(
                f"{cls.__name__!r} subclasses Event directly. Use one of: "
                f"DomainEvent (inside Aggregate/Command), IntegrationEvent "
                f"(cross-boundary facts), Command (inside Aggregate), or "
                f"compose with Auditable / MessageEvent."
            )

    def _collect_into(
        self,
        new_events: list[Event],
        interrupt_fn: Any,
    ) -> None:
        """Append this result to *new_events*.

        Framework-internal — overridden by ``Interrupted`` and ``Scatter``.
        """
        new_events.append(self)


_AGGREGATE_REGISTRY: dict[str, type[Aggregate]] = {}
"""Maps ``__aggregate_name__`` -> Aggregate class. Populated in
``Aggregate.__init_subclass__``. Used by ``EventGraph`` to auto-discover
declarative reducers via a handler's subscribed event types."""


class Aggregate:
    """Marker for an aggregate root in the DDD sense.

    Subclasses act as namespaces for nested ``Command`` and ``DomainEvent``
    classes. The class name becomes the aggregate's identifier, used by
    catalog introspection and for stamping ``__aggregate__`` on nested
    commands and events.

    Example::

        class Order(Aggregate):
            class Place(Command):
                customer_id: str

                class Placed(DomainEvent):
                    order_id: str
    """

    __aggregate_name__: ClassVar[str]
    __reducers__: ClassVar[tuple[Any, ...]]

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        existing = _AGGREGATE_REGISTRY.get(cls.__name__)
        if existing is not None and existing is not cls:
            raise TypeError(
                f"Aggregate {cls.__name__!r} is already defined at "
                f"{existing.__module__}.{existing.__qualname__}. Aggregate "
                f"class names must be unique within a process."
            )
        cls.__aggregate_name__ = cls.__name__
        cls.__reducers__ = _collect_aggregate_reducers(cls)
        _AGGREGATE_REGISTRY[cls.__aggregate_name__] = cls
        # Second-pass stamp: DomainEvents nested inside a Command have their
        # __aggregate__ left as None by the metaclass (Command.__aggregate__
        # isn't known at that point). Fill them in now.
        _stamp_nested_aggregate(cls, cls.__name__)
        _attach_command_outcomes(cls)


def _collect_aggregate_reducers(cls: type) -> tuple[Any, ...]:
    """Walk the MRO and collect declarative reducers from class bodies.

    Child aggregates inherit parent aggregate's reducers; dedup by name.
    Runtime import of ``BaseReducer`` to avoid module-level circular
    dependency with ``_reducer``.
    """
    from langgraph_events._reducer import BaseReducer  # noqa: PLC0415

    collected: list[Any] = []
    seen_names: set[str] = set()
    for klass in cls.__mro__[:-1]:
        for attr in klass.__dict__.values():
            if isinstance(attr, BaseReducer) and attr.name not in seen_names:
                collected.append(attr)
                seen_names.add(attr.name)
    return tuple(collected)


def _attach_command_outcomes(aggregate_cls: type) -> None:
    """For each nested ``Command``, expose ``cmd.Outcomes`` — the union of
    its nested ``DomainEvent`` classes.

    - Zero outcomes: no attribute added.
    - One outcome: ``Outcomes`` is that class.
    - Multiple outcomes: ``Outcomes`` is a ``types.UnionType`` (``A | B | ...``).
    - User already declared ``Outcomes`` in the class body: validated for drift
      against the nested outcomes; left in place if consistent.
    """
    for cmd in aggregate_cls.__dict__.values():
        if not (isinstance(cmd, type) and issubclass(cmd, Command)):
            continue
        outcomes = [
            x
            for x in cmd.__dict__.values()
            if isinstance(x, type) and issubclass(x, DomainEvent)
        ]
        if not outcomes:
            continue

        declared = cmd.__dict__.get("Outcomes")
        if declared is not None:
            declared_set = set(typing.get_args(declared)) or {declared}
            if declared_set != set(outcomes):
                raise TypeError(
                    f"Command {cmd.__qualname__!r}: declared Outcomes "
                    f"{sorted(t.__name__ for t in declared_set)} does not match "
                    f"nested DomainEvents "
                    f"{sorted(t.__name__ for t in outcomes)}. Keep them in sync."
                )
            # User-declared matches; leave as-is (mypy-visible).
            continue

        cmd.Outcomes = (  # type: ignore[attr-defined]
            outcomes[0]
            if len(outcomes) == 1
            else functools.reduce(operator.or_, outcomes)
        )


def _stamp_nested_aggregate(container: type, agg_name: str) -> None:
    """Walk ``container`` and set ``__aggregate__`` on any nested ``Event``
    subclass that doesn't have one yet.

    Covers DomainEvents nested inside a Command (their metaclass runs before
    the Command's ``__aggregate__`` is known), and non-DomainEvent events
    nested in an Aggregate for locality (e.g. ``class Blocked(Halted)`` under
    ``class Content(Aggregate)``) — the metaclass only fires for Command /
    DomainEvent, so those would otherwise never be stamped.
    """
    for attr in container.__dict__.values():
        if not isinstance(attr, type):
            continue
        if not issubclass(attr, Event):
            continue
        if getattr(attr, "__aggregate__", None) is None:
            attr.__aggregate__ = agg_name  # type: ignore[attr-defined]
        if issubclass(attr, Command):
            _stamp_nested_aggregate(attr, agg_name)


def _is_nested_in_class(cls: type) -> bool:
    """Return True if *cls* appears to be defined inside another class.

    Detects nesting via ``__qualname__``.  Function-local classes use
    ``<locals>`` markers; we count segments after the last ``<locals>`` (or
    from the start of the qualname when none is present).  Two or more
    segments means the class is nested inside another class.
    """
    parts = cls.__qualname__.split(".")
    try:
        last_locals = len(parts) - 1 - parts[::-1].index("<locals>")
        relevant = parts[last_locals + 1 :]
    except ValueError:
        relevant = parts
    return len(relevant) >= 2


class _NestedEventMeta(type):
    """Metaclass that validates aggregate-nesting when a nested class is
    assigned to its enclosing class.

    ``Command`` and ``DomainEvent`` are referenced by name below; at class
    definition time for these base classes themselves, ``__set_name__`` is
    never called (module-level classes aren't assigned as attributes). By
    the time user code triggers ``__set_name__``, both names resolve via
    module globals.
    """

    def __set_name__(cls, owner: type, name: str) -> None:
        if issubclass(cls, Command):
            if not (isinstance(owner, type) and issubclass(owner, Aggregate)):
                raise TypeError(
                    f"Command {cls.__name__!r} must be nested inside an "
                    f"Aggregate subclass, got owner {owner.__name__!r}"
                )
            cls.__aggregate__ = owner.__name__
        elif issubclass(cls, DomainEvent):
            if isinstance(owner, type) and issubclass(owner, Aggregate):
                cls.__aggregate__ = owner.__name__
                cls.__command__ = None
            elif isinstance(owner, type) and issubclass(owner, Command):
                cls.__command__ = owner
                # __aggregate__ filled in by Aggregate.__init_subclass__ — at
                # this point Command.__aggregate__ isn't known yet.
            else:
                raise TypeError(
                    f"DomainEvent {cls.__name__!r} must be nested inside an "
                    f"Aggregate or Command, got owner {owner.__name__!r}"
                )


def _inherits_aggregate(cls: type) -> bool:
    """True if any base of *cls* (other than itself) already has a stamped
    ``__aggregate__`` — meaning it inherits from an already-validated event."""
    return any(
        getattr(base, "__aggregate__", None) is not None for base in cls.__mro__[1:]
    )


def _validate_handle_signature(cls: type, handle: Any) -> None:
    """Check that an inline ``handle`` takes ``self`` as its first parameter.

    Reducer / log / config / store params are validated later at graph
    construction, where the reducer names are known.
    """
    import inspect  # noqa: PLC0415

    if isinstance(handle, (staticmethod, classmethod)):
        raise TypeError(
            f"Command {cls.__name__!r}: `handle` must be a regular method "
            f"(no @staticmethod / @classmethod)."
        )
    try:
        sig = inspect.signature(handle)
    except (TypeError, ValueError):
        return  # built-ins / C-implemented — skip silently
    params = list(sig.parameters.values())
    if not params or params[0].name != "self":
        raise TypeError(
            f"Command {cls.__name__!r}: `handle` must take `self` as its "
            f"first parameter."
        )


class Command(Event, _event_base=True, metaclass=_NestedEventMeta):
    """Imperative intent. Must be nested inside an ``Aggregate`` subclass.

    Use imperative naming (``Place``, ``Ship``, ``Cancel``).  Outcomes of
    a command are typically nested ``DomainEvent`` subclasses.

    Example::

        class Order(Aggregate):
            class Place(Command):
                customer_id: str

                class Placed(DomainEvent):
                    order_id: str

                class Rejected(DomainEvent):
                    reason: str
    """

    __aggregate__: ClassVar[str | None] = None
    __command_handler__: ClassVar[Any] = None

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if _inherits_aggregate(cls):
            return
        if not _is_nested_in_class(cls):
            raise TypeError(
                f"Command {cls.__name__!r} must be nested inside an Aggregate "
                f"subclass, e.g. "
                f"class Order(Aggregate): class {cls.__name__}(Command): ..."
            )
        # Detect an inline ``handle`` method; auto-registered when the command
        # class is passed to ``EventGraph`` (see _graph.py:_expand_command_handlers).
        handle = cls.__dict__.get("handle")
        if callable(handle):
            _validate_handle_signature(cls, handle)
            cls.__command_handler__ = handle


class DomainEvent(Event, _event_base=True, metaclass=_NestedEventMeta):
    """Fact within the bounded context. Past-participle naming.

    Must be nested inside an ``Aggregate`` (a free-standing event under
    the aggregate) or a ``Command`` (an outcome of that command).

    Example::

        class Order(Aggregate):
            class Shipped(DomainEvent):
                tracking: str
    """

    __aggregate__: ClassVar[str | None] = None
    __command__: ClassVar[type | None] = None

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if _inherits_aggregate(cls):
            return
        if not _is_nested_in_class(cls):
            raise TypeError(
                f"DomainEvent {cls.__name__!r} must be nested inside an "
                f"Aggregate or Command"
            )


class IntegrationEvent(Event, _event_base=True):
    """Fact that crosses a context or system boundary. Past-participle.

    Lives at module level (no nesting requirement), typically serializable,
    intended to be published to or consumed from external systems.

    Example::

        class PaymentConfirmed(IntegrationEvent):
            transaction_id: str
    """


class SystemEvent(Event, _event_base=True):
    """Framework-emitted fact. Past-participle.

    Reserved for events generated by the graph runtime itself
    (``Halted``, ``Interrupted``, ``HandlerRaised`` etc.).  Lives at
    module level.
    """


class MessageEvent:
    """Mixin for events that wrap LangChain messages.

    Compose with an Event branch (``DomainEvent``, ``IntegrationEvent``,
    etc.) — this class is a behavioural mixin, not an ``Event`` subclass.

    Convention:
    - ``message`` field (single BaseMessage) → ``[self.message]``
    - ``messages`` field (tuple of BaseMessage) → ``list(self.messages)``
    - Override ``as_messages()`` for custom behavior.

    Example::

        class UserMessageReceived(IntegrationEvent, MessageEvent, Auditable):
            message: HumanMessage
    """

    _event_mixin: ClassVar[bool] = True

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


class Auditable:
    """Mixin for events that should be auto-logged.

    Compose with an Event branch — this class is a behavioural mixin,
    not an ``Event`` subclass.  Use ``@on(Auditable)`` to subscribe to
    every auditable event.

    Example::

        class OrderPlaced(DomainEvent, Auditable):
            order_id: str = ""
    """

    _event_mixin: ClassVar[bool] = True

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


class Halted(SystemEvent):
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


class Interrupted(SystemEvent):
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


class FrontendToolCallRequested(Interrupted):
    """Request a frontend-executed tool call and pause the graph.

    Event-native counterpart to LLM-initiated tool calls: a handler returns
    this event, the AG-UI adapter emits ``ToolCallStart``/``ToolCallArgs``/
    ``ToolCallEnd`` for a frontend ``useFrontendTool`` handler to pick up,
    and the graph pauses via the existing ``Interrupted`` machinery.  Resume
    with a domain event (typically ``ToolsExecuted(messages=...)`` built via
    ``detect_new_tool_results`` from the frontend's tool-result message).

    Mirrors the ``ApprovalRequested(Interrupted)`` pattern — tool calls
    become "HITL with typed fields," exactly as the AG-UI spec positions
    them.  Fields are ordered so dataclass defaults follow non-defaults::

        FrontendToolCallRequested(name="confirm", args={"message": "Ship?"})
    """

    name: str = ""
    args: dict[str, Any] = field(default_factory=dict)
    tool_call_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def __post_init__(self) -> None:
        if not self.name or not self.name.strip():
            raise ValueError(
                "FrontendToolCallRequested.name must be a non-empty tool name; "
                "got empty/whitespace. Pass the same `name` your "
                "useFrontendTool({ name: ... }) registration declares."
            )

    def agui_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "args": self.args,
            "tool_call_id": self.tool_call_id,
        }


class Resumed(SystemEvent):
    """Created by the framework when a graph resumes from an interrupt.

    Contains the resume event and a reference to the original
    ``Interrupted`` event.
    """

    value: Event | None = None
    interrupted: Interrupted | None = None


class HandlerRaised(SystemEvent):
    """Emitted when a handler raises an exception declared in its ``raises=`` clause.

    Subscribe via ``@on(HandlerRaised, exception=SomeError)`` for a specific
    exception type, or ``@on(HandlerRaised)`` to catch any declared raise::

        @on(HandlerRaised, exception=RateLimitError)
        def backoff(event: HandlerRaised, exception: RateLimitError):
            exception.retry_after  # typed via field injection

    Fields (framework-populated at emit time):

    - ``handler``: name of the handler that raised.
    - ``source_event``: the event the raising handler was processing.
    - ``exception``: the caught exception instance.

    ``source_event`` is named as such (rather than ``event``) to keep the
    handler's own ``event`` parameter free when this field is used as a
    field matcher — ``@on(HandlerRaised, source_event=SomeType)`` injects
    ``source_event`` as a typed kwarg.
    """

    handler: str = ""
    source_event: Event | None = None
    exception: Exception | None = None


class Invariant:
    """Marker base for typed invariants.

    Subclass to declare one. An instance is emitted inside
    ``InvariantViolated.invariant`` when the predicate returns False;
    the subclass identity (not its instance state) is what matchers compare.

    Subclasses must be zero-arg instantiable — the framework calls ``Cls()``
    at emission time purely to satisfy ``isinstance`` matching.

    Example::

        class CustomerNotBanned(Invariant):
            '''Customer must not be banned.'''

        @on(Order.Place, invariants={CustomerNotBanned: lambda log: ...})
        def place(event: Order.Place) -> Order.Place.Placed: ...

        @on(InvariantViolated, invariant=CustomerNotBanned)
        def explain(event: InvariantViolated) -> ...: ...

    Nesting under an ``Aggregate`` / ``Command`` is encouraged as DDD-idiomatic
    but not enforced.
    """


class InvariantViolated(SystemEvent):
    """Emitted when an invariant predicate declared on a handler returns False.

    The framework evaluates each handler's ``invariants=`` predicates before
    invoking it. If any predicate returns false, the handler is skipped and
    one ``InvariantViolated`` event is emitted for the failing invariant.
    Predicate exceptions propagate — they are NOT turned into violations.

    Subscribe via ``@on(InvariantViolated)`` for all violations, or pin to a
    specific invariant via the ``invariant=`` field matcher::

        @on(InvariantViolated, invariant=CustomerNotBanned)
        def handle_banned(event: InvariantViolated) -> ...: ...

    See ``HandlerRaised`` for the ``source_event`` naming rationale.
    """

    invariant: Invariant | None = None
    handler: str = ""
    source_event: Event | None = None


class SystemPromptSet(IntegrationEvent, MessageEvent):
    """Built-in event for setting the system prompt as a first-class citizen.

    Wraps a ``SystemMessage`` so that the system prompt appears in the event
    log, is queryable via ``EventLog``, and participates in the ``Auditable``
    trail when mixed in.

    Categorized as ``IntegrationEvent`` — it crosses the user↔graph boundary
    (user code constructs it, the graph ingests it); ``SystemEvent`` is
    reserved for framework-emitted facts like ``Halted`` or ``Interrupted``.

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
