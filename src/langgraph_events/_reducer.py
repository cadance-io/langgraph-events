"""Reducer — generic LangGraph-native state channel for event-driven graphs."""

from __future__ import annotations

import operator
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Annotated, Any

if TYPE_CHECKING:
    from collections.abc import Callable

    from langchain_core.messages import BaseMessage

    from langgraph_events._event import Event
    from langgraph_events._types import ReducerFn


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


def _last_write_wins(existing: Any, new: Any) -> Any:
    """Binary operator that always takes the newer value."""
    return new


class BaseReducer(ABC):
    """Abstract base for all reducer types.

    Subclasses declare ``event_type`` so the framework can filter events
    before calling ``fn``.  Use a ``@runtime_checkable Protocol`` for
    structural multi-type matching.
    """

    name: str
    event_type: type

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

    name: str
    event_type: type
    fn: Callable[[Any], list[Any]]
    reducer: ReducerFn = field(default=operator.add)
    default: list[Any] = field(default_factory=list)

    def state_annotation(self) -> Any:
        return Annotated[list, self.reducer]

    @property
    def empty(self) -> Any:
        return list(self.default)

    def collect(self, events: list[Event]) -> Any:
        contributions: list[Any] = []
        for event in events:
            if isinstance(event, self.event_type):
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

    Example::

        strategy = ScalarReducer(
            name="strategy",
            event_type=StrategyChosen,
            fn=lambda e: e.strategy,
        )

        @on(TaskReceived)
        def handle(event: TaskReceived, strategy: str) -> Completed:
            ...
    """

    name: str
    event_type: type
    fn: Callable[[Any], Any]
    default: Any = None

    def state_annotation(self) -> Any:
        return Annotated[Any, _last_write_wins]

    @property
    def empty(self) -> Any:
        return self.default

    def collect(self, events: list[Event]) -> Any:
        last: Any = SKIP
        for event in events:
            if isinstance(event, self.event_type):
                last = event
        return self.fn(last) if last is not SKIP else SKIP

    def has_contributions(self, result: Any) -> bool:
        return result is not SKIP

    def output_type(self) -> Any:
        return Any

    def seed(self, events: list[Event]) -> Any:
        result = self.collect(events)
        return result if self.has_contributions(result) else self.default


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
