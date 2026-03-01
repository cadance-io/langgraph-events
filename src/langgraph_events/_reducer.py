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


def _last_write_wins(existing: Any, new: Any) -> Any:
    """Binary operator that always takes the newer value."""
    return new


class BaseReducer(ABC):
    """Abstract base for all reducer types."""

    name: str

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

    The ``fn`` is called once per event when it is produced. Its return
    value is merged into the state channel using the ``reducer`` function
    (defaults to ``operator.add`` for simple list concatenation).

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
    fn: Callable[[Event], list[Any]]
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

    The ``fn`` is called once per event. It should return ``T | None``;
    the last non-None value wins and is injected directly into the handler.

    Note: ``None`` signals "no contribution". To store ``None`` as
    a meaningful value, wrap it (e.g. use a sentinel dataclass).

    Example::

        strategy = ScalarReducer(
            name="strategy",
            fn=lambda e: e.strategy if isinstance(e, StrategyChosen) else None,
        )

        @on(TaskReceived)
        def handle(event: TaskReceived, strategy: str) -> Done:
            ...
    """

    name: str
    fn: Callable[[Event], Any]
    default: Any = None

    def state_annotation(self) -> Any:
        return Annotated[Any, _last_write_wins]

    @property
    def empty(self) -> Any:
        return self.default

    def collect(self, events: list[Event]) -> Any:
        result = None
        for event in events:
            val = self.fn(event)
            if val is not None:
                result = val
        return result

    def has_contributions(self, result: Any) -> bool:
        return result is not None

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

    resolved_default = default or []

    def fn(event: Event) -> list[BaseMessage]:
        return event.as_messages()

    return Reducer(name=name, fn=fn, reducer=add_messages, default=resolved_default)  # type: ignore[arg-type]
