"""Reducer — generic LangGraph-native state channel for event-driven graphs."""

from __future__ import annotations

import operator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

    from langchain_core.messages import BaseMessage

    from langgraph_events._event import Event
    from langgraph_events._types import ReducerFn


@dataclass
class Reducer:
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

        from langchain_core.messages import SystemMessage

        messages = message_reducer([SystemMessage(content="You are helpful")])
        graph = EventGraph([call_llm], reducers=[messages])
    """
    from langgraph.graph.message import add_messages  # noqa: PLC0415

    def fn(event: Event) -> list[BaseMessage]:
        return event.as_messages()

    return Reducer(name=name, fn=fn, reducer=add_messages, default=default or [])  # type: ignore[arg-type]
