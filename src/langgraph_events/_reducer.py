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
    system: str | None = None,
    name: str = "messages",
) -> Reducer:
    """Built-in reducer for MessageEvent -> BaseMessage projection.

    Calls ``as_messages()`` on any ``MessageEvent`` and accumulates
    using ``langgraph.graph.message.add_messages`` for smart deduplication.

    Args:
        default: Optional list of initial messages (e.g. a SystemMessage).
            Mutually exclusive with ``system``.
        system: Convenience shorthand — a plain string that is wrapped in
            ``SystemMessage(content=system)`` and used as the initial default.
            Mutually exclusive with ``default``.
        name: State channel name (default ``"messages"``).

    Example::

        # Using a SystemPromptSet seed event (preferred — prompt is in the event log):
        messages = message_reducer()
        graph = EventGraph([call_llm], reducers=[messages])
        log = graph.invoke([
            SystemPromptSet.from_str("You are helpful"),
            UserMessageReceived(message=HumanMessage(content="Hi")),
        ])

        # Using the system= shorthand (static prompt, not in the event log):
        messages = message_reducer(system="You are helpful")

        # Using an explicit default list:
        messages = message_reducer([SystemMessage(content="You are helpful")])
    """
    from langgraph.graph.message import add_messages  # noqa: PLC0415

    if system is not None and default is not None:
        raise ValueError("Cannot specify both 'default' and 'system' — use one or the other")

    if system is not None:
        from langchain_core.messages import SystemMessage  # noqa: PLC0415

        resolved_default: list[BaseMessage] = [SystemMessage(content=system)]
    else:
        resolved_default = default or []

    def fn(event: Event) -> list[BaseMessage]:
        return event.as_messages()

    return Reducer(name=name, fn=fn, reducer=add_messages, default=resolved_default)  # type: ignore[arg-type]
