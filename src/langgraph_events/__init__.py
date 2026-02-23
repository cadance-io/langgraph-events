"""langgraph-events — Opinionated event-driven abstraction for LangGraph."""

from langgraph_events._event import (
    Auditable,
    Event,
    Halt,
    Interrupted,
    MessageEvent,
    Resumed,
    Scatter,
    SystemPromptSet,
)
from langgraph_events._event_log import EventLog
from langgraph_events._graph import EventGraph
from langgraph_events._handler import on
from langgraph_events._reducer import Reducer, message_reducer
from langgraph_events._types import HandlerReturn

__all__ = [
    "Auditable",
    "Event",
    "EventGraph",
    "EventLog",
    "Halt",
    "HandlerReturn",
    "Interrupted",
    "MessageEvent",
    "Reducer",
    "Resumed",
    "Scatter",
    "SystemPromptSet",
    "message_reducer",
    "on",
]
