"""langgraph-events — Opinionated event-driven abstraction for LangGraph."""

from langgraph_events._event import (
    Auditable,
    Event,
    Halted,
    Interrupted,
    MessageEvent,
    Resumed,
    Scatter,
    SystemPromptSet,
)
from langgraph_events._event_log import EventLog
from langgraph_events._graph import (
    EventGraph,
    GraphState,
    LLMStreamEnd,
    LLMToken,
    StreamFrame,
)
from langgraph_events._handler import on
from langgraph_events._reducer import Reducer, ScalarReducer, message_reducer
from langgraph_events._types import (
    HandlerReturn,  # noqa: F401 (importable but not promoted)
)

__all__ = [
    "Auditable",
    "Event",
    "EventGraph",
    "EventLog",
    "GraphState",
    "Halted",
    "Interrupted",
    "LLMStreamEnd",
    "LLMToken",
    "MessageEvent",
    "Reducer",
    "Resumed",
    "ScalarReducer",
    "Scatter",
    "StreamFrame",
    "SystemPromptSet",
    "message_reducer",
    "on",
]
