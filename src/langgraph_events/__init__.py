"""langgraph-events — Opinionated event-driven abstraction for LangGraph."""

from langgraph_events._custom_event import (
    STATE_SNAPSHOT_EVENT_NAME,
    aemit_custom,
    aemit_state_snapshot,
    emit_custom,
    emit_state_snapshot,
)
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
    CustomEventFrame,
    EventGraph,
    GraphState,
    LLMStreamEnd,
    LLMToken,
    StateSnapshotFrame,
    StreamFrame,
)
from langgraph_events._handler import on
from langgraph_events._reducer import Reducer, ScalarReducer, message_reducer
from langgraph_events._types import (
    HandlerReturn,  # noqa: F401 (importable but not promoted)
)

__all__ = [
    "STATE_SNAPSHOT_EVENT_NAME",
    "Auditable",
    "CustomEventFrame",
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
    "StateSnapshotFrame",
    "StreamFrame",
    "SystemPromptSet",
    "aemit_custom",
    "aemit_state_snapshot",
    "emit_custom",
    "emit_state_snapshot",
    "message_reducer",
    "on",
]
