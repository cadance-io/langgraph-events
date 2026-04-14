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
    Cancelled,
    Event,
    Halted,
    HandlerRaised,
    Interrupted,
    MaxRoundsExceeded,
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
    OrphanedEventWarning,
    StateSnapshotFrame,
    StreamFrame,
)
from langgraph_events._handler import on
from langgraph_events._reducer import SKIP, Reducer, ScalarReducer, message_reducer
from langgraph_events._types import (
    HandlerReturn,  # noqa: F401 (importable but not promoted)
)

__all__ = [
    "SKIP",
    "STATE_SNAPSHOT_EVENT_NAME",
    "Auditable",
    "Cancelled",
    "CustomEventFrame",
    "Event",
    "EventGraph",
    "EventLog",
    "GraphState",
    "Halted",
    "HandlerRaised",
    "Interrupted",
    "LLMStreamEnd",
    "LLMToken",
    "MaxRoundsExceeded",
    "MessageEvent",
    "OrphanedEventWarning",
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
