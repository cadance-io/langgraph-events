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
    Command,
    DomainEvent,
    Event,  # noqa: F401 (importable for reducer event_type=Event catch-all)
    FrontendToolCallRequested,
    Halted,
    HandlerRaised,
    IntegrationEvent,
    Interrupted,
    Invariant,
    InvariantViolated,
    MaxRoundsExceeded,
    MessageEvent,
    Namespace,
    Resumed,
    Scatter,
    SystemEvent,
    SystemPromptSet,
)
from langgraph_events._event_log import EventLog
from langgraph_events._graph import (
    EventGraph,
    GraphState,
    LLMStreamEnd,
    LLMToken,
    LLMToolCallChunk,
    OrphanedEventWarning,
)
from langgraph_events._handler import on
from langgraph_events._namespace import NamespaceModel
from langgraph_events._reducer import SKIP, Reducer, ScalarReducer, message_reducer
from langgraph_events._types import (
    HandlerReturn,  # noqa: F401 (importable but not promoted)
)

__all__ = [
    "SKIP",
    "STATE_SNAPSHOT_EVENT_NAME",
    "Auditable",
    "Cancelled",
    "Command",
    "DomainEvent",
    "EventGraph",
    "EventLog",
    "FrontendToolCallRequested",
    "GraphState",
    "Halted",
    "HandlerRaised",
    "IntegrationEvent",
    "Interrupted",
    "Invariant",
    "InvariantViolated",
    "LLMStreamEnd",
    "LLMToken",
    "LLMToolCallChunk",
    "MaxRoundsExceeded",
    "MessageEvent",
    "Namespace",
    "NamespaceModel",
    "OrphanedEventWarning",
    "Reducer",
    "Resumed",
    "ScalarReducer",
    "Scatter",
    "SystemEvent",
    "SystemPromptSet",
    "aemit_custom",
    "aemit_state_snapshot",
    "emit_custom",
    "emit_state_snapshot",
    "message_reducer",
    "on",
]
