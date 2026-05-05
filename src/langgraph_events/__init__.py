"""langgraph-events — Opinionated event-driven abstraction for LangGraph."""

from typing import Any

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
    on_namespace_finalize,
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
from langgraph_events._namespace._smells import DomainPatternWarning
from langgraph_events._reducer import SKIP, Reducer, ScalarReducer, message_reducer
from langgraph_events._types import (
    HandlerReturn,  # noqa: F401 (importable but not promoted)
)

# --- Deprecated top-level aliases ----------------------------------------
# Kept callable for one minor version; emits ``DeprecationWarning`` per access.
# Drop in a future release. Names listed here are intentionally absent from
# ``__all__`` so ``from langgraph_events import *`` does not pull them in.
_DEPRECATED_AGUI_ALIASES: dict[str, str] = {
    "FrontendToolCallRequested": "langgraph_events.agui",
}


def __getattr__(name: str) -> Any:
    target_module = _DEPRECATED_AGUI_ALIASES.get(name)
    if target_module is None:
        raise AttributeError(f"module 'langgraph_events' has no attribute {name!r}")
    import importlib  # noqa: PLC0415
    import warnings  # noqa: PLC0415

    warnings.warn(
        f"'langgraph_events.{name}' has moved to '{target_module}.{name}'. "
        f"Update imports to 'from {target_module} import {name}'. The "
        f"top-level alias will be removed in a future release.",
        DeprecationWarning,
        stacklevel=2,
    )
    return getattr(importlib.import_module(target_module), name)


__all__ = [
    "SKIP",
    "STATE_SNAPSHOT_EVENT_NAME",
    "Auditable",
    "Cancelled",
    "Command",
    "DomainEvent",
    "DomainPatternWarning",
    "EventGraph",
    "EventLog",
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
    "on_namespace_finalize",
]
