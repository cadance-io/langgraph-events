"""AG-UI adapter layer for langgraph-events."""

from typing import TYPE_CHECKING, NoReturn

from langgraph_events.agui._adapter import AGUIAdapter
from langgraph_events.agui._context import MapperContext
from langgraph_events.agui._events import (
    FrontendStateMutated,
    FrontendToolCallRequested,
    InterruptedWithPayload,
)
from langgraph_events.agui._extras import (
    AGUI_EXTRAS_KEY,
    AGUI_EXTRAS_MAX_BYTES,
    AGUI_EXTRAS_OPEN_FIELDS,
)
from langgraph_events.agui._mappers import (
    FallbackMapper,
    FrontendToolCallRequestedMapper,
    InterruptedMapper,
    SkipInternalMapper,
    UnmappedEventError,
)
from langgraph_events.agui._protocols import (
    AGUICustomEvent,
    AGUISerializable,
    EventMapper,
    ResumeFactory,
    SeedFactory,
)
from langgraph_events.agui._resume import (
    agui_messages_to_langchain,
    detect_new_frontend_messages,
    extract_resume_input,
)
from langgraph_events.agui._tools import (
    build_langchain_tools,
    detect_new_tool_results,
)
from langgraph_events.agui._transport import (
    create_starlette_response,
    encode_sse_stream,
)

# --- Removed names ------------------------------------------------------
# Each import of a removed name raises ``ImportError`` with its migration.
# Names listed here are intentionally absent from ``__all__``.
_REMOVED: dict[str, str] = {
    "merge_frontend_messages": (
        "'merge_frontend_messages' was removed. Use "
        "'detect_new_frontend_messages'. It returns only the inbound messages "
        "that the thread does not store, not the full history. A handler that "
        "reads 'event.messages' as the whole conversation must inject the "
        "messages reducer instead."
    ),
}


if not TYPE_CHECKING:

    def __getattr__(name: str) -> NoReturn:
        migration = _REMOVED.get(name)
        if migration is None:
            raise AttributeError(
                f"module 'langgraph_events.agui' has no attribute {name!r}"
            )
        raise ImportError(migration)


__all__ = [
    "AGUI_EXTRAS_KEY",
    "AGUI_EXTRAS_MAX_BYTES",
    "AGUI_EXTRAS_OPEN_FIELDS",
    "AGUIAdapter",
    "AGUICustomEvent",
    "AGUISerializable",
    "EventMapper",
    "FallbackMapper",
    "FrontendStateMutated",
    "FrontendToolCallRequested",
    "FrontendToolCallRequestedMapper",
    "InterruptedMapper",
    "InterruptedWithPayload",
    "MapperContext",
    "ResumeFactory",
    "SeedFactory",
    "SkipInternalMapper",
    "UnmappedEventError",
    "agui_messages_to_langchain",
    "build_langchain_tools",
    "create_starlette_response",
    "detect_new_frontend_messages",
    "detect_new_tool_results",
    "encode_sse_stream",
    "extract_resume_input",
]
