"""Public custom event emission helpers.

These helpers provide an EventGraph-native way to emit LangGraph custom stream
events from handlers without importing LangGraph callback APIs directly.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from contextvars import ContextVar, Token
from typing import Any

STATE_SNAPSHOT_EVENT_NAME = "intermediate_state"

_SyncEmitter = Callable[[str, Any], None]
_AsyncEmitter = Callable[[str, Any], Awaitable[None]]

_SYNC_EMITTER: ContextVar[_SyncEmitter | None] = ContextVar(
    "langgraph_events_sync_custom_emitter",
    default=None,
)
_ASYNC_EMITTER: ContextVar[_AsyncEmitter | None] = ContextVar(
    "langgraph_events_async_custom_emitter",
    default=None,
)


def _set_custom_emitters(
    *,
    sync_emitter: _SyncEmitter,
    async_emitter: _AsyncEmitter,
) -> tuple[Token[_SyncEmitter | None], Token[_AsyncEmitter | None]]:
    """Bind custom-event emitters for the current handler execution context."""
    sync_token = _SYNC_EMITTER.set(sync_emitter)
    async_token = _ASYNC_EMITTER.set(async_emitter)
    return sync_token, async_token


def _reset_custom_emitters(
    tokens: tuple[Token[_SyncEmitter | None], Token[_AsyncEmitter | None]],
) -> None:
    """Reset custom-event emitters after handler execution."""
    sync_token, async_token = tokens
    _SYNC_EMITTER.reset(sync_token)
    _ASYNC_EMITTER.reset(async_token)


def _require_sync_emitter() -> _SyncEmitter:
    emitter = _SYNC_EMITTER.get()
    if emitter is None:
        raise RuntimeError(
            "Custom emission helpers can only be called while an EventGraph "
            "handler is running."
        )
    return emitter


def _require_async_emitter() -> _AsyncEmitter:
    emitter = _ASYNC_EMITTER.get()
    if emitter is None:
        raise RuntimeError(
            "Custom emission helpers can only be called while an EventGraph "
            "handler is running."
        )
    return emitter


def emit_custom(name: str, data: Any) -> None:
    """Emit a LangGraph custom stream event from inside a handler.

    The payload is surfaced in ``EventGraph.astream_events(...,
    include_custom_events=True)`` as ``CustomEventFrame(name, data)``.
    """
    _require_sync_emitter()(name, data)


async def aemit_custom(name: str, data: Any) -> None:
    """Async variant of ``emit_custom`` for async handlers."""
    await _require_async_emitter()(name, data)


def emit_state_snapshot(data: dict[str, Any]) -> None:
    """Emit a typed state snapshot frame from inside a handler."""
    _require_sync_emitter()(STATE_SNAPSHOT_EVENT_NAME, data)


async def aemit_state_snapshot(data: dict[str, Any]) -> None:
    """Async variant of ``emit_state_snapshot`` for async handlers."""
    await _require_async_emitter()(STATE_SNAPSHOT_EVENT_NAME, data)
