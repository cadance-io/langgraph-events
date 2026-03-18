"""SSE encoding helpers for the AG-UI adapter."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from ag_ui.core import BaseEvent


async def encode_sse_stream(
    events: AsyncIterator[BaseEvent],
    accept: str | None = None,
) -> AsyncIterator[str]:
    """Encode AG-UI events as SSE strings.

    Framework-agnostic async generator that yields ``"data: ...\\n\\n"``
    strings ready for an SSE response body.

    Args:
        events: Async iterator of AG-UI BaseEvent objects.
        accept: The ``Accept`` header value (passed to EventEncoder).
    """
    from ag_ui.encoder import EventEncoder  # noqa: PLC0415

    encoder = EventEncoder(accept=accept)  # type: ignore[arg-type]
    async for event in events:
        yield encoder.encode(event)


def create_starlette_response(
    events: AsyncIterator[BaseEvent],
    accept: str | None = None,
) -> Any:
    """Create a Starlette ``StreamingResponse`` from AG-UI events.

    Convenience wrapper — lazy-imports starlette so the dependency
    is optional at the library level.

    Args:
        events: Async iterator of AG-UI BaseEvent objects.
        accept: The ``Accept`` header value (passed to EventEncoder).
    """
    from ag_ui.encoder import EventEncoder  # noqa: PLC0415
    from starlette.responses import (  # noqa: PLC0415
        StreamingResponse,
    )

    encoder = EventEncoder(accept=accept)  # type: ignore[arg-type]
    return StreamingResponse(
        encode_sse_stream(events, accept=accept),
        media_type=encoder.get_content_type(),
    )
