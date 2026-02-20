"""@on decorator and handler metadata extraction."""

from __future__ import annotations

import asyncio
import inspect
from dataclasses import dataclass
from typing import Any, Callable

from langgraph_events._event import Event
from langgraph_events._event_log import EventLog


def on(event_type: type[Event]) -> Callable:
    """Decorator that subscribes a handler to an event type.

    Example::

        @on(DocumentReceived)
        async def classify(event: DocumentReceived) -> DocumentClassified:
            return DocumentClassified(...)
    """
    if not (isinstance(event_type, type) and issubclass(event_type, Event)):
        raise TypeError(
            f"@on() requires an Event subclass, got {event_type!r}"
        )

    def decorator(fn: Callable) -> Callable:
        fn._event_type = event_type  # type: ignore[attr-defined]
        return fn

    return decorator


@dataclass(frozen=True)
class HandlerMeta:
    """Extracted metadata about a registered handler."""

    name: str
    fn: Callable[..., Any]
    event_type: type[Event]
    wants_log: bool
    is_async: bool


def extract_handler_meta(fn: Callable) -> HandlerMeta:
    """Extract handler metadata from a decorated function."""
    event_type = getattr(fn, "_event_type", None)
    if event_type is None:
        raise ValueError(
            f"Function {fn.__qualname__!r} is not decorated with @on(EventType)"
        )

    sig = inspect.signature(fn)
    wants_log = False
    for param in sig.parameters.values():
        annotation = param.annotation
        if annotation is EventLog:
            wants_log = True
            break

    return HandlerMeta(
        name=fn.__qualname__,
        fn=fn,
        event_type=event_type,
        wants_log=wants_log,
        is_async=asyncio.iscoroutinefunction(fn),
    )
