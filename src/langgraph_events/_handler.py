"""@on decorator and handler metadata extraction."""

from __future__ import annotations

import asyncio
import inspect
from dataclasses import dataclass
from typing import Any, Callable

from langgraph_events._event import Event
from langgraph_events._event_log import EventLog


def on(*event_types: type[Event]) -> Callable:
    """Decorator that subscribes a handler to one or more event types.

    Multi-subscription: the handler fires when ANY of the listed types arrive.

    Examples::

        @on(DocumentReceived)
        async def classify(event: DocumentReceived) -> DocumentClassified:
            return DocumentClassified(...)

        @on(UserMessage, ToolResults)
        async def call_llm(event: Event, log: EventLog) -> AssistantMessage:
            ...
    """
    if not event_types:
        raise TypeError("@on() requires at least one Event subclass")

    for et in event_types:
        if not (isinstance(et, type) and issubclass(et, Event)):
            raise TypeError(
                f"@on() requires Event subclasses, got {et!r}"
            )

    def decorator(fn: Callable) -> Callable:
        fn._event_types = event_types  # type: ignore[attr-defined]
        return fn

    return decorator


@dataclass(frozen=True)
class HandlerMeta:
    """Extracted metadata about a registered handler."""

    name: str
    fn: Callable[..., Any]
    event_types: tuple[type[Event], ...]
    wants_log: bool
    is_async: bool


def extract_handler_meta(fn: Callable) -> HandlerMeta:
    """Extract handler metadata from a decorated function."""
    event_types = getattr(fn, "_event_types", None)
    if event_types is None:
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
        event_types=tuple(event_types),
        wants_log=wants_log,
        is_async=asyncio.iscoroutinefunction(fn),
    )
