"""@on decorator and handler metadata extraction."""

from __future__ import annotations

import asyncio
import inspect
import typing
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from langgraph_events._event import Event
from langgraph_events._event_log import (
    EventLog,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from langgraph_events._types import F, HandlerReturn


def on(*event_types: type[Event]) -> Callable[[F], F]:
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
            raise TypeError(f"@on() requires Event subclasses, got {et!r}")

    def decorator(fn: F) -> F:
        fn._event_types = event_types  # type: ignore[attr-defined]
        return fn

    return decorator


@dataclass(frozen=True)
class HandlerMeta:
    """Extracted metadata about a registered handler."""

    name: str
    fn: Callable[..., HandlerReturn]
    event_types: tuple[type[Event], ...]
    log_param: str | None
    is_async: bool
    reducer_params: tuple[str, ...] = ()

    @property
    def wants_log(self) -> bool:
        """Backward-compatible property."""
        return self.log_param is not None


def extract_handler_meta(
    fn: Callable[..., Any],
    reducer_names: frozenset[str] = frozenset(),
) -> HandlerMeta:
    """Extract handler metadata from a decorated function."""
    event_types = getattr(fn, "_event_types", None)
    if event_types is None:
        raise ValueError(
            f"Function {fn.__qualname__!r} is not decorated with @on(EventType)"
        )

    try:
        hints = typing.get_type_hints(fn)
    except Exception:
        hints = {}

    # Find the actual parameter name annotated with EventLog
    log_param: str | None = None
    for param_name, hint in hints.items():
        if hint is EventLog:
            log_param = param_name
            break

    # Detect reducer parameters by name match
    sig = inspect.signature(fn)
    reducer_params = tuple(name for name in sig.parameters if name in reducer_names)

    # Warn about handler params that don't match any known injection source
    if reducer_names:
        first_param = next(iter(sig.parameters), None)
        known_params = {first_param}  # first param is always the event
        if log_param:
            known_params.add(log_param)
        known_params.update(reducer_names)
        unknown = [
            name
            for name in sig.parameters
            if name not in known_params and name != "self"
        ]
        if unknown:
            warnings.warn(
                f"Handler {fn.__qualname__!r} has parameter(s) {unknown} that "
                f"don't match any reducer. "
                f"Available reducers: {sorted(reducer_names)}. Typo?",
                stacklevel=3,
            )

    return HandlerMeta(
        name=fn.__qualname__,
        fn=fn,
        event_types=tuple(event_types),
        log_param=log_param,
        is_async=asyncio.iscoroutinefunction(fn),
        reducer_params=reducer_params,
    )
