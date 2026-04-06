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


def on(*event_types: type[Event], **field_matchers: type[Event]) -> Callable[[F], F]:
    """Decorator that subscribes a handler to one or more event types.

    Multi-subscription: the handler fires when ANY of the listed types arrive.

    Field matchers narrow dispatch further — the handler only fires when the
    named field is an instance of the given type::

        @on(Resumed, interrupted=ApprovalRequested)
        def handle(event: Resumed, interrupted: ApprovalRequested):
            interrupted.draft  # type-safe, framework-guaranteed

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

    # Validate field matchers
    for field_name, field_type in field_matchers.items():
        if not (isinstance(field_type, type) and issubclass(field_type, Event)):
            raise TypeError(
                f"@on() field matcher values must be Event subclasses, "
                f"got {field_type!r} for field {field_name!r}"
            )
        # Check that at least one event type declares this field
        has_field = any(
            field_name in getattr(et, "__dataclass_fields__", {}) for et in event_types
        )
        if not has_field:
            raise TypeError(
                f"@on() field matcher references {field_name!r}, but "
                f"no field {field_name!r} exists on {event_types!r}"
            )

    def decorator(fn: F) -> F:
        fn._event_types = event_types  # type: ignore[attr-defined]
        if field_matchers:
            fn._field_matchers = dict(field_matchers)  # type: ignore[attr-defined]
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
    config_param: str | None = None
    store_param: str | None = None
    field_matchers: tuple[tuple[str, type[Event]], ...] = ()
    field_inject_params: frozenset[str] = frozenset()

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
    except Exception as exc:
        warnings.warn(
            f"Failed to resolve type hints for handler {fn.__qualname__!r}; "
            f"falling back to signature-only detection. ({exc})",
            stacklevel=3,
        )
        hints = {}

    # Find the actual parameter name annotated with EventLog
    log_param: str | None = None
    for param_name, hint in hints.items():
        if hint is EventLog:
            log_param = param_name
            break

    # Detect config and store parameters by type hint
    from langchain_core.runnables import RunnableConfig  # noqa: PLC0415
    from langgraph.store.base import BaseStore  # noqa: PLC0415

    config_param: str | None = None
    store_param: str | None = None
    for param_name, hint in hints.items():
        if hint is RunnableConfig:
            config_param = param_name
        elif hint is BaseStore:
            store_param = param_name

    # Detect reducer parameters by name match
    sig = inspect.signature(fn)
    reducer_params = tuple(name for name in sig.parameters if name in reducer_names)

    # Extract field matchers
    raw_field_matchers: dict[str, type[Event]] = getattr(fn, "_field_matchers", {})
    field_matchers = tuple(raw_field_matchers.items())
    field_inject_params = frozenset(
        name for name in sig.parameters if name in raw_field_matchers
    )

    # Warn about handler params that don't match any known injection source
    if reducer_names:
        first_param = next(iter(sig.parameters), None)
        known_params = {first_param}  # first param is always the event
        if log_param:
            known_params.add(log_param)
        if config_param:
            known_params.add(config_param)
        if store_param:
            known_params.add(store_param)
        known_params.update(reducer_names)
        known_params.update(raw_field_matchers.keys())
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
        name=fn.__name__,
        fn=fn,
        event_types=tuple(event_types),
        log_param=log_param,
        is_async=asyncio.iscoroutinefunction(fn),
        reducer_params=reducer_params,
        config_param=config_param,
        store_param=store_param,
        field_matchers=field_matchers,
        field_inject_params=field_inject_params,
    )
