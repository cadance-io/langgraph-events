"""Built-in AG-UI event mappers."""

from __future__ import annotations

import json
import warnings
from collections.abc import Mapping
from functools import cache
from typing import TYPE_CHECKING, Any

from ag_ui.core import (
    BaseEvent,
    CustomEvent,
    EventType,
    MessagesSnapshotEvent,
    StateSnapshotEvent,
    ToolCallArgsEvent,
    ToolCallEndEvent,
    ToolCallStartEvent,
)

from langgraph_events._event import (
    Event,
    Interrupted,
    Resumed,
    SystemPromptSet,
)

from ._events import (
    FrontendStateMutated,
    FrontendToolCallRequested,
    InterruptedWithPayload,
)
from ._protocols import AGUICustomEvent, AGUISerializable

if TYPE_CHECKING:
    from ag_ui.core import Message
    from pydantic import BaseModel

    from ._context import MapperContext


_warned_classes: set[type] = set()

AGUI_EXTRAS_KEY = "agui"
"""Reserved ``BaseMessage.additional_kwargs`` key for AG-UI passthrough fields.

A consumer puts a mapping under this key on a LangChain message. Every entry
becomes an extra field on the AG-UI message the adapter sends to the client.
The same key receives the client's extra fields on the way back in. Nothing
else in ``additional_kwargs`` crosses the wire.
"""

TOOL_ERROR_STATUS = "error"
"""The LangChain ``ToolMessage.status`` value that marks a failed tool result.

The literal doubles as the fallback for AG-UI ``ToolMessage.error``. AG-UI
declares ``error`` as a string, and an empty string is falsy, so a client
writing ``if (msg.error)`` reads a failed tool result as a success. An errored
tool message with empty content therefore sends this literal. The literal is
fixed, so it carries nothing about the tool, its arguments or its result.
"""


class UnmappedEventError(TypeError):
    """Raised by ``AGUIAdapter(on_unmapped="raise")`` for an event that reaches
    the fallback path without implementing ``AGUISerializable``."""

    def __init__(self, cls: type) -> None:
        super().__init__(
            f"{cls.__name__} has no AG-UI mapping and does not implement "
            f"agui_dict(). Implement AGUISerializable to serialize it, register "
            f"a custom EventMapper, or pass on_unmapped='ignore' to drop it."
        )


def _warn_missing_agui_dict(cls: type) -> None:
    if cls not in _warned_classes:
        _warned_classes.add(cls)
        warnings.warn(
            f"{cls.__name__} does not implement agui_dict(); "
            f"skipping AG-UI serialization. Implement AGUISerializable "
            f"to include this event in the AG-UI stream.",
            stacklevel=3,
        )


def _handle_unmapped(cls: type, on_unmapped: str) -> list[BaseEvent]:
    """Apply the ``on_unmapped`` policy to an event with no AG-UI mapping.

    ``"raise"`` raises ``UnmappedEventError``; ``"warn"`` emits the
    once-per-class warning; ``"ignore"`` is silent. ``warn`` and ``ignore``
    both drop the event by returning ``[]``.
    """
    if on_unmapped == "raise":
        raise UnmappedEventError(cls)
    if on_unmapped == "warn":
        _warn_missing_agui_dict(cls)
    return []


@cache
def _declared_names(cls: type[BaseModel]) -> frozenset[str]:
    """Return every name that already addresses a field on *cls*.

    An AG-UI model has ``populate_by_name=True`` with a camelCase alias
    generator, so ``tool_call_id`` and ``toolCallId`` both reach the same
    declared field. Both spellings are reserved.
    """
    names: set[str] = set()
    for name, field in cls.model_fields.items():
        names.add(name)
        if field.alias:
            names.add(field.alias)
    return frozenset(names)


def _build_agui_message(
    cls: Any,
    source: Any,
    **fields: Any,
) -> Any:
    """Build an AG-UI message, adding the passthrough fields *source* carries.

    The passthrough fields come from ``source.additional_kwargs[AGUI_EXTRAS_KEY]``.
    An entry that addresses a declared field would silently rewrite protocol
    data, so a collision raises ``ValueError`` naming the key.
    """
    extras = getattr(source, "additional_kwargs", None) or {}
    passthrough = extras.get(AGUI_EXTRAS_KEY)
    if passthrough is None:
        return cls(**fields)
    if not isinstance(passthrough, Mapping):
        raise ValueError(
            f"additional_kwargs[{AGUI_EXTRAS_KEY!r}] must be a mapping of "
            f"AG-UI message fields, got {type(passthrough).__name__}."
        )
    collisions = sorted(set(passthrough) & _declared_names(cls))
    if collisions:
        raise ValueError(
            f"additional_kwargs[{AGUI_EXTRAS_KEY!r}] must not carry "
            f"{', '.join(collisions)} — {cls.__name__} declares that field. "
            f"Rename the entry, or set the field through the LangChain message."
        )
    return cls(**fields, **passthrough)


def _langchain_to_agui_messages(
    messages: list[Any],
) -> list[Message]:
    """Convert LangChain BaseMessage list to AG-UI Message format."""
    from ag_ui.core import (  # noqa: PLC0415
        AssistantMessage,
        SystemMessage,
        ToolCall,
        UserMessage,
    )
    from ag_ui.core import ToolMessage as AguiToolMessage  # noqa: PLC0415
    from ag_ui.core.types import FunctionCall  # noqa: PLC0415

    result: list[Message] = []
    for msg in messages:
        msg_type = msg.type
        msg_id = getattr(msg, "id", None) or ""
        msg_name = getattr(msg, "name", None)
        if msg_type == "human":
            result.append(
                _build_agui_message(
                    UserMessage,
                    msg,
                    id=msg_id,
                    role="user",
                    content=msg.content,
                    name=msg_name,
                )
            )
        elif msg_type == "ai":
            tool_calls = None
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                tool_calls = [
                    ToolCall(
                        id=tc.get("id", ""),
                        type="function",
                        function=FunctionCall(
                            name=tc.get("name", ""),
                            arguments=json.dumps(tc.get("args", {})),
                        ),
                    )
                    for tc in msg.tool_calls
                ]
            result.append(
                _build_agui_message(
                    AssistantMessage,
                    msg,
                    id=msg_id,
                    role="assistant",
                    content=msg.content if isinstance(msg.content, str) else None,
                    name=msg_name,
                    tool_calls=tool_calls,
                )
            )
        elif msg_type == "system":
            result.append(
                _build_agui_message(
                    SystemMessage,
                    msg,
                    id=msg_id,
                    role="system",
                    content=msg.content if isinstance(msg.content, str) else "",
                    name=msg_name,
                )
            )
        elif msg_type == "tool":
            # AG-UI declares tool content as a plain string, and ToolMessage
            # has no name field. Block content has no lossless mapping, so it
            # degrades to "" rather than raising inside the snapshot build.
            content = msg.content if isinstance(msg.content, str) else ""
            # An errored result must reach the client as a truthy `error`, or
            # `if (msg.error)` reads the failure as a success. The content is
            # the reason when there is one. Empty content — genuine, or block
            # content degraded above — falls back to the status literal.
            is_error = getattr(msg, "status", None) == TOOL_ERROR_STATUS
            result.append(
                _build_agui_message(
                    AguiToolMessage,
                    msg,
                    id=msg_id,
                    role="tool",
                    content=content,
                    tool_call_id=getattr(msg, "tool_call_id", ""),
                    error=(content or TOOL_ERROR_STATUS) if is_error else None,
                )
            )
    return result


class SkipInternalMapper:
    """Suppress framework-internal events (Resumed, SystemPromptSet,
    FrontendStateMutated).

    ``FrontendStateMutated`` originates from the client — echoing it back
    over the wire is redundant.  Its downstream reducer changes surface
    through the usual ``StateSnapshotEvent`` path.
    """

    def map(self, event: Event, ctx: MapperContext) -> list[BaseEvent] | None:
        if isinstance(event, (Resumed, SystemPromptSet, FrontendStateMutated)):
            return []
        return None


class FrontendToolCallRequestedMapper:
    """Emit ToolCallStart/Args/End for a FrontendToolCallRequested event.

    Runs before ``InterruptedMapper`` so the generic interrupt mapping never
    sees a FrontendToolCallRequested — the frontend receives the tool-call
    streaming triple and then the graph pauses via the existing Interrupted
    machinery.
    """

    def map(self, event: Event, ctx: MapperContext) -> list[BaseEvent] | None:
        if not isinstance(event, FrontendToolCallRequested):
            return None
        args_delta = json.dumps(event.args)
        return [
            ToolCallStartEvent(
                type=EventType.TOOL_CALL_START,
                tool_call_id=event.tool_call_id,
                tool_call_name=event.name,
            ),
            ToolCallArgsEvent(
                type=EventType.TOOL_CALL_ARGS,
                tool_call_id=event.tool_call_id,
                delta=args_delta,
            ),
            ToolCallEndEvent(
                type=EventType.TOOL_CALL_END,
                tool_call_id=event.tool_call_id,
            ),
        ]


class InterruptedMapper:
    """Map Interrupted events to AG-UI CustomEvent.

    ``InterruptedWithPayload`` subclasses are recognized via their
    ``interrupt_payload()`` method (no ``agui_dict()`` override needed);
    other ``Interrupted`` subclasses must implement ``AGUISerializable``.
    """

    def __init__(self, on_unmapped: str = "warn") -> None:
        self._on_unmapped = on_unmapped

    def map(self, event: Event, ctx: MapperContext) -> list[BaseEvent] | None:
        if not isinstance(event, Interrupted):
            return None
        if isinstance(event, InterruptedWithPayload):
            return [
                CustomEvent(
                    type=EventType.CUSTOM,
                    name="interrupted",
                    value=event.interrupt_payload(),
                )
            ]
        if not isinstance(event, AGUISerializable):
            return _handle_unmapped(type(event), self._on_unmapped)
        return [
            CustomEvent(
                type=EventType.CUSTOM,
                name="interrupted",
                value=event.agui_dict(),
            )
        ]


class FallbackMapper:
    """Map any unclaimed event to AG-UI CustomEvent."""

    def __init__(self, on_unmapped: str = "warn") -> None:
        self._on_unmapped = on_unmapped

    def map(self, event: Event, ctx: MapperContext) -> list[BaseEvent] | None:
        if not isinstance(event, AGUISerializable):
            return _handle_unmapped(type(event), self._on_unmapped)
        name = (
            event.agui_event_name
            if isinstance(event, AGUICustomEvent)
            else type(event).__name__
        )
        return [
            CustomEvent(
                type=EventType.CUSTOM,
                name=name,
                value=event.agui_dict(),
            )
        ]


def default_mappers(on_unmapped: str = "warn") -> list[Any]:
    """Return the default mapper chain in priority order."""
    return [
        SkipInternalMapper(),
        FrontendToolCallRequestedMapper(),
        InterruptedMapper(on_unmapped),
        # FallbackMapper is always last — added by the adapter after user mappers
    ]


def build_state_snapshot(reducers: dict[str, Any]) -> StateSnapshotEvent:
    """Build a StateSnapshotEvent from reducer data."""
    return StateSnapshotEvent(
        type=EventType.STATE_SNAPSHOT,
        snapshot=reducers,
    )


def build_messages_snapshot(
    messages: list[Any],
) -> MessagesSnapshotEvent:
    """Build a MessagesSnapshotEvent from a LangChain message list."""
    return MessagesSnapshotEvent(
        type=EventType.MESSAGES_SNAPSHOT,
        messages=_langchain_to_agui_messages(messages),
    )
