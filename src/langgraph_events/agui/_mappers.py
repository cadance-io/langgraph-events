"""Built-in AG-UI event mappers."""

from __future__ import annotations

import json
import warnings
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

    from ._context import MapperContext


_warned_classes: set[type] = set()


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
        if msg_type == "human":
            result.append(UserMessage(id=msg_id, role="user", content=msg.content))
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
                AssistantMessage(
                    id=msg_id,
                    role="assistant",
                    content=msg.content if isinstance(msg.content, str) else None,
                    tool_calls=tool_calls,
                )
            )
        elif msg_type == "system":
            result.append(
                SystemMessage(
                    id=msg_id,
                    role="system",
                    content=msg.content if isinstance(msg.content, str) else "",
                )
            )
        elif msg_type == "tool":
            result.append(
                AguiToolMessage(
                    id=msg_id,
                    role="tool",
                    content=msg.content or "",
                    tool_call_id=getattr(msg, "tool_call_id", ""),
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
