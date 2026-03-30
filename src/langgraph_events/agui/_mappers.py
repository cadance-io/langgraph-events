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
    TextMessageContentEvent,
    TextMessageEndEvent,
    TextMessageStartEvent,
    ToolCallArgsEvent,
    ToolCallEndEvent,
    ToolCallResultEvent,
    ToolCallStartEvent,
)

from langgraph_events._event import (
    Event,
    Interrupted,
    MessageEvent,
    Resumed,
    SystemPromptSet,
)

from ._protocols import AGUICustomEvent, AGUISerializable

if TYPE_CHECKING:
    from collections.abc import Mapping

    from ag_ui.core import Message

    from ._context import MapperContext


_warned_classes: set[type] = set()


def _warn_missing_agui_dict(cls: type) -> None:
    if cls not in _warned_classes:
        _warned_classes.add(cls)
        warnings.warn(
            f"{cls.__name__} does not implement agui_dict(); "
            f"skipping AG-UI serialization. Implement AGUISerializable "
            f"to include this event in the AG-UI stream.",
            stacklevel=3,
        )


def _langchain_to_agui_messages(
    messages: list[Any],
    *,
    id_overrides: Mapping[str, str] | None = None,
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
        raw_id = getattr(msg, "id", None) or ""
        msg_id = id_overrides.get(raw_id, raw_id) if id_overrides and raw_id else raw_id
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
                    content=msg.content or None,
                    tool_calls=tool_calls,
                )
            )
        elif msg_type == "system":
            result.append(SystemMessage(id=msg_id, role="system", content=msg.content))
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
    """Suppress framework-internal events (Resumed, SystemPromptSet)."""

    def map(self, event: Event, ctx: MapperContext) -> list[BaseEvent] | None:
        if isinstance(event, (Resumed, SystemPromptSet)):
            return []
        return None


class InterruptedMapper:
    """Map Interrupted events to AG-UI CustomEvent."""

    def map(self, event: Event, ctx: MapperContext) -> list[BaseEvent] | None:
        if not isinstance(event, Interrupted):
            return None
        if not isinstance(event, AGUISerializable):
            _warn_missing_agui_dict(type(event))
            return []
        return [
            CustomEvent(
                type=EventType.CUSTOM,
                name="interrupted",
                value=event.agui_dict(),
            )
        ]


class MessageEventMapper:
    """Map MessageEvent with AIMessage content to AG-UI text/tool events."""

    def map(self, event: Event, ctx: MapperContext) -> list[BaseEvent] | None:
        if not isinstance(event, MessageEvent):
            return None
        messages = event.as_messages()
        ai_messages = [m for m in messages if m.type == "ai"]
        tool_messages = [m for m in messages if m.type == "tool"]
        if not ai_messages and not tool_messages:
            return None

        result: list[BaseEvent] = []
        for msg in ai_messages:
            lc_msg_id = getattr(msg, "id", None)
            if ctx.was_streamed_ai_message(lc_msg_id):
                continue
            if ctx.was_emitted_message(lc_msg_id):
                continue
            msg_id = ctx.next_message_id()

            # Text content
            content = msg.content if isinstance(msg.content, str) else ""
            if content:
                result.append(
                    TextMessageStartEvent(
                        type=EventType.TEXT_MESSAGE_START,
                        message_id=msg_id,
                        role="assistant",
                    )
                )
                result.append(
                    TextMessageContentEvent(
                        type=EventType.TEXT_MESSAGE_CONTENT,
                        message_id=msg_id,
                        delta=content,
                    )
                )
                result.append(
                    TextMessageEndEvent(
                        type=EventType.TEXT_MESSAGE_END,
                        message_id=msg_id,
                    )
                )

            # Tool calls
            tool_calls = getattr(msg, "tool_calls", None)
            if tool_calls:
                for tc in tool_calls:
                    tc_id = tc.get("id", "") if isinstance(tc, dict) else tc.id
                    tc_name = tc.get("name", "") if isinstance(tc, dict) else tc.name
                    tc_args = tc.get("args", {}) if isinstance(tc, dict) else tc.args
                    result.append(
                        ToolCallStartEvent(
                            type=EventType.TOOL_CALL_START,
                            tool_call_id=tc_id,
                            tool_call_name=tc_name,
                            parent_message_id=msg_id,
                        )
                    )
                    result.append(
                        ToolCallArgsEvent(
                            type=EventType.TOOL_CALL_ARGS,
                            tool_call_id=tc_id,
                            delta=json.dumps(tc_args),
                        )
                    )
                    result.append(
                        ToolCallEndEvent(
                            type=EventType.TOOL_CALL_END,
                            tool_call_id=tc_id,
                        )
                    )

            if lc_msg_id:
                ctx.mark_emitted_message(lc_msg_id)

        for msg in tool_messages:
            lc_msg_id = getattr(msg, "id", None)
            if ctx.was_emitted_message(lc_msg_id):
                continue
            result.append(
                ToolCallResultEvent(
                    type=EventType.TOOL_CALL_RESULT,
                    message_id=ctx.next_message_id(),
                    tool_call_id=getattr(msg, "tool_call_id", ""),
                    content=msg.content if isinstance(msg.content, str) else "",
                    role="tool",
                )
            )
            if lc_msg_id:
                ctx.mark_emitted_message(lc_msg_id)

        return result


class FallbackMapper:
    """Map any unclaimed event to AG-UI CustomEvent."""

    def map(self, event: Event, ctx: MapperContext) -> list[BaseEvent] | None:
        if not isinstance(event, AGUISerializable):
            _warn_missing_agui_dict(type(event))
            return []
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


def default_mappers() -> list[Any]:
    """Return the default mapper chain in priority order."""
    return [
        SkipInternalMapper(),
        InterruptedMapper(),
        MessageEventMapper(),
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
    *,
    id_overrides: Mapping[str, str] | None = None,
) -> MessagesSnapshotEvent:
    """Build a MessagesSnapshotEvent from a LangChain message list."""
    return MessagesSnapshotEvent(
        type=EventType.MESSAGES_SNAPSHOT,
        messages=_langchain_to_agui_messages(messages, id_overrides=id_overrides),
    )
