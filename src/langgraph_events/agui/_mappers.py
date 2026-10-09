"""Built-in AG-UI event mappers."""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from functools import cache
from typing import TYPE_CHECKING, Any, TypeVar

from ag_ui.core import (
    AudioPart,
    BaseEvent,
    CustomEvent,
    DataSource,
    DocumentPart,
    EventType,
    FileSource,
    ImagePart,
    InputContentPart,
    MessagesSnapshotEvent,
    StateSnapshotEvent,
    TextPart,
    ToolCallArgsEvent,
    ToolCallEndEvent,
    ToolCallStartEvent,
    UrlSource,
    VideoPart,
)
from pydantic import BaseModel

from langgraph_events._event import (
    Event,
    Interrupted,
    Resumed,
    SystemPromptSet,
)
from langgraph_events._warn import warn_user

from ._events import (
    FrontendStateMutated,
    FrontendToolCallRequested,
    InterruptedWithPayload,
)
from ._extras import AGUI_EXTRAS_KEY
from ._protocols import AGUICustomEvent, AGUISerializable

if TYPE_CHECKING:
    from ag_ui.core import Message

    from ._context import MapperContext

logger = logging.getLogger(__name__)

_warned_classes: set[type] = set()
_warned_extras: set[tuple[type, str]] = set()

AGUIMessageT = TypeVar("AGUIMessageT", bound=BaseModel)

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
        warn_user(
            f"{cls.__name__} does not implement agui_dict(); "
            f"skipping AG-UI serialization. Implement AGUISerializable "
            f"to include this event in the AG-UI stream.",
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


def _warn_dropped_extras(cls: type, kind: str, detail: str) -> None:
    """Warn once that AG-UI passthrough fields were dropped from *cls*.

    *kind* is the dedupe key, not the message. It takes one of four fixed
    values, so the dedupe set stays bounded however many bad messages arrive.
    *detail* carries the specifics for the reader.
    """
    key = (cls, kind)
    if key in _warned_extras:
        return
    _warned_extras.add(key)
    warn_user(
        f"Dropping AG-UI passthrough fields from a {cls.__name__}: {detail} "
        f"The rest of the message is unchanged.",
    )


@cache
def _declared_names(cls: type[BaseModel]) -> frozenset[str]:
    """Return every name that already addresses a field on *cls*.

    An AG-UI model has ``populate_by_name=True`` with a camelCase alias
    generator, so ``tool_call_id`` and ``toolCallId`` both reach the same
    declared field. Both spellings are reserved.

    ``field.alias`` is the only alias an AG-UI model sets today. A model that
    set ``validation_alias`` or ``serialization_alias`` instead would need
    those read here too.
    """
    names: set[str] = set()
    for name, field in cls.model_fields.items():
        names.add(name)
        if field.alias:
            names.add(field.alias)
    return frozenset(names)


def _build_agui_message(
    cls: type[AGUIMessageT],
    source: Any,
    **fields: Any,
) -> AGUIMessageT:
    """Build an AG-UI message, adding the passthrough fields *source* carries.

    The passthrough fields come from ``source.additional_kwargs[AGUI_EXTRAS_KEY]``.

    This function never raises. It runs inside :func:`build_messages_snapshot`,
    which ``connect()`` calls on the **checkpointed** message list. A raise here
    escapes the adapter's async generator into the consumer's HTTP handler and
    never becomes a ``RUN_ERROR``, so one bad value in a checkpoint would break
    every later connect on that thread until someone edited the checkpoint. A
    value that cannot ride through is dropped, and warned about once per class
    and cause. This matches the ``tool`` branch below, which degrades block
    content rather than raising.

    Four causes drop something:

    - the reserved key holds a non-mapping — the whole value goes;
    - an entry key is not a string — that entry goes, because ``**`` refuses it;
    - the entry ``metadata`` holds a non-mapping — that entry goes;
    - an entry addresses a declared field — that entry goes, because it would
      rewrite protocol data.

    ``metadata`` is the one declared field that an entry can set. AG-UI 1.0
    declares it as an open container of extra data, and 0.x carried it as an
    extra field. The inbound side collects it into the same slot.
    """
    extras = getattr(source, "additional_kwargs", None) or {}
    passthrough = extras.get(AGUI_EXTRAS_KEY)
    if passthrough is None:
        return cls(**fields)
    if not isinstance(passthrough, Mapping):
        _warn_dropped_extras(
            cls,
            "not-a-mapping",
            f"additional_kwargs[{AGUI_EXTRAS_KEY!r}] must be a mapping of AG-UI "
            f"message fields, got {type(passthrough).__name__}.",
        )
        return cls(**fields)

    usable = {k: v for k, v in passthrough.items() if isinstance(k, str)}
    unusable = [k for k in passthrough if not isinstance(k, str)]
    if unusable:
        _warn_dropped_extras(
            cls,
            "non-string-key",
            f"an AG-UI message field name must be a string, and "
            f"{', '.join(repr(k) for k in unusable)} is not.",
        )

    if "metadata" in usable and not isinstance(usable["metadata"], Mapping):
        del usable["metadata"]
        _warn_dropped_extras(
            cls,
            "metadata-not-a-mapping",
            f"the entry metadata must be a mapping, because {cls.__name__} "
            f"declares metadata as one.",
        )
    collisions = sorted(set(usable) & _declared_names(cls) - {"metadata"})
    for name in collisions:
        del usable[name]
    if collisions:
        _warn_dropped_extras(
            cls,
            "declared-field",
            f"{cls.__name__} declares {', '.join(collisions)}, so the entry "
            f"would rewrite protocol data. Rename the entry, or set the field "
            f"through the LangChain message.",
        )
    return cls(**fields, **usable)


_PART_TYPES: dict[str, Any] = {
    "image": ImagePart,
    "audio": AudioPart,
    "video": VideoPart,
    "file": DocumentPart,
}
"""The AG-UI content part for each LangChain standard media block type."""


def _content_to_agui(
    content: str | list[Any], label: str
) -> str | list[InputContentPart]:
    """Convert LangChain message content to AG-UI message content.

    Each standard block becomes the AG-UI part of the same modality. A block
    with no AG-UI part is dropped, with one WARNING that names *label*. A
    raise here would break each later ``connect()`` on the thread, as
    :func:`_build_agui_message` explains.
    """
    if isinstance(content, str):
        return content
    parts = [_block_to_part(block) for block in content]
    kept = [part for part in parts if part is not None]
    if len(kept) < len(parts):
        logger.warning(
            "Dropping %d content block(s) from message %s — they have no AG-UI "
            "content part. The rest of the content is unchanged.",
            len(parts) - len(kept),
            label,
        )
    return kept


def _block_to_part(block: Any) -> InputContentPart | None:
    if isinstance(block, str):
        return TextPart(text=block)
    if not isinstance(block, Mapping):
        return None
    block_id = _text_value(block, "id")
    if block.get("type") == "text":
        text = block.get("text")
        return TextPart(id=block_id, text=text) if isinstance(text, str) else None
    part_type = _PART_TYPES.get(block.get("type"))  # type: ignore[arg-type]
    source = _block_source(block)
    if part_type is None or source is None:
        return None
    return part_type(id=block_id, source=source)


def _block_source(block: Mapping[str, Any]) -> Any:
    """Return the AG-UI part source of a standard block, or ``None``.

    An inline block needs its ``mime_type``, because AG-UI requires one on a
    ``data`` source.
    """
    mime_type = _text_value(block, "mime_type")
    if url := _text_value(block, "url"):
        return UrlSource(value=url, mime_type=mime_type)
    if (data := _text_value(block, "base64")) and mime_type:
        return DataSource(value=data, mime_type=mime_type)
    if file_id := _text_value(block, "file_id"):
        return FileSource(value=file_id)
    return None


def _text_value(block: Mapping[str, Any], key: str) -> str | None:
    value = block.get(key)
    return value if isinstance(value, str) and value else None


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
                    content=_content_to_agui(msg.content, msg_id),
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
            # AG-UI's ToolMessage has no name field.
            tool_call_id = getattr(msg, "tool_call_id", "")
            content = _content_to_agui(msg.content, tool_call_id or msg_id)
            # An errored result must reach the client as a truthy `error`, or
            # `if (msg.error)` reads the failure as a success. String content
            # is the reason when there is one. Empty or block content falls
            # back to the status literal, because `error` is a string.
            is_error = getattr(msg, "status", None) == TOOL_ERROR_STATUS
            reason = content if isinstance(content, str) else ""
            result.append(
                _build_agui_message(
                    AguiToolMessage,
                    msg,
                    id=msg_id,
                    role="tool",
                    content=content,
                    tool_call_id=tool_call_id,
                    error=(reason or TOOL_ERROR_STATUS) if is_error else None,
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
