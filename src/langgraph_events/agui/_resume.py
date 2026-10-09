"""Helpers for AG-UI ResumeFactory implementations.

Bridges AG-UI ``RunAgentInput`` shapes into LangChain/langgraph state suitable
for resume events. All public helpers here are pure functions — no I/O, no
global state.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

from ._extras import collect_inbound_extras

if TYPE_CHECKING:
    from ag_ui.core import InputContentPart, Message
    from ag_ui.core.types import RunAgentInput
    from langchain_core.messages import BaseMessage
    from langchain_core.messages.tool_call import ToolCall as LCToolCall

logger = logging.getLogger(__name__)

BLOCK_TYPES = {"image": "image", "audio": "audio", "video": "video", "document": "file"}
"""The LangChain standard block type for each AG-UI media part type."""

SOURCE_KEYS = {"data": "base64", "url": "url"}
"""The LangChain block key that holds the value of each AG-UI source type.

A ``file`` source is not in the table, so an inbound part with one is dropped.
That source names a handle in the provider account. The server sends the
handle with its own credentials, so a client handle can read a file that the
client does not own.
"""


def _content_to_langchain(
    content: str | list[InputContentPart], message_id: str
) -> str | list[str | dict[str, Any]]:
    """Convert AG-UI message content to LangChain message content.

    Each part becomes the LangChain standard block of the same modality. A part
    with a source type outside :data:`SOURCE_KEYS` is dropped with a WARNING.
    Part ``metadata`` is dropped with a WARNING, because a LangChain block has
    no field for it.
    """
    if isinstance(content, str):
        return content
    kept = [p for p in content if p.type == "text" or p.source.type in SOURCE_KEYS]
    if len(kept) < len(content):
        logger.warning(
            "Dropping %d content part(s) with a file source from AG-UI message "
            "%s — a client must not name a provider file handle.",
            len(content) - len(kept),
            message_id,
        )
    with_metadata = sum(part.metadata is not None for part in kept)
    if with_metadata:
        logger.warning(
            "Dropping the metadata of %d content part(s) from AG-UI message %s "
            "— a LangChain content block has no field for it.",
            with_metadata,
            message_id,
        )
    return [_part_to_block(part) for part in kept]


def _part_to_block(part: InputContentPart) -> dict[str, Any]:
    if part.type == "text":
        block: dict[str, Any] = {"type": "text", "text": part.text}
    else:
        block = {
            "type": BLOCK_TYPES[part.type],
            SOURCE_KEYS[part.source.type]: part.source.value,
        }
        if part.source.mime_type:
            block["mime_type"] = part.source.mime_type
    if part.id is not None:
        block["id"] = part.id
    return block


def agui_messages_to_langchain(  # noqa: PLR0912
    messages: list[Message],
    *,
    drop_invalid_tool_calls: bool = False,
) -> list[BaseMessage]:
    """Convert AG-UI protocol messages to LangChain ``BaseMessage`` instances.

    Reasoning and developer messages are skipped (logged at DEBUG); activity
    and unknown roles raise ``ValueError``.

    A truthy AG-UI ``ToolMessage.error`` becomes ``ToolMessage.status="error"``.
    Fields the client added beyond the AG-UI schema land under the reserved
    ``additional_kwargs[AGUI_EXTRAS_KEY]`` key. Those fields are unvalidated
    client input that reaches the checkpoint and is served back out. Read
    :func:`~langgraph_events.agui._extras.collect_inbound_extras` before you
    rely on them, and note the size cap in :data:`AGUI_EXTRAS_MAX_BYTES`.

    The default ``drop_invalid_tool_calls=False`` propagates
    ``json.JSONDecodeError`` for parity with upstream ``ag-ui-langgraph`` —
    drop-in replacement for migrators. Set ``True`` for production resume
    factories that need resilience: ``AssistantMessage`` tool_calls whose
    ``function.arguments`` fail ``json.loads`` are dropped (WARNING-logged);
    if all tool_calls in a message are invalid, the message itself is dropped.
    """
    from ag_ui.core import AssistantMessage, UserMessage  # noqa: PLC0415
    from ag_ui.core import SystemMessage as AGUISystemMessage  # noqa: PLC0415
    from ag_ui.core import ToolMessage as AGUIToolMessage  # noqa: PLC0415
    from langchain_core.messages import (  # noqa: PLC0415
        AIMessage,
        HumanMessage,
        SystemMessage,
        ToolMessage,
    )

    out: list[BaseMessage] = []
    for m in messages:
        if isinstance(m, UserMessage):
            out.append(
                HumanMessage(
                    id=m.id,
                    content=_content_to_langchain(m.content, m.id),
                    name=m.name,
                    additional_kwargs=collect_inbound_extras(m),
                )
            )
        elif isinstance(m, AssistantMessage):
            tool_calls: list[LCToolCall] = []
            for tc in m.tool_calls or []:
                raw = tc.function.arguments
                if not raw:
                    args: Any = {}
                else:
                    try:
                        args = json.loads(raw)
                    except json.JSONDecodeError:
                        if drop_invalid_tool_calls:
                            logger.warning(
                                "Dropping AG-UI tool_call %s — unparseable arguments",
                                tc.id,
                            )
                            continue
                        raise
                tool_calls.append(
                    {
                        "id": tc.id,
                        "name": tc.function.name,
                        "args": args,
                        "type": "tool_call",
                    }
                )
            if drop_invalid_tool_calls and m.tool_calls and not tool_calls:
                logger.warning(
                    "Dropping AG-UI assistant message %s — all tool_calls invalid",
                    m.id,
                )
                continue
            out.append(
                AIMessage(
                    id=m.id,
                    content=m.content or "",
                    tool_calls=tool_calls,
                    name=m.name,
                    additional_kwargs=collect_inbound_extras(m),
                )
            )
        elif isinstance(m, AGUISystemMessage):
            out.append(
                SystemMessage(
                    id=m.id,
                    content=m.content,
                    name=m.name,
                    additional_kwargs=collect_inbound_extras(m),
                )
            )
        elif isinstance(m, AGUIToolMessage):
            out.append(
                ToolMessage(
                    id=m.id,
                    content=_content_to_langchain(m.content, m.id),
                    tool_call_id=m.tool_call_id,
                    # A truthy `error` marks the failure, in both directions.
                    # The outbound mapper never sends an empty `error`, so a
                    # client that initialises `error: ""` reports no failure.
                    status="error" if m.error else "success",
                    additional_kwargs=collect_inbound_extras(m),
                )
            )
        else:
            role = getattr(m, "role", type(m).__name__)
            if role in ("reasoning", "developer"):
                logger.debug(
                    "Skipping AG-UI %s message %s", role, getattr(m, "id", "?")
                )
            else:
                raise ValueError(f"Unsupported message role: {role}")
    return out


def merge_frontend_messages(
    input_data: RunAgentInput,
    checkpoint_state: dict[str, Any] | None,
    *,
    reducer_name: str = "messages",
    drop_invalid_tool_calls: bool = True,
) -> tuple[BaseMessage, ...]:
    """Merge frontend AG-UI messages into the existing reducer message list.

    Reads existing messages from
    ``checkpoint_state["reducers"][reducer_name]`` (empty if missing or
    ``None``), converts ``input_data.messages`` via
    :func:`agui_messages_to_langchain`, and merges via langgraph's
    ``add_messages`` (id-based dedup).

    A stored message wins over an inbound message with the same id. The
    server owns each message that its ``MessagesSnapshot`` sent, as AG-UI
    defines that event. A client echoes that history back, and an echo can
    be lossy, for example when a part is dropped. So only a message with a
    new id is converted and added.

    Defensive default: malformed tool-call JSON is dropped (with a WARNING).
    Pass ``drop_invalid_tool_calls=False`` for strict parity with upstream.
    """
    from langgraph.graph.message import add_messages  # noqa: PLC0415

    reducers = (checkpoint_state or {}).get("reducers") or {}
    existing = list(reducers.get(reducer_name) or [])
    stored_ids = {m.id for m in existing}
    new = agui_messages_to_langchain(
        [m for m in input_data.messages or [] if m.id not in stored_ids],
        drop_invalid_tool_calls=drop_invalid_tool_calls,
    )
    # langgraph's add_messages signature accepts dict/tuple/str shapes for
    # checkpoint-stored messages; our inputs and output are always BaseMessage.
    merged: list[BaseMessage] = add_messages(existing, new)  # type: ignore[arg-type,assignment]
    return tuple(merged)


def extract_resume_input(input_data: RunAgentInput) -> Any:
    """Pull resume input from ``RunAgentInput.forwarded_props.command.resume``.

    If the value is a string, attempts ``json.loads`` (returns the decoded
    JSON value on success; the raw string on ``JSONDecodeError``). Dicts,
    lists, and numbers pass through unchanged. Returns ``None`` if absent or
    falsy.
    """
    forwarded = input_data.forwarded_props or {}
    resume = (forwarded.get("command") or {}).get("resume")
    if not resume:
        return None
    if isinstance(resume, str):
        try:
            return json.loads(resume)
        except json.JSONDecodeError:
            return resume
    return resume
