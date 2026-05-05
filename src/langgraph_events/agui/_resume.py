"""Helpers for AG-UI ResumeFactory implementations.

Bridges AG-UI ``RunAgentInput`` shapes into LangChain/langgraph state suitable
for resume events. All public helpers here are pure functions — no I/O, no
global state.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ag_ui.core import Message
    from ag_ui.core.types import RunAgentInput
    from langchain_core.messages import BaseMessage
    from langchain_core.messages.tool_call import ToolCall as LCToolCall

logger = logging.getLogger(__name__)


def agui_messages_to_langchain(  # noqa: PLR0912
    messages: list[Message],
    *,
    drop_invalid_tool_calls: bool = False,
) -> list[BaseMessage]:
    """Convert AG-UI protocol messages to LangChain ``BaseMessage`` instances.

    Reasoning and developer messages are skipped (logged at DEBUG); activity
    and unknown roles raise ``ValueError``.

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
    from ag_ui.core.types import (  # noqa: PLC0415
        BinaryInputContent,
        TextInputContent,
    )
    from langchain_core.messages import (  # noqa: PLC0415
        AIMessage,
        HumanMessage,
        SystemMessage,
        ToolMessage,
    )

    out: list[BaseMessage] = []
    for m in messages:
        if isinstance(m, UserMessage):
            content: str | list[str | dict[Any, Any]]
            if isinstance(m.content, list):
                parts: list[str | dict[Any, Any]] = []
                for p in m.content:
                    if isinstance(p, TextInputContent):
                        parts.append({"type": "text", "text": p.text})
                    elif isinstance(p, BinaryInputContent):
                        url = p.url or (
                            f"data:{p.mime_type};base64,{p.data}" if p.data else p.id
                        )
                        parts.append({"type": "image_url", "image_url": {"url": url}})
                content = parts
            else:
                content = m.content
            out.append(HumanMessage(id=m.id, content=content, name=m.name))
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
                )
            )
        elif isinstance(m, AGUISystemMessage):
            out.append(SystemMessage(id=m.id, content=m.content, name=m.name))
        elif isinstance(m, AGUIToolMessage):
            out.append(
                ToolMessage(id=m.id, content=m.content, tool_call_id=m.tool_call_id)
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

    Defensive default: malformed tool-call JSON is dropped (with a WARNING).
    Pass ``drop_invalid_tool_calls=False`` for strict parity with upstream.
    """
    from langgraph.graph.message import add_messages  # noqa: PLC0415

    reducers = (checkpoint_state or {}).get("reducers") or {}
    existing = list(reducers.get(reducer_name) or [])
    new = agui_messages_to_langchain(
        input_data.messages or [],
        drop_invalid_tool_calls=drop_invalid_tool_calls,
    )
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
