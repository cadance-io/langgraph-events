"""Helpers for bridging AG-UI tool definitions and tool results to LangChain."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ag_ui.core import RunAgentInput, Tool
    from langchain_core.messages import ToolMessage


def build_langchain_tools(tools: list[Tool] | None) -> list[dict[str, Any]]:
    """Convert AG-UI tool definitions into OpenAI-format bindings.

    The returned list is suitable for ``llm.bind_tools(...)`` on any
    LangChain chat model that accepts OpenAI-style function specs.  Frontend
    tools are never executed on the backend — the LLM only needs the schema
    so it can decide to call one, which the AG-UI adapter then streams to
    the frontend as ``ToolCallStart``/``ToolCallArgs``/``ToolCallEnd``.
    """
    if not tools:
        return []
    return [
        {
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description,
                "parameters": t.parameters,
            },
        }
        for t in tools
    ]


def detect_new_tool_results(
    input_data: RunAgentInput,
    checkpoint_state: dict[str, Any] | None,
) -> list[ToolMessage]:
    """Return frontend tool-result messages that are new since the checkpoint.

    Walks ``input_data.messages`` forward and collects every ``role: "tool"``
    entry whose ``tool_call_id`` is not already present among the tool
    messages in ``checkpoint_state["messages"]``.  Handles the fresh-run
    case (``checkpoint_state is None``) by returning ``[]``.
    """
    from langchain_core.messages import ToolMessage as LCToolMessage  # noqa: PLC0415

    if checkpoint_state is None:
        return []
    existing: set[str] = set()
    checkpointed = checkpoint_state.get("messages") or []
    for msg in checkpointed:
        if getattr(msg, "type", None) == "tool":
            tc_id = getattr(msg, "tool_call_id", None)
            if tc_id:
                existing.add(tc_id)

    results: list[ToolMessage] = []
    for msg in input_data.messages or []:
        if getattr(msg, "role", None) != "tool":
            continue
        tc_id = getattr(msg, "tool_call_id", None) or ""
        if tc_id in existing:
            continue
        results.append(
            LCToolMessage(
                content=getattr(msg, "content", "") or "",
                tool_call_id=tc_id,
                id=getattr(msg, "id", None) or None,
            )
        )
    return results
