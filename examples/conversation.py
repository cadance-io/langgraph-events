"""DDD Conversation Agent — langgraph-events demo.

A DDD domain wrapping a ReAct tool-calling agent with **AG-UI frontend
tools** (CopilotKit v2 `useFrontendTool`). Illustrates:

- ``DomainEvent + MessageEvent`` mixin — ``Conversation.Send.Sent`` is
  both a domain fact *and* a message carrier, feeding into
  ``message_reducer()`` alongside the ``IntegrationEvent`` responses from
  the LLM.
- Content moderation via ``Command.handle`` — ``Conversation.Send``
  checks content policy before wrapping the user's text in a
  ``HumanMessage``. Rejected messages produce a ``Blocked`` domain event
  and never reach the LLM.
- **AG-UI frontend tools (LLM-initiated path)** — the LLM streams
  ``tool_call_chunks``; the ``AGUIAdapter`` translates them to
  ``ToolCallStart`` / ``ToolCallArgs`` / ``ToolCallEnd`` events for the
  frontend's ``useFrontendTool`` handler. When the frontend returns its
  result, ``detect_new_tool_results`` surfaces it and the resume factory
  continues the ReAct loop with ``ToolsExecuted``.
- Mixed taxonomy — DDD events (``Sent``, ``Blocked``) and integration
  events (``LLMResponded``, ``ToolsExecuted``, ``AnswerProduced``)
  coexist in one graph with a shared ``message_reducer()``.

For the **handler-initiated path** (backend returns
``FrontendToolCallRequested(Interrupted)`` to pause for a confirm
dialog), see the snippet in ``docs/agui.md``.

Frontend (React + CopilotKit v2) — wire the page to the AG-UI adapter::

    import { useFrontendTool } from "@copilotkit/react-core";
    import { z } from "zod";

    useFrontendTool({
      name: "get_weather",
      parameters: z.object({ city: z.string() }),
      handler: async ({ city }) =>
        JSON.stringify({ temp: 22, conditions: "partly cloudy" }),
    });

Usage (stand-alone, no HTTP; module-import only requires no key)::

    export OPENAI_API_KEY="sk-..."  # pragma: allowlist secret
    python examples/conversation.py
"""

from __future__ import annotations

import asyncio
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver

from langgraph_events import (
    Auditable,
    Command,
    DomainEvent,
    Event,
    EventGraph,
    EventLog,
    IntegrationEvent,
    MessageEvent,
    Namespace,
    SystemPromptSet,
    message_reducer,
    on,
)
from langgraph_events.agui import (
    AGUIAdapter,
    build_langchain_tools,
    detect_new_tool_results,
)

# ---------------------------------------------------------------------------
# Content policy — simple keyword blocklist (swap for an LLM classifier in
# production)
# ---------------------------------------------------------------------------

BLOCKED_WORDS = ["hack", "bomb", "exploit"]


# ---------------------------------------------------------------------------
# Namespace: Conversation
# ---------------------------------------------------------------------------


class Conversation(Namespace):
    """A moderated conversation with an LLM agent.

    ``Send`` is the domain's only entry point; it enforces content
    policy before the user's message reaches the LLM. Everything
    downstream (LLM call, frontend tool results) lives outside the
    domain as ``IntegrationEvent`` handlers — the LLM and frontend
    tools are external services, not owned by the conversation.
    """

    class Send(Command):
        """User sends a message.

        The inline ``handle`` checks content policy and wraps accepted
        text in a ``HumanMessage``. Rejected messages produce ``Blocked``
        and never trigger the LLM.
        """

        content: str = ""

        class Sent(DomainEvent, MessageEvent):
            """User message accepted and recorded."""

            message: HumanMessage = None  # type: ignore[assignment]

        class Blocked(DomainEvent):
            """User message rejected by content moderation."""

            reason: str = ""

        def handle(self) -> Conversation.Send.Sent | Conversation.Send.Blocked:
            for word in BLOCKED_WORDS:
                if word in self.content.lower():
                    return Conversation.Send.Blocked(
                        reason=f"Content policy: contains '{word}'",
                    )
            return Conversation.Send.Sent(
                message=HumanMessage(content=self.content),
            )


# ---------------------------------------------------------------------------
# Integration events — LLM / frontend-tool boundary
# ---------------------------------------------------------------------------


class ToolsRegistered(IntegrationEvent):
    """Seed event carrying the frontend-declared tool definitions.

    The ``call_llm`` handler binds these to the LLM so the model knows
    what it can call; the actual execution happens in the frontend.
    """

    tools: tuple[Any, ...] = ()


class LLMResponded(IntegrationEvent, MessageEvent, Auditable):
    """LLM returned a response (may include frontend tool calls)."""

    message: AIMessage = None  # type: ignore[assignment]


class ToolsExecuted(IntegrationEvent, MessageEvent, Auditable):
    """Resume event — the frontend returned one or more tool messages."""

    messages: tuple[ToolMessage, ...] = ()


class AnswerProduced(IntegrationEvent, Auditable):
    """Final text answer — terminal event."""

    content: str = ""


# ---------------------------------------------------------------------------
# Reducer — accumulates BaseMessage from any MessageEvent so handlers
# receive the full conversation history via the ``messages`` parameter.
# ---------------------------------------------------------------------------

messages = message_reducer()


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------


@on(Auditable)
def audit_trail(event: Auditable) -> None:
    """Side-effect handler: logs all auditable events as they flow."""
    print(f"  {event.trail()}")


@on(Conversation.Send.Sent, ToolsExecuted)
async def call_llm(
    event: Event,
    messages: list[BaseMessage],
    log: EventLog,
) -> LLMResponded:
    """Call the LLM with the full conversation history, binding the
    frontend-declared tools so the model can call them.

    Fires on both ``Sent`` (initial query) and ``ToolsExecuted`` (after
    the frontend returns tool results), creating the ReAct loop.
    """
    registered = log.latest(ToolsRegistered)
    tool_specs = build_langchain_tools(list(registered.tools)) if registered else []
    llm = ChatOpenAI(model="gpt-4o-mini")
    if tool_specs:
        llm = llm.bind_tools(tool_specs)
    response = await llm.ainvoke(messages)
    return LLMResponded(message=response)


@on(LLMResponded)
def finalize_answer(event: LLMResponded) -> AnswerProduced | None:
    """Emit the terminal ``AnswerProduced`` when the LLM answered without
    calling tools. When ``tool_calls`` are present, do nothing — the
    AG-UI adapter's streaming path has already emitted
    ``ToolCallStart`` / ``ToolCallArgs`` / ``ToolCallEnd`` for the
    frontend; the graph naturally pauses waiting for the frontend's
    returning tool message, which the resume factory will surface as
    ``ToolsExecuted`` to continue the loop.
    """
    if event.message.tool_calls:
        return None
    return AnswerProduced(content=event.message.content)


# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------

graph = EventGraph.from_namespaces(
    Conversation,
    handlers=[call_llm, finalize_answer, audit_trail],
    reducers=[messages],
    checkpointer=MemorySaver(),
)


# ---------------------------------------------------------------------------
# AG-UI adapter — production deployment mounts ``adapter.stream(inp)``
# on a FastAPI route via ``create_starlette_response(...)``.
# ---------------------------------------------------------------------------


def seed_factory(input_data, checkpoint_state=None):
    del checkpoint_state
    user_msg = next(
        (
            HumanMessage(content=m.content)
            for m in reversed(input_data.messages or [])
            if getattr(m, "role", None) == "user"
        ),
        HumanMessage(content=""),
    )
    return [
        SystemPromptSet.from_str(
            "You are a helpful assistant with access to frontend tools. "
            "Use them when needed to answer the user's question accurately."
        ),
        ToolsRegistered(tools=tuple(input_data.tools or [])),
        Conversation.Send(content=user_msg.content),
    ]


def resume_factory(input_data, checkpoint_state=None):
    results = detect_new_tool_results(input_data, checkpoint_state)
    if not results:
        return None
    return ToolsExecuted(messages=tuple(results))


adapter = AGUIAdapter(
    graph=graph,
    seed_factory=seed_factory,
    resume_factory=resume_factory,
)


# ---------------------------------------------------------------------------
# Main — stand-alone smoke, no HTTP. For the full frontend-tools
# round-trip, mount ``adapter.stream(input_data)`` on a FastAPI route.
# ---------------------------------------------------------------------------


async def main() -> None:
    print("--- Normal conversation (no frontend tools registered) ---")
    log1 = await graph.ainvoke(
        [
            SystemPromptSet.from_str("You are a helpful assistant."),
            ToolsRegistered(tools=()),
            Conversation.Send(content="Say hello in three words."),
        ]
    )
    answer = log1.latest(AnswerProduced)
    if answer:
        print(f"\nAnswer: {answer.content}")

    print("\n--- Blocked by moderation (never reaches LLM) ---")
    log2 = await graph.ainvoke(
        [
            ToolsRegistered(tools=()),
            Conversation.Send(content="How do I hack into a database?"),
        ]
    )
    blocked = log2.latest(Conversation.Send.Blocked)
    if blocked:
        print(f"Result: {blocked.reason}")
    print(f"LLM called: {log2.has(LLMResponded)}")


if __name__ == "__main__":
    asyncio.run(main())
