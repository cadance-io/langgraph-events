"""DDD Conversation Agent — langgraph-events demo.

A namespace wrapping a ReAct tool-calling agent, with an ``AGUIAdapter``
built for **AG-UI frontend tools** (CopilotKit v2 ``useFrontendTool``).
Illustrates:

- ``DomainEvent + MessageEvent`` mixin — ``Conversation.Send.Sent`` is
  both a domain fact *and* a message carrier, feeding into
  ``message_reducer()`` alongside the ``IntegrationEvent`` responses from
  the LLM.
- **AG-UI frontend tools (LLM-initiated path)** — the LLM streams
  ``tool_call_chunks``. The ``AGUIAdapter`` translates them to
  ``ToolCallStart`` / ``ToolCallArgs`` / ``ToolCallEnd`` events for the
  frontend's ``useFrontendTool`` handler. When the frontend returns its
  result, ``detect_new_tool_results`` surfaces it and the resume factory
  continues the ReAct loop with ``ToolsExecuted``.
- Mixed taxonomy — domain events (``Sent``, ``AnswerProduced``) and
  integration events (``LLMResponded``, ``ToolsExecuted``) coexist in one
  graph with a shared ``message_reducer()``.

``main()`` runs the graph directly with ``graph.ainvoke`` and no frontend
tool registered. It does not drive ``adapter.stream``. To exercise the
frontend-tool round trip, mount ``adapter.stream(input_data)`` on a FastAPI
route via ``create_starlette_response(...)``.

For the **handler-initiated path** (backend returns
``FrontendToolCallRequested(Interrupted)`` to pause for a confirm
dialog), see the snippet in ``docs/agui.md``.

Frontend (React + CopilotKit v2). Wire the page to the AG-UI adapter::

    import { useFrontendTool } from "@copilotkit/react-core";
    import { z } from "zod";

    useFrontendTool({
      name: "get_weather",
      parameters: z.object({ city: z.string() }),
      handler: async ({ city }) =>
        JSON.stringify({ temp: 22, conditions: "partly cloudy" }),
    });

Usage. The script is stand-alone, with no HTTP server. Importing the module
needs no key. Running ``main()`` needs the key::

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
# Namespace: Conversation
# ---------------------------------------------------------------------------


class Conversation(Namespace):
    """A conversation with an LLM agent.

    ``Send`` is the namespace's only entry point. Everything downstream
    (LLM call, frontend tool results) lives outside the namespace as
    ``IntegrationEvent`` handlers. The LLM and frontend tools are external
    services, not owned by the conversation. The final answer comes back
    into the namespace as ``AnswerProduced``.
    """

    class Send(Command):
        """User sends a message.

        The inline ``handle`` wraps the text in a ``HumanMessage``.
        """

        content: str = ""

        class Sent(DomainEvent, MessageEvent):
            """User message accepted and recorded."""

            message: HumanMessage = None  # type: ignore[assignment]

        def handle(self) -> Conversation.Send.Sent:
            return Conversation.Send.Sent(
                message=HumanMessage(content=self.content),
            )

    class AnswerProduced(DomainEvent):
        """Final text answer. Terminal fact of the conversation."""

        content: str = ""


# ---------------------------------------------------------------------------
# Integration events: LLM / frontend-tool boundary
# ---------------------------------------------------------------------------


class ToolsRegistered(IntegrationEvent):
    """Seed event carrying the frontend-declared tool definitions.

    The ``call_llm`` handler binds these to the LLM so the model knows
    what it can call. The actual execution happens in the frontend.
    """

    tools: tuple[Any, ...] = ()


class LLMResponded(IntegrationEvent, MessageEvent):
    """LLM returned a response (may include frontend tool calls)."""

    message: AIMessage = None  # type: ignore[assignment]


class ToolsExecuted(IntegrationEvent, MessageEvent):
    """Resume event. The frontend returned one or more tool messages."""

    messages: tuple[ToolMessage, ...] = ()


# ---------------------------------------------------------------------------
# Reducer: projects BaseMessage from any MessageEvent so handlers
# receive the full conversation history via the ``messages`` parameter.
# ---------------------------------------------------------------------------

messages = message_reducer()


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------


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
def finalize_answer(event: LLMResponded) -> Conversation.AnswerProduced | None:
    """Emit the terminal ``AnswerProduced`` when the LLM answered without
    calling tools. When ``tool_calls`` are present, do nothing. The
    AG-UI adapter's streaming path has already emitted
    ``ToolCallStart`` / ``ToolCallArgs`` / ``ToolCallEnd`` for the
    frontend. The graph pauses until the frontend returns its tool
    message, which the resume factory surfaces as ``ToolsExecuted`` to
    continue the loop.
    """
    if event.message.tool_calls:
        return None
    return Conversation.AnswerProduced(content=event.message.content)


# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------

graph = EventGraph.from_namespaces(
    Conversation,
    handlers=[call_llm, finalize_answer],
    reducers=[messages],
    checkpointer=MemorySaver(),
)


# ---------------------------------------------------------------------------
# AG-UI adapter. A production deployment mounts ``adapter.stream(inp)``
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
# Main: stand-alone smoke, no HTTP, no frontend tool registered. It does
# not drive ``adapter.stream``.
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
    for ev in log1:
        print(f"  {ev!r}")
    answer = log1.latest(Conversation.AnswerProduced)
    if answer:
        print(f"\nAnswer: {answer.content}")


if __name__ == "__main__":
    asyncio.run(main())
