"""AG-UI Frontend Tools (LLM-initiated) — langgraph-events demo.

Wires CopilotKit's ``useFrontendTool`` (v2) to an ``EventGraph`` end to end.
The LLM streams ``tool_call_chunks``; the AG-UI adapter translates them to
``ToolCallStart``/``ToolCallArgs``/``ToolCallEnd``; the frontend handler fires
and sends its return value back as a ``role: "tool"`` message; a resume
factory detects it and continues the graph.

Frontend (React + CopilotKit v2) — wire the page to this backend::

    import { useFrontendTool } from "@copilotkit/react-core";
    import { z } from "zod";

    useFrontendTool({
      name: "run_scenario",
      description: "Execute a scenario on the client and stream results.",
      parameters: z.object({
        scenario_id: z.string(),
        environment_id: z.string(),
      }),
      handler: async ({ scenario_id, environment_id }, { signal }) => {
        const res = await fetch(`/scenarios/${scenario_id}/run`, { signal });
        return JSON.stringify(await res.json());
      },
    });

The agent-side wiring is below.  Run a minimal FastAPI app that exposes
``AGUIAdapter(...).stream(input_data)`` via
``create_starlette_response(...)`` on an endpoint the CopilotKit runtime
will POST to; the frontend tools registered on the page arrive as
``RunAgentInput.tools`` on every request.
"""

from __future__ import annotations

import asyncio

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver

from langgraph_events import (
    Auditable,
    Event,
    EventGraph,
    EventLog,
    MessageEvent,
    message_reducer,
    on,
)
from langgraph_events.agui import (
    AGUIAdapter,
    build_langchain_tools,
    detect_new_tool_results,
)

# ---------------------------------------------------------------------------
# Events (past-participle: "what just happened")
# ---------------------------------------------------------------------------


class ToolsRegistered(Event):
    """Seed event carrying the frontend-declared tool definitions."""

    tools: tuple = ()


class UserMessageReceived(MessageEvent, Auditable):
    message: HumanMessage = None  # type: ignore[assignment]


class LLMResponded(MessageEvent, Auditable):
    message: AIMessage = None  # type: ignore[assignment]


class ToolsExecuted(MessageEvent, Auditable):
    messages: tuple[ToolMessage, ...] = ()


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------


@on(Auditable)
def audit_trail(event: Auditable) -> None:
    print(f"  {event.trail()}")


@on(UserMessageReceived, ToolsExecuted)
async def call_llm(
    event: Event,
    messages: list[BaseMessage],
    log: EventLog,
) -> LLMResponded:
    """Call the LLM, binding the frontend tools declared by the page."""
    registered = log.latest(ToolsRegistered)
    tool_specs = build_langchain_tools(list(registered.tools)) if registered else []
    llm = ChatOpenAI(model="gpt-4o-mini")
    if tool_specs:
        llm = llm.bind_tools(tool_specs)
    response = await llm.ainvoke(messages)
    return LLMResponded(message=response)


# ---------------------------------------------------------------------------
# Graph + adapter
# ---------------------------------------------------------------------------


graph = EventGraph(
    [call_llm, audit_trail],
    reducers=[message_reducer()],
    checkpointer=MemorySaver(),
)


def seed_factory(input_data, checkpoint_state=None):
    """Produce the seed events for a fresh run.

    Captures both the frontend-registered tools (as a ``ToolsRegistered``
    event so any handler can read them via ``EventLog``) and the inbound
    user message.
    """
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
        ToolsRegistered(tools=tuple(input_data.tools or [])),
        UserMessageReceived(message=user_msg),
    ]


def resume_factory(input_data, checkpoint_state=None):
    """Resume the graph when the frontend returns one or more tool messages."""
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
# Stand-alone smoke driver (no HTTP)
# ---------------------------------------------------------------------------


async def _smoke() -> None:
    """Drive the graph without CopilotKit to verify the wiring locally.

    A real deployment would wrap ``adapter.stream(...)`` in
    ``create_starlette_response(...)`` and mount it on a FastAPI route.
    """
    from ag_ui.core import RunAgentInput, Tool, UserMessage

    inp = RunAgentInput(
        thread_id="demo-1",
        run_id="run-1",
        state={},
        messages=[
            UserMessage(id="u-1", role="user", content="Run scenario S-42"),
        ],
        tools=[
            Tool(
                name="run_scenario",
                description="Execute a scenario on the client.",
                parameters={
                    "type": "object",
                    "properties": {
                        "scenario_id": {"type": "string"},
                        "environment_id": {"type": "string"},
                    },
                    "required": ["scenario_id"],
                },
            ),
        ],
        context=[],
        forwarded_props={},
    )
    async for evt in adapter.stream(inp):
        print(evt.type, getattr(evt, "tool_call_name", ""))


if __name__ == "__main__":
    asyncio.run(_smoke())
