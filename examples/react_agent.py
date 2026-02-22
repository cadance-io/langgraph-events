"""ReAct Tool-Calling Agent — langgraph-events demo.

Demonstrates the ReAct (Reason + Act) loop using multi-subscription handlers.
The `@on(UserMessageReceived, ToolsExecuted)` pattern creates the agent loop
implicitly — no routing functions, no conditional edges, no manual graph wiring.

Events wrap LangChain message types directly (``MessageEvent`` pattern).
The ``message_reducer`` handles incremental message accumulation automatically.

Also demonstrates the **Auditable trait pattern**: events inherit from a marker
class, and a single `@on(Auditable)` handler auto-logs every marked event as
it flows through the graph — no manual isinstance printing needed.

Usage:
    export OPENAI_API_KEY="sk-..."
    python examples/react_agent.py
"""

from __future__ import annotations

import asyncio
import math
from dataclasses import dataclass

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

from langgraph_events import (
    Auditable,
    Event,
    EventGraph,
    MessageEvent,
    message_reducer,
    on,
)

# ---------------------------------------------------------------------------
# Events (past-participle: "what just happened")
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class UserMessageReceived(MessageEvent, Auditable):
    message: HumanMessage = None  # type: ignore[assignment]


@dataclass(frozen=True)
class LLMResponded(MessageEvent, Auditable):
    message: AIMessage = None  # type: ignore[assignment]


@dataclass(frozen=True)
class ToolsExecuted(MessageEvent, Auditable):
    messages: tuple[ToolMessage, ...] = ()


@dataclass(frozen=True)
class AnswerProduced(Auditable):
    content: str = ""


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    weather_data = {
        "tokyo": "22°C, partly cloudy with 60% humidity",
        "new york": "18°C, sunny with light breeze",
        "london": "14°C, overcast with occasional rain",
        "paris": "20°C, clear skies",
        "san francisco": "16°C, foggy in the morning",
    }
    return weather_data.get(city.lower(), f"25°C, clear skies in {city}")


@tool
def calculator(expression: str) -> str:
    """Evaluate a mathematical expression. Examples: '2 + 2', 'sqrt(144)', '(22 * 9/5) + 32'."""
    # Restricted to math operations only — no builtins, no imports
    allowed_names = {
        "sqrt": math.sqrt,
        "pi": math.pi,
        "abs": abs,
        "round": round,
        "pow": pow,
    }
    code = compile(expression, "<calc>", "eval")
    # Reject any name not in our allowlist
    for name in code.co_names:
        if name not in allowed_names:
            return (
                f"Error: '{name}' is not allowed. Only math operations are supported."
            )
    # Safe: restricted to math-only namespace with no builtins
    return str(eval(code, {"__builtins__": {}}, allowed_names))


TOOLS = [get_weather, calculator]
TOOL_REGISTRY = {t.name: t for t in TOOLS}

# ---------------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------------

llm = ChatOpenAI(model="gpt-4o-mini").bind_tools(TOOLS)


# ---------------------------------------------------------------------------
# Reducer — one-liner replaces the manual to_messages() chain
# ---------------------------------------------------------------------------

messages = message_reducer(
    [
        SystemMessage(
            content="You are a helpful assistant with access to tools. "
            "Use them when needed to answer the user's question accurately."
        )
    ]
)


# ---------------------------------------------------------------------------
# Handlers — the entire agent is just these three functions
# ---------------------------------------------------------------------------


@on(Auditable)
def audit_trail(event: Auditable) -> None:
    """Side-effect handler: logs all auditable events as they flow."""
    print(f"  {event.trail()}")


@on(UserMessageReceived, ToolsExecuted)
async def call_llm(event: Event, messages: list[BaseMessage]) -> LLMResponded:
    """Call the LLM with the full conversation history.

    Fires on both UserMessageReceived (initial query) and ToolsExecuted
    (after tool execution), creating the ReAct loop automatically.
    The ``messages`` reducer is maintained incrementally by the framework —
    no rebuild needed.
    """
    response = await llm.ainvoke(messages)
    return LLMResponded(message=response)


@on(LLMResponded)
def execute_tools(event: LLMResponded) -> ToolsExecuted | AnswerProduced:
    """Execute tool calls or return the final answer.

    If the LLM didn't request any tools, the response is the final answer.
    """
    if not event.message.tool_calls:
        return AnswerProduced(content=event.message.content)

    results = []
    for tc in event.message.tool_calls:
        tool_fn = TOOL_REGISTRY[tc["name"]]
        result = tool_fn.invoke(tc["args"])
        results.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))
    return ToolsExecuted(messages=tuple(results))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def main():
    graph = EventGraph(
        [call_llm, execute_tools, audit_trail],
        reducers=[messages],
    )

    question = (
        "What's the weather in Tokyo? "
        "If the temperature is in Celsius, convert it to Fahrenheit."
    )
    print(f"Question: {question}\n")
    print("--- Event Flow ---")

    log = await graph.ainvoke(
        UserMessageReceived(message=HumanMessage(content=question))
    )

    print()
    answer = log.latest(AnswerProduced)
    print(f"Final Answer: {answer.content}")


if __name__ == "__main__":
    asyncio.run(main())
