"""Multi-Agent Supervisor — langgraph-events demo.

Demonstrates a supervisor/coordinator pattern where typed events route tasks
to specialist agents automatically. The supervisor emits `ResearchDispatched`
or `CodeDispatched` events and specialist handlers are wired implicitly via
`@on` — no routing functions, no subgraph state adapters.

The **Auditable trait** auto-logs every event as it flows, replacing manual
isinstance printing with a single side-effect handler.

Usage:
    export OPENAI_API_KEY="sk-..."
    python examples/supervisor.py
"""

from __future__ import annotations

import asyncio
from typing import Protocol, runtime_checkable

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

from langgraph_events import Auditable, Event, EventGraph, EventLog, Reducer, on

# ---------------------------------------------------------------------------
# Events (past-participle: "what just happened")
# ---------------------------------------------------------------------------


@runtime_checkable
class Contextualizable(Protocol):
    """Events that contribute context to the supervisor reducer."""

    def context_part(self) -> str: ...


class TaskReceived(Auditable):
    task: str = ""

    def context_part(self) -> str:
        return f"[User Task] {self.task}"


class ResearchDispatched(Auditable):
    query: str = ""


class CodeDispatched(Auditable):
    spec: str = ""
    context: str = ""


class ResearchCompleted(Auditable):
    findings: str = ""

    def context_part(self) -> str:
        return f"[Research Result] {self.findings}"


class CodeProduced(Auditable):
    code: str = ""

    def context_part(self) -> str:
        return f"[Code Result]\n{self.code}"


class ResultFinalized(Auditable):
    answer: str = ""


# ---------------------------------------------------------------------------
# Supervisor routing tools (for structured LLM output, not execution)
# ---------------------------------------------------------------------------


@tool
def delegate_research(query: str) -> str:
    """Delegate a research task to the research specialist."""
    return query


@tool
def delegate_code(spec: str) -> str:
    """Delegate a coding task to the coding specialist."""
    return spec


@tool
def final_answer(answer: str) -> str:
    """Provide the final synthesized answer combining all specialist results."""
    return answer


ROUTING_TOOLS = [delegate_research, delegate_code, final_answer]

# ---------------------------------------------------------------------------
# LLMs
# ---------------------------------------------------------------------------

supervisor_llm = ChatOpenAI(model="gpt-4o-mini").bind_tools(
    ROUTING_TOOLS, tool_choice="required"
)
researcher_llm = ChatOpenAI(model="gpt-4o-mini")
coder_llm = ChatOpenAI(model="gpt-4o-mini")


# ---------------------------------------------------------------------------
# Reducer — projects events into context parts incrementally
# ---------------------------------------------------------------------------


context_reducer = Reducer(
    "context_parts",
    event_type=Contextualizable,
    fn=lambda e: [e.context_part()],
)


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------


@on(Auditable)
def audit_trail(event: Auditable) -> None:
    """Side-effect handler: logs all auditable events as they flow."""
    print(f"  {event.trail()}")


@on(TaskReceived, ResearchCompleted, CodeProduced)
async def supervisor(
    event: Event, log: EventLog, context_parts: list
) -> ResearchDispatched | CodeDispatched | ResultFinalized:
    """Supervisor hub — decides next step based on accumulated results.

    Fires on the initial TaskReceived, and again whenever a specialist reports
    back. Uses tool-calling to make structured routing decisions.
    The ``context_parts`` reducer is maintained incrementally by the
    framework — no rebuild needed.
    """
    context = "\n\n".join(context_parts)
    messages = [
        SystemMessage(
            content=(
                "You are a supervisor agent coordinating specialists.\n"
                "You have two specialists:\n"
                "- Research specialist: looks up information (use delegate_research)\n"
                "- Coding specialist: writes code (use delegate_code)\n\n"
                "Analyze the conversation so far and decide the next step.\n"
                "If you have enough information from all specialists, "
                "provide the final answer (use final_answer).\n"
                "Call exactly one tool."
            )
        ),
        HumanMessage(content=context),
    ]

    response = await supervisor_llm.ainvoke(messages)
    tc = response.tool_calls[0]

    if tc["name"] == "delegate_research":
        return ResearchDispatched(query=tc["args"]["query"])
    elif tc["name"] == "delegate_code":
        research = log.latest(ResearchCompleted)
        context_str = research.findings if research else ""
        return CodeDispatched(spec=tc["args"]["spec"], context=context_str)
    else:
        return ResultFinalized(answer=tc["args"]["answer"])


@on(ResearchDispatched)
async def researcher(event: ResearchDispatched) -> ResearchCompleted:
    """Research specialist — answers research queries."""
    messages = [
        SystemMessage(
            content=(
                "You are a research specialist. Provide clear, concise findings "
                "about the topic. Focus on key facts, use cases, and practical details."
            )
        ),
        HumanMessage(content=event.query),
    ]
    response = await researcher_llm.ainvoke(messages)
    return ResearchCompleted(findings=response.content)


@on(CodeDispatched)
async def coder(event: CodeDispatched) -> CodeProduced:
    """Coding specialist — writes code based on specs."""
    prompt = event.spec
    if event.context:
        prompt = f"Context:\n{event.context}\n\nTask:\n{event.spec}"

    messages = [
        SystemMessage(
            content=(
                "You are an expert Python programmer. Write clean, working code "
                "based on the specification. Include brief comments. "
                "Return only the code, no explanation."
            )
        ),
        HumanMessage(content=prompt),
    ]
    response = await coder_llm.ainvoke(messages)
    return CodeProduced(code=response.content)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


graph = EventGraph(
    [supervisor, researcher, coder, audit_trail],
    reducers=[context_reducer],
)


async def main():
    task = "Research what FastAPI is, then write a simple hello world API endpoint."
    print(f"Task: {task}\n")
    print("--- Event Flow ---")

    log = await graph.ainvoke(TaskReceived(task=task))

    # Show the specialist outputs
    print()
    research = log.latest(ResearchCompleted)
    if research:
        print("=== Research Findings ===")
        print(research.findings)
        print()

    code = log.latest(CodeProduced)
    if code:
        print("=== Generated Code ===")
        print(code.code)
        print()

    result = log.latest(ResultFinalized)
    print("=== Final Result ===")
    print(result.answer)


if __name__ == "__main__":
    asyncio.run(main())
