"""Multi-Agent Supervisor — langgraph-events demo.

Demonstrates a supervisor/coordinator pattern where typed events route tasks
to specialist agents automatically. The supervisor handler subscribes to the
``Task.Run`` command and to each specialist's completion event, and emits
sub-commands (``Task.Research``, ``Task.Code``) or the final ``Finalized``
fact — no routing functions, no subgraph state adapters.

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

from langgraph_events import (
    Aggregate,
    Auditable,
    Command,
    DomainEvent,
    EventGraph,
    EventLog,
    Reducer,
    on,
)

# ---------------------------------------------------------------------------
# Aggregate: Task
# ---------------------------------------------------------------------------


@runtime_checkable
class Contextualizable(Protocol):
    """Events that contribute context to the supervisor reducer."""

    def context_part(self) -> str: ...


class Task(Aggregate, Auditable):
    """A coordinated multi-step task.

    ``Run`` is the user intent — the entry command. The supervisor handler
    picks it up (and each specialist completion) and dispatches either a
    ``Research`` or ``Code`` sub-command, or emits the ``Finalized`` fact
    when enough context has been gathered.
    """

    class Run(Command, Auditable):
        description: str = ""

        def context_part(self) -> str:
            return f"[User Task] {self.description}"

    class Research(Command, Auditable):
        query: str = ""

        class Completed(DomainEvent, Auditable):
            findings: str = ""

            def context_part(self) -> str:
                return f"[Research Result] {self.findings}"

    class Code(Command, Auditable):
        spec: str = ""
        context: str = ""

        class Produced(DomainEvent, Auditable):
            code: str = ""

            def context_part(self) -> str:
                return f"[Code Result]\n{self.code}"

    class Finalized(DomainEvent, Auditable):
        """Final synthesized answer — terminal domain fact."""

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


@on(Task.Run, Task.Research.Completed, Task.Code.Produced)
async def supervisor(
    event: Task.Run | Task.Research.Completed | Task.Code.Produced,
    log: EventLog,
    context_parts: list,
) -> Task.Research | Task.Code | Task.Finalized:
    """Supervisor hub — decides next step based on accumulated results.

    Fires on the initial ``Task.Run``, and again whenever a specialist reports
    back (``Task.Research.Completed`` / ``Task.Code.Produced``). Uses
    tool-calling to make structured routing decisions. The ``context_parts``
    reducer is maintained incrementally by the framework — no rebuild needed.
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
        return Task.Research(query=tc["args"]["query"])
    elif tc["name"] == "delegate_code":
        research = log.latest(Task.Research.Completed)
        context_str = research.findings if research else ""
        return Task.Code(spec=tc["args"]["spec"], context=context_str)
    else:
        return Task.Finalized(answer=tc["args"]["answer"])


@on(Task.Research)
async def researcher(event: Task.Research) -> Task.Research.Completed:
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
    return Task.Research.Completed(findings=response.content)


@on(Task.Code)
async def coder(event: Task.Code) -> Task.Code.Produced:
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
    return Task.Code.Produced(code=response.content)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


graph = EventGraph(
    [supervisor, researcher, coder, audit_trail],
    reducers=[context_reducer],
)


async def main() -> None:
    task = "Research what FastAPI is, then write a simple hello world API endpoint."
    print(f"Task: {task}\n")
    print("--- Event Flow ---")

    log = await graph.ainvoke(Task.Run(description=task))

    # Show the specialist outputs
    print()
    research = log.latest(Task.Research.Completed)
    if research:
        print("=== Research Findings ===")
        print(research.findings)
        print()

    code = log.latest(Task.Code.Produced)
    if code:
        print("=== Generated Code ===")
        print(code.code)
        print()

    result = log.latest(Task.Finalized)
    print("=== Final Result ===")
    print(result.answer)


if __name__ == "__main__":
    asyncio.run(main())
