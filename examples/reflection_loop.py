"""Reflection Loop — langgraph-events demo.

Demonstrates the autonomous generate/critique/revise pattern using
multi-subscription handlers. The `@on(WriteRequest, Critique)` pattern
creates the revision cycle implicitly — the generate handler fires on
both the initial request and on critique feedback, looping until a
quality threshold or revision cap is reached.

Uses `EventLog.latest()` for lookback to enforce a revision cap,
and union return types (`Critique | FinalDraft`) to branch the flow.

Also demonstrates the **Auditable trait pattern**: events inherit from a marker
class, and a single `@on(Auditable)` handler auto-logs every marked event as
it flows through the graph.

Usage:
    export OPENAI_API_KEY="sk-..."
    python examples/reflection_loop.py
"""

from __future__ import annotations

import asyncio

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from langgraph_events import Auditable, Event, EventGraph, EventLog, on

# ---------------------------------------------------------------------------
# Events (past-participle: "what just happened")
# ---------------------------------------------------------------------------


class WriteRequest(Auditable):
    topic: str = ""
    max_revisions: int = 3


class Draft(Auditable):
    content: str = ""
    revision: int = 0


class Critique(Auditable):
    feedback: str = ""
    revision: int = 0


class FinalDraft(Auditable):
    content: str = ""


# ---------------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------------

llm = ChatOpenAI(model="gpt-4o-mini")


# ---------------------------------------------------------------------------
# Handlers — the entire loop is just these three functions
# ---------------------------------------------------------------------------


@on(Auditable)
def audit_trail(event: Auditable) -> None:
    """Side-effect handler: logs all auditable events as they flow."""
    print(f"  {event.trail()}")


@on(WriteRequest, Critique)
async def generate(event: Event, log: EventLog) -> Draft:
    """Generate or revise a draft.

    Fires on both WriteRequest (initial generation) and Critique
    (revision based on feedback), creating the revision cycle automatically.
    """
    if isinstance(event, Critique):
        prev_draft = log.latest(Draft)
        messages = [
            SystemMessage(
                content=(
                    "You are a skilled writer revising a draft based on feedback. "
                    "Preserve the overall structure but address every point in the feedback."
                )
            ),
            HumanMessage(
                content=(
                    f"Previous draft:\n{prev_draft.content}\n\n"
                    f"Feedback:\n{event.feedback}\n\n"
                    "Please revise the draft."
                )
            ),
        ]
        revision = event.revision + 1
    else:
        messages = [
            SystemMessage(
                content=(
                    "You are a skilled writer. Write a clear, engaging short essay "
                    "(2-3 paragraphs) on the given topic."
                )
            ),
            HumanMessage(content=f"Write about: {event.topic}"),
        ]
        revision = 0

    response = await llm.ainvoke(messages)
    return Draft(content=response.content, revision=revision)


@on(Draft)
async def evaluate(event: Draft, log: EventLog) -> Critique | FinalDraft:
    """Evaluate a draft — either critique it or accept it as final.

    Uses EventLog.latest() to look back at the original WriteRequest
    for the revision cap.
    """
    request = log.latest(WriteRequest)
    if event.revision >= request.max_revisions:
        return FinalDraft(content=event.content)

    messages = [
        SystemMessage(
            content=(
                "You are a sharp editorial critic. Read the draft and provide "
                "specific, actionable feedback in 2-3 bullet points. "
                "Focus on clarity, structure, and persuasiveness."
            )
        ),
        HumanMessage(content=event.content),
    ]
    response = await llm.ainvoke(messages)
    return Critique(feedback=response.content, revision=event.revision)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def main():
    graph = EventGraph([generate, evaluate, audit_trail])

    topic = "why event-driven architecture leads to simpler agent designs"
    print(f"Topic: {topic}")
    print("Max revisions: 3\n")
    print("--- Event Flow ---")

    log = await graph.ainvoke(WriteRequest(topic=topic, max_revisions=3))

    print()
    drafts = log.filter(Draft)
    print(f"Total drafts: {len(drafts)} (1 initial + {len(drafts) - 1} revisions)")

    final = log.latest(FinalDraft)
    print(f"\n=== Final Draft ===\n{final.content}")


if __name__ == "__main__":
    asyncio.run(main())
