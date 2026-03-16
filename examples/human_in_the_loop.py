"""Human-in-the-Loop Approval — langgraph-events demo.

Demonstrates the Interrupted/Resumed pattern for pausing execution and
collecting human feedback. The graph generates a draft, pauses for approval,
and supports revision cycles — all with typed events instead of raw
interrupt() calls and manual state threading.

The **Auditable trait** auto-logs every event as it flows, replacing manual
isinstance printing with a single side-effect handler.

Usage:
    export OPENAI_API_KEY="sk-..."
    python examples/human_in_the_loop.py
"""

from __future__ import annotations

import asyncio

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver

from langgraph_events import (
    Auditable,
    Event,
    EventGraph,
    EventLog,
    Interrupted,
    on,
)

# ---------------------------------------------------------------------------
# Events (past-participle: "what just happened")
# ---------------------------------------------------------------------------


class ContentRequested(Auditable):
    topic: str = ""
    tone: str = "professional"


class ReviewApproved(Auditable):
    """Human approved the draft."""


class RevisionFeedback(Auditable):
    feedback: str = ""


class RevisionRequested(Auditable):
    feedback: str = ""
    revision: int = 0


class DraftGenerated(Auditable):
    content: str = ""
    revision: int = 0


class ContentPublished(Auditable):
    content: str = ""


# ---------------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------------

llm = ChatOpenAI(model="gpt-4o-mini")


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------


@on(Auditable)
def audit_trail(event: Auditable) -> None:
    """Side-effect handler: logs all auditable events as they flow."""
    print(f"  {event.trail()}")


@on(ContentRequested, RevisionRequested)
async def generate_draft(event: Event, log: EventLog) -> DraftGenerated:
    """Generate or revise a content draft.

    Fires on both ContentRequested (initial generation) and RevisionRequested
    (revision based on human feedback).
    """
    if isinstance(event, ContentRequested):
        messages = [
            SystemMessage(
                content=(
                    f"You are a content writer. Write in a {event.tone} tone. "
                    "Keep the content concise — about 2-3 paragraphs."
                )
            ),
            HumanMessage(content=f"Write about: {event.topic}"),
        ]
        revision = 0
    else:
        prev_draft = log.latest(DraftGenerated)
        prev_content = prev_draft.content if prev_draft else "(no previous draft)"
        messages = [
            SystemMessage(
                content=(
                    "You are a content writer revising a draft based on feedback. "
                    "Keep the same general structure but address the feedback."
                )
            ),
            HumanMessage(
                content=(
                    f"Previous draft:\n{prev_content}\n\n"
                    f"Feedback:\n{event.feedback}\n\n"
                    "Please revise the draft."
                )
            ),
        ]
        revision = event.revision

    response = await llm.ainvoke(messages)
    return DraftGenerated(content=response.content, revision=revision)


@on(DraftGenerated)
def request_approval(event: DraftGenerated) -> Interrupted:
    """Pause the graph and ask the human to review the draft."""
    return Interrupted(
        prompt=(
            f"--- Draft (revision {event.revision}) ---\n\n"
            f"{event.content}\n\n"
            "---\n"
            "Type 'approve' to publish, or provide feedback for revision:"
        ),
        payload={"revision": event.revision},
    )


@on(ReviewApproved)
def publish(event: ReviewApproved, log: EventLog) -> ContentPublished:
    """Publish the draft when the human approves."""
    draft = log.latest(DraftGenerated)
    return ContentPublished(content=draft.content)


@on(RevisionFeedback)
def request_revision(event: RevisionFeedback, log: EventLog) -> RevisionRequested:
    """Request a revision cycle with the human's feedback."""
    interrupted = log.latest(Interrupted)
    revision = (interrupted.payload.get("revision", 0) + 1) if interrupted else 1
    return RevisionRequested(feedback=event.feedback, revision=revision)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


graph = EventGraph(
    [generate_draft, request_approval, publish, request_revision, audit_trail],
    checkpointer=MemorySaver(),
)


async def main():
    config = {"configurable": {"thread_id": "human-in-the-loop-demo"}}
    topic = "the benefits of event-driven architecture in modern software systems"
    print(f"Topic: {topic}\n")
    print("--- Event Flow ---")

    # First invoke — generates draft and pauses
    log = await graph.ainvoke(ContentRequested(topic=topic), config=config)

    while True:
        state = graph.get_state(config)

        if not state.is_interrupted:
            # Graph completed — extract final events
            if log.latest(ContentPublished):
                print("\n=== Published! ===")
                draft = log.latest(DraftGenerated)
                if draft:
                    print(draft.content)
            break

        # Graph is paused — show the interrupt prompt
        if state.interrupted:
            print(state.interrupted.prompt)

        # Get human input
        human_input = input("\n> ").strip()
        if not human_input:
            human_input = "approve"

        print()

        # Resume with a typed event
        if human_input.strip().lower() == "approve":
            log = await graph.aresume(ReviewApproved(), config=config)
        else:
            log = await graph.aresume(
                RevisionFeedback(feedback=human_input), config=config
            )


if __name__ == "__main__":
    asyncio.run(main())
