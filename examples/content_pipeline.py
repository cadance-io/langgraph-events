"""Content Pipeline — langgraph-events demo.

Demonstrates event streaming, halt-based early termination, and post-run
inspection in a content classification + analysis pipeline.

Covers APIs not shown in other examples:
  - ``Halted`` — immediate graph termination (unsafe content stops the pipeline)
  - ``StreamFrame`` — NamedTuple yielded by ``astream_events()`` with reducer snapshots
  - ``stream_events()`` / ``astream_events()`` — high-level event streaming
  - ``EventLog.has()`` — boolean existence check
  - ``Reducer`` — polymorphic stage tracking via a shared parent event class

No LLM dependency — runs standalone with keyword-based classification rules.

Usage:
    python examples/content_pipeline.py
"""

from __future__ import annotations

import asyncio

from langgraph_events import (
    Auditable,
    EventGraph,
    EventLog,
    Halted,
    Reducer,
    StreamFrame,
    on,
)

# ---------------------------------------------------------------------------
# Events (past-participle: "what just happened")
# ---------------------------------------------------------------------------


class PipelineStage(Auditable):
    """Event that contributes to the pipeline stage tracker."""

    def stage_label(self) -> str:
        return ""


class ContentReceived(Auditable):
    """Seed event — user-submitted content. Not a pipeline stage."""

    text: str = ""


class ContentClassified(PipelineStage):
    category: str = ""
    safe: bool = True

    def stage_label(self) -> str:
        return f"classified:{self.category}"


class ContentApproved(PipelineStage):
    text: str = ""
    category: str = ""

    def stage_label(self) -> str:
        return "approved"


class AnalysisProduced(PipelineStage):
    summary: str = ""
    word_count: int = 0

    def stage_label(self) -> str:
        return f"analyzed:{self.word_count} words"


class ContentBlocked(Halted):
    category: str = ""


# ---------------------------------------------------------------------------
# Reducer — polymorphic stage tracking
# ---------------------------------------------------------------------------

UNSAFE_KEYWORDS = {"hack", "exploit", "attack", "malware", "phishing"}


stages = Reducer(
    "stages",
    event_type=PipelineStage,
    fn=lambda e: [e.stage_label()],
)


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------


@on(ContentReceived)
def classify(event: ContentReceived) -> ContentClassified:
    """Classify content by category and safety. Keyword-based rules."""
    words = set(event.text.lower().split())
    if words & UNSAFE_KEYWORDS:
        return ContentClassified(category="unsafe", safe=False)
    if any(w in words for w in ("python", "code", "function", "class")):
        return ContentClassified(category="technical")
    return ContentClassified(category="general")


@on(ContentClassified)
def gate(event: ContentClassified, log: EventLog) -> ContentBlocked | ContentApproved:
    """Safety gate — halts the pipeline for unsafe content.

    Uses ``log.latest(ContentReceived)`` to pass the original text forward.
    """
    if not event.safe:
        return ContentBlocked(category=event.category)
    original = log.latest(ContentReceived)
    return ContentApproved(text=original.text, category=event.category)


@on(ContentApproved)
def analyze(event: ContentApproved) -> AnalysisProduced:
    """Produce a simple analysis of approved content."""
    words = event.text.split()
    return AnalysisProduced(
        summary=f"{event.category} content ({len(words)} words)",
        word_count=len(words),
    )


# ---------------------------------------------------------------------------
# Main — three scenarios demonstrating different APIs
# ---------------------------------------------------------------------------


graph = EventGraph(
    [classify, gate, analyze],
    reducers=[stages],
)


async def main():
    # --- Scenario 1: Streaming with reducer snapshots ---
    print("=== Scenario 1: astream_events + StreamFrame ===")
    print("Input: safe technical content\n")

    async for frame in graph.astream_events(
        ContentReceived(text="A Python function that sorts a list"),
        include_reducers=True,
    ):
        assert isinstance(frame, StreamFrame)
        event_name = type(frame.event).__name__
        print(f"  {event_name:25s} stages={frame.reducers['stages']}")

    # --- Scenario 2: Halted with post-run inspection ---
    print("\n=== Scenario 2: Halted + EventLog.has() ===")
    print("Input: unsafe content\n")

    log = await graph.ainvoke(
        ContentReceived(text="How to hack a server exploit attack")
    )

    print(f"  Halted in log?   {log.has(Halted)}")
    print(f"  Analysis done?   {log.has(AnalysisProduced)}")
    blocked = log.latest(ContentBlocked)
    print(f"  Blocked category: {blocked.category}")

    # --- Scenario 3: Bare event stream (sync) ---
    print("\n=== Scenario 3: stream_events (sync, no reducers) ===")
    print("Input: safe general content\n")

    for event in graph.stream_events(
        ContentReceived(text="The weather today is sunny and warm")
    ):
        print(f"  {type(event).__name__}: {event}")


if __name__ == "__main__":
    asyncio.run(main())
