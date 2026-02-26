"""Content Pipeline — langgraph-events demo.

Demonstrates event streaming, halt-based early termination, and post-run
inspection in a content classification + analysis pipeline.

Covers APIs not shown in other examples:
  - ``Halt`` — immediate graph termination (unsafe content stops the pipeline)
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
    Event,
    EventGraph,
    EventLog,
    Halt,
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


# ---------------------------------------------------------------------------
# Reducer — polymorphic stage tracking
# ---------------------------------------------------------------------------

UNSAFE_KEYWORDS = {"hack", "exploit", "attack", "malware", "phishing"}


def to_pipeline_stage(event: Event) -> list[str]:
    """Map events to pipeline stage labels via polymorphic dispatch."""
    if isinstance(event, PipelineStage):
        return [event.stage_label()]
    return []


stages = Reducer("stages", fn=to_pipeline_stage)


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
def gate(event: ContentClassified, log: EventLog) -> Halt | ContentApproved:
    """Safety gate — halts the pipeline for unsafe content.

    Uses ``log.latest(ContentReceived)`` to pass the original text forward.
    """
    if not event.safe:
        return Halt(reason=f"blocked: {event.category} content")
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


async def main():
    graph = EventGraph(
        [classify, gate, analyze],
        reducers=[stages],
    )

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

    # --- Scenario 2: Halt with post-run inspection ---
    print("\n=== Scenario 2: Halt + EventLog.has() ===")
    print("Input: unsafe content\n")

    log = await graph.ainvoke(
        ContentReceived(text="How to hack a server exploit attack")
    )

    print(f"  Halt in log?     {log.has(Halt)}")
    print(f"  Analysis done?   {log.has(AnalysisProduced)}")
    halt = log.latest(Halt)
    print(f"  Halt reason:     {halt.reason}")

    # --- Scenario 3: Bare event stream (sync) ---
    print("\n=== Scenario 3: stream_events (sync, no reducers) ===")
    print("Input: safe general content\n")

    for event in graph.stream_events(
        ContentReceived(text="The weather today is sunny and warm")
    ):
        print(f"  {type(event).__name__}: {event}")


if __name__ == "__main__":
    asyncio.run(main())
