"""Content Pipeline — langgraph-events demo.

Demonstrates event streaming, halt-based early termination, and post-run
inspection in a DDD ``Content`` aggregate.

Covers APIs not shown in other examples:
  - ``Halted`` — immediate graph termination (unsafe content stops the pipeline)
  - ``StreamFrame`` — NamedTuple yielded by ``astream_events()`` with reducer snapshots
  - ``stream_events()`` / ``astream_events()`` — high-level event streaming
  - ``EventLog.has()`` — boolean existence check
  - ``Reducer`` — polymorphic stage tracking via a ``Protocol``

No LLM dependency — runs standalone with keyword-based classification rules.

Usage:
    python examples/content_pipeline.py
"""

from __future__ import annotations

import asyncio
from typing import Protocol, runtime_checkable

from langgraph_events import (
    Aggregate,
    Auditable,
    Command,
    DomainEvent,
    EventGraph,
    EventLog,
    Halted,
    Reducer,
    on,
)
from langgraph_events.stream import StreamFrame

# ---------------------------------------------------------------------------
# Aggregate: Content
# ---------------------------------------------------------------------------

UNSAFE_KEYWORDS = {"hack", "exploit", "attack", "malware", "phishing"}


@runtime_checkable
class StageLabelled(Protocol):
    """Events that contribute a stage label to the pipeline tracker."""

    def stage_label(self) -> str: ...


class Content(Aggregate):
    """Content flowing through safety classification and analysis.

    ``Process`` is the entry command. Its inline ``handle`` classifies the
    text (``Classified`` outcome). Downstream reactions approve or block,
    then analyze — those facts are free-standing under the aggregate.
    """

    class Process(Command, Auditable):
        """Process a piece of content through the pipeline."""

        text: str = ""

        class Classified(DomainEvent, Auditable):
            """Classification outcome — category + safety flag."""

            category: str = ""
            safe: bool = True

            def stage_label(self) -> str:
                return f"classified:{self.category}"

        def handle(self) -> Content.Process.Classified:
            """Inline keyword classifier — trivial rules, colocated with the command."""
            words = set(self.text.lower().split())
            if words & UNSAFE_KEYWORDS:
                return Content.Process.Classified(category="unsafe", safe=False)
            if any(w in words for w in ("python", "code", "function", "class")):
                return Content.Process.Classified(category="technical")
            return Content.Process.Classified(category="general")

    class Approved(DomainEvent, Auditable):
        """Content cleared the safety gate — free-standing under Content."""

        text: str = ""
        category: str = ""

        def stage_label(self) -> str:
            return "approved"

    class Analyzed(DomainEvent, Auditable):
        """Analysis produced — terminal free-standing fact."""

        summary: str = ""
        word_count: int = 0

        def stage_label(self) -> str:
            return f"analyzed:{self.word_count} words"

    class Blocked(Halted):
        """Halted: content failed the safety gate.

        ``Halted`` subclass stays a ``SystemEvent`` (framework termination);
        nested inside the aggregate class for locality only.
        """

        category: str = ""


# ---------------------------------------------------------------------------
# Reducer — polymorphic stage tracking via Protocol
# ---------------------------------------------------------------------------


stages = Reducer(
    "stages",
    event_type=StageLabelled,
    fn=lambda e: [e.stage_label()],
)


# ---------------------------------------------------------------------------
# Handlers — ``Process`` is inline; gate and analyze are external reactions
# ---------------------------------------------------------------------------


@on(Content.Process.Classified)
def gate(
    event: Content.Process.Classified, log: EventLog
) -> Content.Blocked | Content.Approved:
    """Safety gate — halts the pipeline for unsafe content.

    Uses ``log.latest(Content.Process)`` to pass the original text forward.
    """
    if not event.safe:
        return Content.Blocked(category=event.category)
    original = log.latest(Content.Process)
    return Content.Approved(text=original.text, category=event.category)


@on(Content.Approved)
def analyze(event: Content.Approved) -> Content.Analyzed:
    """Produce a simple analysis of approved content."""
    words = event.text.split()
    return Content.Analyzed(
        summary=f"{event.category} content ({len(words)} words)",
        word_count=len(words),
    )


# ---------------------------------------------------------------------------
# Main — three scenarios demonstrating different APIs
# ---------------------------------------------------------------------------


graph = EventGraph.from_aggregates(
    Content,
    handlers=[gate, analyze],
    reducers=[stages],
)


async def main() -> None:
    # --- Scenario 1: Streaming with reducer snapshots ---
    print("=== Scenario 1: astream_events + StreamFrame ===")
    print("Input: safe technical content\n")

    async for frame in graph.astream_events(
        Content.Process(text="A Python function that sorts a list"),
        include_reducers=True,
    ):
        assert isinstance(frame, StreamFrame)
        event_name = type(frame.event).__name__
        print(f"  {event_name:25s} stages={frame.reducers['stages']}")

    # --- Scenario 2: Halted with post-run inspection ---
    print("\n=== Scenario 2: Halted + EventLog.has() ===")
    print("Input: unsafe content\n")

    log = await graph.ainvoke(
        Content.Process(text="How to hack a server exploit attack")
    )

    print(f"  Halted in log?   {log.has(Halted)}")
    print(f"  Analysis done?   {log.has(Content.Analyzed)}")
    blocked = log.latest(Content.Blocked)
    print(f"  Blocked category: {blocked.category}")

    # --- Scenario 3: Bare event stream (sync) ---
    print("\n=== Scenario 3: stream_events (sync, no reducers) ===")
    print("Input: safe general content\n")

    for event in graph.stream_events(
        Content.Process(text="The weather today is sunny and warm")
    ):
        print(f"  {type(event).__name__}: {event}")


if __name__ == "__main__":
    asyncio.run(main())
