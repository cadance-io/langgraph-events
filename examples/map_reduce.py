"""Map-Reduce Pipeline — langgraph-events demo.

Demonstrates fan-out/fan-in using ``Scatter`` and ``EventLog.filter()`` inside
a DDD ``Batch`` aggregate. Replaces LangGraph's verbose ``Send`` API and
worker subgraph patterns with a simple command-plus-scatter flow.

The **Auditable trait** auto-logs every event as it flows, replacing manual
isinstance printing with a single side-effect handler.

Usage:
    export OPENAI_API_KEY="sk-..."
    python examples/map_reduce.py
"""

from __future__ import annotations

import asyncio

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from langgraph_events import (
    Aggregate,
    Auditable,
    Command,
    DomainEvent,
    EventGraph,
    EventLog,
    Scatter,
    on,
)

# ---------------------------------------------------------------------------
# Aggregate: Batch
# ---------------------------------------------------------------------------


class Batch(Aggregate):
    """A batch of documents to summarize via map-reduce.

    ``Summarize`` is the entry command. A fan-out handler returns
    ``Scatter[Batch.DocDispatched]``; per-doc handlers emit ``DocSummarized``;
    a gather handler collects them and produces ``Summarize.Summarized`` when
    all documents have been processed.
    """

    class Summarize(Command, Auditable):
        """Summarize this batch of documents."""

        documents: tuple = ()  # tuple of (title, content) pairs

        class Summarized(DomainEvent, Auditable):
            """Final outcome — combined summary of the whole batch."""

            combined: str = ""
            individual: tuple = ()  # tuple of (title, summary) pairs

    class DocDispatched(DomainEvent, Auditable):
        """Scatter target — one document dispatched for summarization."""

        title: str = ""
        content: str = ""
        batch_size: int = 0

    class DocSummarized(DomainEvent, Auditable):
        """Per-document summary — gathered into the final outcome."""

        title: str = ""
        summary: str = ""


# ---------------------------------------------------------------------------
# Sample documents
# ---------------------------------------------------------------------------

DOCUMENTS = (
    (
        "The History of Python",
        "Python was conceived in the late 1980s by Guido van Rossum at Centrum "
        "Wiskunde & Informatica (CWI) in the Netherlands as a successor to the ABC "
        "programming language. Its implementation began in December 1989, and the "
        "first version (0.9.0) was released in February 1991. Python 2.0 was released "
        "in 2000, introducing features like list comprehensions and garbage collection. "
        "Python 3.0, released in 2008, was a major revision not fully backward-compatible "
        "with Python 2. Today Python is one of the most popular programming languages "
        "worldwide, used extensively in web development, data science, AI, and automation.",
    ),
    (
        "Introduction to Machine Learning",
        "Machine learning is a subset of artificial intelligence that focuses on building "
        "systems that learn from data. Rather than being explicitly programmed, these "
        "systems identify patterns in training data and use them to make predictions or "
        "decisions. The three main types are supervised learning (labeled data), "
        "unsupervised learning (unlabeled data), and reinforcement learning (reward-based). "
        "Key algorithms include linear regression, decision trees, neural networks, and "
        "support vector machines. Machine learning powers applications from spam filters "
        "and recommendation engines to self-driving cars and medical diagnosis.",
    ),
    (
        "Cloud Computing Fundamentals",
        "Cloud computing delivers computing services — servers, storage, databases, "
        "networking, software — over the internet. The three main service models are "
        "Infrastructure as a Service (IaaS), Platform as a Service (PaaS), and Software "
        "as a Service (SaaS). Major providers include AWS, Microsoft Azure, and Google "
        "Cloud Platform. Benefits include scalability, cost-efficiency (pay-as-you-go), "
        "and global availability. Cloud-native architectures leverage containers, "
        "microservices, and serverless functions to build resilient, scalable applications. "
        "Security and compliance remain key considerations for enterprise adoption.",
    ),
)

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


@on(Batch.Summarize)
def split_batch(event: Batch.Summarize) -> Scatter[Batch.DocDispatched]:
    """Fan out: split the batch into individual summarization tasks."""
    return Scatter(
        [
            Batch.DocDispatched(
                title=title, content=content, batch_size=len(event.documents)
            )
            for title, content in event.documents
        ]
    )


@on(Batch.DocDispatched)
async def summarize_one(event: Batch.DocDispatched) -> Batch.DocSummarized:
    """Map: summarize a single document using the LLM."""
    messages = [
        SystemMessage(content="Summarize the following text in 2-3 sentences."),
        HumanMessage(content=f"Title: {event.title}\n\n{event.content}"),
    ]
    response = await llm.ainvoke(messages)
    return Batch.DocSummarized(title=event.title, summary=response.content)


@on(Batch.DocSummarized)
def gather_summaries(
    event: Batch.DocSummarized, log: EventLog
) -> Batch.Summarize.Summarized | None:
    """Reduce: collect all summaries once all documents are processed.

    This handler fires once per ``DocSummarized`` in the pending batch. Since
    all ``DocSummarized`` events arrive in the same dispatch round, the
    handler runs N times. We check completion via ``EventLog.filter()`` and
    produce the combined result — duplicates are harmless, ``log.latest()``
    picks the final one.
    """
    all_summaries = log.filter(Batch.DocSummarized)
    batch = log.latest(Batch.Summarize)

    if len(all_summaries) < len(batch.documents):
        return None

    individual = tuple((s.title, s.summary) for s in all_summaries)
    combined = "\n\n".join(f"**{title}**: {summary}" for title, summary in individual)
    return Batch.Summarize.Summarized(combined=combined, individual=individual)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


graph = EventGraph([split_batch, summarize_one, gather_summaries, audit_trail])


async def main() -> None:
    print(f"Summarizing {len(DOCUMENTS)} documents...\n")
    for title, _ in DOCUMENTS:
        print(f"  - {title}")
    print()
    print("--- Event Flow ---")

    log = await graph.ainvoke(Batch.Summarize(documents=DOCUMENTS))

    print("\n--- Combined Summary ---")
    result = log.latest(Batch.Summarize.Summarized)
    print(result.combined)


if __name__ == "__main__":
    asyncio.run(main())
