"""Error Recovery — langgraph-events demo.

Demonstrates declared handler exceptions via ``raises=`` + the built-in
``HandlerRaised`` event.

A handler declares which exceptions the framework should catch; when one of
those exceptions fires, the framework surfaces it as a ``HandlerRaised`` event
so other handlers can react (retry, back off, halt) without try/except
boilerplate at the raise site.

Covers APIs not shown in other examples:
  - ``raises=`` on ``@on(...)`` — declare catchable exceptions
  - ``HandlerRaised`` — built-in event wrapping a caught exception
  - Field injection of the exception (``exception: RateLimitError``)
  - Chained error handling: the catcher itself declares ``raises=`` to escalate

Usage:
    python examples/error_recovery.py
"""

from __future__ import annotations

import warnings

from langgraph_events import (
    Auditable,
    EventGraph,
    EventLog,
    Halted,
    HandlerRaised,
    OrphanedEventWarning,
    on,
)

warnings.filterwarnings("ignore", category=OrphanedEventWarning)

# ---------------------------------------------------------------------------
# Exceptions (NOT Events — they stay plain Python exceptions)
# ---------------------------------------------------------------------------


class RateLimitError(Exception):
    """Simulated upstream rate limit."""

    def __init__(self, retry_after: float) -> None:
        super().__init__(f"rate limited, retry after {retry_after}s")
        self.retry_after = retry_after


class QuotaExhaustedError(Exception):
    """Escalation after too many retries — no more budget."""


# ---------------------------------------------------------------------------
# Events (past-participle: "what just happened")
# ---------------------------------------------------------------------------


class QuestionAsked(Auditable):
    """Seed event — the user's question enters the graph here."""

    question: str = ""


class RetryScheduled(Auditable):
    """Internal retry event — emitted by the recovery handler."""

    question: str = ""
    attempt: int = 2


class AnswerReceived(Auditable):
    answer: str = ""


class GaveUp(Halted):
    reason: str = ""


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------

MAX_ATTEMPTS = 3
_scenario = {"succeed_after": 3}  # attempt # at which the call starts succeeding


@on(QuestionAsked, RetryScheduled, raises=RateLimitError)
def call_llm(event: QuestionAsked | RetryScheduled) -> AnswerReceived:
    """Simulated LLM call — rate-limits until the scenario's success threshold."""
    attempt = event.attempt if isinstance(event, RetryScheduled) else 1
    if attempt < _scenario["succeed_after"]:
        raise RateLimitError(retry_after=0.1 * attempt)
    return AnswerReceived(answer=f"Answer to: {event.question!r}")


@on(HandlerRaised, exception=RateLimitError, raises=QuotaExhaustedError)
def backoff_and_retry(
    event: HandlerRaised,
    exception: RateLimitError,
) -> RetryScheduled:
    """Catches rate-limit failures and schedules a retry of the original question."""
    original = event.source_event
    assert isinstance(original, (QuestionAsked, RetryScheduled))
    prev_attempt = original.attempt if isinstance(original, RetryScheduled) else 1
    next_attempt = prev_attempt + 1
    if next_attempt > MAX_ATTEMPTS:
        raise QuotaExhaustedError(f"Exceeded {MAX_ATTEMPTS} attempts")
    print(
        f"  [backoff] attempt {prev_attempt} hit rate limit "
        f"(retry_after={exception.retry_after}s); retrying as attempt {next_attempt}"
    )
    return RetryScheduled(question=original.question, attempt=next_attempt)


@on(HandlerRaised, exception=QuotaExhaustedError)
def give_up(event: HandlerRaised) -> GaveUp:
    """Escalation catcher — the retry loop itself surrendered."""
    return GaveUp(reason=str(event.exception))


# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------


graph = EventGraph([call_llm, backoff_and_retry, give_up])


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------


def main() -> None:
    print("== Successful recovery after transient rate limits ==")
    _scenario["succeed_after"] = 3
    log = graph.invoke(QuestionAsked(question="What is langgraph-events?"))
    _print_trail(log)
    answer = log.latest(AnswerReceived)
    print(f"  [result] {answer}\n")

    print("== Escalation when retries are exhausted ==")
    _scenario["succeed_after"] = 999  # unreachable — always rate-limits
    log = graph.invoke(QuestionAsked(question="Will this ever succeed?"))
    _print_trail(log)
    halted = log.latest(GaveUp)
    print(f"  [result] {halted}")


def _print_trail(log: EventLog) -> None:
    for ev in log:
        if isinstance(ev, Auditable):
            print(f"  {ev.trail()}")
        elif isinstance(ev, HandlerRaised):
            exc_name = type(ev.exception).__name__
            print(f"  [HandlerRaised] handler={ev.handler} exception={exc_name}")
        elif isinstance(ev, Halted):
            print(f"  [{type(ev).__name__}] {ev}")


if __name__ == "__main__":
    main()
