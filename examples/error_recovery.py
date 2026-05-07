"""Error Recovery — langgraph-events demo.

Demonstrates declared handler exceptions via class-level ``raises`` + the
built-in ``HandlerRaised`` event, organized around a DDD ``Question`` domain.

A Command declares which exceptions the framework should catch; when one of
those exceptions fires, the framework surfaces it as a ``HandlerRaised`` event
so other handlers can react (retry, back off, halt) without try/except
boilerplate at the raise site.

Covers APIs not shown in other examples:
  - ``raises = (...,)`` as a class-level attribute on a ``Command``
  - ``HandlerRaised`` — built-in event wrapping a caught exception
  - Field injection of the exception (``exception: RateLimitError``)
  - Chained error handling: a catcher itself declares ``raises=`` to escalate
  - Retry loop expressed as the catcher emitting a fresh ``Ask`` command —
    Cmd-private outcomes (``Ask.Answered``) only ever come from
    ``Ask.handle()``.

Usage:
    python examples/error_recovery.py
"""

from __future__ import annotations

import warnings

from langgraph_events import (
    Auditable,
    Command,
    DomainEvent,
    EventGraph,
    EventLog,
    Halted,
    HandlerRaised,
    Namespace,
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
# Namespace: Question
# ---------------------------------------------------------------------------


MAX_ATTEMPTS = 3
_scenario = {"succeed_after": 3}  # attempt # at which the call starts succeeding


class Question(Namespace):
    """A user question answered via a rate-limit-tolerant LLM call.

    ``Ask`` is the entry command. Its inline ``handle`` declares
    ``raises = (RateLimitError,)`` as a class attribute; a catcher schedules
    a retry by emitting a fresh ``Ask`` command (with an incremented
    ``attempt``), which goes back through ``Ask.handle()``. After
    ``MAX_ATTEMPTS``, the catcher escalates to ``QuotaExhaustedError`` which
    a second catcher turns into the terminal ``GaveUp`` ``Halted`` signal.
    """

    class Ask(Command, Auditable):
        """Entry command — asks a question, may take several tries to answer."""

        question: str = ""
        attempt: int = 1

        raises = (RateLimitError,)

        class Answered(DomainEvent, Auditable):
            """Question answered — terminal outcome of Ask."""

            answer: str = ""

        def handle(self) -> Question.Ask.Answered:
            if self.attempt < _scenario["succeed_after"]:
                raise RateLimitError(retry_after=0.1 * self.attempt)
            return Question.Ask.Answered(answer=f"Answer to: {self.question!r}")

    class GaveUp(Halted):
        """Terminal halt — retry budget exhausted."""

        reason: str = ""


# ---------------------------------------------------------------------------
# Handlers — recovery only; the success path lives on Ask.handle().
# ---------------------------------------------------------------------------


@on(HandlerRaised, exception=RateLimitError, raises=QuotaExhaustedError)
def backoff_and_retry(
    event: HandlerRaised,
    exception: RateLimitError,
) -> Question.Ask:
    """Catches rate-limit failures and re-issues ``Ask`` with attempt+1.

    Re-emitting the Command is the canonical retry pattern under the strict
    Command-privacy rule: only ``Ask.handle()`` may produce ``Ask.Answered``,
    so the recovery reactor schedules a fresh ``Ask`` rather than emitting an
    answer itself.
    """
    original = event.source_event
    assert isinstance(original, Question.Ask)
    prev_attempt = original.attempt
    next_attempt = prev_attempt + 1
    if next_attempt > MAX_ATTEMPTS:
        raise QuotaExhaustedError(f"Exceeded {MAX_ATTEMPTS} attempts")
    print(
        f"  [backoff] attempt {prev_attempt} hit rate limit "
        f"(retry_after={exception.retry_after}s); retrying as attempt {next_attempt}"
    )
    return Question.Ask(question=original.question, attempt=next_attempt)


@on(HandlerRaised, exception=QuotaExhaustedError)
def give_up(event: HandlerRaised) -> Question.GaveUp:
    """Escalation catcher — the retry loop itself surrendered."""
    return Question.GaveUp(reason=str(event.exception))


# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------


graph = EventGraph([Question.Ask, backoff_and_retry, give_up])


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------


def main() -> None:
    print("== Successful recovery after transient rate limits ==")
    _scenario["succeed_after"] = 3
    log = graph.invoke(Question.Ask(question="What is langgraph-events?"))
    _print_trail(log)
    answer = log.latest(Question.Ask.Answered)
    print(f"  [result] {answer}\n")

    print("== Escalation when retries are exhausted ==")
    _scenario["succeed_after"] = 999  # unreachable — always rate-limits
    log = graph.invoke(Question.Ask(question="Will this ever succeed?"))
    _print_trail(log)
    halted = log.latest(Question.GaveUp)
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
