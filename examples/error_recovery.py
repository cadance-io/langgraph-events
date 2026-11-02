"""Error Recovery — langgraph-events demo.

Demonstrates declared handler exceptions via class-level ``raises=`` plus a
declarative ``retry = RetryPolicy(...)``, organized around a ``Question``
namespace.

A Command declares which exceptions the library catches. A ``RetryPolicy`` on
the same Command tells the library to re-invoke the handler in place, with a
full-jitter exponential backoff between tries. Only when the retry budget is
spent does the exception surface as ``HandlerRaised``. The handler that
receives it never counts attempts and never schedules a retry.

Covers:
  - ``retry = RetryPolicy(...)`` — declarative backoff, enforced by the library
  - ``HandlerRetried`` — built-in event emitted before each backoff wait
  - ``raises = (...,)`` as a class-level attribute on a ``Command``
  - ``HandlerRaised`` — built-in event wrapping a caught exception
  - ``@on(HandlerRaised, exception=...)`` as a field matcher narrowing a
    handler to one exception type

Usage:
    python examples/error_recovery.py
"""

from __future__ import annotations

from langgraph_events import (
    Command,
    DomainEvent,
    EventGraph,
    EventLog,
    Halted,
    HandlerRaised,
    HandlerRetried,
    Namespace,
    RetryPolicy,
    on,
)

# ---------------------------------------------------------------------------
# Exceptions (not Events: they stay plain Python exceptions)
# ---------------------------------------------------------------------------


class RateLimitError(Exception):
    """Simulated upstream rate limit."""

    def __init__(self, retry_after: float) -> None:
        super().__init__(f"rate limited, retry after {retry_after}s")
        self.retry_after = retry_after


# ---------------------------------------------------------------------------
# Namespace: Question
# ---------------------------------------------------------------------------


MAX_ATTEMPTS = 3
# Attempt # at which the simulated upstream starts succeeding, plus the call
# counter the retried handler increments. A retried handler is re-run from the
# top, so this stands in for the non-idempotent upstream it is protecting.
_scenario = {"succeed_after": 3, "calls": 0}


class Question(Namespace):
    """A user question answered by a simulated upstream that rate-limits early calls.

    ``Ask`` is the entry command. Its inline ``handle`` declares
    ``raises = (RateLimitError,)`` and a ``retry`` policy as class attributes.
    The library absorbs up to ``MAX_ATTEMPTS - 1`` rate limits on its own. Only
    once the budget is spent does ``HandlerRaised`` reach ``give_up``, which
    turns it into the terminal ``RetriesExhausted`` ``Halted`` signal.
    """

    class Ask(Command):
        """Entry command. Asks a question. Can take several tries to answer."""

        question: str = ""

        raises = (RateLimitError,)
        # Full jitter: the wait before retry n is uniform in
        # [0, min(base_delay * 2**(n-1), max_delay)]. The cap applies to the
        # ceiling, before sampling. Set ``respect_retry_after=True`` to prefer
        # a server-supplied ``exception.retry_after`` over the computed curve.
        retry = RetryPolicy(max_attempts=MAX_ATTEMPTS, base_delay=0.05, max_delay=1.0)

        class Answered(DomainEvent):
            """Question answered. Terminal outcome of Ask."""

            answer: str = ""

        def handle(self) -> Question.Ask.Answered:
            _scenario["calls"] += 1
            if _scenario["calls"] < _scenario["succeed_after"]:
                raise RateLimitError(retry_after=round(0.1 * _scenario["calls"], 2))
            return Question.Ask.Answered(answer=f"Answer to: {self.question!r}")

    class RetriesExhausted(Halted):
        """Terminal halt. The retry budget is spent."""

        reason: str = ""


# ---------------------------------------------------------------------------
# Handlers. Escalation only: the success path lives on Ask.handle(), and the
# library runs the retries in between.
# ---------------------------------------------------------------------------


@on(HandlerRaised, exception=RateLimitError)
def give_up(event: HandlerRaised) -> Question.RetriesExhausted:
    """Escalation handler. Reached only after the retry budget is spent."""
    return Question.RetriesExhausted(
        reason=f"Exceeded {MAX_ATTEMPTS} attempts: {event.exception}"
    )


# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------


graph = EventGraph([Question.Ask, give_up])


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------


def main() -> None:
    print("== Successful recovery after transient rate limits ==")
    _scenario.update(succeed_after=3, calls=0)
    log = graph.invoke(Question.Ask(question="What is langgraph-events?"))
    _print_trail(log)
    answer = log.latest(Question.Ask.Answered)
    print(f"  [result] {answer}\n")

    print("== Escalation when retries are exhausted ==")
    _scenario.update(succeed_after=999, calls=0)  # unreachable: always rate-limits
    log = graph.invoke(Question.Ask(question="Will this ever succeed?"))
    _print_trail(log)
    halted = log.latest(Question.RetriesExhausted)
    print(f"  [result] {halted}")


def _print_trail(log: EventLog) -> None:
    for ev in log:
        if isinstance(ev, HandlerRetried):
            print(
                f"  [HandlerRetried] attempt {ev.attempt} raised "
                f"{type(ev.exception).__name__}; backing off "
                f"{ev.delay_seconds:.3f}s"
            )
        elif isinstance(ev, HandlerRaised):
            exc_name = type(ev.exception).__name__
            print(f"  [HandlerRaised] handler={ev.handler} exception={exc_name}")
        elif isinstance(ev, Halted):
            print(f"  [{type(ev).__name__}] {ev}")
        else:
            print(f"  {ev!r}")


if __name__ == "__main__":
    main()
