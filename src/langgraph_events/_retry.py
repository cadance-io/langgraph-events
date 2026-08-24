"""Declarative retry with exponential backoff for handler invocations.

``RetryPolicy`` is attached to a handler either as a class-level modifier on a
:class:`~langgraph_events._event.Command` (``retry = RetryPolicy(...)``) or via
``@on(..., retry=RetryPolicy(...))``.  The framework then re-invokes the handler
in place when it raises one of its declared ``raises=`` exceptions, waiting
``delay_for(attempt)`` seconds between tries.  Only the *final* failure surfaces
as ``HandlerRaised``, so catchers become pure escalation handlers.

Not to be confused with ``langgraph.types.RetryPolicy``, which is a *node*-level
policy that re-runs an entire LangGraph node.  This one is scoped to a single
handler call and composes with ``raises=``/``HandlerRaised``.
"""

from __future__ import annotations

import asyncio
import random
import time
from dataclasses import dataclass
from typing import Literal

_STRATEGIES = ("exponential", "constant")
_OBSERVE_MODES = ("emit", "log", "silent")

Strategy = Literal["exponential", "constant"]
Observe = Literal["emit", "log", "silent"]

# Indirection seams. Tests monkeypatch these to assert on the requested delays
# without ever waiting, and to make jitter deterministic without seeding a
# process-global RNG. Keep them module-level attributes, not imported names.
_random_uniform = random.uniform


def _sleep(delay: float) -> None:
    """Block the current thread — the sync dispatch path's backoff."""
    time.sleep(delay)


async def _asleep(delay: float) -> None:
    """Yield to the event loop — the async dispatch path's backoff."""
    await asyncio.sleep(delay)


@dataclass(frozen=True)
class RetryPolicy:
    """How many times to re-invoke a failing handler, and how long to wait.

    ``max_attempts`` is the *total* number of calls, not the number of extra
    ones: ``max_attempts=3`` means one initial call plus two retries.

    ``on`` narrows which exceptions retry.  It must be a subset of the
    handler's ``raises=``; the empty default means "every declared raise".
    An exception that is declared in ``raises=`` but excluded from ``on``
    surfaces as ``HandlerRaised`` on its first raise — that is how a
    non-transient error stays non-transient.

    Retried handlers are re-run from the top, so **they must be idempotent**:
    any side effect performed before the raise happens again on every attempt.
    """

    max_attempts: int = 3
    base_delay: float = 0.1
    max_delay: float = 30.0
    strategy: Strategy = "exponential"
    jitter: bool = True
    on: tuple[type[Exception], ...] = ()
    respect_retry_after: bool = False
    observe: Observe = "emit"

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ValueError(
                f"RetryPolicy max_attempts must be >= 1 (it counts the initial "
                f"call), got {self.max_attempts!r}."
            )
        if self.base_delay < 0:
            raise ValueError(
                f"RetryPolicy base_delay must be >= 0, got {self.base_delay!r}."
            )
        if self.max_delay < self.base_delay:
            raise ValueError(
                f"RetryPolicy max_delay ({self.max_delay!r}) must be >= "
                f"base_delay ({self.base_delay!r})."
            )
        if self.strategy not in _STRATEGIES:
            raise ValueError(
                f"RetryPolicy strategy must be one of {_STRATEGIES}, "
                f"got {self.strategy!r}."
            )
        if self.observe not in _OBSERVE_MODES:
            raise ValueError(
                f"RetryPolicy observe must be one of {_OBSERVE_MODES}, "
                f"got {self.observe!r}."
            )
        normalized = self.on if isinstance(self.on, tuple) else (self.on,)
        for entry in normalized:
            if not (isinstance(entry, type) and issubclass(entry, Exception)):
                raise TypeError(
                    f"RetryPolicy on= entries must be Exception subclasses, got "
                    f"{entry!r}. Non-Exception BaseException subclasses "
                    f"(KeyboardInterrupt, SystemExit, GeneratorExit, "
                    f"asyncio.CancelledError) are not allowed — they are "
                    f"runtime/exit signals, not domain errors."
                )
        object.__setattr__(self, "on", normalized)

    def delay_for(self, attempt: int, exc: Exception | None = None) -> float:
        """Seconds to wait before the attempt after *attempt*.

        *attempt* is 1-based and names the call that just **failed**, so the
        first backoff is ``delay_for(1)``.

        With ``respect_retry_after`` a server-supplied ``exc.retry_after`` wins
        over the computed curve and is used verbatim (capped by ``max_delay``,
        never jittered) — the upstream told us exactly how long to wait.
        """
        if self.respect_retry_after:
            hint = getattr(exc, "retry_after", None)
            if isinstance(hint, (int, float)) and not isinstance(hint, bool):
                return min(float(hint), self.max_delay)
        raw = (
            self.base_delay * 2 ** (attempt - 1)
            if self.strategy == "exponential"
            else self.base_delay
        )
        ceiling = min(raw, self.max_delay)
        return _random_uniform(0.0, ceiling) if self.jitter else ceiling

    def retries(self, exc: Exception, declared: tuple[type[Exception], ...]) -> bool:
        """Whether *exc* is in scope for this policy.

        *declared* is the handler's ``raises=`` tuple, used when ``on`` is empty.
        """
        return isinstance(exc, self.on or declared)
