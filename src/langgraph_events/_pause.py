"""The pause concept: when a run must pause, and why.

The caller passes ``deadline=``, ``pause_on=``, or both. Each entry point
writes one :class:`_PauseBudget` into the run's ``configurable``. The router
and the retry loop read it from there.
"""

from __future__ import annotations

import dataclasses
import time
from typing import TYPE_CHECKING, Any, NamedTuple, cast

from langgraph_events._event import (
    HandlerAbandoned,
    HandlerAbandonedOnDeadline,
    HandlerAbandonedOnSignal,
    RunPaused,
    RunPausedOnDeadline,
    RunPausedOnSignal,
)

if TYPE_CHECKING:
    from langchain_core.runnables import RunnableConfig

    from langgraph_events._types import PauseSignal

# Every ``configurable`` key the framework writes starts with this prefix.
# The AG-UI adapter drops a client key with it, so a client cannot plant one.
# The checkpoint does not store ``configurable``, and tracing metadata skips
# a key with a ``__`` prefix, so the pause signal object is never serialized.
_FRAMEWORK_KEY_PREFIX = "__lge_"
_PAUSE_BUDGET_KEY = f"{_FRAMEWORK_KEY_PREFIX}pause_budget"


class _SignalLatch:
    """The run's ``pause_on=`` signal, held set from its first set read.

    The retry loop and the router both read the signal. When a caller clears
    it between their reads, the retry loop has already given up for the
    pause, so the router must still pause. The first ``True`` from
    ``is_set()`` therefore commits the run to a pause.
    """

    __slots__ = ("_latched", "_signal")

    def __init__(self, signal: PauseSignal) -> None:
        self._signal = signal
        self._latched = False

    def is_set(self) -> bool:
        if not self._latched:
            self._latched = self._signal.is_set()
        return self._latched


class _PauseCause(NamedTuple):
    """One reason to pause, as the event that each reader emits for it."""

    paused: type[RunPaused]
    abandoned: type[HandlerAbandoned]


_ON_SIGNAL = _PauseCause(RunPausedOnSignal, HandlerAbandonedOnSignal)
_ON_DEADLINE = _PauseCause(RunPausedOnDeadline, HandlerAbandonedOnDeadline)


@dataclasses.dataclass(frozen=True)
class _PauseBudget:
    """When a run must pause: its ``deadline=``, its ``pause_on=``, or both.

    One per run, in the run's ``configurable``. The router asks it for the
    pause event between rounds. The retry loop asks it whether to refuse a
    backoff wait. Both answers come from :meth:`_cause_at`, so the two
    readers cannot disagree on the cause. The fields are frozen. The
    :class:`_SignalLatch` is the run's one mutable flag.
    """

    started_at: float
    deadline: float | None
    signal: _SignalLatch | None

    def _cause_at(self, instant: float) -> _PauseCause | None:
        """Why the run must pause at *instant*, or ``None`` to continue.

        A set signal wins over an expired deadline: the caller asked for it.
        """
        if self.signal is not None and self.signal.is_set():
            return _ON_SIGNAL
        if self.deadline is not None and instant >= self.deadline:
            return _ON_DEADLINE
        return None

    def pause_now(self) -> RunPaused | None:
        """The pause event the router must emit now, or ``None`` to continue."""
        now = time.monotonic()
        cause = self._cause_at(now)
        if cause is None:
            return None
        return cause.paused(elapsed_seconds=now - self.started_at)

    def refuse_wait(self, delay: float) -> type[HandlerAbandoned] | None:
        """The give-up the retry loop emits instead of a *delay* wait, or ``None``.

        The wait is refused when the run must pause at its end: the router
        pauses only between rounds, so the wait only delays the pause.
        """
        cause = self._cause_at(time.monotonic() + delay)
        return None if cause is None else cause.abandoned


_NO_PAUSE = _PauseBudget(started_at=0.0, deadline=None, signal=None)
"""The budget of a run with neither ``deadline=`` nor ``pause_on=``."""


def _with_pause_budget(
    config: RunnableConfig | None,
    deadline: float | None,
    pause_on: PauseSignal | None,
) -> RunnableConfig:
    """Return *config* with the run's :class:`_PauseBudget` in ``configurable``.

    Single writer for every entry point that accepts ``deadline=`` or
    ``pause_on=``. With neither set, *config* comes back unchanged, or as
    ``{}`` for ``None``. Otherwise the result is a copy, so the caller's dict
    is never mutated.

    Raises ``TypeError`` when *pause_on* is a class, or has no callable
    ``is_set``, before any round runs. ``pause_on=event.is_set`` and
    ``pause_on=threading.Event`` are the usual mistakes.
    """
    if deadline is None and pause_on is None:
        return config or {}
    if pause_on is not None and (
        isinstance(pause_on, type) or not callable(getattr(pause_on, "is_set", None))
    ):
        raise TypeError(
            f"pause_on= must be an object with an is_set() method, for "
            f"example an asyncio.Event(), got {type(pause_on).__name__}: "
            f"{pause_on!r}. Pass the event itself, not event.is_set or the "
            f"Event class."
        )
    latch = None if pause_on is None else _SignalLatch(pause_on)
    budget = _PauseBudget(time.monotonic(), deadline, latch)
    out: dict[str, Any] = dict(config or {})
    configurable: dict[str, Any] = out.get("configurable", {})
    out["configurable"] = {**configurable, _PAUSE_BUDGET_KEY: budget}
    return cast("RunnableConfig", out)


def _pause_budget(config: RunnableConfig | None) -> _PauseBudget:
    """The run's :class:`_PauseBudget`, or :data:`_NO_PAUSE` when it has none."""
    return (config or {}).get("configurable", {}).get(_PAUSE_BUDGET_KEY, _NO_PAUSE)
