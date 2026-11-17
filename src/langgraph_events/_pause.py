"""The pause concept: when a run must pause, and why.

The caller passes ``deadline=``, ``pause_on=``, or both. The run's
:class:`~langgraph_events._run_signals._RunSignals` writes one
:class:`_PauseBudget` into the run's ``configurable``. The router and the
retry loop read it from there.
"""

import dataclasses
import time
from typing import TYPE_CHECKING, NamedTuple

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

    from langgraph_events._run_signals import SignalLatch

# Every ``configurable`` key the framework writes starts with this prefix.
# The AG-UI adapter drops a client key with it, so a client cannot plant one.
# The checkpoint does not store ``configurable``, and tracing metadata skips
# a key with a ``__`` prefix, so a signal object is never serialized.
_FRAMEWORK_KEY_PREFIX = "__lge_"
_PAUSE_BUDGET_KEY = f"{_FRAMEWORK_KEY_PREFIX}pause_budget"


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
    :class:`~langgraph_events._run_signals.SignalLatch` is a mutable flag of
    the run.
    """

    started_at: float
    deadline: float | None
    signal: SignalLatch | None

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


def _pause_budget(config: RunnableConfig | None) -> _PauseBudget:
    """The run's :class:`_PauseBudget`, or :data:`_NO_PAUSE` when it has none."""
    return (config or {}).get("configurable", {}).get(_PAUSE_BUDGET_KEY, _NO_PAUSE)
