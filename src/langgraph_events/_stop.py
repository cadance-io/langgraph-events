"""The stop concept: cancel the running handler when ``stop_on=`` is set."""

import asyncio
import logging
from typing import TYPE_CHECKING, Any

from langgraph.errors import GraphInterrupt
from langgraph.types import Command as LGCommand

from langgraph_events._causes import FRAMEWORK, pad_causes
from langgraph_events._event import (
    CancelledOnSignal,
    Event,
    Halted,
    HandlerCutOnSignal,
)
from langgraph_events._run_signals import stop_signal

if TYPE_CHECKING:
    from collections.abc import Coroutine

    from langchain_core.runnables import RunnableConfig
    from langgraph.types import StateSnapshot

STOPPED_HANDLERS = "_stopped_handlers"
"""The state channel of the handlers whose step the stop cut."""

_logger = logging.getLogger(__name__)

_STOP_TICK_SECONDS = 0.05
"""Cost: each running async handler wakes the event loop once per tick."""

_STOP_GRACE_SECONDS = 1.0
"""How long a handler can clean up after the first cancel.

A usual cleanup closes a connection or writes one record, well inside one
second. A lost cancel then delays the stop by at most this time.
"""


async def _cancel_until_done(task: asyncio.Task[None]) -> None:
    """Cancel *task* until it ends.

    *task* has :data:`_STOP_GRACE_SECONDS` to clean up after the first
    cancel. A layer under a handler can lose a cancel, so the cancel then
    repeats at each tick. A cancel of the caller sends the cancel again at
    once, and is raised when *task* has ended.
    """
    caller_cancelled = False
    timeout = _STOP_GRACE_SECONDS
    while not task.done():
        task.cancel()
        try:
            await asyncio.wait({task}, timeout=timeout)
        except asyncio.CancelledError:
            caller_cancelled = True
            continue
        timeout = _STOP_TICK_SECONDS
    _log_exception(task)
    if caller_cancelled:
        raise asyncio.CancelledError


def _log_exception(task: asyncio.Task[None]) -> None:
    """Log the exception of a discarded *task*, which no caller reads.

    An interrupt after the stop is no failure: the stop discards it.
    """
    if task.cancelled():
        return
    exc = task.exception()
    if exc is not None and not isinstance(exc, GraphInterrupt):
        _logger.warning("An async handler raised after its cancel.", exc_info=exc)


def stop_is_set(config: RunnableConfig | None) -> bool:
    stop = stop_signal(config)
    return stop is not None and stop.is_set()


async def run_step(work: Coroutine[Any, Any, None], config: RunnableConfig) -> bool:
    """Run *work*, and cancel it when the run's ``stop_on=`` is set.

    The cancel goes to *work* only, so it never reaches a checkpoint write.
    Returns whether the stop cut *work*.
    """
    stop = stop_signal(config)
    if stop is None:
        await work
        return False
    task = asyncio.ensure_future(work)
    try:
        while not task.done() and not stop.is_set():
            await asyncio.wait({task}, timeout=_STOP_TICK_SECONDS)
    except asyncio.CancelledError:
        await _cancel_until_done(task)
        raise
    if stop.is_set():
        await _cancel_until_done(task)
        return True
    task.result()
    return False


def add_or_clear(left: list[str], right: list[str] | None) -> list[str]:
    return [] if right is None else left + right


def _cut_records(snapshot: StateSnapshot | None) -> list[Event]:
    """A :class:`HandlerCutOnSignal` for each cut sibling of a paused round.

    The interrupt of a sibling handler ended that round, so its router did
    not run. The cut handler wrote its name as a pending write of its task.
    An earlier resume of the same round already wrote a record past the
    cursor, so that name is skipped.
    """
    if snapshot is None:
        return []
    values = snapshot.values
    recorded = {
        e.handler
        for e in values.get("events", [])[values.get("_cursor", 0) :]
        if isinstance(e, HandlerCutOnSignal)
    }
    return [
        HandlerCutOnSignal(handler=name)
        for task in snapshot.tasks
        if isinstance(task.result, dict)
        for name in task.result.get(STOPPED_HANDLERS) or []
        if name not in recorded
    ]


def resume_command(value: Event, snapshot: StateSnapshot | None) -> LGCommand:
    """The input that resumes *snapshot* with *value*.

    It also stores a :class:`HandlerCutOnSignal` for each handler that the
    stop cut in the paused round.
    """
    cuts = _cut_records(snapshot)
    if not cuts:
        return LGCommand(resume=value)
    return LGCommand(resume=value, update=pad_causes({"events": cuts}, FRAMEWORK))


def cancelled_on_signal(
    stopped: list[str], new_events: list[Event]
) -> CancelledOnSignal | None:
    """The event that ends the run on its set stop, or ``None`` after a ``Halted``.

    A name that a :class:`HandlerCutOnSignal` in *new_events* records was cut
    before the resume, so ``discarded`` does not name it.
    """
    if any(isinstance(e, Halted) for e in new_events):
        return None
    recorded = {e.handler for e in new_events if isinstance(e, HandlerCutOnSignal)}
    discarded = [name for name in dict.fromkeys(stopped) if name not in recorded]
    return CancelledOnSignal(discarded=", ".join(discarded))
