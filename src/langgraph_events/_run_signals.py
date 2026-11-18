"""The ``deadline=``, ``pause_on=`` and ``stop_on=`` of one run call."""

import dataclasses
import time
from typing import TYPE_CHECKING, Any, cast

from langgraph_events._pause import (
    _FRAMEWORK_KEY_PREFIX,
    _PAUSE_BUDGET_KEY,
    _PauseBudget,
)

if TYPE_CHECKING:
    from langchain_core.runnables import RunnableConfig
    from langgraph.types import StateSnapshot

    from langgraph_events._types import PauseSignal


_STOP_SIGNAL_KEY = f"{_FRAMEWORK_KEY_PREFIX}stop_signal"


class ResumeStoppedError(RuntimeError):
    """The ``stop_on=`` signal stopped a resume before it applied its answers.

    The signal was set at the call, or while the resume replayed the paused
    handlers. Each cut question stays open, with the same interrupt id, and
    the resume stores no answer for it. A replay that ended before the stop
    keeps its answer. The resume stores no seed.
    """

    def __init__(self) -> None:
        super().__init__(
            "stop_on= was set before the resume applied its answers. The thread "
            "stays paused on its questions."
        )


_REPLAY_STEP_KEY = f"{_FRAMEWORK_KEY_PREFIX}replay_step"


class SignalLatch:
    """A ``pause_on=`` or ``stop_on=`` signal, held set from its first set read.

    The retry loop and the router both read the ``pause_on=`` signal. When a
    caller clears it between their reads, the retry loop has already given
    up for the pause, so the router must still pause. The first ``True`` from
    ``is_set()`` therefore commits the run to a pause, or to a stop.
    """

    __slots__ = ("_latched", "_signal")

    def __init__(self, signal: PauseSignal) -> None:
        self._signal = signal
        self._latched = False

    def is_set(self) -> bool:
        if not self._latched:
            self._latched = self._signal.is_set()
        return self._latched


def stop_signal(config: RunnableConfig | None) -> SignalLatch | None:
    """The ``stop_on=`` latch that :meth:`_RunSignals.apply` put in *config*."""
    signal: SignalLatch | None = (
        (config or {}).get("configurable", {}).get(_STOP_SIGNAL_KEY)
    )
    return signal


def is_replay(config: RunnableConfig | None) -> bool:
    """Whether the step of *config* replays the paused handler of a resume.

    A resume replays only the paused handlers, all in its first step.
    """
    conf = config or {}
    replay_step = conf.get("configurable", {}).get(_REPLAY_STEP_KEY)
    step = conf.get("metadata", {}).get("langgraph_step")
    return replay_step is not None and step == replay_step


@dataclasses.dataclass(frozen=True)
class _RunSignals:
    """The ``deadline=``, ``pause_on=`` and ``stop_on=`` of one run call.

    Each public entry point builds one, so a bad signal raises ``TypeError``
    before any round runs. ``pause_on=event.is_set`` and
    ``pause_on=threading.Event`` are the usual mistakes.
    """

    deadline: float | None
    pause_on: PauseSignal | None
    stop_on: PauseSignal | None
    replay_step: int | None = None

    def __post_init__(self) -> None:
        for name, signal in (("pause_on", self.pause_on), ("stop_on", self.stop_on)):
            if signal is not None and (
                isinstance(signal, type)
                or not callable(getattr(signal, "is_set", None))
            ):
                raise TypeError(
                    f"{name}= must be an object with an is_set() method, for "
                    f"example an asyncio.Event(), got {type(signal).__name__}: "
                    f"{signal!r}. Pass the event itself, not event.is_set or "
                    f"the Event class."
                )

    def refuse_set_stop(self, config: RunnableConfig | None) -> None:
        """Raise :class:`ResumeStoppedError` when a stop is set at a resume.

        The stop is ``stop_on=``, or a latch that the caller put in *config*.
        """
        if (self.stop_on is not None and self.stop_on.is_set()) or (
            (latch := stop_signal(config)) is not None and latch.is_set()
        ):
            raise ResumeStoppedError

    def for_resume(self, snapshot: StateSnapshot | None) -> _RunSignals:
        """These signals for the resume of *snapshot*.

        The resume replays the paused handlers in the step after *snapshot*.
        A thread with no open interrupt has no paused handler to replay, for
        example after a failed run, so the signals come back unchanged.
        """
        if snapshot is None or snapshot.metadata is None or not snapshot.interrupts:
            return self
        return dataclasses.replace(self, replay_step=snapshot.metadata["step"] + 1)

    def apply(self, config: RunnableConfig | None) -> RunnableConfig:
        """Return *config* with the run's pause budget and stop signal.

        With no signal, *config* comes back unchanged, or as ``{}`` for
        ``None``. Otherwise the result is a copy, so the caller's dict is
        never mutated.
        """
        added: dict[str, Any] = {}
        if self.deadline is not None or self.pause_on is not None:
            latch = None if self.pause_on is None else SignalLatch(self.pause_on)
            added[_PAUSE_BUDGET_KEY] = _PauseBudget(
                time.monotonic(), self.deadline, latch
            )
        if self.stop_on is not None:
            added[_STOP_SIGNAL_KEY] = SignalLatch(self.stop_on)
        if self.replay_step is not None:
            added[_REPLAY_STEP_KEY] = self.replay_step
        if not added:
            return config or {}
        out: dict[str, Any] = dict(config or {})
        out["configurable"] = {**out.get("configurable", {}), **added}
        return cast("RunnableConfig", out)
