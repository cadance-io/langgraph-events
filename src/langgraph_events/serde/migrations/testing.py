"""User-facing test helpers for migration verification.

:func:`assert_migration_gate` runs the one gate in a test.
:func:`assert_resume_recovers` proves a renamed handler resumes a paused
thread. :func:`synthesize_legacy_payload` builds the bytes a prior release
wrote.
"""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING

from langgraph_events._event import Event, Resumed
from langgraph_events.serde.migrations._gate import GateError, Invocation, check
from langgraph_events.serde.migrations._revive import synthesize_legacy_payload

if TYPE_CHECKING:
    from pathlib import Path

    from langchain_core.runnables import RunnableConfig

    from langgraph_events import EventGraph
    from langgraph_events._event_log import EventLog
    from langgraph_events.serde.migrations._target import MigrationTarget

__all__ = [
    "assert_migration_gate",
    "assert_resume_recovers",
    "synthesize_legacy_payload",
]

_PYTHON_TARGET = "<module:attr of the MigrationTarget>"


def assert_migration_gate(
    target: MigrationTarget, *, previous: Path | None = None
) -> None:
    """Run :func:`~langgraph_events.serde.migrations.check` and raise on a finding.

    A test has no previous release, so without *previous* the release rule
    does not run, and the report says so.

    Raises:
        GateError: an ``AssertionError`` whose ``report`` holds the findings.
    """
    report = check(target, previous=previous)
    if not report.passed:
        invocation = Invocation(
            _PYTHON_TARGET, str(previous) if previous is not None else None
        )
        raise GateError(report, invocation)


def assert_resume_recovers(
    before: EventGraph,
    after: EventGraph,
    *,
    seed: Event | list[Event],
    resume_with: Event,
    thread_id: str | None = None,
) -> EventLog:
    """Assert a thread paused inside a handler on *before* resumes on *after*.

    The behavioral twin of the gate's ``handlers`` rule: instead of a
    static name check it exercises the real interrupt→resume path.
    Invokes *before* with *seed* (which must pause via an ``Interrupted``),
    then resumes *after* with *resume_with* on the same checkpoint and asserts
    recovery actually happened — a ``Resumed`` is emitted, which a silent drop
    or a ``halt`` would not produce. Use it to prove an ``@on(previously=...)``
    rename keeps old checkpoints resumable. Returns the post-resume log for
    further assertions.

    *before* and *after* must be constructed with the **same** checkpointer
    instance so the paused checkpoint survives the rebuild.
    """
    if before._checkpointer is None or before._checkpointer is not after._checkpointer:
        raise ValueError(
            "assert_resume_recovers: `before` and `after` must share one "
            "checkpointer instance so the paused checkpoint survives the "
            "rename — e.g. EventGraph(..., checkpointer=saver) for both."
        )
    tid = thread_id or f"resume-recovers-{uuid.uuid4().hex}"
    config: RunnableConfig = {"configurable": {"thread_id": tid}}
    before.invoke(seed, config=config)
    if not before.get_state(config).is_interrupted:
        raise AssertionError(
            "assert_resume_recovers: `seed` did not pause `before` — it must "
            "trigger an Interrupted so there is a paused checkpoint to resume."
        )
    log = after.resume(resume_with, config=config)
    if log.latest(Resumed) is None:
        raise AssertionError(
            "assert_resume_recovers: resume did not recover the paused thread "
            "(no Resumed emitted) — the handler rename is not covered by "
            "@on(previously=...)."
        )
    return log
