"""The deploy step: rewrite the live set of a target's store until it reads clean.

:func:`rewrite_live_set` refuses before it opens the store when the running
code does not hold the expected migrations. It then rewrites the latest
checkpoint of each thread in each checkpoint namespace and plans again,
until a plan reads every checkpoint unchanged or the passes run out. Only
that last plan decides the outcome.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import TYPE_CHECKING

from langgraph_events._rewrite import RewriteReport
from langgraph_events.serde.migrations._target import TargetError, ThreadKey

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable
    from typing import TextIO

    from langgraph_events import EventGraph
    from langgraph_events.serde.migrations._core import MigrationSet
    from langgraph_events.serde.migrations._gate import Invocation
    from langgraph_events.serde.migrations._target import MigrationTarget, Snapshot


@dataclass(frozen=True)
class WrongRelease:
    """The running code does not hold the migrations the caller expects.

    ``running_labels`` lists the label of each migration the running code
    holds, so the operator can see which release runs.
    """

    expected: str
    running: str
    running_labels: tuple[str, ...]

    def render(self, invocation: Invocation, *, apply: bool) -> str:
        """The two fingerprints, the running migrations, then the next command.

        *apply* keeps the caller's ``--apply`` in the command.
        """
        labels = ", ".join(self.running_labels) or "none"
        return "\n".join(
            (
                f"lge rewrite: wrong release. The running code holds the "
                f"migrations {self.running}, not {self.expected}.",
                f"  running migrations: {labels}",
                f"  Run the command in the release whose baseline gives "
                f"{self.expected}: {invocation.rewrite(self.expected, apply=apply)}",
            )
        )


@dataclass(frozen=True)
class LiveSetOutcome:
    """Every apply pass and the last plan.

    Each report holds one ``ThreadRewrite`` per checkpoint namespace of
    each thread, the root namespace first.
    """

    passes: tuple[RewriteReport, ...]
    final: RewriteReport

    @property
    def clean(self) -> bool:
        """The last plan reads every checkpoint unchanged."""
        return not _not_clean(self.final)

    def render(self, invocation: Invocation, expect: str) -> str:
        """The verdict, each checkpoint that is not clean, then the next command."""
        head = str(self.final).splitlines()[0]
        if self.clean:
            return f"lge rewrite: clean (last plan: {head})"
        lines = [f"lge rewrite: not clean (last plan: {head})"]
        for thread in self.final.threads:
            if thread.status == "rewrite":
                lines.append(f"  {thread.location}: needs a rewrite")
            elif thread.status == "refused":
                lines.append(f"  {thread.location}: refused, {thread.reason}")
        lines.append(f"  Run: {invocation.rewrite(expect, apply=True)}")
        return "\n".join(lines)


def rewrite_live_set(
    target: MigrationTarget,
    *,
    expect: str,
    apply: bool,
    passes: int = 3,
    log: TextIO | None = None,
) -> LiveSetOutcome | WrongRelease:
    """Rewrite every checkpoint namespace of every thread in the store of
    *target*, then plan again.

    *expect* is the migration fingerprint of the release that must run.
    The code is compared before the store is opened. With *apply*, each
    pass rewrites the checkpoints that the last plan did not read
    unchanged, and logs one line to *log*, by default ``sys.stderr``. A
    run that writes during a pass can leave a checkpoint with old bytes,
    so the next plan reads it as not clean and the next pass rewrites it.
    Without *apply*, the function only plans.

    Raises:
        ValueError: when *passes* is below 1, or when the graph cannot be
            rewritten, for example under ``legacy_write=True``.
        TargetError: when *target* has no ``store``, or no
            ``NamespaceAwareSerde``, or when the saver that ``store``
            opens holds a serde with other migrations than the code.
    """
    if passes < 1:
        raise ValueError(f"passes must be 1 or more, got {passes}.")
    if target.store is None:
        raise TargetError(
            "The target has no store. Set store= on the MigrationTarget to a "
            "function that opens the checkpointer, for example "
            "lambda: PostgresSaver.from_conn_string(url)."
        )
    running = target.snapshot().migrations
    if running.fingerprint != expect:
        return WrongRelease(
            expected=expect,
            running=running.fingerprint,
            running_labels=tuple(sorted(key.label for key in running.keys)),
        )
    out = sys.stderr if log is None else log
    with target.store() as saver:
        graph = _same_serde(target.build(saver), running)

        def plan() -> RewriteReport:
            return _walk(graph.plan_rewrite, target.thread_keys(saver), applied=False)

        reports: list[RewriteReport] = []
        final = plan()
        while apply and _not_clean(final) and len(reports) < passes:
            report = _walk(graph.rewrite_store, _not_clean(final), applied=True)
            reports.append(report)
            out.write(f"pass {len(reports)}: {report}\n")
            final = plan()
    return LiveSetOutcome(tuple(reports), final)


def _same_serde(live: Snapshot, code: MigrationSet) -> EventGraph:
    """The graph on the real store, when its serde holds the migrations of
    the code. A saver that already holds a ``NamespaceAwareSerde`` keeps
    it, so the graph factory does not decide the serde that writes."""
    if live.migrations.fingerprint != code.fingerprint:
        raise TargetError(
            f"The saver that store= opens holds its own serde, with the "
            f"migrations {live.migrations.fingerprint}. The graph built on an "
            f"empty saver holds {code.fingerprint}, and the gate and the "
            f"--expect fence judge those. Set the serde in the graph factory, "
            f"not in the store."
        )
    return live.graph


def _walk(
    method: Callable[..., RewriteReport], keys: Iterable[ThreadKey], *, applied: bool
) -> RewriteReport:
    """One call of *method* per checkpoint namespace, in one report."""
    by_namespace: dict[str, list[str]] = {}
    for key in keys:
        by_namespace.setdefault(key.checkpoint_ns, []).append(key.thread_id)
    return RewriteReport(
        applied,
        tuple(
            thread
            for namespace, thread_ids in sorted(by_namespace.items())
            for thread in method(thread_ids=thread_ids, checkpoint_ns=namespace).threads
        ),
    )


def _not_clean(report: RewriteReport) -> list[ThreadKey]:
    return [
        ThreadKey(t.thread_id, t.checkpoint_ns)
        for t in report.threads
        if t.status != "unchanged"
    ]
