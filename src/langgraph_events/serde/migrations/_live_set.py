"""The deploy step: rewrite the live set of a target's store until it reads clean.

:func:`rewrite_live_set` refuses before it opens the store when the running
code does not hold the expected migrations. It then rewrites each root thread
and plans again, until a plan reads every root thread unchanged or the passes
run out. Only that last plan decides the outcome. It reads each subgraph
checkpoint and never writes one.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import TYPE_CHECKING

from langgraph_events.serde.migrations._target import TargetError, ThreadKey

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import TextIO

    from langgraph_events import EventGraph
    from langgraph_events._rewrite import RewriteReport, ThreadRewrite
    from langgraph_events.serde.migrations._core import MigrationKey, MigrationSet
    from langgraph_events.serde.migrations._gate import Invocation
    from langgraph_events.serde.migrations._target import MigrationTarget

_METHOD = "rewrite_live_set"


@dataclass(frozen=True)
class WrongRelease:
    """The running code does not hold the migrations the caller expects.

    ``running_labels`` lists the label of each migration the running code
    holds, so the operator can see which release runs.
    """

    expected: str
    running: str
    running_labels: tuple[str, ...]

    def render(self, invocation: Invocation) -> str:
        """The two fingerprints, the running migrations, then the next command."""
        labels = ", ".join(self.running_labels) or "none"
        return "\n".join(
            (
                f"lge rewrite: wrong release. The running code holds the "
                f"migrations {self.running}, not {self.expected}.",
                f"  running migrations: {labels}",
                f"  Run the command in the release whose baseline gives "
                f"{self.expected}: {invocation.rewrite(self.expected)}",
            )
        )


@dataclass(frozen=True)
class SubgraphTouched:
    """A subgraph checkpoint that the running migrations still change.

    ``thread`` is the read-only plan of the checkpoint. ``migrations`` holds
    each running migration keyed on an identity that the plan migrated:
    those migrations must stay while the checkpoint exists.
    """

    key: ThreadKey
    thread: ThreadRewrite
    migrations: tuple[MigrationKey, ...]


@dataclass(frozen=True)
class LiveSetOutcome:
    """Every apply pass, the last plan, and each subgraph checkpoint it touched."""

    passes: tuple[RewriteReport, ...]
    final: RewriteReport
    subgraphs: tuple[SubgraphTouched, ...]

    @property
    def clean(self) -> bool:
        """The last plan reads every root thread unchanged, and no subgraph
        checkpoint is touched."""
        return not self.subgraphs and all(
            thread.status == "unchanged" for thread in self.final.threads
        )

    def render(self, invocation: Invocation, expect: str) -> str:
        """The verdict, each checkpoint that is not clean, then the next commands."""
        head = str(self.final).splitlines()[0]
        if self.clean:
            return f"lge rewrite: clean (last plan: {head})"
        lines = [f"lge rewrite: not clean (last plan: {head})"]
        for thread in self.final.threads:
            if thread.status == "rewrite":
                lines.append(f"  {thread.thread_id}: needs a rewrite")
            elif thread.status == "refused":
                lines.append(f"  {thread.thread_id}: refused, {thread.reason}")
        for touched in self.subgraphs:
            labels = ", ".join(key.label for key in touched.migrations) or "none"
            lines.append(
                f"  {touched.key.thread_id} in subgraph namespace "
                f"{touched.key.checkpoint_ns!r}: {touched.thread.status}, "
                f"read by: {labels}"
            )
        if _not_clean(self.final):
            lines.append(f"  Run: {invocation.rewrite(expect, apply=True)}")
        lines.extend(
            "  The rewrite never writes a subgraph checkpoint. Keep the migration "
            f"one more release: {invocation.baseline_hold(key.label)}"
            for key in dict.fromkeys(k for t in self.subgraphs for k in t.migrations)
        )
        return "\n".join(lines)


def rewrite_live_set(
    target: MigrationTarget,
    *,
    expect: str,
    apply: bool,
    passes: int = 3,
    log: TextIO | None = None,
) -> LiveSetOutcome | WrongRelease:
    """Rewrite every root thread of the store of *target*, then plan again.

    *expect* is the migration fingerprint of the release that must run.
    The code is compared before the store is opened. With *apply*, each
    pass rewrites the root threads that the last plan did not read
    unchanged, and logs one line to *log*, by default ``sys.stderr``. A
    run that writes during a pass can leave a thread with old bytes, so
    the next plan reads it as not clean and the next pass rewrites it.
    Without *apply*, the function only plans.

    Raises:
        ValueError: when *passes* is below 1, or when the graph cannot be
            rewritten, for example under ``legacy_write=True``.
        TargetError: when *target* has no ``store``, or no
            ``NamespaceAwareSerde``.
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
        graph = target.graph(saver)

        def plan() -> RewriteReport:
            roots = [
                k.thread_id for k in target.thread_keys(saver) if not k.checkpoint_ns
            ]
            return graph.plan_rewrite(thread_ids=roots)

        reports: list[RewriteReport] = []
        final = plan()
        while apply and _not_clean(final) and len(reports) < passes:
            report = graph.rewrite_store(thread_ids=_not_clean(final))
            reports.append(report)
            out.write(f"pass {len(reports)}: {report}\n")
            final = plan()
        subgraphs = _plan_subgraphs(graph, target.thread_keys(saver), running)
    return LiveSetOutcome(tuple(reports), final, subgraphs)


def _not_clean(report: RewriteReport) -> list[str]:
    return [t.thread_id for t in report.threads if t.status != "unchanged"]


def _plan_subgraphs(
    graph: EventGraph, keys: Sequence[ThreadKey], running: MigrationSet
) -> tuple[SubgraphTouched, ...]:
    """A read-only plan of each subgraph checkpoint. Keeps each one that is
    not unchanged."""
    by_namespace: dict[str, list[str]] = {}
    for key in keys:
        if key.checkpoint_ns:
            by_namespace.setdefault(key.checkpoint_ns, []).append(key.thread_id)
    touched: list[SubgraphTouched] = []
    for namespace, thread_ids in sorted(by_namespace.items()):
        report = graph._rewrite_walk(
            _METHOD,
            apply=False,
            drop=(),
            thread_ids=thread_ids,
            checkpoint_ns=namespace,
        )
        for thread in report.threads:
            if thread.status == "unchanged":
                continue
            identities = {label for pair in thread.migrated for label in pair}
            touched.append(
                SubgraphTouched(
                    ThreadKey(thread.thread_id, namespace),
                    thread,
                    tuple(
                        sorted(
                            key
                            for key in running.keys
                            if f"{key.module}:{key.qualname}" in identities
                        )
                    ),
                )
            )
    return tuple(sorted(touched, key=lambda t: t.key))
