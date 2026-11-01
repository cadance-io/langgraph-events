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
from typing import TYPE_CHECKING, Literal

from langgraph_events._rewrite import RewriteReport, ThreadRewrite
from langgraph_events.serde.migrations._backup import BackupCounts, _Backup
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


LiveSetState = Literal["clean", "rewrite", "refused"]
"""The worst state in the last plan of a live-set rewrite."""

_STATES: tuple[LiveSetState, ...] = ("clean", "rewrite", "refused")


@dataclass(frozen=True)
class LiveSetOutcome:
    """Every apply pass, the last plan, and the backup taken before the
    first pass.

    Each report holds one ``ThreadRewrite`` per checkpoint namespace of
    each thread, the root namespace first. ``backup`` is ``None`` when the
    caller asked for no backup.
    """

    passes: tuple[RewriteReport, ...]
    final: RewriteReport
    backup: BackupCounts | None = None

    @property
    def clean(self) -> bool:
        """The last plan reads every checkpoint unchanged."""
        return self.state == "clean"

    @property
    def state(self) -> LiveSetState:
        """The worst state in the last plan: ``"refused"`` when it refuses
        a checkpoint, else ``"rewrite"`` when a checkpoint needs a rewrite,
        else ``"clean"``."""
        statuses = {thread.status for thread in self.final.threads}
        if "refused" in statuses:
            return "refused"
        if "rewrite" in statuses:
            return "rewrite"
        return "clean"

    def within(self, accept: LiveSetState) -> bool:
        """The state is *accept* or better, in the order ``"clean"``,
        ``"rewrite"``, ``"refused"``.

        Raises:
            ValueError: when *accept* is not one of the three states.
        """
        if accept not in _STATES:
            raise ValueError(
                f"accept must be one of {', '.join(_STATES)}, got {accept!r}."
            )
        return _STATES.index(self.state) <= _STATES.index(accept)

    def render(
        self, invocation: Invocation, expect: str, accept: LiveSetState = "clean"
    ) -> str:
        """The verdict, each checkpoint that is not clean, then the next command.

        A checkpoint that the last plan reads as a rewrite also names the
        reason when the last pass refused to write it. The next command is
        printed only when the state is worse than *accept* and the last
        plan reads a checkpoint as a rewrite. A refused checkpoint needs
        the action that its reason names first.
        """
        head = str(self.final).splitlines()[0]
        if self.clean:
            return f"lge rewrite: clean (last plan: {head})"
        accepted = self.within(accept)
        if accepted:
            verdict = f"accepted, {self.state} (--accept {accept}; last plan: {head})"
        else:
            verdict = f"not clean (last plan: {head})"
        lines = [f"lge rewrite: {verdict}"]
        last_pass = self.passes[-1].threads if self.passes else ()
        pass_refusals = {_key(t): t.reason for t in last_pass if t.status == "refused"}
        for thread in self.final.threads:
            if thread.status == "rewrite":
                refusal = pass_refusals.get(_key(thread))
                suffix = f", the last pass refused it: {refusal}" if refusal else ""
                lines.append(f"  {thread.location}: needs a rewrite{suffix}")
            elif thread.status == "refused":
                lines.append(f"  {thread.location}: refused, {thread.reason}")
        if not accepted and any(t.status == "rewrite" for t in self.final.threads):
            lines.append(f"  Run: {invocation.rewrite(expect, apply=True)}")
        return "\n".join(lines)


def rewrite_live_set(
    target: MigrationTarget,
    *,
    expect: str,
    apply: bool,
    passes: int = 3,
    log: TextIO | None = None,
    backup: str | None = None,
) -> LiveSetOutcome | WrongRelease:
    """Rewrite every checkpoint namespace of every thread in the store of
    *target*, then plan again.

    *expect* is the migration fingerprint of the release that must run.
    The code is compared before the store is opened. With *apply*, each
    pass rewrites the checkpoints that the last plan did not read
    unchanged, and logs one line to *log*, by default ``sys.stderr``. A
    run that writes during a pass can leave a checkpoint with old bytes,
    so the next plan reads it as not clean and the next pass rewrites it.
    A pass rewrites only the checkpoints that the last plan reads as a
    rewrite. The loop stops when no such checkpoint is left, because a
    refused checkpoint changes only when a person acts. Without *apply*,
    the function only plans.

    With *backup*, a tag, the function backs up the live set of a
    ``PostgresSaver`` store before the first plan, in one transaction:
    the latest checkpoint of each thread in each namespace into
    ``checkpoints_bak_<tag>``, and its pending writes into
    ``checkpoint_writes_bak_<tag>``. The same transaction drops the backup
    tables of every other tag in the current schema of the store. A rerun
    with the same tag keeps each row the backup holds. It logs one line to
    *log*. :func:`restore_live_set` puts the backup back.

    Raises:
        ValueError: when *passes* is below 1, when *backup* is given
            without *apply* or is not 1 to 32 characters of ``a-z``,
            ``0-9`` and ``_``, or when the graph cannot be rewritten, for
            example under ``legacy_write=True``.
        TargetError: when *target* has no ``store``, or no
            ``NamespaceAwareSerde``, when ``store`` returns an async context
            manager, when the saver that ``store`` opens holds a serde
            with other migrations than the code, or when the backup
            fails or the saver is not a ``PostgresSaver``. A failed backup
            writes nothing, and no pass runs.
    """
    if passes < 1:
        raise ValueError(f"passes must be 1 or more, got {passes}.")
    if backup is not None and not apply:
        raise ValueError(
            "backup= needs apply=True. A plan writes nothing, so it needs no backup."
        )
    tables = _Backup.of(backup) if backup is not None else None
    running = target.snapshot().migrations
    if running.fingerprint != expect:
        return WrongRelease(
            expected=expect,
            running=running.fingerprint,
            running_labels=tuple(sorted(key.label for key in running.keys)),
        )
    out = sys.stderr if log is None else log
    with target.open_store() as saver:
        graph = _same_serde(target.build(saver), running)
        counts = None
        if tables is not None:
            counts = tables.take(saver)
            out.write(
                f"backup {tables.tag}: {counts} in {tables.checkpoints} and "
                f"{tables.writes}\n"
            )

        def plan() -> RewriteReport:
            return _walk(graph.plan_rewrite, target.thread_keys(saver), applied=False)

        reports: list[RewriteReport] = []
        final = plan()
        while apply and _to_rewrite(final) and len(reports) < passes:
            report = _walk(graph.rewrite_store, _to_rewrite(final), applied=True)
            reports.append(report)
            out.write(f"pass {len(reports)}: {report}\n")
            final = plan()
    return LiveSetOutcome(tuple(reports), final, counts)


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


def _key(thread: ThreadRewrite) -> ThreadKey:
    return ThreadKey(thread.thread_id, thread.checkpoint_ns)


def _to_rewrite(report: RewriteReport) -> list[ThreadKey]:
    """The checkpoints that a pass can change. A plan refuses a checkpoint
    that only a person can clear, so a pass leaves it out."""
    return [_key(t) for t in report.threads if t.status == "rewrite"]
