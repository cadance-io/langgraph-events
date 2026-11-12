"""The deploy step: rewrite the live set of a target's store until it reads clean.

:func:`rewrite_live_set` refuses before it opens the store when the running
code does not hold the expected migrations. It then rewrites the latest
checkpoint of each thread in each checkpoint namespace and plans again,
until a plan reads every checkpoint unchanged or the passes run out. Only
that last plan decides the outcome.
"""

from __future__ import annotations

import enum
import functools
import sys
from dataclasses import dataclass
from typing import TYPE_CHECKING

from langgraph_events._rewrite import RewriteReport, RewriteStatus, ThreadRewrite
from langgraph_events.serde.migrations._target import TargetError, ThreadKey

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable
    from typing import TextIO

    from langgraph_events import EventStore
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


@functools.total_ordering
class Verdict(enum.Enum):
    """The worst state of a checkpoint in the last plan of a live-set
    rewrite.

    The members are in order, from best to worst, and compare in that
    order. ``str()`` gives the value, the word that ``--accept`` takes.
    """

    CLEAN = "clean"
    """The last plan reads every checkpoint unchanged."""
    REWRITE = "rewrite"
    """The last plan reads a checkpoint as a rewrite, and refuses none."""
    REFUSED = "refused"
    """The last plan refuses a checkpoint."""

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, Verdict):
            return NotImplemented
        members = list(Verdict)
        return members.index(self) < members.index(other)

    def __str__(self) -> str:
        return self.value


def _verdict_of(status: RewriteStatus) -> Verdict:
    if status == "unchanged":
        return Verdict.CLEAN
    return Verdict(status)


@dataclass(frozen=True)
class LiveSetOutcome:
    """Every apply pass and the last plan.

    Each report holds one ``ThreadRewrite`` per checkpoint namespace of
    each thread, the root namespace first.
    """

    passes: tuple[RewriteReport, ...]
    final: RewriteReport

    @property
    def verdict(self) -> Verdict:
        """The worst state of a checkpoint in the last plan."""
        return max(
            (_verdict_of(thread.status) for thread in self.final.threads),
            default=Verdict.CLEAN,
        )

    @property
    def clean(self) -> bool:
        """The last plan reads every checkpoint unchanged."""
        return self.verdict is Verdict.CLEAN

    def accepted(self, accept: Verdict) -> bool:
        """The verdict is *accept* or better. The ``rewrite`` command exits
        0 exactly when this is true for its ``--accept``."""
        return self.verdict <= accept

    def render(
        self,
        invocation: Invocation,
        expect: str,
        accept: Verdict = Verdict.CLEAN,
    ) -> str:
        """The verdict, each checkpoint that is not clean, then the next command.

        A checkpoint that the last plan reads as a rewrite also names the
        reason when the last pass refused to write it, and each event class
        that the rewrite drops, with its count of entries. The next command is
        printed when the last plan reads a checkpoint as a rewrite, and the
        verdict is not accepted or the passes ran out. A refused
        checkpoint needs the action that its reason names first.
        """
        head = str(self.final).splitlines()[0]
        if self.clean:
            return f"lge rewrite: clean (last plan: {head})"
        accepted = self.accepted(accept)
        if accepted:
            verdict = f"accepted, {self.verdict} (--accept {accept}; last plan: {head})"
        else:
            verdict = f"not clean (last plan: {head})"
        lines = [f"lge rewrite: {verdict}"]
        last_pass = self.passes[-1].threads if self.passes else ()
        pass_refusals = {_key(t): t.reason for t in last_pass if t.status == "refused"}
        for thread in self.final.threads:
            if thread.status == "rewrite":
                refusal = pass_refusals.get(_key(thread))
                drops = f", drops {thread.drops}" if thread.dropped else ""
                suffix = f", the last pass refused it: {refusal}" if refusal else ""
                lines.append(f"  {thread.location}: needs a rewrite{drops}{suffix}")
            elif thread.status == "refused":
                lines.append(f"  {thread.location}: refused, {thread.reason}")
        if _to_rewrite(self.final) and (self.passes or not accepted):
            if self.passes:
                noun = "pass" if len(self.passes) == 1 else "passes"
                lines.append(
                    f"  The store did not settle after {len(self.passes)} {noun}."
                )
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
    A pass rewrites only the checkpoints that the last plan reads as a
    rewrite. The loop stops when no such checkpoint is left, because a
    refused checkpoint changes only when a person acts. Without *apply*,
    the function only plans.

    Each plan reports a stored instance of a class in ``drop`` of *target*
    as a rewrite, and each apply pass drops it. See :class:`MigrationTarget`.

    Raises:
        ValueError: when *passes* is below 1, or when the graph cannot be
            rewritten, for example under ``legacy_write=True``.
        TargetError: when *target* has no ``store``, or no
            ``NamespaceAwareSerde``, when ``retired=`` of the graph does not
            list a class in ``drop``, when the serde does not revive such a
            class, when ``store`` returns an async context manager, or
            when the saver that ``store`` opens holds a serde with other
            migrations than the code.
    """
    if passes < 1:
        raise ValueError(f"passes must be 1 or more, got {passes}.")
    running = target.snapshot().migrations
    if running.fingerprint != expect:
        return WrongRelease(
            expected=expect,
            running=running.fingerprint,
            running_labels=tuple(sorted(key.label for key in running.keys)),
        )
    out = sys.stderr if log is None else log
    with target.open_store() as saver:
        store = _same_serde(target.build(saver), running)
        plan_rewrite = functools.partial(store.plan_rewrite, drop=target.drop)
        rewrite_store = functools.partial(store.rewrite_store, drop=target.drop)

        def plan() -> RewriteReport:
            return _walk(plan_rewrite, target.thread_keys(saver), applied=False)

        reports: list[RewriteReport] = []
        final = plan()
        while apply and _to_rewrite(final) and len(reports) < passes:
            report = _walk(rewrite_store, _to_rewrite(final), applied=True)
            reports.append(report)
            out.write(f"pass {len(reports)}: {report}\n")
            final = plan()
    return LiveSetOutcome(tuple(reports), final)


def _same_serde(live: Snapshot, code: MigrationSet) -> EventStore:
    """The store built on the real saver, when its serde holds the
    migrations of the code. A saver that already holds a
    ``NamespaceAwareSerde`` keeps it, so the graph factory does not decide
    the serde that writes."""
    if live.migrations.fingerprint != code.fingerprint:
        raise TargetError(
            f"The saver that store= opens holds its own serde, with the "
            f"migrations {live.migrations.fingerprint}. The graph built on an "
            f"empty saver holds {code.fingerprint}, and the gate and the "
            f"--expect fence judge those. Set the serde in the graph factory, "
            f"not in the store."
        )
    return live.store


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
