"""The one migration gate: ``check`` and the two commands that write a baseline.

The CLI and :func:`~langgraph_events.serde.migrations.assert_migration_gate`
both call :func:`check`. Each finding owns its explanation and one next
command, rendered with the caller's own arguments.
"""

from __future__ import annotations

import shlex
import typing
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, Literal

from langgraph_events.serde.migrations._baseline import (
    Baseline,
    Hold,
    MigrationRecord,
    MigrationsUnrecorded,
)
from langgraph_events.serde.migrations._core import (
    PROGRAM,
    TARGET_PLACEHOLDER,
    MigrationKey,
)
from langgraph_events.serde.migrations._revive import revival_failure

if TYPE_CHECKING:
    from pathlib import Path

    from langgraph_events._identity import Identity
    from langgraph_events.serde.migrations._target import MigrationTarget, Snapshot

Rule = Literal["baseline", "names", "coverage", "revive", "handlers", "release"]
_RULE_ORDER: tuple[Rule, ...] = typing.get_args(Rule)
ReleaseNotRun = Literal[
    "no previous baseline was given",
    "the previous baseline records no migrations",
]
_PREVIOUS_PLACEHOLDER = "<baseline file of the previous release>"


@dataclass(frozen=True)
class Invocation:
    """The caller's own arguments, so that each remedy is a command to run.

    ``target`` is the ``module:attr`` of the target. ``previous`` is the
    path of the previous release's baseline, when the caller gave one.
    """

    target: str
    previous: str | None = None

    def baseline_update(self, *, prune: bool = False) -> str:
        """The ``baseline update`` command. *prune* asks for ``--previous``."""
        command = f"{PROGRAM} baseline update {self.target}"
        if self.previous is not None:
            return f"{command} --previous {shlex.quote(self.previous)}"
        if prune:
            return f"{command} --previous {_PREVIOUS_PLACEHOLDER}"
        return command

    def baseline_hold(self, label: str) -> str:
        """The ``baseline hold`` command for the migration *label*."""
        previous = (
            _PREVIOUS_PLACEHOLDER
            if self.previous is None
            else shlex.quote(self.previous)
        )
        return (
            f"{PROGRAM} baseline hold {self.target} {shlex.quote(label)} "
            f"--reason <why> --previous {previous}"
        )

    def rewrite(self, expect: str, *, apply: bool = False) -> str:
        """The ``rewrite`` command. *apply* asks for ``--apply``."""
        flag = " --apply" if apply else ""
        return f"{PROGRAM} rewrite {self.target}{flag} --expect {expect}"


class PreviousRequiredError(ValueError):
    """Migrations left the code, and the update has no previous baseline.

    Only the previous release's baseline says whether a release shipped
    them, and so whether the history they covered can go.
    """

    def __init__(self, deleted: tuple[MigrationKey, ...]) -> None:
        self.deleted = deleted
        super().__init__(self.render(Invocation(TARGET_PLACEHOLDER)))

    def render(self, invocation: Invocation) -> str:
        labels = ", ".join(key.label for key in self.deleted)
        return (
            f"These migrations left the code: {labels}. Only the baseline of "
            f"the previous release says whether a release shipped them. "
            f"Run: {invocation.baseline_update(prune=True)}"
        )


def _identity_label(identity: Identity, *, retired: bool) -> str:
    return f"{identity.label} (retired)" if retired else identity.label


@dataclass(frozen=True)
class BaselineStale:
    """The committed baseline differs from the baseline the code needs."""

    code: ClassVar[str] = "baseline-stale"
    changes: tuple[str, ...]
    deleted: tuple[MigrationKey, ...]

    @property
    def subject(self) -> str:
        return "the committed baseline"

    def lines(self, invocation: Invocation) -> tuple[str, ...]:
        """The explanation, the differences, then the next command."""
        explanation = ("The baseline does not match the code:",)
        changes = tuple(f"  {change}" for change in self.changes)
        if not self.deleted:
            return (*explanation, *changes, f"Run: {invocation.baseline_update()}")
        return (
            *explanation,
            *changes,
            "A migration left the code. When the previous release shipped it, "
            "the update with --previous drops the history it covered.",
            f"Run: {invocation.baseline_update(prune=True)}",
        )


@dataclass(frozen=True)
class AnonymousCallable:
    """A transform or a split whose callable has no stable name."""

    code: ClassVar[str] = "anonymous-callable"
    key: MigrationKey

    @property
    def subject(self) -> str:
        return self.key.label

    def lines(self, invocation: Invocation) -> tuple[str, ...]:
        """The explanation, then the next command."""
        return (
            "A lambda, a callable object, or a local function that captures a "
            "value with no stable form has no stable name, so a replacement "
            "callable reads as the same migration.",
            "Give the callable a module-level name, then run: "
            f"{invocation.baseline_update()}",
        )


@dataclass(frozen=True)
class Uncovered:
    """A recorded identity that no live class and no migration covers.

    ``renamed_to`` holds the live identities with the same last qualname
    segment: a hint for a rename with no ``@migrate_from``.
    """

    code: ClassVar[str] = "uncovered"
    identity: Identity
    retired: bool
    renamed_to: tuple[Identity, ...] = ()

    @property
    def subject(self) -> str:
        return _identity_label(self.identity, retired=self.retired)

    def lines(self, invocation: Invocation) -> tuple[str, ...]:
        """The explanation, any rename hint, then the next command."""
        hint: tuple[str, ...] = ()
        if self.renamed_to:
            joined = ", ".join(identity.label for identity in self.renamed_to)
            hint = (f"Live classes with the same name: {joined}.",)
        return (
            "No live class and no migration covers this recorded identity, "
            "so a stored payload of it cannot revive.",
            *hint,
            f"Add @migrate_from({self.identity.qualname!r}) to the class that "
            f"replaces it, or a Migration to migrations=, then run: "
            f"{invocation.baseline_update()}",
        )


@dataclass(frozen=True)
class MigrationOutOfScope:
    """A class of the graph declares a read-time migration that the serde
    does not collect, because the serde does not read the class."""

    code: ClassVar[str] = "migration-out-of-scope"
    identity: Identity

    @property
    def subject(self) -> str:
        return self.identity.label

    def lines(self, invocation: Invocation) -> tuple[str, ...]:
        """The explanation, then the fix and the next command."""
        return (
            "The class declares a read-time migration, but the serde does not "
            "read the class, so the migration does nothing.",
            "Pass its namespace in namespaces= of the serde, or the class in "
            f"events=, then run: {invocation.baseline_update()}",
        )


@dataclass(frozen=True)
class Unrevivable:
    """A recorded shape whose synthesized payload fails to revive."""

    code: ClassVar[str] = "unrevivable"
    identity: Identity
    retired: bool
    error: str

    @property
    def subject(self) -> str:
        return _identity_label(self.identity, retired=self.retired)

    def lines(self, invocation: Invocation) -> tuple[str, ...]:
        """The read error, then the next command."""
        return (
            f"A stored payload of the recorded shape does not revive: {self.error}",
            "Add or fix the migration that covers it, then run: "
            f"{invocation.baseline_update()}",
        )


@dataclass(frozen=True)
class HandlerMissing:
    """A recorded handler node name that no live handler or alias has."""

    code: ClassVar[str] = "handler-missing"
    name: str

    @property
    def subject(self) -> str:
        return self.name

    def lines(self, invocation: Invocation) -> tuple[str, ...]:
        """The explanation, then the next command."""
        return (
            "No live handler and no @on(previously=...) alias has this node "
            "name, so a thread paused at it cannot resume.",
            f"Add @on(previously={self.name!r}) to the handler that replaces "
            f"it, then run: {invocation.baseline_update()}",
        )


@dataclass(frozen=True)
class SurvivedRelease:
    """A migration that the previous release shipped is still in the code.

    ``expired`` is a hold made against an older previous baseline.
    """

    code: ClassVar[str] = "survived-release"
    key: MigrationKey
    expired: Hold | None = None

    @property
    def subject(self) -> str:
        return self.key.label

    def lines(self, invocation: Invocation) -> tuple[str, ...]:
        """The explanation, any expired hold, then the next command."""
        delete = "Delete it, then run: " + invocation.baseline_update(prune=True)
        if self.expired is None:
            return ("The previous release shipped this migration.", delete)
        return (
            "The previous release shipped this migration.",
            f"Its hold ({self.expired.reason}) was made against "
            f"{self.expired.against}, an older release, so it expired.",
            delete,
            "Or keep it one more release: " + invocation.baseline_hold(self.key.label),
        )


Finding = (
    BaselineStale
    | AnonymousCallable
    | MigrationOutOfScope
    | Uncovered
    | Unrevivable
    | HandlerMissing
    | SurvivedRelease
)


@dataclass(frozen=True)
class GateReport:
    """The findings of one :func:`check`, and the rules it ran.

    ``release_not_run`` says why the release rule did not run. ``held``
    lists each migration whose hold applies. An expired hold is in the
    ``SurvivedRelease`` finding of its migration.
    """

    findings: tuple[Finding, ...]
    rules_run: frozenset[Rule]
    release_not_run: ReleaseNotRun | None
    held: tuple[MigrationRecord, ...]

    @property
    def passed(self) -> bool:
        return not self.findings

    def render(self, invocation: Invocation) -> str:
        """One block per finding: what, why, and one next command."""
        scope = "rules run: " + ", ".join(
            rule for rule in _RULE_ORDER if rule in self.rules_run
        )
        if self.release_not_run is not None:
            scope += f"; release rule not run: {self.release_not_run}"
        count = len(self.findings)
        head = (
            "passed" if self.passed else f"{count} finding{'' if count == 1 else 's'}"
        )
        lines = [f"lge migration gate: {head} ({scope})"]
        for finding in self.findings:
            lines.append(f"  {finding.code}  {finding.subject}")
            lines.extend(f"    {line}" for line in finding.lines(invocation))
        for record in self.held:
            if record.hold is not None:
                lines.append(
                    f"  held  {record.key.label}: {record.hold.reason} "
                    f"(made against {record.hold.against})"
                )
        return "\n".join(lines)


class GateError(AssertionError):
    """The gate found a problem. ``report`` holds the typed findings."""

    def __init__(self, report: GateReport, invocation: Invocation) -> None:
        self.report = report
        super().__init__(report.render(invocation))


def check(target: MigrationTarget, *, previous: Path | None = None) -> GateReport:
    """Judge the code of *target* against its committed baseline.

    *previous* is the baseline file of the previous release. Without it,
    or when it records no migrations, the release rule does not run and
    the report says why. The baseline rule runs first, and the other
    rules judge the baseline the code needs. So a deleted shipped
    migration gives one ``BaselineStale``, never a request to add a
    migration back. Without *previous*, those rules judge the committed
    baseline as the previous one, for the same reason.
    """
    code = target.snapshot()
    committed = Baseline.read(target.baseline)
    prev = _read_previous(previous)
    want = committed.updated(code, prev)
    judged = want if prev is not None else committed.updated(code, committed)
    findings: list[Finding] = []
    if want != committed:
        findings.append(
            BaselineStale(committed.changes_to(want), _deleted(committed, code))
        )
    findings.extend(
        AnonymousCallable(k) for k in sorted(code.migrations.keys) if k.anonymous
    )
    findings.extend(MigrationOutOfScope(i) for i in code.migrations_out_of_scope())
    findings.extend(_coverage_and_revival(code, judged))
    findings.extend(
        HandlerMissing(name)
        for name in sorted(committed.handlers - code.graph.resumable_node_names)
    )
    rules = set(_RULE_ORDER) - {"release"}
    not_run: ReleaseNotRun | None = None
    if prev is None:
        not_run = "no previous baseline was given"
    elif isinstance(prev.code, MigrationsUnrecorded):
        not_run = "the previous baseline records no migrations"
    else:
        rules.add("release")
        findings.extend(_survivors(code, want, prev.code.keys, prev.digest))
    expired = {f.key for f in findings if isinstance(f, SurvivedRelease)}
    return GateReport(
        findings=tuple(findings),
        rules_run=frozenset(rules),
        release_not_run=not_run,
        held=tuple(
            r for r in want.records if r.hold is not None and r.key not in expired
        ),
    )


def update_baseline(target: MigrationTarget, *, previous: Path | None = None) -> None:
    """Write the baseline the code of *target* needs.

    With *previous*, the baseline of the previous release, the update drops
    the history of each migration that release shipped and the code no
    longer holds.

    Raises:
        PreviousRequiredError: when *previous* is not given and a migration
            that the committed baseline records left the code.
    """
    code = target.snapshot()
    committed = Baseline.read(target.baseline)
    if previous is None:
        deleted = _deleted(committed, code)
        if deleted:
            raise PreviousRequiredError(deleted)
    committed.updated(code, _read_previous(previous)).write(target.baseline)


def hold_migration(
    target: MigrationTarget, label: str, *, reason: str, previous: Path
) -> None:
    """Keep the shipped migration *label* for one more release.

    The hold records the digest of *previous*. A check against another
    previous baseline ignores it.

    Raises:
        ValueError: when *label* is not a migration the code holds and
            *previous* shipped.
    """
    code = target.snapshot()
    previous_baseline = _read_existing(previous)
    previous_keys = previous_baseline.recorded_keys
    if isinstance(previous_keys, MigrationsUnrecorded):
        raise ValueError(
            f"The previous baseline at {previous} is version "
            f"{previous_keys.version} and records no migrations, so no "
            f"migration in it can be held."
        )
    holdable = code.migrations.shipped_in(previous_keys)
    choices = ", ".join(key.label for key in sorted(holdable)) or "none"
    try:
        key = MigrationKey.parse(label)
    except ValueError as error:
        raise ValueError(f"{error} Labels you can hold: {choices}.") from error
    if key not in code.migrations.keys:
        raise ValueError(
            f"The code does not hold {label!r}. A hold needs a live migration. "
            f"Labels you can hold: {choices}."
        )
    if key not in holdable:
        raise ValueError(
            f"The previous release did not ship {label!r}, so the release rule "
            f"does not ask to delete it. There is nothing to hold."
        )
    hold = Hold(reason, previous_baseline.digest)
    Baseline.read(target.baseline).held(key, hold).write(target.baseline)


def _read_previous(previous: Path | None) -> Baseline | None:
    return None if previous is None else _read_existing(previous)


def _read_existing(path: Path) -> Baseline:
    if not path.exists():
        raise ValueError(f"No previous baseline at {path}.")
    return Baseline.read(path)


def _deleted(committed: Baseline, code: Snapshot) -> tuple[MigrationKey, ...]:
    keys = committed.recorded_keys
    if isinstance(keys, MigrationsUnrecorded):
        return ()
    return tuple(sorted(code.migrations.deleted_since(keys)))


def _coverage_and_revival(
    code: Snapshot, judged: Baseline
) -> list[Uncovered | Unrevivable]:
    recorded = judged.history
    revivable = code.revivable_identities()
    findings: list[Uncovered | Unrevivable] = []
    for identity in sorted(recorded.keys() - revivable):
        leaf = identity.qualname.rsplit(".", 1)[-1]
        findings.append(
            Uncovered(
                identity,
                retired=identity in judged.retired,
                renamed_to=tuple(
                    live
                    for live in sorted(judged.events)
                    if live != identity and live.qualname.rsplit(".", 1)[-1] == leaf
                ),
            )
        )
    for identity, fields in sorted(recorded.items()):
        if identity not in revivable:
            continue
        failure = revival_failure(code.serde, identity, fields)
        if failure is not None:
            findings.append(
                Unrevivable(identity, retired=identity in judged.retired, error=failure)
            )
    return findings


def _survivors(
    code: Snapshot, want: Baseline, shipped: frozenset[MigrationKey], digest: str
) -> list[SurvivedRelease]:
    """Shipped keys still in the code, except an anonymous key or a key held
    against the previous baseline of *digest*.

    An anonymous key has its own finding: a replacement callable reads as
    the same key, so it must never count as a survivor.
    """
    holds = {record.key: record.hold for record in want.records}
    survivors = []
    for key in sorted(code.migrations.shipped_in(shipped)):
        hold = holds.get(key)
        if key.anonymous or (hold is not None and hold.against == digest):
            continue
        survivors.append(SurvivedRelease(key, expired=hold))
    return survivors
