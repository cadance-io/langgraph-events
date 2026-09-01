"""Baseline-snapshot diff for migration authoring.

The detection tool compares an :class:`~langgraph_events.EventGraph`'s
current event topology against a stored baseline, surfacing renames that
need migration entries. It is intentionally a *suggestion engine*, not an
applicator — leaf-name + module heuristics are good enough to start a
conversation, never good enough to silently rewrite a project's migration
list.

Typical wiring (project-side pre-commit hook)::

    from cadance.graph import build_graph
    from langgraph_events.serde.migrations.detect import (
        detect_changes,
        write_baseline,
    )

    graph = build_graph()
    report = detect_changes(graph, Path("cadance/migrations/baseline.json"))
    if report.has_changes():
        # Render report; fail commit unless covered by migration entries.
        ...
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterable

    from langgraph_events import EventGraph

from langgraph_events._warn import warn_user
from langgraph_events.serde.migrations._core import RenameEvent

BASELINE_VERSION = 3
# Readable versions: v1 had ``events`` only. v2 adds ``handlers`` (node
# names). v3 adds ``fields`` on each event entry and the ``retired`` list.
# A v1 or v2 baseline still loads: its handler set is empty, every
# ``fields`` value is ``None`` and ``retired`` is empty.
_SUPPORTED_BASELINE_VERSIONS = (1, 2, 3)


@dataclass(frozen=True)
class RenameSuggestion:
    """A removed identity plus the candidate additions that might replace it.

    Used for the ``ambiguous`` bucket — leaf-name + module match yielded
    more than one candidate so the user must disambiguate before applying.
    """

    removed: tuple[str, str]
    candidates: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class ChangeReport:
    """Diff between a baseline snapshot and current graph topology.

    ``confident_renames`` is safe to feed directly into a ``Migration`` list.
    ``ambiguous`` requires user input. ``unmatched_removed`` is most likely
    a delete (no migration needed) but could also be a rename whose new
    location bears no resemblance to the old — surface, don't guess.
    """

    added: tuple[tuple[str, str], ...]
    removed: tuple[tuple[str, str], ...]
    confident_renames: tuple[RenameEvent, ...]
    ambiguous: tuple[RenameSuggestion, ...]
    unmatched_removed: tuple[tuple[str, str], ...]

    def has_changes(self) -> bool:
        """``True`` if the baseline disagrees with the graph in any way."""
        return bool(
            self.added
            or self.removed
            or self.confident_renames
            or self.ambiguous
            or self.unmatched_removed
        )


def write_baseline(
    graph: EventGraph, path: Path | str, *, allow_removed: bool = False
) -> None:
    """Snapshot every event identity reachable from *graph* to *path*.

    The baseline is the answer to "what did the topology look like at the
    point this commit was authored?" — committed alongside migrations so
    diffs against future commits classify changes deterministically.

    Each ``events`` entry records the ``fields`` of its class: the init
    fields only, the same set the serde writes. An ``init=False`` field is
    computed at construction and never sits in a payload. The record
    is cumulative: a field that an earlier baseline recorded stays in the
    record after the live class drops it. A recorded field can sit in a
    checkpoint, so the revive gate must keep exercising it. Removing a
    field from the record is a hand edit.

    A write never erases an identity. An identity the old baseline recorded
    and the topology no longer reaches moves to ``retired``, with the
    ``fields`` last recorded for it. A ``retired`` entry persists across
    later writes until a hand edit removes it, or until the identity is
    live again. The coverage gates walk ``retired`` too, so a forgotten
    migration for a retired identity fails in CI. This compares baseline ↔
    topology only; it never inspects the serde or migration table.

    *allow_removed* is deprecated and does nothing. Passing ``True`` emits a
    ``DeprecationWarning``. The flag will be removed in a later release.

    *path* takes a ``str`` too. It is coerced to ``Path`` immediately,
    matching every ``assert_all_baselined_*`` gate's
    ``baseline_path: Path | str``.
    """
    if allow_removed:
        warn_user(
            "write_baseline(allow_removed=True) does nothing: a write now "
            "moves a dropped identity to the baseline's `retired` list "
            "instead of erasing it. The flag will be removed in a later "
            "release. Remove it from the call.",
            DeprecationWarning,
        )
    path = Path(path)
    live = {
        (module, qualname): frozenset(f.name for f in dataclasses.fields(cls) if f.init)
        for module, qualname, cls in _enumerate_event_classes(graph)
    }
    current = set(live)
    recorded: dict[tuple[str, str], frozenset[str] | None] = {}
    if path.exists():
        raw = _read_baseline(path)
        recorded = _fields_by_identity(raw.get("retired", [])) | _fields_by_identity(
            raw["events"]
        )
    events = {
        identity: fields | (recorded.get(identity) or frozenset())
        for identity, fields in live.items()
    }
    retired = {
        identity: fields
        for identity, fields in recorded.items()
        if identity not in current
    }
    payload = {
        "version": BASELINE_VERSION,
        "events": [
            {"module": module, "qualname": qualname, "fields": sorted(fields)}
            for (module, qualname), fields in sorted(events.items())
        ],
        "retired": [
            _retired_entry(module, qualname, fields)
            for (module, qualname), fields in sorted(retired.items())
        ],
        "handlers": [
            {"name": name} for name in sorted(_enumerate_handler_names(graph))
        ],
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def _retired_entry(
    module: str, qualname: str, fields: frozenset[str] | None
) -> dict[str, Any]:
    """One ``retired`` entry. The ``fields`` key is absent when the last
    record of the identity predates v3."""
    entry: dict[str, Any] = {"module": module, "qualname": qualname}
    if fields is not None:
        entry["fields"] = sorted(fields)
    return entry


def _enumerate_handler_names(graph: EventGraph) -> Iterable[str]:
    """Yield the canonical node name of every handler registered on *graph*.

    These are the names an interrupted checkpoint can be paused at; the
    handler coverage gate asserts each is still live or alias-covered.
    """
    return graph.handler_names


def _load_baseline(baseline_path: Path) -> set[tuple[str, str]]:
    """Parse a baseline file and return its ``(module, qualname)`` set.

    Raises ``ValueError`` on an unsupported version. Shared by
    :func:`detect_changes` and ``assert_all_baselined_cover`` so the
    version-bump error wording lives in exactly one place.
    """
    raw = _read_baseline(baseline_path)
    return {(entry["module"], entry["qualname"]) for entry in raw["events"]}


def _load_baseline_fields(
    baseline_path: Path,
) -> dict[tuple[str, str], frozenset[str] | None]:
    """Parse a baseline file and map each ``events`` identity to its
    recorded field names.

    The value is ``None`` for a v1 or v2 baseline, which predates field
    tracking. The revive gate then synthesizes required placeholders only.
    """
    return _fields_by_identity(_read_baseline(baseline_path)["events"])


def _load_baseline_retired(
    baseline_path: Path,
) -> dict[tuple[str, str], frozenset[str] | None]:
    """Parse a baseline file and map each ``retired`` identity to its
    recorded field names.

    A retired identity is one the topology no longer reaches. The value is
    ``None`` when its last record predates v3. Empty for a v1 or v2 file.
    """
    return _fields_by_identity(_read_baseline(baseline_path).get("retired", []))


def _fields_by_identity(
    entries: list[dict[str, Any]],
) -> dict[tuple[str, str], frozenset[str] | None]:
    """Map each entry's ``(module, qualname)`` to its ``fields``, or to
    ``None`` when the entry has no ``fields`` key."""
    return {
        (entry["module"], entry["qualname"]): (
            frozenset(entry["fields"]) if "fields" in entry else None
        )
        for entry in entries
    }


def _load_baseline_handlers(baseline_path: Path) -> set[str]:
    """Parse a baseline file and return its handler node-name set.

    Returns an empty set for a v1 baseline (which predates handler tracking),
    so the handler coverage gate is a no-op against pre-v2 snapshots rather
    than a spurious failure.
    """
    raw = _read_baseline(baseline_path)
    return {entry["name"] for entry in raw.get("handlers", [])}


def _read_baseline(baseline_path: Path) -> dict[str, Any]:
    """Parse + version-check a baseline file. Raises ``ValueError`` on an
    unsupported version. Single source of the version-bump error wording for
    :func:`_load_baseline` and :func:`_load_baseline_handlers`.
    """
    raw: dict[str, Any] = json.loads(baseline_path.read_text())
    file_version = raw.get("version")
    if file_version not in _SUPPORTED_BASELINE_VERSIONS:
        raise ValueError(
            f"Unsupported baseline version {file_version!r} at "
            f"{baseline_path}; this library reads baseline versions "
            f"{_SUPPORTED_BASELINE_VERSIONS}. Regenerate the baseline with the "
            f"current version of langgraph-events."
        )
    if file_version >= 3:
        _check_v3_shape(raw, baseline_path)
    return raw


def _check_v3_shape(raw: dict[str, Any], baseline_path: Path) -> None:
    """Reject a v3 file whose ``events`` entry lacks ``fields``, or whose
    ``events`` and ``retired`` lists share an identity.

    ``fields`` is mandatory from v3. A file that omits it would silently
    degrade the revive gate. The writer never produces an overlap, so one
    means a hand edit went wrong.
    """
    events = raw["events"]
    incomplete = [e["qualname"] for e in events if "fields" not in e]
    if incomplete:
        raise ValueError(
            f"Baseline at {baseline_path} is version {raw['version']} but "
            f"these events entries have no `fields`: {', '.join(incomplete)}. "
            f'Give each entry "fields": [] and run write_baseline(). The '
            f"write fills the list from the live class."
        )
    live = {(e["module"], e["qualname"]) for e in events}
    retired = {(e["module"], e["qualname"]) for e in raw.get("retired", [])}
    overlap = sorted(live & retired)
    if overlap:
        joined = ", ".join(f"{m}:{q}" for m, q in overlap)
        raise ValueError(
            f"Baseline at {baseline_path} lists these identities under both "
            f"`events` and `retired`: {joined}. Remove it from `retired` when "
            f"the class is live, or from `events` when it is not."
        )


class CoverageError(AssertionError):
    """Base for baseline-coverage failures.

    Subclasses ``AssertionError`` so every coverage gate raises a single
    catchable base (`except CoverageError`). Deliberately message-only — it
    declares no ``uncovered`` attribute, so each concrete subclass is free to
    expose its own correctly-typed payload without a Liskov violation.
    """


MIGRATION_REMEDY = (
    "For each: add @migrate_from to the surviving class or a tombstone, or "
    "append a Migration to migrations=."
)
"""Shared remedy line for every gate that fails on an unreachable event
identity (``MigrationCoverageError`` and both ``_assert_baselined``
gates in ``testing.py``). One wording, so a fix that works for one
gate's failure reads the same for the others."""


RETIRED_REMEDY = (
    "A write keeps a retired identity in the baseline: delete its `retired` "
    "entry by hand once every thread that names it is settled "
    "(graph.unrevivable_threads() == {})."
)
"""The hand-delete rule for an identity the baseline lists under
``retired``. Follows :data:`MIGRATION_REMEDY`, which names the migration
fix. A write never removes the entry."""


class MigrationCoverageError(CoverageError):
    """Raised when a baselined event identity has no migration and no live class.

    ``uncovered`` is the tuple of offending ``(module, qualname)`` identities
    for custom CI reporters. ``retired`` is the subset the baseline lists
    under ``retired``. Raised by ``assert_all_baselined_cover`` /
    ``_resolve`` / ``_revive``.
    """

    def __init__(
        self,
        uncovered: tuple[tuple[str, str], ...],
        retired: tuple[tuple[str, str], ...] = (),
    ) -> None:
        self.uncovered = uncovered
        self.retired = retired
        plural = "y" if len(uncovered) == 1 else "ies"
        verb = "is" if len(uncovered) == 1 else "are"
        message = (
            f"{len(uncovered)} identit{plural} in the baseline {verb} neither "
            f"currently live nor covered by a migration"
        )
        # Each identity is listed once: under ``retired`` when the baseline
        # retires it, in the first sentence otherwise.
        live_missing = tuple(i for i in uncovered if i not in retired)
        if live_missing:
            message += ": " + _join(live_missing)
        message += f". {MIGRATION_REMEDY}"
        if retired:
            message += f" The baseline lists as retired: {_join(retired)}. "
            message += RETIRED_REMEDY
        super().__init__(message)


def _join(identities: tuple[tuple[str, str], ...]) -> str:
    """``module:qualname`` for each identity, comma separated."""
    return ", ".join(f"{m}:{q}" for m, q in identities)


class HandlerCoverageError(CoverageError):
    """Raised when a baselined handler node name is neither live on the graph
    nor covered by an ``@on(previously=...)`` alias.

    Sibling of :class:`MigrationCoverageError` under :class:`CoverageError`;
    ``uncovered`` is the tuple of offending handler node names (plain strings,
    not event identities) — caught together via ``except CoverageError``.
    """

    def __init__(self, uncovered: tuple[str, ...]) -> None:
        self.uncovered = uncovered
        joined = ", ".join(uncovered)
        plural = "" if len(uncovered) == 1 else "s"
        verb = "resolves" if len(uncovered) == 1 else "resolve"
        hint = uncovered[0] if uncovered else "old_name"
        super().__init__(
            f"{len(uncovered)} baselined handler{plural} no longer {verb} to a "
            f"live node: {joined}. For each: add @on(previously={hint!r}) to the "
            f"surviving handler, or regenerate the baseline if the handler is "
            f"intentionally removed."
        )


class BaselineRegressionError(ValueError):
    """Retained for one release so an existing ``except`` clause still
    imports. :func:`write_baseline` no longer raises it.

    Earlier releases raised it when a write would erase an identity the
    existing baseline recorded. A write now moves such an identity to the
    ``retired`` list, and the coverage gates catch a forgotten migration
    for it. Attribute ``removed`` is the tuple of ``(module, qualname)``
    identities the write would have erased.
    """

    def __init__(self, removed: tuple[tuple[str, str], ...]) -> None:
        self.removed = removed
        joined = ", ".join(f"{m}:{q}" for m, q in removed)
        plural = "y" if len(removed) == 1 else "ies"
        super().__init__(
            f"{len(removed)} identit{plural} in the existing baseline would "
            f"be erased by this write: {joined}. Add @migrate_from / "
            f"@backfill (or a Migration) covering them and regenerate."
        )


def detect_changes(graph: EventGraph, baseline_path: Path) -> ChangeReport:
    """Diff the current graph topology against the stored baseline."""
    baseline = _load_baseline(baseline_path)
    current = set(_enumerate_identities(graph))
    return _diff_identities(current, baseline)


def _enumerate_identities(graph: EventGraph) -> Iterable[tuple[str, str]]:
    """Yield ``(module, qualname)`` for every event reachable from *graph*.

    Covers: Commands themselves, all Command outcomes, free-standing
    DomainEvents, IntegrationEvents, and SystemEvents. Matches the surface
    the serde encodes — anything reachable here is something that could
    appear in a checkpoint payload.
    """
    for module, qualname, _cls in _enumerate_event_classes(graph):
        yield (module, qualname)


def _enumerate_event_classes(
    graph: EventGraph,
) -> Iterable[tuple[str, str, type]]:
    """Yield ``(module, qualname, cls)`` for every event reachable from *graph*.

    The class is the source of the ``fields`` a v3 baseline records.
    :func:`_enumerate_identities` derives from this walk.
    """
    model = graph.namespaces()
    for namespace in model.namespaces.values():
        for command in namespace.commands.values():
            yield (command.cls.__module__, command.cls.__qualname__, command.cls)
            for outcome in command.outcomes:
                yield (outcome.__module__, outcome.__qualname__, outcome)
        for event in namespace.events:
            yield (event.__module__, event.__qualname__, event)
    for event in model.integration_events:
        yield (event.__module__, event.__qualname__, event)
    for event in model.system_events:
        yield (event.__module__, event.__qualname__, event)


def _leaf(qualname: str) -> str:
    """Last dotted segment — used as the rename heuristic key."""
    return qualname.rsplit(".", 1)[-1]


def _diff_identities(
    current: set[tuple[str, str]],
    baseline: set[tuple[str, str]],
) -> ChangeReport:
    """Bucket the symmetric diff into rename candidates.

    Heuristic: a removed identity matches an added identity when their
    leaf names agree. Multiple matches per removed → ambiguous. Zero
    matches → unmatched_removed (likely delete). Pure additions land in
    ``added`` (not a migration concern).
    """
    added_only = tuple(sorted(current - baseline))
    removed_only = tuple(sorted(baseline - current))

    additions_by_leaf: dict[str, list[tuple[str, str]]] = {}
    for module, qualname in added_only:
        additions_by_leaf.setdefault(_leaf(qualname), []).append((module, qualname))

    confident: list[RenameEvent] = []
    ambiguous: list[RenameSuggestion] = []
    unmatched: list[tuple[str, str]] = []

    consumed_additions: set[tuple[str, str]] = set()
    for old_module, old_qualname in removed_only:
        candidates = additions_by_leaf.get(_leaf(old_qualname), [])
        if not candidates:
            unmatched.append((old_module, old_qualname))
        elif len(candidates) == 1:
            new_module, new_qualname = candidates[0]
            confident.append(
                RenameEvent(
                    old_module=old_module,
                    old_qualname=old_qualname,
                    new_module=new_module,
                    new_qualname=new_qualname,
                )
            )
            consumed_additions.add((new_module, new_qualname))
        else:
            ambiguous.append(
                RenameSuggestion(
                    removed=(old_module, old_qualname),
                    candidates=tuple(candidates),
                )
            )

    # Additions that didn't get consumed by a confident rename remain in
    # ``added`` so users can see them — they're either pure additions
    # (fine) or ambiguous candidates already surfaced.
    remaining_added = tuple(a for a in added_only if a not in consumed_additions)

    return ChangeReport(
        added=remaining_added,
        removed=removed_only,
        confident_renames=tuple(confident),
        ambiguous=tuple(ambiguous),
        unmatched_removed=tuple(unmatched),
    )
