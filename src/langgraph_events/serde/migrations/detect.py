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

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path

    from langgraph_events import EventGraph

from langgraph_events.serde.migrations._core import RenameEvent

BASELINE_VERSION = 1


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
    graph: EventGraph, path: Path, *, allow_removed: bool = False
) -> None:
    """Snapshot every event identity reachable from *graph* to *path*.

    The baseline is the answer to "what did the topology look like at the
    point this commit was authored?" — committed alongside migrations so
    diffs against future commits classify changes deterministically.

    Refuses to silently regress: if *path* already exists and the new
    snapshot would drop identities the old one recorded, raise
    :class:`BaselineRegressionError` — overwriting them away would make
    :func:`detect_changes` / ``assert_covers`` permanently blind to a
    forgotten migration for those identities. Pass ``allow_removed=True``
    to overwrite anyway (intentional deletes). This compares baseline ↔
    topology only; it never inspects the serde or migration table —
    coverage stays with ``assert_covers`` / ``assert_all_baselined_revive``.
    """
    current = set(_enumerate_identities(graph))
    if path.exists() and not allow_removed:
        removed = tuple(sorted(_load_baseline(path) - current))
        if removed:
            raise BaselineRegressionError(removed)
    identities = sorted(current)
    payload = {
        "version": BASELINE_VERSION,
        "events": [
            {"module": module, "qualname": qualname} for module, qualname in identities
        ],
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def _load_baseline(baseline_path: Path) -> set[tuple[str, str]]:
    """Parse a baseline file and return its ``(module, qualname)`` set.

    Raises ``ValueError`` on an unsupported version. Shared by
    :func:`detect_changes` and ``NamespaceAwareSerde.assert_covers`` so the
    version-bump error wording lives in exactly one place.
    """
    raw = json.loads(baseline_path.read_text())
    file_version = raw.get("version")
    if file_version != BASELINE_VERSION:
        raise ValueError(
            f"Unsupported baseline version {file_version!r} at "
            f"{baseline_path}; this library reads baseline version "
            f"{BASELINE_VERSION}. Regenerate the baseline with the "
            f"current version of langgraph-events."
        )
    return {(entry["module"], entry["qualname"]) for entry in raw["events"]}


class MigrationCoverageError(ValueError):
    """Raised when a baseline identity has no migration and no live class.

    Attribute ``uncovered`` is the tuple of offending ``(module, qualname)``
    identities so a custom CI reporter can format them however it wants.
    """

    def __init__(self, uncovered: tuple[tuple[str, str], ...]) -> None:
        self.uncovered = uncovered
        joined = ", ".join(f"{m}:{q}" for m, q in uncovered)
        plural = "y" if len(uncovered) == 1 else "ies"
        super().__init__(
            f"{len(uncovered)} identit{plural} in the baseline are neither "
            f"currently live nor covered by a migration: {joined}. For each: "
            f"either add @migrate_from to the surviving class, append a "
            f"Migration to migrations=, or regenerate the baseline if the "
            f"identity is intentionally dropped."
        )


class BaselineRegressionError(ValueError):
    """Raised when :func:`write_baseline` would erase identities the
    existing baseline recorded.

    Overwriting them away makes :func:`detect_changes` / ``assert_covers``
    blind to a forgotten migration for those identities. Attribute
    ``removed`` is the tuple of dropped ``(module, qualname)`` identities
    so a custom CI reporter can format them however it wants.
    """

    def __init__(self, removed: tuple[tuple[str, str], ...]) -> None:
        self.removed = removed
        joined = ", ".join(f"{m}:{q}" for m, q in removed)
        plural = "y" if len(removed) == 1 else "ies"
        super().__init__(
            f"{len(removed)} identit{plural} in the existing baseline would "
            f"be erased by this write: {joined}. Add @migrate_from / "
            f"@backfill (or a Migration) covering them and regenerate, or "
            f"pass allow_removed=True if they are intentional deletes."
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
    model = graph.namespaces()
    for namespace in model.namespaces.values():
        for command in namespace.commands.values():
            yield (command.cls.__module__, command.cls.__qualname__)
            for outcome in command.outcomes:
                yield (outcome.__module__, outcome.__qualname__)
        for event in namespace.events:
            yield (event.__module__, event.__qualname__)
    for event in model.integration_events:
        yield (event.__module__, event.__qualname__)
    for event in model.system_events:
        yield (event.__module__, event.__qualname__)


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
