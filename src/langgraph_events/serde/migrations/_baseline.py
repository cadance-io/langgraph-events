"""The committed baseline: what earlier releases could have stored.

Version 4. A read accepts versions 1 to 4. A write always gives version 4.
:meth:`Baseline.updated` computes every value a write stores.
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from langgraph_events.serde.migrations._core import (
    MigrationKey,
    MigrationSet,
    migration_fingerprint,
)

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping
    from pathlib import Path

    from langgraph_events import EventGraph
    from langgraph_events.serde.migrations._target import Snapshot

Identity = tuple[str, str]
Fields = frozenset[str] | None
"""The init fields recorded for one identity. ``None`` when the record
predates version 3, which recorded no fields."""

BASELINE_VERSION = 4
_SUPPORTED_BASELINE_VERSIONS = (1, 2, 3, 4)


@dataclass(frozen=True)
class Hold:
    """Why a shipped migration stays one more release.

    ``against`` is the fingerprint of the previous baseline the hold was
    made against. A check against another previous baseline ignores the
    hold, so a hold lasts one release.
    """

    reason: str
    against: str


@dataclass(frozen=True)
class MigrationRecord:
    """One migration the baseline records, with its hold when one exists."""

    key: MigrationKey
    hold: Hold | None = None


@dataclass(frozen=True)
class MigrationsUnrecorded:
    """A baseline of version 1 to 3. It recorded no migration.

    This is not "no migration": the release rule cannot run against it.
    """

    version: int


@dataclass(frozen=True)
class Baseline:
    """The identities, fields, handlers and migrations of a baseline file."""

    events: Mapping[Identity, Fields]
    retired: Mapping[Identity, Fields]
    handlers: frozenset[str]
    migrations: tuple[MigrationRecord, ...] | MigrationsUnrecorded

    @classmethod
    def read(cls, path: Path) -> Baseline:
        """The baseline at *path*. A missing file is an empty version 4 baseline.

        Raises:
            ValueError: when the file has an unsupported version or shape.
        """
        if not path.exists():
            return cls({}, {}, frozenset(), ())
        raw: dict[str, Any] = json.loads(path.read_text())
        version = raw.get("version")
        if version not in _SUPPORTED_BASELINE_VERSIONS:
            raise ValueError(
                f"Unsupported baseline version {version!r} at {path}. This "
                f"library reads baseline versions {_SUPPORTED_BASELINE_VERSIONS}."
            )
        if version >= 3:
            _check_v3_shape(raw, path)
        return cls(
            events=_fields_by_identity(raw["events"]),
            retired=_fields_by_identity(raw.get("retired", [])),
            handlers=frozenset(entry["name"] for entry in raw.get("handlers", [])),
            migrations=(
                _read_migrations(raw.get("migrations", []), path)
                if version >= 4
                else MigrationsUnrecorded(version)
            ),
        )

    def write(self, path: Path) -> None:
        """Write this baseline to *path* as version 4."""
        payload = {
            "version": BASELINE_VERSION,
            "events": [_entry(i, f) for i, f in sorted(self.events.items())],
            "retired": [_entry(i, f) for i, f in sorted(self.retired.items())],
            "handlers": [{"name": name} for name in sorted(self.handlers)],
            "migrations": [
                _record_entry(record)
                for record in sorted(self.records, key=lambda r: r.key)
            ],
        }
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    @property
    def recorded_keys(self) -> frozenset[MigrationKey] | MigrationsUnrecorded:
        """The recorded migration keys, or the reason none are recorded."""
        if isinstance(self.migrations, MigrationsUnrecorded):
            return self.migrations
        return frozenset(record.key for record in self.migrations)

    @property
    def fingerprint(self) -> str | MigrationsUnrecorded:
        """The fingerprint of :attr:`recorded_keys`, as on a ``MigrationSet``."""
        keys = self.recorded_keys
        if isinstance(keys, MigrationsUnrecorded):
            return keys
        return migration_fingerprint(keys)

    def updated(self, code: Snapshot, previous: Baseline | None) -> Baseline:
        """The baseline the code now needs. Idempotent for one *previous*.

        ``events`` and ``retired`` are cumulative: a recorded field stays,
        and an identity the graph no longer reaches moves to ``retired``.
        One exception is a migration that *previous* recorded and the code
        no longer holds. The deploy rewrote every record it covered, so
        the history of its identity resets. See :meth:`_reset`.
        ``handlers`` is rebuilt. ``migrations`` is exactly the code's keys,
        and a hold stays on a key the code still holds.
        """
        live = _live_fields(code.graph)
        recorded = {**self.retired, **self.events}
        merged: dict[Identity, Fields] = {
            identity: fields | (recorded.get(identity) or frozenset())
            for identity, fields in live.items()
        }
        retired = {i: f for i, f in recorded.items() if i not in live}
        previous_keys = previous.recorded_keys if previous is not None else None
        if isinstance(previous_keys, frozenset):
            for key in code.migrations.deleted_since(previous_keys):
                _reset(key, previous_keys, code.migrations, live, merged, retired)
        holds = {record.key: record.hold for record in self.records}
        return Baseline(
            events=merged,
            retired=retired,
            handlers=frozenset(code.graph.handler_names),
            migrations=tuple(
                MigrationRecord(key, holds.get(key))
                for key in sorted(code.migrations.keys)
            ),
        )

    def held(self, key: MigrationKey, hold: Hold) -> Baseline:
        """This baseline with *hold* on the record of *key*.

        Raises:
            ValueError: when the baseline does not record *key*.
        """
        records = self.records
        if key not in {record.key for record in records}:
            raise ValueError(
                f"The baseline does not record {key.label!r}. Hold only a "
                f"migration the code holds, and run `baseline update` first."
            )
        return dataclasses.replace(
            self,
            migrations=tuple(
                MigrationRecord(record.key, hold if record.key == key else record.hold)
                for record in records
            ),
        )

    def changes_to(self, other: Baseline) -> tuple[str, ...]:
        """One line per difference from this baseline to *other*."""
        lines: list[str] = []
        for name, mine, theirs in (
            ("event", self.events, other.events),
            ("retired", self.retired, other.retired),
        ):
            for identity in sorted(mine.keys() | theirs.keys()):
                label = f"{name} {identity[0]}:{identity[1]}"
                if identity not in theirs:
                    lines.append(f"- {label}")
                elif identity not in mine:
                    lines.append(f"+ {label}")
                elif mine[identity] != theirs[identity]:
                    lines.append(
                        f"~ {label} fields {_listed(mine[identity])} -> "
                        f"{_listed(theirs[identity])}"
                    )
        for handler in sorted(self.handlers - other.handlers):
            lines.append(f"- handler {handler}")
        for handler in sorted(other.handlers - self.handlers):
            lines.append(f"+ handler {handler}")
        mine_keys, their_keys = self.recorded_keys, other.recorded_keys
        if isinstance(mine_keys, MigrationsUnrecorded):
            lines.append(f"~ version {mine_keys.version} -> {BASELINE_VERSION}")
            mine_keys = frozenset()
        if isinstance(their_keys, frozenset):
            lines.extend(
                f"- migration {k.label}" for k in sorted(mine_keys - their_keys)
            )
            lines.extend(
                f"+ migration {k.label}" for k in sorted(their_keys - mine_keys)
            )
        return tuple(lines)

    @property
    def records(self) -> tuple[MigrationRecord, ...]:
        """The migration records. Empty for a baseline that records none."""
        if isinstance(self.migrations, MigrationsUnrecorded):
            return ()
        return self.migrations


def _reset(
    key: MigrationKey,
    previous: frozenset[MigrationKey],
    code: MigrationSet,
    live: Mapping[Identity, frozenset[str]],
    events: dict[Identity, Fields],
    retired: dict[Identity, Fields],
) -> None:
    """Reset the history of the identity a deleted shipped *key* covered.

    A rename source, or an identity a previous rename read away, has no
    record left after the deploy: it leaves ``retired``. Any other identity
    keeps records in the shape of its live class only, so its fields are
    the live fields, read through a current rename when one exists.
    """
    identity = key.identity
    sources = {k.identity for k in previous if k.kind == "rename"}
    if key.kind == "rename" or identity in sources:
        retired.pop(identity, None)
        return
    terminus = code.renames.get(identity, identity)
    if terminus not in live:
        return
    if identity in events:
        events[identity] = live[terminus]
    elif identity in retired:
        retired[identity] = live[terminus]


def _live_fields(graph: EventGraph) -> dict[Identity, frozenset[str]]:
    """The init fields of every event class reachable from *graph*.

    Init fields only: the serde writes those, and an ``init=False`` field
    is rebuilt by the constructor.
    """
    return {
        (cls.__module__, cls.__qualname__): frozenset(
            f.name for f in dataclasses.fields(cls) if f.init
        )
        for cls in _event_classes(graph)
    }


def _event_classes(graph: EventGraph) -> Iterable[type]:
    """Commands, their outcomes, namespace events, integration and system
    events: the surface the serde encodes."""
    model = graph.namespaces()
    for namespace in model.namespaces.values():
        for command in namespace.commands.values():
            yield command.cls
            yield from command.outcomes
        yield from namespace.events
    yield from model.integration_events
    yield from model.system_events


def _listed(fields: Fields) -> str:
    return "unrecorded" if fields is None else f"[{', '.join(sorted(fields))}]"


def _entry(identity: Identity, fields: Fields) -> dict[str, Any]:
    entry: dict[str, Any] = {"module": identity[0], "qualname": identity[1]}
    if fields is not None:
        entry["fields"] = sorted(fields)
    return entry


def _record_entry(record: MigrationRecord) -> dict[str, Any]:
    entry: dict[str, Any] = {"label": record.key.label}
    if record.hold is not None:
        entry["hold"] = {"reason": record.hold.reason, "against": record.hold.against}
    return entry


def _read_migrations(
    entries: list[dict[str, Any]], path: Path
) -> tuple[MigrationRecord, ...]:
    records = []
    for entry in entries:
        try:
            key = MigrationKey.parse(entry["label"])
        except ValueError as error:
            raise ValueError(f"Baseline at {path}: {error}") from error
        hold = entry.get("hold")
        records.append(
            MigrationRecord(
                key, Hold(hold["reason"], hold["against"]) if hold else None
            )
        )
    return tuple(sorted(records, key=lambda r: r.key))


def _fields_by_identity(entries: list[dict[str, Any]]) -> dict[Identity, Fields]:
    return {
        (entry["module"], entry["qualname"]): (
            frozenset(entry["fields"]) if "fields" in entry else None
        )
        for entry in entries
    }


def _check_v3_shape(raw: dict[str, Any], path: Path) -> None:
    """Reject an ``events`` entry with no ``fields``, or an identity in both
    ``events`` and ``retired``.

    ``fields`` is mandatory from version 3. The writer never produces an
    overlap, so one means a hand edit went wrong.
    """
    events = raw["events"]
    incomplete = [e["qualname"] for e in events if "fields" not in e]
    if incomplete:
        raise ValueError(
            f"Baseline at {path} is version {raw['version']} but these events "
            f"entries have no `fields`: {', '.join(incomplete)}. Give each "
            f'entry "fields": [] and run `baseline update`. The update fills '
            f"the list from the live class."
        )
    live = {(e["module"], e["qualname"]) for e in events}
    retired = {(e["module"], e["qualname"]) for e in raw.get("retired", [])}
    overlap = sorted(live & retired)
    if overlap:
        joined = ", ".join(f"{m}:{q}" for m, q in overlap)
        raise ValueError(
            f"Baseline at {path} lists these identities under both `events` "
            f"and `retired`: {joined}. Remove it from `retired` when the class "
            f"is live, or from `events` when it is not."
        )
