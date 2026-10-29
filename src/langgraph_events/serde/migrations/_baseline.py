"""The committed baseline: what earlier releases could have stored.

Version 4. A read accepts versions 1 to 4. A write always gives version 4.
:meth:`Baseline.updated` computes every value a write stores.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from langgraph_events._identity import Identity
from langgraph_events.serde.migrations._core import (
    MigrationKey,
    MigrationSet,
    _rename_sources,
    migration_fingerprint,
)

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

    from langgraph_events.serde.migrations._target import Snapshot

Fields = frozenset[str] | None
"""The init fields recorded for one identity. ``None`` when the record
predates version 3, which recorded no fields."""

BASELINE_VERSION = 4
_SUPPORTED_BASELINE_VERSIONS = (1, 2, 3, 4)


@dataclass(frozen=True)
class Hold:
    """Why a shipped migration stays one more release.

    ``against`` is the :attr:`Baseline.digest` of the previous baseline
    the hold was made against. The next release's baseline holds this hold,
    so its digest differs, and a check against it ignores the hold. So a
    hold lasts one release, also when the migrations did not change.
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
    """A baseline of version 1 to 3. It recorded no migration and no live
    fields.

    This is not "no migration": the release rule cannot run against it,
    and an update against it drops no history.
    """

    version: int


@dataclass(frozen=True)
class CodeRecord:
    """What the code held when a version 4 baseline was written.

    ``migrations`` holds one record per client migration. ``live`` holds
    the init fields of each live event class. After a clean deploy, every
    record of a class is in its live shape, so ``live`` is the history
    that stays when a migration of the class leaves the code.
    """

    migrations: tuple[MigrationRecord, ...]
    live: Mapping[Identity, frozenset[str]]

    @property
    def keys(self) -> frozenset[MigrationKey]:
        return frozenset(record.key for record in self.migrations)

    def prune(
        self, history: Mapping[Identity, Fields], current: MigrationSet
    ) -> dict[Identity, Fields]:
        """*history* after the deploy of this code rewrote the store.

        Each migration of this code that *current* no longer holds resets
        the identity it keyed. A rename source, or an identity a rename of
        this code read away, has no record left: it leaves the history.
        Any other identity keeps the live fields of this code.
        """
        pruned = dict(history)
        sources = _rename_sources(self.keys)
        for key in current.deleted_since(self.keys):
            if key.kind == "rename" or key.identity in sources:
                pruned.pop(key.identity, None)
            elif key.identity in self.live:
                pruned[key.identity] = self.live[key.identity]
        return pruned


@dataclass(frozen=True)
class Baseline:
    """The identities, fields, handlers and migrations of a baseline file."""

    events: Mapping[Identity, Fields]
    retired: Mapping[Identity, Fields]
    handlers: frozenset[str]
    code: CodeRecord | MigrationsUnrecorded

    @classmethod
    def read(cls, path: Path) -> Baseline:
        """The baseline at *path*. A missing file is an empty version 4 baseline.

        Raises:
            ValueError: when the file has an unsupported version or shape.
        """
        if not path.exists():
            return cls({}, {}, frozenset(), CodeRecord((), {}))
        try:
            raw: object = json.loads(path.read_text())
        except json.JSONDecodeError as error:
            raise ValueError(
                f"Baseline at {path} is not valid JSON: {error}. Restore the "
                f"file from version control, or give the baseline file of the "
                f"release."
            ) from error
        if not isinstance(raw, dict):
            raise ValueError(
                f"Baseline at {path} is malformed: the file holds a JSON "
                f"{type(raw).__name__}, not an object. Restore the file from "
                f"version control, then run `baseline update`."
            )
        version = raw.get("version")
        if version not in _SUPPORTED_BASELINE_VERSIONS:
            raise ValueError(
                f"Unsupported baseline version {version!r} at {path}. This "
                f"library reads baseline versions {_SUPPORTED_BASELINE_VERSIONS}."
            )
        try:
            return cls._parse(raw, version, path)
        except (KeyError, TypeError) as error:
            raise ValueError(
                f"Baseline at {path} is malformed: {type(error).__name__} "
                f"{error}. Restore the file from version control, then run "
                f"`baseline update`."
            ) from error

    @classmethod
    def _parse(cls, raw: dict[str, Any], version: int, path: Path) -> Baseline:
        if version >= 3:
            _check_v3_shape(raw, path)
        if version >= 4:
            _check_v4_shape(raw, path)
        return cls(
            events=_fields_by_identity(raw["events"]),
            retired=_fields_by_identity(raw.get("retired", [])),
            handlers=frozenset(entry["name"] for entry in raw.get("handlers", [])),
            code=(
                CodeRecord(
                    _read_migrations(raw.get("migrations", []), path),
                    {
                        Identity(e["module"], e["qualname"]): frozenset(e["live"])
                        for e in raw["events"]
                    },
                )
                if version >= 4
                else MigrationsUnrecorded(version)
            ),
        )

    def write(self, path: Path) -> None:
        """Write this baseline to *path* as version 4."""
        path.write_text(json.dumps(self._payload(), indent=2, sort_keys=True) + "\n")

    @property
    def digest(self) -> str:
        """``lge-b1-`` and 16 hex digits of the SHA-256 of the content.

        Two files with the same content give the same digest, whatever
        their whitespace or key order.
        """
        content = json.dumps(self._payload(), sort_keys=True, separators=(",", ":"))
        return "lge-b1-" + hashlib.sha256(content.encode()).hexdigest()[:16]

    def _payload(self) -> dict[str, Any]:
        """The version 4 JSON content.

        Raises:
            TypeError: when the baseline records no code. Only a read of a
                version 1 to 3 file gives one, and no write path keeps it.
        """
        if isinstance(self.code, MigrationsUnrecorded):
            raise TypeError("A baseline of version 1 to 3 is never written.")
        live = self.code.live
        return {
            "version": BASELINE_VERSION,
            "events": [
                {**_entry(i, f), "live": sorted(live[i])}
                for i, f in sorted(self.events.items())
            ],
            "retired": [_entry(i, f) for i, f in sorted(self.retired.items())],
            "handlers": [{"name": name} for name in sorted(self.handlers)],
            "migrations": [
                _record_entry(record)
                for record in sorted(self.records, key=lambda r: r.key)
            ],
        }

    @property
    def recorded_keys(self) -> frozenset[MigrationKey] | MigrationsUnrecorded:
        """The recorded migration keys, or the reason none are recorded."""
        if isinstance(self.code, MigrationsUnrecorded):
            return self.code
        return self.code.keys

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
        and an identity the serde no longer speaks for moves to
        ``retired``. One exception is a migration that *previous* recorded
        and the code no longer holds. The deploy of *previous* rewrote every
        record it covered, so the history of its identity resets. See
        :meth:`CodeRecord.prune`. ``handlers`` is rebuilt. ``migrations``
        is exactly the code's keys, and a hold stays on a key the code
        still holds.
        """
        live = code.live_fields()
        history = self.history
        if previous is not None and isinstance(previous.code, CodeRecord):
            history = previous.code.prune(history, code.migrations)
        holds = {record.key: record.hold for record in self.records}
        return Baseline(
            events={
                identity: fields | (history.get(identity) or frozenset())
                for identity, fields in live.items()
            },
            retired={i: f for i, f in history.items() if i not in live},
            handlers=frozenset(code.graph.handler_names),
            code=CodeRecord(
                tuple(
                    MigrationRecord(key, holds.get(key))
                    for key in sorted(code.migrations.keys)
                ),
                live,
            ),
        )

    def held(self, key: MigrationKey, hold: Hold) -> Baseline:
        """This baseline with *hold* on the record of *key*.

        Raises:
            ValueError: when the baseline does not record *key*.
        """
        code = self.code
        if isinstance(code, MigrationsUnrecorded) or key not in code.keys:
            raise ValueError(
                f"The baseline does not record {key.label!r}. Hold only a "
                f"migration the code holds, and run `baseline update` first."
            )
        return dataclasses.replace(
            self,
            code=CodeRecord(
                tuple(
                    MigrationRecord(
                        record.key, hold if record.key == key else record.hold
                    )
                    for record in code.migrations
                ),
                code.live,
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
                label = f"{name} {identity.label}"
                if identity not in theirs:
                    lines.append(f"- {label}")
                elif identity not in mine:
                    lines.append(f"+ {label}")
                elif mine[identity] != theirs[identity]:
                    lines.append(
                        f"~ {label} fields {_listed(mine[identity])} -> "
                        f"{_listed(theirs[identity])}"
                    )
        if isinstance(self.code, CodeRecord) and isinstance(other.code, CodeRecord):
            for identity in sorted(self.code.live.keys() & other.code.live.keys()):
                mine_live, their_live = (
                    self.code.live[identity],
                    other.code.live[identity],
                )
                if mine_live != their_live:
                    lines.append(
                        f"~ event {identity.label} live "
                        f"{_listed(mine_live)} -> {_listed(their_live)}"
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
    def history(self) -> Mapping[Identity, Fields]:
        """Every recorded identity, live or retired, with its fields."""
        return {**self.retired, **self.events}

    @property
    def records(self) -> tuple[MigrationRecord, ...]:
        """The migration records. Empty for a baseline that records none."""
        if isinstance(self.code, MigrationsUnrecorded):
            return ()
        return self.code.migrations


def _listed(fields: Fields) -> str:
    return "unrecorded" if fields is None else f"[{', '.join(sorted(fields))}]"


def _entry(identity: Identity, fields: Fields) -> dict[str, Any]:
    entry: dict[str, Any] = {"module": identity.module, "qualname": identity.qualname}
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
    labels: set[str] = set()
    for entry in entries:
        label = entry["label"]
        if label in labels:
            raise ValueError(
                f"Baseline at {path} lists the migration {label!r} twice. "
                f"Delete one entry, then run `baseline update`."
            )
        labels.add(label)
        try:
            key = MigrationKey.parse(label)
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
        Identity(entry["module"], entry["qualname"]): (
            frozenset(entry["fields"]) if "fields" in entry else None
        )
        for entry in entries
    }


def _check_v4_shape(raw: dict[str, Any], path: Path) -> None:
    """Reject an ``events`` entry with no ``live``. It is mandatory from
    version 4: a prune against the file reads it."""
    incomplete = [e["qualname"] for e in raw["events"] if "live" not in e]
    if incomplete:
        raise ValueError(
            f"Baseline at {path} is version 4 but these events entries have "
            f"no `live`: {', '.join(incomplete)}. Give each entry "
            f'"live": [] and run `baseline update`. The update fills the list '
            f"from the live class."
        )


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
    live = {Identity(e["module"], e["qualname"]) for e in events}
    retired = {Identity(e["module"], e["qualname"]) for e in raw.get("retired", [])}
    overlap = sorted(live & retired)
    if overlap:
        joined = ", ".join(identity.label for identity in overlap)
        raise ValueError(
            f"Baseline at {path} lists these identities under both `events` "
            f"and `retired`: {joined}. Remove it from `retired` when the class "
            f"is live, or from `events` when it is not."
        )
