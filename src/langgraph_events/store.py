"""A durable event log beside ``EventGraph``, with no checkpointer.

The log is the save file. An :class:`EventStore` keeps one JSON record per
event. An :class:`EventCodec` turns events into records and back, through the
serde migration tables. An :class:`EventStream` runs a graph over the stored
history and appends each superstep's new events before the next superstep
runs. A load rebuilds the log from the store and runs no handler.
"""

from __future__ import annotations

import contextlib
import dataclasses
import json
import os
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, BinaryIO, Protocol, cast

from langgraph_events._event import Event
from langgraph_events._event_log import EventLog
from langgraph_events.serde import NamespaceAwareSerde

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
    from pathlib import Path
    from types import TracebackType

    from langgraph_events._event import Namespace
    from langgraph_events.serde import Migration, UnrevivedIdentity


def _reject_json_constant(constant: str) -> object:
    raise ValueError(f"not valid JSON: {constant}")


@dataclass(frozen=True)
class Record:
    """One stored event: its class identity and its field values.

    *fields* holds JSON values only. A link to an earlier event in the same
    log is ``{"$ref": index}``. See ``docs/event-store.md`` for every marker.
    """

    module: str
    type: str
    fields: Mapping[str, object]

    def to_json(self) -> str:
        """The record as one line of JSON, with no newline."""
        return json.dumps(
            {"module": self.module, "type": self.type, "fields": dict(self.fields)},
            ensure_ascii=False,
            allow_nan=False,
        )

    @classmethod
    def from_json(cls, text: str | bytes) -> Record:
        """Parse one line written by :meth:`to_json`.

        Raises ``ValueError`` when the line is not a record object.
        """
        row = json.loads(text, parse_constant=_reject_json_constant)
        if not (
            isinstance(row, dict)
            and isinstance(row.get("module"), str)
            and isinstance(row.get("type"), str)
            and isinstance(row.get("fields"), dict)
        ):
            raise ValueError(
                f"not an event record: {text!r}. A record is an object with "
                f"string 'module', string 'type' and object 'fields'."
            )
        return cls(row["module"], row["type"], row["fields"])


class EventStore(Protocol):
    """The port: an append-only sequence of records."""

    def append(self, records: Sequence[Record]) -> None:
        """Make *records* durable, in order, after every earlier record."""

    def load(self) -> list[Record]:
        """Every durable record, in order."""


class MemoryEventStore:
    """An in-memory store. It keeps JSON text, so it also proves the codec."""

    def __init__(self) -> None:
        self._lines: list[str] = []

    def append(self, records: Sequence[Record]) -> None:
        """Keep *records* as JSON lines."""
        self._lines.extend(record.to_json() for record in records)

    def load(self) -> list[Record]:
        """Parse every kept line."""
        return [Record.from_json(line) for line in self._lines]


class StoreLockedError(RuntimeError):
    """Another :class:`JsonlEventStore` holds the lock on this log."""


class JsonlEventStore:
    """One JSON record per line in *path*. POSIX only: it uses ``flock``.

    The store takes an exclusive lock on ``<path>.lock`` for its lifetime,
    so one writer owns the log. A line counts only once its newline is on
    disk. Load ignores a last line with no newline, which a crash leaves.
    The first append after that truncates the torn line.
    """

    def __init__(self, path: Path) -> None:
        import fcntl  # noqa: PLC0415

        self._path = path
        self._file: BinaryIO | None = None
        lock_path = path.with_name(path.name + ".lock")
        self._lock: BinaryIO | None = lock_path.open("ab")
        try:
            fcntl.flock(self._lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            self._lock.close()
            self._lock = None
            raise StoreLockedError(
                f"{path} has another writer: {lock_path} is locked. Close the "
                f"other JsonlEventStore, or stop the process that holds it."
            ) from exc

    def load(self) -> list[Record]:
        """Every line that ends with a newline, parsed. A torn last line is
        ignored.

        Raises ``ValueError`` naming the line number when a complete line is
        not a record.
        """
        try:
            data = self._path.read_bytes()
        except FileNotFoundError:
            return []
        records = []
        for number, line in enumerate(data.split(b"\n")[:-1], start=1):
            try:
                records.append(Record.from_json(line))
            except ValueError as exc:
                raise ValueError(f"{self._path}:{number}: {exc}") from exc
        return records

    def append(self, records: Sequence[Record]) -> None:
        """Write *records*, flush, and ``fsync`` before returning."""
        if not records:
            return
        if self._file is None:
            self._file = self._open_for_append()
        self._file.write(b"".join(r.to_json().encode() + b"\n" for r in records))
        self._file.flush()
        os.fsync(self._file.fileno())

    def _open_for_append(self) -> BinaryIO:
        """Open the log and cut a torn last line, so the next record starts
        on its own line."""
        created = not self._path.exists()
        file = self._path.open("a+b")
        size = file.seek(0, os.SEEK_END)
        if size:
            file.seek(0)
            keep = file.read().rfind(b"\n") + 1
            if keep != size:
                file.truncate(keep)
        if created:
            directory = os.open(self._path.parent, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        return file

    def close(self) -> None:
        """Close the log and release the lock. A second call does nothing."""
        if self._file is not None:
            self._file.close()
            self._file = None
        if self._lock is not None:
            self._lock.close()
            self._lock = None

    def __enter__(self) -> JsonlEventStore:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()


_MARKERS = frozenset({"$ref", "$tuple", "$dict", "$repr"})
"""The single keys that make a stored JSON object a marker, not a dict."""

_NO_REPLAY: Mapping[type[Event], Callable[[Event], Iterable[type[Event]]]] = (
    MappingProxyType({})
)


class EventCodec:
    """Events to records and back, for one log.

    The codec remembers the position of every event it encodes or decodes,
    so a field that holds an earlier event of the log is stored as
    ``{"$ref": index}`` and revives as that same object. Use one codec per
    log.

    Decode passes each record through the serde migration tables
    (:meth:`NamespaceAwareSerde.revive_event`). *replay* maps an event type
    to a function that returns the classes that event defines. The codec
    calls it as soon as such an event decodes, and registers each class
    before the next record decodes. The function must be the factory the
    live handler calls. It runs no handler.
    """

    def __init__(
        self,
        migrations: Sequence[Migration] = (),
        *,
        namespaces: Sequence[type[Namespace]] = (),
        events: Sequence[type[Event]] = (),
        replay: Mapping[
            type[Event], Callable[[Event], Iterable[type[Event]]]
        ] = _NO_REPLAY,
    ) -> None:
        self._serde = NamespaceAwareSerde(
            migrations, namespaces=namespaces, events=events
        )
        self._replay = dict(replay)
        self._registry: dict[tuple[str, str], type[Event]] = {}
        self._book: list[Event | UnrevivedIdentity] = []
        self._index_of: dict[int, int] = {}

    def register(self, cls: type[Event]) -> None:
        """Make *cls* revivable by its identity, ahead of the import walk."""
        self._registry[(cls.__module__, cls.__qualname__)] = cls

    @contextlib.contextmanager
    def tolerate_unresolved(self) -> Iterator[list[UnrevivedIdentity]]:
        """Decode an unrevivable record to an ``UnrevivedIdentity``.

        Delegates to :meth:`NamespaceAwareSerde.tolerate_unresolved`. Yields
        the collector. A reader uses this to inspect a log whose classes are
        gone. Do not build an :class:`EventStream` over such a codec.
        """
        with self._serde.tolerate_unresolved() as missing:
            yield missing

    def encode(self, events: Sequence[Event]) -> list[Record]:
        """One record per event. Each event takes the next log position."""
        base = len(self._book)
        fresh: dict[int, int] = {}
        records = []
        for offset, event in enumerate(events):
            cls = type(event)
            records.append(
                Record(
                    cls.__module__,
                    cls.__qualname__,
                    {
                        field.name: self._dump(getattr(event, field.name), fresh)
                        for field in dataclasses.fields(cast("Any", event))
                    },
                )
            )
            fresh[id(event)] = base + offset
        self._book.extend(events)
        self._index_of.update(fresh)
        return records

    def decode(self, records: Sequence[Record]) -> EventLog:
        """Revive *records* as the next events of the log.

        Raises ``ValueError`` naming the record position and identity when a
        record does not revive, a ``$ref`` does not point to an earlier
        event, or a replay function raises. The codec then keeps no event
        of this call.
        """
        base = len(self._book)
        try:
            for index, record in enumerate(records, start=base):
                self._book.append(self._decode_one(index, record))
        except BaseException:
            del self._book[base:]
            raise
        decoded = self._book[base:]
        self._index_of.update(
            (id(event), base + index) for index, event in enumerate(decoded)
        )
        return EventLog._from_owned(tuple(decoded))

    def _decode_one(self, index: int, record: Record) -> Event | UnrevivedIdentity:
        identity = f"{record.module}.{record.type}"
        try:
            kwargs = {
                key: self._load(value, index) for key, value in record.fields.items()
            }
            event = self._serde.revive_event(
                record.module, record.type, kwargs, resolve=self._resolve
            )
            for kind, replay in self._replay.items():
                if isinstance(event, kind):
                    for cls in replay(event):
                        self.register(cls)
        except Exception as exc:
            raise ValueError(f"record #{index} {identity}: {exc}") from exc
        return event

    def _resolve(self, module: str, qualname: str) -> type[Event] | None:
        return self._registry.get((module, qualname))

    def _dump(self, value: object, fresh: Mapping[int, int]) -> object:
        if value is None or isinstance(value, bool | int | float | str):
            return value
        if isinstance(value, Event):
            index = self._index_of.get(id(value), fresh.get(id(value)))
            if index is not None:
                return {"$ref": index}
        if isinstance(value, list | tuple):
            items = [self._dump(item, fresh) for item in value]
            return items if isinstance(value, list) else {"$tuple": items}
        if isinstance(value, dict) and all(isinstance(key, str) for key in value):
            dumped = {key: self._dump(item, fresh) for key, item in value.items()}
            escaped = any(key.startswith("$") for key in value)
            return {"$dict": dumped} if escaped else dumped
        return {"$repr": repr(value)}

    def _load(self, value: object, index: int) -> object:
        if isinstance(value, list):
            return [self._load(item, index) for item in value]
        if not isinstance(value, dict):
            return value
        if len(value) == 1 and next(iter(value)) in _MARKERS:
            return self._load_marker(*next(iter(value.items())), index)
        return {key: self._load(item, index) for key, item in value.items()}

    def _load_marker(self, marker: str, body: object, index: int) -> object:
        if marker == "$ref":
            if isinstance(body, bool) or not isinstance(body, int):
                raise ValueError(f"$ref {body!r} is not an integer")
            if not 0 <= body < index:
                raise ValueError(
                    f"$ref {body} must point to an earlier event, below {index}"
                )
            return self._book[body]
        if marker == "$tuple" and isinstance(body, list):
            return tuple(self._load(item, index) for item in body)
        if marker == "$dict" and isinstance(body, dict):
            return {key: self._load(item, index) for key, item in body.items()}
        if marker == "$repr" and isinstance(body, str):
            return body
        raise ValueError(f"{marker} holds {type(body).__name__}")
