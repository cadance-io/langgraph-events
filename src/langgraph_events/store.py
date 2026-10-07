"""A durable event log beside ``EventGraph``, with no checkpointer.

The log is the save file. An :class:`EventStore` keeps one JSON record per
event. An :class:`EventCodec` turns events into records and back, through the
serde migration tables. An :class:`EventStream` runs a graph over the stored
history and appends each superstep's new events before the next superstep
runs. A load rebuilds the log from the store and runs no handler.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, BinaryIO, Protocol

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path
    from types import TracebackType


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
