"""The backup of the live set of a ``PostgresSaver`` store, and its restore.

The live set is the latest checkpoint of each thread in each checkpoint
namespace, and the pending writes of that checkpoint. The rewrite
overwrites only these rows. It stores each changed channel under a new
blob version and keeps the old version, so the blobs need no copy.

A backup through ``get_tuple()`` cannot work: the serde migrates each
record on read. So the backup copies the stored rows, in SQL, through
the connection of the saver.
"""

from __future__ import annotations

import contextlib
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from langgraph_events.serde.migrations._target import TargetError, _postgres_saver

if TYPE_CHECKING:
    from collections.abc import Iterator

    from langgraph.checkpoint.base import BaseCheckpointSaver
    from langgraph.checkpoint.postgres import PostgresSaver
    from psycopg import Cursor, sql
    from psycopg.rows import DictRow

    from langgraph_events.serde.migrations._target import MigrationTarget

_TAG = re.compile(r"[a-z0-9_]{1,32}")

_BACKUP_TABLES = """
SELECT relname FROM pg_class
WHERE relkind = 'r'
  AND relnamespace = (SELECT oid FROM pg_namespace WHERE nspname = current_schema())
  AND relname ~ '^checkpoint(s|_writes)_bak_[a-z0-9_]{1,32}$'
  AND relname <> ALL(%(keep)s)
"""

_SHARED_COLUMNS = """
SELECT b.attname FROM pg_attribute AS b
JOIN pg_attribute AS l
  ON l.attrelid = %(live)s::regclass AND l.attname = b.attname
  AND l.attnum > 0 AND NOT l.attisdropped
WHERE b.attrelid = %(backup)s::regclass AND b.attnum > 0 AND NOT b.attisdropped
ORDER BY b.attnum
"""

_COPY_CHECKPOINTS = """
INSERT INTO {checkpoints} ({columns})
SELECT {columns} FROM (
    SELECT DISTINCT ON (thread_id, checkpoint_ns) * FROM checkpoints
    ORDER BY thread_id, checkpoint_ns, checkpoint_id DESC
) AS latest
ON CONFLICT DO NOTHING
"""

_COPY_WRITES = """
INSERT INTO {writes} ({columns})
SELECT {w_columns} FROM checkpoint_writes AS w
JOIN {checkpoints} USING (thread_id, checkpoint_ns, checkpoint_id)
ON CONFLICT DO NOTHING
"""

_COUNT = """
SELECT (SELECT count(*) FROM {checkpoints}) AS checkpoints,
       (SELECT count(*) FROM {writes}) AS writes
"""

_RESTORE = """
WITH restored_checkpoints AS (
    UPDATE checkpoints AS t SET checkpoint = b.checkpoint, metadata = b.metadata
    FROM {checkpoints} AS b
    WHERE (t.thread_id, t.checkpoint_ns, t.checkpoint_id)
        = (b.thread_id, b.checkpoint_ns, b.checkpoint_id)
    RETURNING 1
), restored_writes AS (
    UPDATE checkpoint_writes AS t SET channel = b.channel, type = b.type, blob = b.blob
    FROM {writes} AS b
    WHERE (t.thread_id, t.checkpoint_ns, t.checkpoint_id, t.task_id, t.idx)
        = (b.thread_id, b.checkpoint_ns, b.checkpoint_id, b.task_id, b.idx)
    RETURNING 1
)
SELECT (SELECT count(*) FROM restored_checkpoints) AS checkpoints,
       (SELECT count(*) FROM restored_writes) AS writes
"""

_BACKUP_REMEDY = (
    "Back up the store with the tools of its database, then run rewrite "
    "without --backup."
)
_RESTORE_REMEDY = "Restore the store with the tools of its database."


@dataclass(frozen=True)
class BackupCounts:
    """The rows that a backup holds, or that a restore put back."""

    checkpoints: int
    writes: int

    def __str__(self) -> str:
        """``1 checkpoint and 2 writes``."""
        checkpoints = "checkpoint" if self.checkpoints == 1 else "checkpoints"
        writes = "write" if self.writes == 1 else "writes"
        return f"{self.checkpoints} {checkpoints} and {self.writes} {writes}"


@dataclass(frozen=True)
class _Taken:
    """The rows that a backup held before this run, and after it."""

    kept: BackupCounts
    total: BackupCounts

    @property
    def added(self) -> BackupCounts:
        return BackupCounts(
            self.total.checkpoints - self.kept.checkpoints,
            self.total.writes - self.kept.writes,
        )


@dataclass(frozen=True)
class _Backup:
    """The two backup tables of one tag, in the current schema of the store.

    Raises ``ValueError`` unless the tag is 1 to 32 characters of ``a-z``,
    ``0-9`` and ``_``.
    """

    tag: str

    def __post_init__(self) -> None:
        if not _TAG.fullmatch(self.tag):
            raise ValueError(
                f"The backup tag {self.tag!r} is not valid. Give 1 to 32 "
                f"characters of a-z, 0-9 and _."
            )

    @property
    def checkpoints(self) -> str:
        return f"checkpoints_bak_{self.tag}"

    @property
    def writes(self) -> str:
        return f"checkpoint_writes_bak_{self.tag}"

    def take(self, saver: BaseCheckpointSaver[Any]) -> _Taken:
        """Copy the live set into the two tables, in one transaction, and
        drop the backup tables of every other tag.

        A second backup with the same tag keeps each row it holds, and adds
        each new latest row. On an error, the transaction writes nothing.
        """
        postgres = _require_postgres(saver, "backup", _BACKUP_REMEDY)
        import psycopg  # noqa: PLC0415
        from psycopg import sql  # noqa: PLC0415

        try:
            with _transaction(postgres) as cursor:
                cursor.execute(
                    _BACKUP_TABLES, {"keep": [self.checkpoints, self.writes]}
                )
                for row in cursor.fetchall():
                    cursor.execute(
                        sql.SQL("DROP TABLE {}").format(sql.Identifier(row["relname"]))
                    )
                cursor.execute(
                    self._sql(
                        "CREATE TABLE IF NOT EXISTS {checkpoints} "
                        "(LIKE checkpoints INCLUDING ALL)"
                    )
                )
                cursor.execute(
                    self._sql(
                        "CREATE TABLE IF NOT EXISTS {writes} "
                        "(LIKE checkpoint_writes INCLUDING ALL)"
                    )
                )
                kept = _counts(cursor, self._sql(_COUNT))
                columns = _shared_columns(cursor, "checkpoints", self.checkpoints)
                cursor.execute(self._sql(_COPY_CHECKPOINTS, columns))
                columns = _shared_columns(cursor, "checkpoint_writes", self.writes)
                cursor.execute(self._sql(_COPY_WRITES, columns))
                total = _counts(cursor, self._sql(_COUNT))
        except psycopg.Error as error:
            raise TargetError(
                f"The backup {self.tag} failed, so the rewrite did not run and "
                f"the store is unchanged: {_one_line(error)}"
            ) from error
        return _Taken(kept, total)

    def restore(self, saver: BaseCheckpointSaver[Any]) -> BackupCounts:
        """Put back each backed-up row whose key the store still holds, in
        one transaction.

        Raises:
            TargetError: when the store holds no backup tables of the tag,
                or when the restore fails. It then changes nothing.
        """
        postgres = _require_postgres(saver, "restore", _RESTORE_REMEDY)
        import psycopg  # noqa: PLC0415

        try:
            with _transaction(postgres) as cursor:
                cursor.execute(
                    "SELECT to_regclass(%s) IS NOT NULL "
                    "AND to_regclass(%s) IS NOT NULL AS found",
                    (self.checkpoints, self.writes),
                )
                found = cursor.fetchone()
                if found is None or not found["found"]:
                    raise TargetError(self._missing(cursor))
                restored = _counts(cursor, self._sql(_RESTORE))
        except psycopg.Error as error:
            raise TargetError(
                f"The restore of the backup {self.tag} failed and changed "
                f"nothing: {_one_line(error)}"
            ) from error
        return restored

    def _missing(self, cursor: Cursor[DictRow]) -> str:
        """The message of a restore that finds no backup of the tag. It
        names the tag of each backup that the store holds."""
        cursor.execute(_BACKUP_TABLES, {"keep": []})
        tags = sorted(
            {row["relname"].split("_bak_", 1)[1] for row in cursor.fetchall()}
        )
        held = (
            f"The store holds the backup {', '.join(tags)}"
            if tags
            else "The store holds no backup"
        )
        return (
            f"The store holds no backup {self.tag}: no {self.checkpoints} and "
            f"{self.writes}. {held}. Nothing changed."
        )

    def _sql(self, template: str, columns: list[str] | None = None) -> sql.Composed:
        """*template* with the two table names, with *columns* as
        ``{columns}``, and with *columns* qualified by ``w`` as
        ``{w_columns}``."""
        from psycopg import sql  # noqa: PLC0415

        names = columns or []
        return sql.SQL(template).format(
            checkpoints=sql.Identifier(self.checkpoints),
            writes=sql.Identifier(self.writes),
            columns=sql.SQL(", ").join(map(sql.Identifier, names)),
            w_columns=sql.SQL(", ").join(sql.Identifier("w", n) for n in names),
        )


def restore_live_set(target: MigrationTarget, tag: str) -> BackupCounts:
    """Put back the backup *tag* of the live set into the store of *target*.

    The restore runs in one transaction. It overwrites each checkpoint row
    and each pending write that the backup holds and the store still holds
    under the same key. A checkpoint that a run wrote after the backup has
    a new key, so the restore leaves it. The restored rows name the blob
    versions of the backup.

    Raises:
        ValueError: when *tag* is not 1 to 32 characters of ``a-z``,
            ``0-9`` and ``_``.
        TargetError: when *target* sets ``threads``, when the store is not
            a ``PostgresSaver``, when the store holds no backup of *tag*,
            or when the restore fails. The message of a missing backup
            names each backup that the store holds. A failed restore
            changes nothing.
    """
    backup = _Backup(tag)
    _require_whole_store(target, "restore", _RESTORE_REMEDY)
    with target.open_store() as saver:
        return backup.restore(saver)


def _require_whole_store(target: MigrationTarget, action: str, remedy: str) -> None:
    """Refuse a target that sets ``threads``. That target declares a store
    that other graphs can share, and a backup or a restore acts on every
    row of the store."""
    if target.threads is not None:
        raise TargetError(
            f"The {action} acts on the whole store, and the target sets "
            f"threads=, so other graphs can share the store. The {action} "
            f"needs a store that one graph owns. {remedy}"
        )


@contextlib.contextmanager
def _transaction(saver: PostgresSaver) -> Iterator[Cursor[DictRow]]:
    """A cursor of *saver* inside one transaction."""
    with saver._cursor() as cursor, cursor.connection.transaction():
        yield cursor


def _shared_columns(cursor: Cursor[DictRow], live: str, backup: str) -> list[str]:
    """The columns that the table *backup* and the table *live* share, in
    the order of *backup*. A copy that names them still runs after an
    upgrade of the saver adds a column to *live*."""
    cursor.execute(_SHARED_COLUMNS, {"live": live, "backup": backup})
    return [row["attname"] for row in cursor.fetchall()]


def _counts(cursor: Cursor[DictRow], query: sql.Composed) -> BackupCounts:
    cursor.execute(query)
    row = cursor.fetchone()
    assert row is not None  # noqa: S101 - a SELECT of two counts returns one row
    return BackupCounts(row["checkpoints"], row["writes"])


def _require_postgres(
    saver: BaseCheckpointSaver[Any], action: str, remedy: str
) -> PostgresSaver:
    postgres = _postgres_saver(saver)
    if postgres is None:
        raise TargetError(
            f"The {action} needs a PostgresSaver, and the saver of the store "
            f"is {type(saver).__name__}. {remedy}"
        )
    return postgres


def _one_line(error: BaseException) -> str:
    return f"{type(error).__name__}: {' '.join(str(error).split())}"
