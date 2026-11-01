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
from typing import TYPE_CHECKING, Any, Literal

from langgraph_events.serde.migrations._target import TargetError

if TYPE_CHECKING:
    from collections.abc import Iterator

    from langgraph.checkpoint.base import BaseCheckpointSaver
    from langgraph.checkpoint.postgres import PostgresSaver
    from psycopg import Cursor, sql
    from psycopg.rows import DictRow

    from langgraph_events.serde.migrations._target import MigrationTarget

_TAG = re.compile(r"[a-z0-9_]{1,32}")

_OTHER_BACKUPS = """
SELECT relname FROM pg_class
WHERE relkind = 'r'
  AND relnamespace = (SELECT oid FROM pg_namespace WHERE nspname = current_schema())
  AND relname ~ '^checkpoint(s|_writes)_bak_[a-z0-9_]{1,32}$'
  AND relname <> ALL(%(keep)s)
"""

_COPY_CHECKPOINTS = """
INSERT INTO {checkpoints}
SELECT * FROM (
    SELECT DISTINCT ON (thread_id, checkpoint_ns) * FROM checkpoints
    ORDER BY thread_id, checkpoint_ns, checkpoint_id DESC
) AS latest
ON CONFLICT DO NOTHING
"""

_COPY_WRITES = """
INSERT INTO {writes}
SELECT w.* FROM checkpoint_writes AS w
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


class NoBackupError(TargetError):
    """The store holds no backup tables for the tag that the restore names."""


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
class _Backup:
    """The two backup tables of one tag, in the current schema of the store."""

    tag: str

    @classmethod
    def of(cls, tag: str) -> _Backup:
        """Raises ``ValueError`` unless *tag* is 1 to 32 characters of
        ``a-z``, ``0-9`` and ``_``."""
        if not _TAG.fullmatch(tag):
            raise ValueError(
                f"The backup tag {tag!r} is not valid. Give 1 to 32 characters "
                f"of a-z, 0-9 and _."
            )
        return cls(tag)

    @property
    def checkpoints(self) -> str:
        return f"checkpoints_bak_{self.tag}"

    @property
    def writes(self) -> str:
        return f"checkpoint_writes_bak_{self.tag}"

    def take(self, saver: BaseCheckpointSaver[Any]) -> BackupCounts:
        """Copy the live set into the two tables, in one transaction, and
        drop the backup tables of every other tag.

        A second backup with the same tag keeps each row it holds, and adds
        each new latest row. On an error, the transaction writes nothing.
        """
        postgres = _postgres_saver(saver, "backup")
        import psycopg  # noqa: PLC0415
        from psycopg import sql  # noqa: PLC0415

        try:
            with _transaction(postgres) as cursor:
                cursor.execute(
                    _OTHER_BACKUPS, {"keep": [self.checkpoints, self.writes]}
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
                cursor.execute(self._sql(_COPY_CHECKPOINTS))
                cursor.execute(self._sql(_COPY_WRITES))
                cursor.execute(self._sql(_COUNT))
                counts = cursor.fetchone()
        except psycopg.Error as error:
            raise TargetError(
                f"The backup {self.tag} failed, so the rewrite did not run and "
                f"the store is unchanged: {_one_line(error)}"
            ) from error
        assert counts is not None  # noqa: S101 - a SELECT of two counts returns one row
        return BackupCounts(counts["checkpoints"], counts["writes"])

    def restore(self, saver: BaseCheckpointSaver[Any]) -> BackupCounts:
        """Put back each backed-up row whose key the store still holds, in
        one transaction.

        Raises:
            NoBackupError: when the store holds no backup tables of the tag.
            TargetError: when the restore fails. It then changes nothing.
        """
        postgres = _postgres_saver(saver, "restore")
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
                    raise NoBackupError(
                        f"The store holds no backup {self.tag}: no {self.checkpoints} "
                        f"and {self.writes}. Nothing changed."
                    )
                cursor.execute(self._sql(_RESTORE))
                counts = cursor.fetchone()
        except psycopg.Error as error:
            raise TargetError(
                f"The restore of the backup {self.tag} failed and changed "
                f"nothing: {_one_line(error)}"
            ) from error
        assert counts is not None  # noqa: S101 - a SELECT of two counts returns one row
        return BackupCounts(counts["checkpoints"], counts["writes"])

    def _sql(self, template: str) -> sql.Composed:
        from psycopg import sql  # noqa: PLC0415

        return sql.SQL(template).format(
            checkpoints=sql.Identifier(self.checkpoints),
            writes=sql.Identifier(self.writes),
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
        NoBackupError: when the store holds no backup of *tag*.
        TargetError: when the store is not a ``PostgresSaver``, or when the
            restore fails. A failed restore changes nothing.
    """
    backup = _Backup.of(tag)
    with target.open_store() as saver:
        return backup.restore(saver)


@contextlib.contextmanager
def _transaction(saver: PostgresSaver) -> Iterator[Cursor[DictRow]]:
    """A cursor of *saver* inside one transaction.

    ``PostgresSaver._cursor()`` is private to the saver package. It takes
    the lock of the saver, and a connection of its pool when the saver
    holds a pool.
    """
    with saver._cursor() as cursor, cursor.connection.transaction():
        yield cursor


_OTHER_SAVER_REMEDY = {
    "backup": (
        "Back up the store with the tools of its database, then run "
        "rewrite --apply without --backup."
    ),
    "restore": "Restore the store with the tools of its database.",
}


def _postgres_saver(
    saver: BaseCheckpointSaver[Any], action: Literal["backup", "restore"]
) -> PostgresSaver:
    """*saver*, when it is a ``PostgresSaver``. The saver is an optional
    dependency, so its import is local."""
    try:
        from langgraph.checkpoint.postgres import PostgresSaver  # noqa: PLC0415
    except ImportError:
        pass
    else:
        if isinstance(saver, PostgresSaver):
            return saver
    raise TargetError(
        f"The {action} needs a PostgresSaver, and the saver of the store is "
        f"{type(saver).__name__}. {_OTHER_SAVER_REMEDY[action]}"
    )


def _one_line(error: BaseException) -> str:
    return f"{type(error).__name__}: {' '.join(str(error).split())}"
