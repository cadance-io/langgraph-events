"""The SQL and the private API of ``PostgresSaver`` that the commands use:
the thread listing, the backup of the live set, its restore, and the drop
of old backups.

The saver and psycopg are optional dependencies, so their imports are
local. Each operation reads the store through ``PostgresSaver._cursor()``,
which is private to the saver package, so a release of that package can
change it. It takes the lock of the saver, and a connection of its pool
when the saver holds a pool.

The live set is the latest checkpoint of each thread in each checkpoint
namespace, and the pending writes of that checkpoint. The rewrite
overwrites only these rows. It stores each changed channel under a new
blob version and keeps the old version, so the blobs need no copy.

A backup through ``get_tuple()`` cannot work: the serde migrates each
record on read. So the backup, the restore and the drop read and write
the stored rows, in SQL, through the connection of the saver, in one
transaction.

A backup is named after the migration fingerprint of the running code.
So the operator gives no name, and a rerun under the same migrations
adds to the same backup.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, TypeVar

from langgraph_events.serde.migrations._target import TargetError

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    from langgraph.checkpoint.base import BaseCheckpointSaver
    from langgraph.checkpoint.postgres import PostgresSaver
    from psycopg import Cursor, sql
    from psycopg.rows import DictRow

    from langgraph_events.serde.migrations._target import MigrationTarget

_T = TypeVar("_T")

_THREAD_KEYS_QUERY = "SELECT DISTINCT thread_id, checkpoint_ns FROM checkpoints"

_BACKUP_TABLES_SQL = r"""
SELECT relname FROM pg_class
WHERE relkind = 'r'
  AND relnamespace = (SELECT oid FROM pg_namespace WHERE nspname = current_schema())
  AND relname ~ '^checkpoint(s|_writes)_bak_lge_m[0-9]+_[0-9a-f]{16}$'
ORDER BY relname
"""

_SHARED_COLUMNS_SQL = """
SELECT b.attname FROM pg_attribute AS b
JOIN pg_attribute AS l
  ON l.attrelid = %(live)s::regclass AND l.attname = b.attname
  AND l.attnum > 0 AND NOT l.attisdropped
WHERE b.attrelid = %(backup)s::regclass AND b.attnum > 0 AND NOT b.attisdropped
ORDER BY b.attnum
"""

_COPY_SQL = """
WITH added AS (
    INSERT INTO {checkpoints} ({c_columns})
    SELECT {c_columns} FROM (
        SELECT DISTINCT ON (thread_id, checkpoint_ns) * FROM checkpoints
        ORDER BY thread_id, checkpoint_ns, checkpoint_id DESC
    ) AS latest
    ON CONFLICT DO NOTHING
    RETURNING thread_id, checkpoint_ns, checkpoint_id
), copied AS (
    INSERT INTO {writes} ({w_columns})
    SELECT {w_qualified} FROM checkpoint_writes AS w
    WHERE EXISTS (
        SELECT 1 FROM added AS a
        WHERE (a.thread_id, a.checkpoint_ns, a.checkpoint_id)
            = (w.thread_id, w.checkpoint_ns, w.checkpoint_id)
    ) OR EXISTS (
        SELECT 1 FROM {checkpoints} AS b
        WHERE (b.thread_id, b.checkpoint_ns, b.checkpoint_id)
            = (w.thread_id, w.checkpoint_ns, w.checkpoint_id)
    )
    ON CONFLICT DO NOTHING
    RETURNING 1
)
SELECT (SELECT count(*) FROM added) AS checkpoints,
       (SELECT count(*) FROM copied) AS writes
"""

_COUNT_SQL = """
SELECT (SELECT count(*) FROM {checkpoints}) AS checkpoints,
       (SELECT count(*) FROM {writes}) AS writes
"""

# A checkpoint names a version of each channel, but PostgresSaver stores a
# blob only for a channel that is absent from ``channel_values``, or that
# holds the ``true`` marker of a delta snapshot. A primitive value stays
# inline, and a consumed channel has no value. So the check covers each
# version that the restore changes and that needs a blob.
_MISSING_BLOBS_SQL = """
SELECT count(*) AS missing FROM (
    SELECT DISTINCT b.thread_id, b.checkpoint_ns, v.channel, v.version
    FROM {checkpoints} AS b
    JOIN checkpoints AS t
      ON (t.thread_id, t.checkpoint_ns, t.checkpoint_id)
       = (b.thread_id, b.checkpoint_ns, b.checkpoint_id)
    CROSS JOIN LATERAL jsonb_each_text(b.checkpoint -> 'channel_versions')
      AS v(channel, version)
    WHERE v.version IS DISTINCT FROM t.checkpoint -> 'channel_versions' ->> v.channel
      AND coalesce(b.checkpoint -> 'channel_values' -> v.channel, 'true'::jsonb)
        = 'true'::jsonb
) AS needed
WHERE NOT EXISTS (
    SELECT 1 FROM checkpoint_blobs AS bl
    WHERE (bl.thread_id, bl.checkpoint_ns, bl.channel, bl.version)
        = (needed.thread_id, needed.checkpoint_ns, needed.channel, needed.version)
)
"""

_RESTORE_SQL = """
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
), removed_writes AS (
    DELETE FROM checkpoint_writes AS t
    USING {checkpoints} AS b
    WHERE (t.thread_id, t.checkpoint_ns, t.checkpoint_id)
        = (b.thread_id, b.checkpoint_ns, b.checkpoint_id)
      AND NOT EXISTS (
        SELECT 1 FROM {writes} AS w
        WHERE (w.thread_id, w.checkpoint_ns, w.checkpoint_id, w.task_id, w.idx)
            = (t.thread_id, t.checkpoint_ns, t.checkpoint_id, t.task_id, t.idx)
      )
    RETURNING 1
)
SELECT (SELECT count(*) FROM restored_checkpoints) AS checkpoints,
       (SELECT count(*) FROM restored_writes) AS writes,
       (SELECT count(*) FROM removed_writes) AS removed_writes
"""


@dataclass(frozen=True)
class _Action:
    """A command that acts on the whole store, and the remedy that its
    refusal names."""

    name: str
    remedy: str


_BACKUP = _Action("backup", "Back up the store with the tools of its database.")
_RESTORE = _Action("restore", "Restore the store with the tools of its database.")
_DROP = _Action("drop-backups", "Drop the tables with the tools of its database.")


@dataclass(frozen=True)
class BackupRows:
    """A count of checkpoint rows and of pending-write rows of a backup."""

    checkpoints: int
    writes: int

    def __str__(self) -> str:
        """``1 checkpoint and 2 writes``."""
        checkpoints = "checkpoint" if self.checkpoints == 1 else "checkpoints"
        writes = "write" if self.writes == 1 else "writes"
        return f"{self.checkpoints} {checkpoints} and {self.writes} {writes}"


@dataclass(frozen=True)
class BackupOutcome:
    """What :func:`backup_live_set` did to the backup of ``fingerprint``.

    ``kept`` holds the rows that the backup held before the run. A backup
    keeps each of them. ``added`` holds the rows that the run copied.
    """

    fingerprint: str
    kept: BackupRows
    added: BackupRows


@dataclass(frozen=True)
class RestoreOutcome:
    """What :func:`restore_live_set` did to the store.

    ``restored`` holds the rows of the backup of ``fingerprint`` that the
    restore put back. ``removed_writes`` counts the pending writes that
    the restore deleted: each write that a run added to a backed-up
    checkpoint after the backup.
    """

    fingerprint: str
    restored: BackupRows
    removed_writes: int


@dataclass(frozen=True)
class _Backup:
    """The two backup tables of one fingerprint, in the current schema."""

    fingerprint: str

    @classmethod
    def of_table(cls, name: str) -> _Backup:
        """The backup that the backup table *name* belongs to."""
        return cls(name.split("_bak_", 1)[1].replace("_", "-"))

    @property
    def checkpoints(self) -> str:
        return f"checkpoints_bak_{self._suffix}"

    @property
    def writes(self) -> str:
        return f"checkpoint_writes_bak_{self._suffix}"

    @property
    def _suffix(self) -> str:
        return self.fingerprint.replace("-", "_")

    def take(self, cursor: Cursor[DictRow]) -> BackupOutcome:
        """Copy the live set into the two tables. Keep each row that the
        tables hold, and copy each pending write of a backed-up checkpoint
        that the tables do not hold."""
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
        kept = _rows(cursor, self._sql(_COUNT_SQL))
        added = _rows(
            cursor,
            self._sql(
                _COPY_SQL,
                checkpoints=_shared_columns(cursor, "checkpoints", self.checkpoints),
                writes=_shared_columns(cursor, "checkpoint_writes", self.writes),
            ),
        )
        return BackupOutcome(self.fingerprint, kept, added)

    def restore(self, cursor: Cursor[DictRow]) -> RestoreOutcome:
        """Put back the rows of the backup, after the checks that refuse a
        missing backup and a missing blob version."""
        held = _held_backups(cursor)
        if self not in held:
            raise TargetError(self._missing(held))
        # A row lock on a blob does not stop a prune whose snapshot is older
        # than the restore. The table lock waits for a running prune, and
        # holds back a new one until the restore commits.
        cursor.execute("LOCK TABLE checkpoint_blobs IN SHARE MODE")
        cursor.execute(self._sql(_MISSING_BLOBS_SQL))
        missing = _one(cursor)["missing"]
        if missing:
            versions = "version" if missing == 1 else "versions"
            raise TargetError(
                f"The backup {self.fingerprint} names {missing} blob {versions} "
                f"that checkpoint_blobs no longer holds, so the restore would "
                f"lose a channel. A prune of old blob versions deletes such "
                f"versions. Nothing changed. {_RESTORE.remedy}"
            )
        cursor.execute(self._sql(_RESTORE_SQL))
        row = _one(cursor)
        restored = BackupRows(row["checkpoints"], row["writes"])
        return RestoreOutcome(self.fingerprint, restored, row["removed_writes"])

    def _missing(self, held: tuple[_Backup, ...]) -> str:
        """The message of a restore that finds no backup of the running
        code. It names each backup that the store holds."""
        names = ", ".join(backup.fingerprint for backup in held)
        holds = (
            f"The store holds the backup {names}"
            if held
            else "The store holds no backup"
        )
        return (
            f"The store holds no backup {self.fingerprint}, the migration "
            f"fingerprint of the running code. {holds}. Nothing changed. Run "
            f"restore in a container of the release that took the backup."
        )

    def _sql(
        self,
        template: str,
        *,
        checkpoints: list[str] | None = None,
        writes: list[str] | None = None,
    ) -> sql.Composed:
        """*template* with the two table names, and with the column lists
        ``{c_columns}``, ``{w_columns}`` and ``{w_qualified}``."""
        from psycopg import sql  # noqa: PLC0415

        def columns(names: list[str] | None, *table: str) -> sql.Composed:
            return sql.SQL(", ").join(sql.Identifier(*table, n) for n in names or [])

        return sql.SQL(template).format(
            checkpoints=sql.Identifier(self.checkpoints),
            writes=sql.Identifier(self.writes),
            c_columns=columns(checkpoints),
            w_columns=columns(writes),
            w_qualified=columns(writes, "w"),
        )


def thread_rows(saver: BaseCheckpointSaver[Any]) -> list[tuple[str, str]] | None:
    """The thread id and checkpoint namespace of each checkpoint row of a
    ``PostgresSaver``, or ``None`` for another saver."""
    postgres = _postgres_saver(saver)
    if postgres is None:
        return None
    with postgres._cursor() as cursor:
        cursor.execute(_THREAD_KEYS_QUERY)
        rows = cursor.fetchall()
    return [(r["thread_id"], r["checkpoint_ns"]) for r in rows]


def backup_live_set(target: MigrationTarget) -> BackupOutcome:
    """Back up the live set of the ``PostgresSaver`` store of *target*.

    The backup is named after the migration fingerprint of the running
    code: ``checkpoints_bak_<fp>`` holds the latest checkpoint row of each
    thread in each namespace, and ``checkpoint_writes_bak_<fp>`` holds the
    pending writes of those rows. ``<fp>`` is the fingerprint with ``_``
    for each ``-``. The copy runs in one transaction, and names the
    columns that each backup table and its live table share.

    A rerun under the same fingerprint keeps each row that the backup
    holds. It adds the latest row of each thread that appeared or changed
    since, and each pending write of a backed-up checkpoint that a run
    added since. The function drops no table.

    Raises:
        TargetError: when *target* sets ``threads``, when the saver is not
            a ``PostgresSaver``, or when the backup fails. A failed backup
            writes nothing.
    """
    _require_whole_store(target, _BACKUP)
    backup = _Backup(target.snapshot().migrations.fingerprint)
    return _in_transaction(target, _BACKUP, backup.take)


def restore_live_set(target: MigrationTarget) -> RestoreOutcome:
    """Put back the backup of the running code into the store of *target*.

    The backup is the one named after the migration fingerprint of the
    running code. The restore runs in one transaction. It overwrites each
    checkpoint row and each pending write that the backup holds and the
    store still holds under the same key. It deletes each pending write
    of a backed-up checkpoint that the backup does not hold: a run added
    it after the last backup. A checkpoint that a run wrote after the
    backup has a new key, so the restore leaves it.

    Before it writes, the restore checks each blob version that it puts
    back: each channel version of a backed-up checkpoint that differs
    from the version of the live row and that needs a blob. The rewrite
    keeps the old version, but a prune of old blob versions can delete
    it. The restore locks ``checkpoint_blobs`` against writes until the
    end of the transaction, so a prune waits for it.

    Raises:
        TargetError: when *target* sets ``threads``, when the saver is not
            a ``PostgresSaver``, when the store holds no backup of the
            running code (the message names each backup that the store
            holds), when a blob version that the restore puts back is
            missing, or when the restore fails. Then nothing changed.
    """
    _require_whole_store(target, _RESTORE)
    backup = _Backup(target.snapshot().migrations.fingerprint)
    return _in_transaction(target, _RESTORE, backup.restore)


def drop_backups(target: MigrationTarget) -> tuple[str, ...]:
    """Drop each backup of the store of *target* except the backup of the
    running code, in one transaction. Return the fingerprint of each
    dropped backup.

    The function drops only the tables in the current schema of the store
    whose name is the backup name of a fingerprint.

    Raises:
        TargetError: when *target* sets ``threads``, when the saver is not
            a ``PostgresSaver``, or when the drop fails. A failed drop
            drops nothing.
    """
    _require_whole_store(target, _DROP)
    keep = _Backup(target.snapshot().migrations.fingerprint)

    def drop(cursor: Cursor[DictRow]) -> tuple[str, ...]:
        from psycopg import sql  # noqa: PLC0415

        dropped: set[str] = set()
        for name in _backup_tables(cursor):
            backup = _Backup.of_table(name)
            if backup != keep:
                cursor.execute(sql.SQL("DROP TABLE {}").format(sql.Identifier(name)))
                dropped.add(backup.fingerprint)
        return tuple(sorted(dropped))

    return _in_transaction(target, _DROP, drop)


def _require_whole_store(target: MigrationTarget, action: _Action) -> None:
    """Refuse a target that sets ``threads``. That target declares a store
    that other graphs can share. The action acts on every row of the
    store, and the backup name does not name the graph."""
    if target.threads is not None:
        raise TargetError(
            f"The {action.name} acts on the whole store, and the target sets "
            f"threads=, so other graphs can share the store. The {action.name} "
            f"needs a store that one graph owns. {action.remedy}"
        )


def _in_transaction(
    target: MigrationTarget,
    action: _Action,
    work: Callable[[Cursor[DictRow]], _T],
) -> _T:
    """*work* on a cursor of the store of *target*, inside one transaction.

    Raises:
        TargetError: when the saver is not a ``PostgresSaver``, or when the
            database refuses a statement. The transaction then writes
            nothing.
    """
    import psycopg  # noqa: PLC0415

    with target.open_store() as saver:
        postgres = _require_postgres(saver, action)
        try:
            with _transaction(postgres) as cursor:
                return work(cursor)
        except psycopg.Error as error:
            raise TargetError(
                f"The {action.name} failed and changed nothing: {_one_line(error)}"
            ) from error


@contextlib.contextmanager
def _transaction(saver: PostgresSaver) -> Iterator[Cursor[DictRow]]:
    """A cursor of *saver* inside one transaction."""
    with saver._cursor() as cursor, cursor.connection.transaction():
        yield cursor


def _backup_tables(cursor: Cursor[DictRow]) -> list[str]:
    """The name of each backup table in the current schema, sorted."""
    cursor.execute(_BACKUP_TABLES_SQL)
    return [row["relname"] for row in cursor.fetchall()]


def _held_backups(cursor: Cursor[DictRow]) -> tuple[_Backup, ...]:
    """Each backup whose two tables the current schema holds."""
    names = _backup_tables(cursor)
    backups = dict.fromkeys(_Backup.of_table(name) for name in names)
    return tuple(
        backup
        for backup in backups
        if backup.checkpoints in names and backup.writes in names
    )


def _shared_columns(cursor: Cursor[DictRow], live: str, backup: str) -> list[str]:
    """The columns that the table *backup* and the table *live* share, in
    the order of *backup*. A copy that names them still runs after an
    upgrade of the saver adds a column to *live*."""
    cursor.execute(_SHARED_COLUMNS_SQL, {"live": live, "backup": backup})
    return [row["attname"] for row in cursor.fetchall()]


def _one(cursor: Cursor[DictRow]) -> DictRow:
    row = cursor.fetchone()
    assert row is not None  # noqa: S101 - a SELECT of counts returns one row
    return row


def _rows(cursor: Cursor[DictRow], query: sql.Composed) -> BackupRows:
    cursor.execute(query)
    row = _one(cursor)
    return BackupRows(row["checkpoints"], row["writes"])


def _require_postgres(
    saver: BaseCheckpointSaver[Any], action: _Action
) -> PostgresSaver:
    postgres = _postgres_saver(saver)
    if postgres is None:
        raise TargetError(
            f"The {action.name} needs a PostgresSaver, and the saver of the "
            f"store is {type(saver).__name__}. {action.remedy}"
        )
    return postgres


def _postgres_saver(saver: BaseCheckpointSaver[Any]) -> PostgresSaver | None:
    """*saver* when it is a ``PostgresSaver``, else ``None``."""
    try:
        from langgraph.checkpoint.postgres import PostgresSaver  # noqa: PLC0415
    except ImportError:
        return None
    return saver if isinstance(saver, PostgresSaver) else None


def _one_line(error: BaseException) -> str:
    return f"{type(error).__name__}: {' '.join(str(error).split())}"
