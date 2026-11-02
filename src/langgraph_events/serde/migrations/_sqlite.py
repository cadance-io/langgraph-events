"""The SQL and the internal API of ``SqliteSaver`` that the commands use.

The saver is an optional dependency, so its import is local.
"""

from __future__ import annotations

import sqlite3
from typing import TYPE_CHECKING, Any

from langgraph_events.serde.migrations._target import TargetError

if TYPE_CHECKING:
    from langgraph.checkpoint.base import BaseCheckpointSaver

_THREAD_KEYS_QUERY = "SELECT DISTINCT thread_id, checkpoint_ns FROM checkpoints"
_HAS_CHECKPOINTS_QUERY = (
    "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'checkpoints'"
)


def thread_rows(saver: BaseCheckpointSaver[Any]) -> list[tuple[str, str]] | None:
    """The thread id and checkpoint namespace of each checkpoint row of a
    ``SqliteSaver``, or ``None`` for another saver.

    ``SqliteSaver.cursor()`` creates the tables, so the check for the
    ``checkpoints`` table reads the public ``conn`` of the saver first, and
    a plan writes nothing.

    The docstring of ``SqliteSaver.cursor()`` says that a user must not
    call it, so a release of the saver package can change it.

    Raises:
        TargetError: when the database refuses a statement.
    """
    try:
        from langgraph.checkpoint.sqlite import SqliteSaver  # noqa: PLC0415
    except ImportError:
        return None
    if not isinstance(saver, SqliteSaver):
        return None
    try:
        with saver.lock:
            if saver.conn.execute(_HAS_CHECKPOINTS_QUERY).fetchone() is None:
                return []
        with saver.cursor(transaction=False) as cursor:
            cursor.execute(_THREAD_KEYS_QUERY)
            return [(tid, ns) for tid, ns in cursor.fetchall()]
    except sqlite3.Error as error:
        raise TargetError.refused_listing(error) from error
