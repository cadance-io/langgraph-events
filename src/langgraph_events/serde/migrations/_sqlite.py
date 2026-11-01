"""The SQL and the internal API of ``SqliteSaver`` that the commands use.

The saver is an optional dependency, so its import is local.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from langgraph.checkpoint.base import BaseCheckpointSaver

_THREAD_KEYS_QUERY = "SELECT DISTINCT thread_id, checkpoint_ns FROM checkpoints"


def thread_rows(saver: BaseCheckpointSaver[Any]) -> list[tuple[str, str]] | None:
    """The thread id and checkpoint namespace of each checkpoint row of a
    ``SqliteSaver``, or ``None`` for another saver.

    The docstring of ``SqliteSaver.cursor()`` says that a user must not
    call it, so a release of the saver package can change it.
    """
    try:
        from langgraph.checkpoint.sqlite import SqliteSaver  # noqa: PLC0415
    except ImportError:
        return None
    if not isinstance(saver, SqliteSaver):
        return None
    with saver.cursor(transaction=False) as cursor:
        cursor.execute(_THREAD_KEYS_QUERY)
        return [(tid, ns) for tid, ns in cursor.fetchall()]
