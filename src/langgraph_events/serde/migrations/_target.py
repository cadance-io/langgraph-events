"""The graph, the baseline file and the store that every command reads."""

from __future__ import annotations

import contextlib
import inspect
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from langgraph.checkpoint.memory import MemorySaver

from langgraph_events._identity import Identity
from langgraph_events.serde.migrations._core import (
    _declares_migration,
    _init_fields,
    _serde_event_classes,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable
    from contextlib import AbstractContextManager
    from pathlib import Path

    from langgraph.checkpoint.base import BaseCheckpointSaver

    from langgraph_events import EventGraph
    from langgraph_events.serde._jsonplus import NamespaceAwareSerde
    from langgraph_events.serde.migrations._core import MigrationSet


class TargetError(Exception):
    """The target cannot give what a command needs from it."""

    @classmethod
    def refused_listing(cls, error: BaseException) -> TargetError:
        """The error of a store that refuses the thread listing with
        *error*."""
        return cls(f"The store refused the thread listing ({_one_line(error)}).")


@dataclass(frozen=True, order=True)
class ThreadKey:
    """One checkpoint namespace of one thread. ``""`` is the root namespace."""

    thread_id: str
    checkpoint_ns: str


@dataclass(frozen=True)
class MigrationTarget:
    """One graph, its committed baseline and its store.

    ``graph`` builds the graph on the checkpointer it receives. The gate
    passes an empty in-memory saver, so building reads no store.
    ``baseline`` is the path of the committed baseline file. ``store``
    returns a context manager that opens the real sync checkpointer. The
    ``backup``, ``rewrite``, ``restore`` and ``drop-backups`` commands use
    it. ``threads`` lists the checkpoint namespaces of
    the store. It is needed only for a saver other than ``PostgresSaver``,
    ``SqliteSaver`` or ``MemorySaver``, or for a store that more than one
    graph shares. :func:`list_thread_keys` lists the threads of any saver
    through its public ``list()``, for a small store. A target with
    ``threads`` declares a store that other graphs can share, so
    :func:`backup_live_set`, :func:`restore_live_set` and
    :func:`drop_backups`, which act on the whole store, refuse it.
    """

    graph: Callable[[BaseCheckpointSaver[Any]], EventGraph]
    baseline: Path
    store: Callable[[], AbstractContextManager[BaseCheckpointSaver[Any]]] | None = None
    threads: Callable[[BaseCheckpointSaver[Any]], Iterable[ThreadKey]] | None = None

    def thread_keys(self, saver: BaseCheckpointSaver[Any]) -> tuple[ThreadKey, ...]:
        """Every checkpoint namespace of every thread in *saver*, sorted.

        Raises:
            TargetError: when the target has no ``threads`` and the saver
                is not a ``PostgresSaver``, a ``SqliteSaver`` or a
                ``MemorySaver``, when the built-in listing of the saver
                fails, or when the store refuses the listing.
        """
        if self.threads is not None:
            return tuple(sorted(set(self.threads(saver))))
        if isinstance(saver, MemorySaver):
            return tuple(
                sorted(
                    ThreadKey(thread_id, ns)
                    for thread_id, namespaces in saver.storage.items()
                    for ns, checkpoints in namespaces.items()
                    if checkpoints
                )
            )
        # ``_postgres`` imports this module, so the import is local.
        from langgraph_events.serde.migrations import (  # noqa: PLC0415
            _postgres,
            _sqlite,
        )

        try:
            rows = _postgres.thread_rows(saver)
            if rows is None:
                rows = _sqlite.thread_rows(saver)
        except TargetError:
            raise
        except Exception as error:
            # The listing reads an internal method of the saver package,
            # which a release of that package can change.
            failure = f" ({type(error).__name__}: {error})"
            raise TargetError(_no_thread_listing(saver, failure)) from error
        if rows is not None:
            return tuple(sorted(ThreadKey(tid, ns) for tid, ns in rows))
        raise TargetError(_no_thread_listing(saver, ""))

    def open_store(self) -> AbstractContextManager[BaseCheckpointSaver[Any]]:
        """The context manager that ``store`` returns.

        Raises:
            TargetError: when the target has no ``store``, or when
                ``store`` returns an async context manager.
        """
        if self.store is None:
            raise TargetError(
                "The target has no store. Set store= on the MigrationTarget to "
                "a function that opens the checkpointer, for example "
                "lambda: PostgresSaver.from_conn_string(url)."
            )
        opened = self.store()
        if not hasattr(opened, "__enter__") and hasattr(opened, "__aenter__"):
            raise TargetError(
                "The store= function of the target returned an async context "
                "manager. The command needs a sync checkpointer. Return "
                "the sync saver of the same database, for example "
                "SqliteSaver.from_conn_string(path) for an AsyncSqliteSaver, or "
                "PostgresSaver.from_conn_string(url) for an AsyncPostgresSaver."
            )
        return opened

    def snapshot(self) -> Snapshot:
        """What the code holds, built once for every rule on an empty
        in-memory saver.

        Raises:
            TargetError: see :meth:`build`.
        """
        return self.build(MemorySaver())

    def build(self, saver: BaseCheckpointSaver[Any]) -> Snapshot:
        """The graph on *saver*, its serde and its client migrations.

        Raises:
            TargetError: when ``graph`` cannot take a checkpointer, when the
                graph has no ``NamespaceAwareSerde``, or when a client
                migration has no valid key.
        """
        try:
            inspect.signature(self.graph).bind(saver)
        except TypeError as error:
            raise TargetError(
                f"The graph factory of the target cannot take a checkpointer: "
                f"{error}. Write it as def build(checkpointer) -> EventGraph, "
                f"and build the graph on that checkpointer."
            ) from error
        graph = self.graph(saver)
        try:
            serde = graph._require_namespace_aware_serde("MigrationTarget.graph")
        except ValueError as error:
            raise TargetError(str(error)) from error
        try:
            migrations = serde.client_migrations()
        except ValueError as error:
            raise TargetError(
                f"A client migration has no valid key: {error}"
            ) from error
        return Snapshot(graph, serde, migrations)


def list_thread_keys(saver: BaseCheckpointSaver[Any]) -> list[ThreadKey]:
    """Every checkpoint namespace of every thread in *saver*, sorted, read
    through the public ``saver.list(None)``.

    Set ``threads=list_thread_keys`` on a :class:`MigrationTarget` whose
    saver the target cannot list. A checkpoint that names a class the
    serde cannot revive still lists its thread, so the plan can report it.

    WARNING: ``list(None)`` reads and decodes every checkpoint of the
    store, older checkpoints included. On a large store this costs time in
    proportion to the whole store, and memory when the saver loads all
    rows at once. For a large store, write a ``threads`` function that
    asks the store for each distinct thread id and checkpoint namespace.
    """
    from langgraph_events.serde._jsonplus import (  # noqa: PLC0415
        NamespaceAwareSerde,
    )

    serde = getattr(saver, "serde", None)
    tolerant = (
        serde.tolerate_unresolved()
        if isinstance(serde, NamespaceAwareSerde)
        else contextlib.nullcontext()
    )
    with tolerant:
        keys = {
            ThreadKey(
                tup.config["configurable"]["thread_id"],
                tup.config["configurable"].get("checkpoint_ns", ""),
            )
            for tup in saver.list(None)
        }
    return sorted(keys)


def _one_line(error: BaseException) -> str:
    """*error* and its type on one line."""
    return f"{type(error).__name__}: {' '.join(str(error).split())}"


def _no_thread_listing(saver: BaseCheckpointSaver[Any], failure: str) -> str:
    """The message of a target that cannot list the threads of *saver*.
    *failure* names the error of the built-in listing, or is empty."""
    return (
        f"The target cannot list the threads of a {type(saver).__name__}"
        f"{failure}. Set threads= on the MigrationTarget to a function that "
        f"takes the saver and returns a ThreadKey for each thread and "
        f"checkpoint namespace. For a small store, set threads=list_thread_keys. "
        f"It reads every checkpoint through saver.list(None), so its time "
        f"grows with the whole store."
    )


@dataclass(frozen=True)
class Snapshot:
    """The graph, its serde and its client migrations.

    The event classes of the snapshot are the classes the serde reads and
    the classes the graph can store. A graph class outside the serde
    revives by import, so it is live too.
    """

    graph: EventGraph
    serde: NamespaceAwareSerde
    migrations: MigrationSet

    def live_fields(self) -> dict[Identity, frozenset[str]]:
        """The init fields of every live event class of the snapshot."""
        fields = self.serde.live_fields()
        for cls in self._graph_classes():
            fields.setdefault(
                Identity(cls.__module__, cls.__qualname__), _init_fields(cls)
            )
        return fields

    def revivable_identities(self) -> frozenset[Identity]:
        """Every identity the serde revives, by a live class or a rename."""
        return self.serde.revivable_identities() | self.live_fields().keys()

    def migrations_out_of_scope(self) -> tuple[Identity, ...]:
        """Each graph class with a read-time migration that the serde does
        not read, so that the serde never collects the migration."""
        read = self.serde.live_fields().keys()
        return tuple(
            sorted(
                identity
                for cls in self._graph_classes()
                if _declares_migration(cls)
                and (identity := Identity(cls.__module__, cls.__qualname__)) not in read
            )
        )

    def _graph_classes(self) -> list[type]:
        return _serde_event_classes(*self.graph._serde_scope())
