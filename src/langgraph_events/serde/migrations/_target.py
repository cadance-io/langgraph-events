"""The graph, the baseline file and the store that every command reads."""

from __future__ import annotations

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


_THREAD_KEYS_QUERY = "SELECT DISTINCT thread_id, checkpoint_ns FROM checkpoints"


class TargetError(Exception):
    """The target cannot give what a command needs from it."""


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
    opens the real checkpointer. Only ``rewrite`` uses it. ``threads``
    lists the checkpoint namespaces of the store. It is needed only for a
    saver other than ``PostgresSaver`` or ``MemorySaver``, or for a store
    that more than one graph shares.
    """

    graph: Callable[[BaseCheckpointSaver[Any]], EventGraph]
    baseline: Path
    store: Callable[[], AbstractContextManager[BaseCheckpointSaver[Any]]] | None = None
    threads: Callable[[BaseCheckpointSaver[Any]], Iterable[ThreadKey]] | None = None

    def thread_keys(self, saver: BaseCheckpointSaver[Any]) -> tuple[ThreadKey, ...]:
        """Every checkpoint namespace of every thread in *saver*, sorted.

        Raises:
            TargetError: when the target has no ``threads`` and the saver
                is not a ``PostgresSaver`` or a ``MemorySaver``.
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
        postgres = _postgres_thread_keys(saver)
        if postgres is not None:
            return postgres
        raise TargetError(
            f"The target cannot list the threads of a {type(saver).__name__}. "
            f"Set threads= on the MigrationTarget to a function that takes the "
            f"saver and returns a ThreadKey for each thread and checkpoint "
            f"namespace, for example from: {_THREAD_KEYS_QUERY}."
        )

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


def _postgres_thread_keys(
    saver: BaseCheckpointSaver[Any],
) -> tuple[ThreadKey, ...] | None:
    """The keys of a ``PostgresSaver``, or ``None`` for another saver.

    The import is local because the Postgres saver is an optional
    dependency.
    """
    try:
        from langgraph.checkpoint.postgres import (  # noqa: PLC0415
            PostgresSaver,
        )
    except ImportError:
        return None
    if not isinstance(saver, PostgresSaver):
        return None
    with saver._cursor() as cursor:
        cursor.execute(_THREAD_KEYS_QUERY)
        rows = cursor.fetchall()
    return tuple(sorted(ThreadKey(r["thread_id"], r["checkpoint_ns"]) for r in rows))


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
