"""Two releases of one graph over one saver, shared by the live-set tests
and the CLI tests.

A class defined inside a helper never imports, so a serde swap on one saver
simulates two releases against real bytes. This module has no
``from __future__ import annotations``: a handler defined inside a helper
must keep a return annotation that resolves at run time.
"""

import contextlib
import io
from collections.abc import Iterator
from pathlib import Path
from typing import NamedTuple

import pytest
from conftest import Started
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver

from langgraph_events import EventGraph, IntegrationEvent, on
from langgraph_events.serde import NamespaceAwareSerde, migrate_from
from langgraph_events.serde.migrations import (
    LiveSetOutcome,
    Migration,
    MigrationTarget,
    TransformFields,
    rewrite_live_set,
)


def _graph(saver):
    saver.serde = NamespaceAwareSerde(events=(Started,))
    return EventGraph([_noop], checkpointer=saver)


@on(Started)
def _noop(event: Started) -> None:
    return None


def _cfg(tid: str, ns: str = "") -> dict:
    return {"configurable": {"thread_id": tid, "checkpoint_ns": ns}}


class _Releases(NamedTuple):
    """Two releases of one graph over one saver.

    ``current`` renames the event that ``previous`` stores, so a thread
    that ``previous`` wrote holds the old shape under ``current``.
    """

    previous: MigrationTarget
    current: MigrationTarget
    saver: BaseCheckpointSaver

    @property
    def expect(self) -> str:
        return self.current.snapshot().migrations.fingerprint

    def seed(self, *thread_ids: str) -> None:
        graph = self.previous.graph(self.saver)
        for tid in thread_ids:
            graph.invoke(Started(data=tid), config=_cfg(tid))

    def plan(self, *thread_ids: str) -> list[str]:
        graph = self.current.graph(self.saver)
        report = graph.event_store.plan_rewrite(thread_ids=thread_ids)
        return [thread.status for thread in report.threads]


def _releases(tmp_path: Path, saver: BaseCheckpointSaver | None = None) -> _Releases:
    store = saver if saver is not None else MemorySaver()

    class _OldPlaced(IntegrationEvent):
        note: str = ""

    @migrate_from(_OldPlaced.__qualname__, in_module=_OldPlaced.__module__)
    class _Placed(IntegrationEvent):
        note: str = ""

    def previous(s):
        @on(Started)
        def place(event: Started) -> _OldPlaced:
            return _OldPlaced(note=event.data)

        s.serde = NamespaceAwareSerde(events=(Started, _OldPlaced))
        return EventGraph([place], checkpointer=s)

    def current(s):
        @on(Started)
        def place(event: Started) -> _Placed:
            return _Placed(note=event.data)

        s.serde = NamespaceAwareSerde(events=(Started, _Placed))
        return EventGraph([place], checkpointer=s)

    def open_store():
        return contextlib.nullcontext(store)

    baseline = tmp_path / "baseline.json"
    return _Releases(
        MigrationTarget(graph=previous, baseline=baseline, store=open_store),
        MigrationTarget(graph=current, baseline=baseline, store=open_store),
        store,
    )


def _copy_to_namespace(saver: BaseCheckpointSaver, tid: str, ns: str) -> None:
    """Store the latest root checkpoint of *tid* again under *ns*, through
    the serde the saver holds now."""
    tup = saver.get_tuple(_cfg(tid))
    assert tup is not None
    saver.put(
        _cfg(tid, ns), tup.checkpoint, tup.metadata, tup.checkpoint["channel_versions"]
    )


class _Counted(IntegrationEvent):
    count: int = 0


@on(Started)
def _count(event: Started) -> _Counted:
    return _Counted(count=1)


def _bump(kw: dict) -> dict:
    """Not idempotent: each read adds one again."""
    kw["count"] += 1
    return kw


def _unsettling(tmp_path: Path) -> MigrationTarget:
    """A target whose class-stage transform of ``_Counted`` adds one on each
    read, over a store whose thread ``t1`` was written with no transform."""
    store = MemorySaver()
    store.serde = NamespaceAwareSerde(events=(Started, _Counted))
    EventGraph([_count], checkpointer=store).invoke(
        Started(data="t1"), config=_cfg("t1")
    )

    def current(s):
        bump = TransformFields(_Counted.__module__, _Counted.__qualname__, _bump)
        s.serde = NamespaceAwareSerde(
            events=(Started, _Counted), migrations=[Migration(operations=(bump,))]
        )
        return EventGraph([_count], checkpointer=s)

    return MigrationTarget(
        graph=current,
        baseline=tmp_path / "baseline.json",
        store=lambda: contextlib.nullcontext(store),
    )


class _PostgresRows(NamedTuple):
    """The stored rows of a ``PostgresSaver`` store, read in SQL, as the
    serde never sees them."""

    url: str

    def execute(self, query: str, params: tuple = ()) -> list[tuple]:
        import psycopg

        with psycopg.connect(self.url, autocommit=True) as conn:
            cursor = conn.execute(query, params)  # type: ignore[arg-type]
            return cursor.fetchall() if cursor.description else []

    def checkpoint(self, tid: str, table: str = "checkpoints") -> tuple:
        """The latest checkpoint row of *tid* in the root namespace."""
        rows = self.execute(
            f"SELECT checkpoint_id, checkpoint, metadata FROM {table} "  # noqa: S608 - a test-owned table name
            "WHERE thread_id = %s AND checkpoint_ns = '' "
            "ORDER BY checkpoint_id DESC LIMIT 1",
            (tid,),
        )
        return rows[0]

    def writes(self, tid: str) -> list[tuple]:
        """The pending writes of the latest checkpoint of *tid*."""
        return self.execute(
            "SELECT task_id, idx, channel, type, blob FROM checkpoint_writes "
            "WHERE thread_id = %s AND checkpoint_ns = '' AND checkpoint_id = "
            "(SELECT max(checkpoint_id) FROM checkpoints "
            "WHERE thread_id = %s AND checkpoint_ns = '') ORDER BY 1, 2",
            (tid, tid),
        )

    def backup_tables(self) -> list[str]:
        rows = self.execute(
            "SELECT tablename FROM pg_tables WHERE schemaname = current_schema() "
            "AND tablename LIKE '%%\\_bak\\_%%' ORDER BY 1"
        )
        return [row[0] for row in rows]


def _write_pending(
    saver: BaseCheckpointSaver, tid: str, task_id: str = "task-1"
) -> None:
    """Store one completed write of *task_id* on the latest checkpoint of
    *tid*."""
    tup = saver.get_tuple(_cfg(tid))
    assert tup is not None
    saver.put_writes(tup.config, [("audit", "kept")], task_id)


class _PostgresStore(NamedTuple):
    """Two releases over one ``PostgresSaver`` store, and its stored rows."""

    releases: _Releases
    rows: _PostgresRows

    @property
    def fingerprint(self) -> str:
        return self.releases.expect

    def backup_table(self, table: str = "checkpoints") -> str:
        """The name of the backup table of *table* under the current release."""
        return f"{table}_bak_" + self.fingerprint.replace("-", "_")

    def rewrite(self, *, apply: bool = True) -> LiveSetOutcome:
        outcome = rewrite_live_set(
            self.releases.current,
            expect=self.releases.expect,
            apply=apply,
            log=io.StringIO(),
        )
        assert isinstance(outcome, LiveSetOutcome)
        return outcome

    def live_tables(self) -> list[list[tuple]]:
        """Every row of the three live tables, in a stable order."""
        return [
            self.rows.execute(f"SELECT * FROM {table} ORDER BY 1, 2, 3, 4")  # noqa: S608 - a test-owned table name
            for table in ("checkpoints", "checkpoint_writes", "checkpoint_blobs")
        ]


@pytest.fixture
def postgres_store(tmp_path: Path, postgres_url: str) -> Iterator[_PostgresStore]:
    """Thread ``t1``, written by the previous release, with one pending
    write on its latest checkpoint. The current release reads it as a
    rewrite."""
    from langgraph.checkpoint.postgres import PostgresSaver

    with PostgresSaver.from_conn_string(postgres_url) as saver:
        releases = _releases(tmp_path, saver)
        releases.seed("t1")
        _write_pending(saver, "t1")
        yield _PostgresStore(releases, _PostgresRows(postgres_url))


def _no_store() -> contextlib.AbstractContextManager[MemorySaver]:
    raise AssertionError("the store opened")
