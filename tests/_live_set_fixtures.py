"""Two releases of one graph over one saver, shared by the live-set tests
and the CLI tests.

A class defined inside a helper never imports, so a serde swap on one saver
simulates two releases against real bytes. This module has no
``from __future__ import annotations``: a handler defined inside a helper
must keep a return annotation that resolves at run time.
"""

import contextlib
from pathlib import Path
from typing import NamedTuple

from conftest import Started
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver

from langgraph_events import EventGraph, IntegrationEvent, on
from langgraph_events.serde import NamespaceAwareSerde, migrate_from
from langgraph_events.serde.migrations import MigrationTarget


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
        report = graph.plan_rewrite(thread_ids=thread_ids)
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
