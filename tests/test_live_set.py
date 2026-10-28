"""Tests for the live-set rewrite: ``MigrationTarget.thread_keys``,
``rewrite_live_set`` and the ``rewrite`` CLI command.

Idiom, shared with ``tests/test_rewrite_store.py``: a class defined inside
a helper never imports, so a serde swap on one saver simulates two
releases against real bytes.
"""

import contextlib
import io
from pathlib import Path
from typing import NamedTuple

import pytest
from conftest import Started
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.base.id import uuid6
from langgraph.checkpoint.memory import MemorySaver

from langgraph_events import EventGraph, IntegrationEvent, on
from langgraph_events.serde import NamespaceAwareSerde, migrate_from
from langgraph_events.serde.migrations import (
    LiveSetOutcome,
    MigrationTarget,
    TargetError,
    ThreadKey,
    WrongRelease,
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


def describe_MigrationTarget():
    def describe_thread_keys():
        def when_the_saver_is_a_memory_saver():
            def it_lists_each_thread_and_namespace(tmp_path: Path):
                saver = MemorySaver()
                graph = _graph(saver)
                graph.invoke(Started(), config=_cfg("b"))
                graph.invoke(Started(), config=_cfg("a"))
                tup = saver.get_tuple(_cfg("a"))
                assert tup is not None
                saver.put(
                    _cfg("a", "child:1"),
                    tup.checkpoint,
                    tup.metadata,
                    tup.checkpoint["channel_versions"],
                )
                target = MigrationTarget(graph=_graph, baseline=tmp_path / "b.json")

                assert target.thread_keys(saver) == (
                    ThreadKey("a", ""),
                    ThreadKey("a", "child:1"),
                    ThreadKey("b", ""),
                )

        def when_the_target_names_its_threads():
            def it_uses_the_target_function(tmp_path: Path):
                target = MigrationTarget(
                    graph=_graph,
                    baseline=tmp_path / "b.json",
                    threads=lambda saver: [ThreadKey("x", ""), ThreadKey("x", "")],
                )

                assert target.thread_keys(MemorySaver()) == (ThreadKey("x", ""),)

        def when_the_saver_has_no_built_in_enumeration():
            def it_raises_a_target_error_naming_threads(tmp_path: Path):
                target = MigrationTarget(graph=_graph, baseline=tmp_path / "b.json")

                with pytest.raises(TargetError, match=r"threads="):
                    target.thread_keys(BaseCheckpointSaver())


class _Releases(NamedTuple):
    """Two releases of one graph over one saver.

    ``current`` renames the event that ``previous`` stores, so a thread
    that ``previous`` wrote holds the old shape under ``current``.
    """

    previous: MigrationTarget
    current: MigrationTarget
    saver: MemorySaver

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


def _releases(tmp_path: Path, saver: MemorySaver | None = None) -> _Releases:
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


def _copy_to_namespace(saver: MemorySaver, tid: str, ns: str) -> None:
    """Store the latest root checkpoint of *tid* again under *ns*, through
    the serde the saver holds now."""
    tup = saver.get_tuple(_cfg(tid))
    assert tup is not None
    saver.put(
        _cfg(tid, ns), tup.checkpoint, tup.metadata, tup.checkpoint["channel_versions"]
    )


class _StaleRunSaver(MemorySaver):
    """A run loaded the checkpoint before the rewrite and commits its next
    checkpoint after the rewrite proved that the thread did not move.

    The child keeps the parent's channel versions, so it points at the
    old blobs, which the rewrite does not delete.
    """

    def __init__(self) -> None:
        super().__init__()
        self.stale_runs = 0
        self._loaded = None

    def put(self, config, checkpoint, metadata, new_versions):  # type: ignore[override]
        if self.stale_runs and self._loaded is None:
            self.stale_runs -= 1
            tid = config["configurable"]["thread_id"]
            self._loaded = super().get_tuple(_cfg(tid))
        return super().put(config, checkpoint, metadata, new_versions)

    def get_tuple(self, config):  # type: ignore[override]
        result = super().get_tuple(config)
        loaded, self._loaded = self._loaded, None
        if loaded is not None:
            child = {**loaded.checkpoint, "id": str(uuid6())}
            super().put(loaded.config, child, loaded.metadata, {})
        return result


def describe_rewrite_live_set():
    def when_the_running_release_is_not_the_expected_one():
        def it_refuses_before_it_opens_the_store(tmp_path: Path):
            releases = _releases(tmp_path)

            def no_store():
                raise AssertionError("the store was opened")

            previous = MigrationTarget(
                graph=releases.previous.graph,
                baseline=releases.previous.baseline,
                store=no_store,
            )

            outcome = rewrite_live_set(previous, expect=releases.expect, apply=True)

            assert outcome == WrongRelease(
                expected=releases.expect,
                running=previous.snapshot().migrations.fingerprint,
                running_labels=(),
            )

    def when_the_target_has_no_store():
        def it_raises_a_target_error(tmp_path: Path):
            releases = _releases(tmp_path)
            target = MigrationTarget(
                graph=releases.current.graph, baseline=releases.current.baseline
            )

            with pytest.raises(TargetError, match=r"store="):
                rewrite_live_set(target, expect=releases.expect, apply=False)

    def when_passes_is_below_one():
        def it_raises_a_value_error(tmp_path: Path):
            releases = _releases(tmp_path)

            with pytest.raises(ValueError, match=r"passes"):
                rewrite_live_set(
                    releases.current, expect=releases.expect, apply=True, passes=0
                )

    def when_apply_is_false():
        def it_plans_and_writes_nothing(tmp_path: Path):
            releases = _releases(tmp_path)
            releases.seed("t1")

            outcome = rewrite_live_set(
                releases.current, expect=releases.expect, apply=False
            )

            assert isinstance(outcome, LiveSetOutcome)
            assert outcome.passes == ()
            assert [t.status for t in outcome.final.threads] == ["rewrite"]
            assert not outcome.clean
            assert releases.plan("t1") == ["rewrite"]

    def when_apply_is_true():
        def it_rewrites_every_root_thread_until_a_plan_reads_clean(tmp_path: Path):
            releases = _releases(tmp_path)
            releases.seed("t1", "t2")

            outcome = rewrite_live_set(
                releases.current, expect=releases.expect, apply=True
            )

            assert isinstance(outcome, LiveSetOutcome)
            assert outcome.clean
            assert len(outcome.passes) == 1
            assert [t.status for t in outcome.final.threads] == [
                "unchanged",
                "unchanged",
            ]
            assert releases.plan("t1", "t2") == ["unchanged", "unchanged"]

        def it_logs_one_line_per_pass(tmp_path: Path):
            releases = _releases(tmp_path)
            releases.seed("t1")
            log = io.StringIO()

            rewrite_live_set(
                releases.current, expect=releases.expect, apply=True, log=log
            )

            assert log.getvalue() == (
                "pass 1: applied: 1 thread, 1 rewrite, 0 unchanged, 0 refused\n"
            )

    def when_a_run_writes_old_bytes_after_the_rewrite():
        def describe_and_one_pass_is_allowed():
            def it_reports_the_thread_as_not_clean(tmp_path: Path):
                saver = _StaleRunSaver()
                releases = _releases(tmp_path, saver)
                releases.seed("t1")
                saver.stale_runs = 1

                outcome = rewrite_live_set(
                    releases.current, expect=releases.expect, apply=True, passes=1
                )

                assert isinstance(outcome, LiveSetOutcome)
                assert [t.status for t in outcome.passes[0].threads] == ["rewrite"]
                assert [t.status for t in outcome.final.threads] == ["rewrite"]
                assert not outcome.clean

        def describe_and_a_second_pass_is_allowed():
            def it_rewrites_the_new_checkpoint_in_the_next_pass(tmp_path: Path):
                saver = _StaleRunSaver()
                releases = _releases(tmp_path, saver)
                releases.seed("t1")
                saver.stale_runs = 1

                outcome = rewrite_live_set(
                    releases.current, expect=releases.expect, apply=True, passes=2
                )

                assert isinstance(outcome, LiveSetOutcome)
                assert len(outcome.passes) == 2
                assert outcome.clean
                assert releases.plan("t1") == ["unchanged"]

    def when_a_subgraph_checkpoint_holds_an_old_shape():
        def it_reads_it_and_reports_the_run_as_not_clean(tmp_path: Path):
            releases = _releases(tmp_path)
            releases.seed("t1")
            _copy_to_namespace(releases.saver, "t1", "child:1")

            outcome = rewrite_live_set(
                releases.current, expect=releases.expect, apply=True
            )

            assert isinstance(outcome, LiveSetOutcome)
            assert [t.status for t in outcome.final.threads] == ["unchanged"]
            assert [s.key for s in outcome.subgraphs] == [ThreadKey("t1", "child:1")]
            assert outcome.subgraphs[0].thread.status == "rewrite"
            assert not outcome.clean

        def it_writes_nothing_under_the_subgraph_namespace(tmp_path: Path):
            releases = _releases(tmp_path)
            releases.seed("t1")
            _copy_to_namespace(releases.saver, "t1", "child:1")
            before = dict(releases.saver.storage["t1"]["child:1"])

            rewrite_live_set(releases.current, expect=releases.expect, apply=True)

            assert releases.saver.storage["t1"]["child:1"] == before
