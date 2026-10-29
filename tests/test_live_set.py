"""Tests for the live-set rewrite: ``MigrationTarget.thread_keys`` and
``rewrite_live_set``. The two-release fixtures live in ``_live_set_fixtures``.
"""

import contextlib
import io
from collections.abc import Iterator
from pathlib import Path

import pytest
from _dict_saver import DictSaver
from _live_set_fixtures import (
    _cfg,
    _copy_to_namespace,
    _graph,
    _noop,
    _releases,
)
from conftest import Started
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.base.id import uuid6
from langgraph.checkpoint.memory import MemorySaver

from langgraph_events import EventGraph
from langgraph_events.serde import NamespaceAwareSerde
from langgraph_events.serde.migrations import (
    LiveSetOutcome,
    Migration,
    MigrationTarget,
    TargetError,
    ThreadKey,
    TransformFields,
    WrongRelease,
    list_thread_keys,
    rewrite_live_set,
)


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

        def when_the_saver_is_a_postgres_saver():
            def it_lists_each_thread_and_namespace(tmp_path: Path, postgres_url: str):
                from langgraph.checkpoint.postgres import PostgresSaver

                with PostgresSaver.from_conn_string(postgres_url) as saver:
                    graph = _graph(saver)
                    graph.invoke(Started(), config=_cfg("b"))
                    graph.invoke(Started(), config=_cfg("a"))
                    _copy_to_namespace(saver, "a", "child:1")
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

        def when_the_saver_is_a_sqlite_saver():
            def it_lists_each_thread_and_namespace(tmp_path: Path):
                from langgraph.checkpoint.sqlite import SqliteSaver

                with SqliteSaver.from_conn_string(str(tmp_path / "s.db")) as saver:
                    graph = _graph(saver)
                    graph.invoke(Started(), config=_cfg("b"))
                    graph.invoke(Started(), config=_cfg("a"))
                    _copy_to_namespace(saver, "a", "child:1")
                    target = MigrationTarget(graph=_graph, baseline=tmp_path / "b.json")

                    assert target.thread_keys(saver) == (
                        ThreadKey("a", ""),
                        ThreadKey("a", "child:1"),
                        ThreadKey("b", ""),
                    )

            def when_its_internal_cursor_fails():
                def it_names_list_thread_keys(
                    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
                ):
                    from langgraph.checkpoint.sqlite import SqliteSaver

                    def gone(*_args: object, **_kwargs: object) -> None:
                        raise AttributeError("cursor")

                    with SqliteSaver.from_conn_string(str(tmp_path / "s.db")) as saver:
                        monkeypatch.setattr(saver, "cursor", gone)
                        target = MigrationTarget(
                            graph=_graph, baseline=tmp_path / "b.json"
                        )

                        with pytest.raises(TargetError) as raised:
                            target.thread_keys(saver)

                    message = str(raised.value)
                    assert message.startswith(
                        "The target cannot list the threads of a SqliteSaver "
                        "(AttributeError: cursor). Set threads= "
                    )
                    assert "threads=list_thread_keys" in message

        def when_the_saver_has_no_built_in_enumeration():
            def it_names_list_thread_keys_and_no_sql(tmp_path: Path):
                target = MigrationTarget(graph=_graph, baseline=tmp_path / "b.json")

                with pytest.raises(TargetError) as raised:
                    target.thread_keys(DictSaver())

                message = str(raised.value)
                assert message.startswith(
                    "The target cannot list the threads of a DictSaver. Set threads= "
                )
                assert "threads=list_thread_keys" in message
                assert "SELECT" not in message


def describe_list_thread_keys():
    def when_a_custom_saver_holds_threads_and_namespaces():
        def it_lists_each_one_once():
            saver = DictSaver()
            graph = _graph(saver)
            graph.invoke(Started(), config=_cfg("b"))
            graph.invoke(Started(), config=_cfg("a"))
            graph.invoke(Started(), config=_cfg("a"))
            _copy_to_namespace(saver, "a", "child:1")

            assert sorted(list_thread_keys(saver)) == [
                ThreadKey("a", ""),
                ThreadKey("a", "child:1"),
                ThreadKey("b", ""),
            ]

    def when_a_checkpoint_names_a_class_the_serde_cannot_revive():
        def it_still_lists_the_thread(tmp_path: Path):
            saver = DictSaver()
            releases = _releases(tmp_path, saver)
            releases.seed("t1")
            _graph(saver)

            assert list(list_thread_keys(saver)) == [ThreadKey("t1", "")]


class _StaleRun:
    """A run loaded the checkpoint before the rewrite and commits its next
    checkpoint after the rewrite proved that the thread did not move.

    The child keeps the parent's channel versions, so it points at the
    old blobs, which the rewrite does not delete. Mix it into a saver.
    """

    stale_runs = 0
    _loaded = None

    def put(self, config, checkpoint, metadata, new_versions):
        if self.stale_runs and self._loaded is None:
            self.stale_runs -= 1
            tid = config["configurable"]["thread_id"]
            self._loaded = super().get_tuple(_cfg(tid))  # type: ignore[misc]
        return super().put(config, checkpoint, metadata, new_versions)  # type: ignore[misc]

    def get_tuple(self, config):
        result = super().get_tuple(config)  # type: ignore[misc]
        loaded, self._loaded = self._loaded, None
        if loaded is not None:
            child = {**loaded.checkpoint, "id": str(uuid6())}
            super().put(loaded.config, child, loaded.metadata, {})  # type: ignore[misc]
        return result


class _StaleRunSaver(_StaleRun, MemorySaver):
    pass


@contextlib.contextmanager
def _stale_postgres_saver(url: str) -> Iterator[BaseCheckpointSaver]:
    from langgraph.checkpoint.postgres import PostgresSaver
    from psycopg import Connection
    from psycopg.rows import dict_row

    class _StalePostgresSaver(_StaleRun, PostgresSaver):
        pass

    with Connection.connect(
        url, autocommit=True, prepare_threshold=0, row_factory=dict_row
    ) as conn:
        yield _StalePostgresSaver(conn)


def _documented_prune_sql() -> str:
    """The SQL that *Old blob versions* in docs/event-migrations.md gives."""
    docs = Path(__file__).resolve().parents[1] / "docs" / "event-migrations.md"
    section = docs.read_text().split("### Old blob versions", 1)[1]
    return section.split("```sql\n", 1)[1].split("```", 1)[0]


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

    def when_the_plan_refuses_a_checkpoint():
        def _fail_the_task(saver: BaseCheckpointSaver, tid: str) -> None:
            tup = saver.get_tuple(_cfg(tid))
            assert tup is not None
            saver.put_writes(tup.config, [("__error__", "boom")], "task-1")

        def it_stops_after_the_last_pass_that_can_rewrite(tmp_path: Path):
            releases = _releases(tmp_path)
            releases.seed("t1", "t2")
            _fail_the_task(releases.saver, "t2")
            log = io.StringIO()

            outcome = rewrite_live_set(
                releases.current, expect=releases.expect, apply=True, log=log
            )

            assert isinstance(outcome, LiveSetOutcome)
            assert log.getvalue() == (
                "pass 1: applied: 1 thread, 1 rewrite, 0 unchanged, 0 refused\n"
            )
            assert [t.status for t in outcome.final.threads] == [
                "unchanged",
                "refused",
            ]
            assert not outcome.clean

        def when_it_refuses_every_checkpoint():
            def it_runs_no_pass(tmp_path: Path):
                releases = _releases(tmp_path)
                releases.seed("t1")
                _fail_the_task(releases.saver, "t1")

                outcome = rewrite_live_set(
                    releases.current, expect=releases.expect, apply=True
                )

                assert isinstance(outcome, LiveSetOutcome)
                assert outcome.passes == ()
                assert [t.status for t in outcome.final.threads] == ["refused"]

    def when_a_run_writes_old_bytes_after_the_rewrite():
        def when_one_pass_is_allowed():
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

        def when_a_second_pass_is_allowed():
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

        def when_the_store_is_postgres():
            def it_rewrites_the_new_checkpoint_in_the_next_pass(
                tmp_path: Path, postgres_url: str
            ):
                with _stale_postgres_saver(postgres_url) as saver:
                    releases = _releases(tmp_path, saver)
                    releases.seed("t1")
                    saver.stale_runs = 1

                    outcome = rewrite_live_set(
                        releases.current, expect=releases.expect, apply=True
                    )

                    assert isinstance(outcome, LiveSetOutcome)
                    assert [
                        [t.status for t in report.threads] for report in outcome.passes
                    ] == [["rewrite"], ["rewrite"]]
                    assert outcome.clean
                    assert releases.plan("t1") == ["unchanged"]

    def when_a_subgraph_checkpoint_holds_the_live_shape():
        def it_reports_the_run_as_clean(tmp_path: Path):
            releases = _releases(tmp_path)
            releases.seed("t1")
            rewrite_live_set(releases.current, expect=releases.expect, apply=True)
            _copy_to_namespace(releases.saver, "t1", "child:1")

            outcome = rewrite_live_set(
                releases.current, expect=releases.expect, apply=True
            )

            assert isinstance(outcome, LiveSetOutcome)
            assert outcome.passes == ()
            assert [t.location for t in outcome.final.threads] == [
                "t1",
                "t1 in subgraph namespace 'child:1'",
            ]
            assert outcome.clean

    def when_a_subgraph_checkpoint_holds_an_old_shape():
        def when_apply_is_false():
            def it_reports_the_run_as_not_clean(tmp_path: Path):
                releases = _releases(tmp_path)
                releases.seed("t1")
                _copy_to_namespace(releases.saver, "t1", "child:1")

                outcome = rewrite_live_set(
                    releases.current, expect=releases.expect, apply=False
                )

                assert isinstance(outcome, LiveSetOutcome)
                assert [(t.checkpoint_ns, t.status) for t in outcome.final.threads] == [
                    ("", "rewrite"),
                    ("child:1", "rewrite"),
                ]
                assert not outcome.clean

        def when_apply_is_true():
            def it_rewrites_it_until_a_plan_reads_clean(tmp_path: Path):
                releases = _releases(tmp_path)
                releases.seed("t1")
                _copy_to_namespace(releases.saver, "t1", "child:1")

                outcome = rewrite_live_set(
                    releases.current, expect=releases.expect, apply=True
                )

                assert isinstance(outcome, LiveSetOutcome)
                assert outcome.clean
                assert [
                    (t.checkpoint_ns, t.status) for t in outcome.passes[0].threads
                ] == [("", "rewrite"), ("child:1", "rewrite")]
                graph = releases.current.graph(releases.saver)
                [child] = graph.plan_rewrite(
                    thread_ids=["t1"], checkpoint_ns="child:1"
                ).threads
                assert child.status == "unchanged"

        def when_the_store_is_postgres():
            def it_rewrites_it_until_a_plan_reads_clean(
                tmp_path: Path, postgres_url: str
            ):
                from langgraph.checkpoint.postgres import PostgresSaver

                with PostgresSaver.from_conn_string(postgres_url) as saver:
                    releases = _releases(tmp_path, saver)
                    releases.seed("t1")
                    _copy_to_namespace(saver, "t1", "child:1")

                    outcome = rewrite_live_set(
                        releases.current, expect=releases.expect, apply=True
                    )

                    assert isinstance(outcome, LiveSetOutcome)
                    assert outcome.clean
                    assert len(outcome.passes) == 1
                    assert releases.plan("t1") == ["unchanged"]
                    graph = releases.current.graph(saver)
                    [child] = graph.plan_rewrite(
                        thread_ids=["t1"], checkpoint_ns="child:1"
                    ).threads
                    assert child.status == "unchanged"

    def when_a_subgraph_checkpoint_holds_a_failed_task_write():
        def _stuck(tmp_path: Path):
            releases = _releases(tmp_path)
            releases.seed("t1", "t2")
            _copy_to_namespace(releases.saver, "t1", "child:1")
            child = releases.saver.get_tuple(_cfg("t1", "child:1"))
            assert child is not None
            releases.saver.put_writes(child.config, [("__error__", "boom")], "task-1")
            return releases

        def it_reports_the_run_as_not_clean_after_every_pass(tmp_path: Path):
            releases = _stuck(tmp_path)

            outcome = rewrite_live_set(
                releases.current, expect=releases.expect, apply=True
            )

            assert isinstance(outcome, LiveSetOutcome)
            assert [t.location for t in outcome.final.refused] == [
                "t1 in subgraph namespace 'child:1'"
            ]
            assert "delete_thread()" in outcome.final.refused[0].reason

        def when_the_thread_is_deleted():
            def it_reports_the_run_as_clean(tmp_path: Path):
                releases = _stuck(tmp_path)
                rewrite_live_set(releases.current, expect=releases.expect, apply=True)
                releases.saver.delete_thread("t1")

                outcome = rewrite_live_set(
                    releases.current, expect=releases.expect, apply=True
                )

                assert isinstance(outcome, LiveSetOutcome)
                assert outcome.clean

    def when_the_store_saver_holds_its_own_serde():
        def it_raises_a_target_error_before_it_reads_a_thread(tmp_path: Path):
            releases = _releases(tmp_path)
            releases.seed("t1")
            own = NamespaceAwareSerde(
                [
                    Migration(
                        operations=(
                            TransformFields(
                                Started.__module__, Started.__qualname__, _keep
                            ),
                        )
                    )
                ],
                events=(Started,),
            )
            store = MemorySaver(serde=own)

            def from_namespaces(saver):
                return EventGraph.from_namespaces(handlers=[_noop], checkpointer=saver)

            target = MigrationTarget(
                graph=from_namespaces,
                baseline=tmp_path / "b.json",
                store=lambda: contextlib.nullcontext(store),
                threads=_no_read,
            )
            expect = target.snapshot().migrations.fingerprint

            with pytest.raises(TargetError) as raised:
                rewrite_live_set(target, expect=expect, apply=True)

            assert str(raised.value) == (
                f"The saver that store= opens holds its own serde, with the "
                f"migrations {own.client_migrations().fingerprint}. The graph "
                f"built on an empty saver holds {expect}, and the gate and the "
                f"--expect fence judge those. Set the serde in the graph "
                f"factory, not in the store."
            )


def _keep(kw: dict) -> dict:
    return kw


def _no_read(saver: BaseCheckpointSaver) -> list[ThreadKey]:
    raise AssertionError("the rewrite read the store")


def describe_the_documented_prune_of_old_blob_versions():
    def _blob_count(url: str) -> int:
        import psycopg

        with psycopg.connect(url) as conn:
            row = conn.execute("SELECT count(*) FROM checkpoint_blobs").fetchone()
        assert row is not None
        return int(row[0])

    def _unnamed_blob_count(url: str) -> int:
        import psycopg

        with psycopg.connect(url) as conn:
            row = conn.execute(
                "SELECT count(*) FROM checkpoint_blobs AS b WHERE NOT EXISTS ("
                "SELECT 1 FROM checkpoints AS c WHERE c.thread_id = b.thread_id"
                " AND c.checkpoint_ns = b.checkpoint_ns AND "
                "c.checkpoint -> 'channel_versions' ->> b.channel = b.version)"
            ).fetchone()
        assert row is not None
        return int(row[0])

    def _prune(url: str) -> None:
        import psycopg

        with psycopg.connect(url, autocommit=True) as conn:
            conn.execute(_documented_prune_sql())  # type: ignore[arg-type]

    def it_deletes_each_blob_version_the_rewrite_left_unnamed(
        tmp_path: Path, postgres_url: str
    ):
        from langgraph.checkpoint.postgres import PostgresSaver

        with PostgresSaver.from_conn_string(postgres_url) as saver:
            releases = _releases(tmp_path, saver)
            releases.seed("t1", "t2")
            assert _unnamed_blob_count(postgres_url) == 0
            rewrite_live_set(releases.current, expect=releases.expect, apply=True)
            assert _unnamed_blob_count(postgres_url) == 2

            _prune(postgres_url)

            assert _unnamed_blob_count(postgres_url) == 0
            assert releases.plan("t1", "t2") == ["unchanged", "unchanged"]
            graph = releases.current.graph(saver)
            notes = [e.note for e in graph.get_state(_cfg("t1")).events[1:]]
            assert notes == ["t1"]

    def it_keeps_a_blob_newer_than_each_version_a_checkpoint_names(
        tmp_path: Path, postgres_url: str
    ):
        import psycopg
        from langgraph.checkpoint.postgres import PostgresSaver

        with PostgresSaver.from_conn_string(postgres_url) as saver:
            releases = _releases(tmp_path, saver)
            releases.seed("t1")
            newer = saver.get_next_version(
                saver.get_tuple(_cfg("t1")).checkpoint["channel_versions"]["events"],
                None,
            )
        with psycopg.connect(postgres_url, autocommit=True) as conn:
            conn.execute(
                "INSERT INTO checkpoint_blobs (thread_id, checkpoint_ns, "
                "channel, version, type, blob) VALUES "
                "('t1', '', 'events', %s, 'empty', NULL)",
                (newer,),
            )
        before = _blob_count(postgres_url)

        _prune(postgres_url)

        assert _blob_count(postgres_url) == before
