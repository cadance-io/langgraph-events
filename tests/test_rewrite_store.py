"""EventGraph.plan_rewrite() / rewrite_store(): the apply-side migration (#179).

A retired identity stays in the ``events`` channel of a thread's latest
checkpoint until something rewrites the stored bytes. These suites pin
that rewrite at the API boundary: what the plan reports, what the store
holds after the write, and when a thread is refused.

Idiom, shared with ``tests/test_event_graph.py``: a class defined inside
a helper never imports, so a serde swap on the saver simulates "the
class was renamed" or "the class was deleted" against real bytes.
"""

import typing

import pytest
from conftest import Ended, Started
from langgraph.checkpoint.base.id import uuid6
from langgraph.checkpoint.memory import MemorySaver
from test_event_graph import _AsyncOnlySaver

from langgraph_events import (
    EventGraph,
    IntegrationEvent,
    Interrupted,
    Reducer,
    Resumed,
    RewriteReport,
    ThreadRewrite,
    on,
)
from langgraph_events.serde import NamespaceAwareSerde, backfill, migrate_from
from langgraph_events.serde._jsonplus import _scan_identities


class _Go(IntegrationEvent):
    pass


class _Again(IntegrationEvent):
    pass


class _Inner(IntegrationEvent):
    pass


class _Carrier(IntegrationEvent):
    inner: _Inner | None = None


class _SideDone(IntegrationEvent):
    pass


@on(Started)
def _side_effect(event: Started) -> _SideDone:
    """Fan-out sibling that completes in the superstep another pauses."""
    return _SideDone()


class _AdvancingSaver(MemorySaver):
    """A run lands a newer checkpoint on the thread while the rewrite is
    writing: the put succeeds, but the latest id is no longer the one
    the plan was built from."""

    def put(self, config, checkpoint, metadata, new_versions):  # type: ignore[override]
        result = super().put(config, checkpoint, metadata, new_versions)
        newer = {**checkpoint, "id": str(uuid6())}
        super().put(result, newer, metadata, {})
        return result


@on(_Go)
def _go_ends(event: _Go) -> Ended:
    return Ended(result="went")


@on(Started)
def _completes(event: Started) -> None:
    return None


def _cfg(tid: str) -> dict[str, typing.Any]:
    return {"configurable": {"thread_id": tid}}


def _identity(cls: type) -> str:
    return f"{cls.__module__}:{cls.__qualname__}"


def _stored_identities(saver: MemorySaver, tid: str) -> set[tuple[str, str]]:
    """Every event identity in the thread's latest checkpoint blobs, read
    from the bytes, not through the serde."""
    tup = saver.get_tuple(_cfg(tid))
    assert tup is not None
    found: set[tuple[str, str]] = set()
    for channel, version in tup.checkpoint["channel_versions"].items():
        blob = saver.blobs.get((tid, "", channel, version))
        if blob is None or blob[0] != "msgpack":
            continue
        found.update(_scan_identities(blob[1]))
    return found


async def _astored_identities(saver: MemorySaver, tid: str) -> set[tuple[str, str]]:
    """Async sibling of ``_stored_identities``, via ``aget_tuple``."""
    tup = await saver.aget_tuple(_cfg(tid))
    assert tup is not None
    found: set[tuple[str, str]] = set()
    for channel, version in tup.checkpoint["channel_versions"].items():
        blob = saver.blobs.get((tid, "", channel, version))
        if blob is None or blob[0] != "msgpack":
            continue
        found.update(_scan_identities(blob[1]))
    return found


def _renamed_history_pair(saver: MemorySaver, tid: str):
    """A settled thread whose history holds an interrupt written under an
    identity that release N+1 renamed with ``@migrate_from``.

    Returns the release N+1 graph, the config, the old class and the new
    class.
    """

    class _OldGate(Interrupted):
        pass

    @on(Started)
    def wait(event: Started) -> _OldGate:
        return _OldGate()

    cfg = _cfg(tid)
    saver.serde = NamespaceAwareSerde(events=(Started, _OldGate))
    graph = EventGraph([wait, _go_ends], checkpointer=saver)
    graph.invoke(Started(data="x"), config=cfg)
    graph.resume(_Go(), config=cfg)
    assert graph.get_state(cfg).events.has(Ended)

    @migrate_from(_OldGate.__qualname__, in_module=_OldGate.__module__)
    class _NewGate(Interrupted):
        pass

    saver.serde = NamespaceAwareSerde(events=(Started, _NewGate))
    return EventGraph([_completes], checkpointer=saver), cfg, _OldGate, _NewGate


def _settled_drop_pair(saver: MemorySaver, tid: str):
    """A settled thread whose history holds an interrupt on a class that is
    still live and about to be retired. Returns the graph, the config and
    the class."""

    class _Retiring(Interrupted):
        pass

    @on(Started)
    def wait(event: Started) -> _Retiring:
        return _Retiring()

    cfg = _cfg(tid)
    saver.serde = NamespaceAwareSerde(events=(Started, _Retiring))
    graph = EventGraph([wait, _go_ends], checkpointer=saver)
    graph.invoke(Started(data="x"), config=cfg)
    graph.resume(_Go(), config=cfg)
    assert graph.get_state(cfg).events.has(Ended)
    return graph, cfg, _Retiring


def _paused_renamed_pair(saver: MemorySaver, tid: str):
    """A thread paused on a live interrupt, whose settled history holds an
    interrupt written under an identity release N+1 renamed.

    Returns the release N+1 graph, the config, and a dict of handler fire
    counts that the graph's handlers update.
    """

    class _OldGate(Interrupted):
        pass

    class _LivePause(Interrupted):
        pass

    fired: dict[str, int] = {"wait_old": 0, "wait_live": 0, "finish": 0}

    @on(Started)
    def wait_old(event: Started) -> _OldGate:
        fired["wait_old"] += 1
        return _OldGate()

    @on(_Go)
    def wait_live(event: _Go) -> _LivePause:
        fired["wait_live"] += 1
        return _LivePause()

    @on(_Again)
    def finish(event: _Again) -> Ended:
        fired["finish"] += 1
        return Ended(result="finished")

    cfg = _cfg(tid)
    saver.serde = NamespaceAwareSerde(events=(Started, _OldGate, _LivePause))
    graph = EventGraph([wait_old, wait_live, finish], checkpointer=saver)
    graph.invoke(Started(data="x"), config=cfg)
    graph.resume(_Go(), config=cfg)
    assert graph.get_state(cfg).is_interrupted
    fired.update(wait_old=0, wait_live=0, finish=0)

    @migrate_from(_OldGate.__qualname__, in_module=_OldGate.__module__)
    class _NewGate(Interrupted):
        pass

    saver.serde = NamespaceAwareSerde(events=(Started, _NewGate, _LivePause))
    return EventGraph([wait_old, wait_live, finish], checkpointer=saver), cfg, fired


async def _arenamed_history_pair(saver: MemorySaver, tid: str):
    """Async sibling of ``_renamed_history_pair``, built through
    ``ainvoke``/``aresume`` so an async-only saver accepts it."""

    class _OldGate(Interrupted):
        pass

    @on(Started)
    def wait(event: Started) -> _OldGate:
        return _OldGate()

    cfg = _cfg(tid)
    saver.serde = NamespaceAwareSerde(events=(Started, _OldGate))
    graph = EventGraph([wait, _go_ends], checkpointer=saver)
    await graph.ainvoke(Started(data="x"), config=cfg)
    await graph.aresume(_Go(), config=cfg)

    @migrate_from(_OldGate.__qualname__, in_module=_OldGate.__module__)
    class _NewGate(Interrupted):
        pass

    saver.serde = NamespaceAwareSerde(events=(Started, _NewGate))
    return EventGraph([_completes], checkpointer=saver), cfg, _OldGate, _NewGate


def describe_plan_rewrite():
    def when_preconditions_fail():
        def it_requires_a_checkpointer():
            with pytest.raises(ValueError, match=r"plan_rewrite\(\) requires"):
                EventGraph([_completes]).plan_rewrite()

        def it_requires_a_namespace_aware_serde():
            graph = EventGraph([_completes], checkpointer=MemorySaver())

            with pytest.raises(ValueError, match=r"plan_rewrite\(\) needs a Namespace"):
                graph.plan_rewrite()

        def it_refuses_legacy_write():
            saver = MemorySaver()
            saver.serde = NamespaceAwareSerde(events=(Started,), legacy_write=True)
            graph = EventGraph([_completes], checkpointer=saver)

            with pytest.raises(ValueError, match=r"legacy_write"):
                graph.plan_rewrite()

        def it_refuses_a_drop_class_the_serde_cannot_revive():
            class _Unknown(IntegrationEvent):
                pass

            saver = MemorySaver()
            saver.serde = NamespaceAwareSerde(events=(Started,))
            graph = EventGraph([_completes], checkpointer=saver)

            with pytest.raises(ValueError, match=r"_Unknown"):
                graph.plan_rewrite(drop=(_Unknown,))

    def when_a_thread_holds_a_renamed_identity():
        def it_reports_the_migrated_pair():
            saver = MemorySaver()
            graph, _config, old, new = _renamed_history_pair(saver, "t1")

            report = graph.plan_rewrite()

            assert report.applied is False
            [thread] = report.threads
            assert thread.thread_id == "t1"
            assert thread.status == "rewrite"
            assert thread.migrated == ((_identity(old), _identity(new)),)
            assert thread.dropped == {}

        def it_writes_nothing():
            saver = MemorySaver()
            graph, cfg, _old, _new = _renamed_history_pair(saver, "t1")
            blobs_before = set(saver.blobs)
            versions_before = saver.get_tuple(cfg).checkpoint["channel_versions"]

            graph.plan_rewrite()

            assert set(saver.blobs) == blobs_before
            after = saver.get_tuple(cfg).checkpoint["channel_versions"]
            assert after == versions_before

    def when_a_thread_holds_no_migrated_identity():
        def it_reports_unchanged():
            saver = MemorySaver()
            saver.serde = NamespaceAwareSerde(events=(Started,))
            graph = EventGraph([_completes], checkpointer=saver)
            graph.invoke(Started(data="x"), config=_cfg("plain"))

            [thread] = graph.plan_rewrite().threads

            assert thread.status == "unchanged"
            assert thread.migrated == ()
            assert thread.dropped == {}


def describe_rewrite_store():
    def when_a_thread_holds_a_renamed_identity():
        def it_stores_the_live_identity_and_reports_the_rewrite():
            saver = MemorySaver()
            graph, _config, old, new = _renamed_history_pair(saver, "t1")
            assert (old.__module__, old.__qualname__) in _stored_identities(saver, "t1")

            report = graph.rewrite_store()

            assert report.applied is True
            [thread] = report.threads
            assert thread.status == "rewrite"
            assert thread.migrated == ((_identity(old), _identity(new)),)
            stored = _stored_identities(saver, "t1")
            assert (old.__module__, old.__qualname__) not in stored
            assert (new.__module__, new.__qualname__) in stored

    def when_drop_names_a_live_class():
        def it_removes_the_stored_events_and_lowers_the_cursor():
            saver = MemorySaver()
            graph, cfg, retiring = _settled_drop_pair(saver, "t1")

            report = graph.rewrite_store(drop=(retiring,))

            [thread] = report.threads
            assert thread.status == "rewrite"
            # The stored event. The Resumed.interrupted back-reference is
            # cleared, not counted: the count matches the log entries.
            assert thread.dropped == {_identity(retiring): 1}
            values = saver.get_tuple(cfg).checkpoint["channel_values"]
            assert not any(isinstance(e, retiring) for e in values["events"])
            assert not any(isinstance(e, retiring) for e in values["_pending"])
            assert values["_cursor"] == len(values["events"])
            [resumed] = [e for e in values["events"] if isinstance(e, Resumed)]
            assert resumed.interrupted is None
            assert resumed.value == _Go()

        def it_leaves_nothing_for_unrevivable_threads_once_the_class_is_gone():
            saver = MemorySaver()
            graph, _config, retiring = _settled_drop_pair(saver, "t1")
            graph.rewrite_store(drop=(retiring,))

            saver.serde = NamespaceAwareSerde(events=(Started,))
            after = EventGraph([_completes], checkpointer=saver)

            assert after.unrevivable_threads() == {}

        def it_dispatches_the_next_input_on_the_thread():
            saver = MemorySaver()
            graph, cfg, retiring = _settled_drop_pair(saver, "t1")
            graph.rewrite_store(drop=(retiring,))
            fired: list[int] = []

            @on(_Again)
            def again(event: _Again) -> Ended:
                fired.append(1)
                return Ended(result="again")

            log = EventGraph([again], checkpointer=saver).invoke(_Again(), config=cfg)

            assert fired == [1]
            assert log.latest(Ended) == Ended(result="again")

    def when_drop_names_a_base_class():
        def it_leaves_a_subclass_instance_in_place():
            # drop= matches the stored identity, not the class hierarchy,
            # the same rule validate_drop() and the byte scan apply.
            saver = MemorySaver()

            class _Base(Interrupted):
                pass

            class _Sub(_Base):
                pass

            @on(Started)
            def wait(event: Started) -> _Sub:
                return _Sub()

            cfg = _cfg("t1")
            saver.serde = NamespaceAwareSerde(events=(Started, _Base, _Sub))
            graph = EventGraph([wait, _go_ends], checkpointer=saver)
            graph.invoke(Started(data="x"), config=cfg)
            graph.resume(_Go(), config=cfg)

            [thread] = graph.rewrite_store(drop=(_Base,)).threads

            assert thread.status == "unchanged"

    def when_a_live_class_carries_a_fill():
        def it_converges_on_the_second_run():
            saver = MemorySaver()

            class _OldGate(Interrupted):
                pass

            @on(Started)
            def wait(event: Started) -> _OldGate:
                return _OldGate()

            cfg = _cfg("t1")
            saver.serde = NamespaceAwareSerde(events=(Started, _OldGate))
            graph = EventGraph([wait, _go_ends], checkpointer=saver)
            graph.invoke(Started(data="x"), config=cfg)
            graph.resume(_Go(), config=cfg)

            @backfill("flag", default=True)
            @migrate_from(_OldGate.__qualname__, in_module=_OldGate.__module__)
            class _NewGate(Interrupted):
                flag: bool

            saver.serde = NamespaceAwareSerde(events=(Started, _NewGate))
            graph = EventGraph([_completes], checkpointer=saver)
            blobs_before = len(saver.blobs)

            [first] = graph.rewrite_store().threads
            [second] = graph.rewrite_store().threads

            assert first.status == "rewrite"
            assert second.status == "unchanged"
            assert len(saver.blobs) > blobs_before
            assert graph.plan_rewrite().threads[0].status == "unchanged"

    def when_a_thread_is_paused_on_a_live_interrupt():
        def it_rewrites_the_history_and_the_thread_still_resumes():
            saver = MemorySaver()
            graph, cfg, fired = _paused_renamed_pair(saver, "t1")

            [thread] = graph.rewrite_store().threads
            log = graph.resume(_Again(), config=cfg)

            assert thread.status == "rewrite"
            assert log.latest(Ended) == Ended(result="finished")
            # The paused handler re-runs once on resume by LangGraph design.
            assert fired == {"wait_old": 0, "wait_live": 1, "finish": 1}

    def when_a_thread_has_one_checkpoint():
        def it_rewrites_the_thread():
            saver = MemorySaver()
            cfg = _cfg("seeded")

            class _OldGate(Interrupted):
                pass

            saver.serde = NamespaceAwareSerde(events=(Started, _OldGate))
            EventGraph([_completes], checkpointer=saver).pre_seed(
                cfg, {"events": [_OldGate()]}
            )
            assert saver.get_tuple(cfg).parent_config is None

            @migrate_from(_OldGate.__qualname__, in_module=_OldGate.__module__)
            class _NewGate(Interrupted):
                pass

            saver.serde = NamespaceAwareSerde(events=(Started, _NewGate))
            graph = EventGraph([_completes], checkpointer=saver)

            [thread] = graph.rewrite_store().threads

            assert thread.status == "rewrite"
            assert (_NewGate.__module__, _NewGate.__qualname__) in _stored_identities(
                saver, "seeded"
            )

        def it_keeps_the_checkpoint_id_and_reports_unchanged_on_a_second_run():
            saver = MemorySaver()
            graph, cfg, _old, _new = _renamed_history_pair(saver, "t1")
            checkpoint_id = saver.get_tuple(cfg).checkpoint["id"]

            graph.rewrite_store()

            assert saver.get_tuple(cfg).checkpoint["id"] == checkpoint_id
            [thread] = graph.plan_rewrite().threads
            assert thread.status == "unchanged"


def describe_rewrite_store_refusals():
    # Each refusal names what to do. None writes anything.

    def _only(report):
        [thread] = report.threads
        assert thread.status == "refused", thread
        return thread

    def when_the_history_cannot_revive():
        def it_asks_for_a_tombstone_first():
            saver = MemorySaver()

            class _Deleted(Interrupted):
                pass

            @on(Started)
            def wait(event: Started) -> _Deleted:
                return _Deleted()

            cfg = _cfg("t1")
            saver.serde = NamespaceAwareSerde(events=(Started, _Deleted))
            graph = EventGraph([wait, _go_ends], checkpointer=saver)
            graph.invoke(Started(data="x"), config=cfg)
            graph.resume(_Go(), config=cfg)
            saver.serde = NamespaceAwareSerde(events=(Started,))
            with saver.serde.tolerate_unresolved():
                before = saver.get_tuple(cfg).checkpoint["channel_versions"]

            thread = _only(EventGraph([_completes], checkpointer=saver).rewrite_store())

            assert "_Deleted" in thread.reason
            assert "tombstone" in thread.reason
            with saver.serde.tolerate_unresolved():
                assert saver.get_tuple(cfg).checkpoint["channel_versions"] == before

    def when_a_completed_sibling_write_is_pending():
        def it_asks_to_resume_or_abandon_first():
            saver = MemorySaver()

            class _OldGate(Interrupted):
                pass

            @on(Started)
            def wait(event: Started) -> _OldGate:
                return _OldGate()

            cfg = _cfg("t1")
            saver.serde = NamespaceAwareSerde(events=(Started, _OldGate))
            EventGraph([wait, _side_effect], checkpointer=saver).invoke(
                Started(data="x"), config=cfg
            )
            channels = {c for _t, c, _v in saver.get_tuple(cfg).pending_writes}
            assert "__interrupt__" in channels and len(channels) > 1

            @migrate_from(_OldGate.__qualname__, in_module=_OldGate.__module__)
            class _NewGate(Interrupted):
                pass

            saver.serde = NamespaceAwareSerde(events=(Started, _NewGate))
            graph = EventGraph([wait, _side_effect], checkpointer=saver)

            thread = _only(graph.rewrite_store())

            assert "completed task write" in thread.reason
            assert "resume or abandon" in thread.reason

    def when_the_pending_interrupt_is_a_dropped_class():
        def it_asks_to_abandon_first():
            saver = MemorySaver()

            class _Retiring(Interrupted):
                pass

            @on(Started)
            def wait(event: Started) -> _Retiring:
                return _Retiring()

            cfg = _cfg("t1")
            saver.serde = NamespaceAwareSerde(events=(Started, _Retiring))
            graph = EventGraph([wait], checkpointer=saver)
            graph.invoke(Started(data="x"), config=cfg)

            thread = _only(graph.rewrite_store(drop=(_Retiring,)))

            assert "paused on" in thread.reason
            assert "abandon" in thread.reason

    def when_a_dropped_event_is_pending_dispatch_on_a_paused_thread():
        def it_asks_to_resume_or_abandon_first():
            saver = MemorySaver()

            class _Retiring(Interrupted):
                pass

            class _LivePause(Interrupted):
                pass

            @on(Started)
            def wait(event: Started) -> _Retiring:
                return _Retiring()

            @on(_Retiring)
            def after(event: _Retiring) -> _LivePause:
                return _LivePause()

            cfg = _cfg("t1")
            saver.serde = NamespaceAwareSerde(events=(Started, _Retiring, _LivePause))
            graph = EventGraph([wait, after], checkpointer=saver)
            graph.invoke(Started(data="x"), config=cfg)
            graph.resume(_Go(), config=cfg)
            pending = saver.get_tuple(cfg).checkpoint["channel_values"]["_pending"]
            assert any(isinstance(e, _Retiring) for e in pending)

            thread = _only(graph.rewrite_store(drop=(_Retiring,)))

            assert "pending dispatch" in thread.reason
            assert "resume or abandon" in thread.reason

    def when_a_dropped_event_sits_in_a_reducer_value():
        def it_refuses_naming_the_channel():
            saver = MemorySaver()

            class _Retiring(Interrupted):
                pass

            @on(Started)
            def wait(event: Started) -> _Retiring:
                return _Retiring()

            gates = Reducer("gates", event_type=_Retiring, fn=lambda e: [e])
            cfg = _cfg("t1")
            saver.serde = NamespaceAwareSerde(events=(Started, _Retiring))
            graph = EventGraph([wait, _go_ends], reducers=[gates], checkpointer=saver)
            graph.invoke(Started(data="x"), config=cfg)
            graph.resume(_Go(), config=cfg)
            values = saver.get_tuple(cfg).checkpoint["channel_values"]
            assert any(isinstance(e, _Retiring) for e in values["gates"])

            thread = _only(graph.rewrite_store(drop=(_Retiring,)))

            assert " remains in channel 'gates' after the drop; " in thread.reason
            assert thread.reason.endswith("leave the class in place")

    def when_a_dropped_event_is_nested_in_a_field():
        def it_refuses_naming_the_channel():
            saver = MemorySaver()

            @on(Started)
            def carry(event: Started) -> _Carrier:
                return _Carrier(inner=_Inner())

            cfg = _cfg("t1")
            saver.serde = NamespaceAwareSerde(events=(Started, _Carrier, _Inner))
            graph = EventGraph([carry], checkpointer=saver)
            graph.invoke(Started(data="x"), config=cfg)

            thread = _only(graph.rewrite_store(drop=(_Inner,)))

            assert thread.reason.startswith(
                "_Inner remains in channel 'events' after the drop; "
            )

    def when_a_thread_id_has_no_checkpoint():
        def it_reports_the_id_so_a_typo_is_visible():
            saver = MemorySaver()
            graph, _config, _old, _new = _renamed_history_pair(saver, "t1")

            report = graph.plan_rewrite(thread_ids=["t1", "t-typo"])

            assert [(t.thread_id, t.status) for t in report.threads] == [
                ("t1", "rewrite"),
                ("t-typo", "refused"),
            ]
            assert report.threads[1].reason == "no checkpoint for this thread id"

    def when_a_failed_task_left_an_error_write():
        def it_asks_to_run_or_abandon_the_thread():
            saver = MemorySaver()

            class _OldGate(Interrupted):
                pass

            @on(Started)
            def wait(event: Started) -> _OldGate:
                return _OldGate()

            @on(_Go)
            def explode(event: _Go) -> Ended:
                raise RuntimeError("boom")

            cfg = _cfg("t1")
            saver.serde = NamespaceAwareSerde(events=(Started, _OldGate))
            graph = EventGraph([wait, explode], checkpointer=saver)
            graph.invoke(Started(data="x"), config=cfg)
            with pytest.raises(RuntimeError, match="boom"):
                graph.resume(_Go(), config=cfg)
            channels = {c for _t, c, _v in saver.get_tuple(cfg).pending_writes}
            assert "__error__" in channels

            @migrate_from(_OldGate.__qualname__, in_module=_OldGate.__module__)
            class _NewGate(Interrupted):
                pass

            saver.serde = NamespaceAwareSerde(events=(Started, _NewGate))
            graph = EventGraph([wait, explode], checkpointer=saver)

            thread = _only(graph.rewrite_store())

            assert thread.reason == (
                "thread has a pending __error__ write from a failed task; run "
                "it again or abandon it, then rerun"
            )

    def when_a_value_cannot_be_encoded():
        def it_refuses_the_thread_instead_of_aborting_the_walk():
            from langgraph_events._rewrite import _verify_value

            serde = NamespaceAwareSerde(events=(Started,))

            problem, _identities = _verify_value(
                serde, [Started(), object()], "events", set(), set()
            )

            assert problem is not None
            assert problem.startswith("cannot encode the value")

    def when_the_thread_advances_during_the_write():
        def it_reports_the_thread_and_asks_for_a_rerun():
            saver = _AdvancingSaver()
            graph, _config, _old, _new = _renamed_history_pair(saver, "t1")

            thread = _only(graph.rewrite_store())

            assert thread.reason == "thread advanced during the rewrite; rerun"


def describe_rewrite_report():
    def it_is_exported_at_the_top_level():
        saver = MemorySaver()
        graph, _config, _old, _new = _renamed_history_pair(saver, "t1")

        report = graph.plan_rewrite()

        assert isinstance(report, RewriteReport)
        assert all(isinstance(t, ThreadRewrite) for t in report.threads)

    def it_limits_the_walk_to_thread_ids():
        saver = MemorySaver()
        _renamed_history_pair(saver, "t1")
        graph, _config, _old, _new = _renamed_history_pair(saver, "t2")

        report = graph.plan_rewrite(thread_ids=["t2"])

        assert [t.thread_id for t in report.threads] == ["t2"]

    def it_prints_a_summary_line_then_each_refused_thread():
        saver = MemorySaver()
        _settled_drop_pair(saver, "gone-a")
        _renamed_history_pair(saver, "gone-b")
        saver.serde = NamespaceAwareSerde(events=(Started,))
        graph = EventGraph([_completes], checkpointer=saver)
        graph.invoke(Started(data="x"), config=_cfg("plain"))

        report = graph.plan_rewrite()

        lines = str(report).splitlines()
        assert lines[0] == "plan: 3 threads, 0 rewrite, 1 unchanged, 2 refused"
        assert [t.thread_id for t in report.refused] == ["gone-a", "gone-b"]
        assert lines[1].startswith("  gone-a: refused, history names")
        assert lines[2].startswith("  gone-b: refused, history names")

    def it_says_rewritten_once_applied():
        saver = MemorySaver()
        graph, _config, _old, _new = _renamed_history_pair(saver, "t1")

        report = graph.rewrite_store()

        assert str(report) == "applied: 1 thread, 1 rewrite, 0 unchanged, 0 refused"


def describe_async_twins():
    # The consumer's AsyncPostgresSaver refuses a sync checkpointer call
    # from the running loop. Every read and write must go through the
    # async API.

    async def it_plans_through_the_async_saver():
        saver = _AsyncOnlySaver()
        graph, _config, old, new = await _arenamed_history_pair(saver, "t1")

        report = await graph.aplan_rewrite()

        assert report.applied is False
        [thread] = report.threads
        assert thread.migrated == ((_identity(old), _identity(new)),)

    async def it_rewrites_through_the_async_saver():
        saver = _AsyncOnlySaver()
        graph, _config, old, new = await _arenamed_history_pair(saver, "t1")

        report = await graph.arewrite_store()

        assert report.applied is True
        [thread] = report.threads
        assert thread.status == "rewrite"
        stored = await _astored_identities(saver, "t1")
        assert (old.__module__, old.__qualname__) not in stored
        assert (new.__module__, new.__qualname__) in stored
        [again] = (await graph.aplan_rewrite()).threads
        assert again.status == "unchanged"


def describe_documented_retirement_sequence():
    # The sequence under "Retiring an Interrupted subclass" in
    # docs/event-migrations.md, run verbatim against a store that holds
    # one thread paused on the class and one that already answered it.

    def it_retires_the_class_end_to_end():
        saver = MemorySaver()

        class EventClass(Interrupted):
            pass

        @on(Started)
        def wait(event: Started) -> EventClass:
            return EventClass()

        saver.serde = NamespaceAwareSerde(events=(Started, EventClass))
        graph = EventGraph([wait, _go_ends], checkpointer=saver)
        graph.invoke(Started(data="x"), config=_cfg("paused"))
        graph.invoke(Started(data="x"), config=_cfg("answered"))
        graph.resume(_Go(), config=_cfg("answered"))

        # 1. and 2.
        for config in graph.threads_paused_on(EventClass):
            graph.abandon(config, reason="retiring EventClass")
        assert graph.threads_paused_on(EventClass) == []
        # 3.
        report = graph.plan_rewrite(drop=(EventClass,))
        assert not report.refused
        assert {t.thread_id for t in report.threads} == {"paused", "answered"}
        # 4.
        report = graph.rewrite_store(drop=(EventClass,))
        assert not report.refused
        # 5. The class is deleted: a serde that no longer reaches it.
        saver.serde = NamespaceAwareSerde(events=(Started,))
        graph = EventGraph([_completes], checkpointer=saver)
        # 6.
        assert graph.unrevivable_threads() == {}
