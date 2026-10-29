"""EventGraph.plan_rewrite() / rewrite_store(): the apply-side migration (#179).

A retired identity stays in the ``events`` channel of a thread's latest
checkpoint until something rewrites the stored bytes. These suites pin
that rewrite at the API boundary: what the plan reports, what the store
holds after the write, and when a thread is refused.

Idiom, shared with ``tests/test_event_graph.py``: a class defined inside
a helper never imports, so a serde swap on the saver simulates "the
class was renamed" or "the class was deleted" against real bytes.
"""

import contextlib
import copy
import typing

import pytest
from _dict_saver import DictSaver
from conftest import Ended, Started
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.base.id import uuid6
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationInfo,
    computed_field,
    field_serializer,
    field_validator,
)
from test_event_graph import _AsyncOnlySaver

from langgraph_events import (
    Cause,
    EventGraph,
    IntegrationEvent,
    Interrupted,
    Reducer,
    Resumed,
    RewriteReport,
    SourceDropped,
    ThreadRewrite,
    on,
)
from langgraph_events.serde import NamespaceAwareSerde, backfill, migrate_from
from langgraph_events.serde._jsonplus import _scan_identities
from langgraph_events.serde.migrations import Migration, TransformFields


class _Go(IntegrationEvent):
    pass


class _Spec(BaseModel):
    """A field model whose ``kind`` has no default."""

    kind: typing.Literal["a", "b"]


def _unvalidated_spec(kw: dict[str, typing.Any]) -> dict[str, typing.Any]:
    """Builds the field model with no validation, so ``kind`` stays ``None``."""
    kw.setdefault("spec", _Spec.model_construct(kind=None))
    return kw


class _Aliased(BaseModel):
    """Valid, but its dump uses the field name and its constructor the alias."""

    x: int = Field(alias="X")


class _ForbidComputed(BaseModel):
    """Valid, but its dump holds a computed field that ``extra`` forbids."""

    model_config = ConfigDict(extra="forbid")
    a: int

    @computed_field  # type: ignore[prop-decorator]
    @property
    def b(self) -> int:
        return self.a


class _Serialized(BaseModel):
    """Valid, but its serializer dumps a value its validator refuses."""

    when: int

    @field_serializer("when")
    def _text(self, value: int) -> str:
        return f"{value}s"


class _NeedsContext(BaseModel):
    """Valid, but its validator reads a context the constructor cannot give."""

    a: int

    @field_validator("a")
    @classmethod
    def _context(cls, value: int, info: ValidationInfo) -> int:
        if info.context is None:
            raise ValueError("needs context")
        return value


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


class _Hold(Interrupted):
    pass


class _CopyingSaver(MemorySaver):
    """Returns a copy of each pending write value, not the value that the
    serde decoded."""

    def get_tuple(self, config):  # type: ignore[override]
        tup = super().get_tuple(config)
        if tup is None:
            return None
        writes = [(t, c, copy.deepcopy(v)) for t, c, v in tup.pending_writes or ()]
        return tup._replace(pending_writes=writes)


@on(_Go)
def _go_ends(event: _Go) -> Ended:
    return Ended(result="went")


@on(Started)
def _completes(event: Started) -> None:
    return None


class _StrayEntrySaver(MemorySaver):
    """Once ``armed``, stores each put with one more ``events`` entry that
    is not an event. The checkpoint id and the channel versions are the
    ones put received."""

    armed = False

    def put(self, config, checkpoint, metadata, new_versions):  # type: ignore[override]
        if not self.armed:
            return super().put(config, checkpoint, metadata, new_versions)
        values = dict(checkpoint["channel_values"])
        values["events"] = [*values["events"], {"stray": 1}]
        stray = {**checkpoint, "channel_values": values}
        return super().put(config, stray, metadata, new_versions)


class _StaleBlobSaver(MemorySaver):
    """Once ``armed``, put() stores the new channel versions, but keeps the
    bytes of the stored checkpoint under each new version."""

    armed = False

    def put(self, config, checkpoint, metadata, new_versions):  # type: ignore[override]
        if not self.armed:
            return super().put(config, checkpoint, metadata, new_versions)
        thread = config["configurable"]["thread_id"]
        ns = config["configurable"].get("checkpoint_ns", "")
        stored, _metadata, _parent = self.storage[thread][ns][checkpoint["id"]]
        old = self.serde.loads_typed(stored)["channel_versions"]
        result = super().put(config, checkpoint, metadata, new_versions)
        for channel, version in new_versions.items():
            self.blobs[(thread, ns, channel, version)] = self.blobs[
                (thread, ns, channel, old[channel])
            ]
        return result


def _class_stage_transform(saver: MemorySaver, transform, value):
    """A settled thread whose ``_Held`` event holds *value*, read back by a
    release whose class-stage transform of ``_Held`` is *transform*.

    Returns that release's graph and the ``_Held`` class.
    """

    class _Held(IntegrationEvent):
        value: typing.Any = None

    @on(Started)
    def make(event: Started) -> _Held:
        return _Held(value=value)

    saver.serde = NamespaceAwareSerde(events=(Started, _Held))
    EventGraph([make], checkpointer=saver).invoke(Started(data="x"), config=_cfg("t1"))
    operation = TransformFields(_Held.__module__, _Held.__qualname__, transform)
    saver.serde = NamespaceAwareSerde(
        events=(Started, _Held), migrations=[Migration(operations=(operation,))]
    )
    return EventGraph([_completes], checkpointer=saver), _Held


def _bump(kw: dict[str, typing.Any]) -> dict[str, typing.Any]:
    """Not idempotent: each read adds one again."""
    kw["value"] += 1
    return kw


def _opaque(kw: dict[str, typing.Any]) -> dict[str, typing.Any]:
    """Builds a value that msgpack cannot encode."""
    kw["value"] = object()
    return kw


def _as_tuple(kw: dict[str, typing.Any]) -> dict[str, typing.Any]:
    """Idempotent: a stored list and a tuple encode as the same bytes."""
    kw["value"] = tuple(kw["value"])
    return kw


def _cfg(tid: str, ns: str = "") -> dict[str, typing.Any]:
    return {"configurable": {"thread_id": tid, "checkpoint_ns": ns}}


def _identity(cls: type) -> str:
    return f"{cls.__module__}:{cls.__qualname__}"


def _stored_identities(
    saver: MemorySaver, tid: str, ns: str = ""
) -> set[tuple[str, str]]:
    """Every event identity in the latest checkpoint blobs of the thread
    under *ns*, read from the bytes, not through the serde."""
    tup = saver.get_tuple(_cfg(tid, ns))
    assert tup is not None
    found: set[tuple[str, str]] = set()
    for channel, version in tup.checkpoint["channel_versions"].items():
        blob = saver.blobs.get((tid, ns, channel, version))
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


def _subgraph_pair(saver: MemorySaver, tid: str):
    """A thread whose handler ran an inner graph on the same saver.

    The inner graph left its checkpoints under a subgraph namespace. They
    hold an event whose class release N+1 renamed with ``@migrate_from``.
    Returns the release N+1 graph, the namespace, the old class and the
    new class.
    """

    class _OldStep(IntegrationEvent):
        pass

    @on(_OldStep)
    def step(event: _OldStep) -> None:
        return None

    saver.serde = NamespaceAwareSerde(events=(Started, _OldStep))
    inner = EventGraph([step], checkpointer=saver)

    @on(Started)
    def run_inner(event: Started) -> None:
        inner.invoke(_OldStep())

    EventGraph([run_inner], checkpointer=saver).invoke(Started(), config=_cfg(tid))
    [ns] = [ns for ns in saver.storage[tid] if ns]

    @migrate_from(_OldStep.__qualname__, in_module=_OldStep.__module__)
    class _NewStep(IntegrationEvent):
        pass

    saver.serde = NamespaceAwareSerde(events=(Started, _NewStep))
    return EventGraph([_completes], checkpointer=saver), ns, _OldStep, _NewStep


def _failed_subgraph(saver: MemorySaver, tid: str):
    """A thread whose handler caught the failure of an inner graph.

    The root run completed. The latest checkpoint of the inner graph keeps
    the ``__error__`` write, and its history holds an event that release
    N+1 renamed. Returns the release N+1 graph and the namespace.
    """

    class _OldStep(IntegrationEvent):
        pass

    @on(Started)
    def emit(event: Started) -> _OldStep:
        return _OldStep()

    @on(_OldStep)
    def fail(event: _OldStep) -> None:
        raise RuntimeError("inner failed")

    saver.serde = NamespaceAwareSerde(events=(Started, _OldStep))
    inner = EventGraph([emit, fail], checkpointer=saver)

    @on(Started)
    def run_inner(event: Started) -> None:
        with contextlib.suppress(RuntimeError):
            inner.invoke(Started())

    EventGraph([run_inner], checkpointer=saver).invoke(Started(), config=_cfg(tid))
    [ns] = [ns for ns in saver.storage[tid] if ns]

    @migrate_from(_OldStep.__qualname__, in_module=_OldStep.__module__)
    class _NewStep(IntegrationEvent):
        pass

    saver.serde = NamespaceAwareSerde(events=(Started, _NewStep))
    return EventGraph([_completes], checkpointer=saver), ns


def _paused_subgraph(saver: MemorySaver, tid: str):
    """A root thread paused because its inner graph paused on a class about
    to be retired. Returns the graph, the namespace and the class."""

    class _Retiring(Interrupted):
        pass

    @on(Started)
    def wait(event: Started) -> _Retiring:
        return _Retiring()

    saver.serde = NamespaceAwareSerde(events=(Started, Ended, _Go, _Retiring))
    inner = EventGraph([wait, _go_ends], checkpointer=saver)

    @on(Started)
    def run_inner(event: Started) -> None:
        inner.invoke(Started())

    graph = EventGraph([run_inner], checkpointer=saver)
    graph.invoke(Started(), config=_cfg(tid))
    [ns] = [ns for ns in saver.storage[tid] if ns]
    return graph, ns, _Retiring


def _paused_subgraph_beside_a_sibling(saver: MemorySaver, tid: str):
    """A root thread paused because its inner graph paused. A sibling task
    of the inner graph completed in the superstep that paused. Its write
    holds an event whose class release N+1 renamed with ``@migrate_from``.
    Returns the release N+1 graph, the namespace and the old class."""

    class _Gate(Interrupted):
        pass

    class _OldSide(IntegrationEvent):
        pass

    @on(Started)
    def wait(event: Started) -> _Gate:
        return _Gate()

    @on(Started)
    def side(event: Started) -> _OldSide:
        return _OldSide()

    def root(inner: EventGraph) -> EventGraph:
        @on(Started)
        def run_inner(event: Started) -> None:
            inner.invoke(Started())

        return EventGraph([run_inner], checkpointer=saver)

    saver.serde = NamespaceAwareSerde(events=(Started, Ended, _Go, _Gate, _OldSide))
    root(EventGraph([wait, side, _go_ends], checkpointer=saver)).invoke(
        Started(), config=_cfg(tid)
    )
    [ns] = [ns for ns in saver.storage[tid] if ns]

    @migrate_from(_OldSide.__qualname__, in_module=_OldSide.__module__)
    class _NewSide(IntegrationEvent):
        pass

    saver.serde = NamespaceAwareSerde(events=(Started, Ended, _Go, _Gate, _NewSide))
    return root(EventGraph([wait, _go_ends], checkpointer=saver)), ns, _OldSide


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


def _paused_beside_a_sibling(saver: BaseCheckpointSaver, tid: str, *, emits: bool):
    """A thread paused on a live interrupt, whose settled history holds an
    interrupt written under an identity that release N+1 renamed. A sibling
    task completed in the superstep that paused. It returns ``_SideDone``
    when *emits* is true. Otherwise its write holds only empty values.

    Returns the release N+1 graph, the config, and a dict of handler fire
    counts that the graph's handlers update.
    """

    class _OldGate(Interrupted):
        pass

    class _LivePause(Interrupted):
        pass

    fired: dict[str, int] = {"sibling": 0, "finish": 0}

    @on(Started)
    def wait_old(event: Started) -> _OldGate:
        return _OldGate()

    @on(_Go)
    def wait_live(event: _Go) -> _LivePause:
        return _LivePause()

    @on(_Go)
    def sibling(event: _Go) -> _SideDone | None:
        fired["sibling"] += 1
        return _SideDone() if emits else None

    @on(_Again)
    def finish(event: _Again) -> Ended:
        fired["finish"] += 1
        return Ended(result="finished")

    handlers = [wait_old, wait_live, sibling, finish]
    cfg = _cfg(tid)
    saver.serde = NamespaceAwareSerde(events=(Started, _OldGate, _LivePause))
    graph = EventGraph(handlers, checkpointer=saver)
    graph.invoke(Started(data="x"), config=cfg)
    graph.resume(_Go(), config=cfg)
    tup = saver.get_tuple(cfg)
    assert tup is not None
    assert {c for _t, c, _v in tup.pending_writes} > {"__interrupt__", "events"}
    fired.update(sibling=0, finish=0)

    @migrate_from(_OldGate.__qualname__, in_module=_OldGate.__module__)
    class _NewGate(Interrupted):
        pass

    saver.serde = NamespaceAwareSerde(events=(Started, _NewGate, _LivePause))
    return EventGraph(handlers, checkpointer=saver), cfg, fired


@pytest.fixture(params=["memory", "postgres"])
def any_saver(request: pytest.FixtureRequest):
    """A ``MemorySaver``, then a ``PostgresSaver`` on a new database."""
    if request.param == "memory":
        yield MemorySaver()
        return
    from langgraph.checkpoint.postgres import PostgresSaver

    with PostgresSaver.from_conn_string(
        request.getfixturevalue("postgres_url")
    ) as saver:
        yield saver


def _paused_on_renamed_interrupt(saver: BaseCheckpointSaver, tid: str):
    """A thread paused on an interrupt whose class release N+1 renamed.

    Returns the release N+1 graph and the config.
    """

    class _OldGate(Interrupted):
        pass

    @on(Started)
    def wait(event: Started) -> _OldGate:
        return _OldGate()

    cfg = _cfg(tid)
    saver.serde = NamespaceAwareSerde(events=(Started, _OldGate))
    EventGraph([wait], checkpointer=saver).invoke(Started(data="x"), config=cfg)

    @migrate_from(_OldGate.__qualname__, in_module=_OldGate.__module__)
    class _NewGate(Interrupted):
        pass

    @on(Started)
    def wait_new(event: Started) -> _NewGate:
        return _NewGate()

    saver.serde = NamespaceAwareSerde(events=(Started, _NewGate))
    return EventGraph([wait_new], checkpointer=saver), cfg


def _task_writing(
    saver: BaseCheckpointSaver, cfg: dict[str, typing.Any], channel: str
) -> str:
    """The id of the one task whose pending write is on *channel*."""
    tup = saver.get_tuple(cfg)
    assert tup is not None
    [task] = {t for t, c, _v in tup.pending_writes if c == channel}
    return task


def _checkpoint_id(saver: BaseCheckpointSaver, cfg: dict[str, typing.Any]) -> str:
    tup = saver.get_tuple(cfg)
    assert tup is not None
    return tup.config["configurable"]["checkpoint_id"]


def _dict_identities(saver: DictSaver, cfg: dict[str, typing.Any]) -> set:
    """Every event identity in the stored bytes of the latest checkpoint."""
    thread = cfg["configurable"]["thread_id"]
    rows = saver.rows[(thread, "")]
    checkpoint, _metadata, _parent = rows[max(rows)]
    return set(_scan_identities(checkpoint[1]))


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
                EventGraph([_completes]).plan_rewrite(thread_ids=[])

        def it_requires_a_namespace_aware_serde():
            graph = EventGraph([_completes], checkpointer=MemorySaver())

            with pytest.raises(ValueError, match=r"plan_rewrite\(\) needs a Namespace"):
                graph.plan_rewrite(thread_ids=[])

        def it_refuses_legacy_write():
            saver = MemorySaver()
            saver.serde = NamespaceAwareSerde(events=(Started,), legacy_write=True)
            graph = EventGraph([_completes], checkpointer=saver)

            with pytest.raises(ValueError, match=r"legacy_write"):
                graph.plan_rewrite(thread_ids=[])

        def it_refuses_a_drop_class_the_serde_cannot_revive():
            class _Unknown(IntegrationEvent):
                pass

            saver = MemorySaver()
            saver.serde = NamespaceAwareSerde(events=(Started,))
            graph = EventGraph([_completes], checkpointer=saver)

            with pytest.raises(ValueError, match=r"_Unknown"):
                graph.plan_rewrite(drop=(_Unknown,), thread_ids=[])

    def when_a_thread_holds_a_renamed_identity():
        def it_reports_the_migrated_pair():
            saver = MemorySaver()
            graph, _config, old, new = _renamed_history_pair(saver, "t1")

            report = graph.plan_rewrite(thread_ids=["t1"])

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

            graph.plan_rewrite(thread_ids=["t1"])

            assert set(saver.blobs) == blobs_before
            after = saver.get_tuple(cfg).checkpoint["channel_versions"]
            assert after == versions_before

    def when_a_subgraph_checkpoint_holds_a_renamed_identity():
        def it_reads_the_checkpoint_under_that_namespace():
            saver = MemorySaver()
            graph, ns, old, new = _subgraph_pair(saver, "t1")

            report = graph.plan_rewrite(thread_ids=["t1"], checkpoint_ns=ns)

            [thread] = report.threads
            assert thread.checkpoint_ns == ns
            assert thread.status == "rewrite"
            assert thread.migrated == ((_identity(old), _identity(new)),)

    def when_a_thread_holds_no_migrated_identity():
        def it_reports_unchanged():
            saver = MemorySaver()
            saver.serde = NamespaceAwareSerde(events=(Started,))
            graph = EventGraph([_completes], checkpointer=saver)
            graph.invoke(Started(data="x"), config=_cfg("plain"))

            [thread] = graph.plan_rewrite(thread_ids=["plain"]).threads

            assert thread.status == "unchanged"
            assert thread.migrated == ()
            assert thread.dropped == {}


def describe_rewrite_store():
    def when_a_thread_holds_a_renamed_identity():
        def it_stores_the_live_identity_and_reports_the_rewrite():
            saver = MemorySaver()
            graph, _config, old, new = _renamed_history_pair(saver, "t1")
            assert (old.__module__, old.__qualname__) in _stored_identities(saver, "t1")

            report = graph.rewrite_store(thread_ids=["t1"])

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

            report = graph.rewrite_store(drop=(retiring,), thread_ids=["t1"])

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
            graph.rewrite_store(drop=(retiring,), thread_ids=["t1"])

            saver.serde = NamespaceAwareSerde(events=(Started,))
            after = EventGraph([_completes], checkpointer=saver)

            assert after.unrevivable_threads(thread_ids=["t1"]) == {}

        def it_dispatches_the_next_input_on_the_thread():
            saver = MemorySaver()
            graph, cfg, retiring = _settled_drop_pair(saver, "t1")
            graph.rewrite_store(drop=(retiring,), thread_ids=["t1"])
            fired: list[int] = []

            @on(_Again)
            def again(event: _Again) -> Ended:
                fired.append(1)
                return Ended(result="again")

            log = EventGraph([again], checkpointer=saver).invoke(_Again(), config=cfg)

            assert fired == [1]
            assert log.latest(Ended) == Ended(result="again")

        def it_remaps_each_cause_to_the_kept_events():
            saver = MemorySaver()
            graph, cfg, retiring = _settled_drop_pair(saver, "t1")

            graph.rewrite_store(drop=(retiring,), thread_ids=["t1"])
            log = graph.get_state(cfg).events

            assert log.cause(log.latest(Ended)) == Cause(source=_Go(), via="_go_ends")
            assert log.cause(log.latest(Ended)).source is log.first(_Go)
            assert log.cause(log.first(_Go)) == SourceDropped(
                via="wait", source_type=retiring.__qualname__
            )

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

            [thread] = graph.rewrite_store(drop=(_Base,), thread_ids=["t1"]).threads

            assert thread.status == "unchanged"

    def when_a_dropped_event_sits_below_a_paused_handler():
        def _resumed():
            saver = MemorySaver()

            class _Noise(IntegrationEvent):
                pass

            class _Gate(Interrupted):
                pass

            @on(_Noise)
            def promote(event: _Noise) -> Started:
                return Started(data="promoted")

            @on(Started)
            def wait(event: Started) -> _Gate:
                return _Gate()

            cfg = _cfg("t1")
            saver.serde = NamespaceAwareSerde(events=(Started, _Noise, _Gate))
            graph = EventGraph([promote, wait], checkpointer=saver)
            graph.invoke(_Noise(), config=cfg)
            graph.rewrite_store(drop=(_Noise,), thread_ids=["t1"])
            return graph, graph.resume(_Go(), config=cfg), _Gate

        def it_keeps_the_trigger_of_the_resumed_handler():
            _graph, log, gate = _resumed()

            assert log.cause(log.first(gate)) == Cause(
                source=Started(data="promoted"), via="wait"
            )
            assert log.cause(log.first(gate)).source is log.first(Started)

        def it_shows_the_cause_of_a_dropped_source_as_unknown():
            graph, log, _gate = _resumed()
            started = next(i for i, e in enumerate(log) if e is log.first(Started))

            answer = graph.reflect(log).tool().run(op="cause", index=started)

            assert answer.startswith("unknown, source ")
            assert answer.endswith("_Noise dropped by rewrite_store, via promote")

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

            [first] = graph.rewrite_store(thread_ids=["t1"]).threads
            [second] = graph.rewrite_store(thread_ids=["t1"]).threads

            assert first.status == "rewrite"
            assert second.status == "unchanged"
            assert len(saver.blobs) > blobs_before
            assert (
                graph.plan_rewrite(thread_ids=["t1"]).threads[0].status == "unchanged"
            )

    def when_a_thread_is_paused_on_a_live_interrupt():
        def it_rewrites_the_history_and_the_thread_still_resumes():
            saver = MemorySaver()
            graph, cfg, fired = _paused_renamed_pair(saver, "t1")

            [thread] = graph.rewrite_store(thread_ids=["t1"]).threads
            log = graph.resume(_Again(), config=cfg)

            assert thread.status == "rewrite"
            assert log.latest(Ended) == Ended(result="finished")
            # The paused handler re-runs once on resume by LangGraph design.
            assert fired == {"wait_old": 0, "wait_live": 1, "finish": 1}

    def when_a_paused_thread_holds_a_completed_sibling_write():
        @pytest.mark.parametrize("emits", [False, True])
        def it_rewrites_the_thread_and_the_resume_applies_the_write(
            any_saver: BaseCheckpointSaver, emits: bool
        ):
            graph, cfg, fired = _paused_beside_a_sibling(any_saver, "t1", emits=emits)

            [thread] = graph.rewrite_store(thread_ids=["t1"]).threads
            log = graph.resume(_Again(), config=cfg)

            assert thread.status == "rewrite"
            assert log.latest(Ended) == Ended(result="finished")
            assert log.count(_SideDone) == (1 if emits else 0)
            assert fired == {"sibling": 0, "finish": 1}
            assert graph.plan_rewrite(thread_ids=["t1"]).threads[0].status == (
                "unchanged"
            )

        async def it_rewrites_the_thread_through_the_async_path():
            saver = MemorySaver()
            graph, cfg, fired = _paused_beside_a_sibling(saver, "t1", emits=True)

            [thread] = (await graph.arewrite_store(thread_ids=["t1"])).threads
            log = await graph.aresume(_Again(), config=cfg)

            assert thread.status == "rewrite"
            assert log.count(_SideDone) == 1
            assert fired == {"sibling": 0, "finish": 1}

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

            [thread] = graph.rewrite_store(thread_ids=["seeded"]).threads

            assert thread.status == "rewrite"
            assert (_NewGate.__module__, _NewGate.__qualname__) in _stored_identities(
                saver, "seeded"
            )

        def it_keeps_the_checkpoint_id_and_reports_unchanged_on_a_second_run():
            saver = MemorySaver()
            graph, cfg, _old, _new = _renamed_history_pair(saver, "t1")
            checkpoint_id = saver.get_tuple(cfg).checkpoint["id"]

            graph.rewrite_store(thread_ids=["t1"])

            assert saver.get_tuple(cfg).checkpoint["id"] == checkpoint_id
            [thread] = graph.plan_rewrite(thread_ids=["t1"]).threads
            assert thread.status == "unchanged"

    def when_a_paused_subgraph_checkpoint_holds_a_completed_sibling_write():
        def it_rewrites_the_subgraph_checkpoint():
            saver = MemorySaver()

            class _OldGate(Interrupted):
                pass

            @on(Started)
            def wait(event: Started) -> _OldGate:
                return _OldGate()

            saver.serde = NamespaceAwareSerde(events=(Started, _OldGate))
            inner = EventGraph([wait, _side_effect], checkpointer=saver)

            @on(Started)
            def run_inner(event: Started) -> None:
                inner.invoke(Started())

            graph = EventGraph([run_inner], checkpointer=saver)
            graph.invoke(Started(), config=_cfg("t1"))
            [ns] = [ns for ns in saver.storage["t1"] if ns]
            assert _task_writing(saver, _cfg("t1", ns), "events")

            @migrate_from(_OldGate.__qualname__, in_module=_OldGate.__module__)
            class _NewGate(Interrupted):
                pass

            saver.serde = NamespaceAwareSerde(events=(Started, _NewGate))

            report = graph.rewrite_store(thread_ids=["t1"], checkpoint_ns=ns)

            assert [t.status for t in report.threads] == ["rewrite"]

    def when_a_subgraph_checkpoint_holds_a_renamed_identity():
        def it_stores_the_live_identity_and_keeps_the_parent_link():
            saver = MemorySaver()
            graph, ns, old, new = _subgraph_pair(saver, "t1")
            before = saver.get_tuple(_cfg("t1", ns))
            assert before is not None

            report = graph.rewrite_store(thread_ids=["t1"], checkpoint_ns=ns)

            assert [t.status for t in report.threads] == ["rewrite"]
            stored = _stored_identities(saver, "t1", ns)
            assert (old.__module__, old.__qualname__) not in stored
            assert (new.__module__, new.__qualname__) in stored
            after = saver.get_tuple(_cfg("t1", ns))
            assert after is not None
            assert after.checkpoint["id"] == before.checkpoint["id"]
            assert after.parent_config == before.parent_config
            [thread] = graph.plan_rewrite(thread_ids=["t1"], checkpoint_ns=ns).threads
            assert thread.status == "unchanged"

        def when_the_namespace_has_one_checkpoint():
            def it_writes_under_that_namespace_only():
                source = MemorySaver()
                _graph, ns, _old, new = _subgraph_pair(source, "t1")
                tup = source.get_tuple(_cfg("t1", ns))
                assert tup is not None
                latest = tup.config["configurable"]["checkpoint_id"]
                saver = MemorySaver(serde=source.serde)
                checkpoint, metadata, _parent = source.storage["t1"][ns][latest]
                saver.storage["t1"][ns][latest] = (checkpoint, metadata, None)
                saver.blobs.update(
                    (key, blob)
                    for key, blob in source.blobs.items()
                    if key[:2] == ("t1", ns)
                )
                graph = EventGraph([_completes], checkpointer=saver)

                report = graph.rewrite_store(thread_ids=["t1"], checkpoint_ns=ns)

                assert [t.status for t in report.threads] == ["rewrite"]
                assert list(saver.storage["t1"]) == [ns]
                stored = _stored_identities(saver, "t1", ns)
                assert (new.__module__, new.__qualname__) in stored

    def when_a_class_stage_transform_returns_bytes_equal_to_the_stored_ones():
        def it_reads_the_thread_as_unchanged():
            saver = MemorySaver()
            graph, _held = _class_stage_transform(saver, _as_tuple, ("a",))

            [thread] = graph.rewrite_store(thread_ids=["t1"]).threads

            assert thread.status == "unchanged"

        def it_rewrites_a_renamed_thread_once():
            saver = MemorySaver()

            class _OldHeld(IntegrationEvent):
                value: typing.Any = None

            @on(Started)
            def make(event: Started) -> _OldHeld:
                return _OldHeld(value=("a",))

            saver.serde = NamespaceAwareSerde(events=(Started, _OldHeld))
            EventGraph([make], checkpointer=saver).invoke(
                Started(data="x"), config=_cfg("t1")
            )

            @migrate_from(_OldHeld.__qualname__, in_module=_OldHeld.__module__)
            class _Held(IntegrationEvent):
                value: typing.Any = None

            operation = TransformFields(_Held.__module__, _Held.__qualname__, _as_tuple)
            saver.serde = NamespaceAwareSerde(
                events=(Started, _Held), migrations=[Migration(operations=(operation,))]
            )
            graph = EventGraph([_completes], checkpointer=saver)

            [first] = graph.rewrite_store(thread_ids=["t1"]).threads
            [second] = graph.plan_rewrite(thread_ids=["t1"]).threads

            assert (first.status, second.status) == ("rewrite", "unchanged")

    def when_a_custom_saver_overwrites_a_checkpoint():
        def it_stores_the_live_identity():
            saver = DictSaver()
            graph, cfg, old, new = _renamed_history_pair(saver, "t1")

            report = graph.rewrite_store(thread_ids=["t1"])

            assert [t.status for t in report.threads] == ["rewrite"]
            stored = _dict_identities(saver, cfg)
            assert (old.__module__, old.__qualname__) not in stored
            assert (new.__module__, new.__qualname__) in stored


def describe_rewrite_store_refusals():
    # Each refusal names what to do. None writes anything.

    def _only(report):
        [thread] = report.threads
        assert thread.status == "refused", thread
        return thread

    def when_the_log_holds_an_entry_that_is_not_an_event():
        def _langgraph_written(saver: MemorySaver) -> EventGraph:
            """A thread that LangGraph's own serializer wrote. Its class is
            defined inside this helper, so it never imports again."""

            class _Gone(IntegrationEvent):
                note: str = ""

            @on(Started)
            def make(event: Started) -> _Gone:
                return _Gone(note="x")

            saver.serde = JsonPlusSerializer()
            EventGraph([make], checkpointer=saver).invoke(
                Started(data="x"), config=_cfg("t1")
            )
            saver.serde = NamespaceAwareSerde(events=(Started,))
            return EventGraph([_completes], checkpointer=saver)

        def it_refuses_the_thread_and_names_the_channel():
            graph = _langgraph_written(MemorySaver())

            thread = _only(graph.plan_rewrite(thread_ids=["t1"]))

            assert thread.reason.startswith(
                "channel 'events' holds 1 entry that revives as NoneType, not as "
                "an event."
            )
            assert "delete_thread()" in thread.reason

        def when_a_completed_sibling_write_holds_the_entry():
            # The thread paused in its first superstep, so its history
            # holds no record that a migration rewrites.
            def _paused_beside_the_entry(saver: MemorySaver) -> EventGraph:
                class _Gone(IntegrationEvent):
                    pass

                @on(Started)
                def wait(event: Started) -> _Hold:
                    return _Hold()

                @on(Started)
                def make(event: Started) -> _Gone:
                    return _Gone()

                saver.serde = JsonPlusSerializer()
                graph = EventGraph([wait, make], checkpointer=saver)
                graph.invoke(Started(data="x"), config=_cfg("t1"))
                saver.serde = NamespaceAwareSerde(events=(Started, _Hold))
                return graph

            def it_names_the_task_and_the_channel():
                saver = MemorySaver()
                graph = _paused_beside_the_entry(saver)

                thread = _only(graph.rewrite_store(thread_ids=["t1"]))

                assert thread.reason.startswith(
                    f"thread has a completed task write on task "
                    f"{_task_writing(saver, _cfg('t1'), 'events')!r} in channel "
                    f"'events' that holds 1 entry that revives as NoneType, not "
                    f"as an event. "
                )
                assert thread.reason.endswith(
                    "delete_thread() if nobody needs it, then rerun"
                )

            async def it_refuses_the_thread_through_the_async_saver():
                graph = _paused_beside_the_entry(MemorySaver())

                thread = _only(await graph.arewrite_store(thread_ids=["t1"]))

                assert thread.reason.startswith(
                    "thread has a completed task write on task "
                )

        def when_the_allow_list_of_the_serde_blocks_its_class():
            def it_names_the_allow_list():
                saver = MemorySaver(serde=JsonPlusSerializer())

                @on(Started)
                def make(event: Started) -> _Go:
                    return _Go()

                EventGraph([make], checkpointer=saver).invoke(
                    Started(data="x"), config=_cfg("t1")
                )
                saver.serde = NamespaceAwareSerde(
                    events=(Started, _Go), allowed_msgpack_modules=None
                )
                graph = EventGraph([_completes], checkpointer=saver)

                thread = _only(graph.plan_rewrite(thread_ids=["t1"]))

                assert thread.reason.startswith(
                    "channel 'events' holds 2 entries that revive as dict, not as "
                    "events."
                )
                assert thread.reason.endswith(
                    "Then add each class to allowed_msgpack_modules, then rerun"
                )

        def it_writes_nothing():
            saver = MemorySaver()
            graph = _langgraph_written(saver)
            blobs = dict(saver.blobs)

            graph.rewrite_store(thread_ids=["t1"])

            assert saver.blobs == blobs

    def when_a_migration_builds_a_value_that_fails_validation():
        def it_refuses_the_thread_and_names_the_class_and_field():
            saver = MemorySaver()

            class _Old(IntegrationEvent):
                note: str = ""

            @on(Started)
            def make(event: Started) -> _Old:
                return _Old(note="x")

            saver.serde = NamespaceAwareSerde(events=(Started, _Old))
            EventGraph([make], checkpointer=saver).invoke(
                Started(data="x"), config=_cfg("t1")
            )

            @migrate_from(
                _Old.__qualname__,
                in_module=_Old.__module__,
                transform=_unvalidated_spec,
            )
            class _New(IntegrationEvent):
                note: str = ""
                spec: _Spec | None = None

            saver.serde = NamespaceAwareSerde(events=(Started, _New))
            graph = EventGraph([_completes], checkpointer=saver)
            blobs = dict(saver.blobs)

            thread = _only(graph.rewrite_store(thread_ids=["t1"]))

            assert thread.reason.startswith(
                "the rewritten value in channel 'events' holds an invalid _Spec in "
                f"{_New.__qualname__}.spec: 1 validation error for _Spec kind"
            )
            assert saver.blobs == blobs

    def when_a_migration_replaces_a_stored_model():
        def with_a_model_that_fails_validation():
            def it_refuses_the_thread_and_names_the_class_and_field():
                saver = MemorySaver()

                class _Old(IntegrationEvent):
                    spec: object = None

                @on(Started)
                def make(event: Started) -> _Old:
                    return _Old(spec=_Spec(kind="a"))

                saver.serde = NamespaceAwareSerde(events=(Started, _Old))
                EventGraph([make], checkpointer=saver).invoke(
                    Started(data="x"), config=_cfg("t1")
                )

                @migrate_from(
                    _Old.__qualname__,
                    in_module=_Old.__module__,
                    transform=lambda kw: {
                        **kw,
                        "spec": _Spec.model_construct(kind=None),
                    },
                )
                class _New(IntegrationEvent):
                    spec: object = None

                saver.serde = NamespaceAwareSerde(events=(Started, _New))
                graph = EventGraph([_completes], checkpointer=saver)

                thread = _only(graph.plan_rewrite(thread_ids=["t1"]))

                assert thread.reason.startswith(
                    "the rewritten value in channel 'events' holds an invalid _Spec in "
                    f"{_New.__qualname__}.spec: 1 validation error for _Spec kind"
                )

    def when_a_stored_model_does_not_rebuild_from_its_dump():
        # LangGraph revives such a model with ``model_construct``. No
        # migration built it, so the rewrite stores what the store holds.
        @pytest.mark.parametrize(
            "model",
            [
                _Aliased(X=1),
                _ForbidComputed(a=1),
                _Serialized(when=1),
                _NeedsContext.model_validate({"a": 1}, context={"k": 1}),
            ],
            ids=type,
        )
        def it_rewrites_the_thread(model: BaseModel):
            saver = MemorySaver()

            class _Old(IntegrationEvent):
                held: object = None

            @on(Started)
            def make(event: Started) -> _Old:
                return _Old(held=model)

            saver.serde = NamespaceAwareSerde(events=(Started, _Old))
            EventGraph([make], checkpointer=saver).invoke(
                Started(data="x"), config=_cfg("t1")
            )

            @migrate_from(_Old.__qualname__, in_module=_Old.__module__)
            class _New(IntegrationEvent):
                held: object = None

            saver.serde = NamespaceAwareSerde(events=(Started, _New))
            graph = EventGraph([_completes], checkpointer=saver)

            report = graph.rewrite_store(thread_ids=["t1"])

            assert [(t.status, t.reason) for t in report.threads] == [("rewrite", "")]

    def when_a_class_stage_transform_does_not_settle():
        def it_refuses_the_thread_before_the_write():
            saver = MemorySaver()
            graph, held = _class_stage_transform(saver, _bump, 1)
            blobs = dict(saver.blobs)

            thread = _only(graph.rewrite_store(thread_ids=["t1"]))

            assert thread.reason == (
                f"a migration of {_identity(held)} does not settle: a read of the "
                f"rewritten value in channel 'events' changes it again. Make each "
                f"class-stage transform return its own output unchanged, then rerun"
            )
            assert saver.blobs == blobs

        async def it_refuses_the_thread_through_the_async_saver():
            saver = MemorySaver()
            graph, held = _class_stage_transform(saver, _bump, 1)

            thread = _only(await graph.arewrite_store(thread_ids=["t1"]))

            assert thread.reason.startswith(
                f"a migration of {_identity(held)} does not settle: "
            )

    def when_a_class_stage_transform_builds_a_value_that_cannot_be_encoded():
        def it_refuses_the_thread_and_names_the_encode_error():
            saver = MemorySaver()
            graph, _held = _class_stage_transform(saver, _opaque, 1)

            thread = _only(graph.plan_rewrite(thread_ids=["t1"]))

            assert thread.reason.startswith(
                "cannot encode the value in channel 'events' ("
            )

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

            thread = _only(
                EventGraph([_completes], checkpointer=saver).rewrite_store(
                    thread_ids=["t1"]
                )
            )

            assert "_Deleted" in thread.reason
            assert "tombstone" in thread.reason
            with saver.serde.tolerate_unresolved():
                assert saver.get_tuple(cfg).checkpoint["channel_versions"] == before

    def when_a_completed_sibling_write_holds_a_migrated_record():
        def it_names_the_task_the_channel_and_the_record():
            saver = MemorySaver()

            class _Pause(Interrupted):
                pass

            class _OldSide(IntegrationEvent):
                pass

            @on(Started)
            def wait(event: Started) -> _Pause:
                return _Pause()

            @on(Started)
            def side(event: Started) -> _OldSide:
                return _OldSide()

            cfg = _cfg("t1")
            saver.serde = NamespaceAwareSerde(events=(Started, _Pause, _OldSide))
            EventGraph([wait, side], checkpointer=saver).invoke(
                Started(data="x"), config=cfg
            )

            @migrate_from(_OldSide.__qualname__, in_module=_OldSide.__module__)
            class _NewSide(IntegrationEvent):
                pass

            saver.serde = NamespaceAwareSerde(events=(Started, _Pause, _NewSide))
            graph = EventGraph([wait], checkpointer=saver)

            thread = _only(graph.rewrite_store(thread_ids=["t1"]))

            assert thread.reason == (
                f"thread has a completed task write on task "
                f"{_task_writing(saver, cfg, 'events')!r} in channel 'events' "
                f"that holds {_identity(_OldSide)}, which a migration rewrites; "
                f"resume or abandon it, then rerun"
            )

    def when_the_saver_returns_a_write_that_the_serde_did_not_decode():
        def it_refuses_the_write_it_cannot_read():
            saver = _CopyingSaver()
            graph, cfg, _fired = _paused_beside_a_sibling(saver, "t1", emits=False)

            thread = _only(graph.rewrite_store(thread_ids=["t1"]))

            assert thread.reason.startswith(
                f"thread has a completed task write on task "
                f"{_task_writing(saver, cfg, 'events')!r} in channel "
            )
            assert thread.reason.endswith(
                " that get_tuple() of the checkpointer returns as another object "
                "than the one that its serde decoded, for example as a copy, so "
                "the rewrite cannot read it; resume or abandon it, then rerun. "
                "To let the rewrite read such a write, make get_tuple() return "
                "the write values that the serde decoded"
            )

    def when_a_completed_sibling_write_holds_a_dropped_class():
        def it_names_the_task_the_channel_and_the_class():
            saver = MemorySaver()
            graph, cfg, _fired = _paused_beside_a_sibling(saver, "t1", emits=True)

            thread = _only(graph.rewrite_store(drop=(_SideDone,), thread_ids=["t1"]))

            assert thread.reason == (
                f"thread has a completed task write on task "
                f"{_task_writing(saver, cfg, 'events')!r} in channel 'events' "
                f"that holds {_identity(_SideDone)}, a class in drop; resume or "
                f"abandon it, then rerun"
            )

    def when_a_completed_sibling_write_holds_a_position_that_the_drop_moves():
        def it_names_the_task_and_the_channel():
            saver = MemorySaver()
            graph, cfg, _fired = _paused_beside_a_sibling(saver, "t1", emits=True)

            thread = _only(graph.rewrite_store(drop=(Started,), thread_ids=["t1"]))

            assert thread.reason == (
                f"thread has a completed task write on task "
                f"{_task_writing(saver, cfg, 'causes')!r} in channel 'causes' "
                f"that holds an index into channel 'events', and drop removes an "
                f"event from that channel, so the index would point to another "
                f"event; resume or abandon it, then rerun"
            )

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

            thread = _only(graph.rewrite_store(drop=(_Retiring,), thread_ids=["t1"]))

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

            thread = _only(graph.rewrite_store(drop=(_Retiring,), thread_ids=["t1"]))

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

            thread = _only(graph.rewrite_store(drop=(_Retiring,), thread_ids=["t1"]))

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

            thread = _only(graph.rewrite_store(drop=(_Inner,), thread_ids=["t1"]))

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

        async def it_reports_the_id_through_the_async_saver():
            saver = _AsyncOnlySaver()
            graph, _config, _old, _new = await _arenamed_history_pair(saver, "t1")

            report = await graph.aplan_rewrite(thread_ids=["t1", "t-typo"])

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

            thread = _only(graph.rewrite_store(thread_ids=["t1"]))

            assert thread.reason == (
                "thread has a pending __error__ write from a failed task; run "
                "it again or abandon it, then rerun"
            )

    def when_a_failed_task_left_an_error_write_in_a_subgraph():
        def it_names_the_one_call_that_clears_it():
            saver = MemorySaver()
            graph, ns = _failed_subgraph(saver, "t1")

            thread = _only(graph.rewrite_store(thread_ids=["t1"], checkpoint_ns=ns))

            assert thread.reason == (
                "thread has a pending __error__ write from a failed task; "
                "abandon() and resume() reach only the root namespace, so no "
                "call clears this write. If nobody needs the thread, delete "
                "every checkpoint of it with the checkpointer's "
                "delete_thread(), then rerun. Otherwise keep the migrations "
                "and the classes that this checkpoint needs"
            )

    def when_a_paused_subgraph_checkpoint_holds_a_sibling_write_to_migrate():
        def it_asks_to_resume_the_root_thread():
            saver = MemorySaver()
            graph, ns, old = _paused_subgraph_beside_a_sibling(saver, "t1")

            thread = _only(graph.rewrite_store(thread_ids=["t1"], checkpoint_ns=ns))

            task = _task_writing(saver, _cfg("t1", ns), "events")
            assert thread.reason == (
                f"thread has a completed task write on task {task!r} in channel "
                f"'events' that holds {_identity(old)}, which a migration "
                f"rewrites; resume the root thread, then rerun. abandon() does "
                f"not clear a subgraph checkpoint"
            )

        def when_the_root_thread_resumes():
            def it_leaves_nothing_to_refuse():
                saver = MemorySaver()
                graph, ns, _old = _paused_subgraph_beside_a_sibling(saver, "t1")
                graph.resume(_Go(), config=_cfg("t1"))

                [thread] = graph.rewrite_store(
                    thread_ids=["t1"], checkpoint_ns=ns
                ).threads

                assert thread.status == "unchanged"

    def when_a_failed_sibling_left_a_completed_write_in_a_subgraph():
        def it_names_no_call_that_clears_it():
            saver = MemorySaver()

            class _OldSide(IntegrationEvent):
                pass

            @on(Started)
            def side(event: Started) -> _OldSide:
                return _OldSide()

            @on(Started)
            def fail(event: Started) -> None:
                raise RuntimeError("inner failed")

            saver.serde = NamespaceAwareSerde(events=(Started, _OldSide))
            inner = EventGraph([side, fail], checkpointer=saver)

            @on(Started)
            def run_inner(event: Started) -> None:
                with contextlib.suppress(RuntimeError):
                    inner.invoke(Started())

            graph = EventGraph([run_inner], checkpointer=saver)
            graph.invoke(Started(), config=_cfg("t1"))
            [ns] = [ns for ns in saver.storage["t1"] if ns]

            @migrate_from(_OldSide.__qualname__, in_module=_OldSide.__module__)
            class _NewSide(IntegrationEvent):
                pass

            saver.serde = NamespaceAwareSerde(events=(Started, _NewSide))

            thread = _only(graph.rewrite_store(thread_ids=["t1"], checkpoint_ns=ns))

            assert thread.reason.endswith(
                "abandon() and resume() reach only the root namespace, so no "
                "call clears this write. If nobody needs the thread, delete "
                "every checkpoint of it with the checkpointer's "
                "delete_thread(), then rerun. Otherwise keep the migrations "
                "and the classes that this checkpoint needs"
            )

    def when_the_pending_interrupt_in_a_subgraph_is_a_dropped_class():
        def it_asks_to_resume_the_root_thread():
            saver = MemorySaver()
            graph, ns, retiring = _paused_subgraph(saver, "t1")

            thread = _only(
                graph.plan_rewrite(
                    drop=(retiring,), thread_ids=["t1"], checkpoint_ns=ns
                )
            )

            assert thread.reason == (
                f"thread is paused on {retiring.__qualname__}, a dropped class; "
                f"resume the root thread, then rerun. abandon() does not clear "
                f"a subgraph checkpoint"
            )

        def when_the_root_thread_resumes():
            def it_rewrites_the_subgraph_checkpoint():
                saver = MemorySaver()
                graph, ns, retiring = _paused_subgraph(saver, "t1")
                graph.resume(_Go(), config=_cfg("t1"))

                [thread] = graph.rewrite_store(
                    drop=(retiring,), thread_ids=["t1"], checkpoint_ns=ns
                ).threads

                assert thread.status == "rewrite"

    def when_a_value_cannot_be_encoded():
        def it_refuses_the_thread_instead_of_aborting_the_walk():
            from langgraph_events._rewrite import _verify_value

            serde = NamespaceAwareSerde(events=(Started,))

            problem, _identities = _verify_value(
                serde, [Started(), object()], "events", set(), set(), frozenset()
            )

            assert problem is not None
            assert problem.startswith("cannot encode the value")

    def when_the_thread_advances_during_the_write():
        def it_reports_the_thread_and_asks_for_a_rerun():
            saver = _AdvancingSaver()
            graph, _config, _old, _new = _renamed_history_pair(saver, "t1")

            thread = _only(graph.rewrite_store(thread_ids=["t1"]))

            assert thread.reason == "thread advanced during the rewrite; rerun"

    def when_the_saver_does_not_overwrite_a_checkpoint():
        def it_refuses_the_thread_and_names_the_saver_behaviour():
            saver = DictSaver(put_overwrites=False)
            graph, cfg, old, _new = _renamed_history_pair(saver, "t1")
            checkpoint_id = _checkpoint_id(saver, cfg)

            thread = _only(graph.rewrite_store(thread_ids=["t1"]))

            assert thread.reason.startswith(
                f"the saver did not overwrite checkpoint {checkpoint_id}; "
            )
            assert (old.__module__, old.__qualname__) in _dict_identities(saver, cfg)

        async def it_refuses_the_thread_through_the_async_saver():
            saver = DictSaver(put_overwrites=False)
            graph, cfg, _old, _new = await _arenamed_history_pair(saver, "t1")
            checkpoint_id = _checkpoint_id(saver, cfg)

            thread = _only(await graph.arewrite_store(thread_ids=["t1"]))

            assert thread.reason.startswith(
                f"the saver did not overwrite checkpoint {checkpoint_id}; "
            )

    def when_the_saver_does_not_overwrite_an_interrupt_write():
        def it_refuses_the_paused_thread():
            saver = DictSaver(writes_overwrite=False)
            graph, cfg = _paused_on_renamed_interrupt(saver, "t1")
            checkpoint_id = _checkpoint_id(saver, cfg)

            thread = _only(graph.rewrite_store(thread_ids=["t1"]))

            assert thread.reason == (
                f"the saver did not store the bytes written to checkpoint "
                f"{checkpoint_id}: they read unchanged before the write, but a "
                f"read of the store still migrates them. put_writes() of the "
                f"saver must replace an __interrupt__ write of the same task. "
                f"Fix the saver, then rerun. See 'Checkpointers' in "
                f"docs/event-migrations.md"
            )

    def when_the_saver_keeps_the_old_bytes_under_the_new_versions():
        def it_names_the_put_rule():
            saver = _StaleBlobSaver()
            graph, cfg, _old, _new = _renamed_history_pair(saver, "t1")
            saver.armed = True
            checkpoint_id = _checkpoint_id(saver, cfg)

            thread = _only(graph.rewrite_store(thread_ids=["t1"]))

            assert thread.reason == (
                f"the saver did not store the bytes written to checkpoint "
                f"{checkpoint_id}: they read unchanged before the write, but a "
                f"read of the store still migrates them. put() of the saver "
                f"must store each channel value it receives under its new "
                f"version. Fix the saver, then rerun. See 'Checkpointers' in "
                f"docs/event-migrations.md"
            )

    def when_the_read_after_the_write_refuses_the_thread():
        def it_reports_the_reason_of_that_read():
            saver = _StrayEntrySaver()
            graph, _config, _old, _new = _renamed_history_pair(saver, "t1")
            saver.armed = True

            thread = _only(graph.rewrite_store(thread_ids=["t1"]))

            assert thread.reason.startswith(
                "the rewrite is stored, but the read after the write refuses the "
                "thread: channel 'events' holds 1 entry that revives as dict, "
            )

        async def it_reports_the_reason_through_the_async_saver():
            saver = _StrayEntrySaver()
            graph, _config, _old, _new = await _arenamed_history_pair(saver, "t1")
            saver.armed = True

            thread = _only(await graph.arewrite_store(thread_ids=["t1"]))

            assert thread.reason.startswith(
                "the rewrite is stored, but the read after the write refuses the "
                "thread: channel 'events' holds 1 entry that revives as dict, "
            )


def describe_rewrite_report():
    def it_is_exported_at_the_top_level():
        saver = MemorySaver()
        graph, _config, _old, _new = _renamed_history_pair(saver, "t1")

        report = graph.plan_rewrite(thread_ids=["t1"])

        assert isinstance(report, RewriteReport)
        assert all(isinstance(t, ThreadRewrite) for t in report.threads)

    def it_reads_only_the_listed_threads():
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

        report = graph.plan_rewrite(thread_ids=["gone-a", "gone-b", "plain"])

        lines = str(report).splitlines()
        assert lines[0] == "plan: 3 threads, 0 rewrite, 1 unchanged, 2 refused"
        assert [t.thread_id for t in report.refused] == ["gone-a", "gone-b"]
        assert lines[1].startswith("  gone-a: refused, history names")
        assert lines[2].startswith("  gone-b: refused, history names")

    def it_names_the_namespace_of_a_refused_subgraph_checkpoint():
        saver = MemorySaver()
        saver.serde = NamespaceAwareSerde(events=(Started,))
        graph = EventGraph([_completes], checkpointer=saver)

        report = graph.plan_rewrite(thread_ids=["ghost"], checkpoint_ns="child:1")

        assert str(report).splitlines()[1] == (
            "  ghost in subgraph namespace 'child:1': refused, no checkpoint "
            "for this thread id"
        )

    def it_says_rewritten_once_applied():
        saver = MemorySaver()
        graph, _config, _old, _new = _renamed_history_pair(saver, "t1")

        report = graph.rewrite_store(thread_ids=["t1"])

        assert str(report) == "applied: 1 thread, 1 rewrite, 0 unchanged, 0 refused"


def describe_async_twins():
    # The consumer's AsyncPostgresSaver refuses a sync checkpointer call
    # from the running loop. Every read and write must go through the
    # async API.

    def when_a_root_thread_holds_a_renamed_identity():
        async def it_plans_through_the_async_saver():
            saver = _AsyncOnlySaver()
            graph, _config, old, new = await _arenamed_history_pair(saver, "t1")

            report = await graph.aplan_rewrite(thread_ids=["t1"])

            assert report.applied is False
            [thread] = report.threads
            assert thread.migrated == ((_identity(old), _identity(new)),)

        async def it_rewrites_through_the_async_saver():
            saver = _AsyncOnlySaver()
            graph, _config, old, new = await _arenamed_history_pair(saver, "t1")

            report = await graph.arewrite_store(thread_ids=["t1"])

            assert report.applied is True
            [thread] = report.threads
            assert thread.status == "rewrite"
            stored = await _astored_identities(saver, "t1")
            assert (old.__module__, old.__qualname__) not in stored
            assert (new.__module__, new.__qualname__) in stored
            [again] = (await graph.aplan_rewrite(thread_ids=["t1"])).threads
            assert again.status == "unchanged"

    def when_a_subgraph_checkpoint_holds_a_renamed_identity():
        async def it_plans_the_namespace_and_writes_nothing():
            saver = MemorySaver()
            graph, ns, old, _new = _subgraph_pair(saver, "t1")

            report = await graph.aplan_rewrite(thread_ids=["t1"], checkpoint_ns=ns)

            [thread] = report.threads
            assert (thread.status, thread.checkpoint_ns) == ("rewrite", ns)
            assert (old.__module__, old.__qualname__) in _stored_identities(
                saver, "t1", ns
            )

        async def it_rewrites_in_place_and_keeps_the_parent_link():
            saver = MemorySaver()
            graph, ns, old, new = _subgraph_pair(saver, "t1")
            before = saver.get_tuple(_cfg("t1", ns))
            assert before is not None

            report = await graph.arewrite_store(thread_ids=["t1"], checkpoint_ns=ns)

            assert [t.status for t in report.threads] == ["rewrite"]
            stored = _stored_identities(saver, "t1", ns)
            assert (old.__module__, old.__qualname__) not in stored
            assert (new.__module__, new.__qualname__) in stored
            after = saver.get_tuple(_cfg("t1", ns))
            assert after is not None
            assert after.checkpoint["id"] == before.checkpoint["id"]
            assert after.parent_config == before.parent_config


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

        # 1. The server-side query returns every thread id in the store.
        thread_ids = ["paused", "answered"]
        assert thread_ids, "the query returned no thread ids"
        # 2. and 3.
        for config in graph.threads_paused_on(EventClass, thread_ids=thread_ids):
            graph.abandon(config, reason="retiring EventClass")
        assert graph.threads_paused_on(EventClass, thread_ids=thread_ids) == []
        # 4.
        report = graph.plan_rewrite(drop=(EventClass,), thread_ids=thread_ids)
        assert not report.refused
        assert {t.thread_id for t in report.threads} == {"paused", "answered"}
        # 5.
        report = graph.rewrite_store(drop=(EventClass,), thread_ids=thread_ids)
        assert not report.refused
        # 6. The class is deleted: a serde that no longer reaches it.
        saver.serde = NamespaceAwareSerde(events=(Started,))
        graph = EventGraph([_completes], checkpointer=saver)
        # 7.
        assert graph.unrevivable_threads(thread_ids=thread_ids) == {}
