"""A value class is a stored identity: read, rewrite and gate after a move."""

import contextlib
import enum
import json
import sys
import types
from dataclasses import dataclass
from datetime import timedelta
from typing import TYPE_CHECKING, Any

import ormsgpack
import pydantic
import pytest
from _value_move_fixtures import GONE as _GONE
from _value_move_fixtures import HoldsMoved, Moved
from conftest import Started
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.serde.jsonplus import _option

from langgraph_events import (
    EventGraph,
    EventTopology,
    IntegrationEvent,
    Interrupted,
    ScalarReducer,
    on,
)
from langgraph_events.serde import (
    NamespaceAwareSerde,
    UnreachableMigrationWarning,
    migrate_value_from,
)
from langgraph_events.serde.migrations import (
    BaselineStale,
    Migration,
    MigrationTarget,
    MoveValue,
    SurvivedRelease,
    UncoveredValue,
    UnrevivableValue,
    check,
    rewrite_live_set,
    update_baseline,
)
from langgraph_events.serde.migrations._baseline import Baseline, ValuesUnrecorded

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

    from langgraph.checkpoint.base import BaseCheckpointSaver

_HERE = __name__
_EMPTY_FINGERPRINT = "lge-m1-e3b0c44298fc1c14"


class Priority(enum.StrEnum):
    HIGH = "high"
    LOW = "low"


@dataclass(frozen=True)
class Ref:
    id: int
    priority: Priority
    tags: frozenset[Priority] = frozenset()


class Card(pydantic.BaseModel):
    priority: Priority
    ref_id: int


@dataclass
class Base:
    a: int
    b: int


@dataclass
class Sub(Base):
    b: int = 0


class Outer:
    class Inner(enum.Enum):
        ONE = 1


class Other(enum.Enum):
    ONE = 1


class Held(IntegrationEvent):
    priority: Priority | None = None
    ref: Ref | None = None
    card: Card | None = None


class HoldsBase(IntegrationEvent):
    base: Base | None = None
    sub: Sub | None = None


class HoldsNested(IntegrationEvent):
    inner: Outer.Inner | None = None


@dataclass(frozen=True)
class Validated:
    count: int

    def __post_init__(self) -> None:
        if self.count is None:
            raise ValueError("count is required")


class HoldsValidated(IntegrationEvent):
    validated: Validated | None = None


class Chosen(IntegrationEvent):
    label: str = ""


class AskPriority(Interrupted):
    priority: Priority | None = None


_MOVES = [
    Migration(
        (
            MoveValue(_GONE, "Priority", Priority),
            MoveValue(_GONE, "Ref", Ref),
            MoveValue(_GONE, "Card", Card),
        )
    )
]
_STRICT = [(_HERE, "Priority"), (_HERE, "Ref"), (_HERE, "Card")]


@contextlib.contextmanager
def _gone_module() -> Iterator[types.SimpleNamespace]:
    """The value classes as an earlier release held them, in a module that
    leaves ``sys.modules`` when the block ends."""
    module = types.ModuleType(_GONE)
    sys.modules[_GONE] = module

    class OldPriority(enum.StrEnum):
        HIGH = "high"
        LOW = "low"

    OldPriority.__name__ = "Priority"

    @dataclass(frozen=True)
    class OldRef:
        id: int
        priority: Any
        tags: Any = frozenset()

    OldRef.__name__ = "Ref"

    class OldCard(pydantic.BaseModel):
        priority: Any
        ref_id: int

    OldCard.__name__ = "Card"
    classes = (OldPriority, OldRef, OldCard)
    for cls in classes:
        cls.__module__ = _GONE
        setattr(module, cls.__name__, cls)
    try:
        yield types.SimpleNamespace(Priority=OldPriority, Ref=OldRef, Card=OldCard)
    finally:
        del sys.modules[_GONE]


@pytest.fixture
def temp_module() -> Iterator[types.ModuleType]:
    """A module that binds classes under a test-owned name."""
    module = types.ModuleType("temp_value_module")
    sys.modules[module.__name__] = module
    yield module
    del sys.modules[module.__name__]


def _bind(module: types.ModuleType, cls: type) -> None:
    cls.__module__ = module.__name__
    setattr(module, cls.__name__, cls)


def _old_bytes(build: Any) -> tuple[str, bytes]:
    with _gone_module() as old:
        return NamespaceAwareSerde(events=[Held]).dumps_typed(build(old))


def _record(code: int, module: str, name: str, payload: object) -> tuple[str, bytes]:
    inner = ormsgpack.packb([module, name, payload], option=_option)
    return "msgpack", ormsgpack.packb(ormsgpack.Ext(code, inner), option=_option)


def _serde(**kwargs: Any) -> NamespaceAwareSerde:
    return NamespaceAwareSerde(
        _MOVES, events=[Held], allowed_msgpack_modules=_STRICT, **kwargs
    )


def describe_migrate_value_from():
    def when_a_value_class_moved():
        def it_revives_an_enum_in_an_event_field():
            data = _old_bytes(lambda old: Held(priority=old.Priority.HIGH))

            assert _serde().loads_typed(data).priority is Priority.HIGH

        def it_revives_a_dataclass_and_the_enum_inside_it():
            data = _old_bytes(lambda old: Held(ref=old.Ref(1, old.Priority.LOW)))

            assert _serde().loads_typed(data).ref == Ref(1, Priority.LOW)

        def it_revives_an_enum_inside_a_frozenset():
            data = _old_bytes(
                lambda old: Held(
                    ref=old.Ref(1, old.Priority.LOW, frozenset({old.Priority.HIGH}))
                )
            )

            assert _serde().loads_typed(data).ref.tags == frozenset({Priority.HIGH})

        def it_revives_a_pydantic_model_and_its_enum():
            data = _old_bytes(
                lambda old: Held(card=old.Card(priority=old.Priority.HIGH, ref_id=2))
            )

            assert _serde().loads_typed(data).card == Card(
                priority=Priority.HIGH, ref_id=2
            )

        def it_revives_a_bare_channel_value():
            data = _old_bytes(lambda old: {"scalar": old.Priority.LOW})

            assert _serde().loads_typed(data) == {"scalar": Priority.LOW}

        def it_writes_the_new_module_on_a_second_encode():
            data = _old_bytes(lambda old: Held(ref=old.Ref(1, old.Priority.LOW)))
            serde = _serde()

            _kind, again = serde.dumps_typed(serde.loads_typed(data))

            assert _GONE.encode() not in again

    def when_a_decorator_declares_the_move():
        def it_revives_the_old_identity():
            serde = NamespaceAwareSerde(events=[HoldsMoved])
            data = _record(0, _GONE, "Moved", "on")

            assert serde.loads_typed(data) is Moved.ON

        def it_names_the_move_in_the_migration_keys():
            serde = NamespaceAwareSerde(events=[HoldsMoved])

            labels = {key.label for key in serde.client_migrations().keys}

            assert labels == {f"move {_GONE}:Moved"}

    def when_the_decorated_class_is_not_an_enum():
        def it_revives_a_dataclass(temp_module: types.ModuleType):
            @migrate_value_from(_GONE)
            @dataclass
            class Point:
                x: int

            _bind(temp_module, Point)
            serde = NamespaceAwareSerde(values=[Point])

            assert serde.loads_typed(_record(2, _GONE, "Point", {"x": 1})) == Point(1)

        def it_revives_a_pydantic_model(temp_module: types.ModuleType):
            @migrate_value_from(_GONE)
            class Box(pydantic.BaseModel):
                x: int

            _bind(temp_module, Box)
            serde = NamespaceAwareSerde(values=[Box])
            inner = ormsgpack.packb(
                [_GONE, "Box", {"x": 1}, "model_validate_json"], option=_option
            )
            data = ormsgpack.packb(ormsgpack.Ext(5, inner), option=_option)

            assert serde.loads_typed(("msgpack", data)) == Box(x=1)

    def when_a_decorated_sibling_is_not_reached():
        def it_warns_that_the_move_does_nothing(temp_module: types.ModuleType):
            class Reached(enum.Enum):
                ONE = 1

            @migrate_value_from(_GONE)
            class Lonely(enum.Enum):
                ONE = 1

            _bind(temp_module, Reached)
            _bind(temp_module, Lonely)

            with pytest.warns(UnreachableMigrationWarning, match="Lonely"):
                NamespaceAwareSerde(values=[Reached])

    def when_no_move_names_the_old_module():
        def it_gives_langgraph_raw_data():
            data = _old_bytes(lambda old: Held(priority=old.Priority.HIGH))
            serde = NamespaceAwareSerde(events=[Held], allowed_msgpack_modules=_STRICT)

            assert serde.loads_typed(data).priority == "high"

        def it_keeps_the_fingerprint_of_the_earlier_release():
            serde = NamespaceAwareSerde(events=[Held])

            assert serde.client_migrations().fingerprint == _EMPTY_FINGERPRINT

        def it_hands_langgraph_the_stored_bytes_unchanged():
            serde = NamespaceAwareSerde(events=[Held])
            data = serde.dumps_typed(Priority.HIGH)
            seen: list[bytes] = []
            real = serde._unpack_ext_hook

            def spy(code: int, payload: bytes) -> object:
                seen.append(payload)
                return real(code, payload)

            serde._unpack_ext_hook = spy

            serde.loads_typed(data)

            assert seen == [ormsgpack.unpackb(data[1], ext_hook=lambda _c, p: p)]

    def when_a_move_is_declared():
        def when_the_bytes_name_no_old_module():
            def it_hands_langgraph_the_stored_bytes_unchanged():
                serde = _serde()
                data = serde.dumps_typed(Priority.HIGH)
                seen: list[bytes] = []
                real = serde._unpack_ext_hook

                def spy(code: int, payload: bytes) -> object:
                    seen.append(payload)
                    return real(code, payload)

                serde._unpack_ext_hook = spy

                serde.loads_typed(data)

                assert seen == [ormsgpack.unpackb(data[1], ext_hook=lambda _c, p: p)]

        def it_changes_the_fingerprint():
            assert _serde().client_migrations().fingerprint != _EMPTY_FINGERPRINT

    def when_the_private_record_layout_is_read():
        def it_finds_module_and_name_first_in_each_value_code():
            serde = NamespaceAwareSerde(events=[Held])
            found: dict[int, set[tuple[str, str]]] = {}

            def hook(code: int, payload: bytes) -> None:
                inner = ormsgpack.unpackb(
                    payload, ext_hook=hook, option=ormsgpack.OPT_NON_STR_KEYS
                )
                found.setdefault(code, set()).add((inner[0], inner[1]))

            for item in (
                Ref(1, Priority.LOW),
                Card(priority=Priority.LOW, ref_id=1),
                timedelta(days=1),
            ):
                ormsgpack.unpackb(serde.dumps_typed(item)[1], ext_hook=hook)

            assert (_HERE, "Priority") in found[0]
            assert ("datetime", "timedelta") in found[1]
            assert (_HERE, "Ref") in found[2]
            assert (_HERE, "Card") in found[5]

    def when_a_move_is_invalid():
        def it_refuses_a_source_equal_to_its_target():
            with pytest.raises(ValueError, match="moves onto itself"):
                NamespaceAwareSerde(
                    [Migration((MoveValue(_HERE, "Priority", Priority),))]
                )

        def it_refuses_an_event_target():
            with pytest.raises(ValueError, match="is an event"):
                NamespaceAwareSerde([Migration((MoveValue(_GONE, "Held", Held),))])

        def it_refuses_two_targets_for_one_source():
            with pytest.raises(ValueError, match="two targets"):
                NamespaceAwareSerde(
                    [
                        Migration(
                            (
                                MoveValue(_GONE, "X", Priority),
                                MoveValue(_GONE, "X", Other),
                            )
                        )
                    ]
                )

        def it_refuses_a_source_of_this_library():
            with pytest.raises(ValueError, match="this library owns"):
                NamespaceAwareSerde(
                    [Migration((MoveValue("langgraph_events.x", "Y", Priority),))]
                )

        def it_refuses_a_source_that_is_another_live_class():
            with pytest.raises(ValueError, match="another live class"):
                NamespaceAwareSerde([Migration((MoveValue(_HERE, "Other", Priority),))])

        def it_refuses_legacy_write():
            with pytest.raises(ValueError, match="legacy_write"):
                NamespaceAwareSerde(_MOVES, legacy_write=True)

    def when_the_old_module_still_imports_the_target():
        def it_accepts_the_move():
            module = types.ModuleType(_GONE)
            module.Priority = Priority  # type: ignore[attr-defined]
            sys.modules[_GONE] = module
            try:
                NamespaceAwareSerde(
                    [Migration((MoveValue(_GONE, "Priority", Priority),))]
                )
            finally:
                del sys.modules[_GONE]


def describe_NamespaceAwareSerde():
    def when_a_reached_enum_is_nested_in_a_class():
        def it_raises_naming_the_field():
            with pytest.raises(ValueError, match=r"HoldsNested\.inner"):
                NamespaceAwareSerde(events=[HoldsNested])

    def when_a_channel_type_reaches_a_nested_enum():
        def it_raises_naming_the_channel():
            with pytest.raises(ValueError, match="channel 'level'"):
                NamespaceAwareSerde(channel_types={"level": Outer.Inner})

    def when_values_names_a_root():
        def it_records_a_class_no_annotation_reaches():
            serde = NamespaceAwareSerde(values=[Other])

            assert serde.value_identities() == {(_HERE, "Other"): "enum"}

    def when_a_model_holds_an_enum():
        def it_records_the_model_and_the_enum():
            serde = NamespaceAwareSerde(values=[Card])

            assert serde.value_identities() == {
                (_HERE, "Card"): "model",
                (_HERE, "Priority"): "enum",
            }


def describe_record_reads():
    def when_a_move_applies():
        def it_records_a_touched_value_read():
            data = _old_bytes(lambda old: old.Priority.HIGH)
            serde = _serde()

            with serde._record_reads() as reads:
                serde.loads_typed(data)

            assert [(r.stored, r.resolved, r.revived, r.touched) for r in reads] == [
                ((_GONE, "Priority"), (_HERE, "Priority"), True, True)
            ]

    def when_a_recorded_dataclass_lost_a_field():
        def it_records_a_value_that_did_not_revive():
            serde = NamespaceAwareSerde(events=[HoldsBase])

            with serde._record_reads() as reads:
                serde.loads_typed(_record(2, _HERE, "Base", {"a": 1}))

            assert [r.revived for r in reads] == [False]

        def when_a_subclass_instance_revives_beside_it():
            def it_judges_each_record_on_its_exact_type():
                lost = _record(2, _HERE, "Base", {"a": 1})
                kept = _record(2, _HERE, "Sub", {"a": 1})
                data = ormsgpack.packb(
                    [
                        ormsgpack.unpackb(lost[1], ext_hook=ormsgpack.Ext),
                        ormsgpack.unpackb(kept[1], ext_hook=ormsgpack.Ext),
                    ],
                    option=_option,
                )
                serde = NamespaceAwareSerde(events=[HoldsBase])

                with serde._record_reads() as reads:
                    serde.loads_typed(("msgpack", data))

                assert {r.stored[1]: r.revived for r in reads} == {
                    "Base": False,
                    "Sub": True,
                }


def _cfg(thread_id: str) -> dict[str, dict[str, str]]:
    return {"configurable": {"thread_id": thread_id, "checkpoint_ns": ""}}


def _reducer(ref: Any, priority: Any) -> ScalarReducer[Chosen, Any]:
    """A channel that holds a value and no event. It holds a dataclass, because
    ``PostgresSaver`` stores a bare ``str`` subclass inline, with no identity."""
    return ScalarReducer[Chosen, Ref](
        name="ref",
        event_type=Chosen,
        fn=lambda event: ref(1, priority(event.label)),
    )


@on(Started)
def _choose(event: Started) -> Chosen:
    return Chosen(label="high")


def _current_graph(saver: BaseCheckpointSaver[Any]) -> EventGraph:
    reducer = _reducer(Ref, Priority)
    saver.serde = EventTopology([_choose], reducers=[reducer]).serde(_MOVES)
    return EventGraph([_choose], reducers=[reducer], checkpointer=saver)


def _seed_old(saver: BaseCheckpointSaver[Any], thread_id: str) -> None:
    """A real run of the earlier release: its value-only channel names the
    module that leaves ``sys.modules`` after the run."""
    with _gone_module() as old:
        saver.serde = NamespaceAwareSerde(events=[Started, Chosen])
        graph = EventGraph(
            [_choose], reducers=[_reducer(old.Ref, old.Priority)], checkpointer=saver
        )
        graph.invoke(Started(), config=_cfg(thread_id))


def _target(saver: BaseCheckpointSaver[Any], baseline: Path) -> MigrationTarget:
    return MigrationTarget(
        graph=_current_graph,
        baseline=baseline,
        store=lambda: contextlib.nullcontext(saver),
    )


def _rewrite(target: MigrationTarget) -> Any:
    expect = target.snapshot().migrations.fingerprint
    return rewrite_live_set(target, expect=expect, apply=True, log=sys.stderr)


def _memory_blob(saver: MemorySaver, thread_id: str, channel: str) -> bytes:
    tup = saver.get_tuple(_cfg(thread_id))
    assert tup is not None
    version = tup.checkpoint["channel_versions"][channel]
    return saver.blobs[(thread_id, "", channel, version)][1]


def _ext(code: int, module: str, name: str, payload: object) -> ormsgpack.Ext:
    return ormsgpack.Ext(code, ormsgpack.packb([module, name, payload], option=_option))


def _store_blob(saver: MemorySaver, thread_id: str, channel: str, value: Any) -> None:
    tup = saver.get_tuple(_cfg(thread_id))
    assert tup is not None
    version = tup.checkpoint["channel_versions"][channel]
    saver.blobs[(thread_id, "", channel, version)] = (
        "msgpack",
        ormsgpack.packb(value, option=_option),
    )


@on(Held)
def _on_held(event: Held) -> None:
    return None


def _held_graph(saver: BaseCheckpointSaver[Any]) -> EventGraph:
    saver.serde = NamespaceAwareSerde(_MOVES, events=[Held])
    return EventGraph([_on_held], checkpointer=saver)


@on(Started)
def _ask(event: Started) -> AskPriority:
    return AskPriority(priority=_ask.priority)  # type: ignore[attr-defined]


def _ask_graph(saver: BaseCheckpointSaver[Any]) -> EventGraph:
    saver.serde = NamespaceAwareSerde(_MOVES, events=[Started, AskPriority])
    return EventGraph([_ask], checkpointer=saver)


def _pending(saver: MemorySaver, thread_id: str) -> list[Any]:
    tup = saver.get_tuple(_cfg(thread_id))
    assert tup is not None
    checkpoint_id = tup.config["configurable"]["checkpoint_id"]
    return list(saver.writes[(thread_id, "", checkpoint_id)].values())


def describe_rewrite_live_set():
    def when_a_value_moved():
        def when_the_store_is_memory():
            def it_writes_the_new_module_into_a_value_only_channel(tmp_path: Path):
                saver = MemorySaver()
                _seed_old(saver, "t1")
                assert _GONE.encode() in _memory_blob(saver, "t1", "ref")

                _rewrite(_target(saver, tmp_path / "b.json"))

                blob = _memory_blob(saver, "t1", "ref")
                assert _GONE.encode() not in blob
                assert _HERE.encode() in blob

            def it_reads_unchanged_on_a_second_pass(tmp_path: Path):
                saver = MemorySaver()
                _seed_old(saver, "t1")
                target = _target(saver, tmp_path / "b.json")

                outcome = _rewrite(target)

                assert [t.status for t in outcome.final.threads] == ["unchanged"]

            def it_keeps_the_revived_value(tmp_path: Path):
                saver = MemorySaver()
                _seed_old(saver, "t1")

                _rewrite(_target(saver, tmp_path / "b.json"))

                tup = saver.get_tuple(_cfg("t1"))
                assert tup is not None
                assert tup.checkpoint["channel_values"]["ref"] == Ref(1, Priority.HIGH)

        def when_the_store_is_postgres():
            def it_writes_the_new_module_into_a_value_only_channel(
                tmp_path: Path, postgres_url: str
            ):
                import psycopg
                from langgraph.checkpoint.postgres import PostgresSaver

                with PostgresSaver.from_conn_string(postgres_url) as saver:
                    _seed_old(saver, "t1")
                    outcome = _rewrite(_target(saver, tmp_path / "b.json"))
                    tup = saver.get_tuple(_cfg("t1"))

                assert tup is not None
                version = tup.checkpoint["channel_versions"]["ref"]
                with psycopg.connect(postgres_url) as conn:
                    row = conn.execute(
                        "SELECT blob FROM checkpoint_blobs WHERE thread_id = 't1' "
                        "AND channel = 'ref' AND version = %s",
                        (version,),
                    ).fetchone()
                assert row is not None
                assert _GONE.encode() not in bytes(row[0])
                assert _HERE.encode() in bytes(row[0])
                assert [t.status for t in outcome.final.threads] == ["unchanged"]

        def when_the_store_is_sqlite():
            def it_reads_unchanged_on_a_second_pass(tmp_path: Path):
                from langgraph.checkpoint.sqlite import SqliteSaver

                path = str(tmp_path / "s.db")
                with SqliteSaver.from_conn_string(path) as saver:
                    _seed_old(saver, "t1")
                    outcome = _rewrite(_target(saver, tmp_path / "b.json"))
                    tup = saver.get_tuple(_cfg("t1"))

                assert tup is not None
                assert tup.checkpoint["channel_values"]["ref"] == Ref(1, Priority.HIGH)
                assert [t.status for t in outcome.final.threads] == ["unchanged"]

    def when_an_event_field_holds_a_moved_value():
        def it_writes_the_new_module_into_the_log(tmp_path: Path):
            saver = MemorySaver()
            with _gone_module() as old:
                saver.serde = NamespaceAwareSerde(events=[Held])
                graph = EventGraph([_on_held], checkpointer=saver)
                graph.invoke(Held(priority=old.Priority.HIGH), config=_cfg("t1"))
            assert _GONE.encode() in _memory_blob(saver, "t1", "events")

            outcome = _rewrite(
                MigrationTarget(
                    graph=_held_graph,
                    baseline=tmp_path / "b.json",
                    store=lambda: contextlib.nullcontext(saver),
                )
            )

            assert _GONE.encode() not in _memory_blob(saver, "t1", "events")
            assert [t.status for t in outcome.final.threads] == ["unchanged"]

    def when_a_paused_thread_holds_a_moved_value():
        def it_writes_the_new_module_into_the_pending_interrupt(tmp_path: Path):
            saver = MemorySaver()
            with _gone_module() as old:
                saver.serde = NamespaceAwareSerde(events=[Started, AskPriority])
                _ask.priority = old.Priority.HIGH
                graph = EventGraph([_ask], checkpointer=saver)
                graph.invoke(Started(), config=_cfg("t1"))
            assert any(
                _GONE.encode() in blob
                for (_task, _channel, (_kind, blob), _path) in _pending(saver, "t1")
            )

            outcome = _rewrite(
                MigrationTarget(
                    graph=_ask_graph,
                    baseline=tmp_path / "b.json",
                    store=lambda: contextlib.nullcontext(saver),
                )
            )

            assert not any(
                _GONE.encode() in blob
                for (_task, _channel, (_kind, blob), _path) in _pending(saver, "t1")
            )
            assert [t.status for t in outcome.final.threads] == ["unchanged"]

    def when_a_value_in_a_moved_thread_reads_as_raw_data():
        def it_refuses_the_thread(tmp_path: Path):
            saver = MemorySaver()
            _seed_old(saver, "t1")
            _store_blob(
                saver,
                "t1",
                "ref",
                [_ext(0, _GONE, "Priority", "high"), _ext(0, _HERE, "Priority", "no")],
            )
            store = _target(saver, tmp_path / "b.json").build(saver).store

            report = store.plan_rewrite(thread_ids=["t1"])

            assert [t.status for t in report.threads] == ["refused"]
            assert f"{_HERE}:Priority reads back as raw data" in (
                report.threads[0].reason
            )

    def when_a_lost_value_sits_in_an_unchanged_thread():
        def it_reads_unchanged(tmp_path: Path):
            saver = MemorySaver()
            _current_graph(saver).invoke(Started(), config=_cfg("t1"))
            _store_blob(saver, "t1", "ref", [_ext(0, _HERE, "Priority", "no")])
            store = _target(saver, tmp_path / "b.json").build(saver).store

            report = store.plan_rewrite(thread_ids=["t1"])

            assert [t.status for t in report.threads] == ["unchanged"]

    def when_a_completed_task_write_holds_a_moved_value():
        def it_refuses_the_thread(tmp_path: Path):
            saver = MemorySaver()
            _seed_old(saver, "t1")
            tup = saver.get_tuple(_cfg("t1"))
            assert tup is not None
            with _gone_module() as old:
                saver.serde = NamespaceAwareSerde(events=[Started, Chosen])
                saver.put_writes(
                    tup.config, [("ref", old.Priority.LOW)], task_id="task-1"
                )
            store = _target(saver, tmp_path / "b.json").build(saver).store

            report = store.plan_rewrite(thread_ids=["t1"])

            assert [t.status for t in report.threads] == ["refused"]
            assert "a migration rewrites" in report.threads[0].reason


def _gate_graph(saver: BaseCheckpointSaver[Any]) -> EventGraph:
    saver.serde = NamespaceAwareSerde(events=[Started, HoldsMoved])
    return EventGraph([_noop_moved], checkpointer=saver)


def _plain_graph(saver: BaseCheckpointSaver[Any]) -> EventGraph:
    saver.serde = NamespaceAwareSerde(events=[Started, Held])
    return EventGraph([_noop_moved], checkpointer=saver)


def _validated_graph(saver: BaseCheckpointSaver[Any]) -> EventGraph:
    saver.serde = NamespaceAwareSerde(events=[Started, HoldsValidated])
    return EventGraph([_noop_moved], checkpointer=saver)


def _strict_old_graph(saver: BaseCheckpointSaver[Any]) -> EventGraph:
    saver.serde = NamespaceAwareSerde(
        _MOVES,
        events=[Started, Held],
        allowed_msgpack_modules=[
            (_GONE, "Priority"),
            (_GONE, "Ref"),
            (_GONE, "Card"),
            *_STRICT[1:],
        ],
    )
    return EventGraph([_noop_moved], checkpointer=saver)


@on(Started)
def _noop_moved(event: Started) -> None:
    return None


def _add_value(path: Path, module: str, name: str, kind: str) -> None:
    raw = json.loads(path.read_text())
    raw["values"].append({"module": module, "name": name, "kind": kind})
    path.write_text(json.dumps(raw))


def describe_migration_gate():
    def when_a_recorded_value_does_not_import():
        def it_reports_uncovered_value(tmp_path: Path):
            path = tmp_path / "b.json"
            target = MigrationTarget(graph=_plain_graph, baseline=path)
            update_baseline(target)
            _add_value(path, _GONE, "Priority", "enum")

            report = check(target)

            assert [f.subject for f in report.findings] == [f"{_GONE}:Priority"]
            assert isinstance(report.findings[0], UncoveredValue)
            assert report.findings[0].code == "uncovered-value"

        def when_a_move_names_it():
            def it_passes(tmp_path: Path):
                path = tmp_path / "b.json"
                target = MigrationTarget(graph=_gate_graph, baseline=path)
                update_baseline(target)
                _add_value(path, _GONE, "Moved", "enum")

                assert check(target).passed

    def when_only_an_import_at_the_old_module_resolves_it():
        def it_reports_uncovered_value(tmp_path: Path):
            path = tmp_path / "b.json"
            target = MigrationTarget(graph=_plain_graph, baseline=path)
            update_baseline(target)
            _add_value(path, _GONE, "Priority", "enum")
            module = types.ModuleType(_GONE)
            module.Priority = Priority  # type: ignore[attr-defined]
            sys.modules[_GONE] = module
            try:
                report = check(target)
            finally:
                del sys.modules[_GONE]

            assert [type(f) for f in report.findings] == [UncoveredValue]

    def when_the_allowlist_names_the_old_module_only():
        def it_reports_unrevivable_value(tmp_path: Path):
            path = tmp_path / "b.json"
            target = MigrationTarget(graph=_strict_old_graph, baseline=path)
            update_baseline(target)
            _add_value(path, _GONE, "Priority", "enum")

            report = check(target)

            found = [f for f in report.findings if isinstance(f, UnrevivableValue)]
            assert f"{_GONE}:Priority" in {f.subject for f in found}
            assert found[0].code == "unrevivable-value"

    def when_a_dataclass_refuses_the_none_placeholder():
        def it_reports_no_finding(tmp_path: Path):
            path = tmp_path / "b.json"
            target = MigrationTarget(graph=_validated_graph, baseline=path)

            update_baseline(target)

            assert check(target).passed

    def when_a_shipped_move_survives():
        def it_reports_survived_release(tmp_path: Path):
            path = tmp_path / "b.json"
            previous = tmp_path / "previous.json"
            target = MigrationTarget(graph=_gate_graph, baseline=path)
            update_baseline(target)
            previous.write_text(path.read_text())

            report = check(target, previous=previous)

            assert [type(f) for f in report.findings] == [SurvivedRelease]
            assert report.findings[0].subject == f"move {_GONE}:Moved"


def describe_baseline_values():
    def when_the_code_reaches_values():
        def it_records_the_kind_of_each_value(tmp_path: Path):
            path = tmp_path / "b.json"

            update_baseline(MigrationTarget(graph=_plain_graph, baseline=path))

            assert json.loads(path.read_text())["values"] == [
                {"kind": "model", "module": _HERE, "name": "Card"},
                {"kind": "enum", "module": _HERE, "name": "Priority"},
                {"kind": "dataclass", "module": _HERE, "name": "Ref"},
            ]

        def it_writes_version_5(tmp_path: Path):
            path = tmp_path / "b.json"

            update_baseline(MigrationTarget(graph=_plain_graph, baseline=path))

            assert json.loads(path.read_text())["version"] == 5

    def when_the_baseline_is_version_4():
        def it_reads_values_as_unrecorded(tmp_path: Path):
            path = tmp_path / "b.json"
            path.write_text(json.dumps({"version": 4, "events": [], "migrations": []}))

            assert Baseline.read(path).values == ValuesUnrecorded(4)

        def it_reports_a_stale_baseline(tmp_path: Path):
            path = tmp_path / "b.json"
            target = MigrationTarget(graph=_plain_graph, baseline=path)
            update_baseline(target)
            raw = json.loads(path.read_text())
            del raw["values"]
            raw["version"] = 4
            path.write_text(json.dumps(raw))

            report = check(target)

            stale = [f for f in report.findings if isinstance(f, BaselineStale)]
            assert "~ version 4 -> 5" in stale[0].changes

    def when_a_version_5_file_has_no_values():
        def it_refuses_the_file(tmp_path: Path):
            path = tmp_path / "b.json"
            path.write_text(json.dumps({"version": 5, "events": [], "migrations": []}))

            with pytest.raises(ValueError, match="values"):
                Baseline.read(path)

    def when_the_previous_release_shipped_a_move_the_code_dropped():
        def it_drops_the_source_from_values(tmp_path: Path):
            path = tmp_path / "b.json"
            previous = tmp_path / "previous.json"
            update_baseline(MigrationTarget(graph=_gate_graph, baseline=path))
            _add_value(path, _GONE, "Moved", "enum")
            previous.write_text(path.read_text())
            without_move = MigrationTarget(graph=_plain_graph, baseline=path)

            update_baseline(without_move, previous=previous)

            names = {
                (entry["module"], entry["name"])
                for entry in json.loads(path.read_text())["values"]
            }
            assert (_GONE, "Moved") not in names
