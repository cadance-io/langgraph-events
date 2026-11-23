"""A value class is a stored identity: read, rewrite and gate after a move."""

import contextlib
import enum
import json
import sys
import threading
import types
from dataclasses import InitVar, dataclass, field
from datetime import date, timedelta
from typing import TYPE_CHECKING, Any, NamedTuple, TypedDict

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
    Invocation,
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
from langgraph_events.serde.migrations._baseline import Baseline

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


@dataclass
class Exported:
    count: int

    def dict(self) -> dict[str, int]:
        return {"count": self.count}


class Dated(enum.Enum):
    START = date(2020, 1, 1)


@dataclass
class Derived:
    count: int
    total: int = field(init=False, default=0)


@dataclass
class NeedsSeed:
    count: int
    seed: InitVar[int]


@dataclass
class OptionalSeed:
    count: int
    seed: InitVar[int] = 0


@dataclass(init=False)
class RenamedInit:
    count: int

    def __init__(self, amount: int) -> None:
        self.count = amount


@dataclass(init=False)
class MatchingInit:
    count: int

    def __init__(self, count: int) -> None:
        self.count = count


class OtherSlot(TypedDict):
    other: Other


class OtherPair(NamedTuple):
    other: Other


class RefSlot(TypedDict):
    ref: Ref
    other: Other


class SlotCard(pydantic.BaseModel):
    slot: RefSlot


class PairCard(pydantic.BaseModel):
    pair: OtherPair


class NestedEnumCard(pydantic.BaseModel):
    class Kind(enum.Enum):
        X = "x"

    kind: Kind = Kind.X


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

        def it_revives_a_named_tuple(temp_module: types.ModuleType):
            @migrate_value_from(_GONE)
            class Point(NamedTuple):
                x: int

            _bind(temp_module, Point)
            serde = NamespaceAwareSerde(values=[Point])

            revived = serde.loads_typed(_record(2, _GONE, "Point", {"x": 1}))

            assert type(revived) is Point
            assert revived == Point(1)

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

    def when_a_move_is_declared():
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
        @pytest.mark.parametrize(
            ("operations", "match"),
            [
                ((MoveValue(_HERE, "Priority", Priority),), "moves onto itself"),
                ((MoveValue(_GONE, "Held", Held),), "is an event"),
                (
                    (MoveValue(_GONE, "X", Priority), MoveValue(_GONE, "X", Other)),
                    "two targets",
                ),
                (
                    (MoveValue("langgraph_events.x", "Y", Priority),),
                    "this library owns",
                ),
                ((MoveValue(_HERE, "Other", Priority),), "another live class"),
            ],
        )
        def it_refuses_the_move(operations: tuple[MoveValue, ...], match: str):
            with pytest.raises(ValueError, match=match):
                NamespaceAwareSerde([Migration(operations)])

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

            assert serde.value_identities() == {(_HERE, "Other")}

    def when_a_typed_dict_holds_a_value():
        def it_records_the_value_and_not_the_container():
            serde = NamespaceAwareSerde(channel_types={"slot": OtherSlot})

            assert serde.value_identities() == {(_HERE, "Other")}

    def when_a_named_tuple_holds_a_value():
        def it_records_the_named_tuple_and_the_value():
            serde = NamespaceAwareSerde(channel_types={"pair": OtherPair})

            assert serde.value_identities() == {
                (_HERE, "OtherPair"),
                (_HERE, "Other"),
            }

        def when_a_model_holds_the_named_tuple():
            def it_records_only_the_model_and_the_enum():
                serde = NamespaceAwareSerde(values=[PairCard])

                assert serde.value_identities() == {
                    (_HERE, "PairCard"),
                    (_HERE, "Other"),
                }

    def when_a_container_type_holds_a_value():

        def when_a_model_holds_the_container():
            def it_records_only_the_enums_below_the_model():
                serde = NamespaceAwareSerde(values=[SlotCard])

                assert serde.value_identities() == {
                    (_HERE, "SlotCard"),
                    (_HERE, "Other"),
                    (_HERE, "Priority"),
                }

    def when_a_model_holds_a_nested_enum():
        def it_records_the_model_alone():
            serde = NamespaceAwareSerde(values=[NestedEnumCard])

            assert serde.value_identities() == {(_HERE, "NestedEnumCard")}

    def when_a_values_root_sits_in_a_module_named_like_the_standard_library():
        def it_records_the_class(monkeypatch: pytest.MonkeyPatch):
            module = types.ModuleType("tabnanny")
            monkeypatch.setitem(sys.modules, "tabnanny", module)

            class Local(enum.Enum):
                ONE = 1

            _bind(module, Local)

            serde = NamespaceAwareSerde(values=[Local])

            assert serde.value_identities() == {("tabnanny", "Local")}


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


def _retired_graph(saver: BaseCheckpointSaver[Any]) -> EventGraph:
    reducer = _reducer(Ref, Priority)
    saver.serde = EventTopology([_choose], reducers=[reducer]).serde()
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


@pytest.fixture(params=["memory", "sqlite", "postgres"])
def any_saver(request: pytest.FixtureRequest, tmp_path: Path) -> Iterator[Any]:
    if request.param == "memory":
        yield MemorySaver()
    elif request.param == "sqlite":
        from langgraph.checkpoint.sqlite import SqliteSaver

        with SqliteSaver.from_conn_string(str(tmp_path / "s.db")) as saver:
            yield saver
    else:
        from langgraph.checkpoint.postgres import PostgresSaver

        url = request.getfixturevalue("postgres_url")
        with PostgresSaver.from_conn_string(url) as saver:
            yield saver


def describe_rewrite_live_set():
    def when_a_value_moved():
        def it_stores_a_value_that_needs_no_move(any_saver: Any, tmp_path: Path):
            _seed_old(any_saver, "t1")

            outcome = _rewrite(_target(any_saver, tmp_path / "b.json"))

            assert [t.status for t in outcome.final.threads] == ["unchanged"]
            any_saver.serde = NamespaceAwareSerde(events=[Started, Chosen])
            tup = any_saver.get_tuple(_cfg("t1"))
            assert tup is not None
            assert tup.checkpoint["channel_values"]["ref"] == Ref(1, Priority.HIGH)

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
        def it_reads_unchanged_and_names_the_value(tmp_path: Path):
            saver = MemorySaver()
            _current_graph(saver).invoke(Started(), config=_cfg("t1"))
            _store_blob(saver, "t1", "ref", [_ext(0, _HERE, "Priority", "no")])
            store = _target(saver, tmp_path / "b.json").build(saver).store

            report = store.plan_rewrite(thread_ids=["t1"])

            assert [t.status for t in report.threads] == ["unchanged"]
            assert report.threads[0].lost_values == (f"{_HERE}:Priority",)

        def when_a_subclass_instance_revives_beside_it():
            def it_names_only_the_lost_class(tmp_path: Path):
                saver = MemorySaver()
                _current_graph(saver).invoke(Started(), config=_cfg("t1"))
                _store_blob(
                    saver,
                    "t1",
                    "ref",
                    [_ext(2, _HERE, "Base", {"a": 1}), _ext(2, _HERE, "Sub", {"a": 1})],
                )
                store = _target(saver, tmp_path / "b.json").build(saver).store

                report = store.plan_rewrite(thread_ids=["t1"])

                assert report.threads[0].lost_values == (f"{_HERE}:Base",)

    def when_a_stored_identity_is_a_re_export():
        def _lost(saver_path: Path, record: Any) -> Any:
            saver = MemorySaver()
            _current_graph(saver).invoke(Started(), config=_cfg("t1"))
            _store_blob(saver, "t1", "ref", [record])
            store = _target(saver, saver_path / "b.json").build(saver).store
            return store.plan_rewrite(thread_ids=["t1"]).threads[0].lost_values

        def it_counts_the_exact_class_as_revived(
            tmp_path: Path, temp_module: types.ModuleType
        ):
            temp_module.Priority = Priority  # type: ignore[attr-defined]
            record = _ext(0, temp_module.__name__, "Priority", "high")

            assert _lost(tmp_path, record) == ()

        def when_the_read_gives_raw_data():
            def it_stays_lost(tmp_path: Path, temp_module: types.ModuleType):
                temp_module.Priority = Priority  # type: ignore[attr-defined]
                record = _ext(0, temp_module.__name__, "Priority", "no")

                assert _lost(tmp_path, record) == (f"{temp_module.__name__}:Priority",)

        def when_the_read_gives_another_class():
            def it_stays_lost(tmp_path: Path, temp_module: types.ModuleType):
                class Built:
                    pass

                class Shape:
                    def __new__(cls) -> Any:
                        return Built()

                temp_module.Shape = Shape  # type: ignore[attr-defined]
                record = _ext(2, temp_module.__name__, "Shape", {})

                assert _lost(tmp_path, record) == (f"{temp_module.__name__}:Shape",)

        def when_the_name_is_bound_to_a_builtin():
            def it_stays_lost(tmp_path: Path, temp_module: types.ModuleType):
                temp_module.Gone = dict  # type: ignore[attr-defined]
                record = _ext(2, temp_module.__name__, "Gone", {"a": 1})

                assert _lost(tmp_path, record) == (f"{temp_module.__name__}:Gone",)

    def when_a_retired_move_left_old_bytes():
        def it_reads_unchanged_and_names_the_value_in_the_output(tmp_path: Path):
            saver = MemorySaver()
            _seed_old(saver, "t1")
            target = MigrationTarget(
                graph=_retired_graph,
                baseline=tmp_path / "b.json",
                store=lambda: contextlib.nullcontext(saver),
            )

            outcome = _rewrite(target)

            assert [t.status for t in outcome.final.threads] == ["unchanged"]
            assert outcome.clean
            rendered = outcome.render(Invocation("app:target"), "fingerprint")
            assert f"t1: reads as raw data: {_GONE}:" in rendered

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


def _values_graph(value: type) -> Any:
    def graph(saver: BaseCheckpointSaver[Any]) -> EventGraph:
        saver.serde = NamespaceAwareSerde(events=[Started], values=[value])
        return EventGraph([_noop_moved], checkpointer=saver)

    return graph


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


class _RacedSerde(NamespaceAwareSerde):
    """Each read waits for a read in another thread that LangGraph blocks."""

    def loads_typed(self, data: tuple[str, bytes]) -> Any:
        strict = NamespaceAwareSerde(allowed_msgpack_modules=[(_HERE, "Priority")])
        reader = threading.Thread(
            target=strict.loads_typed, args=(_record(0, _HERE, "Other", 1),)
        )
        reader.start()
        reader.join()
        return super().loads_typed(data)


def _raced_graph(saver: BaseCheckpointSaver[Any]) -> EventGraph:
    saver.serde = _RacedSerde(events=[Started], values=[Other])
    return EventGraph([_noop_moved], checkpointer=saver)


def _add_value(path: Path, module: str, name: str) -> None:
    raw = json.loads(path.read_text())
    raw["values"].append({"module": module, "name": name})
    path.write_text(json.dumps(raw))


def describe_migration_gate():
    def when_a_recorded_value_does_not_import():
        def it_reports_uncovered_value(tmp_path: Path):
            path = tmp_path / "b.json"
            target = MigrationTarget(graph=_plain_graph, baseline=path)
            update_baseline(target)
            _add_value(path, _GONE, "Priority")

            report = check(target)

            assert [f.subject for f in report.findings] == [f"{_GONE}:Priority"]
            assert isinstance(report.findings[0], UncoveredValue)
            assert report.findings[0].code == "uncovered-value"

        def when_a_move_names_it():
            def it_passes(tmp_path: Path):
                path = tmp_path / "b.json"
                target = MigrationTarget(graph=_gate_graph, baseline=path)
                update_baseline(target)
                _add_value(path, _GONE, "Moved")

                assert check(target).passed

    def when_only_an_import_at_the_old_module_resolves_it():
        def it_reports_uncovered_value(tmp_path: Path):
            path = tmp_path / "b.json"
            target = MigrationTarget(graph=_plain_graph, baseline=path)
            update_baseline(target)
            _add_value(path, _GONE, "Priority")
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
            _add_value(path, _GONE, "Priority")

            report = check(target)

            found = [f for f in report.findings if isinstance(f, UnrevivableValue)]
            assert f"{_GONE}:Priority" in {f.subject for f in found}
            assert found[0].code == "unrevivable-value"

    def when_a_healthy_class_cannot_take_a_placeholder():
        @pytest.mark.parametrize(
            "value", [Validated, Exported, Dated, OptionalSeed, MatchingInit]
        )
        def it_reports_no_finding(value: type, tmp_path: Path):
            path = tmp_path / "b.json"
            target = MigrationTarget(graph=_values_graph(value), baseline=path)

            update_baseline(target)

            assert check(target).passed

    def when_another_thread_reads_a_blocked_value_during_the_check():
        def it_reports_no_finding(tmp_path: Path):
            path = tmp_path / "b.json"
            target = MigrationTarget(graph=_raced_graph, baseline=path)

            update_baseline(target)

            assert check(target).passed

    def when_the_constructor_refuses_the_stored_fields():
        @pytest.mark.parametrize("value", [Derived, NeedsSeed, RenamedInit])
        def it_reports_unrevivable_value(value: type, tmp_path: Path):
            path = tmp_path / "b.json"
            target = MigrationTarget(graph=_values_graph(value), baseline=path)

            update_baseline(target)

            assert [type(f) for f in check(target).findings] == [UnrevivableValue]

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
        def it_records_each_value(tmp_path: Path):
            path = tmp_path / "b.json"

            update_baseline(MigrationTarget(graph=_plain_graph, baseline=path))

            raw = json.loads(path.read_text())
            assert raw["version"] == 5
            assert raw["values"] == [
                {"module": _HERE, "name": "Card"},
                {"module": _HERE, "name": "Priority"},
                {"module": _HERE, "name": "Ref"},
            ]

    def when_the_baseline_is_version_4():
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

        def when_it_is_the_previous_baseline():
            def it_runs_the_release_rule(tmp_path: Path):
                path = tmp_path / "b.json"
                previous = tmp_path / "previous.json"
                target = MigrationTarget(graph=_gate_graph, baseline=path)
                update_baseline(target)
                raw = json.loads(path.read_text())
                del raw["values"]
                raw["version"] = 4
                previous.write_text(json.dumps(raw))

                report = check(target, previous=previous)

                assert "release" in report.rules_run
                assert [type(f) for f in report.findings] == [SurvivedRelease]

            def it_keeps_the_digest_of_the_earlier_release(tmp_path: Path):
                path = tmp_path / "b.json"
                raw = {"events": [], "retired": [], "handlers": [], "migrations": []}
                path.write_text(json.dumps({"version": 4, **raw}))

                assert Baseline.read(path).digest == "lge-b1-f1b48be375ecf9e1"

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
            _add_value(path, _GONE, "Moved")
            previous.write_text(path.read_text())
            without_move = MigrationTarget(graph=_plain_graph, baseline=path)

            update_baseline(without_move, previous=previous)

            names = {
                (entry["module"], entry["name"])
                for entry in json.loads(path.read_text())["values"]
            }
            assert (_GONE, "Moved") not in names
