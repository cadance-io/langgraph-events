"""A value class is a stored identity: read, rewrite and gate after a move."""

import annotationlib
import contextlib
import copy
import enum
import json
import sys
import threading
import types
from dataclasses import InitVar, dataclass, field, is_dataclass, replace
from datetime import date, timedelta
from typing import TYPE_CHECKING, Any, Literal, NamedTuple, TypedDict

import ormsgpack
import pydantic
import pytest
from _value_move_fixtures import GONE as _GONE
from _value_move_fixtures import HoldsMoved, Moved
from conftest import Started
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.serde.jsonplus import _option
from langgraph.checkpoint.sqlite import SqliteSaver

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
    replay_reducer,
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


AliasOfOther = Other


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


@dataclass
class GenericBox[T]:
    item: T


@dataclass
class ForwardHint:
    path: Path | None = None


type OtherAlias = Other | None
type OtherTree = Other | list[OtherTree]
type BoxAlias[T] = GenericBox[T]
type HiddenAlias = Path


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

    def when_an_annotation_wraps_a_stored_class():
        @pytest.mark.parametrize(
            ("annotation", "names"),
            [
                (OtherAlias, {"Other"}),
                (OtherTree, {"Other"}),
                (BoxAlias[int], {"GenericBox"}),
                (GenericBox[Other], {"GenericBox", "Other"}),
                (Literal[Other.ONE], {"Other"}),
            ],
        )
        def it_records_the_class(annotation: Any, names: set[str]):
            serde = NamespaceAwareSerde(channel_types={"slot": annotation})

            assert serde.value_identities() == {(_HERE, name) for name in names}

    def when_a_type_alias_names_a_type_checking_import():
        def it_records_nothing():
            serde = NamespaceAwareSerde(channel_types={"slot": HiddenAlias})

            assert serde.value_identities() == set()

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


def _note_reducer() -> ScalarReducer[Chosen, Any]:
    """A second value-only channel, which a move in ``ref`` does not write."""
    return ScalarReducer[Chosen, Any](
        name="note", event_type=Chosen, fn=lambda event: [Other.ONE]
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
            assert f"{_HERE}:Priority in channel 'ref' reads back as raw data" in (
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

    def when_only_a_re_export_is_allowed():
        def _plan(saver: MemorySaver, tmp_path: Path, module: str) -> Any:
            def graph(store_saver: BaseCheckpointSaver[Any]) -> EventGraph:
                reducers = [_reducer(Ref, Priority), _note_reducer()]
                store_saver.serde = NamespaceAwareSerde(
                    _MOVES,
                    events=[Started, Chosen],
                    values=[Ref],
                    allowed_msgpack_modules=[*_STRICT, (module, "Other")],
                )
                return EventGraph(
                    [_choose], reducers=reducers, checkpointer=store_saver
                )

            target = MigrationTarget(graph=graph, baseline=tmp_path / "b.json")
            return target.build(saver).store.plan_rewrite(thread_ids=["t1"]).threads[0]

        def _seeded(module: str) -> tuple[MemorySaver, ormsgpack.Ext]:
            saver = MemorySaver()
            reducers = [_reducer(Ref, Priority), _note_reducer()]
            saver.serde = EventTopology([_choose], reducers=reducers).serde(_MOVES)
            EventGraph([_choose], reducers=reducers, checkpointer=saver).invoke(
                Started(), config=_cfg("t1")
            )
            _store_blob(saver, "t1", "ref", [_ext(0, _GONE, "Priority", "high")])
            return saver, _ext(0, module, "Other", 1)

        def when_a_written_channel_holds_the_value():
            def it_refuses_the_thread(tmp_path: Path, temp_module: types.ModuleType):
                temp_module.Other = Other  # type: ignore[attr-defined]
                saver, exported = _seeded(temp_module.__name__)
                _store_blob(
                    saver,
                    "t1",
                    "ref",
                    [_ext(0, _GONE, "Priority", "high"), exported],
                )

                thread = _plan(saver, tmp_path, temp_module.__name__)

                assert thread.status == "refused"
                assert f"{_HERE}:Other in channel 'ref' reads back" in thread.reason

        def when_an_unwritten_channel_holds_the_value():
            def it_plans_the_write(tmp_path: Path, temp_module: types.ModuleType):
                temp_module.Other = Other  # type: ignore[attr-defined]
                saver, exported = _seeded(temp_module.__name__)
                _store_blob(saver, "t1", "note", [exported])

                thread = _plan(saver, tmp_path, temp_module.__name__)

                assert thread.status == "rewrite"

        def when_a_pending_interrupt_holds_the_value():
            def it_refuses_the_thread(tmp_path: Path, temp_module: types.ModuleType):
                temp_module.Other = Other  # type: ignore[attr-defined]
                saver = MemorySaver()
                with _gone_module() as old:
                    saver.serde = NamespaceAwareSerde(events=[Started, AskPriority])
                    _ask.priority = old.Priority.HIGH
                    EventGraph([_ask], checkpointer=saver).invoke(
                        Started(), config=_cfg("t1")
                    )
                tup = saver.get_tuple(_cfg("t1"))
                assert tup is not None
                writes = saver.writes[
                    ("t1", "", tup.config["configurable"]["checkpoint_id"])
                ]
                for key, (task, channel, _blob, path) in list(writes.items()):
                    if channel == "__interrupt__":
                        writes[key] = (
                            task,
                            channel,
                            (
                                "msgpack",
                                ormsgpack.packb(
                                    [
                                        _ext(0, _GONE, "Priority", "high"),
                                        _ext(0, temp_module.__name__, "Other", 1),
                                    ],
                                    option=_option,
                                ),
                            ),
                            path,
                        )

                def graph(store_saver: BaseCheckpointSaver[Any]) -> EventGraph:
                    store_saver.serde = NamespaceAwareSerde(
                        _MOVES,
                        events=[Started, AskPriority],
                        allowed_msgpack_modules=[
                            *_STRICT,
                            (temp_module.__name__, "Other"),
                        ],
                    )
                    return EventGraph([_ask], checkpointer=store_saver)

                target = MigrationTarget(graph=graph, baseline=tmp_path / "b.json")
                store = target.build(saver).store

                thread = store.plan_rewrite(thread_ids=["t1"]).threads[0]

                assert thread.status == "refused"
                assert "the pending __interrupt__ write reads back" in thread.reason

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
            "value",
            [Validated, Exported, Dated, OptionalSeed, MatchingInit, ForwardHint],
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


class AnyCard(pydantic.BaseModel):
    priority: Any
    ref_id: int = 0


class HoldsAny(IntegrationEvent):
    moved: Priority | None = None
    value: Any = None


class DroppedHolder(IntegrationEvent):
    value: Any = None


class Trimmed(IntegrationEvent):
    moved: Priority | None = None
    label: str = ""


class Replaced(IntegrationEvent):
    moved: Priority | None = None
    value: Any = None


class Filled(IntegrationEvent):
    moved: Priority | None = None
    value: Any = None
    note: str = ""


class SplitSource(IntegrationEvent):
    moved: Priority | None = None
    value: Any = None


class SplitTarget(IntegrationEvent):
    moved: Priority | None = None


class RenamedHolder(IntegrationEvent):
    moved: Priority | None = None
    value: Any = None


class FilteredInPlace(IntegrationEvent):
    moved: Priority | None = None
    value: Any = None


def _keep_only_high_in_place(kwargs: dict[str, Any]) -> dict[str, Any]:
    kwargs["value"][:] = [v for v in kwargs["value"] if v is not Priority.LOW]
    return kwargs


class CopiedDeep(IntegrationEvent):
    moved: Priority | None = None
    value: Any = None


class RebuiltValues(IntegrationEvent):
    moved: Priority | None = None
    value: Any = None


class CopiedDict(IntegrationEvent):
    moved: Priority | None = None
    value: Any = None


@dataclass
class Box:
    inner: Any = None


class Wrapped(IntegrationEvent):
    moved: Priority | None = None
    value: Any = None


class ClearsFirst(IntegrationEvent):
    moved: Priority | None = None
    first: Any = None
    second: Any = None


_LOG_EVENTS = [
    CopiedDeep,
    RebuiltValues,
    CopiedDict,
    Wrapped,
    ClearsFirst,
    HoldsAny,
    FilteredInPlace,
    DroppedHolder,
    Trimmed,
    Replaced,
    Filled,
    SplitSource,
    SplitTarget,
    RenamedHolder,
]
_REMOVALS = [
    Migration.transform_fields(target=CopiedDeep, transform=copy.deepcopy),
    Migration.transform_fields(
        target=RebuiltValues,
        transform=lambda kw: {
            **kw,
            "value": [replace(v) if is_dataclass(v) else v for v in kw["value"]],
        },
    ),
    Migration.transform_fields(
        target=CopiedDict, transform=lambda kw: {**kw, "value": dict(kw["value"])}
    ),
    Migration.transform_fields(
        target=Wrapped,
        transform=lambda kw: {
            **kw,
            "value": kw["value"] if isinstance(kw["value"], Box) else Box(kw["value"]),
        },
    ),
    Migration.transform_fields(
        target=ClearsFirst, transform=lambda kw: {**kw, "first": None}
    ),
    Migration.transform_fields(
        target=FilteredInPlace, transform=_keep_only_high_in_place
    ),
    Migration.transform_fields(
        target=Trimmed,
        transform=lambda kw: {k: v for k, v in kw.items() if k != "value"},
    ),
    Migration.transform_fields(
        target=Replaced, transform=lambda kw: {**kw, "value": "replaced"}
    ),
    Migration.add_field(target=Filled, field="note", default="filled"),
    Migration.split_event(
        source=SplitSource,
        select=lambda kw: (SplitTarget, {"moved": kw.get("moved")}),
        targets=[SplitTarget],
    ),
    Migration.rename(old_module=_HERE, old_qualname="OldHolder", to=RenamedHolder),
]
_LOST_MEMBER = _ext(0, _HERE, "Priority", "no")
_HEALTHY_MEMBER = _ext(0, _HERE, "Priority", "low")
_HEALTHY_BASE = _ext(2, _HERE, "Base", {"a": 1, "b": 2})
_LOST_BASE = _ext(2, _HERE, "Base", {"a": 1, "b": 2, "gone": 3})
_DEAD_MODULE = "dead_value_module"
_LOST_MODEL = _ext(5, _DEAD_MODULE, "Card", {"priority": "high"})


@on(HoldsAny)
def _on_any(event: HoldsAny) -> None:
    return None


def _moved() -> ormsgpack.Ext:
    return _ext(0, _GONE, "Priority", "high")


def _event(cls: type, qualname: str | None = None, **fields: Any) -> ormsgpack.Ext:
    record = [cls.__module__, qualname or cls.__qualname__, fields]
    return ormsgpack.Ext(100, ormsgpack.packb(record, option=_option))


def _dumped(value: object) -> ormsgpack.Ext:
    """*value* as the record that a real write stores."""
    data = NamespaceAwareSerde().dumps_typed(value)[1]
    return ormsgpack.unpackb(
        data,
        ext_hook=ormsgpack.Ext,
        option=ormsgpack.OPT_NON_STR_KEYS,
    )


def _log_graph(allowed: Any = None) -> Any:
    def graph(saver: BaseCheckpointSaver[Any]) -> EventGraph:
        limit = {} if allowed is None else {"allowed_msgpack_modules": allowed}
        saver.serde = NamespaceAwareSerde(
            [*_MOVES, *_REMOVALS], events=_LOG_EVENTS, **limit
        )
        return EventGraph([_on_any], checkpointer=saver)

    return graph


def _log_store(saver: MemorySaver, tmp_path: Path, allowed: Any = None) -> Any:
    target = MigrationTarget(graph=_log_graph(allowed), baseline=tmp_path / "b.json")
    return target.build(saver).store


def _log_saver(*log: ormsgpack.Ext) -> MemorySaver:
    saver = MemorySaver()
    _log_graph()(saver).invoke(HoldsAny(), config=_cfg("t1"))
    _store_blob(saver, "t1", "events", list(log))
    return saver


def _value_thread(tmp_path: Path, value: ormsgpack.Ext, allowed: Any = None) -> Any:
    saver = _log_saver(_event(HoldsAny, moved=_moved(), value=value))
    return (
        _log_store(saver, tmp_path, allowed).plan_rewrite(thread_ids=["t1"]).threads[0]
    )


def describe_a_rewrite_that_would_remove_a_stored_class():
    def when_the_class_stays_in_the_new_bytes():
        def it_plans_a_healthy_value(tmp_path: Path):
            thread = _value_thread(tmp_path, _ext(0, _HERE, "Priority", "low"))

            assert (thread.status, thread.lost_values) == ("rewrite", ())

        def it_plans_a_declared_move(tmp_path: Path):
            thread = _value_thread(tmp_path, _ext(1, _GONE, "Ref", [1, _moved()]))

            assert thread.status == "rewrite"

        def it_plans_a_re_export_that_a_complete_allowlist_admits(
            tmp_path: Path, temp_module: types.ModuleType
        ):
            temp_module.Other = Other  # type: ignore[attr-defined]
            allowed = [*_STRICT, (temp_module.__name__, "Other"), (_HERE, "Other")]
            record = _ext(0, temp_module.__name__, "Other", 1)

            assert _value_thread(tmp_path, record, allowed).status == "rewrite"

        def it_plans_a_module_whose_second_getattr_raises_import_error(
            tmp_path: Path, temp_module: types.ModuleType
        ):
            lookups = iter([Other])

            def first_lookup_only(name: str) -> object:
                try:
                    return next(lookups)
                except StopIteration:
                    raise ImportError(name) from None

            temp_module.__getattr__ = first_lookup_only  # type: ignore[method-assign]
            record = _ext(0, temp_module.__name__, "Other", 1)

            assert _value_thread(tmp_path, record).status == "rewrite"

        def it_plans_a_module_level_alias(tmp_path: Path):
            record = _ext(0, _HERE, "AliasOfOther", 1)

            thread = _value_thread(tmp_path, record)

            assert (thread.status, thread.lost_values) == ("rewrite", ())

        def it_does_not_flag_a_framework_re_export(tmp_path: Path):
            from ag_ui.core import Context

            live = (Context.__module__, "Context")
            kwargs = {"description": "d", "value": "v"}
            assert live != ("ag_ui.core", "Context")
            record = _ext(5, "ag_ui.core", "Context", kwargs)

            thread = _value_thread(tmp_path, record)

            assert (thread.status, thread.lost_values) == ("rewrite", ())

        def it_plans_a_nested_enum_that_its_model_restores(tmp_path: Path):
            record = _dumped(Card(priority=Priority.HIGH, ref_id=1))
            saver = _log_saver(_event(RenamedHolder, "OldHolder", value=record))
            store = _log_store(saver, tmp_path, [(_HERE, "Card")])

            thread = store.plan_rewrite(thread_ids=["t1"]).threads[0]

            assert (thread.status, thread.lost_values) == ("rewrite", ())

    def when_the_new_bytes_do_not_revive():
        def it_refuses_a_re_export_only_allowlist(
            tmp_path: Path, temp_module: types.ModuleType
        ):
            temp_module.Other = Other  # type: ignore[attr-defined]
            allowed = [*_STRICT, (temp_module.__name__, "Other")]
            record = _ext(0, temp_module.__name__, "Other", 1)

            thread = _value_thread(tmp_path, record, allowed)

            assert thread.status == "refused"
            assert f"{_HERE}:Other" in thread.reason

    def when_the_class_leaves_the_new_bytes():
        def it_refuses_a_model_whose_enum_the_allowlist_omits(tmp_path: Path):
            record = _dumped(AnyCard(priority=Priority.HIGH))
            saver = _log_saver(_event(RenamedHolder, "OldHolder", value=record))
            store = _log_store(saver, tmp_path, [(_HERE, "AnyCard")])

            thread = store.plan_rewrite(thread_ids=["t1"]).threads[0]

            assert thread.status == "refused"
            assert f"{_HERE}:Priority" in thread.reason

        def it_refuses_a_deleted_member(tmp_path: Path):
            assert _value_thread(tmp_path, _LOST_MEMBER).status == "refused"

        def it_refuses_a_dataclass_that_lost_a_field(tmp_path: Path):
            record = _ext(2, _HERE, "Base", {"a": 1, "b": 2, "gone": 3})

            assert _value_thread(tmp_path, record).status == "refused"

        def it_refuses_an_old_unrecorded_value(tmp_path: Path):
            record = _ext(0, _DEAD_MODULE, "Tone", "aa")

            thread = _value_thread(tmp_path, record)

            assert thread.status == "refused"
            assert f"{_DEAD_MODULE}:Tone" in thread.reason

        def it_counts_records_per_name(tmp_path: Path, temp_module: types.ModuleType):
            temp_module.Other = Other  # type: ignore[attr-defined]
            value = [
                _ext(0, temp_module.__name__, "Other", 1),
                _ext(0, _DEAD_MODULE, "Other", 1),
            ]
            saver = _log_saver(_event(HoldsAny, moved=_moved(), value=value))

            thread = _log_store(saver, tmp_path).plan_rewrite(thread_ids=["t1"])

            assert thread.threads[0].status == "refused"
            assert f"{_DEAD_MODULE}:Other" in thread.threads[0].reason

        def it_keeps_the_bytes_and_a_corrected_allowlist_revives_the_value(
            tmp_path: Path,
        ):
            record = _dumped(AnyCard(priority=Priority.HIGH))
            saver = _log_saver(_event(RenamedHolder, "OldHolder", value=record))
            before = _memory_blob(saver, "t1", "events")

            report = _log_store(saver, tmp_path, [(_HERE, "AnyCard")]).rewrite_store(
                thread_ids=["t1"]
            )

            assert [t.status for t in report.threads] == ["refused"]
            assert _memory_blob(saver, "t1", "events") == before
            allowed = [(_HERE, "AnyCard"), (_HERE, "Priority")]
            _log_store(saver, tmp_path, allowed).rewrite_store(thread_ids=["t1"])
            tup = saver.get_tuple(_cfg("t1"))
            assert tup is not None
            stored = tup.checkpoint["channel_values"]["events"][0]
            assert stored.value == AnyCard(priority=Priority.HIGH)

    def when_an_unwritten_channel_holds_the_lost_class():
        def it_plans_the_write_and_names_the_class(tmp_path: Path):
            saver = MemorySaver()
            reducers = [_reducer(Ref, Priority), _note_reducer()]
            saver.serde = EventTopology([_choose], reducers=reducers).serde(_MOVES)
            EventGraph([_choose], reducers=reducers, checkpointer=saver).invoke(
                Started(), config=_cfg("t1")
            )
            _store_blob(saver, "t1", "ref", [_moved()])
            _store_blob(saver, "t1", "note", [_ext(0, _DEAD_MODULE, "Tone", "aa")])

            def graph(store_saver: BaseCheckpointSaver[Any]) -> EventGraph:
                store_saver.serde = EventTopology([_choose], reducers=reducers).serde(
                    _MOVES
                )
                return EventGraph(
                    [_choose], reducers=reducers, checkpointer=store_saver
                )

            target = MigrationTarget(graph=graph, baseline=tmp_path / "b.json")
            thread = (
                target.build(saver).store.plan_rewrite(thread_ids=["t1"]).threads[0]
            )

            assert thread.status == "rewrite"
            assert thread.lost_values == (f"{_DEAD_MODULE}:Tone",)

    def when_a_lost_value_reads_as_none_beside_a_none_channel():
        def it_refuses_the_write(tmp_path: Path):
            reducers = [_reducer(Ref, Priority), _note_reducer()]

            def graph(store_saver: BaseCheckpointSaver[Any]) -> EventGraph:
                store_saver.serde = EventTopology([_choose], reducers=reducers).serde(
                    _MOVES
                )
                return EventGraph(
                    [_choose], reducers=reducers, checkpointer=store_saver
                )

            saver = MemorySaver()
            graph(saver).invoke(Started(), config=_cfg("t1"))
            _store_blob(saver, "t1", "ref", _ext(0, _GONE, "Priority", "no"))
            _store_blob(saver, "t1", "note", None)
            target = MigrationTarget(graph=graph, baseline=tmp_path / "b.json")

            report = target.build(saver).store.plan_rewrite(thread_ids=["t1"])

            assert report.threads[0].status == "refused"

    def when_a_pending_interrupt_holds_the_value():
        def _paused(tmp_path: Path, *held: ormsgpack.Ext) -> Any:
            saver = MemorySaver()
            saver.serde = NamespaceAwareSerde(_MOVES, events=[Started, AskPriority])
            _ask.priority = Priority.HIGH  # type: ignore[attr-defined]
            EventGraph([_ask], checkpointer=saver).invoke(Started(), config=_cfg("t1"))
            tup = saver.get_tuple(_cfg("t1"))
            assert tup is not None
            writes = saver.writes[
                ("t1", "", tup.config["configurable"]["checkpoint_id"])
            ]
            blob = ("msgpack", ormsgpack.packb([_moved(), *held], option=_option))
            for key, (task, channel, _blob, path) in list(writes.items()):
                if channel == "__interrupt__":
                    writes[key] = (task, channel, blob, path)
            target = MigrationTarget(graph=_ask_graph, baseline=tmp_path / "b.json")
            return target.build(saver).store.plan_rewrite(thread_ids=["t1"]).threads[0]

        def it_refuses_a_deleted_member(tmp_path: Path):
            thread = _paused(tmp_path, _LOST_MEMBER)

            assert thread.status == "refused"
            assert "the pending __interrupt__ write" in thread.reason

        def it_refuses_a_deleted_member_on_a_one_blob_saver(tmp_path: Path):
            with SqliteSaver.from_conn_string(":memory:") as saver:
                saver.serde = NamespaceAwareSerde(_MOVES, events=[Started, AskPriority])
                _ask.priority = Priority.HIGH  # type: ignore[attr-defined]
                EventGraph([_ask], checkpointer=saver).invoke(
                    Started(), config=_cfg("t1")
                )
                blob = ormsgpack.packb([_moved(), _LOST_MEMBER], option=_option)
                saver.conn.execute(
                    "UPDATE writes SET type = 'msgpack', value = ? "
                    "WHERE channel = '__interrupt__'",
                    (blob,),
                )
                target = MigrationTarget(graph=_ask_graph, baseline=tmp_path / "b.json")

                report = target.build(saver).store.plan_rewrite(thread_ids=["t1"])

            assert report.threads[0].status == "refused"
            assert "the pending __interrupt__ write" in report.threads[0].reason

        def it_plans_a_healthy_value(tmp_path: Path):
            thread = _paused(tmp_path, _ext(0, _HERE, "Priority", "low"))

            assert thread.status == "rewrite"

    def when_a_declared_migration_removes_the_value():
        @pytest.mark.parametrize(
            ("log", "drop"),
            [
                pytest.param(
                    [
                        lambda: _event(HoldsAny, moved=_moved()),
                        lambda: _event(DroppedHolder, value=_LOST_MEMBER),
                    ],
                    [DroppedHolder],
                    id="a damaged value held only by a dropped event",
                ),
                pytest.param(
                    [
                        lambda: _event(HoldsAny, moved=_moved()),
                        lambda: _event(
                            DroppedHolder,
                            value=[_event(DroppedHolder, value=_LOST_MEMBER)],
                        ),
                    ],
                    [DroppedHolder],
                    id="a damaged value in a nested dropped event",
                ),
                *(
                    pytest.param(
                        [lambda c=c: _event(c, moved=_moved(), value=_HEALTHY_MEMBER)],
                        [],
                        id=f"a healthy field through {c.__name__}",
                    )
                    for c in (Trimmed, Replaced, SplitSource, Filled)
                ),
                pytest.param(
                    [
                        lambda: _event(
                            RenamedHolder,
                            "OldHolder",
                            moved=_moved(),
                            value=_HEALTHY_MEMBER,
                        )
                    ],
                    [],
                    id="a healthy field through a rename",
                ),
                pytest.param(
                    [
                        lambda: _event(
                            FilteredInPlace,
                            moved=_moved(),
                            value=[_HEALTHY_MEMBER, _moved()],
                        )
                    ],
                    [],
                    id="a healthy value that a transform removes in place",
                ),
                pytest.param(
                    [lambda: _event(CopiedDeep, moved=_moved(), value=[_HEALTHY_BASE])],
                    [],
                    id="a deep copy of a healthy value",
                ),
                pytest.param(
                    [lambda: _event(Wrapped, moved=_moved(), value=_HEALTHY_BASE)],
                    [],
                    id="a wrapper around a healthy value",
                ),
                pytest.param(
                    [
                        lambda: _event(
                            ClearsFirst,
                            moved=_moved(),
                            first=_HEALTHY_MEMBER,
                            second=1,
                        )
                    ],
                    [],
                    id="a transform that clears a healthy value",
                ),
                pytest.param(
                    [
                        lambda: _event(
                            ClearsFirst,
                            moved=_moved(),
                            first=_HEALTHY_MEMBER,
                            second=_HEALTHY_MEMBER,
                        )
                    ],
                    [],
                    id="a transform that clears one of two equal members",
                ),
            ],
        )
        def it_plans_the_removal(tmp_path: Path, log: list[Any], drop: list[type]):
            saver = _log_saver(*(entry() for entry in log))

            thread = (
                _log_store(saver, tmp_path)
                .plan_rewrite(thread_ids=["t1"], drop=drop)
                .threads[0]
            )

            assert thread.status == "rewrite"

        @pytest.mark.parametrize(
            ("log", "drop", "name"),
            [
                *(
                    pytest.param(
                        [lambda c=c: _event(c, moved=_moved(), value=_LOST_MEMBER)],
                        [],
                        "Priority",
                        id=f"a damaged field through {c.__name__}",
                    )
                    for c in (Trimmed, Replaced, SplitSource)
                ),
                pytest.param(
                    [
                        lambda: _event(
                            RenamedHolder,
                            "OldHolder",
                            moved=_moved(),
                            value=_LOST_MEMBER,
                        )
                    ],
                    [],
                    "Priority",
                    id="a damaged field that the migration keeps",
                ),
                pytest.param(
                    [
                        lambda: _event(HoldsAny, moved=_moved(), value=_LOST_MEMBER),
                        lambda: _event(DroppedHolder, value=_HEALTHY_MEMBER),
                    ],
                    [DroppedHolder],
                    "Priority",
                    id="a healthy drop beside a loss of the same class",
                ),
                pytest.param(
                    [
                        lambda: _event(HoldsAny, moved=_moved(), value=_LOST_MEMBER),
                        lambda: _event(
                            DroppedHolder,
                            value=[_event(DroppedHolder, value=_LOST_MEMBER)],
                        ),
                    ],
                    [DroppedHolder],
                    "Priority",
                    id="a nested drop beside a loss of the same class",
                ),
                pytest.param(
                    [
                        lambda: _event(
                            FilteredInPlace,
                            moved=_moved(),
                            value=[_HEALTHY_MEMBER, _LOST_MEMBER],
                        )
                    ],
                    [],
                    "Priority",
                    id="a loss that an in-place transform keeps",
                ),
                pytest.param(
                    [lambda: _event(CopiedDeep, moved=_moved(), value=[_LOST_MEMBER])],
                    [],
                    "Priority",
                    id="a deep copy of a lost enum",
                ),
                *(
                    pytest.param(
                        [
                            lambda c=c: _event(
                                c, moved=_moved(), value=[_HEALTHY_BASE, _LOST_BASE]
                            )
                        ],
                        [],
                        "Base",
                        id=f"a damaged dataclass through {c.__name__}",
                    )
                    for c in (CopiedDeep, RebuiltValues, Wrapped)
                ),
                pytest.param(
                    [lambda: _event(CopiedDict, moved=_moved(), value=_LOST_MODEL)],
                    [],
                    "Card",
                    id="a dict copy of a lost model",
                ),
            ],
        )
        def it_refuses_the_loss(
            tmp_path: Path, log: list[Any], drop: list[type], name: str
        ):
            saver = _log_saver(*(entry() for entry in log))

            thread = (
                _log_store(saver, tmp_path)
                .plan_rewrite(thread_ids=["t1"], drop=drop)
                .threads[0]
            )

            assert thread.status == "refused"
            assert f":{name}" in thread.reason

    def when_the_thread_has_no_migration():
        def it_stays_unchanged(tmp_path: Path):
            saver = _log_saver(_event(HoldsAny, value=_LOST_MEMBER))

            thread = _log_store(saver, tmp_path).plan_rewrite(thread_ids=["t1"])

            assert thread.threads[0].status == "unchanged"
            assert thread.threads[0].lost_values == (f"{_HERE}:Priority",)


class Painted(IntegrationEvent):
    color: Any = None
    note: str = ""


@on(Started)
def _paint(event: Started) -> Painted:
    return Painted(color=sys.modules[_PAINT_MODULE].Color.BLUE, note="keep")


_PAINT_MODULE = "paint_value_module"
_latest_color = ScalarReducer[Painted, Any](
    name="latest_color", event_type=Painted, fn=lambda event: [event.color]
)


def _install_color(monkeypatch: pytest.MonkeyPatch, *, blue: bool) -> type:
    module = types.ModuleType(_PAINT_MODULE)
    members = {"RED": "red", **({"BLUE": "blue"} if blue else {})}
    color = enum.Enum("Color", members, module=_PAINT_MODULE)  # type: ignore[misc]
    module.Color = color  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, _PAINT_MODULE, module)
    return color


def _paint_graph(
    saver: BaseCheckpointSaver[Any], color: type, migrations: list[Migration]
) -> EventGraph:
    saver.serde = NamespaceAwareSerde(
        migrations, events=[Started, Painted], values=[color]
    )
    return EventGraph([_paint], reducers=[_latest_color], checkpointer=saver)


def _replay_latest_color(saver: BaseCheckpointSaver[Any]) -> None:
    tup = saver.get_tuple(_cfg("t1"))
    assert tup is not None
    checkpoint = tup.checkpoint
    rebuilt = replay_reducer(_latest_color, checkpoint["channel_values"]["events"])
    old = checkpoint["channel_versions"]["latest_color"]
    new = saver.get_next_version(old, None)
    checkpoint["channel_values"]["latest_color"] = rebuilt
    checkpoint["channel_versions"]["latest_color"] = new
    for seen in checkpoint["versions_seen"].values():
        if seen.get("latest_color") == old:
            seen["latest_color"] = new
    saver.put(
        tup.parent_config or _cfg("t1"),
        checkpoint,
        tup.metadata,
        {"latest_color": new},
    )


def describe_removing_a_damaged_value_on_purpose():
    def when_the_member_returns_for_one_release():
        @pytest.mark.parametrize("saver_kind", ["memory", "sqlite"])
        def it_stores_the_replacement_and_the_next_release_reads_it(
            monkeypatch: pytest.MonkeyPatch, saver_kind: str
        ):
            with (
                SqliteSaver.from_conn_string(":memory:")
                if saver_kind == "sqlite"
                else contextlib.nullcontext(MemorySaver())
            ) as saver:
                color = _install_color(monkeypatch, blue=True)
                _paint_graph(saver, color, []).invoke(Started(), config=_cfg("t1"))
                color = _install_color(monkeypatch, blue=True)

                def clear_blue(fields: dict[str, Any]) -> dict[str, Any]:
                    held = fields.get("color")
                    return {**fields, "color": None if held is color.BLUE else held}  # type: ignore[attr-defined]

                clear = Migration.transform_fields(target=Painted, transform=clear_blue)
                store = _paint_graph(saver, color, [clear]).event_store
                written = store.rewrite_store(thread_ids=["t1"]).threads[0]
                _replay_latest_color(saver)

                next_color = _install_color(monkeypatch, blue=False)
                next_store = _paint_graph(saver, next_color, []).event_store
                planned = next_store.plan_rewrite(thread_ids=["t1"]).threads[0]
                tup = saver.get_tuple(_cfg("t1"))

            assert tup is not None
            values = tup.checkpoint["channel_values"]
            painted = next(e for e in values["events"] if isinstance(e, Painted))
            assert written.status == "rewrite"
            assert (planned.status, planned.lost_values) == ("unchanged", ())
            assert (painted.color, painted.note) == (None, "keep")
            assert values["latest_color"] == [None]


type _ForwardOnly = "Later" | None  # noqa: F821
type _Dense0 = list[_Dense1] | list[_Dense2] | list[_Dense3] | list[_Dense4]
type _Dense1 = list[_Dense0] | list[_Dense2] | list[_Dense3] | list[_Dense4]
type _Dense2 = list[_Dense0] | list[_Dense1] | list[_Dense3] | list[_Dense4]
type _Dense3 = list[_Dense0] | list[_Dense1] | list[_Dense2] | list[_Dense4]
type _Dense4 = list[_Dense0] | list[_Dense1] | list[_Dense2] | list[_Dense3]


class HoldsOpaqueAlias(IntegrationEvent):
    op: _ForwardOnly = None


class HoldsDenseAlias(IntegrationEvent):
    tree: _Dense0 | None = None


def describe_a_type_alias_in_a_field():
    def when_its_value_cannot_be_evaluated():
        def it_builds_the_serde():
            serde = NamespaceAwareSerde(events=[HoldsOpaqueAlias])

            assert serde.value_identities() == frozenset()

    def when_a_recursive_group_is_dense():
        def it_visits_each_alias_once(monkeypatch: pytest.MonkeyPatch):
            calls = 0
            real = annotationlib.call_evaluate_function

            def counted(*args: Any, **kwargs: Any) -> Any:
                nonlocal calls
                calls += 1
                return real(*args, **kwargs)

            monkeypatch.setattr(annotationlib, "call_evaluate_function", counted)

            NamespaceAwareSerde(events=[HoldsDenseAlias])

            assert calls <= 5
