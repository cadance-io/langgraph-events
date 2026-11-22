"""The value classes a checkpoint stores by name, and the read of a moved one.

LangGraph stores an enum, a dataclass or a pydantic model under
``(module, __name__)``, and it decodes nested values with its own hook. So
this module does not decode a value. It relabels the stored names in the
bytes, at every depth, and then hands the bytes to LangGraph. LangGraph
keeps the allowlist and every constructor.
"""

import dataclasses
import enum
import importlib
import sys
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, NamedTuple, cast

import ormsgpack
from langgraph.checkpoint.serde.jsonplus import (
    EXT_CONSTRUCTOR_KW_ARGS,
    EXT_CONSTRUCTOR_POS_ARGS,
    EXT_CONSTRUCTOR_SINGLE_ARG,
    EXT_PYDANTIC_V1,
    EXT_PYDANTIC_V2,
    _option,
)

from langgraph_events._event import Event, Namespace
from langgraph_events._identity import ValueIdentity, ValueKind
from langgraph_events.serde.migrations._core import (
    _VALUE_SOURCES_ATTR,
    Migration,
    MoveValue,
    _annotation_types,
    _field_hints,
    _is_pydantic_model,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping

_NAMED_RECORD_CODES = frozenset(
    {
        EXT_CONSTRUCTOR_SINGLE_ARG,
        EXT_CONSTRUCTOR_POS_ARGS,
        EXT_CONSTRUCTOR_KW_ARGS,
        EXT_PYDANTIC_V1,
        EXT_PYDANTIC_V2,
    }
)
"""The LangGraph ext codes whose payload starts ``[module, name, ...]``."""

_FRAMEWORK_ROOTS = frozenset(
    {"langchain_core", "langgraph", "langgraph_events", "pydantic"}
)
"""Packages whose classes their own releases keep readable."""


def value_kind(tp: object) -> ValueKind | None:
    """The kind of the value class *tp*, or ``None`` for any other type.

    An event, a namespace, a standard-library class and a framework class
    are not value classes of the client.
    """
    if not isinstance(tp, type) or issubclass(tp, (Event, Namespace)):
        return None
    root = tp.__module__.partition(".")[0]
    if root in sys.stdlib_module_names or root in _FRAMEWORK_ROOTS:
        return None
    if issubclass(tp, enum.Enum):
        return "enum"
    if _is_pydantic_model(tp):
        return "model"
    return "dataclass" if dataclasses.is_dataclass(tp) else None


def _value_hints(cls: type) -> Iterable[object]:
    """The annotation of each stored field of the value class *cls*."""
    if _is_pydantic_model(cls):
        model_fields: Mapping[str, Any] = getattr(cls, "model_fields", {})
        return [info.annotation for info in model_fields.values()]
    return _field_hints(cls).values()


@dataclass(frozen=True)
class ValueScope:
    """Every value class whose identity a checkpoint can store.

    The walk starts at the event fields, the channel types and the explicit
    *roots*. Below a dataclass, each enum, dataclass or model is its own
    record. Below a pydantic model only an enum is, because a model dump
    keeps an enum member and flattens the rest to a dict.
    """

    classes: Mapping[ValueIdentity, type]

    @classmethod
    def reach(
        cls,
        events: Iterable[type],
        channel_types: Mapping[str, object],
        roots: Iterable[type] = (),
    ) -> ValueScope:
        """Raises ``ValueError`` for a reached class that is not bound at
        module level under its ``__name__``. The message names the field or
        the channel."""
        classes: dict[ValueIdentity, type] = {}
        seen: set[tuple[object, bool]] = set()

        def visit(tp: object, where: str, *, dumped: bool) -> None:
            for inner in _annotation_types(tp):
                kind = value_kind(inner)
                if kind is None or (inner, dumped) in seen:
                    continue
                seen.add((inner, dumped))
                value = cast("type", inner)
                if not dumped or kind == "enum":
                    classes[_bound(value, kind, where)] = value
                if kind != "enum":
                    for hint in _value_hints(value):
                        visit(hint, where, dumped=dumped or kind == "model")

        for event in events:
            for field, tp in _field_hints(event).items():
                visit(tp, f"{event.__qualname__}.{field}", dumped=False)
        for channel, tp in channel_types.items():
            visit(tp, f"channel {channel!r}", dumped=False)
        for root in roots:
            visit(root, f"values= entry {root.__qualname__}", dumped=False)
        return cls(classes)

    @property
    def kinds(self) -> dict[ValueIdentity, ValueKind]:
        """The kind of each reached value class."""
        return {
            identity: kind
            for identity, value in self.classes.items()
            if (kind := value_kind(value)) is not None
        }

    def decorated_moves(self) -> tuple[Migration, ...]:
        """One migration per reached class with ``@migrate_value_from``."""
        return tuple(
            Migration(
                tuple(
                    MoveValue(module, name, value)
                    for module, name in value.__dict__[_VALUE_SOURCES_ATTR]
                ),
                name=identity.label,
            )
            for identity, value in self.classes.items()
            if _VALUE_SOURCES_ATTR in value.__dict__
        )

    def unreached_siblings(self, modules: Iterable[str]) -> list[type]:
        """Each ``@migrate_value_from`` class in *modules* or in a module
        of this scope that the walk did not reach. Its move does nothing."""
        names = {*modules, *(identity.module for identity in self.classes)}
        reached = {id(value) for value in self.classes.values()}
        found: dict[int, type] = {}
        for name in names:
            for value in vars(sys.modules.get(name, object)).values():
                if (
                    isinstance(value, type)
                    and value.__module__ == name
                    and _VALUE_SOURCES_ATTR in value.__dict__
                    and id(value) not in reached
                ):
                    found[id(value)] = value
        return list(found.values())


def _bound(cls: type, kind: ValueKind, where: str) -> ValueIdentity:
    try:
        return ValueIdentity.of(cls)
    except ValueError as error:
        label = "pydantic model" if kind == "model" else kind
        raise ValueError(
            f"{where} is annotated with {label} {cls.__qualname__}. {error} The "
            f"LangGraph serializer stores a value by __name__ and revives it "
            f"with getattr on the module. On a miss it returns raw data with "
            f"no error."
        ) from error


def live_value_class(module: str, name: str) -> type | None:
    """The class that *module* itself defines under *name*, or ``None``.

    A class that *module* only imports does not count, because the bytes
    that name *module* then depend on an import that nothing records.
    """
    try:
        found = getattr(importlib.import_module(module), name, None)
    except Exception:  # an import failure of any type means "not live"
        return None
    if isinstance(found, type) and (found.__module__, found.__name__) == (module, name):
        return found
    return None


class ValueRead(NamedTuple):
    """One recorded value identity that a read met in the stored bytes.

    ``revived`` is ``False`` when LangGraph gave back anything other than
    an instance of exactly the recorded class.
    """

    stored: tuple[str, str]
    resolved: tuple[str, str]
    revived: bool

    @property
    def touched(self) -> bool:
        """A rewrite stores this value under another identity."""
        return self.stored != self.resolved


class _Met(NamedTuple):
    """A recorded value record, as the bytes LangGraph can decode alone."""

    stored: tuple[str, str]
    resolved: tuple[str, str]
    code: int
    payload: bytes


@dataclass(frozen=True)
class ValueReader:
    """Relabels each moved value in the stored bytes, then calls LangGraph."""

    moves: Mapping[tuple[str, str], tuple[str, str]]
    recorded: Mapping[tuple[str, str], type]
    needles: tuple[bytes, ...]

    @classmethod
    def build(
        cls, scope: ValueScope, move_table: Mapping[tuple[str, str], type]
    ) -> ValueReader:
        """The reader of *scope* and of the moves in *move_table*."""
        return cls(
            moves={
                source: (target.__module__, target.__name__)
                for source, target in move_table.items()
            },
            recorded={
                **{(i.module, i.name): c for i, c in scope.classes.items()},
                **{(t.__module__, t.__name__): t for t in move_table.values()},
            },
            needles=tuple({module.encode() for module, _name in move_table}),
        )

    def hook(
        self,
        fallback: Callable[[int, bytes], Any],
        reads: list[Any] | None,
    ) -> Callable[[int, bytes], Any]:
        """The read of one value record. With no move and no census to
        take, this is *fallback* itself, so a read pays nothing."""
        if not self.moves and (reads is None or not self.recorded):
            return fallback

        def read(code: int, data: bytes) -> object:
            if reads is None and not any(needle in data for needle in self.needles):
                return fallback(code, data)
            met: list[_Met] = []
            relabelled = self._relabel(code, data, met)
            if reads is not None:
                # Each record is decoded alone and judged on its exact type,
                # so a subclass instance cannot stand in for a lost record.
                reads.extend(
                    ValueRead(
                        m.stored,
                        m.resolved,
                        type(fallback(m.code, m.payload)) is self.recorded[m.resolved],
                    )
                    for m in met
                )
            return fallback(code, relabelled)

        return read

    def _relabel(self, code: int, data: bytes, met: list[_Met]) -> bytes:
        """*data* with each moved name replaced. It is the same object when
        no name inside it moved, so an unmoved record is never repacked."""
        changed = False

        def nested(inner_code: int, payload: bytes) -> ormsgpack.Ext:
            nonlocal changed
            relabelled = self._relabel(inner_code, payload, met)
            changed = changed or relabelled is not payload
            return ormsgpack.Ext(inner_code, relabelled)

        try:
            record = ormsgpack.unpackb(
                data, ext_hook=nested, option=ormsgpack.OPT_NON_STR_KEYS
            )
        except ormsgpack.MsgpackDecodeError:
            return data
        identity = _named_record(code, record)
        resolved = None if identity is None else self.moves.get(identity, identity)
        if identity is not None and resolved != identity:
            record[0:2] = resolved
            changed = True
        payload = ormsgpack.packb(record, option=_option) if changed else data
        if identity is not None and resolved is not None and resolved in self.recorded:
            met.append(_Met(identity, resolved, code, payload))
        return payload


def _named_record(code: int, record: object) -> tuple[str, str] | None:
    """The ``(module, name)`` that the LangGraph record *record* stores."""
    if (
        code in _NAMED_RECORD_CODES
        and isinstance(record, list)
        and len(record) >= 3
        and isinstance(record[0], str)
        and isinstance(record[1], str)
    ):
        return record[0], record[1]
    return None


def stored_value_identities(data: bytes) -> frozenset[tuple[str, str]]:
    """Every ``(module, name)`` of a LangGraph value record in *data*.

    Class-blind: it imports nothing and constructs nothing.
    """
    found: set[tuple[str, str]] = set()

    def hook(code: int, payload: bytes) -> None:
        try:
            record = ormsgpack.unpackb(
                payload, ext_hook=hook, option=ormsgpack.OPT_NON_STR_KEYS
            )
        except ormsgpack.MsgpackDecodeError:
            return
        identity = _named_record(code, record)
        if identity is not None:
            found.add(identity)

    ormsgpack.unpackb(data, ext_hook=hook, option=ormsgpack.OPT_NON_STR_KEYS)
    return frozenset(found)
