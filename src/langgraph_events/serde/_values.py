"""The value classes a checkpoint stores by name, and the read of a moved one.

LangGraph stores an enum, a dataclass, a ``NamedTuple`` or a pydantic model
under ``(module, __name__)``, and it decodes nested values with its own hook. So
this module does not decode a value. It relabels the stored names in the
bytes, at every depth, and then hands the bytes to LangGraph. LangGraph
keeps the allowlist and every constructor.
"""

import annotationlib
import dataclasses
import enum
import sys
import typing
from collections import Counter
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
from langgraph_events._identity import ValueIdentity, live_value_class
from langgraph_events.serde.migrations._core import (
    _VALUE_SOURCES_ATTR,
    Migration,
    MoveValue,
    _annotation_types,
    _enclosing_names,
    _field_hints,
    _is_pydantic_model,
    _unreachable_migrate_from_siblings,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Collection, Iterable, Mapping

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


def _client_module(module: str) -> bool:
    """*module* is outside the standard library and the framework packages."""
    root = module.partition(".")[0]
    return root not in sys.stdlib_module_names and root not in _FRAMEWORK_ROOTS


def _is_named_tuple(tp: object) -> bool:
    return isinstance(tp, type) and issubclass(tp, tuple) and hasattr(tp, "_fields")


def _is_value_class(tp: object, first_party: Collection[str] = ()) -> bool:
    """*tp* is a class that LangGraph stores by name, for this client.

    An event, a namespace, a standard-library class and a framework class
    are not value classes of the client, unless *first_party* names the
    module of *tp*.
    """
    if not isinstance(tp, type) or issubclass(tp, (Event, Namespace)):
        return False
    if tp.__module__ not in first_party and not _client_module(tp.__module__):
        return False
    return (
        issubclass(tp, enum.Enum)
        or _is_pydantic_model(tp)
        or dataclasses.is_dataclass(tp)
        or _is_named_tuple(tp)
    )


def _is_field_container(tp: object) -> bool:
    """*tp* is a ``TypedDict`` or a ``NamedTuple``, whose fields can hold a
    value class."""
    return typing.is_typeddict(tp) or _is_named_tuple(tp)


def _value_hints(cls: type) -> Iterable[object]:
    """The annotation of each stored field of *cls*."""
    if _is_field_container(cls):
        return typing.get_type_hints(
            cls, localns=_enclosing_names(cls), format=annotationlib.Format.FORWARDREF
        ).values()
    if dataclasses.is_dataclass(cls):
        return _field_hints(cls).values()
    model_fields: Mapping[str, Any] = getattr(cls, "model_fields", {})
    return [info.annotation for info in model_fields.values()]


@dataclass(frozen=True)
class ValueScope:
    """Every value class whose identity a checkpoint can store.

    The walk starts at the event fields, the channel types and the explicit
    *roots*. Below a dataclass, each enum, dataclass or model is its own
    record, and so is a ``NamedTuple``. Below a pydantic model only an enum
    is, because a model dump keeps an enum member and flattens the rest to a
    dict or a plain tuple. A ``TypedDict`` passes the walk through and is
    not recorded.
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
        roots = tuple(roots)
        first_party = {root.__module__ for root in roots}
        classes: dict[ValueIdentity, type] = {}
        seen: set[tuple[object, bool]] = set()

        def visit(tp: object, where: str, *, dumped: bool) -> None:
            for inner in _annotation_types(tp):
                stored_by_name = _is_value_class(inner, first_party)
                if not stored_by_name and not _is_field_container(inner):
                    continue
                if (inner, dumped) in seen:
                    continue
                seen.add((inner, dumped))
                value = cast("type", inner)
                is_enum = issubclass(value, enum.Enum)
                # An enum in a model dump with no identity is not an error:
                # a read can still validate the raw value into the field.
                if stored_by_name and (
                    not dumped
                    or (
                        is_enum
                        and live_value_class(value.__module__, value.__name__) is value
                    )
                ):
                    classes[_bound(value, where)] = value
                if not is_enum:
                    below_model = dumped or _is_pydantic_model(value)
                    for hint in _value_hints(value):
                        visit(hint, where, dumped=below_model)

        for event in events:
            for field, tp in _field_hints(event).items():
                visit(tp, f"{event.__qualname__}.{field}", dumped=False)
        for channel, tp in channel_types.items():
            visit(tp, f"channel {channel!r}", dumped=False)
        for root in roots:
            visit(root, f"values= entry {root.__qualname__}", dumped=False)
        return cls(classes)

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
        return _unreachable_migrate_from_siblings(
            self.classes, marker=_VALUE_SOURCES_ATTR, modules=modules
        )


def _bound(cls: type, where: str) -> ValueIdentity:
    try:
        return ValueIdentity.of(cls)
    except ValueError as error:
        raise ValueError(
            f"{where} is annotated with {cls.__qualname__}. {error} The "
            f"LangGraph serializer stores a value by __name__ and revives it "
            f"with getattr on the module. On a miss it returns raw data with "
            f"no error."
        ) from error


class ValueRead(NamedTuple):
    """One named client value record that a read met in the stored bytes."""

    stored: tuple[str, str]
    resolved: tuple[str, str]
    lost: tuple[tuple[str, str], ...] = ()

    @property
    def touched(self) -> bool:
        """A rewrite stores this value under another identity."""
        return self.stored != self.resolved


@dataclass(frozen=True)
class ValueReader:
    """Relabels each moved value in the stored bytes, then calls LangGraph."""

    moves: Mapping[tuple[str, str], tuple[str, str]]
    recorded: frozenset[tuple[str, str]]
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
            recorded=frozenset(
                {*scope.classes, *map(ValueIdentity.of, move_table.values())}
            ),
            needles=tuple({module.encode() for module, _name in move_table}),
        )

    def hook(
        self,
        fallback: Callable[[int, bytes], Any],
        reads: list[Any] | None,
        encode: Callable[[object], tuple[str, bytes]] | None = None,
    ) -> Callable[[int, bytes], Any]:
        """The read of one value record. With no move and no read to
        record, this is *fallback* itself, so a read pays nothing.

        With *encode*, each outermost record is judged here, before any event
        migration can see its value."""
        if not self.moves and reads is None:
            return fallback

        def read(code: int, data: bytes) -> object:
            if reads is None and not any(needle in data for needle in self.needles):
                return fallback(code, data)
            met: list[ValueRead] = []
            relabelled = self._relabel(code, data, met)
            value = fallback(code, relabelled)
            if reads is not None:
                if encode is not None and met:
                    lost = self._lost(value, met, fallback, encode)
                    met[-1] = met[-1]._replace(lost=lost)
                reads.extend(met)
            return value

        return read

    def names(self, data: bytes) -> Counter[tuple[str, str]]:
        """The live name of each value record in *data*, counted per name."""
        met: list[ValueRead] = []
        self._relabel(-1, data, met)
        return Counter(read.resolved for read in met)

    def _lost(
        self,
        value: object,
        met: list[ValueRead],
        fallback: Callable[[int, bytes], Any],
        encode: Callable[[object], tuple[str, bytes]],
    ) -> tuple[tuple[str, str], ...]:
        """Each name of *met* that a write of *value* removes.

        A second read proves that the allowlist admits a name the write
        changed, for example the defining module of a re-export.
        """
        names = {_defining_name(read.resolved, value): read.resolved for read in met}
        stored = Counter(_defining_name(read.resolved, value) for read in met)
        try:
            kind, written = encode(value)
            if kind != "msgpack":
                return tuple(names.values())
            after = self.names(written)
            if after == Counter(read.resolved for read in met):
                return ()
            again = ormsgpack.unpackb(
                written, ext_hook=fallback, option=ormsgpack.OPT_NON_STR_KEYS
            )
            second = self.names(encode(again)[1])
        except ValueError, TypeError:
            return tuple(names.values())
        return (
            *(names[name] for name in (stored - after).elements()),
            *(after - second).elements(),
        )

    def _relabel(self, code: int, data: bytes, met: list[ValueRead]) -> bytes:
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
        if (
            identity is not None
            and resolved is not None
            and (resolved in self.recorded or _client_module(resolved[0]))
        ):
            met.append(ValueRead(identity, resolved))
        return payload


def _defining_name(identity: tuple[str, str], value: object) -> tuple[str, str]:
    """The name that a write of the class bound at *identity* stores.

    Only a module that is already imported counts, so a stored name never
    triggers an import. An unbound name stays as it is. When a module
    ``__getattr__`` raises, *value*, the revived record, names the class.
    """
    try:
        found = getattr(sys.modules.get(identity[0]), identity[1], None)
    except Exception:  # a module __getattr__ can raise any type
        found = type(value) if type(value).__name__ == identity[1] else None
        if found is None:
            return identity
    return (found.__module__, found.__name__) if isinstance(found, type) else identity


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
