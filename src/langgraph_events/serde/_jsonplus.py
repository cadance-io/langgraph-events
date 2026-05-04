"""``NamespaceAwareSerde`` — qualname-keyed roundtrip for nested events.

LangGraph's ``JsonPlusSerializer`` encodes dataclass identity by
``(__module__, __name__)``. For events nested inside a ``Namespace``,
``__name__`` is leaf-only (e.g. ``"Approved"`` for ``Persona.Approve.Approved``)
and therefore collides across namespaces. We override the dataclass branch
to encode by ``(__module__, __qualname__)`` and revive via attribute walk.

We depend on a few private helpers from
``langgraph.checkpoint.serde.jsonplus`` (``_msgpack_default``,
``_msgpack_ext_hook``, ``_option``). They have been stable for some time but
are technically private — pin a compatible LangGraph version.
"""

from __future__ import annotations

import dataclasses
import importlib
from typing import Any

import ormsgpack
from langgraph.checkpoint.serde.jsonplus import (
    JsonPlusSerializer,
    _msgpack_default,
    _msgpack_ext_hook,
    _option,
)

from langgraph_events._event import Event

# Unique among LangGraph's existing ext codes (currently 0..6).
EXT_NAMESPACE_AWARE_EVENT = 100


def _encode(obj: Any) -> bytes:
    return ormsgpack.packb(obj, default=_default, option=_option)


def _default(obj: Any) -> Any:
    if isinstance(obj, Event) and dataclasses.is_dataclass(obj):
        return ormsgpack.Ext(
            EXT_NAMESPACE_AWARE_EVENT,
            _encode(
                (
                    obj.__class__.__module__,
                    obj.__class__.__qualname__,
                    {f.name: getattr(obj, f.name) for f in dataclasses.fields(obj)},
                ),
            ),
        )
    return _msgpack_default(obj)


def _ext_hook(code: int, data: bytes) -> Any:
    if code != EXT_NAMESPACE_AWARE_EVENT:
        return _msgpack_ext_hook(code, data)
    tup = ormsgpack.unpackb(data, ext_hook=_ext_hook, option=ormsgpack.OPT_NON_STR_KEYS)
    module_name, qualname, kwargs = tup
    cls: Any = importlib.import_module(module_name)
    for part in qualname.split("."):
        cls = getattr(cls, part)
    return cls(**kwargs)


class NamespaceAwareSerde(JsonPlusSerializer):
    """JsonPlusSerializer that keys ``Event`` identity by ``__qualname__``.

    Drop-in for any LangGraph checkpointer that accepts ``serde=``::

        MemorySaver(serde=NamespaceAwareSerde())

    Non-event payloads are encoded exactly as the default
    ``JsonPlusSerializer`` would — the override applies only to ``Event``
    subclasses.
    """

    def dumps_typed(self, obj: Any) -> tuple[str, bytes]:
        if obj is None or isinstance(obj, (bytes, bytearray)):
            return super().dumps_typed(obj)
        try:
            return "msgpack", _encode(obj)
        except ormsgpack.MsgpackEncodeError:
            return super().dumps_typed(obj)

    def loads_typed(self, data: tuple[str, bytes]) -> Any:
        type_, data_ = data
        if type_ == "msgpack":
            return ormsgpack.unpackb(
                data_, ext_hook=_ext_hook, option=ormsgpack.OPT_NON_STR_KEYS
            )
        return super().loads_typed(data)
