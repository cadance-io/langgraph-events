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
import warnings
from typing import TYPE_CHECKING, Any

import ormsgpack

if TYPE_CHECKING:
    from collections.abc import Callable
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

try:
    from langgraph.checkpoint.serde.jsonplus import (
        _msgpack_default,
        _msgpack_ext_hook,
        _option,
    )
except ImportError as exc:  # pragma: no cover - smoke fence on LangGraph drift
    raise ImportError(
        "langgraph_events.serde.NamespaceAwareSerde depends on private "
        "helpers from langgraph.checkpoint.serde.jsonplus "
        "(_msgpack_default, _msgpack_ext_hook, _option). They appear to "
        "have moved or been renamed. Pin a compatible LangGraph version, "
        "or open an issue against langgraph-events."
    ) from exc

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


def _make_ext_hook(errors: list[str]) -> Callable[[int, bytes], Any]:
    """Build an ext-hook that records revival errors into *errors*.

    ormsgpack swallows the original exception from an ext-hook and re-raises
    a generic ``ValueError("ext_hook failed")``. The error list lets
    ``loads_typed`` reconstruct an actionable message after the fact.
    """

    def _ext_hook(code: int, data: bytes) -> Any:
        if code != EXT_NAMESPACE_AWARE_EVENT:
            return _msgpack_ext_hook(code, data)
        tup = ormsgpack.unpackb(
            data, ext_hook=_ext_hook, option=ormsgpack.OPT_NON_STR_KEYS
        )
        module_name, qualname, kwargs = tup
        try:
            obj: Any = importlib.import_module(module_name)
            for part in qualname.split("."):
                obj = getattr(obj, part)
            return obj(**kwargs)
        except (ImportError, AttributeError) as exc:
            errors.append(
                f"Cannot revive {module_name}.{qualname}: {type(exc).__name__}: {exc}. "
                f"The class may have been renamed or removed since the "
                f"checkpoint was written."
            )
            raise

    return _ext_hook


from langgraph_events._event import Event  # noqa: E402  (avoid circular import order)


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
        except ormsgpack.MsgpackEncodeError as exc:
            # Falling through to ``JsonPlusSerializer.dumps_typed`` re-encodes
            # under the leaf-``__name__`` scheme — exactly the collision the
            # namespace-aware serde exists to prevent. Warn so users notice
            # and either avoid the unencodable payload or extend ``_default``
            # to handle it.
            warnings.warn(
                f"NamespaceAwareSerde could not encode {type(obj).__name__!r}; "
                f"falling back to JsonPlusSerializer (leaf-name identity, "
                f"collision-prone for nested events). Reason: {exc}",
                stacklevel=2,
            )
            return super().dumps_typed(obj)

    def loads_typed(self, data: tuple[str, bytes]) -> Any:
        type_, data_ = data
        if type_ != "msgpack":
            return super().loads_typed(data)
        errors: list[str] = []
        try:
            return ormsgpack.unpackb(
                data_,
                ext_hook=_make_ext_hook(errors),
                option=ormsgpack.OPT_NON_STR_KEYS,
            )
        except ValueError as exc:
            if errors:
                raise ValueError(errors[-1]) from exc
            raise
