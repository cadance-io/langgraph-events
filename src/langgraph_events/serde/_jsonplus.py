"""``NamespaceAwareSerde`` — qualname-keyed roundtrip for nested events.

LangGraph's ``JsonPlusSerializer`` encodes dataclass identity by
``(__module__, __name__)``. For events nested inside a ``Namespace``,
``__name__`` is leaf-only (e.g. ``"Approved"`` for ``Persona.Approve.Approved``)
and therefore collides across namespaces. We override the dataclass branch
to encode by ``(__module__, __qualname__)`` and revive via attribute walk.

We depend on a few private helpers from
``langgraph.checkpoint.serde.jsonplus`` (``_msgpack_default``, ``_option``).
They have been stable for some time but are technically private — pin a
compatible LangGraph version.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Any

import ormsgpack
from langgraph.types import Interrupt

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from pathlib import Path

    from langgraph_events.serde.migrations._core import AddField, Migration
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

try:
    from langgraph.checkpoint.serde.jsonplus import (
        _msgpack_default,
        _option,
    )
except ImportError as exc:  # pragma: no cover - smoke fence on LangGraph drift
    raise ImportError(
        "langgraph_events.serde.NamespaceAwareSerde depends on private "
        "helpers from langgraph.checkpoint.serde.jsonplus "
        "(_msgpack_default, _option). They appear to have moved or been "
        "renamed. Pin a compatible LangGraph version, or open an issue "
        "against langgraph-events."
    ) from exc

# Smoke fence on the per-instance ``_unpack_ext_hook`` attribute: ``loads_typed``
# threads it through ``_make_ext_hook`` as the fallback for ext codes we don't
# own (#68). Failing fast at import time gives a single actionable error
# instead of an ``AttributeError`` deep in the first checkpoint load if a
# future LangGraph rev renames or hides the attribute.
if not hasattr(JsonPlusSerializer(), "_unpack_ext_hook"):  # pragma: no cover
    raise ImportError(
        "langgraph_events.serde.NamespaceAwareSerde depends on the "
        "per-instance ``JsonPlusSerializer._unpack_ext_hook`` attribute. "
        "It appears to have been renamed or removed. Pin a compatible "
        "LangGraph version (langgraph-checkpoint>=4.0.3 is supported), "
        "or open an issue against langgraph-events."
    )

# Unique among LangGraph's existing ext codes (currently 0..6).
EXT_NAMESPACE_AWARE_EVENT = 100
# Dedicated code so we can recurse via our own ``_default`` when re-encoding
# the wrapped value — LangGraph's generic dataclass path uses ``_msgpack_enc``
# which is hardcoded to ``default=_msgpack_default`` and would bypass us.
EXT_INTERRUPT = 101

# Imported AFTER the EXT constants and ``_option`` are bound: ``_core``
# pulls in the ``serde.migrations`` package whose ``__init__`` re-exports
# ``testing.synthesize_legacy_payload``, which imports the two names above
# back from this module. Defining them first lets that re-entry resolve
# against a partially-initialized ``_jsonplus`` without a circular import.
from langgraph_events.serde.migrations._core import (  # noqa: E402
    _apply_identity_migrations,
    _collect_decorated_migrations,
    _flatten_and_validate,
    _resolve_identity,
)
from langgraph_events.serde.migrations.detect import (  # noqa: E402
    MigrationCoverageError,
    _load_baseline,
)


def _make_default(
    legacy_write: bool,
    oldest_historic: dict[tuple[str, str], tuple[str, str]],
) -> Callable[[Any], Any]:
    """Build the ``default=`` hook ormsgpack uses for unknown types.

    Closure so the per-serde ``legacy_write`` flag and ``oldest_historic``
    map thread down into recursive sub-encodes (Interrupt-wrapped values,
    etc.) without relying on module-level state.

    ``oldest_historic`` is built at construction from the serde's
    ``namespaces=`` scope. Encoding under an oldest historic identity is
    gated on the class being in this map — out-of-scope decorated classes
    fall through to their current qualname so bytes never reference a
    historic name the serde's own read path can't migrate back.
    """

    def _default(obj: Any) -> Any:
        if isinstance(obj, Event) and dataclasses.is_dataclass(obj):
            cls = obj.__class__
            module, qualname = cls.__module__, cls.__qualname__
            if legacy_write:
                # Consult the serde's scoped map (not ``__lge_migrate_from__``
                # on the class) so encode/decode scope stays symmetric: bytes
                # are only relabelled under a historic identity the read-side
                # rename table knows how to migrate. Subclasses inherit
                # neither the parent's history nor its scope mapping —
                # ``oldest_historic`` only records identities the namespace
                # walk reached directly via ``__dict__``.
                if (oldest := oldest_historic.get((module, qualname))) is not None:
                    module, qualname = oldest
            return ormsgpack.Ext(
                EXT_NAMESPACE_AWARE_EVENT,
                ormsgpack.packb(
                    (
                        module,
                        qualname,
                        {f.name: getattr(obj, f.name) for f in dataclasses.fields(obj)},
                    ),
                    default=_default,
                    option=_option,
                ),
            )
        if isinstance(obj, Interrupt):
            # LangGraph wraps every interrupted value in this dataclass
            # before checkpointing. Re-encoding through our own ``default``
            # (rather than letting upstream's dataclass branch handle it)
            # is what keeps a nested namespaced ``Interrupted`` subclass
            # inside ``obj.value`` reachable through this hook and revivable
            # under EXT_NAMESPACE_AWARE_EVENT.
            #
            # Tracks (value, id) explicitly rather than walking
            # ``dataclasses.fields(obj)`` — Interrupt has a custom
            # ``__init__`` that doesn't accept arbitrary kwargs, so a
            # generic walk would not round-trip cleanly anyway.
            # ``it_matches_the_schema_we_encode`` in tests/test_serde.py
            # guards against silent field drift.
            return ormsgpack.Ext(
                EXT_INTERRUPT,
                ormsgpack.packb((obj.value, obj.id), default=_default, option=_option),
            )
        return _msgpack_default(obj)

    return _default


def _make_ext_hook(
    errors: list[str],
    fallback: Callable[[int, bytes], Any],
    rename_table: dict[tuple[str, str], tuple[str, str]],
    addfield_table: dict[tuple[str, str], tuple[AddField, ...]],
) -> Callable[[int, bytes], Any]:
    """Build an ext-hook that records revival errors into *errors*.

    ormsgpack swallows the original exception from an ext-hook and re-raises
    a generic ``ValueError("ext_hook failed")``. The error list lets
    ``loads_typed`` reconstruct an actionable message after the fact.

    *fallback* handles ext codes we don't own (everything emitted by
    upstream's ``_msgpack_default`` — Pydantic models, plain dataclasses,
    ``UUID``s, ``datetime``s, etc.). Callers thread the parent's
    *per-instance* ``_unpack_ext_hook`` here rather than the module-level
    alias from ``langgraph.checkpoint.serde.jsonplus``: in
    ``langgraph-checkpoint>=4.0.3`` that alias is hardcoded strict
    (``allowed_modules=None``) and silently demotes non-event payloads to
    plain ``dict`` regardless of ``LANGGRAPH_STRICT_MSGPACK`` or the
    constructor's ``allowed_msgpack_modules`` argument (#68).
    """

    def _ext_hook(code: int, data: bytes) -> Any:
        if code == EXT_INTERRUPT:
            # Inner unpack uses our hook so a nested EXT_NAMESPACE_AWARE_EVENT
            # inside ``value`` resolves back to its namespaced class.
            value, id_ = ormsgpack.unpackb(
                data, ext_hook=_ext_hook, option=ormsgpack.OPT_NON_STR_KEYS
            )
            try:
                return Interrupt(value=value, id=id_)
            except TypeError as exc:
                # Mirrors the EXT_NAMESPACE_AWARE_EVENT branch below: degrade
                # gracefully through ``loads_typed``'s ``errors`` channel if
                # ``Interrupt.__init__`` shape changes upstream (the static
                # schema guard in tests/test_serde.py catches drift at test
                # time, but this covers an unpinned-LangGraph runtime gap).
                errors.append(
                    f"Cannot revive langgraph.types.Interrupt(value=..., "
                    f"id=...): {type(exc).__name__}: {exc}. The Interrupt "
                    f"dataclass shape may have changed since the checkpoint "
                    f"was written; update NamespaceAwareSerde to track the "
                    f"new fields."
                )
                raise
        if code != EXT_NAMESPACE_AWARE_EVENT:
            return fallback(code, data)
        tup = ormsgpack.unpackb(
            data, ext_hook=_ext_hook, option=ormsgpack.OPT_NON_STR_KEYS
        )
        module_name, qualname, kwargs = tup
        # Rewrite historic identity to current and inject any AddField
        # defaults — shared with the baseline test helper so the read-side
        # migration rule lives in exactly one place.
        module_name, qualname = _apply_identity_migrations(
            module_name, qualname, kwargs, rename_table, addfield_table
        )
        try:
            return _resolve_identity(module_name, qualname)(**kwargs)
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

    Pass ``namespaces=`` to scope decorator-driven (``@migrate_from``)
    collection to the namespaces in play for this graph. Pass
    ``migrations=`` for hand-authored cross-module renames or composite
    operations; the two compose. See :mod:`langgraph_events.serde.migrations`.
    """

    def __init__(
        self,
        migrations: Sequence[Migration] = (),
        *,
        namespaces: Sequence[type] = (),
        legacy_write: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        # Decorator-driven migrations come first so the duplicate-source
        # diagnostic, when a user-passed hand-authored entry conflicts,
        # names the user's migration as the second (more actionable than
        # naming the auto-collected one).
        decorated, oldest_historic, live = _collect_decorated_migrations(namespaces)
        all_migrations = (*decorated, *migrations)
        self._rename_table, self._addfield_table = _flatten_and_validate(all_migrations)
        self._live_identities = live
        self._legacy_write = legacy_write
        self._encode_default = _make_default(legacy_write, oldest_historic)

    def revivable_identities(self) -> frozenset[tuple[str, str]]:
        """Every ``(module, qualname)`` this serde can revive — either still
        live in the namespaces it was constructed with, or covered by a
        rename migration (``@migrate_from`` decorators in ``namespaces=``
        and hand-authored ``migrations=``).

        Read-only view. AddField targets are NOT included: they key on the
        post-rename (currently-live) identity and add no new revivable
        identities.
        """
        return self._live_identities | frozenset(self._rename_table.keys())

    def assert_covers(self, baseline_path: Path) -> None:
        """Raise :class:`MigrationCoverageError` if any identity in
        *baseline_path* is neither currently live in this serde's
        ``namespaces=`` nor covered by a rename migration.

        Construct the serde the same way the runtime does — this verifies
        the production migration table covers every event the production
        cluster could hand it on the next read.
        """
        baseline = _load_baseline(baseline_path)
        missing = tuple(sorted(baseline - self.revivable_identities()))
        if missing:
            raise MigrationCoverageError(missing)

    def dumps_typed(self, obj: Any) -> tuple[str, bytes]:
        if obj is None or isinstance(obj, (bytes, bytearray)):
            return super().dumps_typed(obj)
        # ``_encode_default`` is a strict superset of upstream's
        # ``_msgpack_default``: anything upstream encodes, we encode the
        # same way. So an ``MsgpackEncodeError`` here is genuinely
        # unencodable. The old behaviour warned and called
        # ``super().dumps_typed`` — which in the default config simply
        # re-raised, and with the parent's binary-fallback kwarg enabled
        # would silently emit unsafe-binary bytes that bypass the
        # migration table. Let the encode error propagate at the source
        # so the caller widens ``_default`` or removes the payload from
        # state explicitly.
        return "msgpack", ormsgpack.packb(
            obj, default=self._encode_default, option=_option
        )

    def loads_typed(self, data: tuple[str, bytes]) -> Any:
        type_, data_ = data
        if type_ != "msgpack":
            return super().loads_typed(data)
        errors: list[str] = []
        # Route fallback through the parent's *per-instance* hook — see the
        # docstring on ``_make_ext_hook`` for the #68 backstory.
        try:
            return ormsgpack.unpackb(
                data_,
                ext_hook=_make_ext_hook(
                    errors,
                    self._unpack_ext_hook,
                    self._rename_table,
                    self._addfield_table,
                ),
                option=ormsgpack.OPT_NON_STR_KEYS,
            )
        except ValueError as exc:
            if errors:
                raise ValueError(errors[-1]) from exc
            raise
