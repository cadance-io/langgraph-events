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

import contextlib
import dataclasses
import re
from typing import TYPE_CHECKING, Any, NamedTuple

import ormsgpack
from langgraph.types import Interrupt

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Sequence

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
    _unreachable_migrate_from_siblings,
)


class UnrevivedIdentity(NamedTuple):
    """Placeholder for an interrupt identity that could not be revived,
    produced only inside :meth:`NamespaceAwareSerde.tolerate_unresolved`.

    Carries the identity exactly as stored — ``module`` and ``qualname``
    — with none of the original fields; there is no live class to hold
    them. Not an ``Event`` subclass: a caller must never mistake this for
    a real, dispatchable event.

    Exists for ``EventGraph.threads_paused_on()``/``abandon()`` — the
    retirement tools that must keep working on a thread whose paused
    class was deleted before every thread was settled. Every other read
    path stays strict: an unrevivable identity there is a genuine bug and
    must raise, not degrade silently.
    """

    module: str
    qualname: str


_UNEXPECTED_KWARG_RE = re.compile(r"unexpected keyword argument '(\w+)'")


def _revival_remedy(qualname: str, exc: Exception) -> str:
    """The actionable second sentence of a ``Cannot revive`` message.

    *exc* is ``ImportError``/``AttributeError`` (the identity resolves to
    no live class at all) or ``TypeError`` (it resolves, but the stored
    kwargs and the class's fields disagree). The two need different
    remedies: "map onto a tombstone with @migrate_from" is right for the
    first — there's no live class yet — and wrong for the second, where
    *qualname* may already **be** the tombstone: redecorating it with the
    decorator it already carries fixes nothing.
    """
    if not isinstance(exc, TypeError):
        return (
            "The class may have been renamed or removed since the "
            "checkpoint was written. Settle the thread with "
            "abandon()/aabandon() before deleting the class, or map the "
            f"dead identity onto a tombstone class with "
            f"@migrate_from({qualname!r})."
        )
    match = _UNEXPECTED_KWARG_RE.search(str(exc))
    if match is None:
        return (
            f"{qualname}'s fields no longer match the stored payload — "
            f"the class may have gained a required field with no default "
            f"and no AddField. Align the class with the payload's shape, "
            f"or back-fill the field with a migration."
        )
    field = match.group(1)
    return (
        f"{qualname} does not declare the field {field!r} the stored "
        f"payload carries. Add {field!r} to the class, matching the "
        f"retired class's shape, or drop it from the payload with a "
        f"migration."
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
    origin_addfield_table: dict[tuple[str, str], tuple[AddField, ...]],
    scope: dict[tuple[str, str], type],
    *,
    tolerant: bool = False,
) -> Callable[[int, bytes], Any]:
    """Build an ext-hook that records revival errors into *errors*.

    *scope* is the serde's ``namespaces=`` map, consulted ahead of the
    import walk so revival lands on the classes this serde was built with
    — the read-side mirror of the encoder's ``oldest_historic`` map.

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

    *tolerant*, when ``True``, degrades an unrevivable
    ``EXT_NAMESPACE_AWARE_EVENT`` identity to an :class:`UnrevivedIdentity`
    instead of raising. Only ``NamespaceAwareSerde.tolerate_unresolved``
    sets this — every other caller keeps the strict default.
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
            module_name,
            qualname,
            kwargs,
            rename_table,
            addfield_table,
            origin_addfield_table,
        )
        try:
            return _resolve_identity(module_name, qualname, scope=scope)(**kwargs)
        except (ImportError, AttributeError, TypeError) as exc:
            # ``TypeError`` is the field-shape mismatch: the identity
            # resolves, but the stored kwargs carry a key the live class
            # has dropped, or omit a field it has gained with no AddField.
            if tolerant:
                # Retirement cleanup only (see the *tolerant* parameter
                # docstring above): the caller is a tool that exists to
                # settle exactly this thread, not a normal read — degrade
                # instead of raising, and keep no partial kwargs (there is
                # no live class to hold them).
                return UnrevivedIdentity(module=module_name, qualname=qualname)
            errors.append(
                f"Cannot revive {module_name}.{qualname}: {type(exc).__name__}: "
                f"{exc}. {_revival_remedy(qualname, exc)}"
            )
            raise

    return _ext_hook


from langgraph_events._event import Event  # noqa: E402  (avoid circular import order)
from langgraph_events._warn import warn_user  # noqa: E402


class UnreachableMigrationWarning(UserWarning):
    """A ``@migrate_from``-decorated class was not collected into a
    :class:`NamespaceAwareSerde`'s scope — its migration does nothing."""


class NamespaceAwareSerde(JsonPlusSerializer):
    """JsonPlusSerializer that keys ``Event`` identity by ``__qualname__``.

    Drop-in for any LangGraph checkpointer that accepts ``serde=``::

        MemorySaver(serde=NamespaceAwareSerde())

    Non-event payloads are encoded exactly as the default
    ``JsonPlusSerializer`` would — the override applies only to ``Event``
    subclasses.

    Pass ``namespaces=`` to scope decorator-driven (``@migrate_from``)
    collection to the namespaces in play for this graph. Pass ``events=``
    for event classes that live outside every namespace — module-level
    ``IntegrationEvent``s and framework ``SystemEvent``s — which the
    namespace walk cannot reach; without them those identities resolve by
    import and so are shared between engine lifetimes of one module.
    ``EventGraph.from_namespaces`` fills this in automatically from the
    graph it builds. Pass
    ``migrations=`` for hand-authored cross-module renames or composite
    operations; the two compose. See :mod:`langgraph_events.serde.migrations`.
    """

    def __init__(
        self,
        migrations: Sequence[Migration] = (),
        *,
        namespaces: Sequence[type] = (),
        events: Sequence[type] = (),
        legacy_write: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        # Decorator-driven migrations come first so the duplicate-source
        # diagnostic, when a user-passed hand-authored entry conflicts,
        # names the user's migration as the second (more actionable than
        # naming the auto-collected one).
        decorated, oldest_historic, scope = _collect_decorated_migrations(
            namespaces, events
        )
        all_migrations = (*decorated, *migrations)
        (
            self._rename_table,
            self._addfield_table,
            self._origin_addfield_table,
        ) = _flatten_and_validate(all_migrations, scope)
        # Origin-scoped fills are the fan-in signal, and a fan-in cannot
        # ride legacy_write: writes would relabel EVERY instance under the
        # oldest historic identity, collapsing the per-origin distinction
        # the fills exist to preserve — and the old releases' classes do
        # not accept the back-filled field anyway.
        if legacy_write and self._origin_addfield_table:
            raise ValueError(
                "legacy_write=True cannot be combined with origin-scoped "
                "back-fills (migrate_from(backfill=...) or AddField keyed "
                "on a historic identity). Drain in-flight threads before "
                "the consolidation cutover, or drop legacy_write and accept "
                "read-only compatibility. See 'Consolidating N classes into "
                "one' in docs/event-migrations.md."
            )
        # The read path resolves through ``_scope`` before it falls back to
        # importing — see ``_resolve_identity``. ``_live_identities`` is its
        # key set: the identities revivable with no migration at all.
        self._scope = scope
        self._live_identities = frozenset(scope)
        self._legacy_write = legacy_write
        self._encode_default = _make_default(legacy_write, oldest_historic)
        self._tolerant_depth = 0
        for cls in _unreachable_migrate_from_siblings(scope):
            warn_user(
                f"{cls.__qualname__} is decorated with @migrate_from, but "
                f"this serde's namespaces=/events= never reaches it — its "
                f"migration will not be collected, so a payload under its "
                f"historic identity will not revive. Nest it inside a "
                f"Namespace passed via namespaces=, or pass it directly "
                f"in events=.",
                UnreachableMigrationWarning,
            )

    @property
    def _tolerant(self) -> bool:
        return self._tolerant_depth > 0

    @contextlib.contextmanager
    def tolerate_unresolved(self) -> Iterator[None]:
        """Degrade an unrevivable interrupt identity to
        :class:`UnrevivedIdentity` instead of raising ``Cannot revive``,
        for the duration of the ``with`` block.

        For ``EventGraph.threads_paused_on()``/``abandon()`` only — the
        retirement tools that must keep working on a thread whose paused
        class was already deleted. ``loads_typed()`` stays strict outside
        this block, so a genuine revival bug elsewhere still raises.

        Reentrant: a depth counter, not a flag — ``abandon()`` opens this
        block and then calls a helper that opens it again for one read;
        the inner exit must not turn tolerance off out from under the
        still-open outer block.

        WARNING: toggles per-instance state. Do not read through this
        same serde instance from another thread/task while the block is
        open — the same caveat as calling ``abandon()`` concurrently with
        another run on the thread it targets.
        """
        self._tolerant_depth += 1
        try:
            yield
        finally:
            self._tolerant_depth -= 1

    def revivable_identities(self) -> frozenset[tuple[str, str]]:
        """Every ``(module, qualname)`` this serde can revive — either still
        live in the namespaces and loose events it was constructed with
        (``namespaces=`` / ``events=``), or covered by a rename
        migration (``@migrate_from`` decorators in ``namespaces=``
        and hand-authored ``migrations=``).

        Read-only view. AddField targets are not added: a fill modifies
        kwargs for an identity revived by other means (a live class or a
        rename), it does not make an identity revivable by itself.
        """
        return self._live_identities | frozenset(self._rename_table.keys())

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
                    self._origin_addfield_table,
                    self._scope,
                    tolerant=self._tolerant,
                ),
                option=ormsgpack.OPT_NON_STR_KEYS,
            )
        except ValueError as exc:
            if errors:
                raise ValueError(errors[-1]) from exc
            raise
