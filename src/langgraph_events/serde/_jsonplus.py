"""``NamespaceAwareSerde`` — qualname-keyed roundtrip for nested events.

LangGraph's ``JsonPlusSerializer`` encodes Pydantic identity by
``(__module__, __name__)``. For events nested inside a ``Namespace``,
``__name__`` is leaf-only (e.g. ``"Approved"`` for ``Persona.Approve.Approved``)
and therefore collides across namespaces. We encode by
``(__module__, __qualname__)`` and revive via attribute walk.

We depend on a few private helpers from
``langgraph.checkpoint.serde.jsonplus`` (``_msgpack_default``, ``_option``).
They have been stable for some time but are technically private — pin a
compatible LangGraph version.
"""

from __future__ import annotations

import contextlib
import re
from typing import TYPE_CHECKING, Any, NamedTuple

import ormsgpack
from langgraph.types import Interrupt
from pydantic import BaseModel, ValidationError

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Sequence

    from langgraph_events.serde.migrations._core import (
        AddField,
        Migration,
        SplitEvent,
        TransformFields,
    )
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

# The dataclass ``TypeError`` for a kwarg the class does not declare.
# ``_revival_remedy`` and the revive gate's failure hint both read the
# field name out of it.
_UNEXPECTED_KWARG_RE = re.compile(r"unexpected keyword argument '(\w+)'")

# Imported AFTER the EXT constants, ``_UNEXPECTED_KWARG_RE`` and ``_option``
# are bound: ``_core`` pulls in the ``serde.migrations`` package whose
# ``__init__`` re-exports ``testing``, which imports those names back from
# this module. Defining them first lets that re-entry resolve against a
# partially-initialized ``_jsonplus`` without a circular import.
from langgraph_events.serde.migrations._core import (  # noqa: E402
    SplitError,
    TransformError,
    _apply_identity_migrations,
    _CallableError,
    _collect_decorated_migrations,
    _flatten_and_validate,
    _resolve_identity,
    _unimportable_payload_models,
    _unreachable_migrate_from_siblings,
)


class UnrevivedIdentity(NamedTuple):
    """Placeholder for an interrupt identity that could not be revived,
    produced only inside :meth:`NamespaceAwareSerde.tolerate_unresolved`.

    Carries the identity exactly as stored: ``module`` and ``qualname``.
    None of the original fields carry over. There is no live class to
    hold them. Not an ``Event`` subclass: a caller must never mistake
    this for a real, dispatchable event.

    Exists for ``EventGraph.threads_paused_on()``/``abandon()``. These
    are the retirement tools that must keep working on a thread whose
    paused class was deleted before every thread was settled. Every
    other read path stays strict: an unrevivable identity there is a
    genuine bug and must raise, not degrade silently.
    """

    module: str
    qualname: str


ReadRecord = tuple[tuple[str, str], tuple[str, str], bool]
"""One stored event record: ``(stored, resolved, touched)``. *stored* and
*resolved* are ``(module, qualname)`` identities. *touched* is ``True``
when the migration table rewrote the record."""


def _scan_identities(data: bytes) -> list[tuple[str, str]]:
    """Every ``(module, qualname)`` event identity stored in *data*, in
    read order, outer record before the records nested inside it.

    Class-blind: imports nothing and constructs nothing, so it cannot go
    stale when a class moves. Recurses into every ext payload, ours and
    upstream's, and ignores a payload that is not itself msgpack. An
    event inside a pydantic field is upstream ext 5 under ``__name__``
    only, so it is not seen here, as it is not seen by the migration
    tables on read.
    """
    found: list[tuple[str, str]] = []

    def hook(code: int, payload: bytes) -> None:
        # Remember the slot before recursing, so the outer record lands
        # ahead of the records nested inside its kwargs.
        slot = len(found)
        try:
            inner = ormsgpack.unpackb(
                payload, ext_hook=hook, option=ormsgpack.OPT_NON_STR_KEYS
            )
        except ormsgpack.MsgpackDecodeError:
            return None
        if (
            code == EXT_NAMESPACE_AWARE_EVENT
            and isinstance(inner, (list, tuple))
            and len(inner) == 3
            and isinstance(inner[0], str)
            and isinstance(inner[1], str)
        ):
            found.insert(slot, (inner[0], inner[1]))
        return None

    ormsgpack.unpackb(data, ext_hook=hook, option=ormsgpack.OPT_NON_STR_KEYS)
    return found


def _revival_remedy(qualname: str, exc: Exception) -> str:
    """The actionable second sentence of a ``Cannot revive`` message.

    *exc* is ``ImportError``/``AttributeError`` (the identity resolves to
    no live class at all), ``TypeError`` (it resolves, but the stored
    kwargs and the class's fields disagree), ``TransformError`` (a
    ``TransformFields`` raised, or returned a non-dict), or
    ``SplitError`` (a ``SplitEvent`` raised, or returned the wrong
    shape). Each needs a different remedy. "Map onto a tombstone with
    @migrate_from" is right for the first case: there is no live class
    yet. It is wrong for the second case, where *qualname* may already
    **be** the tombstone: redecorating it with the decorator it already
    carries fixes nothing.
    """
    if isinstance(exc, SplitError):
        op = exc.op
        return (
            f"The select keyed on {op.module}.{op.qualname} must accept a "
            f"payload of every era. It must return None when the "
            f"discriminating value is absent or None, or a (target, kwargs) "
            f"tuple whose target is in targets=. See 'Splitting one stored "
            f"event into two on a payload value' in docs/event-migrations.md."
        )
    if isinstance(exc, TransformError):
        op = exc.op
        return (
            f"The transform keyed on {op.module}.{op.qualname} must accept a "
            f"payload from every era and return a dict. Guard a key that can "
            f"be absent or None: kw.pop('x', None) and a None check. See "
            f"'Dropping, merging or retyping a field' in "
            f"docs/event-migrations.md."
        )
    if not isinstance(exc, (TypeError, ValidationError)):
        return (
            "The class may have been renamed or removed since the "
            "checkpoint was written. Settle the thread with "
            "abandon()/aabandon() before deleting the class, or map the "
            f"dead identity onto a tombstone class with "
            f"@migrate_from({qualname!r})."
        )
    field: str | None = None
    if isinstance(exc, ValidationError):
        for error in exc.errors():
            if error["type"] == "extra_forbidden" and error["loc"]:
                field = str(error["loc"][0])
                break
    else:
        match = _UNEXPECTED_KWARG_RE.search(str(exc))
        field = match.group(1) if match is not None else None
    if field is None:
        return (
            f"{qualname}'s fields no longer match the stored payload — "
            f"the class may have gained a required field with no default "
            f"and no AddField. Align the class with the payload's shape, "
            f"or back-fill the field with a migration."
        )
    return (
        f"{qualname} does not declare the field {field!r} the stored "
        f"payload carries. Add {field!r} to the class, matching the "
        f"old class's shape, or drop it from the payload with "
        f"@transform_fields."
    )


def _contains_unrevived(value: Any) -> bool:
    if isinstance(value, UnrevivedIdentity):
        return True
    if isinstance(value, dict):
        return any(_contains_unrevived(item) for item in value.values())
    if isinstance(value, (list, tuple, set, frozenset)):
        return any(_contains_unrevived(item) for item in value)
    return False


def _make_default(
    legacy_write: bool,
    oldest_historic: dict[tuple[str, str], tuple[str, str]],
    scope: dict[tuple[str, str], type],
    refusals: list[str],
) -> Callable[[Any], Any]:
    """Build the ``default=`` hook ormsgpack uses for unknown types.

    Closure so the per-serde ``legacy_write`` flag and ``oldest_historic``
    map thread down into recursive sub-encodes (Interrupt-wrapped values,
    etc.) without relying on module-level state.

    *refusals* collects the message of every write this hook refuses.
    ormsgpack swallows an exception raised inside ``default`` and raises
    its own ``MsgpackEncodeError`` with no cause chained. The caller
    reads *refusals* after that error to raise the real ``ValueError``.
    This mirrors *errors* in :func:`_make_ext_hook` on the read path.

    ``oldest_historic`` is built at construction from the serde's
    ``namespaces=`` scope. Encoding under an oldest historic identity is
    gated on the class being in this map — out-of-scope decorated classes
    fall through to their current qualname so bytes never reference a
    historic name the serde's own read path can't migrate back.

    Only fields with ``init=True`` are written: an ``init=False`` field is
    rebuilt by the constructor on read, which rejects it as a kwarg (#172).
    """

    def _default(obj: Any) -> Any:
        if isinstance(obj, UnrevivedIdentity):
            # #170: a read-side placeholder must never become a stored
            # value. Upstream's hook would encode this named tuple by
            # class name, so it would round-trip and a strict read could
            # not tell it from a real event.
            refusals.append(
                f"Refusing to store {obj.module}.{obj.qualname}: it is an "
                f"UnrevivedIdentity placeholder, and a placeholder must never "
                f"be stored. Recover the class with a tombstone, see "
                f"'Recovering a delete-first deployment' in "
                f"docs/event-migrations.md."
            )
            raise ValueError(refusals[-1])
        if isinstance(obj, Event):
            cls = obj.__class__
            module, qualname = cls.__module__, cls.__qualname__
            identity = (module, qualname)
            if "<locals>" in qualname:
                bound = scope.get(identity)
                if bound is None:
                    # A checkpoint serde is process-local state too. Bind a
                    # function-local event when this exact serde first writes
                    # it, so the same graph can revive its next checkpoint.
                    scope[identity] = cls
                elif bound is not cls:
                    raise ValueError(
                        "NamespaceAwareSerde cannot bind two different event "
                        f"classes to {module}.{qualname}"
                    )
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
                        {
                            name: getattr(obj, name)
                            for name, field in type(obj).model_fields.items()
                            if field.init is not False
                        },
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
    transform_table: dict[tuple[str, str], TransformFields],
    origin_transform_table: dict[tuple[str, str], TransformFields],
    split_table: dict[tuple[str, str], SplitEvent],
    scope: dict[tuple[str, str], type],
    *,
    unresolved: list[UnrevivedIdentity] | None = None,
    reads: list[ReadRecord] | None = None,
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

    *unresolved*, when not ``None``, is a collector. An unrevivable
    ``EXT_NAMESPACE_AWARE_EVENT`` identity is then appended to it and
    returned as an :class:`UnrevivedIdentity` instead of raising. Only
    ``NamespaceAwareSerde.tolerate_unresolved`` passes one. Every other
    caller keeps the strict default.

    *reads*, when not ``None``, is a second collector. Every stored
    ``EXT_NAMESPACE_AWARE_EVENT`` record is appended to it as a
    :data:`ReadRecord`, touched or not. Only
    ``NamespaceAwareSerde._record_reads`` passes one.
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
        try:
            # Rewrite historic identity to current, run any TransformFields
            # and inject any AddField defaults — shared with the baseline
            # test helper so the read-side migration rule lives in exactly
            # one place. Inside the ``try``: a transform that raises must
            # reach the ``errors`` channel like every other failure, or
            # ormsgpack reports a bare ``ext_hook failed``. The identity
            # stays the STORED one when the migration itself fails.
            stored = (module_name, qualname)
            # A fill mutates ``kwargs`` in place: snapshot it first.
            before = dict(kwargs) if reads is not None else None
            module_name, qualname, kwargs = _apply_identity_migrations(
                module_name,
                qualname,
                kwargs,
                rename_table,
                addfield_table,
                origin_addfield_table,
                transform_table,
                origin_transform_table,
                split_table,
            )
            instance = _resolve_identity(module_name, qualname, scope=scope)(**kwargs)
            if reads is not None:
                # Recorded after the record revived, so a degraded record
                # lands in *unresolved* only. Touched means the table
                # changed the identity or the kwargs: a rewrite would
                # store different bytes. A table entry that changed
                # nothing (a fill whose field the payload already held)
                # is untouched, so a second rewrite converges.
                touched = stored != (module_name, qualname) or kwargs != before
                reads.append((stored, (module_name, qualname), touched))
            return instance
        except (
            ImportError,
            AttributeError,
            TypeError,
            ValidationError,
            _CallableError,
        ) as exc:
            # ``TypeError`` is the field-shape mismatch: the identity
            # resolves, but the stored kwargs carry a key the live class
            # has dropped, or omit a field it has gained with no AddField.
            # ``_CallableError`` wraps whatever a transform or a select
            # raised.
            if isinstance(exc, ValidationError) and unresolved is not None:
                placeholder_fields = {
                    name for name, value in kwargs.items() if _contains_unrevived(value)
                }
                errors_only_cover_placeholders = placeholder_fields and all(
                    error["loc"] and str(error["loc"][0]) in placeholder_fields
                    for error in exc.errors()
                )
                live_cls = _resolve_identity(module_name, qualname, scope=scope)
                if errors_only_cover_placeholders and issubclass(live_cls, BaseModel):
                    instance = live_cls.model_construct(**kwargs)
                    if reads is not None:
                        touched = stored != (module_name, qualname) or kwargs != before
                        reads.append((stored, (module_name, qualname), touched))
                    return instance
            if unresolved is not None:
                # Retirement cleanup only (see the *unresolved* parameter
                # docstring above). The caller is a tool that exists to
                # settle exactly this thread, not a normal read. Degrade
                # instead of raising. Keep no partial kwargs: there is
                # no live class to hold them. The collector sees every
                # degrade, however deep the identity sat in the blob.
                placeholder = UnrevivedIdentity(module=module_name, qualname=qualname)
                unresolved.append(placeholder)
                return placeholder
            failure = (
                str(exc)
                if isinstance(exc, _CallableError)
                else f"{type(exc).__name__}: {exc}"
            )
            errors.append(
                f"Cannot revive {module_name}.{qualname}: {failure}. "
                f"{_revival_remedy(qualname, exc)}"
            )
            raise

    return _ext_hook


from langgraph_events._event import Event  # noqa: E402  (avoid circular import order)
from langgraph_events._warn import warn_user  # noqa: E402


class UnreachableMigrationWarning(UserWarning):
    """A ``@migrate_from``-decorated class was not collected into a
    :class:`NamespaceAwareSerde`'s scope. Its migration does nothing."""


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
        for event_cls, field, model in _unimportable_payload_models(scope.values()):
            # The LangGraph serializer, which owns the pydantic ext code,
            # stores a payload by ``__name__`` and revives it with one
            # ``getattr`` on the module. Every failure path there returns
            # the raw dump dict with no error. The event itself still
            # constructs around that dict, so nothing downstream notices.
            # Fail here, where the author can act (#167).
            nested = "." in model.__qualname__
            remedy = (
                f"Move {model.__qualname__} to module scope in {model.__module__}."
                if nested
                else f"Bind it on module {model.__module__} under the name "
                f"{model.__name__!r}. Alias a parametrized generic at module "
                f"level, for example IntBox = Box[int]."
            )
            raise ValueError(
                f"{event_cls.__qualname__}.{field} is annotated with pydantic "
                f"model {model.__qualname__}. Its checkpoint identity "
                f"{model.__module__}.{model.__name__} does not import. The "
                f"LangGraph serializer stores a pydantic payload by __name__ "
                f"and revives it with getattr on the module. On a miss it "
                f"returns a raw dict with no error. {remedy}"
            )
        all_migrations = (*decorated, *migrations)
        (
            self._rename_table,
            self._addfield_table,
            self._origin_addfield_table,
            self._transform_table,
            self._origin_transform_table,
            self._split_table,
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
        # A transform has no inverse. A write relabelled under the oldest
        # historic identity would carry the CURRENT shape, which the old
        # release's class does not accept, and the transform only runs on
        # read. Refuse at construction, like the origin-fill case above.
        if legacy_write and (self._transform_table or self._origin_transform_table):
            raise ValueError(
                "legacy_write=True cannot be combined with a TransformFields "
                "(transform_fields, migrate_from(transform=...) or a "
                "hand-authored TransformFields). A transform runs on read and "
                "has no inverse, so an old release cannot read what this one "
                "writes. Drain in-flight threads before the cutover, or drop "
                "legacy_write and accept read-only compatibility. See "
                "'Dropping, merging or retyping a field' in "
                "docs/event-migrations.md."
            )
        # A split has no inverse either: an old release has no class for
        # the target, and select only runs on read.
        if legacy_write and self._split_table:
            raise ValueError(
                "legacy_write=True cannot be combined with a SplitEvent "
                "(split_event, Migration.split_event or a hand-authored "
                "SplitEvent). A split runs on read and has no inverse, so an "
                "old release cannot read what this one writes. Drain "
                "in-flight threads before the cutover, or drop legacy_write "
                "and accept read-only compatibility. See 'Splitting one "
                "stored event into two on a payload value' in "
                "docs/event-migrations.md."
            )
        # The read path resolves through ``_scope`` before it falls back to
        # importing — see ``_resolve_identity``.
        self._scope = scope
        self._legacy_write = legacy_write
        self._oldest_historic = oldest_historic
        self._tolerant_depth = 0
        self._unresolved: list[UnrevivedIdentity] | None = None
        self._record_depth = 0
        self._reads: list[ReadRecord] | None = None
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

    @contextlib.contextmanager
    def tolerate_unresolved(self) -> Iterator[list[UnrevivedIdentity]]:
        """Degrade an unrevivable event identity to
        :class:`UnrevivedIdentity` instead of raising ``Cannot revive``,
        for the duration of the ``with`` block.

        Yields a collector. Every identity degraded inside the block is
        appended to it, in read order, wherever it sat in the blob: a
        top-level event, a pending write, or a field nested inside a
        live event. ``EventGraph.unrevivable_threads()`` reads this
        collector. A reader that walked the checkpoint structure instead
        would miss the nested shapes.

        For the retirement tools only: ``EventGraph.threads_paused_on()``,
        ``abandon()`` and ``unrevivable_threads()``. These must keep
        working on a thread whose class was already deleted.
        ``loads_typed()`` stays strict outside this block, so a genuine
        revival bug elsewhere still raises.

        Reentrant: a depth counter, not a flag. ``abandon()`` opens this
        block and then calls a helper that opens it again for one read.
        The inner exit must not turn tolerance off out from under the
        still-open outer block. An inner block shares the outer block's
        collector.

        WARNING: toggles per-instance state. Do not read through this
        same serde instance from another thread/task while the block is
        open. This is the same caveat as calling ``abandon()``
        concurrently with another run on the thread it targets.
        """
        collector = self._unresolved
        if collector is None:
            collector = self._unresolved = []
        self._tolerant_depth += 1
        try:
            yield collector
        finally:
            self._tolerant_depth -= 1
            if self._tolerant_depth == 0:
                self._unresolved = None

    @contextlib.contextmanager
    def _record_reads(self) -> Iterator[list[ReadRecord]]:
        """Collect every stored event record this serde reads, for the
        duration of the ``with`` block.

        Yields a collector of :data:`ReadRecord` entries, one per stored
        ``EXT_NAMESPACE_AWARE_EVENT`` record that revived, in read order,
        wherever the record sat in the blob. A record that degraded to an
        :class:`UnrevivedIdentity` is not collected. The ``touched`` flag
        is ``True`` when the migration table changed the identity or the
        kwargs, so a re-encode would store different bytes. A class that
        gained a dataclass default with no ``AddField`` is untouched: the
        table never saw the missing field.

        For ``EventGraph.plan_rewrite()`` and ``rewrite_store()`` only.
        Reentrant like :meth:`tolerate_unresolved`, and with the same
        per-instance-state caveat: do not read through this serde from
        another thread or task while the block is open.
        """
        collector = self._reads
        if collector is None:
            collector = self._reads = []
        self._record_depth += 1
        try:
            yield collector
        finally:
            self._record_depth -= 1
            if self._record_depth == 0:
                self._reads = None

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
        return frozenset(self._scope) | frozenset(self._rename_table.keys())

    def dumps_typed(self, obj: Any) -> tuple[str, bytes]:
        if obj is None or isinstance(obj, (bytes, bytearray)):
            return super().dumps_typed(obj)
        # The hook from ``_make_default`` is a strict superset of
        # upstream's ``_msgpack_default``: anything upstream encodes, we
        # encode the same way. So an ``MsgpackEncodeError`` here is
        # genuinely unencodable, unless the hook refused the write on
        # purpose (#170). The old behaviour warned and called
        # ``super().dumps_typed`` — which in the default config simply
        # re-raised, and with the parent's binary-fallback kwarg enabled
        # would silently emit unsafe-binary bytes that bypass the
        # migration table. Let the encode error propagate at the source
        # so the caller widens ``_default`` or removes the payload from
        # state explicitly.
        #
        # The hook is built per call so *refusals* is local to this
        # write. A per-instance list would leak across concurrent writes.
        refusals: list[str] = []
        default = _make_default(
            self._legacy_write,
            self._oldest_historic,
            self._scope,
            refusals,
        )
        try:
            return "msgpack", ormsgpack.packb(obj, default=default, option=_option)
        except ormsgpack.MsgpackEncodeError as exc:
            if refusals:
                raise ValueError(refusals[-1]) from exc
            raise

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
                    self._transform_table,
                    self._origin_transform_table,
                    self._split_table,
                    self._scope,
                    unresolved=self._unresolved,
                    reads=self._reads,
                ),
                option=ormsgpack.OPT_NON_STR_KEYS,
            )
        except ValueError as exc:
            if errors:
                raise ValueError(errors[-1]) from exc
            raise
