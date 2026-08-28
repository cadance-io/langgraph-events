"""Event-identity migrations for ``NamespaceAwareSerde``.

Migrations are ordered, idempotent rewrites of the encoded
``(module, qualname, kwargs)`` tuple. They run inside the serde's
ext-hook between unpack and revive. The wire format is unchanged from
prior library versions — this is a read-side affordance only. See #70.
"""

from __future__ import annotations

import importlib
import itertools
from dataclasses import dataclass, fields, is_dataclass
from typing import TYPE_CHECKING, Any

from langgraph_events._event import Event, _iter_nested_events
from langgraph_events._labels import distinct_labels

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping, Sequence

    from langgraph_events._reducer import BaseReducer

# Local sentinel for "no ``default`` supplied". ``dataclasses.MISSING`` looks
# like the canonical choice but it's special-cased by ``@dataclass`` itself
# to mean "this field has no default" — passing it as the value of
# ``AddField.default`` collides with that contract.
_MISSING: Any = object()


@dataclass(frozen=True)
class RenameEvent:
    """Rewrite ``(old_module, old_qualname)`` to ``(new_module, new_qualname)``.

    Idempotent: if the input tuple does not match, returns it unchanged.
    """

    old_module: str
    old_qualname: str
    new_module: str
    new_qualname: str


@dataclass(frozen=True)
class AddField:
    """Inject a dataclass ``field`` into kwargs for events with this identity.

    Scoped to a specific event identity (``module`` + ``qualname``) so it
    doesn't touch unrelated payloads. The identity decides WHEN it runs:
    name the CURRENT (live) class and it applies AFTER any matching
    :class:`RenameEvent` — the common "class gained a required field"
    case. Name a HISTORIC identity covered by a rename and it applies
    BEFORE the rename, only to payloads written under that exact origin —
    the fan-in case where collapsed origins each pin their own value
    (see ``migrate_from(backfill=...)`` for the decorator form).

    Use ``default`` for immutable values (``str``, ``int``, ``None``,
    tuples). Use ``default_factory`` for anything that could be mutated
    in place by a consumer (``list``, ``dict``, custom objects) — the
    serde invokes it per-read so two migrated payloads never share the
    same object.
    """

    module: str
    qualname: str
    field: str
    default: Any = _MISSING
    default_factory: Callable[[], Any] | None = None

    def __post_init__(self) -> None:
        if (self.default is _MISSING) == (self.default_factory is None):
            raise ValueError(
                f"AddField({self.module}:{self.qualname}, {self.field!r}): "
                f"exactly one of `default` or `default_factory` is required."
            )
        if isinstance(self.default, (list, dict, set, bytearray)):
            raise ValueError(
                f"AddField({self.module}:{self.qualname}, {self.field!r}): "
                f"mutable `default` of type {type(self.default).__name__} "
                f"would be shared across every migrated payload of this "
                f"class. Use `default_factory={type(self.default).__name__}` "
                f"instead."
            )


Operation = RenameEvent | AddField


@dataclass(frozen=True)
class Migration:
    """A named, ordered group of operations.

    Migrations exist as an authoring grouping — the serde flattens all
    operations across all migrations into lookup tables at construction
    time.

    ``name`` is an optional label used in validation diagnostics. It costs
    nothing to omit and helps when the validator surfaces a collision
    between two hand-authored migrations.
    """

    operations: tuple[Operation, ...]
    name: str = ""

    @classmethod
    def rename(
        cls,
        name: str = "",
        *,
        old_module: str,
        old_qualname: str,
        to: type | None = None,
        new_module: str | None = None,
        new_qualname: str | None = None,
    ) -> Migration:
        """Single-op rename sugar — the common case is one rewrite per
        migration.

        The new (post-rename) identity always names a live class, so pass
        it as ``to=<class>`` for refactor-safety (an IDE rename moves with
        it). ``new_module``/``new_qualname`` remain for the cross-module
        case where the live class can't be imported at authoring time. The
        old identity stays a string — that class is gone.
        """
        new_module, new_qualname = _target_identity(
            "Migration.rename",
            "to",
            ("new_module", "new_qualname"),
            "live target",
            to,
            new_module,
            new_qualname,
        )
        return cls(
            name=name,
            operations=(
                RenameEvent(
                    old_module=old_module,
                    old_qualname=old_qualname,
                    new_module=new_module,
                    new_qualname=new_qualname,
                ),
            ),
        )

    @classmethod
    def add_field(
        cls,
        name: str = "",
        *,
        target: type | None = None,
        module: str | None = None,
        qualname: str | None = None,
        field: str,
        default: Any = _MISSING,
        default_factory: Callable[[], Any] | None = None,
    ) -> Migration:
        """Single-op add-field sugar.

        ``name`` labels the migration; ``field`` is the dataclass field
        being added. For the common post-rename fill the target is the
        live class — pass it as ``target=<class>`` for refactor-safety.
        ``module``/``qualname`` remain for a class that can't be imported
        at authoring time, and for the origin-scoped fill keyed on a
        HISTORIC identity (that class is gone — strings are the only
        spelling). Same convention as :class:`AddField` — moving between
        the sugar and the raw form needs no kwarg rename.
        """
        module, qualname = _target_identity(
            "Migration.add_field",
            "target",
            ("module", "qualname"),
            "target",
            target,
            module,
            qualname,
        )
        return cls(
            name=name,
            operations=(
                AddField(
                    module=module,
                    qualname=qualname,
                    field=field,
                    default=default,
                    default_factory=default_factory,
                ),
            ),
        )


def _target_identity(
    method: str,
    cls_kw: str,
    str_kws: tuple[str, str],
    target_noun: str,
    cls: type | None,
    module: str | None,
    qualname: str | None,
) -> tuple[str, str]:
    """Resolve a ``(module, qualname)`` from either a live class or the
    explicit string pair, enforcing "exactly one of the two".

    Shared by ``Migration.rename`` and ``Migration.add_field`` — the only
    differences are the kwarg names and noun in the diagnostics, threaded
    through *method* / *cls_kw* / *str_kws* / *target_noun* so the
    user-facing messages stay identical to the hand-written ones.
    """
    kw1, kw2 = str_kws
    if cls is not None:
        if module is not None or qualname is not None:
            raise ValueError(
                f"{method}: pass either `{cls_kw}=<class>` or "
                f"`{kw1}`/`{kw2}`, not both."
            )
        return cls.__module__, cls.__qualname__
    if module is None or qualname is None:
        raise ValueError(
            f"{method}: provide the {target_noun} as `{cls_kw}=<class>` "
            f"or both `{kw1}` and `{kw2}`."
        )
    return module, qualname


def _flatten_and_validate(
    migrations: Sequence[Migration],
    scope: Mapping[tuple[str, str], type] | None = None,
) -> tuple[
    dict[tuple[str, str], tuple[str, str]],
    dict[tuple[str, str], tuple[AddField, ...]],
    dict[tuple[str, str], tuple[AddField, ...]],
]:
    """Collapse all operations into per-purpose lookup tables.

    Rename ops fold into a direct historic→final map (chain ``A→B→C``
    becomes ``A→C`` and ``B→C``) so reads do a single dict lookup
    regardless of chain depth. AddField ops are bucketed by their target
    identity: a target that is a rename source goes into the origin table
    (applied pre-rename), a live target into the post-rename table.

    Validation is intentionally strict — every error here would otherwise
    surface as a ``ValueError`` on first production read, which is the
    worst possible time to discover it. It asks the same "does this
    identity reach a live class?" question the read path asks, so it takes
    the same *scope* map and answers it the same way (see
    :func:`_resolve_identity`); validating import-only would reject a
    chain that terminates on a class the reader resolves perfectly well.

    Raises:
        ValueError: on duplicate rename sources, dead-end chains, cycles,
            shadowing of live classes, or AddField targets that don't
            resolve.
    """
    rename_edges: dict[tuple[str, str], tuple[str, str]] = {}
    # Remember which migration each edge came from so the duplicate-source
    # diagnostic can name the conflicting migrations, not just their targets.
    edge_origin: dict[tuple[str, str], str] = {}
    # AddField ops carry their migration's name for the same reason.
    addfield_ops: list[tuple[AddField, str]] = []
    for migration in migrations:
        for op in migration.operations:
            if isinstance(op, RenameEvent):
                old_key = (op.old_module, op.old_qualname)
                new_key = (op.new_module, op.new_qualname)
                if old_key in rename_edges and rename_edges[old_key] != new_key:
                    first_label = _migration_label(edge_origin[old_key])
                    second_label = _migration_label(migration.name)
                    raise ValueError(
                        f"Duplicate rename source: {op.old_module}:"
                        f"{op.old_qualname!r} is targeted by both "
                        f"{first_label} → {rename_edges[old_key]!r} and "
                        f"{second_label} → {new_key!r}. Each historic identity "
                        f"may map to at most one successor."
                    )
                rename_edges[old_key] = new_key
                edge_origin[old_key] = migration.name
            elif isinstance(op, AddField):
                addfield_ops.append((op, migration.name))
            else:
                # ``Operation`` is closed by design (RenameEvent | AddField).
                # Silently ignoring an unknown type hides authoring errors —
                # surface them at construction with a targeted diagnostic.
                raise TypeError(
                    f"Unknown migration operation type "
                    f"{type(op).__name__!r} in {_migration_label(migration.name)}. "
                    f"Expected RenameEvent or AddField."
                )

    rename_table: dict[tuple[str, str], tuple[str, str]] = {}
    for start in list(rename_edges):
        rename_table[start] = _resolve_chain_terminus(start, rename_edges, scope)

    addfield_table, origin_addfield_table = _bucket_addfields(
        addfield_ops, rename_table, scope
    )
    return rename_table, addfield_table, origin_addfield_table


def _bucket_addfields(
    addfield_ops: Sequence[tuple[AddField, str]],
    rename_table: dict[tuple[str, str], tuple[str, str]],
    scope: Mapping[tuple[str, str], type] | None = None,
) -> tuple[
    dict[tuple[str, str], tuple[AddField, ...]],
    dict[tuple[str, str], tuple[AddField, ...]],
]:
    """Bucket AddField ops by target kind and validate each one.

    Returns ``(post_rename_table, origin_table)``. Each op carries the
    name of the migration that declared it, for diagnostics.
    """
    addfield_table: dict[tuple[str, str], list[AddField]] = {}
    origin_addfield_table: dict[tuple[str, str], list[AddField]] = {}
    # (identity, field) → migration name, for the duplicate-fill diagnostic.
    # "source" as in "which migration declared it" — NOT "origin", which in
    # this module means the historic identity a payload was written under.
    fill_source: dict[tuple[tuple[str, str], str], str] = {}
    for op, migration_name in addfield_ops:
        target = (op.module, op.qualname)
        # Rename-table membership decides the bucket BEFORE the resolve
        # probe: a rename source can never resolve live (the chain walk
        # rejects shadowing), so the buckets are disjoint by construction.
        if target in rename_table:
            bucket = origin_addfield_table
            live_identity = rename_table[target]
        elif _resolves(*target, scope=scope):
            bucket = addfield_table
            live_identity = target
        else:
            raise ValueError(
                f"AddField target {op.module}:{op.qualname!r} neither "
                f"resolves to a live class — by this serde's namespace "
                f"scope or by import — nor matches a "
                f"historic identity covered by a rename migration. Either "
                f"the target was deleted after the migration was authored, "
                f"the module/qualname has a typo, or the historic identity "
                f"is missing its @migrate_from / RenameEvent."
            )
        # A fill naming a field the live class doesn't have would surface
        # as a dataclass TypeError at first production read — catch the
        # typo here instead. ``live_identity`` always resolves: post-rename
        # targets were just probed, origin targets point at a chain
        # terminus ``_resolve_chain_terminus`` already validated.
        live_cls = _resolve_identity(*live_identity, scope=scope)
        if is_dataclass(live_cls) and op.field not in {
            f.name for f in fields(live_cls)
        }:
            raise ValueError(
                f"AddField({op.module}:{op.qualname!r}, {op.field!r}) in "
                f"{_migration_label(migration_name)}: the live class "
                f"{live_identity[1]} has no field {op.field!r} — likely a "
                f"typo in the back-fill field name."
            )
        # The read path applies the FIRST fill for a field and silently
        # skips the rest (``setdefault``) — a second fill on the same
        # (identity, field) is always an authoring mistake, and with
        # origin-scoped fills a likely copy-paste one. Same philosophy as
        # the duplicate-rename-source guard above.
        if (key := (target, op.field)) in fill_source:
            first_label = _migration_label(fill_source[key])
            second_label = _migration_label(migration_name)
            raise ValueError(
                f"Duplicate AddField: {op.module}:{op.qualname!r} field "
                f"{op.field!r} is filled by both {first_label} and "
                f"{second_label}. Each (identity, field) pair may carry "
                f"at most one back-fill."
            )
        fill_source[key] = migration_name
        bucket.setdefault(target, []).append(op)

    return (
        {k: tuple(v) for k, v in addfield_table.items()},
        {k: tuple(v) for k, v in origin_addfield_table.items()},
    )


def _resolve_chain_terminus(
    start: tuple[str, str],
    rename_edges: dict[tuple[str, str], tuple[str, str]],
    scope: Mapping[tuple[str, str], type] | None = None,
) -> tuple[str, str]:
    """Walk *start* through *rename_edges* and return the terminus.

    Raises ``ValueError`` with a specific message for each detectable
    failure: trivial self-loop, cycle, dead-end (non-resolvable terminus),
    or shadowing of a currently-live class.
    """
    # Diagnose the trivial cycle (``A → A``) ahead of the chain walk —
    # "Cycle in migration chain" is technically true but misleads. A
    # self-loop is almost always a typo or stale paste; say so directly.
    if rename_edges[start] == start:
        raise ValueError(
            f"Migration source {start[0]}:{start[1]!r} maps to itself "
            f"— a self-loop trivially shadows the live class. Remove "
            f"the RenameEvent, or fix old_qualname/new_qualname if "
            f"the duplication is a typo."
        )
    seen: set[tuple[str, str]] = {start}
    current = start
    while current in rename_edges:
        current = rename_edges[current]
        if current in seen:
            raise ValueError(
                f"Cycle in migration chain starting at "
                f"{start[0]}:{start[1]!r}. Migrations must form a DAG."
            )
        seen.add(current)
    if not _resolves(*current, scope=scope):
        raise ValueError(
            f"Migration chain from {start[0]}:{start[1]!r} terminates at "
            f"{current[0]}:{current[1]!r}, which does not resolve to a "
            f"live class — by this serde's namespace scope or by import. "
            f"Either the chain target was "
            f"renamed/deleted after the migration was authored, or the "
            f"new module/qualname has a typo."
        )
    # Reject migrations whose old name shadows a class that still exists
    # — would silently rewrite live payloads on read.
    if _resolves(*start, scope=scope):
        raise ValueError(
            f"Migration source {start[0]}:{start[1]!r} resolves to a "
            f"currently-live class — reachable through this serde's "
            f"namespace scope, or by import. A rename whose old name is "
            f"still live would shadow it on read. Remove the old class, "
            f"or drop it from namespaces=, before declaring this migration."
        )
    return current


def _migration_label(name: str) -> str:
    """Render a migration's name for diagnostics, with a placeholder for blanks."""
    return f"migration {name!r}" if name else "an unnamed migration"


def _resolve_identity(
    module: str,
    qualname: str,
    *,
    scope: Mapping[tuple[str, str], type] | None = None,
) -> Any:
    """Resolve an event identity to its live class — *scope* first, import
    walk as the fallback.

    *scope* is the ``(module, qualname) → class`` map a serde builds from
    its own ``namespaces=`` walk, the mirror of the encode-side
    ``oldest_historic`` map. It wins over the import walk because it is
    the authority on which classes THIS serde speaks for: it reaches
    classes no import can (a namespace defined inside a function carries
    ``<locals>`` in its qualname) and it keeps two lifetimes of one module
    apart, which ``(module, qualname)`` alone cannot (#150).

    The import walk still covers every identity the namespace walk never
    reached — module-level ``IntegrationEvent``s, framework
    ``SystemEvent``s, and anything outside ``namespaces=``.

    Raises ``ImportError`` if the module is gone, ``AttributeError`` if the
    qualname no longer resolves — callers decide which to treat as
    "missing". Single source of truth for the identity → live-class walk
    shared by the read path, ``_resolves``, and the test helpers.
    """
    if scope is not None and (in_scope := scope.get((module, qualname))) is not None:
        return in_scope
    obj: Any = importlib.import_module(module)
    for part in qualname.split("."):
        obj = getattr(obj, part)
    return obj


def _resolves(
    module: str,
    qualname: str,
    *,
    scope: Mapping[tuple[str, str], type] | None = None,
) -> bool:
    """``True`` iff ``(module, qualname)`` reaches a live class — found in
    *scope*, or importable and walkable. Mirrors :func:`_resolve_identity`
    exactly, so "resolvable" means the same thing to migration validation
    as it does to the read path."""
    try:
        _resolve_identity(module, qualname, scope=scope)
    except (ImportError, AttributeError):
        return False
    return True


def _resolve_rename(
    module: str,
    qualname: str,
    rename_table: dict[tuple[str, str], tuple[str, str]],
) -> tuple[str, str]:
    """Map a (possibly historic) identity to its current one.

    Returns the input unchanged when no rename applies. The table holds the
    transitive closure (every historic identity points at the FINAL current
    one), so a single lookup covers chains of arbitrary depth. Single source
    of this rule for the read path and the baseline test helper.
    """
    return rename_table.get((module, qualname), (module, qualname))


def _inject_addfields(
    ops: tuple[AddField, ...],
    kwargs: dict[str, Any],
) -> None:
    """Inject each op's default into *kwargs* in place, never overwriting.

    ``setdefault`` is idempotent — a value already present is preserved.
    Mutable defaults go through ``default_factory`` so each migrated
    payload gets its own instance; the ``not in`` guard keeps the factory
    from firing when the field is already supplied.
    """
    for op in ops:
        if op.default_factory is not None:
            if op.field not in kwargs:
                kwargs[op.field] = op.default_factory()
        else:
            kwargs.setdefault(op.field, op.default)


def _apply_identity_migrations(
    module: str,
    qualname: str,
    kwargs: dict[str, Any],
    rename_table: dict[tuple[str, str], tuple[str, str]],
    addfield_table: dict[tuple[str, str], tuple[AddField, ...]],
    origin_addfield_table: dict[tuple[str, str], tuple[AddField, ...]],
) -> tuple[str, str]:
    """Inject origin-scoped AddField defaults keyed on the PRE-rename
    identity, resolve *module*/*qualname* through the rename table, then
    inject post-rename AddField defaults — all into *kwargs* in place.
    Returns the post-rename identity.

    Origin fills run first so the precedence is: explicit payload value >
    origin-scoped fill > class-global fill (``setdefault`` never
    overwrites). An op lives in exactly one table — origin keys are rename
    sources, post keys are live identities, disjoint by validation — so no
    op is ever applied twice.
    """
    _inject_addfields(origin_addfield_table.get((module, qualname), ()), kwargs)
    module, qualname = _resolve_rename(module, qualname, rename_table)
    _inject_addfields(addfield_table.get((module, qualname), ()), kwargs)
    return module, qualname


_MIGRATE_FROM_ATTR = "__lge_migrate_from__"
_ORIGIN_BACKFILL_ATTR = "__lge_origin_backfill__"


def migrate_from(
    *old_qualnames: str,
    in_module: str | None = None,
    backfill: dict[str, Any] | None = None,
) -> Callable[[type], type]:
    """Mark that this ``Event`` class formerly lived at ``old_qualnames``.

    Multiple positional args declare a chain in temporal order — oldest
    first. ``in_module`` defaults to the decorated class's ``__module__``
    and applies to all historic qualnames. Use the manual :class:`Migration`
    list directly for cross-module relocations or per-step module changes.

    ``backfill`` maps field names to defaults injected ONLY into payloads
    that were written under this decorator's historic qualname — the
    fan-in case where N classes collapse into one and each origin pins a
    different value for a discriminator the old payloads never carried.
    Precedence: payload value > origin fill > class-global
    :func:`backfill` (the fallback for origins without a scoped entry).
    Requires exactly one historic qualname per decorator (a multi-qualname
    chain is ambiguous about which origin the fill belongs to); a fill for
    payloads from *every* era is class-global :func:`backfill`'s job.
    Mutable values are rejected at decoration — for a per-origin
    ``default_factory`` hand-author ``Migration.add_field`` keyed on the
    historic identity.

    Metadata is stashed on the class as ``__lge_migrate_from__`` (and
    ``__lge_origin_backfill__`` for the per-origin fills).
    :class:`NamespaceAwareSerde` walks the namespaces passed to its
    ``namespaces=`` argument at construction and assembles a
    :class:`Migration` per decorated class automatically — no separate
    collection step is required.
    """
    if not old_qualnames:
        raise ValueError("@migrate_from requires at least one historic qualname.")
    if backfill is not None:
        if len(old_qualnames) != 1:
            raise ValueError(
                "@migrate_from(backfill=...) requires exactly one historic "
                "qualname per decorator — a multi-qualname chain is ambiguous "
                "about which origin the back-fill belongs to. Stack one "
                "decorator per origin, or use class-global @backfill for a "
                "fill that applies to every era."
            )
        if not backfill:
            raise ValueError(
                "@migrate_from(backfill=...) requires at least one field — "
                "an empty dict back-fills nothing; drop the argument."
            )
        # Mirror AddField's mutable-default guard here, at decoration, where
        # the dict is in hand — and steer to the escape hatch that exists
        # for THIS form (the dict cannot spell `default_factory=`).
        for field_name, value in backfill.items():
            if isinstance(value, (list, dict, set, bytearray)):
                raise ValueError(
                    f"@migrate_from(backfill=...): mutable value of type "
                    f"{type(value).__name__} for {field_name!r} would be "
                    f"shared across every revived payload of this origin. "
                    f"For a per-origin default_factory, hand-author "
                    f"Migration.add_field(module=..., "
                    f"qualname={old_qualnames[0]!r}, field={field_name!r}, "
                    f"default_factory={type(value).__name__})."
                )

    def _wrap(cls: type) -> type:
        module = in_module if in_module is not None else cls.__module__
        history = tuple((module, q) for q in old_qualnames)
        # ``cls.__dict__.get`` (not ``getattr``) so the marker doesn't leak
        # through MRO when a class inherits from a decorated parent — see
        # the read-side ``getattr`` companion in ``_jsonplus._make_default``.
        existing = cls.__dict__.get(_MIGRATE_FROM_ATTR, ())
        # A repeated origin would otherwise build the chain Old → Old →
        # current and surface as a baffling "Duplicate rename source …
        # → itself" at serde construction — say "duplicate origin" here,
        # where the duplication actually happens.
        seen = set(existing)
        for identity in history:
            if identity in seen:
                raise ValueError(
                    f"@migrate_from: duplicate origin {identity[1]!r} (in "
                    f"module {identity[0]!r}) on {cls.__qualname__}. Each "
                    f"historic identity may be declared once — stacked "
                    f"decorators and multi-arg chains accumulate into one "
                    f"history."
                )
            seen.add(identity)
        # Python applies decorators bottom-up, so the bottom decorator runs
        # first and its qualnames are the OLDEST in the chain. Place
        # ``existing`` (from the inner, earlier-applied decorator) ahead of
        # ``history`` (from the outer one). Aligns stacked decorators with
        # the multi-arg form ``@migrate_from("A", "B")`` where A is oldest.
        setattr(cls, _MIGRATE_FROM_ATTR, existing + history)
        if backfill:
            existing_fills = cls.__dict__.get(_ORIGIN_BACKFILL_ATTR, ())
            entry = ((module, old_qualnames[0]), dict(backfill))
            setattr(cls, _ORIGIN_BACKFILL_ATTR, (*existing_fills, entry))
        return cls

    return _wrap


_BACKFILL_ATTR = "__lge_backfill__"


def backfill(
    field: str,
    *,
    default: Any = _MISSING,
    default_factory: Callable[[], Any] | None = None,
) -> Callable[[type], type]:
    """Back-fill ``field`` for payloads written before it existed.

    The class-scoped, auto-collected sibling of :func:`migrate_from` for
    the "added a now-required field" case. Use it when ``field`` is
    required when the event is constructed in code, but pre-existing
    checkpoints predate it and must revive with a legacy value — an
    asymmetry a plain dataclass default cannot express (a default would
    relax the constructor for everyone). A field that *can* carry a
    dataclass default needs no decorator at all; it revives for free.

    ``default`` / ``default_factory`` follow the exact :class:`AddField`
    convention (one is required; mutable ``default`` is rejected) — the
    metadata becomes an :class:`AddField` keyed on this class's current
    identity, so moving between this and the raw form needs no rename.

    Metadata is stashed as ``__lge_backfill__``. :class:`NamespaceAwareSerde`
    collects it from the namespaces it is built with — exactly like
    ``@migrate_from`` — so no ``migrations=`` list is required. Stacked
    decorators accumulate. Composes with ``@migrate_from`` on the same
    class: the rename is applied first, then the back-fill on the
    resulting (current) identity — so the fill is CLASS-GLOBAL, one value
    for payloads from every origin and era. When collapsed origins each
    need their own value, use ``migrate_from(backfill=...)`` instead; a
    class-global fill then acts as the fallback for unscoped origins.
    """

    def _wrap(cls: type) -> type:
        entry = {
            "field": field,
            "default": default,
            "default_factory": default_factory,
        }
        # ``cls.__dict__.get`` (not ``getattr``) so the marker doesn't leak
        # through MRO when a subclass inherits a decorated parent — same
        # contract as ``_MIGRATE_FROM_ATTR``.
        existing = cls.__dict__.get(_BACKFILL_ATTR, ())
        setattr(cls, _BACKFILL_ATTR, (*existing, entry))
        return cls

    return _wrap


def _serde_event_classes(
    namespaces: Sequence[type], events: Sequence[type]
) -> list[type]:
    """Every event class a serde speaks for, in order, without repeats.

    The namespace walk plus any loose events passed directly — module-level
    ``IntegrationEvent``s and framework ``SystemEvent``s live outside every
    namespace, so nothing else reaches them.

    De-duplicated by class object: one class reached twice would have its
    ``@migrate_from`` migration collected twice, and a duplicated ``AddField``
    is rejected by ``_flatten_and_validate`` as a double fill.
    """
    seen: set[int] = set()
    out: list[type] = []
    for source in (
        (
            cls
            for ns in namespaces
            for cls in _iter_nested_events(ns, recurse_commands=True)
        ),
        events,
    ):
        for cls in source:
            if id(cls) not in seen:
                seen.add(id(cls))
                out.append(cls)
    return out


def _collect_decorated_migrations(
    namespaces: Sequence[type],
    events: Sequence[type] = (),
) -> tuple[
    tuple[Migration, ...],
    dict[tuple[str, str], tuple[str, str]],
    dict[tuple[str, str], type],
]:
    """Walk *namespaces* and assemble a :class:`Migration` per
    ``@migrate_from``-decorated class, plus an ``oldest_historic`` map for
    the encode-side ``legacy_write`` path.

    Internal — invoked by :class:`NamespaceAwareSerde` at construction
    with whatever ``namespaces=`` argument the caller passed. An empty
    iterable yields no migrations and an empty map, which is the right
    behaviour when the user opts out of decorator collection entirely.

    For each decorated class with history ``[h0, h1, ..., hn]`` (oldest to
    newest), emit a :class:`Migration` whose operations form the chain
    ``h0 → h1 → ... → hn → current``. ``_flatten_and_validate`` (called by
    the serde) collapses each chain into a single dict lookup.

    The ``oldest_historic`` map keys current ``(module, qualname)`` to the
    oldest historic identity (``history[0]``). The encoder consults it
    under ``legacy_write=True`` so an out-of-scope decorated class is NOT
    relabelled — bytes always go out under a name the read-side rename
    table knows how to migrate back.

    The third element is the serde's *scope*: every live ``(module,
    qualname)`` the namespace walk reached, mapped to the class object
    itself. Events in it revive directly with no migration.
    ``NamespaceAwareSerde`` stores it as the read path's first port of call
    (:func:`_resolve_identity`) and derives its ``revivable_identities`` /
    ``assert_all_baselined_cover`` key set from it; the walk already
    happens here so the map is free.
    """
    out: list[Migration] = []
    oldest_historic: dict[tuple[str, str], tuple[str, str]] = {}
    scope: dict[tuple[str, str], type] = {}
    for cls in _serde_event_classes(namespaces, events):
        current = (cls.__module__, cls.__qualname__)
        claimed = scope.setdefault(current, cls)
        if claimed is not cls:
            # Two lifetimes of one module share (module, qualname), so
            # last-wins would make revival depend on the order of a
            # sequence that reads as insignificant. Each lifetime gets
            # its own serde — EventGraph rejects the same mistake at
            # graph build.
            here, there = distinct_labels(claimed, cls)
            raise ValueError(
                f"Two classes passed to this serde (via namespaces= or "
                f"events=) claim the same event identity "
                f"{current[0]}.{current[1]}: {here} and {there}. Most "
                f"likely two engine lifetimes of one module — give each "
                f"lifetime its own serde, and its own EventGraph if this "
                f"came from EventGraph.from_namespaces, which builds one "
                f"for you."
            )
        # ``__dict__.get`` (not ``getattr``) — neither marker may leak
        # through MRO when a subclass inherits from a decorated parent.
        history = cls.__dict__.get(_MIGRATE_FROM_ATTR, ())
        backfills = cls.__dict__.get(_BACKFILL_ATTR, ())
        origin_backfills = cls.__dict__.get(_ORIGIN_BACKFILL_ATTR, ())
        if not history and not backfills and not origin_backfills:
            continue
        ops: list[Operation] = []
        if history:
            oldest_historic[current] = history[0]
            chain = [*history, current]
            for (old_mod, old_qn), (new_mod, new_qn) in itertools.pairwise(chain):
                ops.append(
                    RenameEvent(
                        old_module=old_mod,
                        old_qualname=old_qn,
                        new_module=new_mod,
                        new_qualname=new_qn,
                    )
                )
        # Class-global AddField keys on the CURRENT identity — the same
        # identity the rename chain (if any) resolves to — so renames and
        # back-fills on one class compose with no extra logic. ``AddField``
        # runs its own ``__post_init__`` validation (mutable-default guard).
        for bf in backfills:
            ops.append(
                AddField(
                    module=cls.__module__,
                    qualname=cls.__qualname__,
                    field=bf["field"],
                    default=bf["default"],
                    default_factory=bf["default_factory"],
                )
            )
        # Origin-scoped fills key on the HISTORIC identity the decorator
        # was declared with — applied before the rename on the read path,
        # so each collapsed origin can pin its own value.
        for (origin_module, origin_qualname), fill_fields in origin_backfills:
            for field_name, default in fill_fields.items():
                ops.append(
                    AddField(
                        module=origin_module,
                        qualname=origin_qualname,
                        field=field_name,
                        default=default,
                    )
                )
        out.append(
            Migration(
                name=f"{cls.__module__}:{cls.__qualname__}",
                operations=tuple(ops),
            )
        )
    return tuple(out), oldest_historic, scope


def replay_reducer(reducer: BaseReducer, events: Iterable[Event]) -> Any:
    """Rebuild a reducer's channel value from *events*.

    Use after a reducer's projection function or output shape changed
    between releases — the cached value in the checkpoint is stale, but
    events are the source of truth and can be replayed.

    Delegates to :meth:`BaseReducer.seed` so the reducer's default,
    namespace filter, and event-type predicate all apply uniformly.
    Events that don't match the reducer's filter are silently skipped,
    matching how the reducer would behave on a fresh run.

    The library does not iterate the checkpointer for you — saver
    semantics vary across MemorySaver / Sqlite / Postgres. Typical
    recipe::

        tup = checkpointer.get_tuple(config)
        events = tup.checkpoint["channel_values"][<your event-log channel>]
        rebuilt = replay_reducer(my_reducer, events)
        # write `rebuilt` back through the checkpointer's put API
    """
    return reducer.seed(list(events))
