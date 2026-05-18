"""Event-identity migrations for ``NamespaceAwareSerde``.

Migrations are ordered, idempotent rewrites of the encoded
``(module, qualname, kwargs)`` tuple. They run inside the serde's
ext-hook between unpack and revive. The wire format is unchanged from
prior library versions — this is a read-side affordance only. See #70.
"""

from __future__ import annotations

import importlib
import itertools
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from langgraph_events._event import Event, _iter_nested_events

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Sequence

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
    doesn't touch unrelated payloads. Applies AFTER any matching
    :class:`RenameEvent` — so ``module``/``qualname`` should name the
    CURRENT (post-rename) class.

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
        being added. The target identity is the live (post-rename) class,
        so pass it as ``target=<class>`` for refactor-safety;
        ``module``/``qualname`` remain for the case where the class can't
        be imported at authoring time. Same convention as :class:`AddField`
        — moving between the sugar and the raw form needs no kwarg rename.
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
) -> tuple[
    dict[tuple[str, str], tuple[str, str]],
    dict[tuple[str, str], tuple[AddField, ...]],
]:
    """Collapse all operations into per-purpose lookup tables.

    Rename ops fold into a direct historic→final map (chain ``A→B→C``
    becomes ``A→C`` and ``B→C``) so reads do a single dict lookup
    regardless of chain depth. AddField ops are bucketed by their target
    identity (post-rename).

    Validation is intentionally strict — every error here would otherwise
    surface as a ``ValueError`` on first production read, which is the
    worst possible time to discover it.

    Raises:
        ValueError: on duplicate rename sources, dead-end chains, cycles,
            shadowing of live classes, or AddField targets that don't
            resolve.
    """
    rename_edges: dict[tuple[str, str], tuple[str, str]] = {}
    # Remember which migration each edge came from so the duplicate-source
    # diagnostic can name the conflicting migrations, not just their targets.
    edge_origin: dict[tuple[str, str], str] = {}
    addfield_ops: list[AddField] = []
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
                addfield_ops.append(op)
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
        rename_table[start] = _resolve_chain_terminus(start, rename_edges)

    addfield_table: dict[tuple[str, str], list[AddField]] = {}
    for op in addfield_ops:
        target = (op.module, op.qualname)
        if not _resolves(*target):
            raise ValueError(
                f"AddField target {op.module}:{op.qualname!r} does not "
                f"resolve to a currently importable class. Either the "
                f"target was deleted after the migration was authored, "
                f"or the module/qualname has a typo."
            )
        addfield_table.setdefault(target, []).append(op)

    return rename_table, {k: tuple(v) for k, v in addfield_table.items()}


def _resolve_chain_terminus(
    start: tuple[str, str],
    rename_edges: dict[tuple[str, str], tuple[str, str]],
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
    if not _resolves(*current):
        raise ValueError(
            f"Migration chain from {start[0]}:{start[1]!r} terminates at "
            f"{current[0]}:{current[1]!r}, which does not resolve to a "
            f"currently importable class. Either the chain target was "
            f"renamed/deleted after the migration was authored, or the "
            f"new module/qualname has a typo."
        )
    # Reject migrations whose old name shadows a class that still exists
    # — would silently rewrite live payloads on read.
    if _resolves(*start):
        raise ValueError(
            f"Migration source {start[0]}:{start[1]!r} resolves to a "
            f"currently-live class. A rename whose old name is still "
            f"importable would shadow the live class on read. Remove "
            f"the old class definition before declaring this migration."
        )
    return current


def _migration_label(name: str) -> str:
    """Render a migration's name for diagnostics, with a placeholder for blanks."""
    return f"migration {name!r}" if name else "an unnamed migration"


def _resolve_identity(module: str, qualname: str) -> Any:
    """Import *module* and walk *qualname* to the live object.

    Raises ``ImportError`` if the module is gone, ``AttributeError`` if the
    qualname no longer resolves — callers decide which to treat as
    "missing". Single source of truth for the identity → live-class walk
    shared by the read path, ``_resolves``, and the test helpers.
    """
    obj: Any = importlib.import_module(module)
    for part in qualname.split("."):
        obj = getattr(obj, part)
    return obj


def _resolves(module: str, qualname: str) -> bool:
    """``True`` iff ``module`` imports and ``qualname`` walks to an attribute."""
    try:
        _resolve_identity(module, qualname)
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


def _apply_identity_migrations(
    module: str,
    qualname: str,
    kwargs: dict[str, Any],
    rename_table: dict[tuple[str, str], tuple[str, str]],
    addfield_table: dict[tuple[str, str], tuple[AddField, ...]],
) -> tuple[str, str]:
    """Resolve *module*/*qualname* through the rename table, then inject
    AddField defaults into *kwargs* in place. Returns the post-rename
    identity.

    AddField is keyed on the POST-rename identity so renames and field
    additions compose cleanly. ``setdefault`` is idempotent — a value
    already present is preserved. Mutable defaults go through
    ``default_factory`` so each migrated payload gets its own instance.
    """
    module, qualname = _resolve_rename(module, qualname, rename_table)
    for op in addfield_table.get((module, qualname), ()):
        if op.default_factory is not None:
            if op.field not in kwargs:
                kwargs[op.field] = op.default_factory()
        else:
            kwargs.setdefault(op.field, op.default)
    return module, qualname


_MIGRATE_FROM_ATTR = "__lge_migrate_from__"


def migrate_from(
    *old_qualnames: str, in_module: str | None = None
) -> Callable[[type], type]:
    """Mark that this ``Event`` class formerly lived at ``old_qualnames``.

    Multiple positional args declare a chain in temporal order — oldest
    first. ``in_module`` defaults to the decorated class's ``__module__``
    and applies to all historic qualnames. Use the manual :class:`Migration`
    list directly for cross-module relocations or per-step module changes.

    Metadata is stashed on the class as ``__lge_migrate_from__``.
    :class:`NamespaceAwareSerde` walks ``_NAMESPACE_REGISTRY`` at
    construction and assembles a :class:`Migration` per decorated class
    automatically — no separate collection step is required.
    """
    if not old_qualnames:
        raise ValueError("@migrate_from requires at least one historic qualname.")

    def _wrap(cls: type) -> type:
        module = in_module if in_module is not None else cls.__module__
        history = tuple((module, q) for q in old_qualnames)
        # ``cls.__dict__.get`` (not ``getattr``) so the marker doesn't leak
        # through MRO when a class inherits from a decorated parent — see
        # the read-side ``getattr`` companion in ``_jsonplus._make_default``.
        existing = cls.__dict__.get(_MIGRATE_FROM_ATTR, ())
        # Python applies decorators bottom-up, so the bottom decorator runs
        # first and its qualnames are the OLDEST in the chain. Place
        # ``existing`` (from the inner, earlier-applied decorator) ahead of
        # ``history`` (from the outer one). Aligns stacked decorators with
        # the multi-arg form ``@migrate_from("A", "B")`` where A is oldest.
        setattr(cls, _MIGRATE_FROM_ATTR, existing + history)
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
    resulting (current) identity.
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


def _collect_decorated_migrations(
    namespaces: Sequence[type],
) -> tuple[
    tuple[Migration, ...],
    dict[tuple[str, str], tuple[str, str]],
    frozenset[tuple[str, str]],
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

    The third element is the frozenset of every live ``(module, qualname)``
    the namespace walk reached — events that revive directly with no
    migration. ``NamespaceAwareSerde`` stores it for ``assert_covers`` /
    ``revivable_identities``; the walk already happens here so the set is
    free.
    """
    out: list[Migration] = []
    oldest_historic: dict[tuple[str, str], tuple[str, str]] = {}
    live: set[tuple[str, str]] = set()
    for namespace_cls in namespaces:
        for cls in _iter_nested_events(namespace_cls, recurse_commands=True):
            current = (cls.__module__, cls.__qualname__)
            live.add(current)
            # ``__dict__.get`` (not ``getattr``) — neither marker may leak
            # through MRO when a subclass inherits from a decorated parent.
            history = cls.__dict__.get(_MIGRATE_FROM_ATTR, ())
            backfills = cls.__dict__.get(_BACKFILL_ATTR, ())
            if not history and not backfills:
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
            # AddField keys on the CURRENT identity — the same identity the
            # rename chain (if any) resolves to — so renames and back-fills
            # on one class compose with no extra logic. ``AddField`` runs
            # its own ``__post_init__`` validation (mutable-default guard).
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
            out.append(
                Migration(
                    name=f"{cls.__module__}:{cls.__qualname__}",
                    operations=tuple(ops),
                )
            )
    return tuple(out), oldest_historic, frozenset(live)


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
