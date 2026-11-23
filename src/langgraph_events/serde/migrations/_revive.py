"""Revive one recorded shape through the real read path.

The gate's ``revive`` rule calls :func:`revival_failure` once per recorded
identity. :func:`synthesize_legacy_payload` is the public byte assembly
behind it.
"""

import annotationlib
import dataclasses
import inspect
import threading
from typing import TYPE_CHECKING, Any

import ormsgpack

# ``_option`` comes straight from its langgraph source (mypy treats
# ``langgraph.*`` as untyped, so no strict ``no_implicit_reexport``
# tripwire). ``EXT_NAMESPACE_AWARE_EVENT`` is defined in ``_jsonplus`` and
# importing that module first means its LangGraph-drift smoke fence runs
# before the line below, so the actionable error is still preserved.
from langgraph.checkpoint.serde.event_hooks import register_serde_event_listener
from langgraph.checkpoint.serde.jsonplus import EXT_CONSTRUCTOR_KW_ARGS, _option

from langgraph_events._event import Event
from langgraph_events._identity import Identity
from langgraph_events.serde._jsonplus import (
    _UNEXPECTED_KWARG_RE,
    EXT_NAMESPACE_AWARE_EVENT,
)
from langgraph_events.serde.migrations._core import (
    _is_pydantic_model,
    _resolve_identity,
    _resolve_rename,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from langgraph.checkpoint.serde.event_hooks import SerdeEvent

    from langgraph_events.serde._jsonplus import NamespaceAwareSerde


def synthesize_legacy_payload(
    module: str,
    qualname: str,
    kwargs: dict[str, Any],
) -> tuple[str, bytes]:
    """Synthesize the ``(format, bytes)`` tuple a prior release would have
    written for an Event at ``(module, qualname)`` with these ``kwargs``.

    The wire format hasn't changed across releases — these bytes are what
    :class:`~langgraph_events.serde.NamespaceAwareSerde` would emit today
    if the legacy class still existed. Pair with ``serde.loads_typed(...)``
    to assert revival succeeds under the current migration table.
    """
    inner = ormsgpack.packb((module, qualname, kwargs), option=_option)
    outer = ormsgpack.packb(
        ormsgpack.Ext(EXT_NAMESPACE_AWARE_EVENT, inner), option=_option
    )
    return ("msgpack", outer)


def revival_failure(
    serde: NamespaceAwareSerde,
    identity: tuple[str, str],
    recorded: frozenset[str] | None,
) -> str | None:
    """Why a stored payload of *identity* does not revive, or ``None``.

    A client identity revives a synthesized payload through the real read
    path: see :func:`_revive_failure`. An identity of this library only has
    to resolve to a live ``Event``. The library proves its own shapes in
    its own suite, and a placeholder cannot satisfy a library class that
    validates a value in ``__post_init__``.
    """
    if Identity(*identity).owned_by_library:
        return _resolve_failure(serde, identity)
    return _revive_failure(serde, identity, recorded)


def value_revival_failure(serde: NamespaceAwareSerde, live: type) -> str | None:
    """Why a stored record of the value class *live* cannot revive, or
    ``None``.

    LangGraph passes each dataclass field to the constructor, so the
    constructor signature must accept exactly those names. No constructor
    runs. The allowlist is read from LangGraph's own ``msgpack_blocked``
    event, from this thread only, because the listener registry is
    process-global. The probe record holds no payload, so LangGraph checks
    the allowlist and then stops before the constructor.
    """
    if dataclasses.is_dataclass(live) and not _is_pydantic_model(live):
        stored = dict.fromkeys(f.name for f in dataclasses.fields(live))
        try:
            signature = inspect.signature(
                live, annotation_format=annotationlib.Format.FORWARDREF
            )
            signature.bind(**stored)
        except TypeError as error:
            return (
                f"LangGraph stores every field of {live.__qualname__} and passes "
                f"each one to the constructor, which refuses them: {error}. Make "
                f"the constructor accept exactly the stored fields"
            )
    identity = (live.__module__, live.__name__)
    probe = ormsgpack.Ext(
        EXT_CONSTRUCTOR_KW_ARGS, ormsgpack.packb(identity, option=_option)
    )
    events: list[SerdeEvent] = []
    probing_thread = threading.get_ident()

    def keep_own_event(event: SerdeEvent) -> None:
        if threading.get_ident() == probing_thread:
            events.append(event)

    unregister = register_serde_event_listener(keep_own_event)
    try:
        serde.loads_typed(("msgpack", ormsgpack.packb(probe, option=_option)))
    finally:
        unregister()
    if not any(
        e["kind"] == "msgpack_blocked" and (e["module"], e["name"]) == identity
        for e in events
    ):
        return None
    return (
        f"LangGraph blocks {live.__qualname__}. "
        f"Add ({live.__module__!r}, {live.__name__!r}) to allowed_msgpack_modules"
    )


def _resolve_failure(
    serde: NamespaceAwareSerde, identity: tuple[str, str]
) -> str | None:
    """Resolves through *serde*'s own scope first, exactly as its read path
    does, so the answer matches the reader in both directions."""
    target_module, target_qualname = _resolve_rename(
        *identity, serde._tables.rename_table
    )
    try:
        obj = _resolve_identity(target_module, target_qualname, scope=serde._scope)
    except (ImportError, AttributeError) as exc:
        return f"{type(exc).__name__}: {exc}"
    if not (isinstance(obj, type) and issubclass(obj, Event)):
        return f"resolved to non-Event {obj!r}"
    return None


def _revive_failure(
    serde: NamespaceAwareSerde,
    identity: tuple[str, str],
    recorded: frozenset[str] | None,
) -> str | None:
    """The payload carries a placeholder for every required field of the
    live class, plus one for every recorded field the live class no longer
    accepts. Fields the migration table back-fills, origin-scoped or
    class-global, get no placeholder, so the rule proves the fills inject.

    Blind spot: an origin fill keyed mid-chain leaves EARLIER eras
    placeholder-masked. A real payload from such an era raises at read.
    """
    module, qualname = identity
    target_module, target_qualname = _resolve_rename(
        module, qualname, serde._tables.rename_table
    )
    backfilled = frozenset(
        op.field
        for op in (
            *serde._tables.origin_addfield_table.get(identity, ()),
            *serde._tables.addfield_table.get((target_module, target_qualname), ()),
        )
    )
    kwargs = _placeholder_kwargs(
        target_module,
        target_qualname,
        recorded,
        skip=backfilled,
        scope=serde._scope,
    )
    try:
        revived = serde.loads_typed(synthesize_legacy_payload(module, qualname, kwargs))
    except Exception as exc:  # report every failure, don't abort the sweep
        return f"{type(exc).__name__}: {exc}" + _failure_hint(exc)
    if not isinstance(revived, Event):
        return f"revived as non-Event {revived!r}"
    return None


def _placeholder_kwargs(
    module: str,
    qualname: str,
    recorded: frozenset[str] | None,
    *,
    skip: frozenset[str] = frozenset(),
    scope: Mapping[tuple[str, str], type] | None = None,
) -> dict[str, Any]:
    """``{name: None}`` for every field the rule must send to the live class
    at ``(module, qualname)``, except those in *skip*.

    A required init field gets a placeholder, so a healthy class
    constructs. A recorded field the live class no longer accepts gets one
    too: a stored payload still carries it. A recorded field the live class
    still accepts gets none. A ``None`` there could trip a ``__post_init__``
    validator that a real payload never trips. An unresolvable target
    yields ``{}``, so ``loads_typed`` reports the real failure.
    """
    try:
        obj = _resolve_identity(module, qualname, scope=scope)
    except ImportError, AttributeError:
        return {}
    if not dataclasses.is_dataclass(obj):
        return {}
    fields = dataclasses.fields(obj)
    accepted = {f.name for f in fields if f.init}
    required = {
        f.name
        for f in fields
        if f.init
        and f.default is dataclasses.MISSING
        and f.default_factory is dataclasses.MISSING
    }
    dropped = (recorded or frozenset()) - accepted
    return dict.fromkeys(sorted((required | dropped) - skip))


def _failure_hint(exc: Exception) -> str:
    """The cause that ties *exc* back to what the rule sent.

    A kwarg the live class rejected: the baseline recorded it, the class
    dropped it, and a stored payload still carries it. A transform or a
    select that raised: the rule sent a ``None`` placeholder for each
    required or dropped field, and the callable may need a real value.
    Empty for every other failure.
    """
    text = str(exc)
    if "TransformFields raised" in text or "SplitEvent raised" in text:
        return (
            " The gate sent a None placeholder for every required field and "
            "every recorded field the live class no longer accepts. Pin real "
            "values with synthesize_legacy_payload if the transform or the "
            "select reads them."
        )
    match = _UNEXPECTED_KWARG_RE.search(text)
    if match is None:
        return ""
    return f" The baseline recorded {match.group(1)!r}."
