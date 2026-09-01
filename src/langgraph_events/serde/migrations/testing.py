"""User-facing test helpers for migration verification.

Public surface for the kind of "would a prior release's bytes revive"
check users want in CI. Sits on top of the read path — no logic of its
own beyond the wire-format byte assembly the serde already speaks, so
users never import ``_option`` / ``EXT_NAMESPACE_AWARE_EVENT`` themselves.
"""

from __future__ import annotations

import dataclasses
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any

import ormsgpack

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from langgraph_events import EventGraph
    from langgraph_events._event_log import EventLog
    from langgraph_events.serde._jsonplus import NamespaceAwareSerde

# ``_option`` comes straight from its langgraph source (mypy treats
# ``langgraph.*`` as untyped, so no strict ``no_implicit_reexport``
# tripwire). ``EXT_NAMESPACE_AWARE_EVENT`` is defined in ``_jsonplus`` and
# importing that module first means its LangGraph-drift smoke fence runs
# before the line below, so the actionable error is still preserved.
from langgraph.checkpoint.serde.jsonplus import _option

from langgraph_events._event import Event, Resumed
from langgraph_events.serde._jsonplus import EXT_NAMESPACE_AWARE_EVENT
from langgraph_events.serde.migrations._core import (
    _resolve_identity,
    _resolve_rename,
)
from langgraph_events.serde.migrations.detect import (
    MIGRATION_REMEDY,
    HandlerCoverageError,
    MigrationCoverageError,
    _load_baseline,
    _load_baseline_handlers,
)


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


def _required_field_placeholders(
    module: str,
    qualname: str,
    *,
    skip: frozenset[str] = frozenset(),
    scope: Mapping[tuple[str, str], type] | None = None,
) -> dict[str, Any]:
    """``{name: None}`` for every required (no-default) field of the live
    class at ``(module, qualname)``, except those in *skip*.

    Reads the fields off whichever class the read path would build — hence
    *scope*, the serde's ``namespaces=`` map, ahead of the import walk.

    The helper only proves the identity reaches a constructible live
    class, not field semantics — ``None`` placeholders suffice unless the
    class validates in ``__post_init__`` (those need the resolve gate, or
    a back-fill covering the validated field). An unresolvable target
    yields ``{}`` so ``loads_typed`` surfaces the real coverage failure
    rather than a synthetic ``TypeError``. *skip* names fields the
    migration table back-fills itself: a placeholder there would mask the
    fill (``setdefault`` never overwrites), so the gate leaves them to the
    read path and actually exercises the injection.
    """
    try:
        obj = _resolve_identity(module, qualname, scope=scope)
    except (ImportError, AttributeError):
        return {}
    if not dataclasses.is_dataclass(obj):
        return {}
    return {
        f.name: None
        for f in dataclasses.fields(obj)
        if f.name not in skip
        and f.default is dataclasses.MISSING
        and f.default_factory is dataclasses.MISSING
    }


def assert_all_baselined_cover(
    serde: NamespaceAwareSerde, baseline_path: Path | str
) -> None:
    """Assert every identity in *baseline_path* is reachable by *serde*.

    Weakest of the three gates: a set-membership check over
    :meth:`NamespaceAwareSerde.revivable_identities` — every baselined
    identity must be either still live in this serde's ``namespaces=`` or
    covered by a rename migration. Neither resolves nor constructs the class.
    Raises :class:`MigrationCoverageError` (an ``AssertionError``) whose
    ``uncovered`` attribute lists the offending identities.
    """
    baseline = _load_baseline(Path(baseline_path))
    missing = tuple(sorted(baseline - serde.revivable_identities()))
    if missing:
        raise MigrationCoverageError(missing)


def assert_all_baselined_handlers_cover(
    graph: EventGraph, baseline_path: Path | str
) -> None:
    """Assert every baselined handler node name is still reachable on *graph*.

    The handler analog of :func:`assert_all_baselined_cover`: a static
    set-membership check that every handler node name in *baseline_path* is
    either a live handler node or covered by an ``@on(previously=...)`` alias.
    A name that is neither means an interrupted checkpoint paused at that node
    would silently drop on resume after a rename/removal — so this raises
    :class:`HandlerCoverageError` (a :class:`CoverageError`/``AssertionError``)
    naming the lost handler(s). Takes the ``EventGraph`` (handler identity is a
    graph concern), not the serde. A pre-v2 baseline records no handlers and
    passes trivially.
    """
    baselined = _load_baseline_handlers(Path(baseline_path))
    reachable = {meta.node_name for meta in graph._handler_metas}
    reachable |= {
        alias for meta in graph._handler_metas for alias in meta.previous_names
    }
    missing = tuple(sorted(baselined - reachable))
    if missing:
        raise HandlerCoverageError(missing)


def assert_resume_recovers(
    before: EventGraph,
    after: EventGraph,
    *,
    seed: Event | list[Event],
    resume_with: Event,
    thread_id: str | None = None,
) -> EventLog:
    """Assert a thread paused inside a handler on *before* resumes on *after*.

    The behavioral handler analog of :func:`assert_all_baselined_revive`:
    instead of a static name check it exercises the real interrupt→resume path.
    Invokes *before* with *seed* (which must pause via an ``Interrupted``),
    then resumes *after* with *resume_with* on the same checkpoint and asserts
    recovery actually happened — a ``Resumed`` is emitted, which a silent drop
    or a ``halt`` would not produce. Use it to prove an ``@on(previously=...)``
    rename keeps old checkpoints resumable. Returns the post-resume log for
    further assertions.

    *before* and *after* must be constructed with the **same** checkpointer
    instance so the paused checkpoint survives the rebuild.
    """
    if before._checkpointer is None or before._checkpointer is not after._checkpointer:
        raise ValueError(
            "assert_resume_recovers: `before` and `after` must share one "
            "checkpointer instance so the paused checkpoint survives the "
            "rename — e.g. EventGraph(..., checkpointer=saver) for both."
        )
    tid = thread_id or f"resume-recovers-{uuid.uuid4().hex}"
    config = {"configurable": {"thread_id": tid}}
    before.invoke(seed, config=config)
    if not before.get_state(config).is_interrupted:
        raise AssertionError(
            "assert_resume_recovers: `seed` did not pause `before` — it must "
            "trigger an Interrupted so there is a paused checkpoint to resume."
        )
    log = after.resume(resume_with, config=config)
    if log.latest(Resumed) is None:
        raise AssertionError(
            "assert_resume_recovers: resume did not recover the paused thread "
            "(no Resumed emitted) — the handler rename is not covered by "
            "@on(previously=...)."
        )
    return log


def _assert_baselined(
    serde: NamespaceAwareSerde,
    baseline_path: Path | str,
    header: str,
    check: Callable[[NamespaceAwareSerde, str, str], str | None],
) -> None:
    """Sweep *check* over every baselined identity; raise ``AssertionError``.

    Shared spine of :func:`assert_all_baselined_resolve` and
    :func:`assert_all_baselined_revive`: load → sweep (collecting *every*
    failure, never aborting early) → raise *header* plus one indented line
    per failure. ``cover`` is a set-diff rather than a per-identity sweep, so
    it stays separate.
    """
    baseline = _load_baseline(Path(baseline_path))
    failures = [
        failure
        for module, qualname in sorted(baseline)
        if (failure := check(serde, module, qualname)) is not None
    ]
    if failures:
        raise AssertionError(
            header + "\n  " + "\n  ".join(failures) + "\n" + MIGRATION_REMEDY
        )


def _resolve_check(
    serde: NamespaceAwareSerde, module: str, qualname: str
) -> str | None:
    """Failure string if ``(module, qualname)`` no longer resolves (rename-
    aware) to a live ``Event`` subclass, else ``None``. Never constructs.

    Resolves through *serde*'s own scope first, exactly as its read path
    does — the gate asks "would a checkpoint revive", and answering it with
    a different resolution rule than the reader uses makes it wrong in both
    directions."""
    target_module, target_qualname = _resolve_rename(
        module, qualname, serde._rename_table
    )
    try:
        obj = _resolve_identity(target_module, target_qualname, scope=serde._scope)
    except (ImportError, AttributeError) as exc:
        return f"{module}:{qualname} -> {type(exc).__name__}: {exc}"
    if not (isinstance(obj, type) and issubclass(obj, Event)):
        return f"{module}:{qualname} resolved to non-Event {obj!r}"
    return None


def _revive_check(serde: NamespaceAwareSerde, module: str, qualname: str) -> str | None:
    """Failure string if a synthesized legacy payload for ``(module,
    qualname)`` does not revive to an ``Event`` through *serde*, else ``None``.

    Resolve the historic identity to its live target via the same rule the
    read path uses, so required-field placeholder kwargs match the class
    actually built. Fields the migration table back-fills — origin-scoped
    (keyed on the baselined identity) or class-global (keyed on the
    resolved target) — get no placeholder, so the gate proves the fills
    actually inject.
    """
    target_module, target_qualname = _resolve_rename(
        module, qualname, serde._rename_table
    )
    backfilled = frozenset(
        op.field
        for op in (
            *serde._origin_addfield_table.get((module, qualname), ()),
            *serde._addfield_table.get((target_module, target_qualname), ()),
        )
    )
    kwargs = _required_field_placeholders(
        target_module, target_qualname, skip=backfilled, scope=serde._scope
    )
    try:
        revived = serde.loads_typed(synthesize_legacy_payload(module, qualname, kwargs))
    except Exception as exc:  # report every failure, don't abort the sweep
        return f"{module}:{qualname} -> {type(exc).__name__}: {exc}"
    if not isinstance(revived, Event):
        return f"{module}:{qualname} revived as non-Event {revived!r}"
    return None


def assert_all_baselined_revive(
    serde: NamespaceAwareSerde, baseline_path: Path | str
) -> None:
    """Assert every identity in *baseline_path* revives through *serde*.

    Strongest of the three gates: pushes a synthesized legacy payload for
    each baselined identity through the real ext-hook and asserts it revives
    to an ``Event``. Required fields of the resolved live class are filled
    with placeholders so a healthy migration table is never flagged for
    normal required-field classes — except fields the table itself
    back-fills (origin-scoped or class-global), which get no placeholder so
    the gate proves the injection actually happens. Explicit kwargs are only
    needed for genuine field-shape-drift checks (use
    :func:`synthesize_legacy_payload` directly there). Events whose
    ``__post_init__`` rejects placeholders on NON-back-filled fields need
    :func:`assert_all_baselined_resolve` instead; validation on a
    back-filled field sees the real injected value and passes here.

    Blind spot: an origin-scoped fill keyed mid-chain leaves EARLIER eras
    placeholder-masked — their baselined identities have no applicable
    fill, so the field falls back to a ``None`` placeholder and the gate
    passes, while a real payload from that era (which never carried the
    field) raises at read. Cover "every era" with class-global
    :func:`~langgraph_events.serde.migrations.backfill`, or pin the
    specific era with :func:`synthesize_legacy_payload`.

    Zero per-event maintenance: a new ``@migrate_from`` plus a regenerated
    baseline is covered with no new test code. Raises ``AssertionError``
    naming every identity that failed to revive.
    """
    _assert_baselined(
        serde,
        baseline_path,
        "Baselined identities failed to revive through the serde:",
        _revive_check,
    )


def assert_all_baselined_resolve(
    serde: NamespaceAwareSerde, baseline_path: Path | str
) -> None:
    """Assert every identity in *baseline_path* resolves to a live Event class.

    Rename-aware reachability check that never constructs the class — unlike
    :func:`assert_all_baselined_revive` it synthesizes no payload and invokes
    no ``__init__``/``__post_init__``. Use when the baseline includes events
    with construction-time validation, framework ``SystemEvents``, or
    module-level ``IntegrationEvents`` that ``revive`` would trip or ``cover``
    would miss. Raises ``AssertionError`` naming every identity that no longer
    resolves to an ``Event`` subclass.
    """
    _assert_baselined(
        serde,
        baseline_path,
        "Baselined identities no longer resolve to a live Event:",
        _resolve_check,
    )
