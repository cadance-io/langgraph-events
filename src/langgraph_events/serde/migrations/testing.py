"""User-facing test helpers for migration verification.

Public surface for the kind of "would a prior release's bytes revive"
check users want in CI. Sits on top of the read path — no logic of its
own beyond the wire-format byte assembly the serde already speaks, so
users never import ``_option`` / ``EXT_NAMESPACE_AWARE_EVENT`` themselves.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import TYPE_CHECKING, Any

import ormsgpack

if TYPE_CHECKING:
    from collections.abc import Callable

    from langgraph_events.serde._jsonplus import NamespaceAwareSerde

# ``_option`` comes straight from its langgraph source (mypy treats
# ``langgraph.*`` as untyped, so no strict ``no_implicit_reexport``
# tripwire). ``EXT_NAMESPACE_AWARE_EVENT`` is defined in ``_jsonplus`` and
# importing that module first means its LangGraph-drift smoke fence runs
# before the line below, so the actionable error is still preserved.
from langgraph.checkpoint.serde.jsonplus import _option

from langgraph_events._event import Event
from langgraph_events.serde._jsonplus import EXT_NAMESPACE_AWARE_EVENT
from langgraph_events.serde.migrations._core import (
    _resolve_identity,
    _resolve_rename,
)
from langgraph_events.serde.migrations.detect import (
    MigrationCoverageError,
    _load_baseline,
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


def _required_field_placeholders(module: str, qualname: str) -> dict[str, Any]:
    """``{name: None}`` for every required (no-default) field of the live
    class at ``(module, qualname)``.

    Events are frozen dataclasses with no construction-time validation, so
    ``None`` is a sufficient placeholder — the helper only proves the
    identity reaches a constructible live class, not field semantics. An
    unresolvable target yields ``{}`` so ``loads_typed`` surfaces the real
    coverage failure rather than a synthetic ``TypeError``.
    """
    try:
        obj = _resolve_identity(module, qualname)
    except (ImportError, AttributeError):
        return {}
    if not dataclasses.is_dataclass(obj):
        return {}
    return {
        f.name: None
        for f in dataclasses.fields(obj)
        if f.default is dataclasses.MISSING and f.default_factory is dataclasses.MISSING
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
        raise AssertionError(header + "\n  " + "\n  ".join(failures))


def _resolve_check(
    serde: NamespaceAwareSerde, module: str, qualname: str
) -> str | None:
    """Failure string if ``(module, qualname)`` no longer resolves (rename-
    aware) to a live ``Event`` subclass, else ``None``. Never constructs."""
    target_module, target_qualname = _resolve_rename(
        module, qualname, serde._rename_table
    )
    try:
        obj = _resolve_identity(target_module, target_qualname)
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
    actually built.
    """
    target_module, target_qualname = _resolve_rename(
        module, qualname, serde._rename_table
    )
    kwargs = _required_field_placeholders(target_module, target_qualname)
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
    normal required-field classes — explicit kwargs are only needed for
    genuine field-shape-drift checks (use :func:`synthesize_legacy_payload`
    directly there). Events with construction-time validation
    (``__post_init__``) that reject placeholders need
    :func:`assert_all_baselined_resolve` instead.

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
