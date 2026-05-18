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
from langgraph_events.serde.migrations.detect import _load_baseline


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


def assert_all_baselined_revive(
    serde: NamespaceAwareSerde, baseline_path: Path | str
) -> None:
    """Assert every identity in *baseline_path* revives through *serde*.

    Stronger than :meth:`NamespaceAwareSerde.assert_covers` (a
    set-membership check): this pushes a synthesized legacy payload for
    each baselined identity through the real ext-hook and asserts it
    revives to an ``Event``. Required fields of the resolved live class
    are filled with placeholders so a healthy migration table is never
    flagged for normal required-field classes — explicit kwargs are only
    needed for genuine field-shape-drift checks (use
    :func:`synthesize_legacy_payload` directly there).

    Zero per-event maintenance: a new ``@migrate_from`` plus a regenerated
    baseline is covered with no new test code. Raises ``AssertionError``
    naming every identity that failed to revive.
    """
    baseline = _load_baseline(Path(baseline_path))
    failures: list[str] = []
    for module, qualname in sorted(baseline):
        # Resolve historic identity to its live target via the same rule
        # the read path uses, so placeholder kwargs match the class
        # actually built.
        target_module, target_qualname = _resolve_rename(
            module, qualname, serde._rename_table
        )
        kwargs = _required_field_placeholders(target_module, target_qualname)
        try:
            revived = serde.loads_typed(
                synthesize_legacy_payload(module, qualname, kwargs)
            )
        except Exception as exc:  # report every failure, don't abort the sweep
            failures.append(f"{module}:{qualname} -> {type(exc).__name__}: {exc}")
            continue
        if not isinstance(revived, Event):
            failures.append(f"{module}:{qualname} revived as non-Event {revived!r}")
    if failures:
        raise AssertionError(
            "Baselined identities failed to revive through the serde:\n  "
            + "\n  ".join(failures)
        )
