"""Read migrations for the stored shapes of lge's own framework events.

Every ``NamespaceAwareSerde`` applies them, whatever ``namespaces=`` or
``events=`` the client passes. They are not relabelled on write: a framework
event is always written under its own live identity. So the ``legacy_write``
guards ignore them, and an old release cannot read a new framework type.

Each migration holds one operation. Its name is
``lge-<major>.<minor>-<source>-<kind>``:

- ``<major>.<minor>`` is the first release that reads the new shape.
- ``<source>`` is the source class qualname in kebab case, for example
  ``handler-raised`` for ``HandlerRaised``.
- ``<kind>`` is ``rename``, ``fill``, ``transform`` or ``split``.

``tests/test_serde_framework.py`` enforces the rule. It also requires a
stored fixture for the source identity of each operation.
"""

from __future__ import annotations

from typing import Any

from langgraph_events._event import (
    HandlerAbandonedOnDeadline,
    HandlerRaised,
    RunPaused,
    RunPausedOnDeadline,
)
from langgraph_events.serde.migrations._core import Migration


def _split_handler_raised(
    kw: dict[str, Any],
) -> tuple[type, dict[str, Any]] | None:
    # 0.40.0 and earlier wrote abandoned_for_deadline on every HandlerRaised.
    # This release writes no such key, so its own payloads keep their class.
    if "abandoned_for_deadline" not in kw:
        return None
    kept = {k: v for k, v in kw.items() if k != "abandoned_for_deadline"}
    if kw["abandoned_for_deadline"]:
        return HandlerAbandonedOnDeadline, kept
    return HandlerRaised, kept


def _split_run_paused(kw: dict[str, Any]) -> tuple[type, dict[str, Any]]:
    # 0.40.0 and earlier paused only on the deadline.
    return RunPausedOnDeadline, kw


FRAMEWORK_MIGRATIONS: tuple[Migration, ...] = (
    Migration.split_event(
        "lge-0.41-handler-raised-split",
        source=HandlerRaised,
        select=_split_handler_raised,
        targets=(HandlerAbandonedOnDeadline, HandlerRaised),
    ),
    Migration.split_event(
        "lge-0.41-run-paused-split",
        source=RunPaused,
        select=_split_run_paused,
        targets=(RunPausedOnDeadline,),
    ),
)
