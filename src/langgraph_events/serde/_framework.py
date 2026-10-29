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

Before 0.6.0, LangGraph's own serializer wrote each event as a constructor
record. ``NamespaceAwareSerde`` reads such a record of an lge event through
these migrations too.
"""

from __future__ import annotations

import importlib.util
from typing import Any

from langgraph_events._event import (
    Halted,
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


def _drop_halted_reason(kw: dict[str, Any]) -> dict[str, Any]:
    # 0.1.x wrote a free-text reason on Halted. 0.2.0 replaced it with typed
    # subclasses, and LangGraph's own serializer wrote those payloads.
    kw.pop("reason", None)
    return kw


# 0.5.x and earlier defined FrontendToolCallRequested in this module. Its
# class needs the agui extra, so the rename exists only when the extra does:
# a rename must end on a class that imports.
_AGUI_RENAMES: tuple[Migration, ...] = (
    (
        Migration.rename(
            "lge-0.42-frontend-tool-call-requested-rename",
            old_module="langgraph_events._event",
            old_qualname="FrontendToolCallRequested",
            new_module="langgraph_events.agui._events",
            new_qualname="FrontendToolCallRequested",
        ),
    )
    if importlib.util.find_spec("ag_ui") is not None
    else ()
)

FRAMEWORK_MIGRATIONS: tuple[Migration, ...] = (
    *_AGUI_RENAMES,
    Migration.transform_fields(
        "lge-0.42-halted-transform", target=Halted, transform=_drop_halted_reason
    ),
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
