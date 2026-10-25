"""Read migrations for the stored shapes of lge's own framework events.

Every ``NamespaceAwareSerde`` applies them, whatever ``namespaces=`` or
``events=`` the client passes. They never relabel a write, so the
``legacy_write`` guards ignore them.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from langgraph_events.serde.migrations._core import Migration

FRAMEWORK_MIGRATIONS: tuple[Migration, ...] = ()
