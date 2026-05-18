"""Event-identity migrations for ``NamespaceAwareSerde``.

Re-export of the public surface from :mod:`langgraph_events.serde.migrations`'s
implementation modules. Authors write::

    from langgraph_events.serde.migrations import (
        AddField,
        Migration,
        RenameEvent,
        migrate_from,
    )

Decorator-driven migrations (``@migrate_from``) are picked up
automatically — :class:`NamespaceAwareSerde` walks the namespace
registry at construction. Hand-written :class:`Migration` lists for
cross-module renames or composite operations flow through the serde's
``migrations=`` kwarg.
"""

from langgraph_events.serde.migrations._core import (
    AddField,
    Migration,
    RenameEvent,
    backfill,
    migrate_from,
    replay_reducer,
)
from langgraph_events.serde.migrations.testing import (
    assert_all_baselined_revive,
    synthesize_legacy_payload,
)

__all__ = [
    "AddField",
    "Migration",
    "RenameEvent",
    "assert_all_baselined_revive",
    "backfill",
    "migrate_from",
    "replay_reducer",
    "synthesize_legacy_payload",
]
