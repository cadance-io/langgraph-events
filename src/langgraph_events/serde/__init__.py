"""Namespace-aware checkpoint serde — opt-in alternative to LangGraph's default.

Default ``JsonPlusSerializer`` keys dataclass identity by ``cls.__name__``,
which is leaf-only for nested events (``Persona.Approve.Approved.__name__``
is ``"Approved"``). Two namespaces that share a leaf event name therefore
collide on checkpoint round-trip. ``NamespaceAwareSerde`` keys ``Event``
subclasses by ``__qualname__`` instead, walking the attribute chain on the
defining module to revive the right class.

Usage::

    from langgraph.checkpoint.memory import MemorySaver
    from langgraph_events.serde import NamespaceAwareSerde

    graph = EventGraph(
        handlers=[...],
        checkpointer=MemorySaver(serde=NamespaceAwareSerde()),
    )

Pass ``migrations=`` to keep old checkpoints readable after a refactor —
see :mod:`langgraph_events.serde.migrations` and ``docs/event-migrations.md``.
"""

from langgraph_events.serde._jsonplus import NamespaceAwareSerde
from langgraph_events.serde.migrations import (
    Migration,
    assert_all_baselined_revive,
    backfill,
    migrate_from,
    replay_reducer,
    synthesize_legacy_payload,
)

# Raw RenameEvent / AddField are deliberately NOT re-exported here. The
# common path is decorator-first (@migrate_from) plus Migration.rename /
# Migration.add_field sugar. The raw operation constructors remain
# importable from ``langgraph_events.serde.migrations`` for the rare
# composite multi-op Migration — keeping the top-level surface small.
__all__ = [
    "Migration",
    "NamespaceAwareSerde",
    "assert_all_baselined_revive",
    "backfill",
    "migrate_from",
    "replay_reducer",
    "synthesize_legacy_payload",
]
