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
    MigrationKey,
    MigrationSet,
    RenameEvent,
    SplitEvent,
    TransformFields,
    backfill,
    migrate_from,
    replay_reducer,
    split_event,
    transform_fields,
)
from langgraph_events.serde.migrations._gate import (
    AnonymousCallable,
    BaselineStale,
    Finding,
    GateError,
    GateReport,
    HandlerMissing,
    Invocation,
    SurvivedRelease,
    Uncovered,
    Unrevivable,
    check,
    hold_migration,
    update_baseline,
)
from langgraph_events.serde.migrations._live_set import (
    LiveSetOutcome,
    SubgraphTouched,
    WrongRelease,
    rewrite_live_set,
)
from langgraph_events.serde.migrations._target import (
    MigrationTarget,
    TargetError,
    ThreadKey,
)
from langgraph_events.serde.migrations.testing import (
    assert_migration_gate,
    assert_resume_recovers,
    synthesize_legacy_payload,
)

__all__ = [
    "AddField",
    "AnonymousCallable",
    "BaselineStale",
    "Finding",
    "GateError",
    "GateReport",
    "HandlerMissing",
    "Invocation",
    "LiveSetOutcome",
    "Migration",
    "MigrationKey",
    "MigrationSet",
    "MigrationTarget",
    "RenameEvent",
    "SplitEvent",
    "SubgraphTouched",
    "SurvivedRelease",
    "TargetError",
    "ThreadKey",
    "TransformFields",
    "Uncovered",
    "Unrevivable",
    "WrongRelease",
    "assert_migration_gate",
    "assert_resume_recovers",
    "backfill",
    "check",
    "hold_migration",
    "migrate_from",
    "replay_reducer",
    "rewrite_live_set",
    "split_event",
    "synthesize_legacy_payload",
    "transform_fields",
    "update_baseline",
]
