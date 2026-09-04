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
    SplitEvent,
    TransformFields,
    backfill,
    migrate_from,
    replay_reducer,
    split_event,
    transform_fields,
)
from langgraph_events.serde.migrations.detect import (
    CoverageError,
    HandlerCoverageError,
    MigrationCoverageError,
)
from langgraph_events.serde.migrations.testing import (
    assert_all_baselined_cover,
    assert_all_baselined_handlers_cover,
    assert_all_baselined_resolve,
    assert_all_baselined_revive,
    assert_resume_recovers,
    synthesize_legacy_payload,
)

__all__ = [
    "AddField",
    "CoverageError",
    "HandlerCoverageError",
    "Migration",
    "MigrationCoverageError",
    "RenameEvent",
    "SplitEvent",
    "TransformFields",
    "assert_all_baselined_cover",
    "assert_all_baselined_handlers_cover",
    "assert_all_baselined_resolve",
    "assert_all_baselined_revive",
    "assert_resume_recovers",
    "backfill",
    "migrate_from",
    "replay_reducer",
    "split_event",
    "synthesize_legacy_payload",
    "transform_fields",
]
