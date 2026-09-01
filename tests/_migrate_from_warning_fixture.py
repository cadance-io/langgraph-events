"""Fixture for the UnreachableMigrationWarning suite in test_serde.py.

One live Namespace plus one @migrate_from-decorated sibling class that
is never passed to the serde alongside it — the exact shape that used
to silently drop a migration.
"""

from langgraph_events import Interrupted, Namespace
from langgraph_events.serde import migrate_from


class Gate(Namespace):
    class Live(Interrupted):
        pass


@migrate_from("Gate.OldName")
class OrphanTombstone(Interrupted):
    """Decorated but never passed via namespaces=/events= in the suite —
    the class the warning must name."""
