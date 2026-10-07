"""``Note``: an event stored once as ``OldNote``, before it had a ``tag``.

A module of its own: a ``@migrate_from`` class in ``conftest.py`` would make
every serde scoped to a ``conftest`` namespace warn that it cannot reach it.
"""

from langgraph_events import IntegrationEvent
from langgraph_events.serde import backfill, migrate_from


@migrate_from("OldNote")
@backfill("tag", default="legacy")
class Note(IntegrationEvent):
    text: str = ""
    tag: str = ""
