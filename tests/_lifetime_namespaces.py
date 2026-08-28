"""A consumer's namespace module, reloaded once per engine lifetime.

Module-level so the serde's identity walk resolves it, and so inline handler
return annotations resolve at runtime. See tests/test_lifetimes.py.
"""

from __future__ import annotations

from langgraph_events import (
    Command,
    DomainEvent,
    IntegrationEvent,
    Namespace,
    ScalarReducer,
)
from langgraph_events.serde.migrations import migrate_from


class Trading(Namespace):
    last_symbol = ScalarReducer(
        event_type=Command,
        fn=lambda e: getattr(e, "sym", None),
    )

    class Noted(DomainEvent):
        sym: str

    class Place(Command):
        sym: str

        class Placed(DomainEvent):
            sym: str

        def handle(self) -> Trading.Place.Placed:
            return Trading.Place.Placed(sym=self.sym)


class Ping(IntegrationEvent):
    """Module-level, outside any namespace — reaches the serde via events=."""

    sym: str


class Filled(Namespace):
    """A nested event whose decorator produces an AddField fill."""

    class Do(Command):
        @migrate_from("Filled.Ancient", backfill={"note": ""})
        class Done(DomainEvent):
            note: str = ""


class Audited(Namespace):
    """Reaches a graph only through `handlers=`, never through `namespaces=`."""

    class Logged(DomainEvent):
        sym: str
