"""A consumer's namespace module, reloaded once per engine lifetime.

Module-level so the serde's identity walk resolves it, and so inline handler
return annotations resolve at runtime. See tests/test_lifetimes.py.
"""

from __future__ import annotations

from langgraph_events import Command, DomainEvent, Namespace, ScalarReducer


class Trading(Namespace):
    last_symbol = ScalarReducer(
        event_type=Command,
        fn=lambda e: getattr(e, "sym", None),
    )

    class Place(Command):
        sym: str

        class Placed(DomainEvent):
            sym: str

        def handle(self) -> Trading.Place.Placed:
            return Trading.Place.Placed(sym=self.sym)
