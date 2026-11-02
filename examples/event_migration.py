"""Event migration smoke example.

Shows the loop for migrating a renamed event class:

1. Define the event under its **current** location, with ``@migrate_from``
   recording the historic qualname it used to live at.
2. Build the graph with ``EventGraph.from_namespaces`` and a checkpointer.
   ``from_namespaces`` scopes a migration-aware serde to the graph's
   namespaces, so an application never constructs ``NamespaceAwareSerde``
   for the read path.
3. Synthesize a checkpoint payload as a prior release would have produced
   it and show that it revives at the current class.

The ``legacy_write=True`` rolling-deploy path is shown last. It is opt-in
via an explicitly constructed serde. A user-supplied ``NamespaceAwareSerde``
is kept as it is, which is the auto-wiring opt-out.

Usage::

    uv run python examples/event_migration.py
"""

from __future__ import annotations

from langgraph.checkpoint.memory import MemorySaver

from langgraph_events import Command, DomainEvent, EventGraph, Namespace
from langgraph_events.serde import (
    NamespaceAwareSerde,
    migrate_from,
    synthesize_legacy_payload,
)


class Order(Namespace):
    """An order namespace, mid-refactor.

    ``Order.Placed`` used to be a sibling of ``Order.Place``. It is now
    nested inside the producing Command. The decorator records the historic
    qualname so old checkpoints still revive.
    """

    class Place(Command):
        note: str = ""

        @migrate_from("Order.Placed")
        class Placed(DomainEvent):
            note: str = ""

        def handle(self) -> Order.Place.Placed:
            return Order.Place.Placed(note=self.note)


def main() -> None:
    # The authoring step: @migrate_from on the class (above) plus a
    # checkpointer. from_namespaces scopes a migration-aware serde to the
    # graph's namespaces. No NamespaceAwareSerde import is needed here.
    graph = EventGraph.from_namespaces(Order, checkpointer=MemorySaver())
    log = graph.invoke(
        Order.Place(note="current"), config={"configurable": {"thread_id": "t-1"}}
    )
    assert log.has(Order.Place.Placed)

    # The same serde that from_namespaces wires, built here explicitly only
    # to exercise the read path directly. The graph uses it on every
    # checkpoint read, so a legacy payload revives on the next run.
    serde = NamespaceAwareSerde(namespaces=(Order,))

    legacy = synthesize_legacy_payload(
        Order.__module__, "Order.Placed", {"note": "from-old-release"}
    )
    revived = serde.loads_typed(legacy)
    print(f"Revived legacy payload as: {type(revived).__qualname__}")
    print(f"  note = {revived.note!r}")
    assert isinstance(revived, Order.Place.Placed)

    # Opt-in: rolling deploys. Construct the serde with legacy_write=True so
    # new pods encode under the OLDEST qualname and old pods (previous
    # release) can still revive during the rollout. from_namespaces keeps a
    # user-supplied NamespaceAwareSerde as it is.
    rolling = MemorySaver(
        serde=NamespaceAwareSerde(namespaces=(Order,), legacy_write=True)
    )
    rolling_graph = EventGraph.from_namespaces(Order, checkpointer=rolling)
    rolling_graph.invoke(
        Order.Place(note="rolling"), config={"configurable": {"thread_id": "t-2"}}
    )

    fresh = Order.Place.Placed(note="rolling-deploy")
    _, bytes_under_old_name = rolling.serde.dumps_typed(fresh)
    assert b"Order.Placed" in bytes_under_old_name
    assert b"Order.Place.Placed" not in bytes_under_old_name
    print(
        "\nlegacy_write=True encodes 'Order.Placed' instead of "
        "'Order.Place.Placed', so old pods revive it."
    )


if __name__ == "__main__":
    main()
