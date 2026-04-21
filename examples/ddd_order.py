"""DDD Order — langgraph-events demo.

Demonstrates the DDD-aligned taxonomy end-to-end:

- ``Aggregate`` — the ``Order`` namespace
- ``Command`` — ``Order.Place`` (with invariants) and ``Order.Ship`` (inline)
- ``DomainEvent`` — nested outcomes auto-unioned as ``Command.Outcomes``
- Handler forms — pick the shortest that fits:
    * **Inline** ``handle`` method on a command — colocated with the
      command, most DDD-idiomatic for trivial commands. ``Order.Ship.handle``
      below.
    * **``@on``** (bare) — reads the event type from the first parameter's
      annotation.
    * **``@on(Cmd, ...)``** — explicit form needed for ``invariants=``,
      ``raises=``, field matchers, or multi-event subscription. ``place``
      below uses the modifiers-only variant ``@on(invariants=...)``.
- ``Invariant`` — typed marker class (nested under the command for DDD
  locality). Declare predicates with ``invariants={Cls: lambda log: ...}``;
  the framework emits ``InvariantViolated(invariant=Cls(), ...)`` when the
  predicate returns False.
- Pinned reactions use the ``invariant=`` field matcher:
  ``@on(InvariantViolated, invariant=Cls)`` fires only for that specific
  invariant class's failures. A graph-construction-time drift check catches
  matchers that reference undeclared invariant classes.
- ``ScalarReducer`` as an aggregate class attribute — auto-named,
  auto-scoped to the aggregate's events, surfaced via ``log.reduced_state``.
- ``EventGraph.from_aggregates`` — auto-registers inline handlers and mixes
  them with extra external handlers.

Usage:
    python examples/ddd_order.py
"""

from __future__ import annotations

from langgraph_events import (
    SKIP,
    Aggregate,
    Command,
    DomainEvent,
    Event,
    EventGraph,
    IntegrationEvent,
    Invariant,
    InvariantViolated,
    ScalarReducer,
    on,
)

# ---------------------------------------------------------------------------
# Aggregate: Order
# ---------------------------------------------------------------------------


class Order(Aggregate):
    """The Order aggregate."""

    # ScalarReducer as a declarative class attribute. Auto-named
    # ``current_status``, auto-scoped to the aggregate's events. Picks a
    # status label from each event's class name; ``SKIP`` for events that
    # don't change status. Surfaced via ``log.reduced_state["current_status"]``.
    current_status = ScalarReducer(
        event_type=Event,
        fn=lambda e: {
            "Placed": "placed",
            "Rejected": "rejected",
            "Shipped": "shipped",
        }.get(type(e).__name__, SKIP),
    )

    class Place(Command):
        """Place an order for a given customer.

        No inline ``handle`` — this command needs an ``invariants=`` clause
        to guard against banned customers, so its handler lives externally
        as ``place`` below.
        """

        customer_id: str = ""

        class CustomerNotBanned(Invariant):
            """The placing customer must not be on the banned list."""

        class Placed(DomainEvent):
            """Order accepted."""

            order_id: str = ""

        class Rejected(DomainEvent):
            """Order rejected (e.g. by an invariant)."""

            reason: str = ""

    class Ship(Command):
        """Ship an accepted order.

        Simple enough to use the **inline** form — the command owns its
        own handler via ``handle``.
        """

        order_id: str = ""

        class Shipped(DomainEvent):
            """Shipment dispatched."""

            tracking: str = ""

        def handle(self) -> Order.Ship.Shipped:
            return Order.Ship.Shipped(tracking=f"track-{self.order_id}")


# Cross-cutting fact (not an Order event) used by the invariant below.
class CustomerBanned(IntegrationEvent):
    customer_id: str = ""


# ---------------------------------------------------------------------------
# External handlers — for what inline can't express
# ---------------------------------------------------------------------------


@on(
    invariants={
        Order.Place.CustomerNotBanned: lambda log: not log.has(CustomerBanned),
    },
)
def place(event: Order.Place) -> Order.Place.Placed:
    """External handler — the invariant is declared here, not inline.

    Uses ``@on(invariants=...)`` — no positional event type, inferred from
    the ``event:`` annotation. The invariant key is the typed class
    ``Order.Place.CustomerNotBanned``, so a reactor can pin on the same
    class with mypy/IDE support (and a typo triggers the graph-compile-time
    drift check rather than silently failing).
    """
    return Order.Place.Placed(order_id=f"order-for-{event.customer_id}")


@on(InvariantViolated, invariant=Order.Place.CustomerNotBanned)
def explain_rejection(event: InvariantViolated) -> Order.Place.Rejected:
    """Turn *this specific* invariant violation into a domain Rejected.

    Uses a pinned ``invariant=`` field matcher so the handler fires only
    for ``CustomerNotBanned`` failures; other invariants (none here yet,
    but we could add more) would fall through to their own handlers or to
    a catch-all ``@on(InvariantViolated)`` if registered.
    """
    return Order.Place.Rejected(reason=type(event.invariant).__name__)


# ---------------------------------------------------------------------------
# Graph — auto-registers Order's inline handlers; extras are appended
# ---------------------------------------------------------------------------


graph = EventGraph.from_aggregates(
    Order,
    handlers=[place, explain_rejection],
)


def main() -> None:
    print("=== Domain ===")
    print(graph.domain().text())
    print()

    print("=== Happy path (external handler + inline handler) ===")
    log = graph.invoke(
        [
            Order.Place(customer_id="alice"),
            Order.Ship(order_id="order-for-alice"),
        ]
    )
    for ev in log:
        print(f"  {type(ev).__qualname__}: {ev}")
    print()

    print("=== Invariant blocks placement (pinned reaction fires) ===")
    log2 = graph.invoke(
        [
            CustomerBanned(customer_id="bob"),
            Order.Place(customer_id="bob"),
        ]
    )
    for ev in log2:
        print(f"  {type(ev).__qualname__}: {ev}")
    print(f"  current_status: {Order.current_status.collect(list(log2))!r}")
    print()

    print("=== Reducer tracks the latest status on the happy path ===")
    print(f"  current_status: {Order.current_status.collect(list(log))!r}")


if __name__ == "__main__":
    main()
