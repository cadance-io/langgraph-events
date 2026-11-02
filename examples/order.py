"""DDD Order — langgraph-events demo.

Shows three things:

- ``Invariant`` — a typed marker class nested under the command. Declare
  predicates in a class-level ``invariants`` dict. The library emits
  ``InvariantViolated(invariant=Cls(), ...)`` when a predicate returns False.
- Pinned handlers — ``@on(InvariantViolated, invariant=Cls)`` fires only for
  that invariant class. Recovery handlers emit namespace-level events
  (``Order.Rejected``), never Command-private outcomes.
- ``ScalarReducer`` as a namespace class attribute — auto-named, auto-scoped
  to the namespace's events, surfaced via ``graph.reflect(log).state()``.

Usage:
    python examples/order.py
"""

from __future__ import annotations

from typing import ClassVar

from langgraph_events import (
    SKIP,
    Command,
    DomainEvent,
    Event,
    EventGraph,
    IntegrationEvent,
    Invariant,
    InvariantViolated,
    Namespace,
    ScalarReducer,
    on,
)

# ---------------------------------------------------------------------------
# Namespace: Order
# ---------------------------------------------------------------------------


# Cross-cutting fact (not an Order event) used by the invariant below.
class CustomerBanned(IntegrationEvent):
    customer_id: str = ""


_ORDER_LIMIT = 100


class Order(Namespace):
    """The Order namespace."""

    # ScalarReducer as a declarative class attribute. Auto-named
    # ``current_status``, auto-scoped to the namespace's events. Picks a
    # status label from each event's class name. ``SKIP`` for events that
    # do not change status. Surfaced via
    # ``graph.reflect(log).state()["current_status"]``.
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

        The inline ``place`` handler produces ``Placed``. ``invariants`` are
        declared as a class-level attribute. Failure modes (banned customer,
        over-limit) are surfaced as ``InvariantViolated``. Namespace-level
        recovery handlers translate those into ``Order.Rejected``.
        """

        customer_id: str = ""
        amount: int = 0

        class CustomerNotBanned(Invariant):
            """The placing customer must not be on the banned list.

            Pure **pre-check** invariant. The predicate reads
            ``CustomerBanned`` events that have *already* been committed.
            Its truth value does not depend on what this handler emits.
            """

        class OrderTotalWithinLimit(Invariant):
            """Cumulative placed amount must stay under the daily limit.

            Pure **post-check** invariant. The predicate sums committed
            ``Placed`` events. The *current* handler's about-to-be-emitted
            ``Placed`` is what pushes the total over the limit, so only the
            post-command check (against log + emitted buffer) can catch it.
            The pre-check passes. The post-check rolls back and emits
            ``InvariantViolated`` carrying the dropped event in
            ``would_emit``.
            """

        class Placed(DomainEvent):
            """Order accepted."""

            order_id: str = ""
            amount: int = 0

        invariants: ClassVar = {
            CustomerNotBanned: lambda log: not log.has(CustomerBanned),
            OrderTotalWithinLimit: lambda log: (
                sum(e.amount for e in log.filter(Order.Place.Placed)) < _ORDER_LIMIT
            ),
        }

        def place(self) -> Order.Place.Placed:
            return Order.Place.Placed(
                order_id=f"order-for-{self.customer_id}", amount=self.amount
            )

    class Rejected(DomainEvent):
        """Order rejected (e.g. by an invariant).

        Sibling to ``Place``, not nested. Emitted by recovery handlers,
        never from ``Place.place()`` itself.
        """

        reason: str = ""

    class Ship(Command):
        """Ship an accepted order. Inline handler named ``ship``."""

        order_id: str = ""

        class Shipped(DomainEvent):
            """Shipment dispatched."""

            tracking: str = ""

        def ship(self) -> Order.Ship.Shipped:
            return Order.Ship.Shipped(tracking=f"track-{self.order_id}")


# ---------------------------------------------------------------------------
# Recovery handlers: emit namespace-level Order.Rejected
# ---------------------------------------------------------------------------


@on(InvariantViolated, invariant=Order.Place.CustomerNotBanned)
def explain_banned(event: InvariantViolated) -> Order.Rejected:
    """Turn a ``CustomerNotBanned`` violation into ``Order.Rejected``."""
    return Order.Rejected(reason=type(event.invariant).__name__)


@on(InvariantViolated, invariant=Order.Place.OrderTotalWithinLimit)
def explain_over_limit(event: InvariantViolated) -> Order.Rejected:
    """Turn an ``OrderTotalWithinLimit`` post-check violation into a
    ``Rejected``. ``would_emit`` carries the rolled-back ``Placed``. Its
    amount goes into the rejection reason for diagnostics.
    """
    rolled_back = event.would_emit[0] if event.would_emit else None
    amount = getattr(rolled_back, "amount", "?")
    return Order.Rejected(reason=f"OrderTotalWithinLimit (would emit amount={amount})")


# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------


graph = EventGraph.from_namespaces(
    Order,
    handlers=[explain_banned, explain_over_limit],
)


def main() -> None:
    print("=== Namespace ===")
    print(graph.namespaces().text())
    print()

    print("=== Happy path ===")
    log = graph.invoke(
        [
            Order.Place(customer_id="alice", amount=30),
            Order.Ship(order_id="order-for-alice"),
        ]
    )
    for ev in log:
        print(f"  {type(ev).__qualname__}: {ev}")
    print()

    print("=== Pre-check: CustomerNotBanned blocks placement ===")
    log_banned = graph.invoke(
        [
            CustomerBanned(customer_id="bob"),
            Order.Place(customer_id="bob", amount=10),
        ]
    )
    for ev in log_banned:
        print(f"  {type(ev).__qualname__}: {ev}")
    print()

    print("=== Post-check: OrderTotalWithinLimit rolls back the Placed ===")
    log_over = graph.invoke(
        [
            Order.Place(customer_id="carol", amount=60),
            Order.Place(customer_id="dave", amount=60),
        ]
    )
    for ev in log_over:
        print(f"  {type(ev).__qualname__}: {ev}")
    print(f"  current_status: {Order.current_status.collect(list(log_over))!r}")
    print()

    print("=== Reducer tracks the latest status on the happy path ===")
    print(f"  current_status: {Order.current_status.collect(list(log))!r}")


if __name__ == "__main__":
    main()
