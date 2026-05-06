"""DDD Order — langgraph-events demo.

Demonstrates the DDD-aligned taxonomy end-to-end:

- ``Namespace`` — the ``Order`` namespace
- ``Command`` — ``Order.Place`` (with invariants) and ``Order.Ship`` (inline)
- ``DomainEvent`` — outcomes nested under their owning Command (private to
  that Command's ``handle()``) or sibling to the Namespace (free-standing
  facts, e.g. ``Order.Rejected`` below — emitted by recovery reactors).
- Handler form — every Command's outcome must come from its inline
  ``handle()``. Use class-level ``invariants`` / ``raises`` attributes when
  the handler needs them.
- ``Invariant`` — typed marker class (nested under the command for DDD
  locality). Declare predicates as a class-level ``invariants`` dict; the
  framework emits ``InvariantViolated(invariant=Cls(), ...)`` when the
  predicate returns False.
- Pinned reactions use the ``invariant=`` field matcher:
  ``@on(InvariantViolated, invariant=Cls)`` fires only for that specific
  invariant class's failures. Recovery reactors emit namespace-level events
  (``Order.Rejected``), never Command-private outcomes.
- ``ScalarReducer`` as an domain class attribute — auto-named,
  auto-scoped to the domain's events, surfaced via ``log.reduced_state``.
- ``EventGraph.from_namespaces`` — auto-registers inline handlers and mixes
  them with extra external handlers.

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
    """The Order domain."""

    # ScalarReducer as a declarative class attribute. Auto-named
    # ``current_status``, auto-scoped to the domain's events. Picks a
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

        Inline ``handle`` produces ``Placed``; ``invariants`` are declared as
        a class-level attribute. Failure modes (banned customer, over-limit)
        are surfaced as ``InvariantViolated``; namespace-level recovery
        reactors translate those into ``Order.Rejected``.
        """

        customer_id: str = ""
        amount: int = 0

        class CustomerNotBanned(Invariant):
            """The placing customer must not be on the banned list.

            Pure **pre-check** invariant — the predicate reads
            ``CustomerBanned`` events that have *already* been committed;
            its truth value doesn't depend on what this handler emits.
            """

        class OrderTotalWithinLimit(Invariant):
            """Cumulative placed amount must stay under the daily limit.

            Pure **post-check** invariant — the predicate sums committed
            ``Placed`` events. The *current* handler's about-to-be-emitted
            ``Placed`` is what pushes the total over the limit, so only the
            post-command check (against log + emitted buffer) can catch it.
            Pre-check passes; post-check rolls back and emits
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

        def handle(self) -> Order.Place.Placed:
            return Order.Place.Placed(
                order_id=f"order-for-{self.customer_id}", amount=self.amount
            )

    class Rejected(DomainEvent):
        """Order rejected (e.g. by an invariant).

        Sibling to ``Place``, not nested — emitted by recovery reactors,
        never from ``Place.handle()`` itself.
        """

        reason: str = ""

    class Ship(Command):
        """Ship an accepted order. Inline ``handle``."""

        order_id: str = ""

        class Shipped(DomainEvent):
            """Shipment dispatched."""

            tracking: str = ""

        def handle(self) -> Order.Ship.Shipped:
            return Order.Ship.Shipped(tracking=f"track-{self.order_id}")


# ---------------------------------------------------------------------------
# Recovery reactors — emit namespace-level Order.Rejected
# ---------------------------------------------------------------------------


@on(InvariantViolated, invariant=Order.Place.CustomerNotBanned)
def explain_banned(event: InvariantViolated) -> Order.Rejected:
    """Turn a ``CustomerNotBanned`` violation into a domain ``Rejected``."""
    return Order.Rejected(reason=type(event.invariant).__name__)


@on(InvariantViolated, invariant=Order.Place.OrderTotalWithinLimit)
def explain_over_limit(event: InvariantViolated) -> Order.Rejected:
    """Turn an ``OrderTotalWithinLimit`` post-check violation into a
    ``Rejected``. ``would_emit`` carries the rolled-back ``Placed`` — we
    surface its amount in the rejection reason for diagnostics.
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
