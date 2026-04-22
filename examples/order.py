"""DDD Order — langgraph-events demo.

Demonstrates the DDD-aligned taxonomy end-to-end:

- ``Namespace`` — the ``Order`` namespace
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
- ``ScalarReducer`` as an domain class attribute — auto-named,
  auto-scoped to the domain's events, surfaced via ``log.reduced_state``.
- ``EventGraph.from_namespaces`` — auto-registers inline handlers and mixes
  them with extra external handlers.

Usage:
    python examples/order.py
"""

from __future__ import annotations

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

        No inline ``handle`` — this command needs an ``invariants=`` clause
        to guard against banned customers, so its handler lives externally
        as ``place`` below.
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


_ORDER_LIMIT = 100


@on(
    invariants={
        Order.Place.CustomerNotBanned: lambda log: not log.has(CustomerBanned),
        Order.Place.OrderTotalWithinLimit: lambda log: (
            sum(e.amount for e in log.filter(Order.Place.Placed)) < _ORDER_LIMIT
        ),
    },
)
def place(event: Order.Place) -> Order.Place.Placed:
    """External handler — the invariants are declared here, not inline.

    Uses ``@on(invariants=...)`` — no positional event type, inferred from
    the ``event:`` annotation. Invariant keys are typed classes so reactors
    can pin on them with mypy/IDE support (and a typo triggers the
    graph-compile-time drift check rather than silently failing).

    Two invariants, two semantics:

    - ``CustomerNotBanned`` — only the pre-check can fire it (predicate
      doesn't depend on this handler's output).
    - ``OrderTotalWithinLimit`` — only the post-check can fire it
      (predicate sums ``Placed`` events, and *this* handler's ``Placed``
      is what pushes the total over the limit).
    """
    return Order.Place.Placed(
        order_id=f"order-for-{event.customer_id}", amount=event.amount
    )


@on(InvariantViolated, invariant=Order.Place.CustomerNotBanned)
def explain_banned(event: InvariantViolated) -> Order.Place.Rejected:
    """Turn a ``CustomerNotBanned`` violation into a domain ``Rejected``."""
    return Order.Place.Rejected(reason=type(event.invariant).__name__)


@on(InvariantViolated, invariant=Order.Place.OrderTotalWithinLimit)
def explain_over_limit(event: InvariantViolated) -> Order.Place.Rejected:
    """Turn an ``OrderTotalWithinLimit`` post-check violation into a
    ``Rejected``. ``would_emit`` carries the rolled-back ``Placed`` — we
    surface its amount in the rejection reason for diagnostics.
    """
    rolled_back = event.would_emit[0] if event.would_emit else None
    amount = getattr(rolled_back, "amount", "?")
    return Order.Place.Rejected(
        reason=f"OrderTotalWithinLimit (would emit amount={amount})"
    )


# ---------------------------------------------------------------------------
# Graph — auto-registers Order's inline handlers; extras are appended
# ---------------------------------------------------------------------------


graph = EventGraph.from_namespaces(
    Order,
    handlers=[place, explain_banned, explain_over_limit],
)


def main() -> None:
    print("=== Namespace ===")
    print(graph.namespaces().text())
    print()

    print("=== Happy path (external handler + inline handler) ===")
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
    # Two placements. The first (60) commits — pre-check passes, post-check
    # sees simulated total 60 < 100. The second (60) also passes pre-check
    # (current total is 60), but post-check sees simulated total 120 ≥ 100
    # and drops the Placed; the reactor turns the violation into a Rejected.
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
