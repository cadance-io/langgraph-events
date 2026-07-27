"""Reflection agent — langgraph-events demo.

A real-world diagnosis loop: an order is cancelled deep inside an
event-driven flow (payment declined → reaction cancels the order), and an
agent works out *why* — lazily, by driving the ``query_log`` tool the way a
ReAct agent would, instead of reading the whole event log into context.

The reflective surface is deterministic: every op returns facts (listings,
field dumps, static topology, verdict-free evidence joins). Correlating those
facts into a causal story is the agent's job — here scripted step by step so
the example runs offline; each ``query_log`` call is exactly what an LLM
would issue as a tool call. ``QueryTool``'s shape (name / description /
JSON-schema parameters / run) maps 1:1 onto an Anthropic tool dict or
LangChain's ``StructuredTool.from_function``.

Covers APIs not shown in other examples:
  - ``EventGraph.reflect(log)`` — the ``Reflection`` read-model over a run
  - ``Reflection.context()`` — bounded prompt card for an agent's context
  - ``Reflection.tool()`` / ``query_log`` — one dispatch tool, ReAct-driven
  - ``Reflection`` injection — a handler receives the mid-run snapshot by
    annotating a parameter, exactly like ``EventLog`` injection

Usage:
    python examples/reflection_agent.py
"""

from __future__ import annotations

from langgraph_events import (
    Command,
    DomainEvent,
    Event,
    EventGraph,
    HandlerRaised,
    Namespace,
    Reflection,
    ScalarReducer,
    on,
)

# ---------------------------------------------------------------------------
# Domain — a sales flow where payment failures cancel the order
# ---------------------------------------------------------------------------


class CardDeclinedError(Exception):
    """Simulated payment-processor decline."""


class Sales(Namespace):
    order_status = ScalarReducer(
        event_type=Event,
        fn=lambda e: {
            "Placed": "placed",
            "Cancelled": "cancelled",
        }.get(type(e).__name__),
    )

    class Place(Command):
        order_id: str = ""
        amount: int = 0

        class Placed(DomainEvent):
            order_id: str = ""
            amount: int = 0

        def place(self) -> Sales.Place.Placed:
            return Sales.Place.Placed(order_id=self.order_id, amount=self.amount)

    class Cancel(Command):
        order_id: str = ""
        reason: str = ""

        class Cancelled(DomainEvent):
            order_id: str = ""
            reason: str = ""

        def cancel(self) -> Sales.Cancel.Cancelled:
            return Sales.Cancel.Cancelled(order_id=self.order_id, reason=self.reason)


class Payments(Namespace):
    class Charge(Command):
        order_id: str = ""
        amount: int = 0
        raises = (CardDeclinedError,)

        class Charged(DomainEvent):
            order_id: str = ""

        def charge(self) -> Payments.Charge.Charged:
            if self.amount > 100:  # simulated issuer limit
                raise CardDeclinedError(f"card declined for {self.amount}")
            return Payments.Charge.Charged(order_id=self.order_id)


@on(Sales.Place.Placed)
def request_payment(event: Sales.Place.Placed) -> Payments.Charge:
    """Orchestrate: every placed order gets charged."""
    return Payments.Charge(order_id=event.order_id, amount=event.amount)


@on(HandlerRaised, exception=CardDeclinedError)
def cancel_on_decline(event: HandlerRaised) -> Sales.Cancel:
    """React: a declined charge cancels the order."""
    charge = event.source_event
    assert isinstance(charge, Payments.Charge)
    return Sales.Cancel(order_id=charge.order_id, reason="payment declined")


@on(Sales.Cancel.Cancelled)
def brief_support_agent(event: Sales.Cancel.Cancelled, run: Reflection) -> None:
    """Injection point: a handler gets the mid-run Reflection snapshot.

    In a real app this is where you'd call your LLM with
    ``system=run.context()`` and ``tools=[run.tool()]``.
    """
    print("--- context card injected mid-run (what the LLM would see) ---")
    print(run.context(tail=3))
    print()


# ---------------------------------------------------------------------------
# The diagnosis loop — each step is one query_log tool call
# ---------------------------------------------------------------------------


def diagnose(run: Reflection) -> None:
    """Scripted ReAct trace: why was the order cancelled?"""
    tool = run.tool()

    def call(**kwargs: object) -> None:
        print(f">>> query_log({', '.join(f'{k}={v!r}' for k, v in kwargs.items())})")
        print(tool.run(**kwargs))  # type: ignore[arg-type]
        print()

    print("=== agent diagnosis: why was order A-1 cancelled? ===\n")
    call(op="overview")  # 1. shape of the run — spot the anomaly
    call(op="filter", type="Cancelled")  # 2. locate the outcome
    call(op="evidence", index=5)  # 3. Cancelled ← owning command Cancel #4
    call(op="evidence", index=4)  # 4. Cancel ← cancel_on_decline ← #3
    call(op="get", index=3)  # 5. HandlerRaised: CardDeclinedError
    call(op="evidence", index=3)  # 6. explicit link: source_event #2 Charge
    call(op="state")  # 7. resulting projections

    print("=== agent conclusion (reasoned from the facts above) ===")
    print(
        "Order A-1 was cancelled because its $250 charge was declined:\n"
        "  #0 Place → #1 Placed → (request_payment) → #2 Charge\n"
        "  #2 Charge raised CardDeclinedError → #3 HandlerRaised\n"
        "  #3 → (cancel_on_decline) → #4 Cancel → #5 Cancelled"
    )


def main() -> None:
    graph = EventGraph(
        [
            Sales.Place,
            Sales.Cancel,
            Payments.Charge,
            request_payment,
            cancel_on_decline,
            brief_support_agent,
        ]
    )
    log = graph.invoke(Sales.Place(order_id="A-1", amount=250))
    diagnose(graph.reflect(log))


if __name__ == "__main__":
    main()
