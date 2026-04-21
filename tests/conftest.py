"""Shared fixtures and event classes for the test suite."""

import pytest

from langgraph_events import (
    Aggregate,
    Command,
    DomainEvent,
    Event,
    EventGraph,
    IntegrationEvent,
    ScalarReducer,
    on,
)

# ---------------------------------------------------------------------------
# Reusable event classes (used across multiple test files)
# ---------------------------------------------------------------------------


class Started(IntegrationEvent):
    data: str = ""


class Processed(IntegrationEvent):
    data: str = ""


class Ended(IntegrationEvent):
    result: str = ""


class MessageReceived(IntegrationEvent):
    text: str = ""


class MessageSent(IntegrationEvent):
    text: str = ""


class Completed(IntegrationEvent):
    result: str = ""


# Canonical DDD aggregate used by test_invariant.py / test_domain.py /
# test_reducer_aggregate.py. The ``current_status`` reducer demonstrates the
# declarative aggregate-reducer form — auto-named "current_status", auto-scoped
# to Order, auto-discovered by any EventGraph that has a handler subscribed to
# an Order event.
class Order(Aggregate):
    current_status = ScalarReducer(
        event_type=Event,
        fn=lambda e: (
            "shipped"
            if type(e).__name__ == "Shipped"
            else "placed"
            if type(e).__name__ == "Placed"
            else "rejected"
            if type(e).__name__ == "Rejected"
            else None
        ),
    )

    class Place(Command):
        customer_id: str = ""

        class Placed(DomainEvent):
            order_id: str = ""

        class Rejected(DomainEvent):
            reason: str = ""

    class Shipped(DomainEvent):
        tracking: str = ""


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def linear_chain():
    """A simple Started -> Processed -> Ended three-step EventGraph."""

    @on(Started)
    def step1(event: Started) -> Processed:
        return Processed(data=f"processed:{event.data}")

    @on(Processed)
    def step2(event: Processed) -> Ended:
        return Ended(result=f"done:{event.data}")

    return EventGraph([step1, step2])
