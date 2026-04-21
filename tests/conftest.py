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


@pytest.fixture(autouse=True)
def _reset_aggregate_registry():
    """Restore ``_AGGREGATE_REGISTRY`` after each test.

    Aggregate names are enforced unique process-wide. Tests that define their
    own ``class X(Aggregate)`` would otherwise leak into the registry and
    collide with later tests using the same short name. conftest-scoped
    aggregates (e.g. ``Order``) are captured in the snapshot and survive.
    """
    from langgraph_events import _event as _e

    snapshot = dict(_e._AGGREGATE_REGISTRY)
    yield
    _e._AGGREGATE_REGISTRY.clear()
    _e._AGGREGATE_REGISTRY.update(snapshot)
