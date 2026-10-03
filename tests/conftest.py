"""Shared fixtures and event classes for the test suite."""

import sys
from typing import Any

import pytest
from langgraph.checkpoint.memory import MemorySaver

from langgraph_events import (
    Command,
    DomainEvent,
    Event,
    EventGraph,
    IntegrationEvent,
    Namespace,
    ScalarReducer,
    on,
)

_WRAPS_SET_NAME_ERRORS = sys.version_info < (3, 12)
"""CPython wrapped a failing ``__set_name__`` in RuntimeError until 3.12."""

SET_NAME_ERRORS = RuntimeError if _WRAPS_SET_NAME_ERRORS else TypeError


def set_name_cause(error: BaseException) -> BaseException:
    """The error ``__set_name__`` raised, unwrapped on the Pythons that wrap it."""
    return error.__cause__ if _WRAPS_SET_NAME_ERRORS else error


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


# Canonical namespace used by test_invariant.py / test_namespace.py /
# test_reducer_namespace.py. The ``current_status`` reducer demonstrates
# the declarative namespace-reducer form — auto-named "current_status",
# auto-scoped to Order, auto-discovered by any EventGraph that has a
# handler subscribed to an Order event.
class Order(Namespace):
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

        def place(self) -> "Order.Place.Placed | Order.Place.Rejected":
            if self.customer_id == "banned":
                return Order.Place.Rejected(reason="banned")
            return Order.Place.Placed(order_id="o1")

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


def strip_channels(saver: MemorySaver, config: dict[str, Any], *channels: str) -> None:
    """Rewrite the latest checkpoint of the thread without *channels*.

    Simulates a checkpoint that a release saved before those channels
    existed. The checkpoint id stays the same, so a pending interrupt write
    still belongs to it.
    """
    tup = saver.get_tuple(config)
    assert tup is not None
    checkpoint = dict(tup.checkpoint)
    checkpoint["channel_values"] = {
        name: value
        for name, value in tup.checkpoint["channel_values"].items()
        if name not in channels
    }
    checkpoint["channel_versions"] = {
        name: version
        for name, version in tup.checkpoint["channel_versions"].items()
        if name not in channels
    }
    base = tup.parent_config or {
        "configurable": {
            "thread_id": config["configurable"]["thread_id"],
            "checkpoint_ns": "",
        }
    }
    saver.put(base, checkpoint, tup.metadata, {})
