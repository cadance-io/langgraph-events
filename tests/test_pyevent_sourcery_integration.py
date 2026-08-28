"""TDD tests for python-event-sourcery integration.

Covers:
1. Event base class migration — our Events are pyES Events
2. EventStore persistence — EventGraph persists to pyES EventStore
"""

from __future__ import annotations

import pytest
from event_sourcery import Event as ESEvent
from event_sourcery import StreamId
from event_sourcery.backend import InMemoryBackend

from langgraph_events import (
    Command,
    DomainEvent,
    Event,
    EventGraph,
    EventLog,
    IntegrationEvent,
    Namespace,
)

# ---------------------------------------------------------------------------
# Shared event fixtures
# ---------------------------------------------------------------------------


class OrderNS(Namespace):
    class Place(Command):
        customer_id: str = ""

        class Placed(DomainEvent):
            order_id: str = ""

        def place(self) -> OrderNS.Place.Placed:
            return OrderNS.Place.Placed(order_id="o1")


class OrderShipped(IntegrationEvent):
    tracking: str = ""


# ---------------------------------------------------------------------------
# 1. Event base class — our Event IS a pyES Event
# ---------------------------------------------------------------------------


def describe_event_as_pyes_event():
    def it_is_subclass_of_pyes_event():
        assert issubclass(Event, ESEvent)

    def it_domain_event_is_subclass_of_pyes_event():
        assert issubclass(DomainEvent, ESEvent)

    def it_integration_event_is_subclass_of_pyes_event():
        assert issubclass(IntegrationEvent, ESEvent)

    def it_command_is_subclass_of_pyes_event():
        assert issubclass(Command, ESEvent)

    def it_instance_is_pyes_event():
        event = OrderNS.Place.Placed(order_id="o1")
        assert isinstance(event, ESEvent)

    def it_integration_event_instance_is_pyes_event():
        event = OrderShipped(tracking="TR123")
        assert isinstance(event, ESEvent)

    def it_is_frozen():
        event = OrderNS.Place.Placed(order_id="o1")
        with pytest.raises((TypeError, Exception)):
            event.order_id = "changed"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# 2. EventStore persistence
# ---------------------------------------------------------------------------


def describe_event_graph_with_event_store():
    @pytest.fixture
    def backend():
        return InMemoryBackend()

    @pytest.fixture
    def stream_id():
        return StreamId(name="thread-1")

    def it_persists_events_to_store(backend, stream_id):
        graph = EventGraph(
            [OrderNS.Place],
            event_store=backend.event_store,
        )
        graph.invoke(
            OrderNS.Place(customer_id="c1"),
            config={"configurable": {"thread_id": "thread-1"}},
        )

        recorded = backend.event_store.load_stream(stream_id)
        assert len(recorded) > 0
        events = [r.event for r in recorded]
        assert any(isinstance(e, OrderNS.Place.Placed) for e in events)

    def it_loads_event_log_from_store(backend, stream_id):
        graph = EventGraph(
            [OrderNS.Place],
            event_store=backend.event_store,
        )
        graph.invoke(
            OrderNS.Place(customer_id="c1"),
            config={"configurable": {"thread_id": "thread-1"}},
        )

        log = EventLog.from_store(backend.event_store, stream_id)
        assert log.has(OrderNS.Place.Placed)

    def it_accumulates_events_across_runs(backend, stream_id):
        graph = EventGraph(
            [OrderNS.Place],
            event_store=backend.event_store,
            checkpointer=__import__(
                "langgraph.checkpoint.memory", fromlist=["MemorySaver"]
            ).MemorySaver(),
        )
        cfg = {"configurable": {"thread_id": "thread-1"}}
        graph.invoke(OrderNS.Place(customer_id="c1"), config=cfg)
        graph.invoke(OrderNS.Place(customer_id="c2"), config=cfg)

        log = EventLog.from_store(backend.event_store, stream_id)
        assert log.count(OrderNS.Place.Placed) == 2
