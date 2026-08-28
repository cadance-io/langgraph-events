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
    on,
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
# 3. Outbox for IntegrationEvent
# ---------------------------------------------------------------------------


class PaymentConfirmed(IntegrationEvent):
    transaction_id: str = ""


def describe_event_graph_with_outbox():
    @pytest.fixture
    def outbox_backend():
        from event_sourcery._event_store.in_memory import InMemoryConfig

        backend = InMemoryBackend()
        backend.configure(InMemoryConfig())
        return backend.with_outbox()

    @pytest.fixture
    def stream_id():
        return StreamId(name="outbox-thread-1")

    def it_publishes_integration_events_to_outbox(outbox_backend, stream_id):
        """IntegrationEvents emitted by a run land in the pyES Outbox."""

        @on(OrderNS.Place.Placed)
        def confirm(
            event: OrderNS.Place.Placed,
        ) -> PaymentConfirmed:
            return PaymentConfirmed(transaction_id="tx-1")

        graph = EventGraph(
            [OrderNS.Place, confirm],
            outbox=outbox_backend.event_store,
        )
        graph.invoke(
            OrderNS.Place(customer_id="c1"),
            config={"configurable": {"thread_id": "outbox-thread-1"}},
        )

        published: list[IntegrationEvent] = []
        outbox_backend.outbox.run(
            lambda record: published.append(record.wrapped_event.event)
            if isinstance(record.wrapped_event.event, IntegrationEvent)
            else None
        )
        assert any(isinstance(e, PaymentConfirmed) for e in published)

    def it_does_not_publish_domain_events_to_outbox(outbox_backend, stream_id):
        """DomainEvents stay inside the bounded context — not in outbox."""

        graph = EventGraph(
            [OrderNS.Place],
            outbox=outbox_backend.event_store,
        )
        graph.invoke(
            OrderNS.Place(customer_id="c1"),
            config={"configurable": {"thread_id": "outbox-thread-1"}},
        )

        published: list[object] = []
        outbox_backend.outbox.run(
            lambda record: published.append(record.wrapped_event.event)
        )
        domain_in_outbox = [e for e in published if isinstance(e, DomainEvent)]
        assert domain_in_outbox == []


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
