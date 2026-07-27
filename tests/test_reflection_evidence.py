"""Tests for the evidence op — a verdict-free join over log + static model."""

from __future__ import annotations

from conftest import Started

from langgraph_events import (
    Command,
    DomainEvent,
    EventGraph,
    EventLog,
    HandlerRaised,
    IntegrationEvent,
    Interrupted,
    Namespace,
    Resumed,
    on,
)


class Fulfillment(Namespace):
    class Ship(Command):
        order_id: str = ""

        class Dispatched(DomainEvent):
            order_id: str = ""

        def ship(self) -> Fulfillment.Ship.Dispatched:
            return Fulfillment.Ship.Dispatched(order_id=self.order_id)


class CustomerNotified(IntegrationEvent):
    order_id: str = ""


@on(Fulfillment.Ship.Dispatched)
def notify_customer(event: Fulfillment.Ship.Dispatched) -> CustomerNotified:
    return CustomerNotified(order_id=event.order_id)


class ShipmentError(Exception):
    pass


@on(Started, raises=ShipmentError)
def flaky_start(event: Started) -> CustomerNotified:
    raise ShipmentError("carrier down")


@on(HandlerRaised, exception=ShipmentError)
def swallow_error(event: HandlerRaised) -> None:
    return None


def _reflect(graph, seed):
    return graph.reflect(graph.invoke(seed))


def describe_evidence():
    def when_the_event_is_a_command_outcome():
        def it_lists_the_owning_command_and_its_preceding_instances():
            graph = EventGraph([Fulfillment.Ship, notify_customer])
            reflection = _reflect(graph, Fulfillment.Ship(order_id="o1"))

            text = reflection.evidence(1)

            assert "owning command: Ship" in text
            assert "#0" in text

        def it_lists_static_edge_candidates_by_causation_kind():
            graph = EventGraph([Fulfillment.Ship, notify_customer])
            reflection = _reflect(graph, Fulfillment.Ship(order_id="o1"))

            text = reflection.evidence(1)

            assert "intent" in text

    def when_the_event_was_produced_by_a_policy():
        def it_lists_the_policy_edge_and_candidate_source_instances():
            graph = EventGraph([Fulfillment.Ship, notify_customer])
            reflection = _reflect(graph, Fulfillment.Ship(order_id="o1"))

            text = reflection.evidence(2)

            assert "notify_customer" in text
            assert "#1" in text

    def when_multiple_candidate_instances_precede():
        def it_lists_every_candidate():
            graph = EventGraph([Fulfillment.Ship, notify_customer])
            reflection = _reflect(
                graph,
                [Fulfillment.Ship(order_id="o1"), Fulfillment.Ship(order_id="o2")],
            )
            dispatched_index = next(
                i
                for i, e in enumerate(reflection.log)
                if isinstance(e, Fulfillment.Ship.Dispatched)
            )

            text = reflection.evidence(dispatched_index)

            assert "#0" in text
            assert "#1" in text

    def when_the_event_has_downstream_edges():
        def it_lists_the_forward_face():
            graph = EventGraph([Fulfillment.Ship, notify_customer])
            reflection = _reflect(graph, Fulfillment.Ship(order_id="o1"))

            text = reflection.evidence(1)

            assert "CustomerNotified" in text
            assert "#2" in text

    def when_the_event_is_a_seed():
        def it_reports_no_backward_evidence():
            graph = EventGraph([Fulfillment.Ship, notify_customer])
            reflection = _reflect(graph, Fulfillment.Ship(order_id="o1"))

            assert "no backward evidence" in reflection.evidence(0)

    def when_a_handler_raised():
        def it_links_handler_raised_to_its_source_event_by_identity():
            graph = EventGraph([flaky_start, swallow_error])
            reflection = _reflect(graph, Started(data="x"))
            raised_index = next(
                i
                for i, e in enumerate(reflection.log)
                if type(e).__name__ == "HandlerRaised"
            )

            text = reflection.evidence(raised_index)

            assert "source_event: #0" in text

    def when_the_run_was_resumed():
        def it_links_resumed_to_its_interrupted_by_identity():
            graph = EventGraph([Fulfillment.Ship, notify_customer])
            interrupted = Interrupted()
            resumed = Resumed(value=Started(data="ok"), interrupted=interrupted)
            log = EventLog([interrupted, Started(data="ok"), resumed])
            reflection = graph.reflect(log)

            assert "interrupted: #0" in reflection.evidence(2)

    def when_addressed_by_event_instance():
        def it_resolves_the_instance_by_identity():
            graph = EventGraph([Fulfillment.Ship, notify_customer])
            reflection = _reflect(graph, Fulfillment.Ship(order_id="o1"))
            dispatched = reflection.log.latest(Fulfillment.Ship.Dispatched)

            assert reflection.evidence(dispatched) == reflection.evidence(1)

    def when_equal_events_repeat_after_the_effect():
        def it_resolves_backward_links_to_a_preceding_instance():
            graph = EventGraph([Fulfillment.Ship, notify_customer])
            raised = HandlerRaised(handler="h", source_event=Started(data="dup"))
            log = EventLog([Started(data="dup"), raised, Started(data="dup")])
            reflection = graph.reflect(log)

            text = reflection.evidence(1)

            assert "source_event: #0 Started (equality match)" in text
