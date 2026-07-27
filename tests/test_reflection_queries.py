"""Tests for Reflection — deterministic query surface over the event log."""

from __future__ import annotations

from conftest import Order, Started

from langgraph_events import (
    EventGraph,
    EventLog,
    Halted,
    IntegrationEvent,
    Interrupted,
    Reflection,
    Resumed,
    RunPaused,
    on,
)


class Stopped(Halted):
    reason: str = ""


class PayloadWrapped(IntegrationEvent):
    payload: object = None


@on(Started)
def _halt_on_start(event: Started) -> Stopped:
    return Stopped(reason="boom")


class _ExplodingRepr:
    def __repr__(self) -> str:
        raise RuntimeError("no repr for you")


def describe_reflect():
    def it_returns_a_reflection_over_the_log():
        graph = EventGraph([Order.Place])
        log = graph.invoke(Order.Place(customer_id="c1"))

        reflection = graph.reflect(log)

        assert isinstance(reflection, Reflection)

    def it_exposes_the_raw_event_log():
        graph = EventGraph([Order.Place])
        log = graph.invoke(Order.Place(customer_id="c1"))

        reflection = graph.reflect(log)

        assert reflection.log is log


def describe_overview():
    def when_the_run_completes():
        def it_reports_totals_and_counts_by_kind():
            graph = EventGraph([Order.Place])
            reflection = graph.reflect(graph.invoke(Order.Place(customer_id="c1")))

            text = reflection.overview()

            assert "2 events" in text
            assert "command: 1" in text
            assert "domain: 1" in text

        def it_reports_counts_by_namespace():
            graph = EventGraph([Order.Place])
            reflection = graph.reflect(graph.invoke(Order.Place(customer_id="c1")))

            assert "Order: 2" in reflection.overview()

        def it_lists_seed_events_as_indexed_lines():
            graph = EventGraph([Order.Place])
            reflection = graph.reflect(graph.invoke(Order.Place(customer_id="c1")))

            assert "#0 Place(customer_id='c1')" in reflection.overview()

        def it_reports_completed_status():
            graph = EventGraph([Order.Place])
            reflection = graph.reflect(graph.invoke(Order.Place(customer_id="c1")))

            assert "status: completed" in reflection.overview()

    def when_the_run_halts():
        def it_reports_halted_status():
            graph = EventGraph([_halt_on_start])
            reflection = graph.reflect(graph.invoke(Started(data="x")))

            assert "status: halted" in reflection.overview()

        def it_lists_the_anomaly():
            graph = EventGraph([_halt_on_start])
            reflection = graph.reflect(graph.invoke(Started(data="x")))

            assert "Stopped(reason='boom')" in reflection.overview()

    def when_anomalies_flood_the_log():
        def it_caps_the_anomaly_listing():
            graph = EventGraph([Order.Place])
            log = EventLog([Stopped(reason=str(i)) for i in range(8)])

            assert "and 3 more" in graph.reflect(log).overview()

    def when_an_interrupt_is_unresumed():
        def it_reports_interrupted_status():
            graph = EventGraph([Order.Place])
            log = EventLog([Order.Place(customer_id="c1"), Interrupted()])

            assert "status: interrupted" in graph.reflect(log).overview()

    def when_the_latest_interrupt_was_resumed():
        def it_reports_completed_status():
            graph = EventGraph([Order.Place])
            interrupted = Interrupted()
            log = EventLog([interrupted, Resumed(value=None, interrupted=interrupted)])

            assert "status: completed" in graph.reflect(log).overview()

    def when_the_run_is_paused():
        def it_reports_paused_status():
            graph = EventGraph([Order.Place])
            log = EventLog([Order.Place(customer_id="c1"), RunPaused()])

            assert "status: paused" in graph.reflect(log).overview()


def describe_context():
    def it_shows_only_the_last_tail_events():
        graph = EventGraph([Order.Place])
        log = graph.invoke(
            [Order.Place(customer_id="c1"), Order.Place(customer_id="c2")]
        )
        reflection = graph.reflect(log)

        text = reflection.context(tail=2)

        assert f"#{len(log) - 1} " in text
        assert "#0 " not in text.split("recent events")[-1]

    def it_points_the_agent_at_the_query_tool():
        graph = EventGraph([Order.Place])
        reflection = graph.reflect(graph.invoke(Order.Place(customer_id="c1")))

        assert "query_log" in reflection.context()

    def it_lists_anomalies():
        graph = EventGraph([_halt_on_start])
        reflection = graph.reflect(graph.invoke(Started(data="x")))

        assert "Stopped(reason='boom')" in reflection.context()


def describe_state():
    def it_projects_each_reducer_over_the_full_log():
        graph = EventGraph([Order.Place])
        reflection = graph.reflect(graph.invoke(Order.Place(customer_id="c1")))

        assert reflection.state() == {"current_status": "placed"}


def describe_schema():
    def it_renders_the_static_topology():
        graph = EventGraph([Order.Place])
        reflection = graph.reflect(graph.invoke(Order.Place(customer_id="c1")))

        text = reflection.schema()

        assert "Place" in text
        assert "Placed" in text


def describe_event():
    def when_addressed_by_instance():
        def it_resolves_like_an_index():
            graph = EventGraph([Order.Place])
            reflection = graph.reflect(graph.invoke(Order.Place(customer_id="c1")))

            assert reflection.event(reflection.log[1]) == reflection.event(1)

    def when_the_index_is_valid():
        def it_dumps_the_full_event_detail():
            graph = EventGraph([Order.Place])
            reflection = graph.reflect(graph.invoke(Order.Place(customer_id="c1")))

            text = reflection.event(1)

            assert "#1 Placed" in text
            assert "order_id: 'o1'" in text
            assert "kind: domain" in text
            assert "namespace: Order" in text
            assert "command: Place" in text

    def when_the_index_is_negative():
        def it_normalizes_to_the_canonical_index():
            graph = EventGraph([Order.Place])
            reflection = graph.reflect(graph.invoke(Order.Place(customer_id="c1")))

            assert reflection.event(-1) == reflection.event(1)

    def when_the_index_is_out_of_range():
        def it_raises_index_error():
            import pytest

            graph = EventGraph([Order.Place])
            reflection = graph.reflect(graph.invoke(Order.Place(customer_id="c1")))

            with pytest.raises(IndexError):
                reflection.event(42)

    def when_the_instance_is_not_in_the_log():
        def it_raises_value_error():
            import pytest

            graph = EventGraph([Order.Place])
            reflection = graph.reflect(graph.invoke(Order.Place(customer_id="c1")))

            with pytest.raises(ValueError, match="not found"):
                reflection.event(Started(data="elsewhere"))

    def when_a_field_repr_throws():
        def it_renders_an_unrepresentable_marker():
            graph = EventGraph([Order.Place])
            log = EventLog([PayloadWrapped(payload=_ExplodingRepr())])
            reflection = graph.reflect(log)

            assert "<unrepresentable _ExplodingRepr>" in reflection.event(0)

    def when_a_field_value_is_huge():
        def it_truncates_the_detail_dump():
            graph = EventGraph([Order.Place])
            log = EventLog([PayloadWrapped(payload="x" * 5000)])
            reflection = graph.reflect(log)

            detail = reflection.event(0)

            assert "..." in detail
            assert len(detail) < 3000
