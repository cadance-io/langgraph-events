"""Graph runs record the cause of each event.

Spec: docs/superpowers/specs/2026-10-02-event-causes-design.md
"""

import asyncio
from typing import Any

import pytest
from conftest import Ended, Order, Processed, Started
from langgraph.checkpoint.memory import MemorySaver

from langgraph_events import Cancelled, Cause, EventGraph, EventLog, Reducer, on
from langgraph_events.serde import NamespaceAwareSerde


@on(Started)
def step(event: Started) -> Processed:
    return Processed(data=event.data)


@on(Processed)
def finish(event: Processed) -> Ended:
    return Ended(result=event.data)


def _config(thread_id: str) -> dict[str, Any]:
    return {"configurable": {"thread_id": thread_id}}


def _checkpointed(
    handlers: list[Any], thread_id: str
) -> tuple[EventGraph, dict[str, Any], MemorySaver]:
    """A graph on a fresh MemorySaver, and the config of one thread."""
    saver = MemorySaver(serde=NamespaceAwareSerde())
    graph = EventGraph(handlers, checkpointer=saver)
    return graph, _config(thread_id), saver


def describe_invoke():
    def when_a_policy_reacts_to_a_seed():
        def it_records_the_trigger_and_the_handler():
            log = EventGraph([step, finish]).invoke(Started(data="x"))

            cause = log.cause(log.first(Processed))

            assert cause == Cause(source=Started(data="x"), via="step")
            assert cause.source is log.first(Started)

    def when_an_inline_command_handler_runs():
        def it_names_the_command_qualname():
            log = EventGraph([Order.Place]).invoke(Order.Place(customer_id="c1"))

            assert log.cause(log.first(Order.Place.Placed)) == Cause(
                source=Order.Place(customer_id="c1"), via="Order.Place"
            )

    def when_a_handler_pins_its_node_name():
        def it_names_the_pinned_node():
            @on(Processed, node_name="closer")
            def close(event: Processed) -> Ended:
                return Ended(result=event.data)

            log = EventGraph([step, close]).invoke(Started(data="x"))

            assert log.cause(log.first(Ended)).via == "closer"

    def when_no_handler_reacts():
        def it_records_no_cause_for_the_seed():
            log = EventGraph([finish]).invoke(Started(data="x"))

            assert log.causes == (None,)

    def when_a_handler_reads_the_injected_log():
        def it_sees_the_causes_of_the_run():
            seen: list[Any] = []

            @on(Started)
            def read_log(event: Started, log: EventLog) -> None:
                seen.append(log.causes)

            EventGraph([read_log]).invoke(Started(data="x"))

            assert seen == [(None,)]

        def it_finds_the_cause_of_its_trigger():
            seen: list[Cause | None] = []

            @on(Processed)
            def read_trigger(event: Processed, log: EventLog) -> None:
                seen.append(log.cause(event))

            EventGraph([step, read_trigger]).invoke(Started(data="x"))

            assert seen == [Cause(source=Started(data="x"), via="step")]

    def when_the_graph_has_reducers():
        def it_returns_the_causes():
            seen = Reducer(name="seen", event_type=Started, fn=lambda e: [e.data])

            log = EventGraph([finish], reducers=[seen]).invoke(Started(data="x"))

            assert log.causes == (None,)

    def when_the_causes_channel_holds_more_entries_than_events():
        def it_raises_runtime_error():
            graph, config, _saver = _checkpointed([finish], "longer")
            graph.compiled.update_state(
                config,
                {"events": [Started(data="a")], "causes": [None, None, None]},
                as_node="__seed__",
            )

            with pytest.raises(RuntimeError, match="3 entries for 2 events"):
                graph.invoke(Started(data="b"), config=config)

    def when_a_stored_source_is_not_an_earlier_event():
        def it_raises_runtime_error():
            graph, config, _saver = _checkpointed([finish], "bad-source")
            graph.compiled.update_state(
                config,
                {"events": [Started(data="a")], "causes": [(0, "h")]},
                as_node="__seed__",
            )

            with pytest.raises(RuntimeError, match="not an earlier event"):
                graph.invoke(Started(data="b"), config=config)


def describe_ainvoke():
    def when_no_handler_reacts():
        async def it_records_no_cause_for_the_seed():
            log = await EventGraph([finish]).ainvoke(Started(data="x"))

            assert log.causes == (None,)

    def when_a_policy_reacts_to_a_seed():
        async def it_records_the_trigger_and_the_handler():
            log = await EventGraph([step, finish]).ainvoke(Started(data="x"))

            assert log.cause(log.first(Ended)) == Cause(
                source=Processed(data="x"), via="finish"
            )

    def when_a_handler_is_cancelled():
        async def it_records_no_cause_for_the_cancellation():
            ready = asyncio.Event()

            @on(Processed)
            async def stall(event: Processed) -> Ended:
                ready.set()
                await asyncio.sleep(100)
                return Ended(result="never")

            task = asyncio.ensure_future(
                EventGraph([step, stall]).ainvoke(Started(data="x"))
            )
            await ready.wait()
            task.cancel()
            log = await task

            assert log.cause(log.first(Processed)) == Cause(
                source=log.first(Started), via="step"
            )
            assert log.cause(log.latest(Cancelled)) is None
