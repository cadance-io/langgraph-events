"""Graph runs record the cause of each event.

Spec: docs/superpowers/specs/2026-10-02-event-causes-design.md
"""

import asyncio
import itertools
import time
from typing import Any

import pytest
from conftest import Ended, Order, Processed, Started
from langgraph.checkpoint.memory import MemorySaver

from langgraph_events import (
    Cancelled,
    Cause,
    EventGraph,
    EventLog,
    HandlerRaised,
    HandlerRetried,
    IntegrationEvent,
    Invariant,
    InvariantViolated,
    MaxRoundsExceeded,
    Reducer,
    RetryPolicy,
    RunPaused,
    Scatter,
    on,
)
from langgraph_events.serde import NamespaceAwareSerde


@on(Started)
def step(event: Started) -> Processed:
    return Processed(data=event.data)


@on(Processed)
def finish(event: Processed) -> Ended:
    return Ended(result=event.data)


class Batch(IntegrationEvent):
    n: int = 0


class Item(IntegrationEvent):
    n: int = 0


class Tally(IntegrationEvent):
    n: int = 0


class Echo(IntegrationEvent):
    n: int = 0


class Tick(IntegrationEvent):
    n: int = 0


class Tock(IntegrationEvent):
    n: int = 0


class Noted(IntegrationEvent):
    pass


class TockCap(Invariant):
    pass


class Closed(Invariant):
    pass


class FlakyError(Exception):
    pass


@on(Batch)
def split(event: Batch) -> Scatter[Item]:
    return Scatter([Item(n=1), Item(n=2)])


@on(Batch)
def count(event: Batch) -> Tally:
    return Tally(n=event.n)


@on(Batch)
def ignore(event: Batch) -> None:
    return None


@on(Tally)
def echo(event: Tally) -> Echo:
    return Echo(n=event.n)


@on(Tick)
def tock(event: Tick) -> Tock:
    return Tock(n=event.n)


@on(Tick, invariants={TockCap: lambda log: log.count(Tock) < 3})
def double(event: Tick) -> Scatter[Tock]:
    return Scatter([Tock(n=event.n), Tock(n=event.n + 10)])


@on(Tick, invariants={Closed: lambda log: False})
def blocked(event: Tick) -> Tock:
    return Tock(n=event.n)


@on(
    Tick,
    raises=FlakyError,
    retry=RetryPolicy(max_attempts=2, base_delay=0.0, jitter=False),
)
def flaky(event: Tick) -> Tock:
    raise FlakyError("down")


@on(HandlerRaised, exception=FlakyError)
def swallow(event: HandlerRaised) -> None:
    return None


@on(Tick)
def again(event: Tick) -> Tick:
    return Tick(n=event.n + 1)


@on(RunPaused)
def note_pause(event: RunPaused) -> Noted:
    return Noted()


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

    def when_parallel_handlers_react_to_one_event():
        def it_records_each_emission_against_its_own_handler():
            log = EventGraph([split, count, ignore, echo]).invoke(Batch(n=3))
            batch = log.first(Batch)

            assert [log.cause(item) for item in log.filter(Item)] == [
                Cause(source=batch, via="split"),
                Cause(source=batch, via="split"),
            ]
            assert log.cause(log.first(Tally)) == Cause(source=batch, via="count")
            assert log.cause(log.first(Echo)) == Cause(
                source=log.first(Tally), via="echo"
            )

    def when_one_node_handles_several_triggers():
        def it_records_each_output_against_its_own_trigger():
            log = EventGraph([tock]).invoke([Tick(n=1), Noted(), Tick(n=2)])

            sources = [log.cause(t).source for t in log.filter(Tock)]

            assert sources == [Tick(n=1), Tick(n=2)]

    def when_an_invariant_rolls_back_an_emission():
        def it_records_the_violation_against_its_trigger():
            log = EventGraph([double]).invoke([Tick(n=1), Tick(n=2)])

            assert [log.cause(t).source for t in log.filter(Tock)] == [
                Tick(n=1),
                Tick(n=1),
            ]
            assert log.cause(log.first(InvariantViolated)) == Cause(
                source=Tick(n=2), via="double"
            )

    def when_an_invariant_blocks_the_handler():
        def it_records_the_violation_against_its_trigger():
            log = EventGraph([blocked]).invoke(Tick(n=1))

            assert log.cause(log.first(InvariantViolated)) == Cause(
                source=Tick(n=1), via="blocked"
            )

    def when_a_handler_raises_after_a_retry():
        def it_records_the_retry_and_the_raise_against_the_trigger():
            log = EventGraph([flaky, swallow]).invoke(Tick(n=1))
            expected = Cause(source=log.first(Tick), via="flaky")

            assert log.cause(log.first(HandlerRetried)) == expected
            assert log.cause(log.first(HandlerRaised)) == expected

    def when_max_rounds_is_exceeded():
        def it_records_no_cause_for_the_halt():
            log = EventGraph([again], max_rounds=2).invoke(Tick(n=0))
            ticks = log.filter(Tick)

            assert log.cause(ticks[0]) is None
            for earlier, later in itertools.pairwise(ticks):
                assert log.cause(later) == Cause(source=earlier, via="again")
            assert log.cause(log.latest(MaxRoundsExceeded)) is None

    def when_the_deadline_passes_after_a_round():
        def it_records_no_cause_for_the_pause():
            deadline = 0.0

            @on(Started)
            def slow(event: Started) -> Processed:
                time.sleep(max(0.0, deadline - time.monotonic()) + 0.01)
                return Processed(data=event.data)

            graph = EventGraph([slow, note_pause])
            assert graph.compiled is not None
            deadline = time.monotonic() + 0.2

            log = graph.invoke(Started(data="x"), deadline=deadline)
            paused = log.first(RunPaused)

            assert log.cause(log.first(Processed)) == Cause(
                source=log.first(Started), via="slow"
            )
            assert log.cause(paused) is None
            assert log.cause(log.first(Noted)) == Cause(source=paused, via="note_pause")

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
