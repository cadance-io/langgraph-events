"""Graph runs record the cause of each event.

Spec: docs/superpowers/specs/2026-10-02-event-causes-design.md
"""

from typing import Any

import pytest
from conftest import Ended, Processed, Started
from langgraph.checkpoint.memory import MemorySaver

from langgraph_events import EventGraph, EventLog, Reducer, on
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
