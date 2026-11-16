"""EventGraph stores each checkpoint before the next step runs."""

import asyncio
import time
from typing import Any

import pytest
from conftest import Ended, Processed, Started
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from langgraph_events import EventGraph, on

_PUT_DELAY = 0.05

_CONFIG: Any = {"configurable": {"thread_id": "t1"}}


class SlowSaver(InMemorySaver):
    """Store each checkpoint after a delay. Record whether the store held a
    ``Processed`` event each time a handler observes it."""

    def __init__(self) -> None:
        super().__init__()
        self._stored: set[str] = set()
        self.seen: list[bool] = []

    def observe(self) -> None:
        self.seen.append("Processed" in self._stored)

    def _store(self, config, checkpoint, metadata, new_versions):  # type: ignore[no-untyped-def]
        result = super().put(config, checkpoint, metadata, new_versions)
        events = checkpoint["channel_values"].get("events", [])
        self._stored.update(type(event).__name__ for event in events)
        return result

    def put(self, config, checkpoint, metadata, new_versions):  # type: ignore[no-untyped-def]
        time.sleep(_PUT_DELAY)
        return self._store(config, checkpoint, metadata, new_versions)

    async def aput(self, config, checkpoint, metadata, new_versions):  # type: ignore[no-untyped-def]
        # InMemorySaver.aput calls self.put, which blocks the loop on the sleep.
        await asyncio.sleep(_PUT_DELAY)
        return self._store(config, checkpoint, metadata, new_versions)


def _handlers(saver: SlowSaver) -> list[Any]:
    """Two handlers that run one after the other. The second handler asks the
    saver whether the store holds the event of the first when it starts."""

    @on(Started)
    def first(event: Started) -> Processed:
        return Processed(data=event.data)

    @on(Processed)
    def second(event: Processed) -> Ended:
        saver.observe()
        return Ended(result=event.data)

    return [first, second]


@on(Started)
def _end_at_once(event: Started) -> Ended:
    return Ended(result=event.data)


def describe_EventGraph_durability():
    def when_the_run_uses_compiled_invoke():
        def it_stores_the_checkpoint_before_the_next_handler_starts():
            saver = SlowSaver()
            graph = EventGraph(_handlers(saver), checkpointer=saver)

            graph.compiled.invoke({"events": [Started(data="x")]}, config=_CONFIG)

            assert saver.seen == [True]

    def when_the_run_uses_astream_events():
        async def it_stores_the_checkpoint_before_the_next_handler_starts():
            saver = SlowSaver()
            graph = EventGraph(_handlers(saver), checkpointer=saver)

            async for _ in graph.astream_events(Started(data="x"), config=_CONFIG):
                pass

            assert saver.seen == [True]

    def when_the_run_passes_durability():
        def it_uses_the_durability_of_the_run():
            saver = SlowSaver()
            graph = EventGraph(_handlers(saver), checkpointer=saver)

            graph.invoke(Started(data="x"), config=_CONFIG, durability="async")

            assert saver.seen == [False]

    def when_the_graph_is_a_subgraph_that_stores_its_own_checkpoints():
        def it_stores_the_checkpoint_before_the_next_handler_starts():
            saver = SlowSaver()
            child = EventGraph(_handlers(saver), checkpointer=True).compiled
            parent = StateGraph(child.builder.state_schema)
            parent.add_node("child", child)
            parent.add_edge(START, "child")
            parent.add_edge("child", END)

            parent.compile(checkpointer=saver).invoke(
                {"events": [Started(data="x")]}, config=_CONFIG
            )

            assert saver.seen == [True]

    def when_a_handler_runs_an_event_graph():
        def without_a_checkpointer():
            def it_completes_the_run():
                inner = EventGraph([_end_at_once], checkpointer=False)

                @on(Started)
                def outer(event: Started) -> Ended:
                    return Ended(result=inner.invoke(event).latest(Ended).result)

                graph = EventGraph([outer], checkpointer=InMemorySaver())

                log = graph.invoke(Started(data="x"), config=_CONFIG)

                assert log.latest(Ended) == Ended(result="x")

    def when_the_run_has_no_thread_id():
        @pytest.mark.parametrize(
            "run",
            [
                pytest.param(lambda g: g.invoke(Started(data="x")), id="invoke"),
                pytest.param(
                    lambda g: g.compiled.invoke({"events": [Started(data="x")]}),
                    id="compiled-invoke",
                ),
                pytest.param(
                    lambda g: g.compiled.invoke(
                        {"events": [Started(data="x")]}, {"configurable": {}}
                    ),
                    id="compiled-invoke-empty-configurable",
                ),
            ],
        )
        def it_raises_a_value_error_that_names_the_keys(run):
            graph = EventGraph([_end_at_once], checkpointer=InMemorySaver())

            with pytest.raises(ValueError, match="thread_id, checkpoint_ns"):
                run(graph)
