"""EventGraph stores each checkpoint before the next step runs."""

import asyncio
import contextvars
import threading
from typing import Any

import pytest
from conftest import Ended, Processed, Started
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from langgraph_events import EventGraph, on

_GATE_TIMEOUT = 1.0

_CONFIG: Any = {"configurable": {"thread_id": "t1"}}


class GatedSaver(InMemorySaver):
    """Hold the first checkpoint that holds a ``Processed`` event until a
    handler observes the store, or until a timeout. Record whether the store
    held a ``Processed`` event each time a handler observes it.

    A run that stores the checkpoint before the next step waits for the
    timeout, then observes ``True``. A run that stores it in the background
    observes ``False``, because the put cannot finish before the observation.
    """

    def __init__(self) -> None:
        super().__init__()
        self._observed = threading.Event()
        self._stored: set[str] = set()
        self.seen: list[bool] = []

    def observe(self) -> None:
        self.seen.append("Processed" in self._stored)
        self._observed.set()

    def _held(self, checkpoint) -> bool:  # type: ignore[no-untyped-def]
        events = checkpoint["channel_values"].get("events", [])
        return any(type(event).__name__ == "Processed" for event in events)

    def _store(self, config, checkpoint, metadata, new_versions):  # type: ignore[no-untyped-def]
        result = super().put(config, checkpoint, metadata, new_versions)
        events = checkpoint["channel_values"].get("events", [])
        self._stored.update(type(event).__name__ for event in events)
        return result

    def put(self, config, checkpoint, metadata, new_versions):  # type: ignore[no-untyped-def]
        if self._held(checkpoint):
            self._observed.wait(timeout=_GATE_TIMEOUT)
        return self._store(config, checkpoint, metadata, new_versions)

    async def aput(self, config, checkpoint, metadata, new_versions):  # type: ignore[no-untyped-def]
        # InMemorySaver.aput calls self.put, which blocks the loop on the wait.
        if self._held(checkpoint):
            await asyncio.to_thread(self._observed.wait, _GATE_TIMEOUT)
        return self._store(config, checkpoint, metadata, new_versions)


def _handlers(saver: GatedSaver) -> list[Any]:
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
            saver = GatedSaver()
            graph = EventGraph(_handlers(saver), checkpointer=saver)

            graph.compiled.invoke({"events": [Started(data="x")]}, config=_CONFIG)

            assert saver.seen == [True]

    def when_the_run_uses_astream_events():
        async def it_stores_the_checkpoint_before_the_next_handler_starts():
            saver = GatedSaver()
            graph = EventGraph(_handlers(saver), checkpointer=saver)

            async for _ in graph.astream_events(Started(data="x"), config=_CONFIG):
                pass

            assert saver.seen == [True]

    def when_the_run_passes_durability():
        def it_uses_the_durability_of_the_run():
            saver = GatedSaver()
            graph = EventGraph(_handlers(saver), checkpointer=saver)

            graph.invoke(Started(data="x"), config=_CONFIG, durability="async")

            assert saver.seen == [False]

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


def describe_EventGraph_run_inside_another_graph():
    def when_the_compiled_graph_is_a_subgraph_node():
        def it_raises_not_implemented_error():
            child = EventGraph([_end_at_once]).compiled
            parent = StateGraph(child.builder.state_schema)
            parent.add_node("child", child)
            parent.add_edge(START, "child")
            parent.add_edge("child", END)

            with pytest.raises(NotImplementedError, match="fresh context"):
                parent.compile().invoke({"events": [Started(data="x")]})

    def when_a_handler_runs_an_event_graph():
        def it_raises_not_implemented_error():
            inner = EventGraph([_end_at_once])

            @on(Started)
            def outer(event: Started) -> Ended:
                return Ended(result=inner.invoke(event).latest(Ended).result)

            graph = EventGraph([outer], checkpointer=InMemorySaver())

            with pytest.raises(NotImplementedError, match="fresh context"):
                graph.invoke(Started(data="x"), config=_CONFIG)

    def when_the_root_run_carries_a_checkpoint_ns():
        def it_raises_not_implemented_error():
            graph = EventGraph([_end_at_once], checkpointer=InMemorySaver())
            config: Any = {"configurable": {"thread_id": "t1", "checkpoint_ns": "x"}}

            with pytest.raises(NotImplementedError, match="checkpoint_ns"):
                graph.invoke(Started(data="x"), config=config)

    def when_a_handler_runs_an_event_graph_in_a_fresh_context():
        def it_completes_both_runs():
            inner = EventGraph([_end_at_once])

            @on(Started)
            def outer(event: Started) -> Ended:
                log = contextvars.Context().run(inner.invoke, event)
                return Ended(result=log.latest(Ended).result)

            graph = EventGraph([outer], checkpointer=InMemorySaver())

            log = graph.invoke(Started(data="x"), config=_CONFIG)

            assert log.latest(Ended) == Ended(result="x")
