"""EventGraph stores each checkpoint before the next step runs."""

import asyncio
import time
from typing import Any

from conftest import Ended, Processed, Started
from langgraph.checkpoint.memory import InMemorySaver

from langgraph_events import EventGraph, EventTopology, on

_PUT_DELAY = 0.2


class SlowSaver(InMemorySaver):
    """Store each checkpoint after a delay, and record the stored event types."""

    def __init__(self) -> None:
        super().__init__()
        self.stored: set[str] = set()

    def _record(self, checkpoint: Any) -> None:
        events = checkpoint["channel_values"].get("events", [])
        self.stored.update(type(event).__name__ for event in events)

    def put(self, config, checkpoint, metadata, new_versions):  # type: ignore[no-untyped-def]
        time.sleep(_PUT_DELAY)
        result = super().put(config, checkpoint, metadata, new_versions)
        self._record(checkpoint)
        return result

    async def aput(self, config, checkpoint, metadata, new_versions):  # type: ignore[no-untyped-def]
        await asyncio.sleep(_PUT_DELAY)
        result = InMemorySaver.put(self, config, checkpoint, metadata, new_versions)
        self._record(checkpoint)
        return result


def _handlers(saver: SlowSaver, seen: list[bool]) -> list[Any]:
    """Build two handlers that run one after the other. The second handler
    records whether the store holds the event of the first when it starts."""

    @on(Started)
    def first(event: Started) -> Processed:
        return Processed(data=event.data)

    @on(Processed)
    def second(event: Processed) -> Ended:
        seen.append("Processed" in saver.stored)
        return Ended(result=event.data)

    return [first, second]


def _graph(saver: SlowSaver, seen: list[bool]) -> EventGraph:
    return EventGraph(_handlers(saver, seen), checkpointer=saver)


def _config(thread: str) -> dict[str, Any]:
    return {"configurable": {"thread_id": thread}}


def describe_EventGraph_durability():
    def when_the_run_passes_no_durability():
        def it_stores_the_checkpoint_before_the_next_handler_starts():
            saver, seen = SlowSaver(), []
            graph = _graph(saver, seen)

            graph.invoke(Started(data="x"), config=_config("t1"))

            assert seen == [True]

        def it_applies_to_a_graph_built_with_of():
            saver, seen = SlowSaver(), []
            topology = EventTopology(_handlers(saver, seen))
            graph = EventGraph.of(topology, checkpointer=saver)

            graph.invoke(Started(data="x"), config=_config("t1"))

            assert seen == [True]

        def it_applies_to_a_run_of_the_compiled_graph():
            saver, seen = SlowSaver(), []
            graph = _graph(saver, seen)

            graph.compiled.invoke({"events": [Started(data="x")]}, config=_config("t1"))

            assert seen == [True]

        async def it_applies_to_astream_events():
            saver, seen = SlowSaver(), []
            graph = _graph(saver, seen)

            async for _ in graph.astream_events(
                Started(data="x"), config=_config("t1")
            ):
                pass

            assert seen == [True]

    def when_the_run_passes_durability():
        def it_uses_the_durability_of_the_run():
            saver, seen = SlowSaver(), []
            graph = _graph(saver, seen)

            graph.invoke(Started(data="x"), config=_config("t1"), durability="async")

            assert seen == [False]
