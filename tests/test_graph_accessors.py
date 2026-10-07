"""``EventGraph.reducers``, ``.checkpointer`` and the compiled input schema."""

from __future__ import annotations

import pytest
from conftest import Order
from langgraph.checkpoint.memory import InMemorySaver

from langgraph_events import EventGraph, FoldReducer, IntegrationEvent, on


class Tick(IntegrationEvent):
    pass


class Tock(IntegrationEvent):
    seen: int = 0


def _ticks() -> FoldReducer[int]:
    return FoldReducer(
        "ticks", event_type=Tick, default_factory=int, fold=lambda s, e: s + 1
    )


@on(Tick)
def guard(event: Tick, ticks: int) -> Tock:
    return Tock(seen=ticks)


@on(Order.Shipped)
def shipped(event: Order.Shipped) -> None:
    return None


def describe_EventGraph():
    def describe_reducers():
        def it_lists_passed_and_namespace_reducers():
            graph = EventGraph([guard, shipped], reducers=[_ticks()])
            assert set(graph.reducers) == {"ticks", "current_status"}

        def it_is_read_only():
            graph = EventGraph([guard], reducers=[_ticks()])
            with pytest.raises(TypeError):
                graph.reducers["other"] = _ticks()  # type: ignore[index]

    def describe_checkpointer():
        def it_returns_none_by_default():
            assert EventGraph([shipped]).checkpointer is None

        def it_returns_the_passed_saver():
            saver = InMemorySaver()
            assert EventGraph([shipped], checkpointer=saver).checkpointer is saver

    def describe_compiled_input():
        def it_takes_the_cursor_and_the_reducer_channels():
            graph = EventGraph([guard], reducers=[_ticks()])
            history = [Tick(), Tock(seen=1)]
            result = graph.compiled.invoke(
                {"events": [*history, Tick()], "_cursor": 2, "ticks": 10}
            )
            assert result["events"][-1] == Tock(seen=11)
