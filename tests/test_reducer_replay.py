"""A reducer channel equals its reducer advanced over the whole log."""

from __future__ import annotations

from typing import Any

import pytest
from conftest import Noted, keyed_merge
from langgraph.checkpoint.memory import InMemorySaver

from langgraph_events import (
    SKIP,
    EventGraph,
    FoldReducer,
    IntegrationEvent,
    Interrupted,
    Reducer,
    ScalarReducer,
    SystemEvent,
    on,
)

CONFIG: Any = {"configurable": {"thread_id": "t1"}}


class Ticked(IntegrationEvent):
    n: int = 0


class Pinged(IntegrationEvent):
    hops: int = 0


class Asked(IntegrationEvent):
    pass


class AskReview(Interrupted):
    pass


@on(Noted)
def tick(event: Noted) -> Ticked:
    return Ticked(n=len(event.text))


@on(Pinged)
def pong(event: Pinged) -> Pinged:
    return Pinged(hops=event.hops + 1)


@on(Asked)
def ask(event: Asked) -> AskReview:
    return AskReview()


def _reducers() -> list[Any]:
    return [
        Reducer(
            "notes",
            event_type=Noted,
            fn=lambda e: [[e.key, e.text]],
            reducer=keyed_merge,
        ),
        ScalarReducer(
            "even", event_type=Ticked, fn=lambda e: e.n if e.n % 2 == 0 else SKIP
        ),
        FoldReducer(
            "total", event_type=Ticked, default_factory=int, fold=lambda s, e: s + e.n
        ),
        Reducer("system", event_type=SystemEvent, fn=lambda e: [type(e).__name__]),
    ]


def _live_graph() -> EventGraph:
    return EventGraph(
        [tick, pong, ask],
        reducers=_reducers(),
        checkpointer=InMemorySaver(),
        max_rounds=3,
    )


def _replayed(graph: EventGraph) -> dict[str, Any]:
    events = list(graph.get_state(CONFIG).events)
    return {r.name: r.advance(r.empty, events) for r in _reducers()}


def _live(graph: EventGraph) -> dict[str, Any]:
    values = graph.compiled.get_state(CONFIG).values
    return {r.name: values[r.name] for r in _reducers()}


def describe_reducer_channels():
    def when_a_run_pauses_halts_and_continues():
        @pytest.mark.parametrize("durability", ["exit", "sync"])
        def it_equals_the_reducers_advanced_over_the_log(durability):
            graph = _live_graph()
            seeds: list[tuple[Any, dict[str, Any]]] = [
                (Noted(key="a", text="xx"), {}),
                (Noted(key="a", text="yyy"), {}),
                ([Noted(key="b", text="z"), Ticked(n=4)], {}),
                (Noted(key="c", text="ww"), {"deadline": 0.0}),
                (Noted(key="d", text="vvv"), {}),
                (Pinged(), {}),
                (Noted(key="e", text="u"), {}),
            ]
            for seed, kwargs in seeds:
                graph.invoke(seed, config=CONFIG, durability=durability, **kwargs)
            live = _live(graph)
            assert live["system"].count("RunPaused") == 1
            assert live["system"].count("MaxRoundsExceeded") == 1
            assert live == _replayed(graph)

    def when_a_paused_thread_is_abandoned():
        def it_folds_the_abandoned_event():
            graph = _live_graph()
            graph.invoke(Asked(), config=CONFIG)
            graph.abandon(CONFIG, reason="retired")
            live = _live(graph)
            assert live["system"][-1] == "Abandoned"
            assert live == _replayed(graph)


def describe_ScalarReducer():
    def when_the_newest_event_is_skipped():
        def it_keeps_the_newest_value_fn_does_not_skip():
            even = ScalarReducer(
                "even", event_type=Ticked, fn=lambda e: e.n if e.n % 2 == 0 else SKIP
            )
            assert even.collect([Ticked(n=2), Ticked(n=4), Ticked(n=1)]) == 4
