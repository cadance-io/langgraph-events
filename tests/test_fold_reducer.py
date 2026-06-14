"""Tests for FoldReducer — folds matching events into accumulating state."""

from __future__ import annotations

from conftest import Completed, Started
from langgraph.checkpoint.memory import MemorySaver

from langgraph_events import EventGraph, IntegrationEvent, on

# Module-level events so handler/annotation forward refs resolve at runtime.
# Each event owns its transition via ``fold(self, state)`` — mirroring the
# polymorphic ``MessageEvent.as_messages()`` convention.


class Incremented(IntegrationEvent):
    by: int = 1

    def fold(self, state: dict) -> dict:
        return {"n": state["n"] + self.by}


class Reset(IntegrationEvent):
    def fold(self, state: dict):
        from langgraph_events import RESET

        return RESET


class Noop(IntegrationEvent):
    def fold(self, state: dict):
        from langgraph_events import SKIP

        return SKIP


class CursorMoved(IntegrationEvent):
    pos: int = 0

    def fold(self, state):
        # Demonstrates None as a legitimate stored state value.
        return None if self.pos < 0 else self.pos


def describe_FoldReducer():

    def describe_seed():

        def when_events_fold_into_accumulating_state():

            def it_folds_from_default_factory():
                from langgraph_events import FoldReducer

                counter = FoldReducer(
                    name="counter",
                    event_type=Incremented,
                    default_factory=lambda: {"n": 0},
                )
                result = counter.seed(
                    [Incremented(by=1), Incremented(by=2), Incremented(by=3)]
                )
                assert result == {"n": 6}

        def when_no_matching_events():

            def it_returns_default_factory_value():
                from langgraph_events import FoldReducer

                counter = FoldReducer(
                    name="counter",
                    event_type=Incremented,
                    default_factory=lambda: {"n": 0},
                )
                assert counter.seed([]) == {"n": 0}

    def describe_fold_default():

        def when_no_explicit_fold_given():

            def it_delegates_to_event_fold_method():
                from langgraph_events import FoldReducer

                counter = FoldReducer(
                    name="counter",
                    event_type=Incremented,
                    default_factory=lambda: {"n": 0},
                )
                assert counter.seed([Incremented(by=5)]) == {"n": 5}

    def describe_fold_override():

        def when_explicit_fold_given():

            def it_uses_the_supplied_fold_over_event_method():
                from langgraph_events import FoldReducer

                counter = FoldReducer(
                    name="counter",
                    event_type=Incremented,
                    default_factory=lambda: {"n": 0},
                    fold=lambda state, event: {"n": state["n"] + 100},
                )
                assert counter.seed([Incremented(by=1)]) == {"n": 100}

    def describe_RESET():

        def when_fold_returns_RESET():

            def it_clears_channel_to_default_factory():
                from langgraph_events import FoldReducer

                counter = FoldReducer(
                    name="counter",
                    event_type=(Incremented, Reset),
                    default_factory=lambda: {"n": 0},
                )
                result = counter.seed([Incremented(by=4), Reset(), Incremented(by=2)])
                assert result == {"n": 2}

    def describe_SKIP():

        def when_fold_returns_SKIP():

            def it_leaves_state_unchanged():
                from langgraph_events import FoldReducer

                counter = FoldReducer(
                    name="counter",
                    event_type=(Incremented, Noop),
                    default_factory=lambda: {"n": 0},
                )
                result = counter.seed([Incremented(by=3), Noop(), Incremented(by=1)])
                assert result == {"n": 4}

    def describe_None_as_value():

        def when_fold_returns_None():

            def it_stores_None_as_a_real_state_value():
                from langgraph_events import FoldReducer

                cursor = FoldReducer(
                    name="cursor",
                    event_type=CursorMoved,
                    default_factory=lambda: 0,
                )
                assert cursor.seed([CursorMoved(pos=5), CursorMoved(pos=-1)]) is None

    def describe_RESET_then_None():

        def when_a_reset_precedes_a_none_fold():

            def it_resets_then_stores_None():
                from langgraph_events import FoldReducer

                cursor = FoldReducer(
                    name="cursor",
                    event_type=(CursorMoved, Reset),
                    default_factory=lambda: 0,
                )
                # 0 -> 5 -> RESET(0) -> None: reset and None compose in one fold.
                result = cursor.seed([CursorMoved(pos=5), Reset(), CursorMoved(pos=-1)])
                assert result is None

    def describe_reducer_attribute():

        def when_accessed_by_the_streaming_shadow_path():

            def it_exposes_the_binary_merge():
                from langgraph_events import FoldReducer

                counter = FoldReducer(
                    name="counter",
                    event_type=(Incremented, Reset),
                    default_factory=lambda: {"n": 0},
                )
                # The streaming shadow folds per event via getattr(r, "reducer").
                merge = counter.reducer
                assert merge({"n": 1}, counter.collect([Incremented(by=2)])) == {"n": 3}


def describe_RESET_sentinel():

    def when_repr():

        def it_renders_as_RESET():
            from langgraph_events import RESET

            assert repr(RESET) == "RESET"


def describe_Foldable():

    def when_event_has_a_fold_method():

        def it_is_a_Foldable_instance():
            from langgraph_events import Foldable

            # Structural: Incremented never inherits Foldable, it just has fold.
            assert isinstance(Incremented(by=1), Foldable)

    def when_event_lacks_a_fold_method():

        def it_is_not_a_Foldable_instance():
            from langgraph_events import Foldable

            assert not isinstance(Started(data="x"), Foldable)


class Logged(IntegrationEvent):
    line: str = ""

    def fold(self, state: list) -> list:
        return [*state, self.line]


def _counter() -> object:
    from langgraph_events import FoldReducer

    return FoldReducer(
        name="counter",
        event_type=(Incremented, Reset),
        default_factory=lambda: {"n": 0},
    )


def describe_FoldReducer_in_a_graph():

    def describe_channel_merge_path():

        def when_handlers_produce_folding_events_across_rounds():

            def it_folds_produced_events_into_the_injected_state():
                captured: list[dict] = []

                @on(Started)
                def emit(event: Started) -> Incremented:
                    return Incremented(by=10)

                @on(Incremented)
                def capture(event: Incremented, counter: dict) -> Completed:
                    captured.append(counter)
                    return Completed(result="ok")

                graph = EventGraph([emit, capture], reducers=[_counter()])
                graph.invoke(Started(data="go"))
                assert captured == [{"n": 10}]

        def when_a_produced_event_resets_mid_run():

            def it_clears_the_channel_across_rounds():
                seen: list[dict] = []

                @on(Started)
                def emit(event: Started) -> Incremented:
                    return Incremented(by=5)

                @on(Incremented)
                def then_reset(event: Incremented, counter: dict) -> Reset:
                    seen.append(counter)
                    return Reset()

                @on(Reset)
                def finish(event: Reset, counter: dict) -> Completed:
                    seen.append(counter)
                    return Completed(result="ok")

                graph = EventGraph([emit, then_reset, finish], reducers=[_counter()])
                graph.invoke(Started(data="go"))
                assert seen == [{"n": 5}, {"n": 0}]

        def when_a_produced_event_folds_to_None():

            def it_injects_None_into_the_next_handler():
                from langgraph_events import FoldReducer

                captured: list[object] = []

                @on(Started)
                def emit(event: Started) -> CursorMoved:
                    return CursorMoved(pos=-1)  # folds to None

                @on(CursorMoved)
                def capture(event: CursorMoved, cursor: int | None) -> Completed:
                    captured.append(cursor)
                    return Completed(result="ok")

                cursor = FoldReducer(
                    name="cursor",
                    event_type=CursorMoved,
                    default_factory=lambda: 0,
                )
                graph = EventGraph([emit, capture], reducers=[cursor])
                graph.invoke(Started(data="go"))
                # None is a real stored state value end-to-end, not "unset".
                assert captured == [None]

    def describe_streaming_path():

        def when_streamed():

            def with_include_reducers():

                async def it_matches_the_invoke_result_per_event():
                    @on(Started)
                    def emit(event: Started) -> Incremented:
                        return Incremented(by=10)

                    @on(Incremented)
                    def stop(event: Incremented) -> Completed:
                        return Completed(result="ok")

                    graph = EventGraph([emit, stop], reducers=[_counter()])
                    frames = [
                        f
                        async for f in graph.astream_events(
                            Started(data="go"),
                            include_reducers=["counter"],
                        )
                    ]
                    # The streaming shadow folds the produced Incremented per
                    # event, converging on the channel path's value.
                    assert frames[-1].reducers["counter"] == {"n": 10}

    def describe_pre_seed_duality():

        def when_channel_pre_seeded():

            def with_a_folded_state():

                def it_folds_new_events_onto_the_pre_seeded_state():
                    captured: list[dict] = []

                    @on(Incremented)
                    def capture(event: Incremented, counter: dict) -> Completed:
                        captured.append(counter)
                        return Completed(result="ok")

                    graph = EventGraph(
                        [capture], reducers=[_counter()], checkpointer=MemorySaver()
                    )
                    config = {"configurable": {"thread_id": "fold-preseed"}}
                    graph.pre_seed(config, {"counter": {"n": 100}})
                    graph.invoke(Incremented(by=1), config=config)
                    assert captured[0] == {"n": 101}

        def when_fold_state_is_itself_a_list():

            def it_does_not_mistake_pre_seeded_list_for_events():
                from langgraph_events import FoldReducer

                captured: list[list] = []
                log = FoldReducer(name="log", event_type=Logged, default_factory=list)

                @on(Logged)
                def capture(event: Logged, log: list) -> Completed:
                    captured.append(log)
                    return Completed(result="ok")

                graph = EventGraph(
                    [capture], reducers=[log], checkpointer=MemorySaver()
                )
                config = {"configurable": {"thread_id": "fold-list"}}
                graph.pre_seed(config, {"log": ["seeded"]})
                graph.invoke(Logged(line="new"), config=config)
                assert captured[0] == ["seeded", "new"]

            def it_replaces_an_existing_list_channel_as_a_whole():
                # update_state on an already-populated channel writes a
                # pre-folded list as the merge "update". The merge must
                # REPLACE it, not iterate its elements as contribution
                # events — the exact corruption an isinstance(list)
                # discriminator would cause.
                from langgraph_events import FoldReducer

                captured: list[list] = []
                log = FoldReducer(name="log", event_type=Logged, default_factory=list)

                @on(Logged)
                def capture(event: Logged, log: list) -> Completed:
                    captured.append(log)
                    return Completed(result="ok")

                graph = EventGraph(
                    [capture], reducers=[log], checkpointer=MemorySaver()
                )
                config = {"configurable": {"thread_id": "fold-list-replace"}}
                graph.invoke(Logged(line="a"), config=config)
                graph.pre_seed(config, {"log": ["x", "y"]})
                graph.invoke(Logged(line="z"), config=config)
                assert captured[-1] == ["x", "y", "z"]

    def describe_replay_reducer():

        def when_rebuilding_from_an_event_log():

            def it_folds_the_events_from_default_factory():
                from langgraph_events.serde import replay_reducer

                rebuilt = replay_reducer(
                    _counter(), [Incremented(by=2), Reset(), Incremented(by=4)]
                )
                assert rebuilt == {"n": 4}


def describe_public_api():

    def when_importing_from_the_package():

        def it_exposes_BaseReducer_FoldReducer_and_RESET():
            import langgraph_events as le

            assert {"BaseReducer", "FoldReducer", "RESET"} <= set(le.__all__)
            from langgraph_events import (  # noqa: F401
                RESET,
                BaseReducer,
                FoldReducer,
            )
