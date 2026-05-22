"""Tests for annotation-driven required ``ScalarReducer`` values.

When a handler declares a reducer parameter whose type annotation rejects
``None`` (e.g. ``strategy: str``), the framework must raise
``ReducerNotSetError`` if the channel value is ``None`` at injection time.
``str | None`` / ``Optional[str]`` / ``Any`` / no-annotation opt out.
"""

# Note: ``from __future__ import annotations`` is intentionally omitted so the
# handler annotations defined inside ``describe_/when_/it_`` blocks resolve
# from real classes (not strings) when ``typing.get_type_hints`` runs at graph
# build time.

from typing import Any, Optional

import pytest

import langgraph_events
from langgraph_events import (
    Command,
    DomainEvent,
    EventGraph,
    HandlerRaised,
    IntegrationEvent,
    Namespace,
    Reducer,
    ReducerNotSetError,
    ScalarReducer,
    on,
)


# Module-level events so handler type hints resolve at runtime.
class Trigger(IntegrationEvent):
    pass


class StrategyChosen(IntegrationEvent):
    name: str = ""


class TempoChosen(IntegrationEvent):
    bpm: int = 0


class Done(IntegrationEvent):
    pass


# Namespace with a declarative ScalarReducer (auto-discovered by EventGraph)
# plus a Command-with-inline-handle (for the Command.handle path) and a plain
# DomainEvent (for the non-Command reactor path).
class Track(Namespace):
    current_strategy = ScalarReducer(
        event_type=StrategyChosen,
        fn=lambda e: e.name,
    )

    class Acknowledged(DomainEvent):
        msg: str = ""

    class Play(Command):
        title: str = ""

        class Played(DomainEvent):
            title: str = ""

        def handle(self, current_strategy: str) -> "Track.Play.Played":
            return Track.Play.Played(title=self.title)


def _make_strategy_reducer() -> ScalarReducer:
    return ScalarReducer("strategy", event_type=StrategyChosen, fn=lambda e: e.name)


def describe_scalar_reducer_required_value():
    def describe_ReducerNotSetError():
        def it_subclasses_ValueError():
            assert issubclass(ReducerNotSetError, ValueError)

        def it_is_exported_from_top_level():
            assert hasattr(langgraph_events, "ReducerNotSetError")
            assert "ReducerNotSetError" in langgraph_events.__all__

    def when_annotation_rejects_none():
        def when_value_is_none():
            def it_raises_ReducerNotSetError():
                strategy = _make_strategy_reducer()

                @on(Trigger)
                def handler(event: Trigger, strategy: str) -> Done:
                    return Done()

                graph = EventGraph([handler], reducers=[strategy])

                with pytest.raises(ReducerNotSetError):
                    graph.invoke([Trigger()])

            def it_message_names_handler_reducer_and_event_type():
                strategy = _make_strategy_reducer()

                @on(Trigger)
                def named_handler(event: Trigger, strategy: str) -> Done:
                    return Done()

                graph = EventGraph([named_handler], reducers=[strategy])

                with pytest.raises(ReducerNotSetError) as exc_info:
                    graph.invoke([Trigger()])

                message = str(exc_info.value)
                assert "named_handler" in message
                assert "strategy" in message
                assert "StrategyChosen" in message

            def when_handler_declares_raises_for_a_supertype():
                def it_still_propagates():
                    # ReducerNotSetError is a ValueError subclass; a handler
                    # that declares raises=ValueError would normally catch it
                    # if it were raised inside the handler body. The framework
                    # raises in _build_inject — outside the try/except meta.raises
                    # boundary — so it must propagate even with a catcher in place.
                    strategy = _make_strategy_reducer()

                    @on(Trigger, raises=ValueError)
                    def handler(event: Trigger, strategy: str) -> Done:
                        return Done()

                    @on(HandlerRaised, exception=ValueError)
                    def catcher(event: HandlerRaised) -> Done:
                        return Done()

                    graph = EventGraph([handler, catcher], reducers=[strategy])

                    with pytest.raises(ReducerNotSetError):
                        graph.invoke([Trigger()])

            def it_raises_in_async_handlers():
                strategy = _make_strategy_reducer()

                @on(Trigger)
                async def handler(event: Trigger, strategy: str) -> Done:
                    return Done()

                graph = EventGraph([handler], reducers=[strategy])

                import asyncio

                with pytest.raises(ReducerNotSetError):
                    asyncio.run(graph.ainvoke([Trigger()]))

    def when_annotation_accepts_none():
        def it_does_not_raise_for_pep604_optional_union():
            strategy = _make_strategy_reducer()

            @on(Trigger)
            def handler(event: Trigger, strategy: str | None) -> Done:
                assert strategy is None
                return Done()

            graph = EventGraph([handler], reducers=[strategy])
            graph.invoke([Trigger()])  # must not raise

        def it_does_not_raise_for_typing_Optional():
            strategy = _make_strategy_reducer()

            @on(Trigger)
            def handler(event: Trigger, strategy: Optional[str]) -> Done:  # noqa: UP045
                assert strategy is None
                return Done()

            graph = EventGraph([handler], reducers=[strategy])
            graph.invoke([Trigger()])

        def it_does_not_raise_for_Any():
            strategy = _make_strategy_reducer()

            @on(Trigger)
            def handler(event: Trigger, strategy: Any) -> Done:
                assert strategy is None
                return Done()

            graph = EventGraph([handler], reducers=[strategy])
            graph.invoke([Trigger()])

        def it_does_not_raise_for_object():
            strategy = _make_strategy_reducer()

            @on(Trigger)
            def handler(event: Trigger, strategy: object) -> Done:
                assert strategy is None
                return Done()

            graph = EventGraph([handler], reducers=[strategy])
            graph.invoke([Trigger()])

        def it_does_not_raise_for_missing_annotation():
            strategy = _make_strategy_reducer()

            @on(Trigger)
            def handler(event: Trigger, strategy) -> Done:
                assert strategy is None
                return Done()

            graph = EventGraph([handler], reducers=[strategy])
            graph.invoke([Trigger()])

    def when_value_is_set():
        def it_does_not_raise_after_event_projects_a_value():
            strategy = _make_strategy_reducer()

            @on(Trigger)
            def handler(event: Trigger, strategy: str) -> Done:
                assert strategy == "fast"
                return Done()

            graph = EventGraph([handler], reducers=[strategy])
            graph.invoke([StrategyChosen(name="fast"), Trigger()])

        def when_default_is_non_none():
            def it_does_not_raise():
                strategy = ScalarReducer(
                    "strategy",
                    event_type=StrategyChosen,
                    fn=lambda e: e.name,
                    default="draft",
                )

                @on(Trigger)
                def handler(event: Trigger, strategy: str) -> Done:
                    assert strategy == "draft"
                    return Done()

                graph = EventGraph([handler], reducers=[strategy])
                graph.invoke([Trigger()])

    def when_param_is_a_list_reducer():
        def it_does_not_raise_for_concrete_list_annotation():
            # ``Reducer`` channels initialise to ``[]``, never to ``None`` —
            # so a ``list[str]`` annotation never triggers the assertion.
            items = Reducer(
                "items",
                event_type=StrategyChosen,
                fn=lambda e: [e.name],
            )

            @on(Trigger)
            def handler(event: Trigger, items: list[str]) -> Done:
                assert items == []
                return Done()

            graph = EventGraph([handler], reducers=[items])
            graph.invoke([Trigger()])

    def when_handler_has_multiple_reducer_params():
        def when_one_required_is_none_and_one_is_permissive():
            def it_raises_only_for_the_required_one():
                strategy = _make_strategy_reducer()
                tempo = ScalarReducer(
                    "tempo", event_type=TempoChosen, fn=lambda e: e.bpm
                )

                @on(Trigger)
                def handler(event: Trigger, strategy: str, tempo: int | None) -> Done:
                    return Done()

                graph = EventGraph([handler], reducers=[strategy, tempo])

                with pytest.raises(ReducerNotSetError) as exc_info:
                    graph.invoke([Trigger()])

                # The required param is named; the permissive one is not.
                assert "strategy" in str(exc_info.value)
                assert "tempo" not in str(exc_info.value)

        def when_required_is_set_and_permissive_is_none():
            def it_runs_cleanly():
                strategy = _make_strategy_reducer()
                tempo = ScalarReducer(
                    "tempo", event_type=TempoChosen, fn=lambda e: e.bpm
                )

                @on(Trigger)
                def handler(event: Trigger, strategy: str, tempo: int | None) -> Done:
                    assert strategy == "fast"
                    assert tempo is None
                    return Done()

                graph = EventGraph([handler], reducers=[strategy, tempo])
                graph.invoke([StrategyChosen(name="fast"), Trigger()])

    def when_handler_is_a_command_handle_method():
        def when_inline_handle_param_rejects_none():
            def it_raises_ReducerNotSetError():
                # Track.Play.handle (synthesized into @on(Track.Play)) declares
                # `current_strategy: str` — the assertion must fire for the
                # inline-Command path just like for a plain function handler.
                graph = EventGraph([Track.Play])

                with pytest.raises(ReducerNotSetError):
                    graph.invoke([Track.Play(title="song")])

    def when_reducer_is_declared_on_a_namespace():
        def when_handler_param_rejects_none():
            def it_raises_ReducerNotSetError():
                # A non-Command reactor subscribing to a Track-namespaced
                # event triggers _discover_namespace_reducers, which auto-
                # registers Track.current_strategy. The reactor's required
                # annotation then raises when no StrategyChosen has fired.
                @on(Track.Acknowledged)
                def react(event: Track.Acknowledged, current_strategy: str) -> Done:
                    return Done()

                graph = EventGraph([react])

                with pytest.raises(ReducerNotSetError):
                    graph.invoke([Track.Acknowledged(msg="x")])
