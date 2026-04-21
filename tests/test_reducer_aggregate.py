"""Tests for declarative reducers on Aggregate classes."""

from __future__ import annotations

import pytest

from langgraph_events import (
    Aggregate,
    Command,
    DomainEvent,
    Event,
    EventGraph,
    IntegrationEvent,
    ScalarReducer,
    on,
)


# Module-level aggregates so handler type hints resolve at runtime.
class Alpha(Aggregate):
    current_status = ScalarReducer(
        event_type=Event,
        fn=lambda e: type(e).__name__,
    )

    class Act(Command):
        class Acted(DomainEvent):
            marker: str = ""


class Beta(Aggregate):
    class Trigger(Command):
        class Triggered(DomainEvent):
            marker: str = ""


def describe_BaseReducer():

    def describe___set_name__():

        def when_declared_on_aggregate():

            def it_fills_name_from_attribute_name():
                assert Alpha.current_status.name == "current_status"

            def it_fills_aggregate_from_owner_class():
                assert Alpha.current_status.aggregate is Alpha

        def when_declared_outside_aggregate():

            def it_leaves_aggregate_as_None():
                external = ScalarReducer(
                    name="external",
                    event_type=Event,
                    fn=lambda e: e,
                )
                assert external.aggregate is None

        def when_name_is_explicitly_set():

            def it_preserves_explicit_name():
                class HasExplicitName(Aggregate):
                    r = ScalarReducer(
                        name="custom",
                        event_type=Event,
                        fn=lambda e: e,
                    )

                assert HasExplicitName.r.name == "custom"


def describe_Aggregate():

    def describe___reducers__():

        def when_class_body_has_reducer_attributes():

            def it_collects_them_into_tuple():
                class WithReducers(Aggregate):
                    r1 = ScalarReducer(event_type=Event, fn=lambda e: None)
                    r2 = ScalarReducer(event_type=Event, fn=lambda e: None)

                names = {r.name for r in WithReducers.__reducers__}
                assert names == {"r1", "r2"}

        def when_no_reducer_attributes():

            def it_is_empty_tuple():
                class Empty(Aggregate):
                    pass

                assert Empty.__reducers__ == ()

        def when_subclass_aggregate_adds_more():

            def it_inherits_parent_reducers_and_adds_child():
                class Parent(Aggregate):
                    shared = ScalarReducer(event_type=Event, fn=lambda e: None)

                class Child(Parent):
                    extra = ScalarReducer(event_type=Event, fn=lambda e: None)

                names = {r.name for r in Child.__reducers__}
                assert names == {"shared", "extra"}


def describe_aggregate_scope_filter():
    """Reducers with aggregate= only contribute when the event matches."""

    def when_event_belongs_to_aggregate():

        def it_contributes_to_collect():
            result = Alpha.current_status.collect([Alpha.Act.Acted(marker="x")])
            assert result == "Acted"

    def when_event_belongs_to_different_aggregate():

        def it_does_not_contribute():
            from langgraph_events._reducer import SKIP

            result = Alpha.current_status.collect([Beta.Trigger.Triggered(marker="x")])
            assert result is SKIP

    def when_event_has_no_aggregate():

        def it_does_not_contribute():
            class LooseEvent(IntegrationEvent):
                pass

            from langgraph_events._reducer import SKIP

            result = Alpha.current_status.collect([LooseEvent()])
            assert result is SKIP

    def when_subclass_event_inherits_aggregate():

        def it_contributes():
            class FastActed(Alpha.Act.Acted):
                pass

            result = Alpha.current_status.collect([FastActed(marker="x")])
            assert result == "FastActed"


def describe_EventGraph_auto_discovery():

    def when_handler_subscribes_to_aggregate_event():

        def it_auto_registers_aggregate_reducers():
            observed: list[str | None] = []

            @on(Alpha.Act)
            def run(event: Alpha.Act, current_status) -> Alpha.Act.Acted:
                observed.append(current_status)
                return Alpha.Act.Acted(marker="after")

            # No explicit reducers= — must be auto-discovered from Alpha.
            graph = EventGraph([run])
            graph.invoke(Alpha.Act())

            # The seed Alpha.Act is an Alpha event, so the reducer sees it
            # and projects type(e).__name__ == "Act" before the handler runs.
            assert observed == ["Act"]

    def when_reducer_names_collide_across_aggregates():

        def it_raises_TypeError_at_init():
            class Gamma(Aggregate):
                shared_name = ScalarReducer(event_type=Event, fn=lambda e: None)

                class Go(Command):
                    pass

            class Delta(Aggregate):
                shared_name = ScalarReducer(event_type=Event, fn=lambda e: None)

                class Go(Command):
                    pass

            @on(Gamma.Go)
            def g(event: Gamma.Go) -> None:
                return None

            @on(Delta.Go)
            def d(event: Delta.Go) -> None:
                return None

            with pytest.raises(TypeError, match=r"[Rr]educer.*shared_name.*collide"):
                EventGraph([g, d])

    def when_explicit_reducer_shares_discovered_name():

        def it_prefers_explicit_over_discovered():
            from langgraph_events import Reducer

            explicit = Reducer(
                name="current_status",
                event_type=Event,
                fn=lambda e: ["EXPLICIT"],
            )

            @on(Alpha.Act)
            def run(event: Alpha.Act, current_status) -> Alpha.Act.Acted:
                # Explicit list-reducer provides ["EXPLICIT"] regardless of events.
                assert current_status == ["EXPLICIT"]
                return Alpha.Act.Acted(marker="done")

            graph = EventGraph([run], reducers=[explicit])
            graph.invoke(Alpha.Act())
