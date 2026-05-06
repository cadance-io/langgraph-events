"""Tests for declarative reducers on Namespace classes."""

from __future__ import annotations

import pytest

from langgraph_events import (
    Command,
    DomainEvent,
    Event,
    EventGraph,
    IntegrationEvent,
    Namespace,
    ScalarReducer,
    on,
)


# Module-level domains so handler type hints resolve at runtime.
class Alpha(Namespace):
    current_status = ScalarReducer(
        event_type=Event,
        fn=lambda e: type(e).__name__,
    )

    class Act(Command):
        class Acted(DomainEvent):
            marker: str = ""


class Beta(Namespace):
    class Trigger(Command):
        class Triggered(DomainEvent):
            marker: str = ""


# Auto-discovery fixture — module-level so handler return-type hints resolve
# at runtime. ``_AUTO_OBSERVED`` is cleared at the start of each test that
# uses it so no leakage occurs between runs.
_AUTO_OBSERVED: list[str | None] = []


class _AlphaAuto(Namespace):
    current_status = ScalarReducer(
        event_type=Event,
        fn=lambda e: type(e).__name__,
    )

    class Act(Command):
        class Acted(DomainEvent):
            marker: str = ""

        def handle(self, current_status) -> _AlphaAuto.Act.Acted:
            _AUTO_OBSERVED.append(current_status)
            return _AlphaAuto.Act.Acted(marker="after")


class _AlphaExplicit(Namespace):
    current_status = ScalarReducer(
        event_type=Event,
        fn=lambda e: type(e).__name__,
    )

    class Act(Command):
        class Acted(DomainEvent):
            marker: str = ""

        def handle(self, current_status) -> _AlphaExplicit.Act.Acted:
            # Explicit list-reducer wins over discovered scalar reducer.
            assert current_status == ["EXPLICIT"]
            return _AlphaExplicit.Act.Acted(marker="done")


def describe_BaseReducer():

    def describe___set_name__():

        def when_declared_on_domain():

            def it_fills_name_from_attribute_name():
                assert Alpha.current_status.name == "current_status"

            def it_fills_domain_from_owner_class():
                assert Alpha.current_status.namespace is Alpha

        def when_declared_outside_domain():

            def it_leaves_domain_as_None():
                external = ScalarReducer(
                    name="external",
                    event_type=Event,
                    fn=lambda e: e,
                )
                assert external.namespace is None

        def when_name_is_explicitly_set():

            def it_preserves_explicit_name():
                class HasExplicitName(Namespace):
                    r = ScalarReducer(
                        name="custom",
                        event_type=Event,
                        fn=lambda e: e,
                    )

                assert HasExplicitName.r.name == "custom"


def describe_Namespace():

    def describe___reducers__():

        def when_class_body_has_reducer_attributes():

            def it_collects_them_into_tuple():
                class WithReducers(Namespace):
                    r1 = ScalarReducer(event_type=Event, fn=lambda e: None)
                    r2 = ScalarReducer(event_type=Event, fn=lambda e: None)

                names = {r.name for r in WithReducers.__reducers__}
                assert names == {"r1", "r2"}

        def when_no_reducer_attributes():

            def it_is_empty_tuple():
                class Empty(Namespace):
                    pass

                assert Empty.__reducers__ == ()

        def when_subclass_domain_adds_more():

            def it_inherits_parent_reducers_and_adds_child():
                class Parent(Namespace):
                    shared = ScalarReducer(event_type=Event, fn=lambda e: None)

                class Child(Parent):
                    extra = ScalarReducer(event_type=Event, fn=lambda e: None)

                names = {r.name for r in Child.__reducers__}
                assert names == {"shared", "extra"}


def describe_namespace_scope_filter():
    """Reducers with namespace= only contribute when the event matches."""

    def when_event_belongs_to_domain():

        def it_contributes_to_collect():
            result = Alpha.current_status.collect([Alpha.Act.Acted(marker="x")])
            assert result == "Acted"

    def when_event_belongs_to_different_domain():

        def it_does_not_contribute():
            from langgraph_events._reducer import SKIP

            result = Alpha.current_status.collect([Beta.Trigger.Triggered(marker="x")])
            assert result is SKIP

    def when_event_has_no_domain():

        def it_does_not_contribute():
            class LooseEvent(IntegrationEvent):
                pass

            from langgraph_events._reducer import SKIP

            result = Alpha.current_status.collect([LooseEvent()])
            assert result is SKIP

    def when_subclass_event_inherits_domain():

        def it_contributes():
            class FastActed(Alpha.Act.Acted):
                pass

            result = Alpha.current_status.collect([FastActed(marker="x")])
            assert result == "FastActed"


def describe_EventGraph_auto_discovery():

    def when_handler_subscribes_to_domain_event():

        def it_auto_registers_domain_reducers():
            _AUTO_OBSERVED.clear()
            # No explicit reducers= — must be auto-discovered from _AlphaAuto.
            graph = EventGraph([_AlphaAuto.Act])
            graph.invoke(_AlphaAuto.Act())

            # The seed _AlphaAuto.Act is an _AlphaAuto event, so the reducer
            # sees it and projects type(e).__name__ == "Act" before the
            # handler runs.
            assert _AUTO_OBSERVED == ["Act"]

    def when_reducer_names_collide_across_domains():

        def it_raises_TypeError_at_init():
            class Gamma(Namespace):
                shared_name = ScalarReducer(event_type=Event, fn=lambda e: None)

                class Go(Command):
                    pass

            class Delta(Namespace):
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

            graph = EventGraph([_AlphaExplicit.Act], reducers=[explicit])
            graph.invoke(_AlphaExplicit.Act())
