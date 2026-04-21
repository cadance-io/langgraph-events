"""Tests for invariants= kwarg runtime dispatch + InvariantViolated emission."""

from __future__ import annotations

import pytest
from conftest import Order

from langgraph_events import (
    EventGraph,
    HandlerRaised,
    IntegrationEvent,
    Invariant,
    InvariantViolated,
    on,
)


class CustomerBanned(IntegrationEvent):
    customer_id: str = ""


class _PredicateError(Exception):
    pass


# Invariant marker classes defined at module level so every test can reuse
# the same identities for matching across decorator and reactor.
class Ok(Invariant):
    pass


class Blocked(Invariant):
    pass


class Buggy(Invariant):
    pass


class AFails(Invariant):
    pass


class BOk(Invariant):
    pass


class NoBannedCustomer(Invariant):
    pass


class Other(Invariant):
    pass


class Undeclared(Invariant):
    pass


def _place_with(invariants, *, async_handler=False):
    """Build a Place handler with the given invariants= dict. Reused across
    tests to avoid repeating the @on/return boilerplate in every case -- each
    test only varies the invariants argument (and occasionally async-ness)."""
    if async_handler:

        @on(Order.Place, invariants=invariants)
        async def place(event: Order.Place) -> Order.Place.Placed:
            return Order.Place.Placed(order_id="o1")

        return place

    @on(Order.Place, invariants=invariants)
    def place(event: Order.Place) -> Order.Place.Placed:
        return Order.Place.Placed(order_id="o1")

    return place


def describe_invariants():

    def describe_runtime_dispatch():

        def when_predicate_returns_true():

            def it_runs_handler_normally():
                graph = EventGraph([_place_with({Ok: lambda log: True})])
                log = graph.invoke(Order.Place(customer_id="c1"))
                assert log.has(Order.Place.Placed)
                assert not log.has(InvariantViolated)

        def when_predicate_returns_false():

            def it_skips_handler():
                graph = EventGraph([_place_with({Blocked: lambda log: False})])
                log = graph.invoke(Order.Place(customer_id="c1"))
                assert not log.has(Order.Place.Placed)

            def with_populated_fields():

                def it_emits_InvariantViolated():
                    graph = EventGraph([_place_with({Blocked: lambda log: False})])
                    log = graph.invoke(Order.Place(customer_id="c1"))
                    assert log.has(InvariantViolated)
                    v = log.latest(InvariantViolated)
                    assert v is not None
                    assert isinstance(v.invariant, Blocked)
                    assert v.handler == "place"
                    assert isinstance(v.source_event, Order.Place)
                    assert v.source_event.customer_id == "c1"

        def when_multiple_invariants_first_fails():

            def it_short_circuits_and_emits_one_violation():
                calls: list[str] = []

                def pred_a(log):
                    calls.append("a")
                    return False

                def pred_b(log):
                    calls.append("b")
                    return True

                graph = EventGraph([_place_with({AFails: pred_a, BOk: pred_b})])
                log = graph.invoke(Order.Place(customer_id="c1"))
                violations = log.filter(InvariantViolated)
                assert len(violations) == 1
                assert isinstance(violations[0].invariant, AFails)
                assert calls == ["a"]  # short-circuited, b never evaluated

        def when_predicate_raises_exception():

            def it_propagates_exception():
                def boom(log):
                    raise _PredicateError("predicate bug")

                graph = EventGraph([_place_with({Buggy: boom})])
                with pytest.raises(_PredicateError, match="predicate bug"):
                    graph.invoke(Order.Place(customer_id="c1"))

        def when_predicate_uses_event_log():

            def it_observes_prior_events_via_has():
                graph = EventGraph(
                    [
                        _place_with(
                            {NoBannedCustomer: lambda log: not log.has(CustomerBanned)}
                        )
                    ]
                )
                # Seed with a CustomerBanned before the Place command
                log = graph.invoke(
                    [CustomerBanned(customer_id="c1"), Order.Place(customer_id="c1")]
                )
                assert log.has(InvariantViolated)
                assert not log.has(Order.Place.Placed)

        def when_handler_also_declares_raises():

            def it_evaluates_invariant_before_handler_body():
                # If the invariant fails, the handler body never runs, so its
                # raised exception is never produced.
                @on(
                    Order.Place,
                    raises=_PredicateError,
                    invariants={Blocked: lambda log: False},
                )
                def place(event: Order.Place) -> Order.Place.Placed:
                    raise _PredicateError("would have run")

                @on(HandlerRaised, exception=_PredicateError)
                def caught(event: HandlerRaised) -> Order.Place.Rejected:
                    return Order.Place.Rejected(reason="should not see this")

                graph = EventGraph([place, caught])
                log = graph.invoke(Order.Place(customer_id="c1"))
                assert log.has(InvariantViolated)
                assert not log.has(HandlerRaised)
                assert not log.has(Order.Place.Rejected)

        def when_InvariantViolated_subscribed():

            def it_dispatches_to_subscriber():
                place = _place_with({Blocked: lambda log: False})

                @on(InvariantViolated)
                def react(event: InvariantViolated) -> Order.Place.Rejected:
                    return Order.Place.Rejected(reason=type(event.invariant).__name__)

                graph = EventGraph([place, react])
                log = graph.invoke(Order.Place(customer_id="c1"))
                assert log.has(Order.Place.Rejected)
                rejected = log.latest(Order.Place.Rejected)
                assert rejected is not None
                assert rejected.reason == "Blocked"

        def when_InvariantViolated_field_matcher_on_invariant():

            def it_only_fires_for_matching_invariant_class():
                place = _place_with(
                    {Blocked: lambda log: False, Other: lambda log: True}
                )

                seen: list[str] = []

                @on(InvariantViolated, invariant=Blocked)
                def only_blocked(event: InvariantViolated) -> None:
                    seen.append(type(event.invariant).__name__)

                @on(InvariantViolated, invariant=Other)
                def only_other(event: InvariantViolated) -> None:
                    seen.append(type(event.invariant).__name__)

                graph = EventGraph([place, only_blocked, only_other])
                graph.invoke(Order.Place(customer_id="c1"))
                assert seen == ["Blocked"]

        def when_handler_is_async():

            def with_failing_invariant():

                def it_evaluates_invariant_before_awaiting_handler():
                    place = _place_with(
                        {Blocked: lambda log: False}, async_handler=True
                    )
                    graph = EventGraph([place])
                    log = graph.invoke(Order.Place(customer_id="c1"))
                    assert log.has(InvariantViolated)
                    assert not log.has(Order.Place.Placed)

    def describe_coverage_drift_check():

        def when_reactor_references_undeclared_invariant():

            def it_raises_TypeError_at_compile_time():
                place = _place_with({Blocked: lambda log: False})

                @on(InvariantViolated, invariant=Undeclared)
                def never_fires(event: InvariantViolated) -> None:
                    pass

                with pytest.raises(TypeError, match="would never fire"):
                    EventGraph([place, never_fires]).invoke(
                        Order.Place(customer_id="c1")
                    )

        def when_reactor_is_catch_all():

            def it_is_not_flagged_by_coverage_check():
                place = _place_with({Blocked: lambda log: False})

                @on(InvariantViolated)
                def catch_all(event: InvariantViolated) -> None:
                    pass

                # No TypeError even though catch_all has no invariant= matcher.
                graph = EventGraph([place, catch_all])
                log = graph.invoke(Order.Place(customer_id="c1"))
                assert log.has(InvariantViolated)

        def when_no_reactor_exists():

            def it_allows_unhandled_violations_to_land_in_the_log():
                # No reactor at all — violation just sits in the log.
                graph = EventGraph([_place_with({Blocked: lambda log: False})])
                log = graph.invoke(Order.Place(customer_id="c1"))
                assert log.has(InvariantViolated)

        def when_reactor_has_invariant_plus_other_field_matchers():
            # Unlike raises= coverage (which conservatively ignores catchers
            # with extra matchers), invariant= coverage DOES apply even when
            # the reactor adds another matcher like handler=.

            def when_invariant_class_is_undeclared():

                def it_raises_TypeError_at_compile_time():
                    place = _place_with({Blocked: lambda log: False})

                    @on(InvariantViolated, invariant=Undeclared, handler="place")
                    def never_fires(event: InvariantViolated) -> None:
                        pass

                    with pytest.raises(TypeError, match="would never fire"):
                        EventGraph([place, never_fires]).invoke(
                            Order.Place(customer_id="c1")
                        )

            def when_invariant_class_is_declared():

                def it_constructs_cleanly():
                    place = _place_with({Blocked: lambda log: False})

                    @on(InvariantViolated, invariant=Blocked, handler="place")
                    def react(event: InvariantViolated) -> None:
                        pass

                    graph = EventGraph([place, react])
                    log = graph.invoke(Order.Place(customer_id="c1"))
                    assert log.has(InvariantViolated)

    def describe_decoration_time_validation():

        def when_invariant_subclass_requires_fields():

            def it_raises():
                class NeedsReason(Invariant):
                    def __init__(self, reason: str) -> None:
                        self.reason = reason

                with pytest.raises(TypeError, match=r"zero-arg instantiable"):

                    @on(Order.Place, invariants={NeedsReason: lambda log: False})
                    def place(event: Order.Place) -> Order.Place.Placed:
                        return Order.Place.Placed(order_id="o1")
