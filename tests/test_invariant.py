"""Tests for invariants= kwarg runtime dispatch + InvariantViolated emission."""

from __future__ import annotations

import pytest
from conftest import Order

from langgraph_events import (
    Command,
    DomainEvent,
    EventGraph,
    HandlerRaised,
    IntegrationEvent,
    Invariant,
    InvariantViolated,
    Namespace,
    Scatter,
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


# ---------------------------------------------------------------------------
# Post-command invariant check — scenarios exercise the post-handler gate
# that runs against log + emitted events.  A Ledger domain with numeric
# amounts gives us an invariant whose result actually depends on what the
# handler emits (a pure "log.has()" predicate can't distinguish pre vs post).
# ---------------------------------------------------------------------------


class Ledger(Namespace):
    class Deposit(Command):
        amount: int = 0

        class Deposited(DomainEvent):
            amount: int = 0


class OverBudget(Invariant):
    pass


class PostBoom(Invariant):
    pass


class AlwaysHolds(Invariant):
    pass


def _deposit_with(invariants, *, async_handler=False):
    """Build a Ledger.Deposit handler with the given invariants= dict."""
    if async_handler:

        @on(Ledger.Deposit, invariants=invariants)
        async def deposit(event: Ledger.Deposit) -> Ledger.Deposit.Deposited:
            return Ledger.Deposit.Deposited(amount=event.amount)

        return deposit

    @on(Ledger.Deposit, invariants=invariants)
    def deposit(event: Ledger.Deposit) -> Ledger.Deposit.Deposited:
        return Ledger.Deposit.Deposited(amount=event.amount)

    return deposit


def _total_under(limit):
    return lambda log: (
        sum(e.amount for e in log.filter(Ledger.Deposit.Deposited)) < limit
    )


def describe_post_command_invariant_check():
    """After the handler returns, re-check invariants against log + emitted.

    On failure, the emitted events are dropped and a single
    ``InvariantViolated`` is committed in their place with ``would_emit``.
    """

    def when_handler_emits_violating_event():

        def it_emits_InvariantViolated_in_place_of_the_event():
            deposit = _deposit_with({OverBudget: _total_under(100)})
            graph = EventGraph([deposit])
            log = graph.invoke([Ledger.Deposit(amount=60), Ledger.Deposit(amount=60)])
            # First deposit commits (total 60 < 100). Second would push total to
            # 120; post-check catches it and rolls back.
            assert log.count(Ledger.Deposit.Deposited) == 1
            assert log.count(InvariantViolated) == 1

        def it_populates_would_emit():
            deposit = _deposit_with({OverBudget: _total_under(100)})
            graph = EventGraph([deposit])
            log = graph.invoke([Ledger.Deposit(amount=60), Ledger.Deposit(amount=60)])
            v = log.latest(InvariantViolated)
            assert v is not None
            assert isinstance(v.invariant, OverBudget)
            assert len(v.would_emit) == 1
            assert isinstance(v.would_emit[0], Ledger.Deposit.Deposited)
            assert v.would_emit[0].amount == 60

        def it_does_not_commit_the_violating_event_to_the_log():
            deposit = _deposit_with({OverBudget: _total_under(100)})
            graph = EventGraph([deposit])
            log = graph.invoke([Ledger.Deposit(amount=60), Ledger.Deposit(amount=60)])
            deposits = log.filter(Ledger.Deposit.Deposited)
            assert len(deposits) == 1
            assert deposits[0].amount == 60

    def when_handler_emits_non_violating_event():

        def it_commits_the_event_normally():
            deposit = _deposit_with({OverBudget: _total_under(100)})
            graph = EventGraph([deposit])
            log = graph.invoke(Ledger.Deposit(amount=10))
            assert log.count(Ledger.Deposit.Deposited) == 1

        def it_does_not_emit_violation():
            deposit = _deposit_with({OverBudget: _total_under(100)})
            graph = EventGraph([deposit])
            log = graph.invoke(Ledger.Deposit(amount=10))
            assert not log.has(InvariantViolated)

    def when_handler_scatters_violating_buffer():

        def _build_graph():
            @on(
                Ledger.Deposit,
                invariants={OverBudget: _total_under(100)},
            )
            def bulk_deposit(event: Ledger.Deposit) -> Scatter:
                # One Deposit command fans out into three Deposited events;
                # combined total 120 exceeds the limit.
                return Scatter(
                    [
                        Ledger.Deposit.Deposited(amount=40),
                        Ledger.Deposit.Deposited(amount=40),
                        Ledger.Deposit.Deposited(amount=40),
                    ]
                )

            return EventGraph([bulk_deposit])

        def it_drops_the_full_scatter_expansion():
            log = _build_graph().invoke(Ledger.Deposit(amount=0))
            assert log.count(Ledger.Deposit.Deposited) == 0
            assert log.count(InvariantViolated) == 1

        def it_lists_all_scatter_events_in_would_emit():
            log = _build_graph().invoke(Ledger.Deposit(amount=0))
            v = log.latest(InvariantViolated)
            assert v is not None
            assert len(v.would_emit) == 3
            assert all(isinstance(e, Ledger.Deposit.Deposited) for e in v.would_emit)

    def when_handler_returns_None():

        def it_skips_post_check():
            calls: list[str] = []

            def predicate(log):
                calls.append("called")
                return True

            @on(Ledger.Deposit, invariants={AlwaysHolds: predicate})
            def noop(event: Ledger.Deposit) -> None:
                return None

            graph = EventGraph([noop])
            graph.invoke(Ledger.Deposit(amount=10))
            # Only the pre-check ran (1 call).  Post-check bypassed because
            # the emitted buffer was empty.
            assert calls == ["called"]

    def when_no_invariants_declared():

        def it_bypasses_post_check_entirely():
            # Plain handler with no invariants — post-check must be a no-op.
            # Verified indirectly: handler runs and commits normally with no
            # violation events ever emitted.
            @on(Ledger.Deposit)
            def deposit(event: Ledger.Deposit) -> Ledger.Deposit.Deposited:
                return Ledger.Deposit.Deposited(amount=event.amount)

            graph = EventGraph([deposit])
            log = graph.invoke(Ledger.Deposit(amount=999))
            assert log.count(Ledger.Deposit.Deposited) == 1
            assert not log.has(InvariantViolated)

    def when_handler_is_async():

        def it_applies_post_check_on_the_async_path():
            deposit = _deposit_with({OverBudget: _total_under(100)}, async_handler=True)
            graph = EventGraph([deposit])
            log = graph.invoke([Ledger.Deposit(amount=60), Ledger.Deposit(amount=60)])
            assert log.count(Ledger.Deposit.Deposited) == 1
            assert log.count(InvariantViolated) == 1
            v = log.latest(InvariantViolated)
            assert isinstance(v.invariant, OverBudget)
            assert len(v.would_emit) == 1

    def when_multiple_events_match_in_one_node_call():

        def it_rolls_back_only_the_violating_handler_call():
            # Two Deposit events in the same node call.  First (60) commits;
            # second (60) pushes total to 120 and gets rolled back.  This
            # proves per-handler-call isolation within the loop.
            deposit = _deposit_with({OverBudget: _total_under(100)})
            graph = EventGraph([deposit])
            log = graph.invoke([Ledger.Deposit(amount=60), Ledger.Deposit(amount=60)])
            # Exactly one committed, exactly one rolled back.
            assert log.count(Ledger.Deposit.Deposited) == 1
            assert log.count(InvariantViolated) == 1
            v = log.latest(InvariantViolated)
            # would_emit contains only the second call's event, not both.
            assert len(v.would_emit) == 1

    def when_pre_check_already_failed():

        def it_does_not_run_post_check():
            # Pre-check fails → handler skipped → post-check sees empty
            # emitted buffer → returns None trivially.  Only one violation
            # emitted (from pre-check), not two.
            deposit = _deposit_with({OverBudget: lambda log: False})
            graph = EventGraph([deposit])
            log = graph.invoke(Ledger.Deposit(amount=10))
            assert log.count(InvariantViolated) == 1
            assert log.count(Ledger.Deposit.Deposited) == 0
            v = log.latest(InvariantViolated)
            # Pre-check failure has empty would_emit (handler never ran).
            assert v.would_emit == ()

    def when_later_invariant_fails_post_check():

        def it_short_circuits_and_emits_one_violation():
            calls: list[str] = []

            def always_true(log):
                calls.append("first")
                return True

            def fails_after_emit(log):
                calls.append("second")
                # Fails only when a Deposited is in the (simulated) log.
                return not log.has(Ledger.Deposit.Deposited)

            deposit = _deposit_with(
                {AlwaysHolds: always_true, PostBoom: fails_after_emit}
            )
            graph = EventGraph([deposit])
            log = graph.invoke(Ledger.Deposit(amount=10))
            violations = log.filter(InvariantViolated)
            assert len(violations) == 1
            assert isinstance(violations[0].invariant, PostBoom)

        def it_names_the_first_failing_invariant_class():
            # Two post-check-failing invariants: order determines which fires.
            deposit = _deposit_with(
                {
                    OverBudget: lambda log: False,
                    PostBoom: lambda log: False,
                }
            )
            graph = EventGraph([deposit])
            log = graph.invoke(Ledger.Deposit(amount=10))
            # Pre-check actually catches OverBudget first (both already fail
            # against empty log), so the scenario degrades to pre-check
            # semantics.  Assert the first-declared invariant class is the
            # one that fires — consistent with pre-check ordering.
            v = log.latest(InvariantViolated)
            assert isinstance(v.invariant, OverBudget)

    def when_predicate_raises_in_post_check():

        def it_propagates_the_exception():
            # Predicate passes on empty log (pre-check), raises when it sees
            # a Deposited in the simulated log (post-check).
            def post_only_boom(log):
                if log.has(Ledger.Deposit.Deposited):
                    raise _PredicateError("post-check bug")
                return True

            deposit = _deposit_with({PostBoom: post_only_boom})
            graph = EventGraph([deposit])
            with pytest.raises(_PredicateError, match="post-check bug"):
                graph.invoke(Ledger.Deposit(amount=10))

    def when_reactor_is_pinned_to_invariant_class():

        def it_fires_on_post_check_violations_too():
            # @on(InvariantViolated, invariant=OverBudget) must catch BOTH
            # pre and post failures without distinguishing between them.
            deposit = _deposit_with({OverBudget: _total_under(100)})

            seen: list[tuple[int, tuple[int, ...]]] = []

            @on(InvariantViolated, invariant=OverBudget)
            def react(event: InvariantViolated) -> None:
                amounts = tuple(e.amount for e in event.would_emit)
                seen.append((len(event.would_emit), amounts))

            graph = EventGraph([deposit, react])
            graph.invoke([Ledger.Deposit(amount=60), Ledger.Deposit(amount=60)])
            # One post-check violation, reactor saw the rolled-back event.
            assert seen == [(1, (60,))]
