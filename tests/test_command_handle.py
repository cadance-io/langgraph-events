"""Tests for inline command handlers (``Command.handle`` + auto-registration)."""

from __future__ import annotations

import pytest

from langgraph_events import (
    Aggregate,
    Command,
    DomainEvent,
    EventGraph,
    EventLog,
    on,
)


# Module-level aggregate for inline-handler dispatch tests.
class Shop(Aggregate):
    class Buy(Command):
        item: str = ""

        class Bought(DomainEvent):
            item: str = ""
            price: float = 0.0

        class OutOfStock(DomainEvent):
            item: str = ""

        def handle(self) -> Shop.Buy.Bought | Shop.Buy.OutOfStock:
            if self.item == "dodo":
                return Shop.Buy.OutOfStock(item=self.item)
            return Shop.Buy.Bought(item=self.item, price=9.99)


class Shop2(Aggregate):
    class NoHandler(Command):
        class Outcome(DomainEvent):
            pass


class Shop3(Aggregate):
    class CmdA(Command):
        class DoneA(DomainEvent):
            pass

        def handle(self) -> Shop3.CmdA.DoneA:
            return Shop3.CmdA.DoneA()

    class CmdB(Command):
        class DoneB(DomainEvent):
            pass

        def handle(self) -> Shop3.CmdB.DoneB:
            return Shop3.CmdB.DoneB()

    class CmdNoHandle(Command):
        class NeverFires(DomainEvent):
            pass


class Shop4(Aggregate):
    class Slow(Command):
        item: str = ""

        class Done(DomainEvent):
            item: str = ""

        async def handle(self) -> Shop4.Slow.Done:
            return Shop4.Slow.Done(item=self.item)


class Shop5(Aggregate):
    class Tell(Command):
        class Told(DomainEvent):
            pass

        class OtherOutcome(DomainEvent):
            pass


class Foreign(Aggregate):
    class Do(Command):
        class Stuff(DomainEvent):
            pass


class WithLog(Aggregate):
    class Cmd(Command):
        class Done(DomainEvent):
            observed_count: int = 0

        def handle(self, log: EventLog) -> WithLog.Cmd.Done:
            return WithLog.Cmd.Done(observed_count=len(log))


# Module-level aggregates for inline-outcome-coverage tests. These can't live
# inside describe_/when_ blocks because Python can't resolve forward refs on
# handle's return annotation from a nested function scope.
class Shop6(Aggregate):
    class Buy(Command):
        class Bought(DomainEvent):
            pass

        class OutOfStock(DomainEvent):
            pass

        def handle(self) -> Shop6.Buy.Bought:
            return Shop6.Buy.Bought()


class Shop7(Aggregate):
    class Buy(Command):
        class Bought(DomainEvent):
            pass

        class OutOfStock(DomainEvent):
            pass

        def handle(self) -> Shop7.Buy.Bought | Shop7.Buy.OutOfStock:
            return Shop7.Buy.Bought()


class Shop8(Aggregate):
    class Buy(Command):
        class Bought(DomainEvent):
            pass

        class OutOfStock(DomainEvent):
            pass

        def handle(self):
            return Shop8.Buy.Bought()


class Shop9(Aggregate):
    class Buy(Command):
        class Bought(DomainEvent):
            pass

        class OutOfStock(DomainEvent):
            pass


@on(Shop9.Buy)
def shop9_partial(event: Shop9.Buy) -> Shop9.Buy.Bought:
    return Shop9.Buy.Bought()


class Shop10(Aggregate):
    class Buy(Command):
        class Bought(DomainEvent):
            pass

        class OutOfStock(DomainEvent):
            pass

        def handle(self) -> Shop10.Buy.Bought | Shop10.Buy.OutOfStock | None:
            return None


def describe_Command_handle():

    def describe_class_creation():

        def when_command_has_handle_method():

            def it_stamps___command_handler__():
                assert Shop.Buy.__command_handler__ is Shop.Buy.__dict__["handle"]

        def when_command_has_no_handle_method():

            def it_leaves___command_handler___as_None():
                assert Shop2.NoHandler.__command_handler__ is None

        def when_handle_is_not_callable():

            def it_leaves___command_handler___as_None():
                class Odd(Aggregate):
                    class Cmd(Command):
                        handle = "not a function"

                        class Outcome(DomainEvent):
                            pass

                assert Odd.Cmd.__command_handler__ is None

    def describe_EventGraph_registration():

        def when_command_class_passed_in_handlers_list():

            def it_dispatches_to_the_handle_method():
                graph = EventGraph([Shop.Buy])
                log = graph.invoke(Shop.Buy(item="apple"))
                assert log.has(Shop.Buy.Bought)
                assert log.latest(Shop.Buy.Bought).item == "apple"

            def it_binds_self_to_the_command_instance():
                graph = EventGraph([Shop.Buy])
                log = graph.invoke(Shop.Buy(item="dodo"))
                # handle() branches on self.item == "dodo"; if self bound,
                # OutOfStock fires; if bound wrong, Bought fires.
                assert log.has(Shop.Buy.OutOfStock)
                assert not log.has(Shop.Buy.Bought)

            def with_EventLog_param():

                def it_injects_the_current_log():
                    graph = EventGraph([WithLog.Cmd])
                    log = graph.invoke(WithLog.Cmd())
                    # One prior event in the log: the seed Cmd itself.
                    assert log.latest(WithLog.Cmd.Done).observed_count == 1

        def when_command_class_has_no_handle():

            def it_raises_TypeError_at_graph_construction():
                with pytest.raises(TypeError, match=r"no `handle` method"):
                    EventGraph([Shop2.NoHandler])

        def when_mixing_command_classes_and_at_on_functions():

            def it_registers_both_independently():
                @on(Shop.Buy.Bought)
                def react(event: Shop.Buy.Bought) -> None:
                    return None

                graph = EventGraph([Shop.Buy, react])
                log = graph.invoke(Shop.Buy(item="pear"))
                assert log.has(Shop.Buy.Bought)

    def describe_async_handle():

        def when_handle_is_async():

            def it_awaits_correctly_in_ainvoke():
                import asyncio

                async def run():
                    graph = EventGraph([Shop4.Slow])
                    return await graph.ainvoke(Shop4.Slow(item="pear"))

                log = asyncio.run(run())
                assert log.has(Shop4.Slow.Done)
                assert log.latest(Shop4.Slow.Done).item == "pear"

    def describe_return_contract():

        def when_handle_returns_declared_outcome():

            def it_accepts():
                graph = EventGraph([Shop.Buy])
                log = graph.invoke(Shop.Buy(item="pear"))
                assert log.has(Shop.Buy.Bought)

        def when_handle_returns_foreign_outcome():

            def it_raises_TypeError_via_Outcomes_contract():
                class RogueAgg(Aggregate):
                    class Cmd(Command):
                        class Good(DomainEvent):
                            pass

                        def handle(self):
                            # No annotation → falls back to Command.Outcomes;
                            # returning a foreign outcome violates it.
                            return Foreign.Do.Stuff()

                graph = EventGraph([RogueAgg.Cmd])
                with pytest.raises(TypeError, match=r"must return|outcomes of"):
                    graph.invoke(RogueAgg.Cmd())

    def describe_from_aggregates():

        def when_all_commands_define_handle():

            def it_registers_each_of_them():
                graph = EventGraph.from_aggregates(Shop3)
                log = graph.invoke(Shop3.CmdA())
                assert log.has(Shop3.CmdA.DoneA)
                log2 = graph.invoke(Shop3.CmdB())
                assert log2.has(Shop3.CmdB.DoneB)

        def when_some_commands_omit_handle():

            def it_skips_them_silently():
                # Shop3.CmdNoHandle has no handle — from_aggregates must not
                # raise. The resulting graph simply doesn't dispatch it.
                graph = EventGraph.from_aggregates(Shop3)
                log = graph.invoke(Shop3.CmdA())
                assert log.has(Shop3.CmdA.DoneA)

        def when_handlers_kwarg_provided():

            def it_appends_them_after_discovered_ones():
                observed: list[str] = []

                @on(Shop3.CmdA.DoneA)
                def react(event: Shop3.CmdA.DoneA) -> None:
                    observed.append("reacted")

                graph = EventGraph.from_aggregates(Shop3, handlers=[react])
                graph.invoke(Shop3.CmdA())
                assert observed == ["reacted"]

        def when_non_aggregate_argument_passed():

            def it_raises_TypeError():
                class NotAnAggregate:
                    pass

                with pytest.raises(TypeError, match=r"Aggregate"):
                    EventGraph.from_aggregates(NotAnAggregate)  # type: ignore[arg-type]

    def describe_handle_signature_validation():

        def when_handle_is_staticmethod():

            def it_rejects_at_class_creation():
                with pytest.raises(TypeError, match="staticmethod"):

                    class BadAgg(Aggregate):
                        class Cmd(Command):
                            class Done(DomainEvent):
                                pass

                            @staticmethod
                            def handle():  # type: ignore[misc]
                                return None

        def when_handle_first_param_is_not_self():

            def it_rejects_at_class_creation():
                with pytest.raises(TypeError, match="self"):

                    class BadAgg2(Aggregate):
                        class Cmd(Command):
                            class Done(DomainEvent):
                                pass

                            def handle(cmd):  # type: ignore[misc]  # noqa: N805
                                return None

    def describe_inline_outcome_coverage():

        def when_annotation_omits_an_outcome():

            def it_raises_at_graph_construction():
                with pytest.raises(
                    TypeError, match=r"does not cover outcome\(s\): OutOfStock"
                ):
                    EventGraph([Shop6.Buy])

        def when_annotation_covers_all_outcomes():

            def it_accepts():
                graph = EventGraph([Shop7.Buy])
                assert graph.invoke(Shop7.Buy()).has(Shop7.Buy.Bought)

        def when_inline_handle_has_no_annotation():

            def it_falls_back_to_outcomes_contract():
                graph = EventGraph([Shop8.Buy])
                assert graph.invoke(Shop8.Buy()).has(Shop8.Buy.Bought)

        def when_external_handler_declares_narrow_return():
            # External (@on) handlers are exempt — distributed outcome
            # production (one handler → Placed, another reacts to
            # InvariantViolated → Rejected) is a valid DDD pattern.

            def it_does_not_enforce_coverage():
                graph = EventGraph([shop9_partial])
                assert graph.invoke(Shop9.Buy()).has(Shop9.Buy.Bought)

        def when_annotation_includes_None_in_union():

            def it_accepts_if_all_outcomes_present():
                EventGraph([Shop10.Buy])
