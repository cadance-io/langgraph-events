"""Tests for inline command handlers (``Command.handle`` + auto-registration)."""

from __future__ import annotations

from typing import ClassVar

import pytest

from langgraph_events import (
    Command,
    DomainEvent,
    EventGraph,
    EventLog,
    Namespace,
    Reducer,
    on,
)


# Module-level domain for inline-handler dispatch tests.
class Shop(Namespace):
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


class Shop2(Namespace):
    class NoHandler(Command):
        class Outcome(DomainEvent):
            pass


class Shop3(Namespace):
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


class Shop4(Namespace):
    class Slow(Command):
        item: str = ""

        class Done(DomainEvent):
            item: str = ""

        async def handle(self) -> Shop4.Slow.Done:
            return Shop4.Slow.Done(item=self.item)


class Shop5(Namespace):
    class Tell(Command):
        class Told(DomainEvent):
            pass

        class OtherOutcome(DomainEvent):
            pass


class Foreign(Namespace):
    class Do(Command):
        class Stuff(DomainEvent):
            pass


class WithLog(Namespace):
    class Cmd(Command):
        class Done(DomainEvent):
            observed_count: int = 0

        def handle(self, log: EventLog) -> WithLog.Cmd.Done:
            return WithLog.Cmd.Done(observed_count=len(log))


# Module-level domains for inline-outcome-coverage tests. These can't live
# inside describe_/when_ blocks because Python can't resolve forward refs on
# handle's return annotation from a nested function scope.
class Shop6(Namespace):
    class Buy(Command):
        class Bought(DomainEvent):
            pass

        class OutOfStock(DomainEvent):
            pass

        def handle(self) -> Shop6.Buy.Bought:
            return Shop6.Buy.Bought()


class Shop7(Namespace):
    class Buy(Command):
        class Bought(DomainEvent):
            pass

        class OutOfStock(DomainEvent):
            pass

        def handle(self) -> Shop7.Buy.Bought | Shop7.Buy.OutOfStock:
            return Shop7.Buy.Bought()


class Shop8(Namespace):
    class Buy(Command):
        class Bought(DomainEvent):
            pass

        class OutOfStock(DomainEvent):
            pass

        def handle(self):
            return Shop8.Buy.Bought()


class Shop10(Namespace):
    class Buy(Command):
        class Bought(DomainEvent):
            pass

        class OutOfStock(DomainEvent):
            pass

        def handle(self) -> Shop10.Buy.Bought | Shop10.Buy.OutOfStock | None:
            return None


# Module-level fixtures for describe_handle_aliased_across_commands.
class LeftAgg(Namespace):
    class Do(Command):
        class Done(DomainEvent):
            pass

        def handle(self) -> LeftAgg.Do.Done:
            return LeftAgg.Do.Done()


class RightAgg(Namespace):
    class Do(Command):
        class Done(DomainEvent):
            pass


# Module-level fixtures for describe_service_injection.
class _StubChatModel:
    """Stand-in for a chat-model service used by inline-handle DI tests."""

    def __init__(self, value: str = "default") -> None:
        self.value = value


class _StubOpenAIChat(_StubChatModel):
    """Subclass used to verify base-class annotations match a subclass instance."""


class _StubAnthropicChat(_StubChatModel):
    """Sibling subclass used to exercise multi-match rejection."""


class _StubSessionFactory:
    """Second distinct service type used to verify multi-service injection."""

    def __init__(self, label: str = "default") -> None:
        self.label = label


class WithService(Namespace):
    class Cmd(Command):
        class Done(DomainEvent):
            value: str = ""

        def handle(self, chat_model: _StubChatModel) -> WithService.Cmd.Done:
            return WithService.Cmd.Done(value=chat_model.value)


class WithAsyncService(Namespace):
    class Cmd(Command):
        class Done(DomainEvent):
            value: str = ""

        async def handle(self, chat_model: _StubChatModel) -> WithAsyncService.Cmd.Done:
            return WithAsyncService.Cmd.Done(value=chat_model.value)


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
                class Odd(Namespace):
                    class Cmd(Command):
                        handle = "not a function"

                        class Outcome(DomainEvent):
                            pass

                assert Odd.Cmd.__command_handler__ is None

    def describe_class_level_modifiers():
        def when_invariants_set_as_class_attribute():
            def it_evaluates_the_predicate_at_dispatch():
                from langgraph_events import Invariant, InvariantViolated

                class _BlockedInv(Invariant):
                    pass

                class _InlineInv(Namespace):
                    class Cmd(Command):
                        invariants: ClassVar = {_BlockedInv: lambda log: False}

                        class Done(DomainEvent):
                            pass

                        def handle(self) -> _InlineInv.Cmd.Done:
                            return _InlineInv.Cmd.Done()

                graph = EventGraph([_InlineInv.Cmd])
                log = graph.invoke(_InlineInv.Cmd())
                assert log.has(InvariantViolated)
                assert not log.has(_InlineInv.Cmd.Done)

        def when_invariants_inherited_from_a_parent_command():
            def it_evaluates_the_inherited_predicate():
                from langgraph_events import Invariant, InvariantViolated

                class _BlockedInheritedInv(Invariant):
                    pass

                class _InlineInherit(Namespace):
                    class Parent(Command):
                        invariants: ClassVar = {
                            _BlockedInheritedInv: lambda log: False,
                        }

                        class Done(DomainEvent):
                            pass

                        def handle(self) -> _InlineInherit.Parent.Done:
                            return _InlineInherit.Parent.Done()

                    class Child(Parent):
                        def handle(self) -> _InlineInherit.Parent.Done:
                            return _InlineInherit.Parent.Done()

                graph = EventGraph([_InlineInherit.Child])
                log = graph.invoke(_InlineInherit.Child())
                assert log.has(InvariantViolated)
                assert not log.has(_InlineInherit.Parent.Done)

        def when_raises_set_as_class_attribute():
            def it_routes_the_exception_to_HandlerRaised():
                from langgraph_events import HandlerRaised

                class _BoomError(Exception):
                    pass

                class _InlineRaises(Namespace):
                    class Cmd(Command):
                        raises: ClassVar = (_BoomError,)

                        class Done(DomainEvent):
                            pass

                        def handle(self) -> _InlineRaises.Cmd.Done:
                            raise _BoomError("nope")

                @on(HandlerRaised, exception=_BoomError)
                def catch(event: HandlerRaised) -> None:
                    return None

                graph = EventGraph([_InlineRaises.Cmd, catch])
                log = graph.invoke(_InlineRaises.Cmd())
                assert log.has(HandlerRaised)

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

    def describe_service_injection():

        def when_handle_declares_a_service_parameter():

            def it_injects_the_registered_service_by_type():
                chat_model = _StubChatModel(value="injected!")
                graph = EventGraph([WithService.Cmd], services=[chat_model])
                log = graph.invoke(WithService.Cmd())
                assert log.latest(WithService.Cmd.Done).value == "injected!"

        def when_handle_declares_a_service_parameter_but_no_service_registered():

            def it_raises_at_graph_construction():
                with pytest.raises(TypeError, match=r"chat_model"):
                    EventGraph([WithService.Cmd])

        def when_two_services_share_the_same_exact_type():

            def it_rejects_at_graph_construction():
                a = _StubChatModel(value="a")
                b = _StubChatModel(value="b")
                with pytest.raises(TypeError, match=r"_StubChatModel.*collision"):
                    EventGraph([WithService.Cmd], services=[a, b])

        def when_service_is_subclass_of_param_annotation():

            def it_satisfies_the_base_class_annotation():
                subclass_instance = _StubOpenAIChat(value="from-subclass")
                graph = EventGraph([WithService.Cmd], services=[subclass_instance])
                log = graph.invoke(WithService.Cmd())
                assert log.latest(WithService.Cmd.Done).value == "from-subclass"

        def when_two_services_both_match_the_param_annotation():

            def it_rejects_at_graph_construction():
                openai = _StubOpenAIChat(value="openai")
                anthropic = _StubAnthropicChat(value="anthropic")
                with pytest.raises(
                    TypeError, match=r"chat_model.*multiple.*registered services"
                ):
                    EventGraph([WithService.Cmd], services=[openai, anthropic])

        def when_param_is_annotated_object():

            def it_does_not_silently_consume_a_service():
                # `param: object` matches every registered type via issubclass,
                # which would silently inject an unrelated service. The framework
                # must treat the annotation as too broad to claim a service —
                # falling through to "unclaimed param" and erroring at build.
                @on(Shop.Buy.Bought)
                def overly_broad(
                    event: Shop.Buy.Bought,
                    foo: object,
                ) -> None:
                    pass

                chat_model = _StubChatModel(value="x")
                with pytest.raises(TypeError, match=r"foo"):
                    EventGraph([overly_broad], services=[chat_model])

        def when_param_is_resolved_as_a_service_alongside_a_reducer():

            def it_does_not_trigger_the_unknown_reducer_warning():
                # The "unknown reducer" warning fires for any handler
                # parameter that isn't claimed by a known injection source.
                # A param resolved via service-type matching MUST count as
                # claimed — otherwise users get a noisy false-positive
                # warning every time they declare a service param while
                # reducers are also registered.
                import warnings as _warnings

                @on(Shop.Buy.Bought)
                def with_both(
                    event: Shop.Buy.Bought,
                    chat_model: _StubChatModel,
                    items: list,
                ) -> None:
                    pass

                from langgraph_events import Reducer as _Reducer

                items_reducer = _Reducer(
                    name="items",
                    event_type=Shop.Buy.Bought,
                    fn=lambda e: [e.item],
                )
                chat_model_svc = _StubChatModel(value="x")

                with _warnings.catch_warnings(record=True) as caught:
                    _warnings.simplefilter("always")
                    EventGraph(
                        [with_both],
                        reducers=[items_reducer],
                        services=[chat_model_svc],
                    )

                # No "don't match any reducer" warning should fire — the
                # service param is claimed; the reducer param is matched.
                offending = [
                    w for w in caught if "don't match any reducer" in str(w.message)
                ]
                assert not offending, (
                    f"unexpected typo warning: {[str(w.message) for w in offending]}"
                )

        def when_services_are_passed_as_a_name_keyed_mapping():

            def it_resolves_each_handler_param_by_its_name():
                # `services={"primary_chat": ..., "backup_chat": ...}` allows
                # two instances of the same type — the type-keyed list form
                # would reject that as a collision. Resolution is by handler
                # parameter name matching the registry key.
                observed: dict[str, object] = {}

                @on(Shop.Buy.Bought)
                def two_chats(
                    event: Shop.Buy.Bought,
                    primary_chat: _StubChatModel,
                    backup_chat: _StubChatModel,
                ) -> None:
                    observed["primary_chat"] = primary_chat
                    observed["backup_chat"] = backup_chat

                primary = _StubChatModel(value="primary")
                backup = _StubChatModel(value="backup")
                graph = EventGraph(
                    [two_chats],
                    services={"primary_chat": primary, "backup_chat": backup},
                )
                graph.invoke(Shop.Buy.Bought(item="apple", price=1.0))
                assert observed["primary_chat"] is primary
                assert observed["backup_chat"] is backup

        def when_handler_param_name_has_no_matching_service_key():

            def it_raises_at_graph_construction():
                # Annotation alone is not enough in name-keyed mode — the
                # framework cannot guess which service to bind. Surface the
                # missing-binding at graph build.
                @on(Shop.Buy.Bought)
                def picky(
                    event: Shop.Buy.Bought,
                    chat_model: _StubChatModel,
                ) -> None:
                    pass

                with pytest.raises(TypeError, match=r"chat_model"):
                    EventGraph(
                        [picky],
                        services={"primary_chat": _StubChatModel(value="x")},
                    )

        def when_handler_uses_args_and_kwargs():

            def it_does_not_flag_them_as_unclaimed():
                # Variadic parameters cannot be filled by name- or type-based
                # injection — they are caller-controlled and should be ignored
                # by the unclaimed-param check. A generic catcher is a valid
                # use case and must not raise at graph build.
                @on(Shop.Buy.Bought)
                def variadic(event: Shop.Buy.Bought, *args, **kwargs) -> None:
                    pass

                # Build should succeed; no unclaimed-param error.
                EventGraph([variadic])

        def when_base_and_subclass_services_are_both_registered():

            def it_resolves_each_param_to_its_exact_type():
                # services=[A(), B()] where B(A). Handler annotates one param
                # as A and another as B. The user has clearly disambiguated by
                # annotation; multi-match should NOT fire here.
                observed: dict[str, object] = {}

                @on(Shop.Buy.Bought)
                def two_typed(
                    event: Shop.Buy.Bought,
                    base: _StubChatModel,
                    sub: _StubOpenAIChat,
                ) -> None:
                    observed["base"] = base
                    observed["sub"] = sub

                base_svc = _StubChatModel(value="base")
                sub_svc = _StubOpenAIChat(value="sub")
                graph = EventGraph([two_typed], services=[base_svc, sub_svc])
                graph.invoke(Shop.Buy.Bought(item="apple", price=1.0))
                assert observed["base"] is base_svc
                assert observed["sub"] is sub_svc

        def when_param_name_matches_a_reducer_and_type_matches_a_service():

            def it_resolves_to_the_reducer_not_the_service():
                observed: dict[str, object] = {}

                @on(Shop.Buy.Bought)
                def collide(
                    event: Shop.Buy.Bought,
                    chat_model: _StubChatModel,
                ) -> None:
                    observed["chat_model"] = chat_model

                # Reducer named "chat_model" — collides with the param name.
                # Per the resolution order (reducer → framework → service),
                # the reducer state wins, so the handler receives a list.
                chat_log = Reducer(
                    name="chat_model",
                    event_type=Shop.Buy.Bought,
                    fn=lambda e: [e.item],
                )
                chat_model_svc = _StubChatModel(value="from-service")
                graph = EventGraph(
                    [collide],
                    reducers=[chat_log],
                    services=[chat_model_svc],
                )
                graph.invoke(Shop.Buy.Bought(item="apple", price=1.0))
                assert observed["chat_model"] == ["apple"]

        def when_external_handler_declares_multiple_service_params():

            def it_injects_each_service_by_its_type():
                observed: dict[str, object] = {}

                @on(Shop.Buy.Bought)
                def two_services(
                    event: Shop.Buy.Bought,
                    chat_model: _StubChatModel,
                    session_factory: _StubSessionFactory,
                ) -> None:
                    observed["chat_model"] = chat_model
                    observed["session_factory"] = session_factory

                chat_model_svc = _StubChatModel(value="chat")
                session_factory_svc = _StubSessionFactory(label="session")
                graph = EventGraph(
                    [two_services],
                    services=[chat_model_svc, session_factory_svc],
                )
                graph.invoke(Shop.Buy.Bought(item="apple", price=1.0))
                assert observed["chat_model"] is chat_model_svc
                assert observed["session_factory"] is session_factory_svc

        def when_inline_handle_is_async_and_declares_a_service_param():

            def it_injects_through_ainvoke():
                import asyncio

                async def run() -> EventLog:
                    chat_model = _StubChatModel(value="async-injected")
                    graph = EventGraph([WithAsyncService.Cmd], services=[chat_model])
                    return await graph.ainvoke(WithAsyncService.Cmd())

                log = asyncio.run(run())
                assert log.latest(WithAsyncService.Cmd.Done).value == "async-injected"

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
                class RogueAgg(Namespace):
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

    def describe_from_namespaces():

        def when_all_commands_define_handle():

            def it_registers_each_of_them():
                graph = EventGraph.from_namespaces(Shop3)
                log = graph.invoke(Shop3.CmdA())
                assert log.has(Shop3.CmdA.DoneA)
                log2 = graph.invoke(Shop3.CmdB())
                assert log2.has(Shop3.CmdB.DoneB)

        def when_some_commands_omit_handle():

            def it_skips_them_silently():
                # Shop3.CmdNoHandle has no handle — from_namespaces must not
                # raise. The resulting graph simply doesn't dispatch it.
                graph = EventGraph.from_namespaces(Shop3)
                log = graph.invoke(Shop3.CmdA())
                assert log.has(Shop3.CmdA.DoneA)

        def when_handlers_kwarg_provided():

            def it_appends_them_after_discovered_ones():
                observed: list[str] = []

                @on(Shop3.CmdA.DoneA)
                def react(event: Shop3.CmdA.DoneA) -> None:
                    observed.append("reacted")

                graph = EventGraph.from_namespaces(Shop3, handlers=[react])
                graph.invoke(Shop3.CmdA())
                assert observed == ["reacted"]

        def when_non_domain_argument_passed():

            def it_raises_TypeError():
                class NotANamespace:
                    pass

                with pytest.raises(TypeError, match=r"Namespace"):
                    EventGraph.from_namespaces(NotANamespace)  # type: ignore[arg-type]

    def describe_handle_signature_validation():

        def when_handle_is_staticmethod():

            def it_rejects_at_class_creation():
                with pytest.raises(TypeError, match="staticmethod"):

                    class BadAgg(Namespace):
                        class Cmd(Command):
                            class Done(DomainEvent):
                                pass

                            @staticmethod
                            def handle():  # type: ignore[misc]
                                return None

        def when_handle_first_param_is_not_self():

            def it_rejects_at_class_creation():
                with pytest.raises(TypeError, match="self"):

                    class BadAgg2(Namespace):
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

        def when_annotation_includes_None_in_union():

            def it_accepts_if_all_outcomes_present():
                EventGraph([Shop10.Buy])


def describe_handle_aliased_across_commands():

    def when_second_command_reuses_first_handle():

        def it_raises():
            RightAgg.Do.__command_handler__ = LeftAgg.Do.__command_handler__

            EventGraph([LeftAgg.Do])
            with pytest.raises(TypeError, match=r"already bound"):
                EventGraph([RightAgg.Do])
