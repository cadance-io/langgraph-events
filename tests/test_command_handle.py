"""Tests for inline command handlers (``Command.handle`` + auto-registration)."""

from __future__ import annotations

import re
import textwrap
from typing import ClassVar
from typing import ClassVar as CV  # noqa: N817 — alias spelling is under test

import pytest

from langgraph_events import (
    Command,
    DomainEvent,
    EventGraph,
    EventLog,
    HandlerRaised,
    IntegrationEvent,
    Interrupted,
    Invariant,
    InvariantViolated,
    Namespace,
    Reducer,
    Scatter,
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


# Module-level domain whose inline handler method is named something other
# than ``handle`` — exercises that node identity comes from the command
# qualname, not the method ``__name__``.
class Bazaar(Namespace):
    class Sell(Command):
        item: str = ""

        class Sold(DomainEvent):
            item: str = ""

        def sell(self) -> Bazaar.Sell.Sold:
            return Bazaar.Sell.Sold(item=self.item)


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


class Shop11(Namespace):
    class Other(DomainEvent):
        pass

    class Buy(Command):
        class Done(DomainEvent):
            pass

        # Annotates a sibling event so the coverage check runs and Done is missing.
        def handle(self) -> Shop11.Other:
            return Shop11.Buy.Done()


class Shop12(Namespace):
    class Buy(Command):
        class Done(DomainEvent):
            pass

        def handle(self) -> Scatter:
            return Scatter([])


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


# Module-level domain for the ``previously`` class-attribute tests (#103): a
# reactor node named ``save`` was replaced by an inline Command; the old node
# name stays resumable via the class attribute.
class _VaultPause(Interrupted):
    pass


class _VaultGo(IntegrationEvent):
    pass


class Vault(Namespace):
    class Persist(Command):
        previously: ClassVar = ("save",)

        class Saved(DomainEvent):
            pass

        def handle(self) -> Vault.Persist.Saved:
            return Vault.Persist.Saved()


# A pausing command for the resume-recovery test: the reactor it replaced
# returned the same Interrupted, so the resumed checkpoint re-enters the
# alias node, re-pauses, and consumes the resume value.
class Vault2(Namespace):
    class Approve(Command):
        previously: ClassVar = ("await_approval",)

        def handle(self) -> _VaultPause:
            return _VaultPause()


def describe_Command_handle():

    def describe_class_creation():

        def when_command_has_handle_method():

            def it_stamps___command_handler__():
                assert Shop.Buy.__command_handler__ is Shop.Buy.__dict__["handle"]

        def when_command_has_no_handle_method():

            def it_leaves___command_handler___as_None():
                assert Shop2.NoHandler.__command_handler__ is None

        def when_handle_is_not_callable():

            def it_raises_at_class_creation():
                """With pydantic, unannotated non-callable attribute is a hard error."""
                from pydantic import PydanticUserError

                with pytest.raises((TypeError, PydanticUserError)):
                    class Odd(Namespace):
                        class Cmd(Command):
                            handle = "not a function"

                            class Outcome(DomainEvent):
                                pass

        def when_command_has_a_meaningfully_named_public_method():
            # The handler can be named anything meaningful — not just
            # ``handle``. The framework picks up the sole public method.

            def it_stamps___command_handler__():
                class Boutique(Namespace):
                    class Buy(Command):
                        item: str = ""

                        class Bought(DomainEvent):
                            item: str = ""

                        def buy(self) -> Boutique.Buy.Bought:
                            return Boutique.Buy.Bought(item=self.item)

                assert Boutique.Buy.__command_handler__ is Boutique.Buy.__dict__["buy"]
                graph = EventGraph([Boutique.Buy])
                log = graph.invoke(Boutique.Buy(item="apple"))
                assert log.latest(Boutique.Buy.Bought).item == "apple"

        def when_command_has_two_public_methods():
            # A Command represents a single intent; two public methods is
            # ambiguous. Helpers must be underscore-prefixed.

            def it_rejects_at_class_creation():
                with pytest.raises(TypeError, match=r"more than one public method"):

                    class _Bad(Namespace):
                        class Cmd(Command):
                            class Done(DomainEvent):
                                pass

                            def place(self) -> _Bad.Cmd.Done:
                                return _Bad.Cmd.Done()

                            def helper(self) -> str:
                                return "x"

        def when_command_has_a_private_helper_alongside_handler():
            # Underscore-prefixed methods are exempt from the public-methods
            # cap; the framework still picks up the sole public method as
            # the handler.

            def it_picks_up_only_the_public_method():
                class _Helped(Namespace):
                    class Cmd(Command):
                        class Done(DomainEvent):
                            note: str = ""

                        def place(self) -> _Helped.Cmd.Done:
                            return _Helped.Cmd.Done(note=self._note())

                        def _note(self) -> str:
                            return "ok"

                graph = EventGraph([_Helped.Cmd])
                log = graph.invoke(_Helped.Cmd())
                assert log.latest(_Helped.Cmd.Done).note == "ok"

    def describe_class_level_modifiers():
        def when_invariants_set_as_class_attribute():
            def it_evaluates_the_predicate_at_dispatch():
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

            def when_the_declaration_is_removed_between_builds():
                def it_stops_evaluating_the_predicate():
                    # Each build must reflect the class's current declaration
                    # — a stale ``_invariants`` stamp from an earlier build
                    # must not keep evaluating a removed predicate.
                    class _StaleInv(Invariant):
                        pass

                    class _StaleInvariants(Namespace):
                        class Cmd(Command):
                            invariants: ClassVar = {_StaleInv: lambda log: False}

                            class Done(DomainEvent):
                                pass

                            def handle(self) -> _StaleInvariants.Cmd.Done:
                                return _StaleInvariants.Cmd.Done()

                    first = EventGraph([_StaleInvariants.Cmd])
                    assert first.invoke(_StaleInvariants.Cmd()).has(InvariantViolated)
                    del _StaleInvariants.Cmd.invariants
                    log = EventGraph([_StaleInvariants.Cmd]).invoke(
                        _StaleInvariants.Cmd()
                    )
                    assert not log.has(InvariantViolated)
                    assert log.has(_StaleInvariants.Cmd.Done)

        def when_raises_set_as_class_attribute():
            def it_routes_the_exception_to_HandlerRaised():
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

            def when_the_declaration_is_removed_between_builds():
                def it_stops_routing_the_exception_to_HandlerRaised():
                    # Each build must reflect the class's current declaration
                    # — a stale ``_raises`` stamp from an earlier build must
                    # not keep catching an exception nobody declares.
                    class _StaleBoomError(Exception):
                        pass

                    class _StaleRaises(Namespace):
                        class Cmd(Command):
                            raises: ClassVar = (_StaleBoomError,)

                            class Done(DomainEvent):
                                pass

                            def handle(self) -> _StaleRaises.Cmd.Done:
                                raise _StaleBoomError("nope")

                    @on(HandlerRaised, exception=_StaleBoomError)
                    def catch(event: HandlerRaised) -> None:
                        return None

                    first = EventGraph([_StaleRaises.Cmd, catch])
                    assert first.invoke(_StaleRaises.Cmd()).has(HandlerRaised)
                    del _StaleRaises.Cmd.raises
                    second = EventGraph([_StaleRaises.Cmd, catch])
                    with pytest.raises(_StaleBoomError, match="nope"):
                        second.invoke(_StaleRaises.Cmd())

        def when_raises_is_declared_as_a_dataclass_field():
            def it_rejects_at_class_creation():
                # An annotated (non-ClassVar) ``raises`` would silently
                # become a frozen dataclass field serializing exception
                # classes into every checkpoint payload while exception
                # routing kept working (same hazard as ``previously``).
                with pytest.raises(TypeError, match=r"'raises'.*ClassVar"):

                    class _BadRaises(Namespace):
                        class Cmd(Command):
                            raises: tuple[type, ...] = (ValueError,)

                            def handle(self) -> None:
                                return None

        def when_invariants_is_declared_as_a_dataclass_field():
            def it_rejects_at_class_creation():
                # Without the guard this dies inside dataclasses with a
                # mutable-default ValueError whose advice (use
                # default_factory) is actively harmful here: a factory
                # field has no class attribute, so invariant enforcement
                # would silently vanish. That stdlib error must also not
                # surface as a chained cause — it would display the very
                # advice the guard exists to override.
                with pytest.raises(
                    TypeError, match=r"'invariants'.*ClassVar"
                ) as excinfo:

                    class _BadInv(Namespace):
                        class Cmd(Command):
                            invariants: dict = {}  # noqa: RUF012

                            def handle(self) -> None:
                                return None

                assert excinfo.value.__cause__ is None

            def when_the_module_evaluates_annotations_eagerly():
                def it_rejects_before_dataclass_processing():
                    # Without ``from __future__ import annotations`` the
                    # annotation is a real type object, so the guard judges
                    # it directly — the error must not arrive as a chained
                    # translation of dataclasses' mutable-default ValueError.
                    src = textwrap.dedent(
                        """
                        class _EagerAnn(Namespace):
                            class Cmd(Command):
                                invariants: dict = {}

                                def handle(self) -> None:
                                    return None
                        """
                    )
                    # dont_inherit: without it the exec'd code inherits this
                    # module's deferred-annotations future flag.
                    code = compile(src, "<eager>", "exec", dont_inherit=True)
                    ns = {"Namespace": Namespace, "Command": Command}
                    with pytest.raises(
                        TypeError, match=r"'invariants'.*ClassVar"
                    ) as excinfo:
                        exec(code, ns)  # noqa: S102
                    assert excinfo.value.__cause__ is None

    def describe_previously_class_attribute():
        # ``previously`` mirrors ``raises``/``invariants`` as a class-level
        # modifier (inline handlers have no decorator slot): it declares the
        # command node's historic names so checkpoints paused under an old
        # name resume into the renamed command. See issue #103.

        def when_previously_set_as_class_attribute():
            def it_registers_an_alias_node():
                nodes = set(EventGraph([Vault.Persist])._compile().get_graph().nodes)
                assert "Vault.Persist" in nodes
                assert "save" in nodes

            def it_does_not_route_new_events_into_the_alias():
                # Same load-bearing invariant as the reactor form: the
                # dispatcher only returns canonical names, so a fresh invoke
                # runs the handler exactly once, not once per alias.
                log = EventGraph([Vault.Persist]).invoke(Vault.Persist())
                assert len([e for e in log if isinstance(e, Vault.Persist.Saved)]) == 1

            def it_registers_aliases_idempotently_across_builds():
                # _expand_command_handlers re-stamps the same handler function
                # on every EventGraph construction; the constant class-attr
                # value makes that idempotent.
                first = set(EventGraph([Vault.Persist])._compile().get_graph().nodes)
                second = set(EventGraph([Vault.Persist])._compile().get_graph().nodes)
                assert "save" in first
                assert first == second

        def when_the_declaration_is_removed_between_builds():
            def it_drops_the_alias_on_the_next_build():
                # Each build must reflect the class's current declaration — a
                # stale stamp from an earlier build must not keep a deleted
                # alias alive.
                class _Era(Namespace):
                    class Cmd(Command):
                        previously: ClassVar = ("first_era",)

                        def handle(self) -> None:
                            return None

                first = set(EventGraph([_Era.Cmd])._compile().get_graph().nodes)
                assert "first_era" in first
                del _Era.Cmd.previously
                second = set(EventGraph([_Era.Cmd])._compile().get_graph().nodes)
                assert "first_era" not in second

        def when_a_paused_reactor_becomes_an_inline_command():
            def it_resumes_the_old_checkpoint_via_the_alias():
                # The acceptance scenario from #103: a reactor node paused a
                # thread, the reactor was replaced by an inline Command, and
                # the checkpoint's snapshot.next still holds the old node
                # name. previously= keeps it resumable.
                from langgraph.checkpoint.memory import MemorySaver

                from langgraph_events.serde import (
                    NamespaceAwareSerde,
                    assert_resume_recovers,
                )

                @on(Vault2.Approve, node_name="await_approval")
                def approve_reactor(event: Vault2.Approve) -> _VaultPause:
                    return _VaultPause()

                @on(_VaultGo)
                def go(event: _VaultGo) -> None:
                    return None

                # Command events ride inside the checkpoint payload, so the
                # paused thread only revives through a namespace-aware serde
                # (LangGraph's default serializer can't reconstruct nested
                # classes).
                saver = MemorySaver(serde=NamespaceAwareSerde())
                before = EventGraph([approve_reactor, go], checkpointer=saver)
                after = EventGraph([Vault2.Approve, go], checkpointer=saver)
                assert_resume_recovers(
                    before, after, seed=Vault2.Approve(), resume_with=_VaultGo()
                )

        def when_previously_is_a_bare_string():
            def it_normalizes_to_a_single_alias():
                class _Solo(Namespace):
                    class Cmd(Command):
                        previously: ClassVar = "old_solo"

                        def handle(self) -> None:
                            return None

                nodes = set(EventGraph([_Solo.Cmd])._compile().get_graph().nodes)
                assert "old_solo" in nodes

        def when_an_alias_collides():
            def with_a_live_handler_name():
                def it_raises_at_build():
                    class _Clash(Namespace):
                        class Cmd(Command):
                            previously: ClassVar = ("live_node",)

                            def handle(self) -> None:
                                return None

                    @on(_VaultGo, node_name="live_node")
                    def live(event: _VaultGo) -> None:
                        return None

                    with pytest.raises(ValueError, match="collides"):
                        EventGraph([_Clash.Cmd, live])

                def it_names_the_command_node_in_the_error():
                    # The error must identify the claimant by its checkpoint
                    # identity (the command qualname), not the method name
                    # ('handle'), which several commands share.
                    class _Clash2(Namespace):
                        class Cmd(Command):
                            previously: ClassVar = ("live_node2",)

                            def handle(self) -> None:
                                return None

                    @on(_VaultGo, node_name="live_node2")
                    def live(event: _VaultGo) -> None:
                        return None

                    with pytest.raises(ValueError, match=r"_Clash2\.Cmd"):
                        EventGraph([_Clash2.Cmd, live])

            def with_another_handler_claiming_the_same_alias():
                def it_names_both_claimants():
                    class _DupA(Namespace):
                        class Cmd(Command):
                            previously: ClassVar = ("shared_old",)

                            def handle(self) -> None:
                                return None

                    class _DupB(Namespace):
                        class Cmd(Command):
                            previously: ClassVar = ("shared_old",)

                            def handle(self) -> None:
                                return None

                    with pytest.raises(ValueError) as excinfo:
                        EventGraph([_DupA.Cmd, _DupB.Cmd])
                    assert "_DupA.Cmd" in str(excinfo.value)
                    assert "_DupB.Cmd" in str(excinfo.value)

            def with_a_reserved_framework_node():
                # ``__seed__``/``__router__`` are the framework's own pregel
                # nodes — they can never have been a historic handler name.
                # Without validation the alias dies in LangGraph add_node/
                # compile with an opaque duplicate-node error.
                def it_rejects_the_command_alias_at_build():
                    class _ReservedAlias(Namespace):
                        class Cmd(Command):
                            previously: ClassVar = ("__seed__",)

                            def handle(self) -> None:
                                return None

                    with pytest.raises(ValueError, match="reserved") as excinfo:
                        EventGraph([_ReservedAlias.Cmd])
                    assert "_ReservedAlias.Cmd" in str(excinfo.value)
                    assert "__seed__" in str(excinfo.value)

                def it_rejects_the_decorator_alias_at_build():
                    @on(_VaultGo, previously="__router__")
                    def react(event: _VaultGo) -> None:
                        return None

                    with pytest.raises(ValueError, match="reserved") as excinfo:
                        EventGraph([react])
                    assert "__router__" in str(excinfo.value)

                def it_rejects_langgraph_reserved_endpoints_too():
                    # LangGraph rejects __start__/__end__ in add_node itself,
                    # but only at first compile/invoke and without naming
                    # the declaration that smuggled the name in.
                    class _ReservedStart(Namespace):
                        class Cmd(Command):
                            previously: ClassVar = ("__start__",)

                            def handle(self) -> None:
                                return None

                    with pytest.raises(ValueError, match="reserved") as excinfo:
                        EventGraph([_ReservedStart.Cmd])
                    assert "_ReservedStart.Cmd" in str(excinfo.value)

        def when_previously_is_an_invalid_value():
            def it_names_the_command_in_the_error():
                # The user wrote a class attribute, not @on() — the error
                # must point at the command, not the decorator they never
                # called.
                class _BadVal(Namespace):
                    class Cmd(Command):
                        previously: ClassVar = 123

                        def handle(self) -> None:
                            return None

                with pytest.raises(TypeError, match=r"_BadVal\.Cmd"):
                    EventGraph([_BadVal.Cmd])

        def when_classvar_is_imported_under_a_module_level_alias():
            def it_accepts_the_aliased_annotation():
                # dataclasses resolves PEP 563 string annotations through
                # the module globals, so ``CV`` is a working ClassVar
                # spelling — the reserved-modifier guard must not
                # second-guess dataclasses and reject it.
                class _Aliased(Namespace):
                    class Cmd(Command):
                        previously: CV = ("legacy_aliased",)

                        def handle(self) -> None:
                            return None

                (meta,) = EventGraph([_Aliased.Cmd])._handler_metas
                assert meta.previous_names == ("legacy_aliased",)

        def when_previously_is_declared_as_a_dataclass_field():
            def it_rejects_at_class_creation():
                # An annotated (non-ClassVar) ``previously`` would silently
                # become a frozen dataclass field serialized into every
                # checkpoint payload while aliasing still appears to work.
                with pytest.raises(TypeError, match="ClassVar"):

                    class _Bad(Namespace):
                        class Cmd(Command):
                            previously: tuple[str, ...] = ("x",)

                            def handle(self) -> None:
                                return None

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
                with pytest.raises(TypeError, match=r"no inline handler"):
                    EventGraph([Shop2.NoHandler])

        def when_mixing_command_classes_and_at_on_functions():

            def it_registers_both_independently():
                @on(Shop.Buy.Bought)
                def react(event: Shop.Buy.Bought) -> None:
                    return None

                graph = EventGraph([Shop.Buy, react])
                log = graph.invoke(Shop.Buy(item="pear"))
                assert log.has(Shop.Buy.Bought)

    def describe_inline_handler_node_identity():
        # An inline command handler's graph-node name must be a stable,
        # order-independent identity derived from the command's __qualname__
        # — not the method __name__ (``handle``) deduplicated positionally
        # (``handle``, ``handle_2``) by registration order. See issue #97.

        def _name_to_command(graph: EventGraph) -> dict[str, type]:
            return {
                meta.node_name: meta.fn._inline_command for meta in graph._handler_metas
            }

        def when_command_registered():

            def it_names_node_after_the_command_qualname():
                graph = EventGraph([Shop3.CmdA])
                assert graph.handler_names == frozenset({"Shop3.CmdA"})

        def when_two_command_handlers_share_a_method_name():
            # CmdA and CmdB both define ``handle``; under the old scheme they
            # collapsed to ``handle``/``handle_2`` by list position.

            def it_assigns_qualname_node_names_regardless_of_order():
                ab = EventGraph([Shop3.CmdA, Shop3.CmdB]).handler_names
                ba = EventGraph([Shop3.CmdB, Shop3.CmdA]).handler_names
                assert ab == ba == frozenset({"Shop3.CmdA", "Shop3.CmdB"})

            def it_keeps_the_name_to_command_mapping_stable_under_reorder():
                ab = _name_to_command(EventGraph([Shop3.CmdA, Shop3.CmdB]))
                ba = _name_to_command(EventGraph([Shop3.CmdB, Shop3.CmdA]))
                assert ab == ba
                assert ab == {
                    "Shop3.CmdA": Shop3.CmdA,
                    "Shop3.CmdB": Shop3.CmdB,
                }

            def it_never_produces_a_positional_handle_node():
                names = EventGraph([Shop3.CmdA, Shop3.CmdB]).handler_names
                assert not any("handle" in name for name in names)
                assert not any(name.endswith("_2") for name in names)

        def when_a_handler_resolves_to_a_reserved_framework_node():
            def it_rejects_at_build_naming_the_function():
                # Reachable via an explicit @on(node_name=...) pin or a
                # function literally named after the reserved node — in
                # both cases the node name IS the reserved string, so the
                # error must name the claimant by the function the user
                # wrote, not echo the node name as the handler identity.
                @on(_VaultGo, node_name="__seed__")
                def react(event: _VaultGo) -> None:
                    return None

                with pytest.raises(ValueError, match="reserved") as excinfo:
                    EventGraph([react])
                assert "__seed__" in str(excinfo.value)
                assert "react" in str(excinfo.value)

            def it_dispatches_to_the_correct_command_under_each_order():
                for handlers in ([Shop3.CmdA, Shop3.CmdB], [Shop3.CmdB, Shop3.CmdA]):
                    graph = EventGraph(handlers)
                    assert graph.invoke(Shop3.CmdA()).has(Shop3.CmdA.DoneA)
                    assert graph.invoke(Shop3.CmdB()).has(Shop3.CmdB.DoneB)

        def when_handler_method_has_a_custom_name():

            def it_still_uses_the_command_qualname():
                graph = EventGraph([Bazaar.Sell])
                assert graph.handler_names == frozenset({"Bazaar.Sell"})

        def when_built_via_from_namespaces():

            def it_also_uses_qualname_node_names():
                graph = EventGraph.from_namespaces(Shop3)
                assert graph.handler_names == frozenset({"Shop3.CmdA", "Shop3.CmdB"})

        def when_two_handlers_resolve_to_the_same_node_name():
            # node_name uniqueness is the invariant checkpoints depend on. The
            # old positional dedup guaranteed it structurally; the qualname
            # scheme must guard it explicitly — otherwise the collision only
            # surfaces as an opaque LangGraph "node already present" error at
            # compile, instead of a clear framework error at construction.

            def it_raises_a_clear_error_at_construction():
                with pytest.raises(ValueError, match=r"Shop3\.CmdA") as excinfo:
                    EventGraph([Shop3.CmdA, Shop3.CmdA])
                assert "node" in str(excinfo.value).lower()

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

        def when_the_declared_type_and_missing_outcome_share_a_name():

            # Two distinct classes of the same name render identically, so
            # the message read "declares return type `Placed` but does not
            # cover outcome(s): Placed" — advice already satisfied. Usually a
            # function-local class whose string annotation resolved to a
            # different object (issue #151).
            def it_distinguishes_them_and_names_the_cause():
                class Twin(Namespace):
                    class Do(Command):
                        class Placed(DomainEvent):
                            pass

                        def handle(self) -> _Decoy.Do.Placed:
                            return _Decoy.Do.Placed()

                with pytest.raises(TypeError) as exc:
                    EventGraph([Twin.Do])

                msg = str(exc.value)
                assert "`Placed` but does not cover outcome(s): Placed" not in msg
                assert "Twin.Do.Placed" in msg

            def it_keeps_the_annotation_advice_for_genuinely_missing_outcomes():
                # A collision on one name says nothing about the others: an
                # outcome that is simply uncovered still needs the edit.
                class Pair(Namespace):
                    class Do(Command):
                        class Placed(DomainEvent):
                            pass

                        class Rejected(DomainEvent):
                            pass

                        def handle(self) -> _Decoy.Do.Placed:
                            return _Decoy.Do.Placed()

                with pytest.raises(TypeError) as exc:
                    EventGraph([Pair.Do])

                msg = str(exc.value)
                assert "Rejected" in msg
                assert "Add them to the annotation" in msg

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

        def when_single_outcome_command_misses_annotation():

            def it_lists_the_outcome_only_once():
                with pytest.raises(TypeError) as exc:
                    EventGraph([Shop11.Buy])
                msg = str(exc.value)
                assert "Done, Done" not in msg
                assert re.search(r"does not cover outcome\(s\): Done\b", msg)

        def when_inline_handle_uses_bare_scatter():

            def it_raises():
                with pytest.raises(TypeError, match=r"bare `Scatter`") as exc:
                    EventGraph([Shop12.Buy])
                assert "Scatter[Done]" in str(exc.value)


def describe_handle_aliased_across_commands():

    def when_second_command_reuses_first_handle():

        def it_raises():
            RightAgg.Do.__command_handler__ = LeftAgg.Do.__command_handler__

            EventGraph([LeftAgg.Do])
            with pytest.raises(TypeError, match=r"already bound"):
                EventGraph([RightAgg.Do])


class _Decoy(Namespace):
    """Carries a second class also called ``Placed`` — the #151 collision."""

    class Do(Command):
        class Placed(DomainEvent):
            pass
