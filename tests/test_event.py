"""Tests for Event base class, Auditable, and MessageEvent."""

import warnings

import pytest
from conftest import SET_NAME_ERRORS, set_name_cause
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from pydantic import ValidationError

from langgraph_events import (
    Auditable,
    Cancelled,
    Command,
    DomainEvent,
    Event,
    Halted,
    HandlerRaised,
    IntegrationEvent,
    Interrupted,
    MaxRoundsExceeded,
    MessageEvent,
    Namespace,
    Resumed,
    RunPaused,
    SystemEvent,
    on_namespace_finalize,
)


def describe_Event():

    def when_base_behavior():

        def it_is_frozen_by_default():
            class MyEvent(IntegrationEvent):
                x: int = 0

            e = MyEvent(x=42)
            assert e.x == 42
            with pytest.raises((AttributeError, ValidationError)):
                e.x = 99  # type: ignore

        def it_auto_applies_dataclass():
            class AutoEvent(IntegrationEvent):
                value: str = ""

            e = AutoEvent(value="hello")
            assert e.value == "hello"
            with pytest.raises((AttributeError, ValidationError)):
                e.value = "nope"  # type: ignore

        def it_is_a_pydantic_model():
            from pydantic import BaseModel

            class SimpleEvent(IntegrationEvent):
                value: str = ""

            assert issubclass(SimpleEvent, BaseModel)

    def when_single_inheritance():

        def it_matches_isinstance_for_parent():
            class Base(IntegrationEvent):
                x: str = ""

            class Child(Base):
                y: str = ""

            e = Child(x="a", y="b")
            assert isinstance(e, Base)
            assert isinstance(e, Event)
            assert isinstance(e, Child)

    def when_multiple_inheritance():

        def it_matches_isinstance_for_both_parents():
            class TypeA(IntegrationEvent):
                a: str = ""

            class TypeB(IntegrationEvent):
                b: str = ""

            class Both(TypeA, TypeB):
                a: str = ""
                b: str = ""

            e = Both(a="x", b="y")
            assert isinstance(e, TypeA)
            assert isinstance(e, TypeB)
            assert isinstance(e, Event)
            assert isinstance(e, Both)

    def when_bare_event_subclass():

        def it_raises_TypeError():
            with pytest.raises(TypeError, match="subclasses Event directly"):

                class Bare(Event):
                    pass

    def when_integration_event_subclass():

        def it_accepts():
            class Ok(IntegrationEvent):
                pass

            assert issubclass(Ok, Event)


def describe_Auditable():

    def describe_trail():

        def when_default_formatting():

            def it_includes_class_name_and_field_values():
                class OrderPlaced(IntegrationEvent, Auditable):
                    order_id: str = ""
                    total: float = 0.0

                e = OrderPlaced(order_id="A1", total=99.99)
                trail = e.trail()
                assert trail.startswith("[OrderPlaced]")
                assert "order_id='A1'" in trail
                assert "total=99.99" in trail

        def when_string_exceeds_80_chars():

            def it_truncates():
                class LongContent(IntegrationEvent, Auditable):
                    content: str = ""

                long_str = "x" * 200
                e = LongContent(content=long_str)
                trail = e.trail()
                assert "..." in trail
                assert len(trail) < len(long_str) + 50

        def when_tuple_exceeds_3_items():

            def it_shows_item_count():
                class BatchEvent(IntegrationEvent, Auditable):
                    items: tuple = ()

                e = BatchEvent(items=(1, 2, 3, 4, 5))
                trail = e.trail()
                assert "(5 items)" in trail

        def when_tuple_has_exactly_3_items():

            def it_shows_full_repr_not_summary():
                class SmallBatch(IntegrationEvent, Auditable):
                    items: tuple = ()

                e = SmallBatch(items=(1, 2, 3))
                trail = e.trail()
                # 3-item tuple should show individual values, not "(3 items)"
                assert "(3 items)" not in trail
                assert "1" in trail
                assert "2" in trail
                assert "3" in trail

        def when_repr_of_non_string_value_exceeds_80_chars():

            def it_truncates_repr():
                class BigData(IntegrationEvent, Auditable):
                    data: list = None  # type: ignore[assignment]

                # Use a list (not str, not tuple) with a repr > 80 chars
                long_list = list(range(50))
                e = BigData(data=long_list)
                trail = e.trail()
                # The repr() of the list is truncated at 77 chars + "..."
                assert "..." in trail
                assert len(trail) < 200

    def it_produces_trail_for_auto_dataclass():
        class TrackedOrder(IntegrationEvent, Auditable):
            order_id: str = ""

        e = TrackedOrder(order_id="A1")
        trail = e.trail()
        assert "[TrackedOrder]" in trail
        assert "order_id='A1'" in trail


def describe_MessageEvent():

    def when_single_message_field():

        def it_returns_message_in_list():
            class UserMsg(IntegrationEvent, MessageEvent):
                message: HumanMessage = None  # type: ignore[assignment]

            msg = HumanMessage(content="hello")
            event = UserMsg(message=msg)
            assert event.as_messages() == [msg]

    def when_messages_field():

        def it_converts_tuple_to_list():
            class ToolResults(IntegrationEvent, MessageEvent):
                messages: tuple[ToolMessage, ...] = ()

            t1 = ToolMessage(content="42", tool_call_id="tc1")
            t2 = ToolMessage(content="7", tool_call_id="tc2")
            event = ToolResults(messages=(t1, t2))
            assert event.as_messages() == [t1, t2]

    def when_empty_messages_field():

        def it_returns_empty_list():
            class Empty(IntegrationEvent, MessageEvent):
                messages: tuple[ToolMessage, ...] = ()

            event = Empty()
            assert event.as_messages() == []

    def when_no_message_or_messages_field():

        def it_raises_not_implemented():
            class BadEvent(IntegrationEvent, MessageEvent):
                text: str = ""

            event = BadEvent(text="hi")
            with pytest.raises(NotImplementedError, match="must declare"):
                event.as_messages()

    def when_custom_override():

        def it_uses_overridden_method():
            class Custom(IntegrationEvent, MessageEvent):
                text: str = ""

                def as_messages(self) -> list[BaseMessage]:
                    return [HumanMessage(content=self.text)]

            event = Custom(text="hello")
            result = event.as_messages()
            assert len(result) == 1
            assert result[0].content == "hello"

    def when_ai_message_has_tool_calls():

        def it_preserves_tool_calls():
            class LLMResponse(IntegrationEvent, MessageEvent):
                message: AIMessage = None  # type: ignore[assignment]

            ai_msg = AIMessage(
                content="Let me check",
                tool_calls=[{"id": "tc1", "name": "search", "args": {"q": "test"}}],
            )
            event = LLMResponse(message=ai_msg)
            result = event.as_messages()
            assert len(result) == 1
            assert result[0] is ai_msg
            assert result[0].tool_calls == ai_msg.tool_calls

    def when_multi_level_inheritance():

        def it_works_through_multi_level_auto_dataclass():
            class Mid(IntegrationEvent, MessageEvent):
                message: HumanMessage = None  # type: ignore[assignment]

            class Leaf(Mid):
                content: str = ""

            msg = HumanMessage(content="hi")
            e = Leaf(message=msg, content="extra")
            assert e.content == "extra"
            assert e.as_messages() == [msg]
            with pytest.raises((AttributeError, ValidationError)):
                e.content = "nope"  # type: ignore


def describe_Namespace():

    def when_subclassed():

        def it_stamps_domain_name_from_class_name():
            class Widget(Namespace):
                pass

            assert Widget.__namespace_name__ == "Widget"

        def it_does_not_warn():
            with warnings.catch_warnings():
                warnings.simplefilter("error", DeprecationWarning)

                class Widget(Namespace):
                    pass

            assert Widget.__namespace_name__ == "Widget"

        def it_is_not_an_event():
            class Widget(Namespace):
                pass

            assert not issubclass(Widget, Event)

    def when_subclassing_another_namespace():

        # Removed in v0.28.0 after one minor version of deprecation. Only
        # reducers ever inherited; nested commands and events did not, so a
        # child namespace was quietly incomplete — its inherited commands
        # skipped by EventGraph.from_namespaces, its inherited events outside
        # a serde's scope and bleeding across engine lifetimes.
        def it_raises():
            class Parent(Namespace):
                pass

            with pytest.raises(TypeError, match=r"Namespace subclassing"):

                class Kid(Parent):
                    pass

        def it_names_only_the_namespace_bases():
            class Mixin:
                pass

            class Root(Namespace):
                pass

            with pytest.raises(TypeError) as exc:

                class Kid(Mixin, Root):
                    pass

            assert "Root" in str(exc.value)
            assert "Mixin" not in str(exc.value)

        def it_still_allows_a_plain_mixin():
            class Mixin:
                pass

            class Solo(Namespace, Mixin):
                pass

            assert Solo.__namespace_name__ == "Solo"

    def when_redefined():

        # Namespace names are scoped to the graph that uses them, not to the
        # process — a second engine lifetime redefining the same name is
        # valid. The collision that matters (two classes, one name, one
        # graph) is caught at graph build. See issue #148.
        def with_colliding_name():

            def it_does_not_raise():
                class Widget(Namespace):
                    pass

                first = Widget

                class Widget(Namespace):
                    pass

                assert Widget is not first

    def when_nested_events_are_stamped():

        def it_records_the_owning_namespace_class():
            class Widget(Namespace):
                class Build(Command):
                    class Built(DomainEvent):
                        pass

            assert Widget.Build.__namespace_cls__ is Widget
            assert Widget.Build.Built.__namespace_cls__ is Widget

        def it_keeps_both_stamps_in_step():
            class Widget(Namespace):
                class Build(Command):
                    class Built(DomainEvent):
                        pass

            for cls in (Widget.Build, Widget.Build.Built):
                assert cls.__namespace__ == cls.__namespace_cls__.__name__


def describe_Command():

    def when_top_level():

        def it_rejects():
            with pytest.raises(TypeError, match=r"Command.*must be nested.*Namespace"):

                class Place(Command):
                    pass

        def when_it_mixes_in_an_already_stamped_event():
            # A base carrying a ``__namespace__`` does not excuse a Command
            # from the nesting rule — the stamp says where *that* event
            # lives, not where this command does.
            def it_rejects():
                class Widget0(Namespace):
                    class Place(Command):
                        class Placed(DomainEvent):
                            pass

                msg = r"Command.*must be nested.*Namespace"
                with pytest.raises(TypeError, match=msg):

                    class Hybrid(Command, Widget0.Place.Placed):
                        pass

    def when_nested_in_domain():

        def it_accepts_and_stamps_domain():
            class Widget(Namespace):
                class Place(Command):
                    customer_id: str = ""

            assert Widget.Place.__namespace__ == "Widget"

    def when_nested_in_non_domain_class():

        def it_rejects():
            with pytest.raises(SET_NAME_ERRORS) as exc_info:

                class NotDomain:
                    class Place(Command):
                        pass

            cause = set_name_cause(exc_info.value)
            assert isinstance(cause, TypeError)
            assert "must be nested" in str(cause)
            assert "Namespace" in str(cause)

    def when_nested_in_command():

        def it_rejects():
            with pytest.raises(SET_NAME_ERRORS) as exc_info:

                class Widget(Namespace):
                    class Place(Command):
                        class Inner(Command):
                            pass

            cause = set_name_cause(exc_info.value)
            assert isinstance(cause, TypeError)
            assert "must be nested" in str(cause)
            assert "Namespace" in str(cause)

    def when_subclassing_a_concrete_command():
        # ``class Ask(Command)`` is how a Command is declared; ``class
        # Child(Ask)`` is a second intent wearing the first one's identity.

        def it_rejects():
            class Widget(Namespace):
                class Place(Command):
                    class Placed(DomainEvent):
                        pass

                    def handle(self) -> None:
                        return None

            with pytest.raises(TypeError, match=r"may not be subclassed"):

                class Rush(Widget.Place):
                    pass

        def it_names_both_classes_and_says_what_to_do_instead():
            class Widget2(Namespace):
                class Place(Command):
                    def handle(self) -> None:
                        return None

            with pytest.raises(TypeError) as exc_info:

                class Rush(Widget2.Place):
                    pass

            msg = str(exc_info.value)
            assert "Rush" in msg and "Widget2.Place" in msg
            assert "helper" in msg and "Namespace" in msg

        def when_the_parent_declares_no_handler():
            # The rule is structural — it does not depend on the parent
            # carrying a handler, so a handler-less intermediate base is
            # rejected just the same.
            def it_rejects():
                class Widget3(Namespace):
                    class Base(Command):
                        pass

                with pytest.raises(TypeError, match=r"may not be subclassed"):

                    class Derived(Widget3.Base):
                        pass

        def when_the_subclass_is_declared_inside_the_same_namespace():
            def it_rejects():
                with pytest.raises(TypeError, match=r"may not be subclassed"):

                    class Widget4(Namespace):
                        class Place(Command):
                            def handle(self) -> None:
                                return None

                        class Rush(Place):
                            pass

        def when_the_command_composes_a_mixin():
            # Mixins are not Commands — composing one stays legal.
            def it_accepts():
                class Widget5(Namespace):
                    class Place(Command, Auditable):
                        def handle(self) -> None:
                            return None

                assert Widget5.Place.__namespace__ == "Widget5"


def describe_DomainEvent():

    def when_top_level():

        def it_rejects():
            msg = r"DomainEvent.*must be nested.*Namespace"
            with pytest.raises(TypeError, match=msg):

                class Placed(DomainEvent):
                    pass

    def when_nested_in_domain():

        def it_accepts_and_stamps_domain():
            class Widget(Namespace):
                class Shipped(DomainEvent):
                    tracking: str = ""

            assert Widget.Shipped.__namespace__ == "Widget"

        def it_leaves_command_attr_unset():
            class Widget(Namespace):
                class Shipped(DomainEvent):
                    tracking: str = ""

            assert Widget.Shipped.__command__ is None

    def when_nested_in_command():

        def it_accepts_and_stamps_domain_and_command():
            class Widget(Namespace):
                class Place(Command):
                    customer_id: str = ""

                    class Placed(DomainEvent):
                        order_id: str = ""

            assert Widget.Place.Placed.__namespace__ == "Widget"
            assert Widget.Place.Placed.__command__ is Widget.Place

    def when_nested_in_non_domain_class():

        def it_rejects():
            with pytest.raises(SET_NAME_ERRORS) as exc_info:

                class NotDomain:
                    class Placed(DomainEvent):
                        pass

            cause = set_name_cause(exc_info.value)
            assert isinstance(cause, TypeError)
            assert "must be nested" in str(cause)

    def when_subclass_of_validated_event():

        def it_inherits_domain_and_command_attrs():
            class Widget(Namespace):
                class Place(Command):
                    class Placed(DomainEvent):
                        order_id: str = ""

            class FastPlaced(Widget.Place.Placed):
                priority: int = 0

            assert FastPlaced.__namespace__ == "Widget"
            assert FastPlaced.__command__ is Widget.Place

    def when_multiple_commands_and_outcomes_in_one_domain():
        # Invariant-pinning test for the two-pass `__namespace__` stamping in
        # _event.py. When a Command's own DomainEvents are processed by the
        # metaclass, the enclosing Command doesn't yet have `__namespace__`
        # set. `Namespace.__init_subclass__` fills it in via a second pass —
        # this test ensures every nested DomainEvent ends up stamped.

        def it_stamps_every_nested_outcome():
            class Widget(Namespace):
                class Place(Command):
                    class Placed(DomainEvent):
                        order_id: str = ""

                    class Rejected(DomainEvent):
                        reason: str = ""

                class Ship(Command):
                    class Shipped(DomainEvent):
                        tracking: str = ""

            for outcome, parent_cmd in [
                (Widget.Place.Placed, Widget.Place),
                (Widget.Place.Rejected, Widget.Place),
                (Widget.Ship.Shipped, Widget.Ship),
            ]:
                assert outcome.__namespace__ == "Widget"
                assert outcome.__command__ is parent_cmd


def describe_Command_Outcomes():

    def when_command_has_single_outcome():

        def it_exposes_the_single_class_as_Outcomes():
            class AggA(Namespace):
                class Cmd(Command):
                    class Done(DomainEvent):
                        pass

            assert AggA.Cmd.Outcomes is AggA.Cmd.Done

    def when_command_has_multiple_outcomes():

        def it_exposes_a_union_of_all_outcomes():
            import typing

            class AggB(Namespace):
                class Cmd(Command):
                    class Ok(DomainEvent):
                        pass

                    class Err(DomainEvent):
                        pass

            args = set(typing.get_args(AggB.Cmd.Outcomes))
            assert args == {AggB.Cmd.Ok, AggB.Cmd.Err}

    def when_command_has_no_outcomes():

        def it_does_not_define_Outcomes():
            class AggC(Namespace):
                class Cmd(Command):
                    pass

            assert "Outcomes" not in AggC.Cmd.__dict__

    def when_Outcomes_used_in_isinstance():

        def it_matches_any_nested_outcome():
            class AggD(Namespace):
                class Cmd(Command):
                    class A(DomainEvent):
                        pass

                    class B(DomainEvent):
                        pass

            assert isinstance(AggD.Cmd.A(), AggD.Cmd.Outcomes)
            assert isinstance(AggD.Cmd.B(), AggD.Cmd.Outcomes)

    def when_user_declares_Outcomes_matching_nested():

        def it_preserves_user_declaration():
            class AggE(Namespace):
                class Cmd(Command):
                    class A(DomainEvent):
                        pass

                    class B(DomainEvent):
                        pass

                    Outcomes = A | B

            # User's declaration kept; framework didn't overwrite.
            assert AggE.Cmd.Outcomes is AggE.Cmd.__dict__["Outcomes"]

    def when_user_declares_Outcomes_missing_an_outcome():

        def it_rejects_as_drift():
            with pytest.raises(TypeError, match=r"Outcomes.*does not match"):

                class AggF(Namespace):
                    class Cmd(Command):
                        class A(DomainEvent):
                            pass

                        class B(DomainEvent):
                            pass

                        Outcomes = A  # B is missing

    def when_user_declares_Outcomes_including_foreign_type():

        def it_rejects_as_drift():
            class Holder(Namespace):
                class Inner(Command):
                    class Unrelated(DomainEvent):
                        pass

            with pytest.raises(TypeError, match=r"Outcomes.*does not match"):

                class AggG(Namespace):
                    class Cmd(Command):
                        class A(DomainEvent):
                            pass

                        Outcomes = A | Holder.Inner.Unrelated


def describe_IntegrationEvent():

    def when_top_level():

        def it_accepts():
            class PaymentConfirmed(IntegrationEvent):
                transaction_id: str = ""

            assert issubclass(PaymentConfirmed, Event)


def describe_SystemEvent():

    def when_subclassed_by_framework_events():

        def it_makes_Halted_isinstance_SystemEvent():
            assert issubclass(Halted, SystemEvent)

        def it_makes_Interrupted_isinstance_SystemEvent():
            assert issubclass(Interrupted, SystemEvent)

        def it_makes_HandlerRaised_isinstance_SystemEvent():
            assert issubclass(HandlerRaised, SystemEvent)

        def it_makes_Resumed_isinstance_SystemEvent():
            assert issubclass(Resumed, SystemEvent)

        def it_makes_Cancelled_isinstance_SystemEvent():
            assert issubclass(Cancelled, SystemEvent)

        def it_makes_MaxRoundsExceeded_isinstance_SystemEvent():
            assert issubclass(MaxRoundsExceeded, SystemEvent)

        def it_makes_RunPaused_isinstance_SystemEvent():
            assert issubclass(RunPaused, SystemEvent)

        def it_does_not_make_RunPaused_a_Halted_subclass():
            """RunPaused is intentionally *not* a Halted subclass.

            Halted is terminal-across-runs (preserved by MaxRoundsExceeded);
            RunPaused must be resumable on fresh /run, which requires the
            router-level cursor advancement that only kicks in for
            non-Halted system events. Guarding the type hierarchy here
            keeps the design decision visible.
            """
            assert not issubclass(RunPaused, Halted)


def describe_on_namespace_finalize():
    def when_callback_registered_during_class_body():
        def it_fires_after_the_enclosing_Namespace_body_completes():
            captured: list[tuple[type, type]] = []

            class MyNs(Namespace):
                class Cmd(Command):
                    class Done(DomainEvent):
                        pass

                on_namespace_finalize(Cmd, lambda c, ns: captured.append((c, ns)))

            assert captured == [(MyNs.Cmd, MyNs)]

    def when_callback_needs_a_sibling_defined_later_in_the_namespace_body():
        def it_receives_the_enclosing_namespace_class_directly():
            captured: list[type] = []

            def capture_sibling(cls):
                # Callback signature: (cls, namespace_cls). The framework
                # passes the enclosing Namespace class as the second arg so
                # decorators can resolve sibling references via vars(ns_cls)
                # without reaching into private registries.
                on_namespace_finalize(
                    cls, lambda c, ns_cls: captured.append(ns_cls.Sibling)
                )
                return cls

            class LateRefNs(Namespace):
                @capture_sibling
                class Target(Command):
                    pass

                # Defined AFTER Target — would be unresolvable at Target's
                # class-body / __init_subclass__ time. The finalize hook
                # ensures the callback fires once Sibling is bound.
                class Sibling(Command):
                    pass

            assert captured == [LateRefNs.Sibling]

    def when_registered_while_the_enclosing_Namespace_is_still_being_created():
        def it_queues_rather_than_firing_on_a_half_built_class():
            # A nested class is stamped with __namespace_cls__ by
            # __set_name__, which runs *before* the Namespace's
            # __init_subclass__. Anything running in that window — here a
            # sibling descriptor's own __set_name__ — must still queue: the
            # enclosing class has not attached Command.Outcomes yet, which
            # is the whole reason this hook exists.
            seen_outcomes: list[object] = []

            class Sentinel:
                def __set_name__(self, owner, name):
                    cmd = owner.__dict__["Cmd"]
                    assert getattr(cmd, "__namespace_cls__", None) is not None
                    on_namespace_finalize(
                        cmd,
                        lambda c, ns: seen_outcomes.append(
                            getattr(c, "Outcomes", None)
                        ),
                    )
                    # Firing now would hand the callback a Command with no
                    # Outcomes attached.
                    assert seen_outcomes == []

            class MidBuildNs(Namespace):
                class Cmd(Command):
                    class Done(DomainEvent):
                        pass

                sentinel = Sentinel()

            assert seen_outcomes == [MidBuildNs.Cmd.Done]

    def when_registered_after_the_enclosing_Namespace_finalized():
        def it_fires_immediately_instead_of_silently_dropping():
            class FinishedNs(Namespace):
                class Cmd(Command):
                    class Done(DomainEvent):
                        pass

            # Namespace body has completed; FinishedNs.Cmd's enclosing
            # namespace already drained its finalize queue. A late
            # registration must not silently dangle — fire eagerly.
            captured: list[tuple[type, type]] = []
            on_namespace_finalize(
                FinishedNs.Cmd, lambda c, ns: captured.append((c, ns))
            )
            assert captured == [(FinishedNs.Cmd, FinishedNs)]
