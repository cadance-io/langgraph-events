"""Tests for Event base class, Auditable, and MessageEvent."""

import dataclasses

import pytest
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage

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
            with pytest.raises(AttributeError):
                e.x = 99  # type: ignore

        def it_auto_applies_dataclass():
            class AutoEvent(IntegrationEvent):
                value: str = ""

            e = AutoEvent(value="hello")
            assert e.value == "hello"
            with pytest.raises(AttributeError):
                e.value = "nope"  # type: ignore

        def it_auto_dataclass_produces_real_dataclass():
            class SimpleEvent(IntegrationEvent):
                value: str = ""

            assert dataclasses.is_dataclass(SimpleEvent)
            assert SimpleEvent.__dataclass_params__.frozen

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
            with pytest.raises(AttributeError):
                e.content = "nope"  # type: ignore


def describe_Namespace():

    def when_subclassed():

        def it_stamps_domain_name_from_class_name():
            class Widget(Namespace):
                pass

            assert Widget.__namespace_name__ == "Widget"

        def it_is_not_an_event():
            class Widget(Namespace):
                pass

            assert not issubclass(Widget, Event)

    def when_redefined():

        def with_colliding_name():

            def it_raises():
                class Widget(Namespace):
                    pass

                with pytest.raises(TypeError, match=r"already defined"):

                    class Widget(Namespace):
                        pass


def describe_Command():

    def when_top_level():

        def it_rejects():
            with pytest.raises(TypeError, match=r"Command.*must be nested.*Namespace"):

                class Place(Command):
                    pass

    def when_nested_in_domain():

        def it_accepts_and_stamps_domain():
            class Widget(Namespace):
                class Place(Command):
                    customer_id: str = ""

            assert Widget.Place.__namespace__ == "Widget"

    def when_nested_in_non_domain_class():

        def it_rejects():
            with pytest.raises(RuntimeError) as exc_info:

                class NotDomain:
                    class Place(Command):
                        pass

            assert isinstance(exc_info.value.__cause__, TypeError)
            assert "must be nested" in str(exc_info.value.__cause__)
            assert "Namespace" in str(exc_info.value.__cause__)

    def when_nested_in_command():

        def it_rejects():
            with pytest.raises(RuntimeError) as exc_info:

                class Widget(Namespace):
                    class Place(Command):
                        class Inner(Command):
                            pass

            assert isinstance(exc_info.value.__cause__, TypeError)
            assert "must be nested" in str(exc_info.value.__cause__)
            assert "Namespace" in str(exc_info.value.__cause__)


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
            with pytest.raises(RuntimeError) as exc_info:

                class NotDomain:
                    class Placed(DomainEvent):
                        pass

            assert isinstance(exc_info.value.__cause__, TypeError)
            assert "must be nested" in str(exc_info.value.__cause__)

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


def describe_on_namespace_finalize():
    def when_callback_registered_during_class_body():
        def it_fires_after_the_enclosing_Namespace_body_completes():
            captured: list[type] = []

            class MyNs(Namespace):
                class Cmd(Command):
                    class Done(DomainEvent):
                        pass

                on_namespace_finalize(Cmd, captured.append)

            assert captured == [MyNs.Cmd]

    def when_callback_needs_a_sibling_defined_later_in_the_namespace_body():
        def it_can_resolve_the_sibling_at_callback_time():
            from langgraph_events._event import _NAMESPACE_REGISTRY

            captured: list[type] = []

            def capture_sibling(cls):
                def cb(c):
                    ns_cls = _NAMESPACE_REGISTRY[c.__namespace__]
                    captured.append(ns_cls.Sibling)

                on_namespace_finalize(cls, cb)
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
