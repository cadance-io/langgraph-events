"""Tests for Event base class, Auditable, and MessageEvent."""

import dataclasses

import pytest
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage

from langgraph_events import (
    Auditable,
    Event,
    FrontendToolCallRequested,
    Interrupted,
    MessageEvent,
)


def describe_Event():

    def when_base_behavior():

        def it_is_frozen_by_default():
            class MyEvent(Event):
                x: int = 0

            e = MyEvent(x=42)
            assert e.x == 42
            with pytest.raises(AttributeError):
                e.x = 99  # type: ignore

        def it_auto_applies_dataclass():
            class AutoEvent(Event):
                value: str = ""

            e = AutoEvent(value="hello")
            assert e.value == "hello"
            with pytest.raises(AttributeError):
                e.value = "nope"  # type: ignore

        def it_auto_dataclass_produces_real_dataclass():
            class SimpleEvent(Event):
                value: str = ""

            assert dataclasses.is_dataclass(SimpleEvent)
            assert SimpleEvent.__dataclass_params__.frozen

    def when_single_inheritance():

        def it_matches_isinstance_for_parent():
            class Base(Event):
                x: str = ""

            class Child(Base):
                y: str = ""

            e = Child(x="a", y="b")
            assert isinstance(e, Base)
            assert isinstance(e, Event)
            assert isinstance(e, Child)

    def when_multiple_inheritance():

        def it_matches_isinstance_for_both_parents():
            class TypeA(Event):
                a: str = ""

            class TypeB(Event):
                b: str = ""

            class Both(TypeA, TypeB):
                a: str = ""
                b: str = ""

            e = Both(a="x", b="y")
            assert isinstance(e, TypeA)
            assert isinstance(e, TypeB)
            assert isinstance(e, Event)
            assert isinstance(e, Both)


def describe_Auditable():

    def describe_trail():

        def when_default_formatting():

            def it_includes_class_name_and_field_values():
                class OrderPlaced(Auditable):
                    order_id: str = ""
                    total: float = 0.0

                e = OrderPlaced(order_id="A1", total=99.99)
                trail = e.trail()
                assert trail.startswith("[OrderPlaced]")
                assert "order_id='A1'" in trail
                assert "total=99.99" in trail

        def when_string_exceeds_80_chars():

            def it_truncates():
                class LongContent(Auditable):
                    content: str = ""

                long_str = "x" * 200
                e = LongContent(content=long_str)
                trail = e.trail()
                assert "..." in trail
                assert len(trail) < len(long_str) + 50

        def when_tuple_exceeds_3_items():

            def it_shows_item_count():
                class BatchEvent(Auditable):
                    items: tuple = ()

                e = BatchEvent(items=(1, 2, 3, 4, 5))
                trail = e.trail()
                assert "(5 items)" in trail

        def when_tuple_has_exactly_3_items():

            def it_shows_full_repr_not_summary():
                class SmallBatch(Auditable):
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
                class BigData(Auditable):
                    data: list = None  # type: ignore[assignment]

                # Use a list (not str, not tuple) with a repr > 80 chars
                long_list = list(range(50))
                e = BigData(data=long_list)
                trail = e.trail()
                # The repr() of the list is truncated at 77 chars + "..."
                assert "..." in trail
                assert len(trail) < 200

    def it_produces_trail_for_auto_dataclass():
        class TrackedOrder(Auditable):
            order_id: str = ""

        e = TrackedOrder(order_id="A1")
        trail = e.trail()
        assert "[TrackedOrder]" in trail
        assert "order_id='A1'" in trail


def describe_MessageEvent():

    def when_single_message_field():

        def it_returns_message_in_list():
            class UserMsg(MessageEvent):
                message: HumanMessage = None  # type: ignore[assignment]

            msg = HumanMessage(content="hello")
            event = UserMsg(message=msg)
            assert event.as_messages() == [msg]

    def when_messages_field():

        def it_converts_tuple_to_list():
            class ToolResults(MessageEvent):
                messages: tuple[ToolMessage, ...] = ()

            t1 = ToolMessage(content="42", tool_call_id="tc1")
            t2 = ToolMessage(content="7", tool_call_id="tc2")
            event = ToolResults(messages=(t1, t2))
            assert event.as_messages() == [t1, t2]

    def when_empty_messages_field():

        def it_returns_empty_list():
            class Empty(MessageEvent):
                messages: tuple[ToolMessage, ...] = ()

            event = Empty()
            assert event.as_messages() == []

    def when_no_message_or_messages_field():

        def it_raises_not_implemented():
            class BadEvent(MessageEvent):
                text: str = ""

            event = BadEvent(text="hi")
            with pytest.raises(NotImplementedError, match="must declare"):
                event.as_messages()

    def when_custom_override():

        def it_uses_overridden_method():
            class Custom(MessageEvent):
                text: str = ""

                def as_messages(self) -> list[BaseMessage]:
                    return [HumanMessage(content=self.text)]

            event = Custom(text="hello")
            result = event.as_messages()
            assert len(result) == 1
            assert result[0].content == "hello"

    def when_ai_message_has_tool_calls():

        def it_preserves_tool_calls():
            class LLMResponse(MessageEvent):
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
            class Mid(MessageEvent):
                message: HumanMessage = None  # type: ignore[assignment]

            class Leaf(Mid):
                content: str = ""

            msg = HumanMessage(content="hi")
            e = Leaf(message=msg, content="extra")
            assert e.content == "extra"
            assert e.as_messages() == [msg]
            with pytest.raises(AttributeError):
                e.content = "nope"  # type: ignore


def describe_FrontendToolCallRequested():

    def when_only_name_provided():

        def it_is_an_interrupted_subclass():
            e = FrontendToolCallRequested(name="confirm")
            assert isinstance(e, Interrupted)
            assert isinstance(e, Event)

        def it_defaults_args_to_empty_dict():
            e = FrontendToolCallRequested(name="confirm")
            assert e.args == {}

        def it_auto_generates_tool_call_id():
            a = FrontendToolCallRequested(name="confirm")
            b = FrontendToolCallRequested(name="confirm")
            assert a.tool_call_id
            assert b.tool_call_id
            assert a.tool_call_id != b.tool_call_id

    def when_explicit_fields():

        def it_preserves_all_fields():
            e = FrontendToolCallRequested(
                name="run_scenario",
                args={"scenario_id": "s-1"},
                tool_call_id="tc-fixed",
            )
            assert e.name == "run_scenario"
            assert e.args == {"scenario_id": "s-1"}
            assert e.tool_call_id == "tc-fixed"

    def when_agui_dict_called():

        def it_returns_name_args_and_id():
            e = FrontendToolCallRequested(
                name="confirm",
                args={"message": "Ship?"},
                tool_call_id="tc-1",
            )
            d = e.agui_dict()
            assert d == {
                "name": "confirm",
                "args": {"message": "Ship?"},
                "tool_call_id": "tc-1",
            }

    def when_name_is_empty():

        def it_raises_on_construction():
            with pytest.raises(ValueError, match=r"non-empty tool name"):
                FrontendToolCallRequested(name="")

    def when_name_is_whitespace():

        def it_raises_on_construction():
            with pytest.raises(ValueError, match=r"non-empty tool name"):
                FrontendToolCallRequested(name="   ")
