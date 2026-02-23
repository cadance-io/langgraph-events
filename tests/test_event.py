"""Tests for Event base class and special events."""

import pytest
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

from langgraph_events import (
    Auditable,
    Event,
    Halt,
    Interrupted,
    MessageEvent,
    Resumed,
    Scatter,
    SystemPromptSet,
)


def test_event_is_frozen():
    class MyEvent(Event):
        x: int = 0

    e = MyEvent(x=42)
    assert e.x == 42
    # Frozen — can't mutate
    with pytest.raises(AttributeError):
        e.x = 99  # type: ignore


def test_halt_event():
    h = Halt(reason="done")
    assert h.reason == "done"
    assert isinstance(h, Event)
    assert isinstance(h, Halt)


def test_interrupted_event():
    i = Interrupted(prompt="Approve?", payload={"doc_id": "123"})
    assert i.prompt == "Approve?"
    assert i.payload == {"doc_id": "123"}
    assert isinstance(i, Event)


def test_resumed_event():
    i = Interrupted(prompt="Approve?")
    r = Resumed(value="yes", interrupted=i)
    assert r.value == "yes"
    assert r.interrupted is i
    assert isinstance(r, Event)


def test_multiple_inheritance():
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


def test_single_inheritance_chain():
    class Base(Event):
        x: str = ""

    class Child(Base):
        y: str = ""

    e = Child(x="a", y="b")
    assert isinstance(e, Base)
    assert isinstance(e, Event)
    assert isinstance(e, Child)


# --- Scatter tests ---


def test_scatter_wraps_events():
    class Item(Event):
        v: int = 0

    s = Scatter([Item(v=1), Item(v=2), Item(v=3)])
    assert len(s.events) == 3
    assert s.events[0] == Item(v=1)


def test_scatter_rejects_empty():
    with pytest.raises(ValueError, match="at least one"):
        Scatter([])


def test_scatter_rejects_non_events():
    with pytest.raises(TypeError, match="Event instances"):
        Scatter(["not an event"])  # type: ignore


# --- Auditable tests ---


def test_auditable_trail_basic():
    class OrderPlaced(Auditable):
        order_id: str = ""
        total: float = 0.0

    e = OrderPlaced(order_id="A1", total=99.99)
    trail = e.trail()
    assert trail.startswith("[OrderPlaced]")
    assert "order_id='A1'" in trail
    assert "total=99.99" in trail


def test_auditable_trail_truncates_long_strings():
    class LongContent(Auditable):
        content: str = ""

    long_str = "x" * 200
    e = LongContent(content=long_str)
    trail = e.trail()
    # String > 80 chars should be truncated with "..."
    assert "..." in trail
    assert len(trail) < len(long_str) + 50  # trail is much shorter than raw value


def test_auditable_trail_truncates_long_tuples():
    class BatchEvent(Auditable):
        items: tuple = ()

    e = BatchEvent(items=(1, 2, 3, 4, 5))
    trail = e.trail()
    assert "(5 items)" in trail


# --- MessageEvent tests ---


def test_message_event_single_message():
    class UserMsg(MessageEvent):
        message: HumanMessage = None  # type: ignore[assignment]

    msg = HumanMessage(content="hello")
    event = UserMsg(message=msg)
    assert event.as_messages() == [msg]


def test_message_event_messages_field():
    class ToolResults(MessageEvent):
        messages: tuple[ToolMessage, ...] = ()

    t1 = ToolMessage(content="42", tool_call_id="tc1")
    t2 = ToolMessage(content="7", tool_call_id="tc2")
    event = ToolResults(messages=(t1, t2))
    assert event.as_messages() == [t1, t2]


def test_message_event_neither_field_raises():
    class BadEvent(MessageEvent):
        text: str = ""

    event = BadEvent(text="hi")
    with pytest.raises(NotImplementedError, match="must declare"):
        event.as_messages()


# --- SystemPromptSet tests ---


def test_system_prompt_set_is_message_event():
    msg = SystemMessage(content="You are helpful")
    event = SystemPromptSet(message=msg)
    assert isinstance(event, MessageEvent)
    assert isinstance(event, Event)
    assert event.message is msg


def test_system_prompt_set_as_messages():
    msg = SystemMessage(content="Be nice")
    event = SystemPromptSet(message=msg)
    result = event.as_messages()
    assert result == [msg]


def test_system_prompt_set_from_str():
    event = SystemPromptSet.from_str("You are a helpful assistant")
    assert isinstance(event, SystemPromptSet)
    assert isinstance(event, MessageEvent)
    msgs = event.as_messages()
    assert len(msgs) == 1
    assert isinstance(msgs[0], SystemMessage)
    assert msgs[0].content == "You are a helpful assistant"


def test_system_prompt_set_is_frozen():
    event = SystemPromptSet.from_str("test")
    with pytest.raises(AttributeError):
        event.message = SystemMessage(content="changed")  # type: ignore


# --- Auto-dataclass via __init_subclass__ tests ---


def test_auto_dataclass_without_decorator():
    """Event subclass without @dataclass is automatically frozen."""

    class AutoEvent(Event):
        value: str = ""

    e = AutoEvent(value="hello")
    assert e.value == "hello"
    with pytest.raises(AttributeError):
        e.value = "nope"  # type: ignore


def test_auto_dataclass_is_real_dataclass():
    """Auto-applied classes are proper dataclasses."""

    class SimpleEvent(Event):
        value: str = ""

    import dataclasses

    assert dataclasses.is_dataclass(SimpleEvent)
    assert SimpleEvent.__dataclass_params__.frozen


def test_auto_dataclass_multi_level_inheritance():
    """Auto-dataclass works through multiple inheritance levels."""

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


def test_auto_dataclass_auditable_trail():
    """Auditable subclass without decorator has working trail()."""

    class TrackedOrder(Auditable):
        order_id: str = ""

    e = TrackedOrder(order_id="A1")
    trail = e.trail()
    assert "[TrackedOrder]" in trail
    assert "order_id='A1'" in trail
