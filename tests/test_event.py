"""Tests for Event base class and special events."""

from dataclasses import dataclass

from langgraph_events import Event, Halt, Interrupted, Resumed


def test_event_is_frozen():
    @dataclass(frozen=True)
    class MyEvent(Event):
        x: int = 0

    e = MyEvent(x=42)
    assert e.x == 42
    # Frozen — can't mutate
    try:
        e.x = 99  # type: ignore
        assert False, "Should have raised"
    except AttributeError:
        pass


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
    @dataclass(frozen=True)
    class TypeA(Event):
        a: str = ""

    @dataclass(frozen=True)
    class TypeB(Event):
        b: str = ""

    @dataclass(frozen=True)
    class Both(TypeA, TypeB):
        a: str = ""
        b: str = ""

    e = Both(a="x", b="y")
    assert isinstance(e, TypeA)
    assert isinstance(e, TypeB)
    assert isinstance(e, Event)
    assert isinstance(e, Both)


def test_single_inheritance_chain():
    @dataclass(frozen=True)
    class Base(Event):
        x: str = ""

    @dataclass(frozen=True)
    class Child(Base):
        y: str = ""

    e = Child(x="a", y="b")
    assert isinstance(e, Base)
    assert isinstance(e, Event)
    assert isinstance(e, Child)
