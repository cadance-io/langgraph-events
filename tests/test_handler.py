"""Tests for @on decorator and handler metadata extraction."""

import pytest

from langgraph_events import Event, EventLog, on
from langgraph_events._handler import extract_handler_meta


class SampleEvent(Event):
    x: int = 0


def test_on_decorator_attaches_event_type():
    @on(SampleEvent)
    async def handler(event: SampleEvent):
        pass

    assert handler._event_types == (SampleEvent,)


def test_on_rejects_non_event():
    with pytest.raises(TypeError, match="Event subclass"):

        @on(str)  # type: ignore
        async def handler(event):
            pass


def test_extract_handler_meta_basic():
    @on(SampleEvent)
    async def my_handler(event: SampleEvent):
        pass

    meta = extract_handler_meta(my_handler)
    assert meta.event_types == (SampleEvent,)
    assert meta.wants_log is False
    assert meta.is_async is True
    assert "my_handler" in meta.name


def test_extract_handler_meta_with_log():
    @on(SampleEvent)
    async def handler(event: SampleEvent, log: EventLog):
        pass

    meta = extract_handler_meta(handler)
    assert meta.wants_log is True


def test_extract_handler_meta_sync():
    @on(SampleEvent)
    def handler(event: SampleEvent):
        pass

    meta = extract_handler_meta(handler)
    assert meta.is_async is False


def test_extract_handler_meta_not_decorated():
    def plain_fn(event):
        pass

    with pytest.raises(ValueError, match="not decorated"):
        extract_handler_meta(plain_fn)


# --- Multi-subscription tests ---


class EventA(Event):
    a: str = ""


class EventB(Event):
    b: str = ""


def test_on_multi_subscription():
    @on(EventA, EventB)
    async def handler(event: Event):
        pass

    assert handler._event_types == (EventA, EventB)


def test_on_multi_subscription_meta():
    @on(EventA, EventB)
    async def handler(event: Event, log: EventLog):
        pass

    meta = extract_handler_meta(handler)
    assert meta.event_types == (EventA, EventB)
    assert meta.wants_log is True


def test_on_rejects_empty():
    with pytest.raises(TypeError, match="at least one"):

        @on()
        async def handler(event):
            pass


def test_on_rejects_mixed_non_event():
    with pytest.raises(TypeError, match="Event subclasses"):

        @on(EventA, str)  # type: ignore
        async def handler(event):
            pass


# --- Reducer param detection tests ---


def test_extract_handler_meta_with_reducer_params():
    @on(SampleEvent)
    def handler(event: SampleEvent, messages: list, history: list):
        pass

    meta = extract_handler_meta(
        handler, reducer_names=frozenset({"messages", "history"})
    )
    assert set(meta.reducer_params) == {"messages", "history"}


def test_extract_handler_meta_ignores_non_reducer_params():
    @on(SampleEvent)
    def handler(event: SampleEvent, messages: list, other: str):
        pass

    meta = extract_handler_meta(handler, reducer_names=frozenset({"messages"}))
    assert meta.reducer_params == ("messages",)
    # "other" is not a reducer param, and "event" is not either
    assert "other" not in meta.reducer_params
    assert "event" not in meta.reducer_params


# --- Reducer name mismatch warning tests ---


def test_warns_on_misspelled_reducer_param():
    @on(SampleEvent)
    def handler(event: SampleEvent, mesages: list):  # typo: "mesages"
        pass

    import warnings

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        extract_handler_meta(handler, reducer_names=frozenset({"messages"}))

    assert len(w) == 1
    assert "mesages" in str(w[0].message)
    assert "messages" in str(w[0].message)
    assert "Typo?" in str(w[0].message)


def test_no_warning_on_correct_reducer_param():
    @on(SampleEvent)
    def handler(event: SampleEvent, messages: list):
        pass

    import warnings

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        extract_handler_meta(handler, reducer_names=frozenset({"messages"}))

    assert len(w) == 0


def test_no_warning_without_reducers():
    @on(SampleEvent)
    def handler(event: SampleEvent, whatever: str):
        pass

    import warnings

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        extract_handler_meta(handler, reducer_names=frozenset())

    assert len(w) == 0
