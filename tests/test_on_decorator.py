"""Tests for the @on decorator and handler metadata extraction."""

import warnings

import pytest

from langgraph_events import Event, EventGraph, EventLog, on
from langgraph_events._handler import extract_handler_meta


class SampleEvent(Event):
    x: int = 0


class EventA(Event):
    a: str = ""


class EventB(Event):
    b: str = ""


def describe_on_decorator():

    def when_single_event_type():

        def it_attaches_event_type_tuple():
            @on(SampleEvent)
            async def handler(event: SampleEvent):
                pass

            assert handler._event_types == (SampleEvent,)

    def when_multiple_event_types():

        def it_attaches_all_event_types():
            @on(EventA, EventB)
            async def handler(event: Event):
                pass

            assert handler._event_types == (EventA, EventB)

    def when_no_arguments():

        def it_raises_type_error():
            with pytest.raises(TypeError, match="at least one"):

                @on()
                async def handler(event):
                    pass

    def when_non_event_class():

        def it_raises_type_error():
            with pytest.raises(TypeError, match="Event subclass"):

                @on(str)  # type: ignore
                async def handler(event):
                    pass

    def when_mixed_valid_and_invalid():

        def it_raises_type_error():
            with pytest.raises(TypeError, match="Event subclasses"):

                @on(EventA, str)  # type: ignore
                async def handler(event):
                    pass


def describe_extract_handler_meta():

    def it_extracts_event_types_and_name():
        @on(SampleEvent)
        async def my_handler(event: SampleEvent):
            pass

        meta = extract_handler_meta(my_handler)
        assert meta.event_types == (SampleEvent,)
        assert "my_handler" in meta.name

    def it_detects_async_handlers():
        @on(SampleEvent)
        async def handler(event: SampleEvent):
            pass

        meta = extract_handler_meta(handler)
        assert meta.is_async is True

    def it_detects_sync_handlers():
        @on(SampleEvent)
        def handler(event: SampleEvent):
            pass

        meta = extract_handler_meta(handler)
        assert meta.is_async is False

    def when_handler_wants_log():

        def it_sets_wants_log_true():
            @on(SampleEvent)
            async def handler(event: SampleEvent, log: EventLog):
                pass

            meta = extract_handler_meta(handler)
            assert meta.wants_log is True

    def when_handler_has_no_log():

        def it_sets_wants_log_false():
            @on(SampleEvent)
            async def handler(event: SampleEvent):
                pass

            meta = extract_handler_meta(handler)
            assert meta.wants_log is False

    def when_function_not_decorated():

        def it_raises_value_error():
            def plain_fn(event):
                pass

            with pytest.raises(ValueError, match="not decorated"):
                extract_handler_meta(plain_fn)

    def when_reducer_params():

        def it_detects_matching_param_names():
            @on(SampleEvent)
            def handler(event: SampleEvent, messages: list, history: list):
                pass

            meta = extract_handler_meta(
                handler, reducer_names=frozenset({"messages", "history"})
            )
            assert set(meta.reducer_params) == {"messages", "history"}

        def it_ignores_non_reducer_params():
            @on(SampleEvent)
            def handler(event: SampleEvent, messages: list, other: str):
                pass

            meta = extract_handler_meta(handler, reducer_names=frozenset({"messages"}))
            assert meta.reducer_params == ("messages",)
            assert "other" not in meta.reducer_params
            assert "event" not in meta.reducer_params

    def when_misspelled_reducer_param():

        def it_warns_about_typo():
            @on(SampleEvent)
            def handler(event: SampleEvent, mesages: list):
                pass

            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                extract_handler_meta(handler, reducer_names=frozenset({"messages"}))

            assert len(w) == 1
            assert "mesages" in str(w[0].message)
            assert "messages" in str(w[0].message)
            assert "Typo?" in str(w[0].message)

        def it_does_not_warn_on_correct_name():
            @on(SampleEvent)
            def handler(event: SampleEvent, messages: list):
                pass

            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                extract_handler_meta(handler, reducer_names=frozenset({"messages"}))

            assert len(w) == 0

        def it_does_not_warn_without_reducers():
            @on(SampleEvent)
            def handler(event: SampleEvent, whatever: str):
                pass

            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                extract_handler_meta(handler, reducer_names=frozenset())

            assert len(w) == 0

    def when_multi_subscription_meta():

        def it_extracts_multiple_event_types_and_log():
            @on(EventA, EventB)
            async def handler(event: Event, log: EventLog):
                pass

            meta = extract_handler_meta(handler)
            assert meta.event_types == (EventA, EventB)
            assert meta.wants_log is True

    def when_type_hints_cannot_be_resolved():

        def it_warns_and_falls_back_to_signature_only_detection():
            @on(SampleEvent)
            def handler(event: SampleEvent, log: EventLog) -> None:
                pass

            handler.__annotations__["event"] = "MissingEvent"
            handler.__annotations__["log"] = "MissingLog"

            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                meta = extract_handler_meta(handler)

            assert len(w) == 1
            assert "Failed to resolve type hints" in str(w[0].message)
            assert meta.log_param is None
            assert meta.event_types == (SampleEvent,)


def describe_return_hint_parsing():

    def when_return_type_hints_cannot_be_resolved():

        def it_warns_and_treats_handler_as_unannotated():
            @on(SampleEvent)
            def handler(event: SampleEvent) -> SampleEvent:
                return SampleEvent()

            handler.__annotations__["return"] = "MissingReturnEvent"

            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                graph = EventGraph([handler])

                messages = [str(item.message) for item in w]
                assert any("Failed to resolve type hints" in msg for msg in messages)
                assert any(
                    "Failed to resolve return type hints" in msg for msg in messages
                )
                assert "-->|handler| ?" in graph.mermaid()
