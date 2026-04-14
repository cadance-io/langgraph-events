"""Tests for the @on decorator and handler metadata extraction."""

import asyncio
import warnings

import pytest

from langgraph_events import Event, EventGraph, EventLog, HandlerRaised, on
from langgraph_events._event import Interrupted, Resumed
from langgraph_events._handler import extract_handler_meta


class _DomainError(Exception):
    pass


class _OtherError(Exception):
    pass


class SampleEvent(Event):
    x: int = 0


class EventA(Event):
    a: str = ""


class EventB(Event):
    b: str = ""


class ApprovalRequested(Interrupted):
    draft: str = ""


class OtherInterrupted(Interrupted):
    reason: str = ""


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

    def when_field_matcher_provided():

        def it_attaches_field_matchers():
            @on(Resumed, interrupted=ApprovalRequested)
            def handler(event: Resumed, interrupted: ApprovalRequested):
                pass

            assert handler._field_matchers == {"interrupted": ApprovalRequested}

        def it_attaches_event_types_alongside():
            @on(Resumed, interrupted=ApprovalRequested)
            def handler(event: Resumed, interrupted: ApprovalRequested):
                pass

            assert handler._event_types == (Resumed,)

    def when_field_matcher_references_nonexistent_field():

        def it_raises_type_error():
            with pytest.raises(TypeError, match=r"no field.*bogus"):

                @on(Resumed, bogus=ApprovalRequested)
                def handler(event: Resumed):
                    pass

    def when_field_matcher_value_is_not_event_subclass():

        def it_raises_type_error():
            with pytest.raises(TypeError, match="Event or Exception"):

                @on(Resumed, interrupted=str)  # type: ignore
                def handler(event: Resumed):
                    pass

    def when_field_matcher_value_is_exception_subclass():

        def it_accepts_it():
            @on(HandlerRaised, exception=_DomainError)
            def handler(event: HandlerRaised):
                pass

            assert handler._field_matchers == {"exception": _DomainError}

    def when_raises_single_exception_class():

        def it_accepts_and_normalises_to_tuple():
            @on(Resumed, raises=_DomainError)
            def handler(event: Resumed):
                raise _DomainError

            assert handler._raises == (_DomainError,)

    def when_raises_tuple():

        def it_accepts_multiple_exceptions():
            @on(Resumed, raises=(_DomainError, _OtherError))
            def handler(event: Resumed):
                pass

            assert handler._raises == (_DomainError, _OtherError)

    def when_raises_empty_tuple():

        def it_accepts_and_stores_empty_tuple():
            @on(Resumed, raises=())
            def handler(event: Resumed):
                pass

            assert getattr(handler, "_raises", ()) == ()

    def when_raises_omitted():

        def it_defaults_to_empty_tuple():
            @on(Resumed)
            def handler(event: Resumed):
                pass

            assert getattr(handler, "_raises", ()) == ()

    def when_raises_is_not_a_type():

        def it_raises_type_error():
            with pytest.raises(TypeError, match="Exception"):

                @on(Resumed, raises=42)  # type: ignore
                def handler(event: Resumed):
                    pass

    def when_raises_is_baseexception():

        def it_rejects_baseexception():
            with pytest.raises(TypeError, match="Exception"):

                @on(Resumed, raises=BaseException)  # type: ignore
                def handler(event: Resumed):
                    pass

        def it_rejects_keyboard_interrupt():
            with pytest.raises(TypeError, match="Exception"):

                @on(Resumed, raises=KeyboardInterrupt)  # type: ignore
                def handler(event: Resumed):
                    pass

        def it_rejects_system_exit():
            with pytest.raises(TypeError, match="Exception"):

                @on(Resumed, raises=SystemExit)  # type: ignore
                def handler(event: Resumed):
                    pass

        def it_rejects_cancelled_error():
            with pytest.raises(TypeError, match="Exception"):

                @on(Resumed, raises=asyncio.CancelledError)  # type: ignore
                def handler(event: Resumed):
                    pass

        def it_rejects_generator_exit():
            with pytest.raises(TypeError, match="Exception"):

                @on(Resumed, raises=GeneratorExit)  # type: ignore
                def handler(event: Resumed):
                    pass

    def when_raises_is_plain_non_exception_class():

        def it_raises_type_error():
            class Plain:
                pass

            with pytest.raises(TypeError, match="Exception"):

                @on(Resumed, raises=Plain)  # type: ignore
                def handler(event: Resumed):
                    pass

    def when_raises_is_event_subclass_but_not_exception():

        def it_raises_type_error():
            with pytest.raises(TypeError, match="Exception"):

                @on(Resumed, raises=EventA)  # type: ignore
                def handler(event: Resumed):
                    pass


def describe_extract_handler_meta():

    def when_basic_handler():

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

        def it_does_not_warn_for_empty_reducer_set():
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

    def when_field_matchers_in_meta():

        def it_extracts_field_matchers():
            @on(Resumed, interrupted=ApprovalRequested)
            def handler(event: Resumed, interrupted: ApprovalRequested):
                pass

            meta = extract_handler_meta(handler)
            assert meta.field_matchers == (("interrupted", ApprovalRequested),)

        def it_identifies_field_inject_params_from_signature():
            @on(Resumed, interrupted=ApprovalRequested)
            def handler(event: Resumed, interrupted: ApprovalRequested):
                pass

            meta = extract_handler_meta(handler)
            assert meta.field_inject_params == frozenset({"interrupted"})

        def when_field_param_not_in_signature():

            def it_omits_field_inject():
                @on(Resumed, interrupted=ApprovalRequested)
                def handler(event: Resumed):
                    pass

                meta = extract_handler_meta(handler)
                assert meta.field_inject_params == frozenset()

        def it_does_not_warn_about_field_inject_params():
            @on(Resumed, interrupted=ApprovalRequested)
            def handler(event: Resumed, interrupted: ApprovalRequested):
                pass

            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                extract_handler_meta(handler, reducer_names=frozenset({"messages"}))

            typo_warnings = [x for x in w if "Typo?" in str(x.message)]
            assert len(typo_warnings) == 0

    def when_raises_declared():

        def it_extracts_raises_tuple():
            @on(SampleEvent, raises=(_DomainError, _OtherError))
            def handler(event: SampleEvent):
                pass

            meta = extract_handler_meta(handler)
            assert meta.raises == (_DomainError, _OtherError)

        def it_extracts_single_as_tuple():
            @on(SampleEvent, raises=_DomainError)
            def handler(event: SampleEvent):
                pass

            meta = extract_handler_meta(handler)
            assert meta.raises == (_DomainError,)

    def when_raises_omitted_from_decorator():

        def it_defaults_raises_to_empty_tuple():
            @on(SampleEvent)
            def handler(event: SampleEvent):
                pass

            meta = extract_handler_meta(handler)
            assert meta.raises == ()

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
                assert all(item.filename == __file__ for item in w)
                assert "-->|handler| ?" in graph.mermaid()
