"""Tests for declared handler exceptions via ``raises=`` + ``HandlerRaised``."""

import asyncio

import pytest
from conftest import Ended, Started

from langgraph_events import (
    Cancelled,
    Event,
    EventGraph,
    HandlerRaised,
    emit_custom,
    on,
)

# Module-level exception + event classes so LangGraph/typing can resolve them.


class DomainError(Exception):
    """A domain exception declared on handlers."""


class ChildError(DomainError):
    """Specialisation of DomainError for MRO matching tests."""


class OtherError(Exception):
    """A second unrelated exception used for multi-raise tests."""


class RecoveryRequested(Event):
    handler: str = ""


class FallbackRan(Event):
    reason: str = ""


def describe_raises():

    def describe_compile_time_coverage():

        def when_declared_raise():

            def with_no_catcher():

                def it_rejects():
                    @on(Started, raises=DomainError)
                    def raiser(event: Started) -> Ended:
                        raise DomainError("boom")

                    with pytest.raises(TypeError, match=r"DomainError.*no handler"):
                        EventGraph([raiser])

            def with_matching_catcher():

                def it_accepts():
                    @on(Started, raises=DomainError)
                    def raiser(event: Started) -> Ended:
                        raise DomainError("boom")

                    @on(HandlerRaised, exception=DomainError)
                    def catcher(event: HandlerRaised) -> FallbackRan:
                        return FallbackRan(reason="caught")

                    EventGraph([raiser, catcher])  # must not raise

            def with_universal_catcher():

                def it_accepts():
                    @on(Started, raises=DomainError)
                    def raiser(event: Started) -> Ended:
                        raise DomainError("boom")

                    @on(HandlerRaised)
                    def catcher(event: HandlerRaised) -> FallbackRan:
                        return FallbackRan(reason="caught any")

                    EventGraph([raiser, catcher])

        def when_child_declared():

            def with_parent_catcher():

                def it_accepts():
                    @on(Started, raises=ChildError)
                    def raiser(event: Started) -> Ended:
                        raise ChildError("boom")

                    @on(HandlerRaised, exception=DomainError)
                    def catcher(event: HandlerRaised) -> FallbackRan:
                        return FallbackRan(reason="caught by parent")

                    EventGraph([raiser, catcher])

        def when_parent_declared():

            def with_child_catcher():

                def it_rejects():
                    @on(Started, raises=DomainError)
                    def raiser(event: Started) -> Ended:
                        raise DomainError("boom")

                    @on(HandlerRaised, exception=ChildError)
                    def catcher(event: HandlerRaised) -> FallbackRan:
                        return FallbackRan(reason="too narrow")

                    with pytest.raises(TypeError, match=r"DomainError.*no handler"):
                        EventGraph([raiser, catcher])

        def when_chained_raise_across_catchers():

            def it_covers_each_link_independently():
                @on(Started, raises=DomainError)
                def raiser(event: Started) -> Ended:
                    raise DomainError("boom")

                @on(HandlerRaised, exception=DomainError, raises=OtherError)
                def intermediate(event: HandlerRaised) -> FallbackRan:
                    raise OtherError("chained")

                @on(HandlerRaised, exception=OtherError)
                def final(event: HandlerRaised) -> FallbackRan:
                    return FallbackRan(reason="final caught")

                EventGraph([raiser, intermediate, final])

    def describe_runtime_sync():

        def it_emits_handler_raised_and_routes_to_catcher():
            @on(Started, raises=DomainError)
            def raiser(event: Started) -> Ended:
                raise DomainError("boom")

            @on(HandlerRaised, exception=DomainError)
            def catcher(event: HandlerRaised) -> FallbackRan:
                return FallbackRan(reason=str(event.exception))

            graph = EventGraph([raiser, catcher])
            log = graph.invoke(Started(data="hi"))
            assert log.latest(FallbackRan) == FallbackRan(reason="boom")
            hr = log.latest(HandlerRaised)
            assert hr is not None
            assert hr.handler == "raiser"
            assert isinstance(hr.exception, DomainError)

        def it_injects_typed_exception_into_catcher_parameter():
            captured: list[Exception] = []

            @on(Started, raises=DomainError)
            def raiser(event: Started) -> Ended:
                raise DomainError("msg")

            @on(HandlerRaised, exception=DomainError)
            def catcher(event: HandlerRaised, exception: DomainError) -> FallbackRan:
                captured.append(exception)
                return FallbackRan(reason="ok")

            graph = EventGraph([raiser, catcher])
            graph.invoke(Started(data="hi"))
            assert len(captured) == 1
            assert isinstance(captured[0], DomainError)

        def it_routes_third_party_exception():
            @on(Started, raises=TimeoutError)
            def raiser(event: Started) -> Ended:
                raise TimeoutError("slow")

            @on(HandlerRaised, exception=TimeoutError)
            def catcher(event: HandlerRaised) -> FallbackRan:
                return FallbackRan(reason="timed out")

            graph = EventGraph([raiser, catcher])
            log = graph.invoke(Started(data="hi"))
            assert log.latest(FallbackRan).reason == "timed out"

        def it_routes_each_exception_in_tuple_to_correct_catcher():
            @on(Started, raises=(DomainError, OtherError))
            def raiser(event: Started) -> Ended:
                raise OtherError("second")

            @on(HandlerRaised, exception=DomainError)
            def catch_domain(event: HandlerRaised) -> FallbackRan:
                return FallbackRan(reason="domain")

            @on(HandlerRaised, exception=OtherError)
            def catch_other(event: HandlerRaised) -> FallbackRan:
                return FallbackRan(reason="other")

            graph = EventGraph([raiser, catch_domain, catch_other])
            log = graph.invoke(Started(data="hi"))
            assert log.latest(FallbackRan).reason == "other"

        def it_universal_handler_raised_catches_any():
            @on(Started, raises=DomainError)
            def raiser(event: Started) -> Ended:
                raise DomainError("boom")

            @on(HandlerRaised)
            def catcher(event: HandlerRaised) -> FallbackRan:
                return FallbackRan(reason="universal")

            graph = EventGraph([raiser, catcher])
            log = graph.invoke(Started(data="hi"))
            assert log.latest(FallbackRan).reason == "universal"

    def describe_runtime_async():

        async def it_emits_handler_raised_from_async_handler():
            @on(Started, raises=DomainError)
            async def raiser(event: Started) -> Ended:
                raise DomainError("async boom")

            @on(HandlerRaised, exception=DomainError)
            async def catcher(event: HandlerRaised) -> FallbackRan:
                return FallbackRan(reason=str(event.exception))

            graph = EventGraph([raiser, catcher])
            log = await graph.ainvoke(Started(data="hi"))
            assert log.latest(FallbackRan) == FallbackRan(reason="async boom")

    def describe_undeclared_raise():

        def when_not_in_raises_clause():

            def it_bubbles_up():
                @on(Started)
                def raiser(event: Started) -> Ended:
                    raise DomainError("undeclared")

                graph = EventGraph([raiser])
                with pytest.raises(DomainError, match="undeclared"):
                    graph.invoke(Started(data="hi"))

        def when_raised_type_differs_from_declaration():

            def it_bubbles_up():
                @on(Started, raises=DomainError)
                def raiser(event: Started) -> Ended:
                    raise OtherError("different")

                @on(HandlerRaised, exception=DomainError)
                def catcher(event: HandlerRaised) -> FallbackRan:
                    return FallbackRan(reason="never")

                graph = EventGraph([raiser, catcher])
                with pytest.raises(OtherError, match="different"):
                    graph.invoke(Started(data="hi"))

    def describe_cancellation_regression():

        async def it_still_produces_cancelled_for_asyncio_cancel():
            @on(Started)
            async def slow(event: Started) -> Ended:
                await asyncio.sleep(10)
                return Ended(result="never")

            graph = EventGraph([slow])
            task = asyncio.create_task(graph.ainvoke(Started(data="hi")))
            await asyncio.sleep(0.01)
            task.cancel()
            try:
                log = await task
            except asyncio.CancelledError:
                return  # acceptable: framework surfaced cancel as-is
            assert log.latest(Cancelled) is not None

    def describe_emit_custom_before_raise():

        def when_handler_raises():

            async def it_still_delivers_custom_event():
                from langgraph_events import CustomEventFrame

                @on(Started, raises=DomainError)
                def raiser(event: Started) -> Ended:
                    emit_custom("tool.progress", {"pct": 50})
                    raise DomainError("after emit")

                @on(HandlerRaised, exception=DomainError)
                def catcher(event: HandlerRaised) -> FallbackRan:
                    return FallbackRan(reason="caught")

                graph = EventGraph([raiser, catcher])
                items = [
                    item
                    async for item in graph.astream_events(
                        Started(data="hi"),
                        include_custom_events=True,
                    )
                ]
                custom = [i for i in items if isinstance(i, CustomEventFrame)]
                assert len(custom) == 1
                assert custom[0].name == "tool.progress"
