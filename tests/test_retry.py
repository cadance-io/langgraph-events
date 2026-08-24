"""Tests for declarative retry via ``RetryPolicy`` + ``HandlerRetried``."""

from __future__ import annotations

import logging
import time
from typing import ClassVar

import pytest
from conftest import Ended, Started

from langgraph_events import (
    Command,
    DomainEvent,
    EventGraph,
    HandlerRaised,
    HandlerRetried,
    IntegrationEvent,
    Namespace,
    RetryPolicy,
    on,
)
from langgraph_events import _retry as _retry_mod


class FlakyError(Exception):
    """A transient failure that a retry policy should absorb."""


class RateLimitedError(FlakyError):
    """Transient failure carrying a server-supplied ``retry_after`` hint."""

    def __init__(self, retry_after: float) -> None:
        super().__init__(f"rate limited, retry after {retry_after}s")
        self.retry_after = retry_after


def describe_retry_policy():
    def describe_delay_for():
        def when_exponential():
            def it_doubles_each_attempt():
                policy = RetryPolicy(base_delay=0.1, jitter=False)
                assert [policy.delay_for(n) for n in (1, 2, 3, 4)] == [
                    0.1,
                    0.2,
                    0.4,
                    0.8,
                ]

        def when_constant():
            def it_returns_the_base_delay():
                policy = RetryPolicy(base_delay=0.25, strategy="constant", jitter=False)
                assert [policy.delay_for(n) for n in (1, 2, 3)] == [0.25, 0.25, 0.25]

        def when_capped():
            def it_never_exceeds_max_delay():
                policy = RetryPolicy(base_delay=1.0, max_delay=3.0, jitter=False)
                assert [policy.delay_for(n) for n in (1, 2, 3, 9)] == [
                    1.0,
                    2.0,
                    3.0,
                    3.0,
                ]

        def when_jitter_enabled():
            def it_samples_between_zero_and_the_ceiling(monkeypatch):
                calls: list[tuple[float, float]] = []

                def fake_uniform(low: float, high: float) -> float:
                    calls.append((low, high))
                    return high / 2

                monkeypatch.setattr(_retry_mod, "_random_uniform", fake_uniform)
                policy = RetryPolicy(base_delay=0.1, jitter=True)
                assert policy.delay_for(3) == 0.2
                assert calls == [(0.0, 0.4)]

        def when_respect_retry_after():
            def it_uses_the_exception_hint():
                policy = RetryPolicy(base_delay=0.1, respect_retry_after=True)
                assert policy.delay_for(1, RateLimitedError(retry_after=5.0)) == 5.0

            def it_still_caps_the_hint():
                policy = RetryPolicy(
                    base_delay=0.1, max_delay=2.0, respect_retry_after=True
                )
                assert policy.delay_for(1, RateLimitedError(retry_after=90.0)) == 2.0

            def it_falls_back_to_the_computed_curve():
                policy = RetryPolicy(
                    base_delay=0.1, jitter=False, respect_retry_after=True
                )
                assert policy.delay_for(2, FlakyError("no hint")) == 0.2

        def when_respect_retry_after_is_off():
            def it_ignores_the_exception_hint():
                policy = RetryPolicy(base_delay=0.1, jitter=False)
                assert policy.delay_for(1, RateLimitedError(retry_after=90.0)) == 0.1

    def describe_sleep_seams():
        def it_blocks_on_the_sync_path():
            start = time.monotonic()
            _retry_mod._sleep(0.01)
            assert time.monotonic() - start >= 0.01

        async def it_awaits_on_the_async_path():
            start = time.monotonic()
            await _retry_mod._asleep(0.01)
            assert time.monotonic() - start >= 0.01

    def describe_validation():
        def when_max_attempts_below_one():
            def it_rejects():
                with pytest.raises(ValueError, match=r"max_attempts"):
                    RetryPolicy(max_attempts=0)

        def when_base_delay_is_negative():
            def it_rejects():
                with pytest.raises(ValueError, match=r"base_delay"):
                    RetryPolicy(base_delay=-1.0)

        def when_max_delay_below_base():
            def it_rejects():
                with pytest.raises(ValueError, match=r"max_delay"):
                    RetryPolicy(base_delay=5.0, max_delay=1.0)

        def when_strategy_is_unknown():
            def it_rejects():
                with pytest.raises(ValueError, match=r"strategy"):
                    RetryPolicy(strategy="fibonacci")  # type: ignore[arg-type]

        def when_observe_is_unknown():
            def it_rejects():
                with pytest.raises(ValueError, match=r"observe"):
                    RetryPolicy(observe="shout")  # type: ignore[arg-type]

        def when_on_is_not_an_exception():
            def it_rejects():
                with pytest.raises(TypeError, match=r"Exception"):
                    RetryPolicy(on=(42,))  # type: ignore[arg-type]

        def when_on_is_a_bare_exception_class():
            def it_normalises_to_a_tuple():
                policy = RetryPolicy(on=FlakyError)  # type: ignore[arg-type]
                assert policy.on == (FlakyError,)


class Recovered(IntegrationEvent):
    """Terminal outcome emitted by the escalation catcher."""

    reason: str = ""


class _Attempts:
    """Mutable call counter shared between a handler and its assertions."""

    def __init__(self, succeed_on: int) -> None:
        self.succeed_on = succeed_on
        self.calls = 0

    def tick(self) -> bool:
        self.calls += 1
        return self.calls >= self.succeed_on


# Inline-command scenarios live at module level so Python can resolve the
# forward references in their ``handle`` return annotations at runtime.
_INLINE = _Attempts(succeed_on=1)


class DeclaredRetry(Namespace):
    """A Command carrying ``retry`` as a class-level modifier."""

    class Cmd(Command):
        raises: ClassVar = (FlakyError,)
        retry: ClassVar = RetryPolicy(max_attempts=3, base_delay=0.1, jitter=False)

        class Done(DomainEvent):
            pass

        def handle(self) -> DeclaredRetry.Cmd.Done:
            if not _INLINE.tick():
                raise FlakyError("transient")
            return DeclaredRetry.Cmd.Done()


class InheritedRetry(Namespace):
    """A child Command inheriting ``raises``/``retry`` through the MRO."""

    class Parent(Command):
        raises: ClassVar = (FlakyError,)
        retry: ClassVar = RetryPolicy(max_attempts=2, base_delay=0.4, jitter=False)

        class Done(DomainEvent):
            pass

        def handle(self) -> InheritedRetry.Parent.Done:
            return InheritedRetry.Parent.Done()

    class Child(Parent):
        def handle(self) -> InheritedRetry.Parent.Done:
            if not _INLINE.tick():
                raise FlakyError("transient")
            return InheritedRetry.Parent.Done()


@pytest.fixture
def inline_attempts():
    """Reset the module-level counter the inline-command scenarios share."""

    def configure(succeed_on: int) -> _Attempts:
        _INLINE.succeed_on = succeed_on
        _INLINE.calls = 0
        return _INLINE

    return configure


@pytest.fixture
def slept(monkeypatch):
    """Record requested backoff delays instead of waiting for them."""
    recorded: list[float] = []

    def record(delay: float) -> None:
        recorded.append(delay)

    async def arecord(delay: float) -> None:
        recorded.append(delay)

    monkeypatch.setattr(_retry_mod, "_sleep", record)
    monkeypatch.setattr(_retry_mod, "_asleep", arecord)
    return recorded


def describe_retry():
    def describe_compile_time():
        def when_retry_is_declared():
            def without_raises():
                def it_rejects():
                    @on(Started, retry=RetryPolicy())
                    def handler(event: Started) -> Ended:
                        return Ended(result="ok")

                    with pytest.raises(TypeError, match=r"retry=.*raises="):
                        EventGraph([handler])

        def when_on_is_absent_from_raises():
            def it_rejects():
                @on(Started, raises=FlakyError, retry=RetryPolicy(on=(ValueError,)))
                def handler(event: Started) -> Ended:
                    raise FlakyError("boom")

                @on(HandlerRaised, exception=FlakyError)
                def catcher(event: HandlerRaised) -> Recovered:
                    return Recovered(reason="caught")

                with pytest.raises(TypeError, match=r"ValueError.*raises="):
                    EventGraph([handler, catcher])

        def when_retry_is_not_a_policy():
            def it_rejects():
                with pytest.raises(TypeError, match=r"RetryPolicy"):

                    @on(Started, raises=FlakyError, retry=3)  # type: ignore[arg-type]
                    def handler(event: Started) -> Ended:
                        raise FlakyError("boom")

    def describe_runtime_sync():
        def it_retries_until_the_handler_succeeds(slept):
            attempts = _Attempts(succeed_on=3)

            @on(
                Started,
                raises=FlakyError,
                retry=RetryPolicy(max_attempts=3, base_delay=0.1, jitter=False),
            )
            def handler(event: Started) -> Ended:
                if not attempts.tick():
                    raise FlakyError("transient")
                return Ended(result="recovered")

            @on(HandlerRaised, exception=FlakyError)
            def catcher(event: HandlerRaised) -> Recovered:
                return Recovered(reason="gave up")

            log = EventGraph([handler, catcher]).invoke(Started(data="go"))
            assert log.latest(Ended) == Ended(result="recovered")
            assert log.latest(HandlerRaised) is None
            assert attempts.calls == 3

        def it_sleeps_the_computed_delays(slept):
            attempts = _Attempts(succeed_on=3)

            @on(
                Started,
                raises=FlakyError,
                retry=RetryPolicy(max_attempts=3, base_delay=0.1, jitter=False),
            )
            def handler(event: Started) -> Ended:
                if not attempts.tick():
                    raise FlakyError("transient")
                return Ended(result="recovered")

            @on(HandlerRaised, exception=FlakyError)
            def catcher(event: HandlerRaised) -> Recovered:
                return Recovered(reason="gave up")

            EventGraph([handler, catcher]).invoke(Started(data="go"))
            assert slept == [0.1, 0.2]

        def it_emits_handler_raised_after_exhaustion(slept):
            attempts = _Attempts(succeed_on=99)

            @on(
                Started,
                raises=FlakyError,
                retry=RetryPolicy(max_attempts=2, base_delay=0.1, jitter=False),
            )
            def handler(event: Started) -> Ended:
                attempts.tick()
                raise FlakyError("always")

            @on(HandlerRaised, exception=FlakyError)
            def catcher(event: HandlerRaised) -> Recovered:
                return Recovered(reason=str(event.exception))

            log = EventGraph([handler, catcher]).invoke(Started(data="go"))
            assert attempts.calls == 2
            assert log.count(HandlerRaised) == 1
            assert log.latest(Recovered) == Recovered(reason="always")

        def it_emits_handler_retried_per_retry(slept):
            attempts = _Attempts(succeed_on=3)

            @on(
                Started,
                raises=FlakyError,
                retry=RetryPolicy(max_attempts=3, base_delay=0.1, jitter=False),
            )
            def handler(event: Started) -> Ended:
                if not attempts.tick():
                    raise FlakyError("transient")
                return Ended(result="recovered")

            @on(HandlerRaised, exception=FlakyError)
            def catcher(event: HandlerRaised) -> Recovered:
                return Recovered(reason="gave up")

            log = EventGraph([handler, catcher]).invoke(Started(data="go"))
            retried = log.filter(HandlerRetried)
            assert [(e.attempt, e.delay_seconds) for e in retried] == [
                (1, 0.1),
                (2, 0.2),
            ]
            assert retried[0].handler == "handler"
            assert isinstance(retried[0].exception, FlakyError)
            assert retried[0].source_event == Started(data="go")

    def describe_runtime_async():
        async def it_retries_until_the_handler_succeeds(slept):
            attempts = _Attempts(succeed_on=2)

            @on(
                Started,
                raises=FlakyError,
                retry=RetryPolicy(max_attempts=4, base_delay=0.5, jitter=False),
            )
            async def handler(event: Started) -> Ended:
                if not attempts.tick():
                    raise FlakyError("transient")
                return Ended(result="async recovered")

            @on(HandlerRaised, exception=FlakyError)
            async def catcher(event: HandlerRaised) -> Recovered:
                return Recovered(reason="gave up")

            log = await EventGraph([handler, catcher]).ainvoke(Started(data="go"))
            assert log.latest(Ended) == Ended(result="async recovered")
            assert slept == [0.5]

    def describe_observe():
        def when_log():
            def it_does_not_emit_handler_retried(slept, caplog):
                attempts = _Attempts(succeed_on=2)

                @on(
                    Started,
                    raises=FlakyError,
                    retry=RetryPolicy(max_attempts=2, jitter=False, observe="log"),
                )
                def handler(event: Started) -> Ended:
                    if not attempts.tick():
                        raise FlakyError("transient")
                    return Ended(result="recovered")

                @on(HandlerRaised, exception=FlakyError)
                def catcher(event: HandlerRaised) -> Recovered:
                    return Recovered(reason="gave up")

                with caplog.at_level(logging.WARNING, logger="langgraph_events"):
                    log = EventGraph([handler, catcher]).invoke(Started(data="go"))
                assert not log.has(HandlerRetried)
                assert "attempt 1" in caplog.text

        def when_silent():
            def it_records_nothing(slept, caplog):
                attempts = _Attempts(succeed_on=2)

                @on(
                    Started,
                    raises=FlakyError,
                    retry=RetryPolicy(max_attempts=2, jitter=False, observe="silent"),
                )
                def handler(event: Started) -> Ended:
                    if not attempts.tick():
                        raise FlakyError("transient")
                    return Ended(result="recovered")

                @on(HandlerRaised, exception=FlakyError)
                def catcher(event: HandlerRaised) -> Recovered:
                    return Recovered(reason="gave up")

                with caplog.at_level(logging.WARNING, logger="langgraph_events"):
                    log = EventGraph([handler, catcher]).invoke(Started(data="go"))
                assert not log.has(HandlerRetried)
                assert caplog.text == ""
                assert slept == [0.1]

    def describe_non_retryable():
        def when_the_exception_is_outside_on():
            def it_raises_immediately(slept):
                attempts = _Attempts(succeed_on=99)

                @on(
                    Started,
                    raises=(FlakyError, ValueError),
                    retry=RetryPolicy(max_attempts=5, jitter=False, on=(FlakyError,)),
                )
                def handler(event: Started) -> Ended:
                    attempts.tick()
                    raise ValueError("permanent")

                @on(HandlerRaised)
                def catcher(event: HandlerRaised) -> Recovered:
                    return Recovered(reason=str(event.exception))

                log = EventGraph([handler, catcher]).invoke(Started(data="go"))
                assert attempts.calls == 1
                assert slept == []
                assert log.latest(Recovered) == Recovered(reason="permanent")

    def describe_no_policy():
        def when_raises_is_declared():
            def without_retry():
                def it_calls_the_handler_once(slept):
                    attempts = _Attempts(succeed_on=99)

                    @on(Started, raises=FlakyError)
                    def handler(event: Started) -> Ended:
                        attempts.tick()
                        raise FlakyError("boom")

                    @on(HandlerRaised, exception=FlakyError)
                    def catcher(event: HandlerRaised) -> Recovered:
                        return Recovered(reason="caught")

                    log = EventGraph([handler, catcher]).invoke(Started(data="go"))
                    assert attempts.calls == 1
                    assert slept == []
                    assert log.has(HandlerRaised)

    def describe_class_attribute():
        def when_declared_on_a_command():
            def it_applies_the_policy(slept, inline_attempts):
                attempts = inline_attempts(succeed_on=3)

                @on(HandlerRaised, exception=FlakyError)
                def catcher(event: HandlerRaised) -> Recovered:
                    return Recovered(reason="gave up")

                log = EventGraph([DeclaredRetry.Cmd, catcher]).invoke(
                    DeclaredRetry.Cmd()
                )
                assert log.has(DeclaredRetry.Cmd.Done)
                assert attempts.calls == 3
                assert slept == [0.1, 0.2]

        def when_inherited_from_a_parent_command():
            def it_applies_the_inherited_policy(slept, inline_attempts):
                inline_attempts(succeed_on=2)

                @on(HandlerRaised, exception=FlakyError)
                def catcher(event: HandlerRaised) -> Recovered:
                    return Recovered(reason="gave up")

                log = EventGraph([InheritedRetry.Child, catcher]).invoke(
                    InheritedRetry.Child()
                )
                assert log.has(InheritedRetry.Parent.Done)
                assert slept == [0.4]

        def when_declared_as_a_dataclass_field():
            def it_rejects_at_class_creation():
                # An annotated (non-ClassVar) ``retry`` would become a frozen
                # dataclass field, serialising the policy into every
                # checkpoint payload while retry kept working — the same
                # hazard ``raises``/``previously`` are guarded against.
                with pytest.raises(TypeError, match=r"'retry'.*ClassVar"):

                    class _BadRetry(Namespace):
                        class Cmd(Command):
                            retry: RetryPolicy = RetryPolicy()

                            def handle(self) -> None:
                                return None


def describe_namespace_model():
    def describe_command_handler():
        def it_carries_the_declared_policy():
            @on(HandlerRaised, exception=FlakyError)
            def catcher(event: HandlerRaised) -> Recovered:
                return Recovered(reason="gave up")

            model = EventGraph([DeclaredRetry.Cmd, catcher]).namespaces()
            (handler,) = [
                ch for ch in model.command_handlers if DeclaredRetry.Cmd in ch.commands
            ]
            assert handler.retry == DeclaredRetry.Cmd.retry

    def describe_policy():
        def it_carries_the_declared_policy():
            @on(
                Started,
                raises=FlakyError,
                retry=RetryPolicy(max_attempts=2),
            )
            def handler(event: Started) -> Ended:
                raise FlakyError("boom")

            @on(HandlerRaised, exception=FlakyError)
            def catcher(event: HandlerRaised) -> Recovered:
                return Recovered(reason="gave up")

            model = EventGraph([handler, catcher]).namespaces()
            (policy,) = [p for p in model.policies if p.name == "handler"]
            assert policy.retry == RetryPolicy(max_attempts=2)

    def describe_text_render():
        def it_annotates_the_command():
            @on(HandlerRaised, exception=FlakyError)
            def catcher(event: HandlerRaised) -> Recovered:
                return Recovered(reason="gave up")

            text = EventGraph([DeclaredRetry.Cmd, catcher]).namespaces().text()
            assert "retry x3" in text

    def describe_json_render():
        def it_encodes_the_policy():
            @on(HandlerRaised, exception=FlakyError)
            def catcher(event: HandlerRaised) -> Recovered:
                return Recovered(reason="gave up")

            payload = EventGraph([DeclaredRetry.Cmd, catcher]).namespaces().to_dict()
            (encoded,) = [
                r
                for r in payload["command_handlers"]
                if r["commands"] == ["DeclaredRetry.Cmd"]
            ]
            assert encoded["retry"] == {
                "max_attempts": 3,
                "base_delay": 0.1,
                "max_delay": 30.0,
                "strategy": "exponential",
                "jitter": False,
                "on": [],
                "respect_retry_after": False,
                "observe": "emit",
            }
