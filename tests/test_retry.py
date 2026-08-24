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
    Interrupted,
    Invariant,
    InvariantViolated,
    MaxRoundsExceeded,
    Namespace,
    Resumed,
    RetryPolicy,
    Scatter,
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

            def it_clamps_a_negative_hint_to_zero():
                # A skewed clock or a past Retry-After date can produce a
                # negative delta. time.sleep() rejects it with a ValueError
                # raised outside the ``except meta.raises`` boundary, which
                # would abort the whole run instead of surfacing as
                # HandlerRaised; asyncio.sleep() silently returns. Clamp so
                # both dispatch paths agree.
                policy = RetryPolicy(respect_retry_after=True)
                assert policy.delay_for(1, RateLimitedError(retry_after=-5.0)) == 0.0

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


class QuietRetry(Namespace):
    """A Command whose retry policy keeps ``HandlerRetried`` out of the log."""

    class Cmd(Command):
        raises: ClassVar = (FlakyError,)
        retry: ClassVar = RetryPolicy(
            max_attempts=3, base_delay=0.1, jitter=False, observe="log"
        )

        class Done(DomainEvent):
            pass

        def handle(self) -> QuietRetry.Cmd.Done:
            return QuietRetry.Cmd.Done()


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

    class Overriding(Parent):
        """Replaces the policy it would otherwise inherit from ``Parent``."""

        retry: ClassVar = RetryPolicy(max_attempts=2, base_delay=0.05, jitter=False)

        def handle(self) -> InheritedRetry.Parent.Done:
            if not _INLINE.tick():
                raise FlakyError("transient")
            return InheritedRetry.Parent.Done()


@pytest.fixture
def inline_attempts():
    """Reset the counter the module-level inline-command scenarios share.

    Inline ``Command`` handlers must be declared at module level for their
    forward-referenced return annotations to resolve, so they cannot close
    over a per-test counter — they share ``_INLINE``, which this resets.
    """
    _INLINE.succeed_on = 1
    _INLINE.calls = 0
    return _INLINE


class Item(IntegrationEvent):
    """One unit of fan-out work."""

    n: int = 0


class FanOut(IntegrationEvent):
    """Seed that scatters into several ``Item``s."""


class Handled(IntegrationEvent):
    """Successful outcome of processing an ``Item``."""

    n: int = 0


class MustHold(Invariant):
    """Predicate evaluated around dispatch, never per retry attempt."""


class NeedsInput(Interrupted):
    """Human-in-the-loop pause returned by a retryable handler."""

    question: str = ""


@on(HandlerRaised, exception=FlakyError)
def gave_up(event: HandlerRaised) -> Recovered:
    """Escalation catcher — satisfies raises= coverage.

    Most retry tests need a catcher only so the graph builds; they assert on
    the retry behaviour, not on the escalation. Use :func:`echo_failure` when
    the exception message itself is under test.
    """
    return Recovered(reason="gave up")


@on(HandlerRaised, exception=FlakyError)
def echo_failure(event: HandlerRaised) -> Recovered:
    """Escalation catcher that surfaces the exception that exhausted the budget."""
    return Recovered(reason=str(event.exception))


@on(HandlerRaised)
def catch_any(event: HandlerRaised) -> Recovered:
    """Universal catcher — for graphs declaring more than one raise."""
    return Recovered(reason=str(event.exception))


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
                def local_catcher(event: HandlerRaised) -> Recovered:
                    return Recovered(reason="caught")

                with pytest.raises(TypeError, match=r"ValueError.*raises="):
                    EventGraph([handler, local_catcher])

        def when_on_is_a_superclass_of_a_declared_raise():
            def it_accepts():
                # Runtime scope is isinstance(exc, policy.on), so on=(OSError,)
                # genuinely retries a declared ConnectionResetError. The gate
                # must not reject a config that works.
                @on(
                    Started,
                    raises=ConnectionResetError,
                    retry=RetryPolicy(on=(OSError,)),
                )
                def handler(event: Started) -> Ended:
                    raise ConnectionResetError("boom")

                @on(HandlerRaised)
                def local_catcher(event: HandlerRaised) -> Recovered:
                    return Recovered(reason="caught")

                EventGraph([handler, local_catcher])  # must not raise

        def when_an_inline_command_declares_it():
            def it_names_the_command_not_the_method():
                # ``handle`` is useless for locating the offender; the error
                # must say ``_BadInline.Cmd``.
                with pytest.raises(TypeError, match=r"_BadInline\.Cmd.*raises="):

                    class _BadInline(Namespace):
                        class Cmd(Command):
                            retry: ClassVar = RetryPolicy()

                            def handle(self) -> None:
                                return None

                    EventGraph([_BadInline.Cmd])

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

            log = EventGraph([handler, gave_up]).invoke(Started(data="go"))
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

            EventGraph([handler, gave_up]).invoke(Started(data="go"))
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

            log = EventGraph([handler, echo_failure]).invoke(Started(data="go"))
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

            log = EventGraph([handler, gave_up]).invoke(Started(data="go"))
            retried = log.filter(HandlerRetried)
            assert [(e.attempt, e.delay_seconds) for e in retried] == [
                (1, 0.1),
                (2, 0.2),
            ]
            assert retried[0].handler == "handler"
            assert isinstance(retried[0].exception, FlakyError)
            assert retried[0].source_event == Started(data="go")

    def describe_runtime_async():
        async def it_emits_handler_raised_and_breadcrumbs(slept):
            attempts = _Attempts(succeed_on=99)

            @on(
                Item,
                raises=FlakyError,
                retry=RetryPolicy(max_attempts=3, base_delay=0.25, jitter=False),
            )
            async def handler(event: Item) -> Handled:
                attempts.tick()
                raise FlakyError("always")

            log = await EventGraph([handler, echo_failure]).ainvoke(Item(n=1))
            assert attempts.calls == 3
            assert slept == [0.25, 0.5]
            assert [e.attempt for e in log.filter(HandlerRetried)] == [1, 2]
            assert log.count(HandlerRaised) == 1
            assert log.latest(Recovered) == Recovered(reason="always")

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

            log = await EventGraph([handler, gave_up]).ainvoke(Started(data="go"))
            assert log.latest(Ended) == Ended(result="async recovered")
            assert slept == [0.5]

    def describe_observe():
        @pytest.mark.parametrize(
            ("mode", "expect_log"), [("log", True), ("silent", False)]
        )
        def it_suppresses_the_event(slept, caplog, mode, expect_log):
            attempts = _Attempts(succeed_on=2)

            @on(
                Started,
                raises=FlakyError,
                retry=RetryPolicy(max_attempts=2, jitter=False, observe=mode),
            )
            def handler(event: Started) -> Ended:
                if not attempts.tick():
                    raise FlakyError("transient")
                return Ended(result="recovered")

            with caplog.at_level(logging.WARNING, logger="langgraph_events"):
                log = EventGraph([handler, gave_up]).invoke(Started(data="go"))
            assert not log.has(HandlerRetried)
            assert slept == [0.1]
            assert ("attempt 1" in caplog.text) is expect_log

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

                log = EventGraph([handler, catch_any]).invoke(Started(data="go"))
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
                    def local_catcher(event: HandlerRaised) -> Recovered:
                        return Recovered(reason="caught")

                    log = EventGraph([handler, local_catcher]).invoke(
                        Started(data="go")
                    )
                    assert attempts.calls == 1
                    assert slept == []
                    assert log.has(HandlerRaised)

    def describe_class_attribute():
        def when_declared_on_a_command():
            def it_applies_the_policy(slept, inline_attempts):
                inline_attempts.succeed_on = 3
                attempts = inline_attempts

                log = EventGraph([DeclaredRetry.Cmd, gave_up]).invoke(
                    DeclaredRetry.Cmd()
                )
                assert log.has(DeclaredRetry.Cmd.Done)
                assert attempts.calls == 3
                assert slept == [0.1, 0.2]

        def when_inherited_from_a_parent_command():
            def it_applies_the_inherited_policy(slept, inline_attempts):
                inline_attempts.succeed_on = 2

                log = EventGraph([InheritedRetry.Child, gave_up]).invoke(
                    InheritedRetry.Child()
                )
                assert log.has(InheritedRetry.Parent.Done)
                assert slept == [0.4]

        def when_a_child_overrides_the_inherited_policy():
            def it_uses_the_child_policy(slept, inline_attempts):
                inline_attempts.succeed_on = 2

                log = EventGraph([InheritedRetry.Overriding, gave_up]).invoke(
                    InheritedRetry.Overriding()
                )
                assert log.has(InheritedRetry.Parent.Done)
                # Parent declares base_delay=0.4; the child's own policy wins.
                assert slept == [0.05]

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


def describe_retry_scope():
    def describe_attempt_budget():
        def when_several_events_match_one_handler():
            def it_gives_each_event_its_own_budget(slept):
                # The attempt loop is nested inside the per-event loop, so a
                # budget spent on Item(0) must not shorten Item(1)'s.
                seen: list[int] = []

                @on(FanOut)
                def split(event: FanOut) -> Scatter[Item]:
                    return Scatter([Item(n=i) for i in range(3)])

                @on(
                    Item,
                    raises=FlakyError,
                    retry=RetryPolicy(max_attempts=2, base_delay=0.1, jitter=False),
                )
                def work(event: Item) -> Handled:
                    seen.append(event.n)
                    if seen.count(event.n) < 2:
                        raise FlakyError(f"transient {event.n}")
                    return Handled(n=event.n)

                log = EventGraph([split, work, gave_up]).invoke(FanOut())
                assert sorted(e.n for e in log.filter(Handled)) == [0, 1, 2]
                assert not log.has(HandlerRaised)
                assert len(slept) == 3

        def when_max_attempts_is_one():
            def it_never_retries(slept):
                attempts = _Attempts(succeed_on=99)

                @on(Item, raises=FlakyError, retry=RetryPolicy(max_attempts=1))
                def handler(event: Item) -> Handled:
                    attempts.tick()
                    raise FlakyError("always")

                log = EventGraph([handler, gave_up]).invoke(Item(n=1))
                assert attempts.calls == 1
                assert slept == []
                assert not log.has(HandlerRetried)
                assert log.count(HandlerRaised) == 1

    def describe_dispatch_rounds():
        def it_does_not_spend_the_round_budget(slept):
            # Retries happen inside the handler node, so six attempts fit
            # inside a two-round budget without tripping MaxRoundsExceeded.
            attempts = _Attempts(succeed_on=6)

            @on(
                Item,
                raises=FlakyError,
                retry=RetryPolicy(max_attempts=6, base_delay=0.1, jitter=False),
            )
            def handler(event: Item) -> Handled:
                if not attempts.tick():
                    raise FlakyError("transient")
                return Handled(n=99)

            graph = EventGraph([handler, gave_up], max_rounds=2)
            log = graph.invoke(Item(n=1))
            assert attempts.calls == 6
            assert log.latest(Handled) == Handled(n=99)
            assert not log.has(MaxRoundsExceeded)

    def describe_invariants():
        def it_evaluates_the_predicate_once_per_dispatch(slept):
            # ``_check_invariants`` sits outside the attempt loop: the pre and
            # post checks run once each, not once per retry.
            attempts = _Attempts(succeed_on=3)
            evaluations: list[int] = []

            @on(
                Item,
                raises=FlakyError,
                invariants={MustHold: lambda log: not evaluations.append(1)},
                retry=RetryPolicy(max_attempts=3, base_delay=0.1, jitter=False),
            )
            def handler(event: Item) -> Handled:
                if not attempts.tick():
                    raise FlakyError("transient")
                return Handled(n=1)

            @on(InvariantViolated)
            def violated(event: InvariantViolated) -> None:
                return None

            log = EventGraph([handler, gave_up, violated]).invoke(Item(n=1))
            assert attempts.calls == 3
            assert len(evaluations) == 2
            assert log.latest(Handled) == Handled(n=1)

    def describe_interrupts():
        def it_does_not_retry_a_pause(slept):
            # ``_collect_and_check`` runs outside the try/except, so the
            # GraphInterrupt it raises is never mistaken for a failed attempt.
            from langgraph.checkpoint.memory import MemorySaver

            attempts = _Attempts(succeed_on=99)

            @on(
                Item,
                raises=FlakyError,
                retry=RetryPolicy(max_attempts=3, base_delay=0.1, jitter=False),
            )
            def pauser(event: Item) -> NeedsInput:
                attempts.tick()
                return NeedsInput(question="ok?")

            @on(Resumed, interrupted=NeedsInput)
            def after(event: Resumed) -> None:
                return None

            graph = EventGraph([pauser, gave_up, after], checkpointer=MemorySaver())
            config = {"configurable": {"thread_id": "retry-interrupt"}}
            graph.invoke(Item(n=1), config=config)
            assert graph.get_state(config).is_interrupted
            assert attempts.calls == 1
            assert slept == []

    def describe_async_handler_on_the_sync_path():
        def it_retries_across_the_asyncio_run_boundary(slept):
            # An async handler reached through ``invoke()`` is driven by
            # ``asyncio.run`` per attempt — a distinct path from ``ainvoke``.
            attempts = _Attempts(succeed_on=3)

            @on(
                Item,
                raises=FlakyError,
                retry=RetryPolicy(max_attempts=3, base_delay=0.2, jitter=False),
            )
            async def handler(event: Item) -> Handled:
                if not attempts.tick():
                    raise FlakyError("transient")
                return Handled(n=5)

            log = EventGraph([handler, gave_up]).invoke(Item(n=1))
            assert attempts.calls == 3
            assert log.latest(Handled) == Handled(n=5)
            assert slept == [0.2, 0.4]

    def describe_retry_after():
        def it_sleeps_the_exception_hint(slept):
            attempts = _Attempts(succeed_on=2)

            @on(
                Item,
                raises=RateLimitedError,
                retry=RetryPolicy(
                    max_attempts=2, base_delay=0.1, respect_retry_after=True
                ),
            )
            def handler(event: Item) -> Handled:
                if not attempts.tick():
                    raise RateLimitedError(retry_after=1.5)
                return Handled(n=1)

            @on(HandlerRaised, exception=RateLimitedError)
            def local_catcher(event: HandlerRaised) -> Recovered:
                return Recovered(reason="gave up")

            log = EventGraph([handler, local_catcher]).invoke(Item(n=1))
            assert slept == [1.5]
            assert log.latest(HandlerRetried).delay_seconds == 1.5


def describe_handler_retried():
    def describe_subscription():
        def it_matches_on_the_exception_field(slept):
            attempts = _Attempts(succeed_on=3)
            noticed: list[int] = []

            @on(
                Item,
                raises=FlakyError,
                retry=RetryPolicy(max_attempts=3, base_delay=0.1, jitter=False),
            )
            def handler(event: Item) -> Handled:
                if not attempts.tick():
                    raise FlakyError("transient")
                return Handled(n=1)

            @on(HandlerRetried, exception=FlakyError)
            def watch(event: HandlerRetried, exception: FlakyError) -> Recovered:
                noticed.append(event.attempt)
                return Recovered(reason=str(exception))

            EventGraph([handler, watch, gave_up]).invoke(Item(n=1))
            assert noticed == [1, 2]


def describe_namespace_model():
    def describe_command_handler():
        def it_carries_the_declared_policy():

            model = EventGraph([DeclaredRetry.Cmd, gave_up]).namespaces()
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

            model = EventGraph([handler, gave_up]).namespaces()
            (policy,) = [p for p in model.policies if p.name == "handler"]
            assert policy.retry == RetryPolicy(max_attempts=2)

    def describe_seeds():
        def it_does_not_classify_handler_retried_as_a_seed():
            # HandlerRetried is framework-written, like HandlerRaised. Left
            # out of _FRAMEWORK_EMITTED it is misread as a run seed: the
            # structure diagram draws an entry arrow into it and the
            # reflection surface treats leading occurrences as seeds.
            @on(HandlerRetried)
            def note(event: HandlerRetried) -> Recovered:
                return Recovered(reason="retried")

            model = EventGraph([DeclaredRetry.Cmd, note, gave_up]).namespaces()
            assert HandlerRetried not in model.seeds

    def describe_edges():
        def when_the_policy_emits():
            # The model owns the edge so text/json/mermaid all see it —
            # without it, a reactor on HandlerRetried renders as a source
            # node with no producer (#132).
            def it_emits_a_retry_edge_to_handler_retried():

                model = EventGraph([DeclaredRetry.Cmd, gave_up]).namespaces()
                retry_edges = [e for e in model.edges if e.kind == "retry"]
                assert [(e.source, e.target, e.via) for e in retry_edges] == [
                    (DeclaredRetry.Cmd, HandlerRetried, "handle")
                ]

        def when_the_policy_does_not_emit():
            # observe="log"/"silent" never writes HandlerRetried to the log,
            # so an edge into it would be a lie.
            def it_emits_no_retry_edge():

                model = EventGraph([QuietRetry.Cmd, gave_up]).namespaces()
                assert [e for e in model.edges if e.kind == "retry"] == []

    def describe_mermaid_render():
        def when_the_policy_emits():
            def it_draws_the_retry_edge_into_handler_retried():

                out = EventGraph([DeclaredRetry.Cmd, gave_up]).namespaces().mermaid()
                assert 'Cmd -.->|"(retry)"| HandlerRetried' in out

            def it_styles_the_retry_edge_apart_from_raises():

                out = EventGraph([DeclaredRetry.Cmd, gave_up]).namespaces().mermaid()
                assert "stroke:#0891b2" in out

        def when_the_policy_does_not_emit():
            def it_draws_no_handler_retried_node():

                out = EventGraph([QuietRetry.Cmd, gave_up]).namespaces().mermaid()
                assert "HandlerRetried" not in out

    def describe_text_render():
        def it_annotates_the_command():

            text = EventGraph([DeclaredRetry.Cmd, gave_up]).namespaces().text()
            assert "retry x3" in text

    def describe_json_render():
        def it_encodes_the_policy():

            payload = EventGraph([DeclaredRetry.Cmd, gave_up]).namespaces().to_dict()
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
