"""Tests for ``ContinueRun``: a seed that continues the work of a paused run."""

from __future__ import annotations

import asyncio
import threading
from collections import Counter

from conftest import strip_channels
from langgraph.checkpoint.memory import MemorySaver

from langgraph_events import (
    ContinueRun,
    EventGraph,
    Halted,
    HandlerAbandonedOnSignal,
    HandlerRaised,
    IntegrationEvent,
    Interrupted,
    MaxRoundsExceeded,
    Reducer,
    RetryPolicy,
    RunPaused,
    RunPausedOnDeadline,
    RunPausedOnSignal,
    Scatter,
    on,
)


class Begun(IntegrationEvent):
    pass


class Stepped(IntegrationEvent):
    n: int = 0


class Finished(IntegrationEvent):
    pass


class Asked(IntegrationEvent):
    text: str = ""


class Answered(IntegrationEvent):
    text: str = ""


class FannedOut(IntegrationEvent):
    pass


class Worked(IntegrationEvent):
    n: int = 0


class WorkDone(IntegrationEvent):
    n: int = 0


class PauseNoted(IntegrationEvent):
    pass


class Approval(Interrupted):
    pass


class Approved(IntegrationEvent):
    pass


class Flaked(IntegrationEvent):
    pass


class FlakyError(Exception):
    pass


class Stopped(Halted):
    pass


LAST_STEP = 3


def _config(thread_id: str) -> dict[str, dict[str, str]]:
    return {"configurable": {"thread_id": thread_id}}


def _chain(calls: Counter[str], signal: threading.Event | None = None) -> list[object]:
    """A chain ``Begun -> Stepped(1) -> ... -> Stepped(3) -> Finished``.

    Each handler call counts in *calls*. The handler of ``Stepped(1)`` sets
    *signal*, so a ``pause_on=`` run pauses after it.
    """

    @on(Begun)
    def begin(event: Begun) -> Stepped:
        calls["begin"] += 1
        return Stepped(n=1)

    @on(Stepped)
    def step(event: Stepped) -> Stepped | Finished:
        calls[f"step{event.n}"] += 1
        if signal is not None and event.n == 1:
            signal.set()
        return Finished() if event.n == LAST_STEP else Stepped(n=event.n + 1)

    @on(Asked)
    def answer(event: Asked) -> Answered:
        calls["answer"] += 1
        return Answered(text=event.text)

    return [begin, step, answer]


def _chain_graph(
    calls: Counter[str], signal: threading.Event | None = None, **kwargs: object
) -> EventGraph:
    return EventGraph(_chain(calls, signal), checkpointer=MemorySaver(), **kwargs)  # type: ignore[arg-type]


def _ran_once(calls: Counter[str]) -> bool:
    return calls == Counter(begin=1, step1=1, step2=1, step3=1)


def describe_ContinueRun():
    def when_a_deadline_pauses_a_chain():
        def it_finishes_the_chain_once():
            calls: Counter[str] = Counter()
            graph = _chain_graph(calls)
            config = _config("deadline")
            paused = graph.invoke(Begun(), config=config, deadline=0.0)
            assert paused.has(RunPausedOnDeadline)
            assert not paused.has(Finished)

            log = graph.invoke(ContinueRun(), config=config)

            assert log.count(Finished) == 1
            assert _ran_once(calls)

        def when_the_continued_run_pauses_again():
            def it_continues_from_the_second_pause():
                calls: Counter[str] = Counter()
                graph = _chain_graph(calls)
                config = _config("deadline-twice")
                graph.invoke(Begun(), config=config, deadline=0.0)
                again = graph.invoke(ContinueRun(), config=config, deadline=0.0)
                assert again.count(RunPaused) == 2
                assert not again.has(Finished)

                log = graph.invoke(ContinueRun(), config=config)

                assert log.count(Finished) == 1
                assert _ran_once(calls)

    def when_a_signal_pauses_a_chain():
        async def it_finishes_the_chain_once():
            calls: Counter[str] = Counter()
            signal = threading.Event()
            graph = _chain_graph(calls, signal)
            config = _config("signal")
            paused = await graph.ainvoke(Begun(), config=config, pause_on=signal)
            assert paused.has(RunPausedOnSignal)
            assert not paused.has(Finished)

            log = await graph.ainvoke(ContinueRun(), config=config)

            assert log.count(Finished) == 1
            assert _ran_once(calls)

    def when_a_fan_out_round_is_stranded():
        async def it_dispatches_each_branch_once():
            done: list[int] = []
            signal = asyncio.Event()

            @on(FannedOut)
            def split(event: FannedOut) -> Scatter[Worked]:
                return Scatter([Worked(n=n) for n in range(3)])

            @on(Worked)
            async def work(event: Worked) -> WorkDone:
                if event.n == 0:
                    signal.set()
                return WorkDone(n=event.n)

            @on(WorkDone)
            def collect(event: WorkDone) -> None:
                done.append(event.n)

            graph = EventGraph([split, work, collect], checkpointer=MemorySaver())
            config = _config("fan-out")
            paused = await graph.ainvoke(FannedOut(), config=config, pause_on=signal)
            assert paused.count(WorkDone) == 3
            assert done == []

            await graph.ainvoke(ContinueRun(), config=config)

            assert sorted(done) == [0, 1, 2]

    def when_a_pause_handler_emits_an_event():
        def it_dispatches_that_event_once():
            noted: list[PauseNoted] = []

            @on(RunPaused)
            def note(event: RunPaused) -> PauseNoted:
                return PauseNoted()

            @on(PauseNoted)
            def record(event: PauseNoted) -> None:
                noted.append(event)

            calls: Counter[str] = Counter()
            graph = EventGraph(
                [*_chain(calls), note, record], checkpointer=MemorySaver()
            )
            config = _config("pause-handler")
            graph.invoke(Begun(), config=config, deadline=0.0)
            assert noted == []

            graph.invoke(ContinueRun(), config=config)

            assert len(noted) == 1
            assert _ran_once(calls)

    def when_a_retry_gives_up_on_the_signal():
        def it_does_not_dispatch_the_give_up():
            caught: list[HandlerRaised] = []

            @on(
                Flaked,
                raises=FlakyError,
                retry=RetryPolicy(base_delay=60.0, max_delay=60.0),
            )
            def flaky(event: Flaked) -> Finished:
                raise FlakyError

            @on(HandlerRaised)
            def catch(event: HandlerRaised) -> None:
                caught.append(event)

            graph = EventGraph([flaky, catch], checkpointer=MemorySaver())
            config = _config("give-up")
            signal = threading.Event()
            signal.set()
            paused = graph.invoke(Flaked(), config=config, pause_on=signal)
            assert paused.has(HandlerAbandonedOnSignal)

            graph.invoke(ContinueRun(), config=config)

            assert caught == []

    def when_the_stopped_work_holds_a_halt():
        def it_ends_the_continued_run():
            calls: Counter[str] = Counter()

            @on(Begun)
            def stop(event: Begun) -> Stopped:
                return Stopped()

            graph = EventGraph([*_chain(calls), stop], checkpointer=MemorySaver())
            config = _config("halt")
            graph.invoke(Begun(), config=config, deadline=0.0)

            graph.invoke(ContinueRun(), config=config)

            assert calls == Counter(begin=1)

    def when_a_plain_run_follows_the_pause():
        def it_does_not_run_the_stranded_work():
            calls: Counter[str] = Counter()
            graph = _chain_graph(calls)
            config = _config("plain")
            graph.invoke(Begun(), config=config, deadline=0.0)

            log = graph.invoke(Asked(text="hi"), config=config)

            assert log.count(Answered) == 1
            assert calls == Counter(begin=1, answer=1)

        def it_discards_the_stranded_work():
            calls: Counter[str] = Counter()
            graph = _chain_graph(calls)
            config = _config("plain-then-continue")
            graph.invoke(Begun(), config=config, deadline=0.0)
            graph.invoke(Asked(text="hi"), config=config)

            log = graph.invoke(ContinueRun(), config=config)

            assert not log.has(Finished)
            assert calls == Counter(begin=1, answer=1)

    def with_other_seeds():
        def it_dispatches_the_stranded_work_and_the_seeds():
            calls: Counter[str] = Counter()
            graph = _chain_graph(calls)
            config = _config("with-seeds")
            graph.invoke(Begun(), config=config, deadline=0.0)

            log = graph.invoke([ContinueRun(), Asked(text="hi")], config=config)

            assert log.count(Finished) == 1
            assert log.count(Answered) == 1
            assert calls == Counter(begin=1, step1=1, step2=1, step3=1, answer=1)

    def when_ContinueRun_comes_twice():
        def it_continues_only_once():
            calls: Counter[str] = Counter()
            graph = _chain_graph(calls)
            config = _config("twice")
            graph.invoke(Begun(), config=config, deadline=0.0)
            graph.invoke(ContinueRun(), config=config)

            log = graph.invoke(ContinueRun(), config=config)

            assert log.count(Finished) == 1
            assert _ran_once(calls)

    def when_nothing_paused():
        def it_runs_nothing_else():
            calls: Counter[str] = Counter()
            graph = _chain_graph(calls)

            log = graph.invoke(ContinueRun(), config=_config("fresh"))

            assert list(log) == [ContinueRun()]
            assert calls == Counter()

    def when_the_checkpoint_has_no_stranded_record():
        def it_runs_nothing_else():
            saver = MemorySaver()
            calls: Counter[str] = Counter()
            graph = EventGraph(_chain(calls), checkpointer=saver)  # type: ignore[arg-type]
            config = _config("old-checkpoint")
            graph.invoke(Begun(), config=config, deadline=0.0)
            strip_channels(saver, config, "_stranded")

            log = graph.invoke(ContinueRun(), config=config)

            assert log.count(ContinueRun) == 1
            assert calls == Counter(begin=1)

    def when_the_continued_run_exceeds_max_rounds():
        def it_does_not_run_the_stranded_work_again():
            calls: Counter[str] = Counter()
            graph = _chain_graph(calls, max_rounds=1)
            config = _config("max-rounds")
            graph.invoke(Begun(), config=config, deadline=0.0)
            halted = graph.invoke(ContinueRun(), config=config)
            assert halted.has(MaxRoundsExceeded)

            graph.invoke(ContinueRun(), config=config)

            assert calls == Counter(begin=1, step1=1, step2=1)

    def when_a_continued_handler_interrupts():
        def it_continues_only_once_after_the_resume():
            calls: Counter[str] = Counter()

            @on(Begun)
            def begin(event: Begun) -> Stepped:
                calls["begin"] += 1
                return Stepped(n=1)

            @on(Stepped)
            def ask(event: Stepped) -> Approval:
                calls["ask"] += 1
                return Approval()

            @on(Approved)
            def finish(event: Approved) -> Finished:
                calls["finish"] += 1
                return Finished()

            graph = EventGraph([begin, ask, finish], checkpointer=MemorySaver())
            config = _config("interrupt")
            graph.invoke(Begun(), config=config, deadline=0.0)
            graph.invoke(ContinueRun(), config=config)
            resumed = graph.resume(Approved(), config=config)
            assert resumed.count(Finished) == 1
            after_resume = Counter(calls)

            log = graph.invoke(ContinueRun(), config=config)

            assert log.count(Approval) == 1
            assert calls == after_resume

        def when_the_interrupt_is_abandoned():
            def it_continues_nothing():
                calls: Counter[str] = Counter()

                @on(Begun)
                def begin(event: Begun) -> Stepped:
                    return Stepped(n=1)

                @on(Stepped)
                def ask(event: Stepped) -> Approval:
                    calls["ask"] += 1
                    return Approval()

                graph = EventGraph([begin, ask], checkpointer=MemorySaver())
                config = _config("abandon")
                graph.invoke(Begun(), config=config, deadline=0.0)
                graph.invoke(ContinueRun(), config=config)
                graph.abandon(config, reason="stale")

                graph.invoke(ContinueRun(), config=config)

                assert calls == Counter(ask=1)

    def when_a_reducer_projects_the_stranded_events():
        def it_projects_each_event_once():
            seen: list[list[int]] = []
            steps = Reducer("steps", event_type=Stepped, fn=lambda e: [e.n])

            @on(Begun)
            def begin(event: Begun) -> Stepped:
                return Stepped(n=1)

            @on(Stepped)
            def step(event: Stepped, steps: list[int]) -> Stepped | Finished:
                seen.append(list(steps))
                return Finished() if event.n == 2 else Stepped(n=event.n + 1)

            graph = EventGraph(
                [begin, step], reducers=[steps], checkpointer=MemorySaver()
            )
            config = _config("reducer")
            graph.invoke(Begun(), config=config, deadline=0.0)

            graph.invoke(ContinueRun(), config=config)

            assert seen == [[1], [1, 2]]


def describe_the_documented_loop():
    def it_continues_until_the_chain_finishes():
        calls: Counter[str] = Counter()
        graph = _chain_graph(calls)
        config = _config("loop")

        log = graph.invoke(Begun(), config=config, deadline=0.0)
        while log.latest(RunPaused, not_followed_by=ContinueRun):
            log = graph.invoke(ContinueRun(), config=config, deadline=0.0)

        assert log.count(Finished) == 1
        assert _ran_once(calls)
