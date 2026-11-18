import asyncio
import contextlib
import logging
import typing

import pytest
from conftest import (
    ApprovalRequested,
    Ended,
    MessageReceived,
    MessageSent,
    Processed,
    ReviewApproved,
    Started,
    to_processed,
)
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt as lg_interrupt

from langgraph_events import (
    Cancelled,
    CancelledOnSignal,
    EventGraph,
    HandlerCutOnSignal,
    IntegrationEvent,
    Reducer,
    ResumeStoppedError,
    Scatter,
    on,
)


@on(Processed)
async def _async_to_ended(event: Processed) -> Ended:
    """Async ``Processed`` leg of the ``stop_on=`` scenarios."""
    return Ended(result=event.data)


@on(Started)
async def _ask_draft(event: Started) -> ApprovalRequested:
    return ApprovalRequested(draft=event.data)


@on(ReviewApproved)
async def _finish(event: ReviewApproved) -> Ended:
    return Ended(result="approved")


class Answered(IntegrationEvent):
    note: str = ""


class _Replay:
    """Two questions. While armed, the replay of the first sets the resume stop.

    With *second_finishes*, the replay of the second ends before the stop.
    Without it, the stop also cuts the replay of the second.
    """

    def __init__(self, *, second_finishes: bool) -> None:
        self.second_finishes = second_finishes
        self.armed = False
        self.stop = asyncio.Event()
        self.answers: list[str] = []

    def graph(self) -> EventGraph:
        @on(Started)
        async def ask(event: Started) -> ApprovalRequested:
            if self.armed:
                await asyncio.sleep(0.05)
                self.stop.set()
                await asyncio.sleep(100)
            return ApprovalRequested(draft="first")

        @on(Started)
        async def ask_again(event: Started) -> ApprovalRequested:
            if self.armed and not self.second_finishes:
                await asyncio.sleep(100)
            return ApprovalRequested(draft="second")

        @on(Answered)
        async def record(event: Answered) -> None:
            self.answers.append(event.note)

        return EventGraph([ask, ask_again, record], checkpointer=MemorySaver())


async def _open_ids(graph: EventGraph, config: dict) -> dict[str, str]:
    snapshot = await graph.compiled.aget_state(config)
    return {i.value.draft: i.id for i in snapshot.interrupts}


async def _stored(graph: EventGraph, config: dict) -> list[object]:
    return list((await graph.compiled.aget_state(config)).values["events"])


def _slow_sibling(stop: asyncio.Event) -> object:
    @on(Started)
    async def slow(event: Started) -> Ended:
        await asyncio.sleep(0.05)
        stop.set()
        await asyncio.sleep(100)
        return Ended(result="late")

    return slow


@pytest.fixture
def short_stop_grace(monkeypatch: pytest.MonkeyPatch) -> float:
    """Cut the ``stop_on=`` grace period to 0.1 seconds, and return it."""
    monkeypatch.setattr("langgraph_events._stop._STOP_GRACE_SECONDS", 0.1)
    return 0.1


def describe_stop_on():

    def when_the_signal_is_set_while_a_handler_awaits():

        async def it_returns_a_log_that_ends_on_cancelled_on_signal():
            stop = asyncio.Event()

            @on(Started)
            async def slow(event: Started) -> Ended:
                stop.set()
                await asyncio.sleep(100)
                return Ended(result="done")

            log = await EventGraph([slow]).ainvoke(Started(data="go"), stop_on=stop)

            assert [type(event) for event in log] == [
                Started,
                CancelledOnSignal,
            ]

        async def it_names_the_cut_handler_in_discarded():
            stop = asyncio.Event()

            @on(Started)
            async def slow(event: Started) -> Ended:
                stop.set()
                await asyncio.sleep(100)
                return Ended(result="done")

            log = await EventGraph([slow]).ainvoke(Started(data="go"), stop_on=stop)

            assert log.latest(CancelledOnSignal) == CancelledOnSignal(discarded="slow")

        def when_a_sibling_handler_interrupts_in_the_same_round():

            async def it_pauses_on_the_question_and_resumes_on_the_answer():
                stop = asyncio.Event()

                @on(Started)
                async def ask(event: Started) -> ApprovalRequested:
                    return ApprovalRequested(draft=event.data)

                @on(Started)
                async def slow(event: Started) -> Ended:
                    await asyncio.sleep(0.05)
                    stop.set()
                    await asyncio.sleep(100)
                    return Ended(result="late")

                @on(ReviewApproved)
                async def finish(event: ReviewApproved) -> Ended:
                    return Ended(result="approved")

                graph = EventGraph([ask, slow, finish], checkpointer=MemorySaver())
                config = {"configurable": {"thread_id": "stop-and-ask"}}

                await graph.ainvoke(Started(data="draft"), stop_on=stop, config=config)
                paused = await graph.aget_state(config)
                resumed = await graph.aresume(ReviewApproved(), config=config)

                assert paused.interrupted == ApprovalRequested(draft="draft")
                assert not paused.events.has(Cancelled)
                assert resumed.latest(Ended) == Ended(result="approved")

            async def it_records_the_cut_of_the_sibling_on_resume():
                stop = asyncio.Event()
                graph = EventGraph(
                    [_ask_draft, _slow_sibling(stop), _finish],
                    checkpointer=MemorySaver(),
                )
                config = {"configurable": {"thread_id": "stop-and-ask-record"}}
                await graph.ainvoke(Started(data="draft"), stop_on=stop, config=config)

                resumed = await graph.aresume(ReviewApproved(), config=config)

                assert [e for e in resumed if isinstance(e, HandlerCutOnSignal)] == [
                    HandlerCutOnSignal(handler="slow")
                ]
                assert resumed.latest(Ended) == Ended(result="approved")

            def when_two_questions_in_the_round_take_two_resumes():

                async def it_records_the_cut_of_the_sibling_once():
                    stop = asyncio.Event()

                    @on(Started)
                    async def slow(event: Started) -> Ended | None:
                        if event.data != "a":
                            return None
                        await asyncio.sleep(0.05)
                        stop.set()
                        await asyncio.sleep(100)
                        return Ended(result="late")

                    graph = EventGraph(
                        [_ask_draft, slow, _finish], checkpointer=MemorySaver()
                    )
                    config = {"configurable": {"thread_id": "stop-two-questions"}}
                    await graph.ainvoke(
                        [Started(data="a"), Started(data="b")],
                        stop_on=stop,
                        config=config,
                    )

                    await graph.aresume(ReviewApproved(), config=config)
                    resumed = await graph.aresume(ReviewApproved(), config=config)

                    assert [
                        e for e in resumed if isinstance(e, HandlerCutOnSignal)
                    ] == [HandlerCutOnSignal(handler="slow")]

            def when_a_later_stop_cuts_the_answer_handler():

                async def it_names_only_the_answer_handler_in_discarded():
                    stop = asyncio.Event()
                    later_stop = asyncio.Event()

                    @on(ReviewApproved)
                    async def finish(event: ReviewApproved) -> Ended:
                        later_stop.set()
                        await asyncio.sleep(100)
                        return Ended(result="approved")

                    graph = EventGraph(
                        [_ask_draft, _slow_sibling(stop), finish],
                        checkpointer=MemorySaver(),
                    )
                    config = {"configurable": {"thread_id": "stop-ask-later-stop"}}
                    await graph.ainvoke(
                        Started(data="draft"), stop_on=stop, config=config
                    )

                    async with asyncio.timeout(5):
                        resumed = await graph.aresume(
                            ReviewApproved(), stop_on=later_stop, config=config
                        )

                    assert resumed.latest(CancelledOnSignal) == CancelledOnSignal(
                        discarded="finish"
                    )

            def when_the_resume_signal_is_set_in_the_resume_round():

                async def it_names_no_earlier_handler_in_discarded():
                    stop = asyncio.Event()
                    later_stop = asyncio.Event()

                    def set_later_stop(event: ReviewApproved) -> list[str]:
                        later_stop.set()
                        return []

                    graph = EventGraph(
                        [_ask_draft, _slow_sibling(stop), _finish],
                        checkpointer=MemorySaver(),
                        reducers=[
                            Reducer[ReviewApproved, str](
                                name="answered",
                                event_type=ReviewApproved,
                                fn=set_later_stop,
                            )
                        ],
                    )
                    config = {"configurable": {"thread_id": "stop-ask-resume-stop"}}
                    await graph.ainvoke(
                        Started(data="draft"), stop_on=stop, config=config
                    )

                    resumed = await graph.aresume(
                        ReviewApproved(), stop_on=later_stop, config=config
                    )

                    assert [
                        e
                        for e in resumed
                        if isinstance(e, (HandlerCutOnSignal, CancelledOnSignal))
                    ] == [
                        HandlerCutOnSignal(handler="slow"),
                        CancelledOnSignal(discarded=""),
                    ]

    def when_the_handler_interrupts_after_the_signal_is_set():

        async def it_logs_no_failed_cleanup(caplog):
            stop = asyncio.Event()

            @on(Started)
            async def ask(event: Started) -> ApprovalRequested:
                stop.set()
                return ApprovalRequested(draft=event.data)

            @on(ReviewApproved)
            async def finish(event: ReviewApproved) -> Ended:
                return Ended(result="approved")

            graph = EventGraph([ask, finish], checkpointer=MemorySaver())
            config = {"configurable": {"thread_id": "ask-after-stop"}}

            with caplog.at_level(logging.WARNING, "langgraph_events"):
                log = await graph.ainvoke(
                    Started(data="go"), stop_on=stop, config=config
                )

            assert [
                record.getMessage()
                for record in caplog.records
                if record.name.startswith("langgraph_events")
            ] == []
            assert [type(event) for event in log] == [
                Started,
                CancelledOnSignal,
            ]

    def when_aresume_gets_the_signal():

        async def it_cancels_the_answer_handler():
            stop = asyncio.Event()

            @on(Started)
            async def ask(event: Started) -> ApprovalRequested:
                return ApprovalRequested(draft=event.data)

            @on(ReviewApproved)
            async def finish(event: ReviewApproved) -> Ended:
                stop.set()
                await asyncio.sleep(100)
                return Ended(result="approved")

            graph = EventGraph([ask, finish], checkpointer=MemorySaver())
            config = {"configurable": {"thread_id": "aresume-stop"}}
            await graph.ainvoke(Started(data="go"), config=config)

            async with asyncio.timeout(5):
                log = await graph.aresume(ReviewApproved(), stop_on=stop, config=config)

            assert type(list(log)[-1]) is CancelledOnSignal
            assert not log.has(Ended)

        def when_the_signal_is_already_set():

            async def it_raises_ResumeStoppedError_and_keeps_the_question():
                stop = asyncio.Event()
                stop.set()
                graph = EventGraph([_ask_draft, _finish], checkpointer=MemorySaver())
                config = {"configurable": {"thread_id": "aresume-set-stop"}}
                await graph.ainvoke(Started(data="go"), config=config)

                with pytest.raises(ResumeStoppedError):
                    await graph.aresume(ReviewApproved(), stop_on=stop, config=config)
                state = await graph.aget_state(config)

                assert state.interrupted == ApprovalRequested(draft="go")

    def when_astream_resume_gets_the_signal():

        async def it_yields_cancelled_on_signal_and_ends():
            stop = asyncio.Event()

            @on(Started)
            async def ask(event: Started) -> ApprovalRequested:
                return ApprovalRequested(draft=event.data)

            @on(ReviewApproved)
            async def finish(event: ReviewApproved) -> Ended:
                stop.set()
                await asyncio.sleep(100)
                return Ended(result="approved")

            graph = EventGraph([ask, finish], checkpointer=MemorySaver())
            config = {"configurable": {"thread_id": "astream-resume-stop"}}
            await graph.ainvoke(Started(data="go"), config=config)

            async with asyncio.timeout(5):
                names = [
                    type(event).__name__
                    async for event in graph.astream_resume(
                        ReviewApproved(), stop_on=stop, config=config
                    )
                ]

            assert names[-1] == "CancelledOnSignal"
            assert "Ended" not in names

        def when_the_signal_is_already_set():

            async def it_raises_ResumeStoppedError_and_keeps_the_question():
                stop = asyncio.Event()
                stop.set()
                graph = EventGraph([_ask_draft, _finish], checkpointer=MemorySaver())
                config = {"configurable": {"thread_id": "astream-resume-set-stop"}}
                await graph.ainvoke(Started(data="go"), config=config)

                with pytest.raises(ResumeStoppedError):
                    async for _ in graph.astream_resume(
                        ReviewApproved(), stop_on=stop, config=config
                    ):
                        pass
                state = await graph.aget_state(config)

                assert state.interrupted == ApprovalRequested(draft="go")

    def when_the_handler_await_loses_the_first_cancel():

        async def it_cancels_the_handler_again(short_stop_grace):
            stop = asyncio.Event()
            cancels: list[str] = []

            @on(Started)
            async def stubborn(event: Started) -> Ended:
                stop.set()
                try:
                    await asyncio.sleep(100)
                except asyncio.CancelledError:
                    cancels.append("lost")
                await asyncio.sleep(100)
                return Ended(result="done")

            async with asyncio.timeout(5):
                log = await EventGraph([stubborn]).ainvoke(
                    Started(data="go"), stop_on=stop
                )

            assert cancels == ["lost"]
            assert [type(event) for event in log] == [
                Started,
                CancelledOnSignal,
            ]

    def when_the_signal_stays_clear():

        async def it_runs_every_handler_to_its_end():
            log = await EventGraph([to_processed, _async_to_ended]).ainvoke(
                Started(data="go"), stop_on=asyncio.Event()
            )

            assert log.latest(Ended) == Ended(result="go")
            assert not log.has(Cancelled)

    def when_the_signal_is_set_after_the_last_handler_step():

        async def it_stores_cancelled_on_signal_after_the_handler_events():
            stop = asyncio.Event()

            def set_stop(event: Ended) -> list[str]:
                stop.set()
                return [event.result]

            log = await EventGraph(
                [_async_to_ended],
                reducers=[
                    Reducer[Ended, str](name="seen", event_type=Ended, fn=set_stop)
                ],
            ).ainvoke(Processed(data="go"), stop_on=stop)

            assert [type(event) for event in log] == [
                Processed,
                Ended,
                CancelledOnSignal,
            ]

        async def it_names_no_handler_in_discarded():
            stop = asyncio.Event()

            def set_stop(event: Ended) -> list[str]:
                stop.set()
                return [event.result]

            log = await EventGraph(
                [_async_to_ended],
                reducers=[
                    Reducer[Ended, str](name="seen", event_type=Ended, fn=set_stop)
                ],
            ).ainvoke(Processed(data="go"), stop_on=stop)

            assert log.latest(CancelledOnSignal) == CancelledOnSignal(discarded="")

    def when_the_signal_is_set_on_the_last_allowed_round():

        async def it_stores_cancelled_on_signal():
            stop = asyncio.Event()

            @on(Processed)
            async def slow(event: Processed) -> Ended:
                stop.set()
                await asyncio.sleep(100)
                return Ended(result="done")

            log = await EventGraph([to_processed, slow], max_rounds=1).ainvoke(
                Started(data="go"), stop_on=stop
            )

            assert [type(event) for event in log] == [
                Started,
                Processed,
                CancelledOnSignal,
            ]

    def when_several_handlers_of_one_round_are_discarded():

        async def it_stores_one_cancelled_on_signal():
            stop = asyncio.Event()

            @on(Started)
            async def first(event: Started) -> Ended:
                stop.set()
                await asyncio.sleep(100)
                return Ended(result="first")

            @on(Started)
            async def second(event: Started) -> Processed:
                await asyncio.sleep(100)
                return Processed(data="second")

            log = await EventGraph([first, second]).ainvoke(
                Started(data="go"), stop_on=stop
            )

            assert [type(event) for event in log] == [
                Started,
                CancelledOnSignal,
            ]

        async def it_names_each_cut_handler_in_discarded():
            stop = asyncio.Event()

            @on(Started)
            async def first(event: Started) -> Ended:
                stop.set()
                await asyncio.sleep(100)
                return Ended(result="first")

            @on(Started)
            async def second(event: Started) -> Processed:
                await asyncio.sleep(100)
                return Processed(data="second")

            log = await EventGraph([first, second]).ainvoke(
                Started(data="go"), stop_on=stop
            )

            stopped = log.latest(CancelledOnSignal)
            assert stopped is not None
            assert sorted(stopped.discarded.split(", ")) == ["first", "second"]

    def when_a_new_run_follows_an_abandoned_stopped_round():

        async def it_names_only_its_own_cut_handler():
            first_stop = asyncio.Event()
            second_stop = asyncio.Event()

            @on(Started)
            async def ask(event: Started) -> ApprovalRequested:
                return ApprovalRequested(draft=event.data)

            @on(Started)
            async def slow(event: Started) -> Ended:
                await asyncio.sleep(0.05)
                first_stop.set()
                await asyncio.sleep(100)
                return Ended(result="late")

            @on(MessageReceived)
            async def reply(event: MessageReceived) -> MessageSent:
                second_stop.set()
                await asyncio.sleep(100)
                return MessageSent(text=event.text)

            graph = EventGraph([ask, slow, reply], checkpointer=MemorySaver())
            config = {"configurable": {"thread_id": "abandon-then-stop"}}
            await graph.ainvoke(Started(data="go"), stop_on=first_stop, config=config)
            await graph.event_store.aabandon(config)

            log = await graph.ainvoke(
                MessageReceived(text="again"), stop_on=second_stop, config=config
            )

            assert log.latest(CancelledOnSignal) == CancelledOnSignal(discarded="reply")

    def when_a_scattered_handler_is_cut():

        async def it_names_the_handler_once_in_discarded():
            stop = asyncio.Event()

            @on(Started)
            async def split(event: Started) -> Scatter[Processed]:
                return Scatter([Processed(data="a"), Processed(data="b")])

            @on(Processed)
            async def slow(event: Processed) -> Ended:
                stop.set()
                await asyncio.sleep(100)
                return Ended(result=event.data)

            log = await EventGraph([split, slow]).ainvoke(
                Started(data="go"), stop_on=stop
            )

            assert log.latest(CancelledOnSignal) == CancelledOnSignal(discarded="slow")

    def when_the_run_task_is_cancelled_during_the_stop_grace():

        async def it_stores_only_cancelled(short_stop_grace):
            stop = asyncio.Event()
            cleaning = asyncio.Event()

            @on(Started)
            async def tidy(event: Started) -> Ended:
                stop.set()
                try:
                    await asyncio.sleep(100)
                except asyncio.CancelledError:
                    cleaning.set()
                    await asyncio.sleep(short_stop_grace / 2)
                    raise
                return Ended(result="done")

            task = asyncio.ensure_future(
                EventGraph([tidy]).ainvoke(Started(data="go"), stop_on=stop)
            )
            await cleaning.wait()
            task.cancel()
            log = await task

            assert [type(event) for event in log] == [Started, Cancelled]

    def when_the_signal_is_set_before_the_run():

        async def it_never_runs_a_handler():
            stop = asyncio.Event()
            stop.set()
            ran: list[str] = []

            @on(Started)
            async def first(event: Started) -> Ended:
                ran.append(event.data)
                return Ended(result=event.data)

            log = await EventGraph([first]).ainvoke(Started(data="go"), stop_on=stop)

            assert ran == []
            assert [type(event) for event in log] == [
                Started,
                CancelledOnSignal,
            ]

    def when_the_handler_returns_after_the_signal_is_set():

        async def it_discards_the_handler_events():
            stop = asyncio.Event()

            @on(Started)
            async def first(event: Started) -> Processed:
                stop.set()
                return Processed(data=event.data)

            log = await EventGraph([first]).ainvoke(Started(data="go"), stop_on=stop)

            assert [type(event) for event in log] == [
                Started,
                CancelledOnSignal,
            ]

    def when_a_sync_handler_runs_as_the_signal_is_set():

        async def it_discards_the_handler_events():
            stop = asyncio.Event()

            @on(Started)
            def first(event: Started) -> Processed:
                stop.set()
                return Processed(data=event.data)

            log = await EventGraph([first]).ainvoke(Started(data="go"), stop_on=stop)

            assert [type(event) for event in log] == [
                Started,
                CancelledOnSignal,
            ]

    def when_a_new_run_starts_on_the_stopped_thread():

        async def it_runs_the_new_seed():
            stop = asyncio.Event()

            @on(Started)
            async def slow(event: Started) -> Ended:
                stop.set()
                await asyncio.sleep(100)
                return Ended(result="done")

            @on(MessageReceived)
            async def reply(event: MessageReceived) -> MessageSent:
                return MessageSent(text=event.text)

            graph = EventGraph([slow, reply], checkpointer=MemorySaver())
            config = {"configurable": {"thread_id": "after-stop"}}
            await graph.ainvoke(Started(data="go"), stop_on=stop, config=config)

            log = await graph.ainvoke(MessageReceived(text="again"), config=config)

            assert log.latest(MessageSent) == MessageSent(text="again")

    def when_the_run_task_is_cancelled():

        async def it_cancels_the_handler_and_ends_on_cancelled():
            ready = asyncio.Event()
            unwound: list[str] = []

            @on(Started)
            async def slow(event: Started) -> Ended:
                ready.set()
                try:
                    await asyncio.sleep(100)
                finally:
                    unwound.append(event.data)
                return Ended(result="done")

            task = asyncio.ensure_future(
                EventGraph([slow]).ainvoke(Started(data="go"), stop_on=asyncio.Event())
            )
            await ready.wait()
            task.cancel()
            log = await task

            assert unwound == ["go"]
            assert [type(event) for event in log] == [Started, Cancelled]

        def when_the_handler_await_loses_the_cancel():

            async def it_cancels_the_handler_again_after_the_grace(
                short_stop_grace,
            ):
                ready = asyncio.Event()
                cancels: list[str] = []

                @on(Started)
                async def stubborn(event: Started) -> Ended:
                    ready.set()
                    try:
                        await asyncio.sleep(100)
                    except asyncio.CancelledError:
                        cancels.append("lost")
                    await asyncio.sleep(100)
                    return Ended(result="done")

                task = asyncio.ensure_future(
                    EventGraph([stubborn]).ainvoke(
                        Started(data="go"), stop_on=asyncio.Event()
                    )
                )
                await ready.wait()
                task.cancel()
                async with asyncio.timeout(5):
                    log = await task

                assert cancels == ["lost"]
                assert [type(event) for event in log] == [Started, Cancelled]

        def when_it_is_cancelled_again_during_the_handler_cleanup():

            async def it_ends_the_handler_task_before_the_run_ends():
                ready = asyncio.Event()
                handler_tasks: list[asyncio.Task[typing.Any]] = []

                @on(Started)
                async def slow(event: Started) -> Ended:
                    handler_tasks.append(
                        typing.cast("asyncio.Task[typing.Any]", asyncio.current_task())
                    )
                    ready.set()
                    try:
                        await asyncio.sleep(100)
                    except asyncio.CancelledError:
                        await asyncio.sleep(0.1)
                        raise
                    return Ended(result="done")

                task = asyncio.ensure_future(
                    EventGraph([slow]).ainvoke(
                        Started(data="go"), stop_on=asyncio.Event()
                    )
                )
                await ready.wait()
                task.cancel()
                await asyncio.sleep(0.01)
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task

                assert [t.done() for t in handler_tasks] == [True]

    def when_the_handler_cleans_up_after_the_cancel():

        async def it_lets_a_cleanup_shorter_than_the_grace_finish(
            short_stop_grace,
        ):
            stop = asyncio.Event()
            cleanups: list[str] = []

            @on(Started)
            async def tidy(event: Started) -> Ended:
                stop.set()
                try:
                    await asyncio.sleep(100)
                except asyncio.CancelledError:
                    await asyncio.sleep(short_stop_grace / 2)
                    cleanups.append("done")
                    raise
                return Ended(result="done")

            log = await EventGraph([tidy]).ainvoke(Started(data="go"), stop_on=stop)

            assert cleanups == ["done"]
            assert [type(event) for event in log] == [
                Started,
                CancelledOnSignal,
            ]

    def when_the_handler_raises_after_the_stop():

        async def it_logs_the_exception(caplog):
            stop = asyncio.Event()

            @on(Started)
            async def broken(event: Started) -> Ended:
                stop.set()
                try:
                    await asyncio.sleep(100)
                except asyncio.CancelledError:
                    raise RuntimeError("cleanup failed") from None
                return Ended(result="done")

            with caplog.at_level(logging.WARNING, "langgraph_events"):
                log = await EventGraph([broken]).ainvoke(
                    Started(data="go"), stop_on=stop
                )

            assert [type(event) for event in log] == [
                Started,
                CancelledOnSignal,
            ]
            assert [
                (record.name, record.getMessage(), str(record.exc_info[1]))
                for record in caplog.records
                if record.exc_info is not None
            ] == [
                (
                    "langgraph_events._stop",
                    "An async handler raised after its cancel.",
                    "cleanup failed",
                )
            ]

    def when_the_signal_is_set_during_a_slow_checkpoint_write():

        async def it_stores_the_handler_events_and_cancelled_on_signal(
            short_stop_grace,
        ):
            stop = asyncio.Event()

            @on(Started)
            async def reply(event: Started) -> Ended:
                return Ended(result="done")

            class _SlowSaver(MemorySaver):
                async def aput_writes(
                    self, *args: typing.Any, **kwargs: typing.Any
                ) -> None:
                    if not stop.is_set() and "Ended(" in repr(args):
                        stop.set()
                        await asyncio.sleep(short_stop_grace * 2)
                    await super().aput_writes(*args, **kwargs)

            graph = EventGraph([reply], checkpointer=_SlowSaver())
            config = {"configurable": {"thread_id": "stop-on-slow-write"}}

            await graph.ainvoke(Started(data="go"), stop_on=stop, config=config)

            events = (await graph.compiled.aget_state(config)).values["events"]
            assert events == [
                Started(data="go"),
                Ended(result="done"),
                CancelledOnSignal(discarded=""),
            ]

    def when_a_stream_is_stopped():

        async def it_yields_cancelled_on_signal_and_ends():
            stop = asyncio.Event()

            @on(Started)
            async def slow(event: Started) -> Ended:
                stop.set()
                await asyncio.sleep(100)
                return Ended(result="done")

            names = [
                type(event).__name__
                async for event in EventGraph([slow]).astream_events(
                    Started(data="go"), stop_on=stop
                )
            ]

            assert names == ["Started", "CancelledOnSignal"]

    def when_the_signal_has_no_is_set_method():

        @pytest.mark.parametrize(
            "bad",
            [asyncio.Event().is_set, True, asyncio.Event],
            ids=["bound-is_set", "bool", "class"],
        )
        async def it_raises_TypeError_before_any_round_runs(bad):
            with pytest.raises(TypeError, match="stop_on"):
                await EventGraph([to_processed]).ainvoke(
                    Started(data="go"), stop_on=bad
                )


async def _stop_the_replay(
    replay: _Replay, graph: EventGraph, config: dict, durability: str
) -> None:
    """Resume both questions, and let the replay of the first set the stop."""
    ids = await _open_ids(graph, config)
    replay.armed = True
    async with asyncio.timeout(5):
        with pytest.raises(ResumeStoppedError):
            await graph.aresume(
                {
                    ids["first"]: Answered(note="stopped first"),
                    ids["second"]: Answered(note="stopped second"),
                },
                stop_on=replay.stop,
                config=config,
                durability=durability,
            )
    replay.armed = False


async def _paused(second_finishes: bool, durability: str) -> tuple:
    replay = _Replay(second_finishes=second_finishes)
    graph = replay.graph()
    config = {"configurable": {"thread_id": f"replay-{second_finishes}-{durability}"}}
    await graph.ainvoke(Started(data="go"), config=config, durability=durability)
    return replay, graph, config


@pytest.mark.parametrize("durability", ["sync", "async", "exit"])
def describe_a_stop_during_the_replay_of_the_questions():
    def when_the_stop_cuts_each_replay():
        async def it_keeps_each_question_open_under_its_id(durability):
            replay, graph, config = await _paused(False, durability)
            ids = await _open_ids(graph, config)

            await _stop_the_replay(replay, graph, config, durability)

            assert await _open_ids(graph, config) == ids

        async def it_stores_no_answer(durability):
            replay, graph, config = await _paused(False, durability)
            stored = await _stored(graph, config)

            await _stop_the_replay(replay, graph, config, durability)

            assert await _stored(graph, config) == stored
            assert replay.answers == []

        def when_a_later_resume_answers_the_questions():
            async def it_dispatches_only_the_later_answers(durability):
                replay, graph, config = await _paused(False, durability)
                await _stop_the_replay(replay, graph, config, durability)
                ids = await _open_ids(graph, config)

                await graph.aresume(
                    {
                        ids["first"]: Answered(note="first"),
                        ids["second"]: Answered(note="second"),
                    },
                    config=config,
                    durability=durability,
                )

                assert sorted(replay.answers) == ["first", "second"]

        def when_the_stream_resumes():
            async def it_raises_from_the_stream(durability):
                replay, graph, config = await _paused(False, durability)
                ids = await _open_ids(graph, config)
                replay.armed = True

                async with asyncio.timeout(5):
                    with pytest.raises(ResumeStoppedError):
                        async for _ in graph.astream_resume(
                            {ids["first"]: Answered(), ids["second"]: Answered()},
                            stop_on=replay.stop,
                            config=config,
                            durability=durability,
                        ):
                            pass

                assert await _open_ids(graph, config) == ids

    def when_a_replay_finishes_before_the_stop():
        async def it_keeps_only_the_cut_question_open(durability):
            replay, graph, config = await _paused(True, durability)
            ids = await _open_ids(graph, config)

            await _stop_the_replay(replay, graph, config, durability)

            assert await _open_ids(graph, config) == {"first": ids["first"]}

        def when_a_later_resume_answers_the_open_question():
            async def it_dispatches_each_answer_once(durability):
                replay, graph, config = await _paused(True, durability)
                await _stop_the_replay(replay, graph, config, durability)
                ids = await _open_ids(graph, config)

                await graph.aresume(
                    {ids["first"]: Answered(note="first")},
                    config=config,
                    durability=durability,
                )

                assert sorted(replay.answers) == ["first", "stopped second"]


def describe_a_resume_of_a_failed_run():
    def when_the_stop_cuts_the_rerun_of_the_failed_handler():
        async def it_stores_cancelled_on_signal():
            stop = asyncio.Event()
            failing = [True]

            @on(Started)
            async def work(event: Started) -> Ended:
                if failing[0]:
                    raise RuntimeError("boom")
                stop.set()
                await asyncio.sleep(100)
                return Ended(result="late")

            graph = EventGraph([work], checkpointer=MemorySaver())
            config = {"configurable": {"thread_id": "failed-rerun"}}
            with pytest.raises(RuntimeError):
                await graph.ainvoke(Started(data="go"), config=config)
            failing[0] = False

            async with asyncio.timeout(5):
                log = await graph.aresume(ReviewApproved(), stop_on=stop, config=config)

            assert type(list(log)[-1]) is CancelledOnSignal


class Noted(IntegrationEvent):
    text: str = ""


async def _notes(graph: EventGraph, config: dict) -> list[str]:
    return [e.text for e in await _stored(graph, config) if isinstance(e, Noted)]


def describe_resume_seeds():
    def when_two_resumes_in_a_row_are_stopped():
        async def it_stores_only_the_seed_of_the_later_resume():
            replay, graph, config = await _paused(False, "sync")
            ids = await _open_ids(graph, config)
            for _ in range(2):
                replay.armed = True
                async with asyncio.timeout(5):
                    with pytest.raises(ResumeStoppedError):
                        async for _item in graph.astream_resume(
                            {ids["first"]: Answered(), ids["second"]: Answered()},
                            seeds=[Noted(text="stopped")],
                            stop_on=replay.stop,
                            config=config,
                        ):
                            pass
                replay.armed = False
            assert await _notes(graph, config) == []

            async for _item in graph.astream_resume(
                {ids["first"]: Answered(), ids["second"]: Answered()},
                seeds=[Noted(text="kept")],
                config=config,
            ):
                pass

            assert await _notes(graph, config) == ["kept"]

    def when_the_handler_pauses_a_second_time():
        async def it_stores_the_seed_of_each_resume():
            @on(Started)
            async def ask(event: Started) -> ApprovalRequested:
                return ApprovalRequested(draft=event.data)

            graph = EventGraph([ask], checkpointer=MemorySaver())
            config = {"configurable": {"thread_id": "seeds-twice"}}
            await graph.ainvoke([Started(data="a"), Started(data="b")], config=config)

            for text in ("one", "two"):
                async for _item in graph.astream_resume(
                    Answered(), seeds=[Noted(text=text)], config=config
                ):
                    pass

            assert await _notes(graph, config) == ["one", "two"]

    def when_the_handler_paused_on_a_raw_interrupt():
        async def it_raises_and_keeps_the_question():
            @on(Started)
            async def ask(event: Started) -> Ended:
                answer = lg_interrupt(ApprovalRequested(draft="raw"))
                return Ended(result=type(answer).__name__)

            graph = EventGraph([ask], checkpointer=MemorySaver())
            config = {"configurable": {"thread_id": "seeds-raw"}}
            await graph.ainvoke(Started(data="go"), config=config)

            with pytest.raises(ValueError, match="raw"):
                async for _item in graph.astream_resume(
                    Answered(), seeds=[Noted(text="x")], config=config
                ):
                    pass

            assert (await graph.aget_state(config)).is_interrupted
