"""Integration tests for EventGraph — the full event-driven graph engine."""

import asyncio
import warnings

import pytest
from conftest import Completed, Ended, MessageReceived, MessageSent, Processed, Started
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)

from langgraph_events import (
    STATE_SNAPSHOT_EVENT_NAME,
    Cancelled,
    Event,
    EventGraph,
    EventLog,
    Halted,
    Interrupted,
    MaxRoundsExceeded,
    MessageEvent,
    OrphanedEventWarning,
    Reducer,
    Resumed,
    ScalarReducer,
    Scatter,
    StateSnapshotFrame,
    StreamFrame,
    SystemPromptSet,
    aemit_custom,
    aemit_state_snapshot,
    emit_custom,
    emit_state_snapshot,
    message_reducer,
    on,
)

# ---------------------------------------------------------------------------
# Helpers (prefixed with _ to exclude from collection)
# ---------------------------------------------------------------------------


class _StepInterrupted(Interrupted):
    """Module-level Interrupted subclass for checkpoint-aware tests.

    Defined here (not inside test functions) so that LangGraph's
    serializer can resolve the class on checkpoint restore.
    """

    step: int = 0


def _data_reducer() -> Reducer:
    """Simple reducer that accumulates Started.data values."""
    return Reducer(name="data_items", event_type=Started, fn=lambda e: [e.data])


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def describe_EventGraph():

    def describe_invoke():

        def describe_linear_chain():

            def it_processes_three_step_chain(linear_chain):
                log = linear_chain.invoke(Started(data="hello"))
                assert isinstance(log, EventLog)
                assert len(log) == 3
                assert log.latest(Ended) == Ended(result="done:processed:hello")

            async def it_processes_async_handlers():
                @on(Started)
                async def step1(event: Started) -> Processed:
                    return Processed(data=event.data.upper())

                @on(Processed)
                async def step2(event: Processed) -> Ended:
                    return Ended(result=event.data)

                graph = EventGraph([step1, step2])
                log = await graph.ainvoke(Started(data="hello"))
                assert log.latest(Ended) == Ended(result="HELLO")

            @pytest.mark.asyncio
            async def it_raises_clear_error_for_invoke_inside_running_loop():
                @on(Started)
                async def step(event: Started) -> Ended:
                    return Ended(result=event.data)

                graph = EventGraph([step])
                with pytest.raises(RuntimeError, match=r"Use ainvoke\(\) instead"):
                    graph.invoke(Started(data="hello"))

        def describe_branching():

            class InputReceived(Event):
                kind: str = ""
                data: str = ""

            class FastPathChosen(Event):
                data: str = ""

            class SlowPathChosen(Event):
                data: str = ""

            class OutputProduced(Event):
                result: str = ""

            @pytest.fixture
            def branching_graph():
                @on(InputReceived)
                def route(
                    event: InputReceived,
                ) -> FastPathChosen | SlowPathChosen | None:
                    if event.kind == "fast":
                        return FastPathChosen(data=event.data)
                    return SlowPathChosen(data=event.data)

                @on(FastPathChosen)
                def handle_fast(event: FastPathChosen) -> OutputProduced:
                    return OutputProduced(result=f"fast:{event.data}")

                @on(SlowPathChosen)
                def handle_slow(event: SlowPathChosen) -> OutputProduced:
                    return OutputProduced(result=f"slow:{event.data}")

                return EventGraph([route, handle_fast, handle_slow])

            def when_fast_path():

                def it_produces_fast_output(branching_graph):
                    log = branching_graph.invoke(InputReceived(kind="fast", data="x"))
                    assert log.latest(OutputProduced) == OutputProduced(result="fast:x")

                def it_does_not_trigger_slow_handler(branching_graph):
                    log = branching_graph.invoke(InputReceived(kind="fast", data="x"))
                    assert not log.has(SlowPathChosen)

            def when_slow_path():

                def it_produces_slow_output(branching_graph):
                    log = branching_graph.invoke(InputReceived(kind="slow", data="y"))
                    assert log.latest(OutputProduced) == OutputProduced(result="slow:y")

                def it_does_not_trigger_fast_handler(branching_graph):
                    log = branching_graph.invoke(InputReceived(kind="slow", data="y"))
                    assert not log.has(FastPathChosen)

        def describe_fan_out_via_inheritance():

            class Tracked(Event):
                action: str = ""

            class ProcessCompleted(Event):
                item: str = ""

            class TrackedItem(Tracked, ProcessCompleted):
                action: str = ""
                item: str = ""

            class AuditDone(Event):
                msg: str = ""

            class ProcessDone(Event):
                msg: str = ""

            def it_triggers_both_parent_handlers():
                @on(Tracked)
                def audit(event: Tracked) -> AuditDone:
                    return AuditDone(msg=f"audited:{event.action}")

                @on(ProcessCompleted)
                def process(event: ProcessCompleted) -> ProcessDone:
                    return ProcessDone(msg=f"processed:{event.item}")

                graph = EventGraph([audit, process])
                log = graph.invoke(TrackedItem(action="create", item="doc1"))
                assert log.has(AuditDone)
                assert log.has(ProcessDone)
                assert log.latest(AuditDone) == AuditDone(msg="audited:create")
                assert log.latest(ProcessDone) == ProcessDone(msg="processed:doc1")

            def it_fires_parent_handler_for_child_event():
                class BaseReceived(Event):
                    x: str = ""

                class ChildReceived(BaseReceived):
                    y: str = ""

                class ResultProduced(Event):
                    v: str = ""

                @on(BaseReceived)
                def handle_base(event: BaseReceived) -> ResultProduced:
                    return ResultProduced(v=event.x)

                graph = EventGraph([handle_base])
                log = graph.invoke(ChildReceived(x="hello", y="world"))
                assert log.latest(ResultProduced) == ResultProduced(v="hello")

        def describe_side_effect_handlers():

            def it_executes_side_effect_on_none_return():
                side_effects: list[str] = []

                @on(Started)
                def produce(event: Started) -> Processed:
                    return Processed(data=event.data)

                @on(Processed)
                def consume(event: Processed) -> None:
                    side_effects.append(event.data)

                graph = EventGraph([produce, consume])
                log = graph.invoke(Started(data="test"))
                assert len(log) == 2
                assert side_effects == ["test"]

        def describe_event_log_injection():

            def it_provides_full_log_to_handler():
                @on(Started)
                def step1(event: Started) -> Processed:
                    return Processed(data=event.data)

                @on(Processed)
                def step2(event: Processed, log: EventLog) -> Ended:
                    assert log.has(Started)
                    count = len(log.filter(Event))
                    return Ended(result=f"saw {count} events")

                graph = EventGraph([step1, step2])
                log = graph.invoke(Started(data="hello"))
                assert log.latest(Ended) == Ended(result="saw 2 events")

            def it_shows_snapshot_not_affected_by_later_events():
                log_lengths: list[int] = []

                @on(Started)
                def step1(event: Started) -> Processed:
                    return Processed(data="from_step1")

                @on(Processed)
                def step2(event: Processed, log: EventLog) -> Ended:
                    log_lengths.append(len(log))
                    assert not log.has(Ended)
                    return Ended(result="done")

                graph = EventGraph([step1, step2])
                final_log = graph.invoke(Started(data="test"))
                assert log_lengths == [2]
                assert len(final_log) == 3

            def it_prevents_mutation_from_corrupting_graph_state():
                @on(Started)
                def evil_handler(event: Started, log: EventLog) -> Processed:
                    with pytest.raises(AttributeError):
                        log._events.append(Ended(result="INJECTED"))  # type: ignore[attr-defined]
                    return Processed(data="honest")

                @on(Processed)
                def step2(event: Processed, log: EventLog) -> Ended:
                    assert log.has(Started)
                    assert log.has(Processed)
                    injected = [
                        e
                        for e in log
                        if isinstance(e, Ended) and e.result == "INJECTED"
                    ]
                    assert injected == []
                    return Ended(result="clean")

                graph = EventGraph([evil_handler, step2])
                final_log = graph.invoke(Started(data="test"))
                assert len(final_log) == 3
                assert final_log.latest(Ended) == Ended(result="clean")
                injected = [
                    e
                    for e in final_log
                    if isinstance(e, Ended) and e.result == "INJECTED"
                ]
                assert injected == []

            def it_provides_independent_snapshots_to_parallel_handlers():
                class Triggered(Event):
                    value: str = ""

                class ResultAProduced(Event):
                    saw_events: int = 0

                class ResultBProduced(Event):
                    saw_events: int = 0

                class Collected(Event):
                    a_saw: int = 0
                    b_saw: int = 0

                @on(Triggered)
                def handler_a(event: Triggered, log: EventLog) -> ResultAProduced:
                    with pytest.raises(AttributeError):
                        log._events.append(Ended(result="from_a"))  # type: ignore[attr-defined]
                    return ResultAProduced(saw_events=len(log))

                @on(Triggered)
                def handler_b(event: Triggered, log: EventLog) -> ResultBProduced:
                    has_end = any(isinstance(e, Ended) for e in log)
                    assert not has_end
                    return ResultBProduced(saw_events=len(log))

                @on(ResultAProduced, ResultBProduced)
                def collect(event: Event, log: EventLog) -> Collected | None:
                    if log.has(ResultAProduced) and log.has(ResultBProduced):
                        a = log.latest(ResultAProduced)
                        b = log.latest(ResultBProduced)
                        return Collected(a_saw=a.saw_events, b_saw=b.saw_events)
                    return None

                graph = EventGraph([handler_a, handler_b, collect])
                final_log = graph.invoke(Triggered(value="go"))
                result = final_log.latest(Collected)
                assert result is not None
                assert result.b_saw == 1

        def describe_multi_subscription():

            class PingSent(Event):
                value: str = ""

            class PongReceived(Event):
                value: str = ""

            class Replied(Event):
                value: str = ""

            def when_single_type_pending():

                def it_fires_on_either_event_type():
                    @on(PingSent, PongReceived)
                    def echo(event: Event) -> Replied:
                        if isinstance(event, PingSent):
                            return Replied(value=f"ping:{event.value}")
                        return Replied(value=f"pong:{event.value}")

                    @on(Replied)
                    def finish(event: Replied) -> Completed:
                        return Completed(result=event.value)

                    graph = EventGraph([echo, finish])
                    log = graph.invoke(PingSent(value="hello"))
                    assert log.latest(Completed) == Completed(result="ping:hello")
                    log = graph.invoke(PongReceived(value="world"))
                    assert log.latest(Completed) == Completed(result="pong:world")

                def it_provides_log_to_multi_sub_handler():
                    class MsgAReceived(Event):
                        text: str = ""

                    class MsgBReceived(Event):
                        text: str = ""

                    class Summarized(Event):
                        count: int = 0

                    @on(MsgAReceived, MsgBReceived)
                    def summarize(event: Event, log: EventLog) -> Summarized:
                        total = len(log.filter(Event))
                        return Summarized(count=total)

                    graph = EventGraph([summarize])
                    log = graph.invoke(MsgAReceived(text="hi"))
                    assert log.latest(Summarized) == Summarized(count=1)

                def it_supports_react_loop_pattern():
                    class UserMsgReceived(Event):
                        content: str = ""

                    class AssistantMsgSent(Event):
                        content: str = ""
                        needs_tool: bool = False

                    class ToolResultReturned(Event):
                        result: str = ""

                    class FinalAnswerProduced(Event):
                        answer: str = ""

                    call_count = 0

                    @on(UserMsgReceived, ToolResultReturned)
                    def call_llm(event: Event, log: EventLog) -> AssistantMsgSent:
                        nonlocal call_count
                        call_count += 1
                        if isinstance(event, UserMsgReceived):
                            return AssistantMsgSent(
                                content="need tool", needs_tool=True
                            )
                        return AssistantMsgSent(
                            content=f"got:{event.result}",
                            needs_tool=False,
                        )

                    @on(AssistantMsgSent)
                    def handle_response(
                        event: AssistantMsgSent,
                    ) -> ToolResultReturned | FinalAnswerProduced:
                        if event.needs_tool:
                            return ToolResultReturned(result="42")
                        return FinalAnswerProduced(answer=event.content)

                    graph = EventGraph([call_llm, handle_response])
                    log = graph.invoke(UserMsgReceived(content="what is 6*7?"))
                    assert call_count == 2
                    assert log.latest(FinalAnswerProduced) == (
                        FinalAnswerProduced(answer="got:42")
                    )
                    assert log.has(ToolResultReturned)
                    assert log.has(AssistantMsgSent)

            def when_both_types_pending():

                def it_dispatches_handler_only_once():
                    @on(PingSent, PongReceived)
                    def echo(event: Event) -> Replied:
                        return Replied(value="seen")

                    @on(Replied)
                    def finish(event: Replied) -> Completed:
                        return Completed(result=event.value)

                    graph = EventGraph([echo, finish])
                    log = graph.invoke([PingSent(value="a"), PongReceived(value="b")])
                    # Handler fires once per matching event, but is dispatched
                    # only once (not duplicated in matched list)
                    replies = log.filter(Replied)
                    assert len(replies) == 2
                    assert log.filter(Completed) == [
                        Completed(result="seen"),
                        Completed(result="seen"),
                    ]

        def describe_multi_seed():

            def it_accepts_list_of_seed_events():
                @on(Started)
                def step1(event: Started) -> Processed:
                    return Processed(data=event.data)

                @on(Processed)
                def step2(event: Processed) -> Ended:
                    return Ended(result=event.data)

                graph = EventGraph([step1, step2])
                log = graph.invoke([Started(data="hello")])
                assert log.latest(Ended) == Ended(result="hello")

            def it_includes_all_seed_events_in_log():
                class ConfigSet(Event):
                    setting: str = ""

                @on(Started)
                def handle(event: Started, log: EventLog) -> Ended:
                    config = log.latest(ConfigSet)
                    return Ended(result=f"{config.setting}:{event.data}")

                graph = EventGraph([handle])
                log = graph.invoke([ConfigSet(setting="v1"), Started(data="go")])
                assert log.has(ConfigSet)
                assert log.has(Started)
                assert log.latest(Ended) == Ended(result="v1:go")

            def it_still_accepts_single_event():
                @on(Started)
                def step(event: Started) -> Ended:
                    return Ended(result=event.data)

                graph = EventGraph([step])
                log = graph.invoke(Started(data="solo"))
                assert log.latest(Ended) == Ended(result="solo")

            def describe_SystemPromptSet():

                def it_from_str_creates_message_event():
                    event = SystemPromptSet.from_str("You are helpful")
                    assert isinstance(event, SystemPromptSet)
                    assert isinstance(event, MessageEvent)
                    msgs = event.as_messages()
                    assert len(msgs) == 1
                    assert isinstance(msgs[0], SystemMessage)
                    assert msgs[0].content == "You are helpful"

                def it_is_frozen():
                    event = SystemPromptSet.from_str("test")
                    with pytest.raises(AttributeError):
                        event.message = SystemMessage(  # type: ignore
                            content="changed"
                        )

                def it_is_queryable_as_seed():
                    @on(Started)
                    def handle(event: Started, log: EventLog) -> Ended:
                        has_prompt = log.has(SystemPromptSet)
                        return Ended(result=f"has_prompt={has_prompt}")

                    graph = EventGraph([handle])
                    log = graph.invoke(
                        [
                            SystemPromptSet.from_str("You are helpful"),
                            Started(data="go"),
                        ]
                    )
                    assert log.has(SystemPromptSet)
                    assert log.latest(Ended) == Ended(result="has_prompt=True")

                def it_contributes_to_message_reducer():
                    class UserMsgReceived(MessageEvent):
                        message: HumanMessage = None  # type: ignore[assignment]

                    class Finished(Event):
                        answer: str = ""

                    r = message_reducer()

                    received_messages: list[list[BaseMessage]] = []

                    @on(UserMsgReceived)
                    def respond(
                        event: UserMsgReceived, messages: list[BaseMessage]
                    ) -> Finished:
                        received_messages.append(list(messages))
                        return Finished(answer="ok")

                    graph = EventGraph([respond], reducers=[r])
                    log = graph.invoke(
                        [
                            SystemPromptSet.from_str("You are a test bot"),
                            UserMsgReceived(message=HumanMessage(content="hello")),
                        ]
                    )
                    assert log.latest(Finished) is not None
                    msgs = received_messages[0]
                    assert len(msgs) == 2
                    assert isinstance(msgs[0], SystemMessage)
                    assert msgs[0].content == "You are a test bot"
                    assert msgs[1].content == "hello"

        def describe_ainvoke():

            async def it_handles_multi_seed():
                @on(Started)
                async def step(event: Started) -> Ended:
                    return Ended(result=event.data)

                graph = EventGraph([step])
                log = await graph.ainvoke([Started(data="a"), Started(data="b")])
                ends = log.filter(Ended)
                assert len(ends) == 2
                assert {e.result for e in ends} == {"a", "b"}

            async def it_stops_on_halt():
                @on(Started)
                async def halter(event: Started) -> Halted:
                    return Halted()

                @on(Halted)
                async def unreachable(event: Halted) -> Ended:
                    return Ended(result="should not run")

                graph = EventGraph([halter, unreachable])
                log = await graph.ainvoke(Started(data="go"))
                assert log.has(Halted)
                assert not log.has(Ended)

            async def it_injects_reducer_values():
                reducer = _data_reducer()

                @on(Started)
                async def step(event: Started, data_items: list) -> Ended:
                    return Ended(result=",".join(data_items))

                graph = EventGraph([step], reducers=[reducer])
                log = await graph.ainvoke(Started(data="x"))
                assert log.latest(Ended) == Ended(result="x")

    def describe_config_and_store():

        def when_handler_requests_config():

            def it_receives_a_runnable_config_dict():
                from langchain_core.runnables import RunnableConfig

                captured: list[RunnableConfig] = []

                @on(Started)
                def step(event: Started, config: RunnableConfig) -> Ended:
                    captured.append(config)
                    return Ended(result="ok")

                graph = EventGraph([step])
                graph.invoke(Started(data="x"))
                assert len(captured) == 1
                assert "configurable" in captured[0]

        def when_handler_requests_store():

            def when_store_configured():

                def it_can_put_and_get_via_store():
                    from langgraph.store.base import BaseStore
                    from langgraph.store.memory import InMemoryStore

                    store = InMemoryStore()

                    @on(Started)
                    async def step(event: Started, store: BaseStore) -> Ended:
                        await store.aput(("test",), "key1", {"val": event.data})
                        items = await store.aget(("test",), "key1")
                        return Ended(result=items.value["val"])

                    graph = EventGraph([step], store=store)
                    log = graph.invoke(Started(data="hello"))
                    assert log.latest(Ended) == Ended(result="hello")

            def when_store_not_configured():

                def it_raises_for_sync_handler():
                    from langgraph.store.base import BaseStore

                    @on(Started)
                    def step(event: Started, store: BaseStore) -> Ended:
                        return Ended(result="ok")

                    graph = EventGraph([step])
                    with pytest.raises(ValueError, match="no store is configured"):
                        graph.invoke(Started(data="hello"))

                async def it_raises_for_async_handler():
                    from langgraph.store.base import BaseStore

                    @on(Started)
                    async def step(event: Started, store: BaseStore) -> Ended:
                        return Ended(result="ok")

                    graph = EventGraph([step])
                    with pytest.raises(ValueError, match="no store is configured"):
                        await graph.ainvoke(Started(data="hello"))

        def when_handler_requests_neither():

            def it_runs_handler():
                @on(Started)
                def step(event: Started) -> Ended:
                    return Ended(result=event.data)

                graph = EventGraph([step])
                log = graph.invoke(Started(data="hi"))
                assert log.latest(Ended) == Ended(result="hi")

        def when_handler_requests_config_and_log():

            def it_injects_both():
                from langchain_core.runnables import RunnableConfig

                captured: list[tuple] = []

                @on(Started)
                def step(
                    event: Started,
                    log: EventLog,
                    config: RunnableConfig,
                ) -> Ended:
                    captured.append((len(log), "configurable" in config))
                    return Ended(result="ok")

                graph = EventGraph([step])
                graph.invoke(Started(data="x"))
                assert len(captured) == 1
                assert captured[0] == (1, True)

    def describe_halt():

        def it_is_Event_subclass():
            h = Halted()
            assert isinstance(h, Event)
            assert isinstance(h, Halted)

        def it_preserves_subtype_fields():
            h = MaxRoundsExceeded(rounds=5)
            assert isinstance(h, Halted)
            assert h.rounds == 5

        def it_stops_execution_immediately():
            @on(Started)
            def step1(event: Started) -> Halted:
                return Halted()

            @on(Halted)
            def should_not_run(event: Halted) -> Ended:
                return Ended(result="should not reach here")

            graph = EventGraph([step1, should_not_run])
            log = graph.invoke(Started(data="test"))
            assert log.has(Halted)
            assert not log.has(Ended)

    def describe_interrupt():

        def it_is_a_bare_marker_supporting_typed_subclasses():
            assert isinstance(Interrupted(), Event)

            class ConfirmationRequested(Interrupted):
                data: str

            c = ConfirmationRequested(data="test")
            assert c.data == "test"
            assert isinstance(c, Interrupted)

        def it_stores_value_and_interrupted_reference():
            class Confirmed(Event):
                pass

            i = Interrupted()
            confirm = Confirmed()
            r = Resumed(value=confirm, interrupted=i)
            assert r.value is confirm
            assert r.interrupted is i
            assert isinstance(r, Event)

        def it_pauses_and_resumes():
            from langgraph.checkpoint.memory import MemorySaver

            class ConfirmationRequested(Interrupted):
                data: str

            class Confirmed(Event):
                pass

            @on(Started)
            def need_input(event: Started) -> ConfirmationRequested:
                return ConfirmationRequested(data=event.data)

            @on(Confirmed)
            def handle_confirm(event: Confirmed) -> Ended:
                return Ended(result="confirmed")

            graph = EventGraph(
                [need_input, handle_confirm],
                checkpointer=MemorySaver(),
            )

            config = {"configurable": {"thread_id": "interrupt-test"}}
            graph.invoke(Started(data="test"), config=config)
            state = graph.get_state(config)
            assert state.is_interrupted

            log = graph.resume(Confirmed(), config=config)
            assert log.latest(Ended) == Ended(result="confirmed")

        def it_raises_on_resume_missing_checkpointer():
            class Confirmed(Event):
                pass

            @on(Started)
            def need_input(event: Started) -> Interrupted:
                return Interrupted()

            graph = EventGraph([need_input])
            with pytest.raises(ValueError, match=r"resume.*requires a checkpointer"):
                graph.resume(Confirmed())

        def it_raises_type_error_for_non_event_resume():
            from langgraph.checkpoint.memory import MemorySaver

            @on(Started)
            def need_input(event: Started) -> Interrupted:
                return Interrupted()

            @on(Resumed)
            def handle_resume(event: Resumed) -> Ended:
                return Ended(result="done")

            graph = EventGraph(
                [need_input, handle_resume],
                checkpointer=MemorySaver(),
            )
            config = {"configurable": {"thread_id": "type-error-test"}}
            graph.invoke(Started(data="test"), config=config)

            with pytest.raises(TypeError, match=r"resume\(\) requires an Event"):
                graph.resume("yes", config=config)  # type: ignore[arg-type]

        def it_auto_dispatches_event_resume_value():
            from langgraph.checkpoint.memory import MemorySaver

            class ApprovalSubmitted(Event):
                approved: bool = False

            handler_fired = []

            @on(Started)
            def need_input(event: Started) -> Interrupted:
                return Interrupted()

            @on(ApprovalSubmitted)
            def handle_approval(event: ApprovalSubmitted) -> Ended:
                handler_fired.append(event)
                return Ended(result=f"approved={event.approved}")

            @on(Resumed)
            def handle_resume(event: Resumed) -> Completed:
                val = event.value
                approved = val.approved if isinstance(val, ApprovalSubmitted) else False
                return Completed(result=f"resumed:approved={approved}")

            graph = EventGraph(
                [need_input, handle_approval, handle_resume],
                checkpointer=MemorySaver(),
            )
            config = {"configurable": {"thread_id": "auto-dispatch-test"}}
            graph.invoke(Started(data="test"), config=config)

            approval = ApprovalSubmitted(approved=True)
            log = graph.resume(approval, config=config)

            # Handler subscribed to ApprovalSubmitted fires
            assert len(handler_fired) == 1
            assert handler_fired[0] == approval

            # ApprovalSubmitted appears before Resumed in the log
            events = list(log)
            approval_idx = next(
                i for i, e in enumerate(events) if isinstance(e, ApprovalSubmitted)
            )
            resumed_idx = next(
                i for i, e in enumerate(events) if isinstance(e, Resumed)
            )
            assert approval_idx < resumed_idx

            # Resumed.value holds the Event reference
            resumed = log.latest(Resumed)
            assert resumed.value is approval

        def it_processes_event_through_reducer():
            from langgraph.checkpoint.memory import MemorySaver

            class UserMsgReceived(MessageEvent):
                message: HumanMessage

            @on(Started)
            def need_input(event: Started) -> Interrupted:
                return Interrupted()

            received_messages: list = []

            @on(Resumed)
            def handle_resume(event: Resumed, messages: list) -> Ended:
                received_messages.extend(messages)
                return Ended(result="done")

            graph = EventGraph(
                [need_input, handle_resume],
                checkpointer=MemorySaver(),
                reducers=[message_reducer()],
            )
            config = {"configurable": {"thread_id": "reducer-dispatch-test"}}
            graph.invoke(Started(data="test"), config=config)

            user_msg = UserMsgReceived(message=HumanMessage(content="hello from human"))
            log = graph.resume(user_msg, config=config)

            # The message reducer saw the auto-dispatched UserMsgReceived
            assert any(
                isinstance(m, HumanMessage) and m.content == "hello from human"
                for m in received_messages
            )
            assert log.latest(Ended) == Ended(result="done")

    def describe_field_matchers():

        def when_field_matches():

            def it_dispatches_the_handler():
                from langgraph.checkpoint.memory import MemorySaver

                class ApprovalRequested(Interrupted):
                    draft: str = ""

                class ReviewApproved(Event):
                    pass

                captured = []

                @on(Started)
                def need_input(event: Started) -> ApprovalRequested:
                    return ApprovalRequested(draft="hello")

                @on(Resumed, interrupted=ApprovalRequested)
                def handle_approval(event: Resumed) -> Ended:
                    captured.append(event.interrupted)
                    return Ended(result="approved")

                graph = EventGraph(
                    [need_input, handle_approval],
                    checkpointer=MemorySaver(),
                )
                config = {"configurable": {"thread_id": "field-match-test"}}
                graph.invoke(Started(data="test"), config=config)
                log = graph.resume(ReviewApproved(), config=config)

                assert log.latest(Ended) == Ended(result="approved")
                assert len(captured) == 1
                assert isinstance(captured[0], ApprovalRequested)

        def when_field_does_not_match():

            def it_skips_the_handler():
                from langgraph.checkpoint.memory import MemorySaver

                class ApprovalRequested(Interrupted):
                    draft: str = ""

                class OtherInterrupted(Interrupted):
                    reason: str = ""

                class ReviewApproved(Event):
                    pass

                captured = []

                @on(Started)
                def need_input(event: Started) -> OtherInterrupted:
                    return OtherInterrupted(reason="different")

                @on(Resumed, interrupted=ApprovalRequested)
                def handle_approval(event: Resumed) -> Ended:
                    captured.append("should not fire")
                    return Ended(result="approved")

                @on(ReviewApproved)
                def fallback(event: ReviewApproved) -> Ended:
                    return Ended(result="fallback")

                graph = EventGraph(
                    [need_input, handle_approval, fallback],
                    checkpointer=MemorySaver(),
                )
                config = {"configurable": {"thread_id": "field-no-match-test"}}
                graph.invoke(Started(data="test"), config=config)
                log = graph.resume(ReviewApproved(), config=config)

                assert len(captured) == 0
                assert log.latest(Ended) == Ended(result="fallback")

        def when_field_is_none():

            def it_skips_the_handler():
                """A None field value does not match a field matcher."""
                from langgraph.checkpoint.memory import MemorySaver

                class ApprovalRequested(Interrupted):
                    draft: str = ""

                class OtherInterrupted(Interrupted):
                    reason: str = ""

                class Acknowledge(Event):
                    pass

                captured = []

                @on(Started)
                def need_input(event: Started) -> OtherInterrupted:
                    return OtherInterrupted(reason="test")

                @on(Resumed, interrupted=ApprovalRequested)
                def approval_handler(event: Resumed) -> Ended:
                    captured.append("should not fire")
                    return Ended(result="approval")

                @on(Acknowledge)
                def fallback(event: Acknowledge) -> Ended:
                    return Ended(result="fallback")

                graph = EventGraph(
                    [need_input, approval_handler, fallback],
                    checkpointer=MemorySaver(),
                )
                config = {"configurable": {"thread_id": "field-none-test"}}
                graph.invoke(Started(data="test"), config=config)
                log = graph.resume(Acknowledge(), config=config)

                assert len(captured) == 0
                assert log.latest(Ended) == Ended(result="fallback")

        def when_handler_requests_field_injection():

            def it_injects_the_narrowed_field():
                from langgraph.checkpoint.memory import MemorySaver

                class ApprovalRequested(Interrupted):
                    draft: str = ""

                class ReviewApproved(Event):
                    pass

                injected_values = []

                @on(Started)
                def need_input(event: Started) -> ApprovalRequested:
                    return ApprovalRequested(draft="my draft")

                @on(Resumed, interrupted=ApprovalRequested)
                def handle_approval(
                    event: Resumed, interrupted: ApprovalRequested
                ) -> Ended:
                    injected_values.append(interrupted)
                    return Ended(result=interrupted.draft)

                graph = EventGraph(
                    [need_input, handle_approval],
                    checkpointer=MemorySaver(),
                )
                config = {"configurable": {"thread_id": "field-inject-test"}}
                graph.invoke(Started(data="test"), config=config)
                log = graph.resume(ReviewApproved(), config=config)

                assert log.latest(Ended) == Ended(result="my draft")
                assert len(injected_values) == 1
                assert isinstance(injected_values[0], ApprovalRequested)
                assert injected_values[0].draft == "my draft"

        def when_multiple_field_matchers():

            def it_requires_all_fields_to_match():
                from langgraph.checkpoint.memory import MemorySaver

                class ApprovalRequested(Interrupted):
                    draft: str = ""

                class ReviewApproved(Event):
                    pass

                captured = []

                @on(Started)
                def need_input(event: Started) -> ApprovalRequested:
                    return ApprovalRequested(draft="hello")

                @on(Resumed, interrupted=ApprovalRequested, value=ReviewApproved)
                def strict_handler(
                    event: Resumed,
                    interrupted: ApprovalRequested,
                    value: ReviewApproved,
                ) -> Ended:
                    captured.append((interrupted, value))
                    return Ended(result="strict")

                graph = EventGraph(
                    [need_input, strict_handler],
                    checkpointer=MemorySaver(),
                )
                config = {"configurable": {"thread_id": "multi-field-test"}}
                graph.invoke(Started(data="test"), config=config)
                log = graph.resume(ReviewApproved(), config=config)

                assert log.latest(Ended) == Ended(result="strict")
                assert len(captured) == 1
                assert isinstance(captured[0][0], ApprovalRequested)
                assert isinstance(captured[0][1], ReviewApproved)

            def when_one_field_does_not_match():

                def it_skips_the_handler():
                    from langgraph.checkpoint.memory import MemorySaver

                    class ApprovalRequested(Interrupted):
                        draft: str = ""

                    class ReviewApproved(Event):
                        pass

                    class OtherEvent(Event):
                        pass

                    captured = []

                    @on(Started)
                    def need_input(event: Started) -> ApprovalRequested:
                        return ApprovalRequested(draft="hello")

                    # value=OtherEvent won't match ReviewApproved
                    @on(Resumed, interrupted=ApprovalRequested, value=OtherEvent)
                    def strict_handler(event: Resumed) -> Ended:
                        captured.append("should not fire")
                        return Ended(result="strict")

                    @on(ReviewApproved)
                    def fallback(event: ReviewApproved) -> Ended:
                        return Ended(result="fallback")

                    graph = EventGraph(
                        [need_input, strict_handler, fallback],
                        checkpointer=MemorySaver(),
                    )
                    config = {"configurable": {"thread_id": "multi-field-skip"}}
                    graph.invoke(Started(data="test"), config=config)
                    log = graph.resume(ReviewApproved(), config=config)

                    assert len(captured) == 0
                    assert log.latest(Ended) == Ended(result="fallback")

    def describe_scatter():

        class BatchReceived(Event):
            items: tuple = ()

        class WorkItemDispatched(Event):
            item: str = ""
            batch_size: int = 0

        class WorkDone(Event):
            item: str = ""
            result: str = ""

        class BatchResultCollected(Event):
            results: tuple = ()

        def describe_construction():

            def when_valid_events():

                def it_wraps_a_list_of_events():
                    class ItemDispatched(Event):
                        v: int = 0

                    s = Scatter(
                        [
                            ItemDispatched(v=1),
                            ItemDispatched(v=2),
                            ItemDispatched(v=3),
                        ]
                    )
                    assert len(s.events) == 3
                    assert s.events[0] == ItemDispatched(v=1)

            def when_empty():

                def it_raises_value_error():
                    with pytest.raises(ValueError, match="at least one"):
                        Scatter([])

            def when_contains_non_events():

                def it_raises_type_error():
                    with pytest.raises(TypeError, match="Event instances"):
                        Scatter(["not an event"])  # type: ignore

        def when_multiple_items():

            def it_fans_out_work_items_and_gathers_results():
                @on(BatchReceived)
                def split(event: BatchReceived) -> Scatter:
                    return Scatter(
                        [
                            WorkItemDispatched(item=item, batch_size=len(event.items))
                            for item in event.items
                        ]
                    )

                @on(WorkItemDispatched)
                def process(event: WorkItemDispatched) -> WorkDone:
                    return WorkDone(item=event.item, result=f"done:{event.item}")

                @on(WorkDone)
                def gather(
                    event: WorkDone, log: EventLog
                ) -> BatchResultCollected | None:
                    all_done = log.filter(WorkDone)
                    batch = log.latest(BatchReceived)
                    if len(all_done) >= len(batch.items):
                        return BatchResultCollected(
                            results=tuple(e.result for e in all_done)
                        )
                    return None

                graph = EventGraph([split, process, gather])
                log = graph.invoke(BatchReceived(items=("a", "b", "c")))
                assert log.has(BatchResultCollected)
                result = log.latest(BatchResultCollected)
                assert len(result.results) == 3
                assert set(result.results) == {"done:a", "done:b", "done:c"}

        def when_single_item():

            def it_still_produces_output():
                @on(BatchReceived)
                def split(event: BatchReceived) -> Scatter:
                    return Scatter([WorkItemDispatched(item=event.items[0])])

                @on(WorkItemDispatched)
                def process(event: WorkItemDispatched) -> WorkDone:
                    return WorkDone(item=event.item, result=f"ok:{event.item}")

                graph = EventGraph([split, process])
                log = graph.invoke(BatchReceived(items=("only",)))
                assert log.latest(WorkDone) == WorkDone(item="only", result="ok:only")

    def describe_reducer():

        def when_reducers_configured():

            def it_returns_frozenset_of_reducer_names():
                r1 = Reducer(name="alpha", event_type=Started, fn=lambda e: [e.data])
                r2 = Reducer(name="beta", event_type=Started, fn=lambda e: [e.data])

                @on(Started)
                def noop(event: Started) -> Completed:
                    return Completed(result="x")

                graph = EventGraph([noop], reducers=[r1, r2])
                assert graph.reducer_names == frozenset({"alpha", "beta"})

        def when_no_reducers():

            def it_returns_empty_frozenset():
                @on(Started)
                def noop(event: Started) -> Completed:
                    return Completed(result="x")

                graph = EventGraph([noop])
                assert graph.reducer_names == frozenset()

        def when_reserved_name():

            @pytest.mark.parametrize(
                "reserved_name",
                ["events", "_cursor", "_pending", "_round"],
            )
            def it_rejects_collisions(reserved_name):
                r = Reducer(name=reserved_name, event_type=Event, fn=lambda e: [])

                @on(Started)
                def noop(event: Started) -> Completed:
                    return Completed(result="x")

                with pytest.raises(
                    ValueError, match="conflict with reserved state fields"
                ):
                    EventGraph([noop], reducers=[r])

        def describe_injection():

            def it_passes_accumulated_values_to_handler():
                def project(event: Event) -> list:
                    if isinstance(event, MessageReceived):
                        return [f"in:{event.text}"]
                    if isinstance(event, MessageSent):
                        return [f"out:{event.text}"]
                    return []

                r = Reducer("history", event_type=Event, fn=project, default=["start"])
                received_history = []

                @on(MessageReceived)
                def respond(event: MessageReceived, history: list) -> MessageSent:
                    received_history.extend(history)
                    return MessageSent(text=event.text.upper())

                @on(MessageSent)
                def finish(event: MessageSent) -> Completed:
                    return Completed(result=event.text)

                graph = EventGraph([respond, finish], reducers=[r])
                log = graph.invoke(MessageReceived(text="hello"))
                assert received_history == ["start", "in:hello"]
                assert log.latest(Completed) == Completed(result="HELLO")

            def it_injects_default_plus_projected_seed():
                r = Reducer("texts", event_type=MessageReceived, fn=lambda e: [e.text])

                @on(MessageReceived)
                def step(
                    event: MessageReceived,
                    log: EventLog,
                    texts: list,
                ) -> Completed:
                    return Completed(result=f"log={len(log)},texts={len(texts)}")

                graph = EventGraph([step], reducers=[r])
                log = graph.invoke(MessageReceived(text="hi"))
                assert log.latest(Completed) == Completed(result="log=1,texts=1")

        def describe_accumulation():

            def when_events_contribute():

                def it_grows_across_multiple_rounds():
                    class ToolResultReturned(Event):
                        result: str = ""

                    def project_all(event: Event) -> list:
                        if isinstance(event, MessageReceived):
                            return [f"in:{event.text}"]
                        if isinstance(event, MessageSent):
                            return [f"out:{event.text}"]
                        if isinstance(event, ToolResultReturned):
                            return [f"tool:{event.result}"]
                        return []

                    r = Reducer("history", event_type=Event, fn=project_all)
                    call_count = 0
                    snapshots: list[list] = []

                    @on(MessageReceived, ToolResultReturned)
                    def call_llm(event: Event, history: list) -> MessageSent:
                        nonlocal call_count
                        call_count += 1
                        snapshots.append(list(history))
                        if isinstance(event, MessageReceived):
                            return MessageSent(text="need_tool")
                        return MessageSent(text=f"final:{event.result}")

                    @on(MessageSent)
                    def handle_response(
                        event: MessageSent,
                    ) -> ToolResultReturned | Completed:
                        if event.text == "need_tool":
                            return ToolResultReturned(result="42")
                        return Completed(result=event.text)

                    graph = EventGraph([call_llm, handle_response], reducers=[r])
                    graph.invoke(MessageReceived(text="question"))
                    assert call_count == 2
                    assert snapshots[0] == ["in:question"]
                    assert snapshots[1] == [
                        "in:question",
                        "out:need_tool",
                        "tool:42",
                    ]

            def when_events_have_no_contribution():

                def it_does_not_change_reducer_value():
                    r = Reducer(
                        "texts",
                        event_type=MessageReceived,
                        fn=lambda e: [e.text],
                    )
                    snapshots: list[list] = []

                    @on(MessageReceived)
                    def respond(event: MessageReceived, texts: list) -> MessageSent:
                        snapshots.append(list(texts))
                        return MessageSent(text=event.text)

                    @on(MessageSent)
                    def finish(event: MessageSent, texts: list) -> Completed:
                        snapshots.append(list(texts))
                        return Completed(result="ok")

                    graph = EventGraph([respond, finish], reducers=[r])
                    graph.invoke(MessageReceived(text="a"))
                    assert snapshots[0] == ["a"]
                    assert snapshots[1] == ["a"]

        def describe_multiple_reducers():

            def it_accumulates_independently():
                def project_upper(event: MessageReceived) -> list:
                    return [event.text.upper()]

                def project_lower(event: MessageReceived) -> list:
                    return [event.text.lower()]

                upper = Reducer(
                    "upper",
                    event_type=MessageReceived,
                    fn=project_upper,
                    default=["INIT"],
                )
                lower = Reducer("lower", event_type=MessageReceived, fn=project_lower)

                @on(MessageReceived)
                def step(event: MessageReceived, upper: list, lower: list) -> Completed:
                    return Completed(result=f"upper={upper},lower={lower}")

                graph = EventGraph([step], reducers=[upper, lower])
                log = graph.invoke(MessageReceived(text="Hello"))
                assert log.latest(Completed) == Completed(
                    result="upper=['INIT', 'HELLO'],lower=['hello']"
                )

        def describe_parallel_handlers():

            def it_accepts_contributions_from_both():
                class Triggered(Event):
                    value: str = ""

                class ResultAProduced(Event):
                    value: str = ""

                class ResultBProduced(Event):
                    value: str = ""

                class Collected(Event):
                    items: tuple = ()

                def project(event: Event) -> list:
                    if isinstance(event, Triggered):
                        return [f"trigger:{event.value}"]
                    if isinstance(event, ResultAProduced):
                        return [f"a:{event.value}"]
                    if isinstance(event, ResultBProduced):
                        return [f"b:{event.value}"]
                    return []

                r = Reducer("items", event_type=Event, fn=project)

                @on(Triggered)
                def handle_a(event: Triggered) -> ResultAProduced:
                    return ResultAProduced(value=event.value)

                @on(Triggered)
                def handle_b(event: Triggered) -> ResultBProduced:
                    return ResultBProduced(value=event.value)

                @on(ResultAProduced, ResultBProduced)
                def collect(
                    event: Event, items: list, log: EventLog
                ) -> Collected | None:
                    if log.has(ResultAProduced) and log.has(ResultBProduced):
                        return Collected(items=tuple(items))
                    return None

                graph = EventGraph([handle_a, handle_b, collect], reducers=[r])
                log = graph.invoke(Triggered(value="x"))
                result = log.latest(Collected)
                assert result is not None
                assert "trigger:x" in result.items
                assert "a:x" in result.items
                assert "b:x" in result.items

        def describe_react_loop():

            def it_accumulates_system_user_assistant_tool_messages():
                class UserMsgReceived(Event):
                    content: str = ""

                class AssistantMsgSent(Event):
                    content: str = ""
                    needs_tool: bool = False

                class ToolResultReturned(Event):
                    result: str = ""

                class FinalAnswerProduced(Event):
                    answer: str = ""

                def to_messages(event: Event) -> list:
                    if isinstance(event, UserMsgReceived):
                        return [("user", event.content)]
                    if isinstance(event, AssistantMsgSent):
                        return [("assistant", event.content)]
                    if isinstance(event, ToolResultReturned):
                        return [("tool", event.result)]
                    return []

                r = Reducer(
                    "messages",
                    event_type=Event,
                    fn=to_messages,
                    default=[("system", "You are helpful")],
                )
                message_snapshots: list[list] = []

                @on(UserMsgReceived, ToolResultReturned)
                def call_llm(event: Event, messages: list) -> AssistantMsgSent:
                    message_snapshots.append(list(messages))
                    if isinstance(event, UserMsgReceived):
                        return AssistantMsgSent(content="need tool", needs_tool=True)
                    return AssistantMsgSent(
                        content=f"got:{event.result}",
                        needs_tool=False,
                    )

                @on(AssistantMsgSent)
                def handle_response(
                    event: AssistantMsgSent,
                ) -> ToolResultReturned | FinalAnswerProduced:
                    if event.needs_tool:
                        return ToolResultReturned(result="42")
                    return FinalAnswerProduced(answer=event.content)

                graph = EventGraph([call_llm, handle_response], reducers=[r])
                log = graph.invoke(UserMsgReceived(content="what is 6*7?"))
                assert log.latest(FinalAnswerProduced) == (
                    FinalAnswerProduced(answer="got:42")
                )
                assert message_snapshots[0] == [
                    ("system", "You are helpful"),
                    ("user", "what is 6*7?"),
                ]
                assert message_snapshots[1] == [
                    ("system", "You are helpful"),
                    ("user", "what is 6*7?"),
                    ("assistant", "need tool"),
                    ("tool", "42"),
                ]

        def describe_backward_compatibility():

            def it_handles_no_reducers():
                @on(MessageReceived)
                def step(event: MessageReceived) -> Completed:
                    return Completed(result=event.text)

                graph = EventGraph([step])
                log = graph.invoke(MessageReceived(text="hello"))
                assert log.latest(Completed) == Completed(result="hello")

            def it_coexists_alongside_event_log_injection():
                def project(event: Event) -> list:
                    return [1]

                r = Reducer("counter", event_type=Event, fn=project)

                @on(MessageReceived)
                def step(event: MessageReceived, log: EventLog) -> Completed:
                    return Completed(result=f"events={len(log)}")

                graph = EventGraph([step], reducers=[r])
                log = graph.invoke(MessageReceived(text="hi"))
                assert log.latest(Completed) == Completed(result="events=1")

        def describe_edge_cases():

            def when_fn_returns_non_list():

                def it_raises_type_error():
                    def bad_project(event: Event) -> list:
                        return "not a list"  # type: ignore

                    r = Reducer("bad", event_type=MessageReceived, fn=bad_project)

                    @on(MessageReceived)
                    def step(event: MessageReceived) -> Completed:
                        return Completed(result="ok")

                    graph = EventGraph([step], reducers=[r])
                    with pytest.raises(TypeError, match="must return a list"):
                        graph.invoke(MessageReceived(text="hello"))

            def when_custom_log_parameter_name():

                def it_supports_custom_log_parameter_name():
                    @on(MessageReceived)
                    def step(event: MessageReceived, event_log: EventLog) -> Completed:
                        return Completed(result=f"events={len(event_log)}")

                    graph = EventGraph([step])
                    log = graph.invoke(MessageReceived(text="hi"))
                    assert log.latest(Completed) == Completed(result="events=1")

            def when_checkpointer():

                def it_does_not_double_values_on_re_invoke():
                    from langgraph.checkpoint.memory import MemorySaver

                    r = Reducer(
                        "texts",
                        event_type=MessageReceived,
                        fn=lambda e: [e.text],
                        default=["init"],
                    )

                    @on(MessageReceived)
                    def step(event: MessageReceived, texts: list) -> Completed:
                        return Completed(result=",".join(texts))

                    graph = EventGraph([step], reducers=[r], checkpointer=MemorySaver())

                    config = {"configurable": {"thread_id": "reducer-test"}}
                    log = graph.invoke(MessageReceived(text="a"), config=config)
                    assert log.latest(Completed).result == "init,a"

                    log = graph.invoke(MessageReceived(text="b"), config=config)
                    assert log.latest(Completed).result == "init,a,b"

            def when_custom_reducer_function():

                def it_supports_custom_reducer_function():
                    def always_keep_last_n(left: list, right: list) -> list:
                        combined = left + right
                        return combined[-3:]

                    class Continued(Event):
                        text: str = ""

                    def project_all(event: Event) -> list:
                        if isinstance(event, MessageReceived):
                            return [event.text]
                        if isinstance(event, Continued):
                            return [event.text]
                        return []

                    r = Reducer(
                        "recent",
                        event_type=Event,
                        fn=project_all,
                        reducer=always_keep_last_n,
                        default=["x", "y", "z"],
                    )
                    snapshots: list[list] = []

                    @on(MessageReceived, Continued)
                    def step(event: Event, recent: list) -> MessageSent | Continued:
                        snapshots.append(list(recent))
                        if isinstance(event, MessageReceived):
                            return Continued(text="b")
                        return MessageSent(text="done")

                    @on(MessageSent)
                    def finish(event: MessageSent) -> Completed:
                        return Completed(result="ok")

                    graph = EventGraph([step, finish], reducers=[r])
                    graph.invoke(MessageReceived(text="a"))
                    assert snapshots[0] == ["y", "z", "a"]
                    assert snapshots[1] == ["z", "a", "b"]

            def when_pre_seeded_via_update_state():

                def _texts_reducer(**kw) -> Reducer:
                    return Reducer(
                        "texts", event_type=MessageReceived, fn=lambda e: [e.text], **kw
                    )

                def _texts_handler(trigger: type) -> tuple:
                    captured: list[list] = []

                    @on(trigger)
                    def _capture_texts(event: Event, texts: list) -> Completed:
                        captured.append(list(texts))
                        return Completed(result="ok")

                    return _capture_texts, captured

                def _pre_seed_graph(
                    handlers: list, reducers: list, seed_values: dict, thread_id: str
                ) -> tuple:
                    from langgraph.checkpoint.memory import MemorySaver

                    graph = EventGraph(
                        handlers, reducers=reducers, checkpointer=MemorySaver()
                    )
                    config: dict = {"configurable": {"thread_id": thread_id}}
                    graph.pre_seed(config, seed_values)
                    return graph, config

                def it_preserves_pre_seeded_list_reducer():
                    handler, captured = _texts_handler(Started)
                    graph, config = _pre_seed_graph(
                        [handler],
                        [_texts_reducer()],
                        {"texts": ["pre-seeded"]},
                        "pre-seed-list",
                    )
                    graph.invoke(Started(data="go"), config=config)
                    assert captured[0] == ["pre-seeded"]

                def it_preserves_pre_seeded_scalar_reducer():
                    captured: list[object] = []

                    @on(Started)
                    def step(event: Started, proposal: object) -> Completed:
                        captured.append(proposal)
                        return Completed(result="ok")

                    sr = ScalarReducer(
                        name="proposal", event_type=MessageReceived, fn=lambda e: e.text
                    )
                    graph, config = _pre_seed_graph(
                        [step],
                        [sr],
                        {"proposal": "my proposal text"},
                        "pre-seed-scalar",
                    )
                    graph.invoke(Started(data="go"), config=config)
                    assert captured[0] == "my proposal text"

                def it_preserves_pre_seeded_falsy_scalar():
                    captured: list[object] = []

                    @on(Started)
                    def step(event: Started, count: object) -> Completed:
                        captured.append(count)
                        return Completed(result="ok")

                    sr = ScalarReducer(
                        name="count",
                        event_type=MessageReceived,
                        fn=lambda e: int(e.text),
                    )
                    graph, config = _pre_seed_graph(
                        [step],
                        [sr],
                        {"count": 0},
                        "pre-seed-falsy",
                    )
                    graph.invoke(Started(data="go"), config=config)
                    assert captured[0] == 0

                def when_seed_event_also_contributes():

                    def it_merges_contributions_into_pre_seeded_list():
                        handler, captured = _texts_handler(MessageReceived)
                        graph, config = _pre_seed_graph(
                            [handler],
                            [_texts_reducer()],
                            {"texts": ["existing"]},
                            "merge-list",
                        )
                        graph.invoke(MessageReceived(text="new"), config=config)
                        assert captured[0] == ["existing", "new"]

                def when_reducer_has_non_empty_default():

                    def it_does_not_duplicate_default():
                        handler, captured = _texts_handler(Started)
                        graph, config = _pre_seed_graph(
                            [handler],
                            [_texts_reducer(default=["init"])],
                            {"texts": ["custom"]},
                            "no-dup-default",
                        )
                        graph.invoke(Started(data="go"), config=config)
                        # "init" default should NOT be re-applied on top
                        # of pre-seeded value.
                        assert captured[0] == ["custom"]

                def it_advances_cursor_after_pre_seeded_run():
                    handler, captured = _texts_handler(MessageReceived)
                    graph, config = _pre_seed_graph(
                        [handler],
                        [_texts_reducer()],
                        {"texts": ["pre"]},
                        "cursor-advance",
                    )
                    # Run 1 — pre-seeded
                    graph.invoke(MessageReceived(text="a"), config=config)
                    assert captured[0] == ["pre", "a"]

                    # Run 2 — re-invoke, cursor now > 0, normal resume
                    graph.invoke(MessageReceived(text="b"), config=config)
                    assert captured[1] == ["pre", "a", "b"]

                def it_handles_mixed_pre_seeded_and_normal_reducers():
                    captured_seeded: list[list] = []
                    captured_normal: list[list] = []

                    @on(Started)
                    def step(event: Started, seeded: list, normal: list) -> Completed:
                        captured_seeded.append(list(seeded))
                        captured_normal.append(list(normal))
                        return Completed(result="ok")

                    r_seeded = Reducer(
                        "seeded", event_type=MessageReceived, fn=lambda e: [e.text]
                    )
                    r_normal = Reducer(
                        "normal",
                        event_type=Started,
                        fn=lambda e: [e.data],
                        default=["init"],
                    )
                    graph, config = _pre_seed_graph(
                        [step],
                        [r_seeded, r_normal],
                        {"seeded": ["external"]},
                        "mixed",
                    )
                    graph.invoke(Started(data="go"), config=config)
                    assert captured_seeded[0] == ["external"]
                    assert captured_normal[0] == ["init", "go"]

                @pytest.mark.asyncio
                async def it_supports_async_apre_seed():
                    from langgraph.checkpoint.memory import MemorySaver

                    handler, captured = _texts_handler(Started)
                    graph = EventGraph(
                        [handler],
                        reducers=[_texts_reducer()],
                        checkpointer=MemorySaver(),
                    )
                    config: dict = {"configurable": {"thread_id": "pre-seed-async"}}
                    await graph.apre_seed(config, {"texts": ["pre-seeded"]})
                    await graph.ainvoke(Started(data="go"), config=config)
                    assert captured[0] == ["pre-seeded"]

    def describe_scalar_reducer():

        def when_matching_events():

            def it_injects_last_value():
                class StrategyChosen(Event):
                    strategy: str = ""

                class TaskDone(Event):
                    result: str = ""

                sr = ScalarReducer(
                    name="strategy",
                    event_type=StrategyChosen,
                    fn=lambda e: e.strategy,
                )

                @on(StrategyChosen)
                def handle(event: StrategyChosen, strategy: str) -> TaskDone:
                    return TaskDone(result=f"used:{strategy}")

                graph = EventGraph([handle], reducers=[sr])
                log = graph.invoke(StrategyChosen(strategy="aggressive"))
                assert log.latest(TaskDone) == TaskDone(result="used:aggressive")

            def it_takes_last_matching_value():
                class StepCompleted(Event):
                    value: str = ""

                class Finalized(Event):
                    result: str = ""

                sr = ScalarReducer(
                    name="chosen",
                    event_type=StepCompleted,
                    fn=lambda e: e.value,
                )

                @on(StepCompleted)
                def advance(
                    event: StepCompleted,
                    chosen: object,
                ) -> StepCompleted | Finalized:
                    if event.value == "b":
                        return Finalized(result=f"chosen={chosen}")
                    return StepCompleted(value="b")

                graph = EventGraph([advance], reducers=[sr])
                log = graph.invoke(StepCompleted(value="a"))
                # After seed "a", handler sees "a"; produces StepCompleted("b"),
                # then handler sees "b" (last non-None wins).
                assert log.latest(Finalized) == Finalized(result="chosen=b")

            def it_collects_from_last_matching_event():
                class StepCompleted(Event):
                    tag: str = ""

                sr = ScalarReducer(
                    name="val",
                    event_type=StepCompleted,
                    fn=lambda e: e.tag,
                )
                events = [
                    StepCompleted(tag="a"),
                    StepCompleted(tag="b"),
                    StepCompleted(tag="c"),
                ]
                result = sr.collect(events)
                assert result == "c"

        def when_no_matching_events():

            def it_defaults_to_none():
                class Triggered(Event):
                    pass

                class Unmatched(Event):
                    pass

                class ResultProduced(Event):
                    got: str = ""

                sr = ScalarReducer(
                    name="mode",
                    event_type=Unmatched,
                    fn=lambda e: "irrelevant",
                )

                @on(Triggered)
                def handle(event: Triggered, mode: object) -> ResultProduced:
                    return ResultProduced(got=str(mode))

                graph = EventGraph([handle], reducers=[sr])
                log = graph.invoke(Triggered())
                assert log.latest(ResultProduced) == ResultProduced(got="None")

            def it_returns_skip():
                class StepCompleted(Event):
                    pass

                class OtherReceived(Event):
                    pass

                from langgraph_events import SKIP

                sr = ScalarReducer(
                    name="val", event_type=OtherReceived, fn=lambda e: "x"
                )
                assert (
                    sr.collect([StepCompleted(), StepCompleted(), StepCompleted()])
                    is SKIP
                )

            def it_treats_skip_from_fn_as_no_contribution():
                class Triggered(Event):
                    pass

                from langgraph_events import SKIP

                sr = ScalarReducer(
                    name="mode",
                    event_type=Triggered,
                    fn=lambda e: SKIP,
                    default="fallback",
                )

                result = sr.collect([Triggered()])
                assert result is SKIP
                assert sr.has_contributions(result) is False
                assert sr.seed([Triggered()]) == "fallback"

            def it_uses_custom_default():
                class Triggered(Event):
                    pass

                class Unmatched(Event):
                    pass

                class ResultProduced(Event):
                    got: str = ""

                sr = ScalarReducer(
                    name="mode",
                    event_type=Unmatched,
                    fn=lambda e: "irrelevant",
                    default="fallback",
                )

                @on(Triggered)
                def handle(event: Triggered, mode: str) -> ResultProduced:
                    return ResultProduced(got=mode)

                graph = EventGraph([handle], reducers=[sr])
                log = graph.invoke(Triggered())
                assert log.latest(ResultProduced) == ResultProduced(got="fallback")

        def when_mixed_list_reducers():

            def it_works_alongside_list_reducers():
                class Triggered(Event):
                    tag: str = ""

                class ResultProduced(Event):
                    summary: str = ""

                list_r = Reducer(
                    name="tags",
                    event_type=Triggered,
                    fn=lambda e: [e.tag] if e.tag else [],
                )
                scalar_r = ScalarReducer(
                    name="last_tag",
                    event_type=Triggered,
                    fn=lambda e: e.tag,
                )

                @on(Triggered)
                def handle(
                    event: Triggered,
                    tags: list,
                    last_tag: object,
                ) -> ResultProduced:
                    return ResultProduced(summary=f"tags={tags},last={last_tag}")

                graph = EventGraph([handle], reducers=[list_r, scalar_r])
                log = graph.invoke(Triggered(tag="x"))
                assert log.latest(ResultProduced) == (
                    ResultProduced(summary="tags=['x'],last=x")
                )

        def when_parallel_handlers():

            def it_handles_parallel_handler_contributions():
                class Triggered(Event):
                    value: str = ""

                class ResultAProduced(Event):
                    data: str = ""

                class ResultBProduced(Event):
                    data: str = ""

                sr = ScalarReducer(
                    name="latest",
                    event_type=Event,
                    fn=lambda e: (
                        e.value
                        if isinstance(e, Triggered)
                        else (
                            e.data
                            if isinstance(e, (ResultAProduced, ResultBProduced))
                            else None
                        )
                    ),
                )

                @on(Triggered)
                def handler_a(event: Triggered, latest: object) -> ResultAProduced:
                    return ResultAProduced(data=f"a:{event.value}")

                @on(Triggered)
                def handler_b(event: Triggered, latest: object) -> ResultBProduced:
                    return ResultBProduced(data=f"b:{event.value}")

                graph = EventGraph([handler_a, handler_b], reducers=[sr])
                log = graph.invoke(Triggered(value="x"))
                # Both handlers run in parallel — should not crash
                assert log.has(ResultAProduced)
                assert log.has(ResultBProduced)

        def when_subsequent_round_has_no_contribution():

            def it_persists_value():
                class ValueSet(Event):
                    value: str = ""

                class UnrelatedReceived(Event):
                    pass

                class ResultProduced(Event):
                    got: str = ""

                sr = ScalarReducer(
                    name="kept",
                    event_type=ValueSet,
                    fn=lambda e: e.value,
                )

                @on(ValueSet)
                def step1(event: ValueSet) -> UnrelatedReceived:
                    return UnrelatedReceived()

                @on(UnrelatedReceived)
                def step2(event: UnrelatedReceived, kept: object) -> ResultProduced:
                    return ResultProduced(got=str(kept))

                graph = EventGraph([step1, step2], reducers=[sr])
                log = graph.invoke(ValueSet(value="hello"))
                # Round 2 produces UnrelatedReceived (doesn't match event_type) —
                # scalar must still be "hello", not reverted.
                assert log.latest(ResultProduced) == ResultProduced(got="hello")

        def when_fn_returns_none():

            def it_stores_none_as_valid_contribution():
                class ClearSignaled(Event):
                    pass

                class ResultProduced(Event):
                    got: str = ""

                sr = ScalarReducer(
                    name="value",
                    event_type=ClearSignaled,
                    fn=lambda e: None,
                    default="initial",
                )

                @on(ClearSignaled)
                def handle(event: ClearSignaled, value: object) -> ResultProduced:
                    return ResultProduced(got=repr(value))

                graph = EventGraph([handle], reducers=[sr])
                log = graph.invoke(ClearSignaled())
                # fn returns None — this is a real contribution, not "no contribution"
                assert log.latest(ResultProduced) == ResultProduced(got="None")

        def when_protocol_event_type():

            def it_supports_protocol_event_type():
                from typing import Protocol, runtime_checkable

                @runtime_checkable
                class HasScore(Protocol):
                    score: int

                class ScoreARecorded(Event):
                    score: int = 0

                class ScoreBRecorded(Event):
                    score: int = 0

                class ResultProduced(Event):
                    got: str = ""

                sr = ScalarReducer(
                    name="last_score",
                    event_type=HasScore,
                    fn=lambda e: e.score,
                )

                @on(ScoreARecorded)
                def step_a(event: ScoreARecorded, last_score: object) -> ScoreBRecorded:
                    return ScoreBRecorded(score=event.score + 10)

                @on(ScoreBRecorded)
                def step_b(event: ScoreBRecorded, last_score: object) -> ResultProduced:
                    return ResultProduced(got=str(last_score))

                graph = EventGraph([step_a, step_b], reducers=[sr])
                log = graph.invoke(ScoreARecorded(score=5))
                # ScoreA(5) → 5, then ScoreB(15) → 15
                assert log.latest(ResultProduced) == ResultProduced(got="15")

        def when_checkpointer():

            def it_does_not_lose_scalar_on_re_invoke():
                from langgraph.checkpoint.memory import MemorySaver

                class Triggered(Event):
                    value: str = ""

                class ResultProduced(Event):
                    got: str = ""

                sr = ScalarReducer(
                    name="latest",
                    event_type=Triggered,
                    fn=lambda e: e.value,
                )

                @on(Triggered)
                def handle(event: Triggered, latest: object) -> ResultProduced:
                    return ResultProduced(got=str(latest))

                graph = EventGraph([handle], reducers=[sr], checkpointer=MemorySaver())
                config = {"configurable": {"thread_id": "scalar-re-invoke"}}

                # Run 1
                log1 = graph.invoke(Triggered(value="first"), config=config)
                assert log1.latest(ResultProduced) == ResultProduced(got="first")

                # Run 2 — re-invoke on same thread
                log2 = graph.invoke(Triggered(value="second"), config=config)
                assert log2.latest(ResultProduced) == ResultProduced(got="second")

    def describe_message_reducer():

        def when_defaults_provided():

            def it_projects_message_events():
                class UserMsgReceived(MessageEvent):
                    message: HumanMessage = None  # type: ignore[assignment]

                class Replied(Event):
                    text: str = ""

                r = message_reducer([SystemMessage(content="You are helpful")])
                msg = HumanMessage(content="hello")
                result = r.fn(UserMsgReceived(message=msg))
                assert result == [msg]

            def it_skips_non_message_events_at_collect_level():
                class Replied(Event):
                    text: str = ""

                r = message_reducer([SystemMessage(content="You are helpful")])
                result = r.collect([Replied(text="hi")])
                assert result == []

            def it_includes_default_messages():
                r = message_reducer([SystemMessage(content="Be nice")])
                assert len(r.default) == 1
                assert r.default[0].content == "Be nice"

        def when_none_given():

            def it_has_empty_default():
                r = message_reducer()
                assert r.default == []

        def when_custom_channel_name():

            def it_respects_custom_channel_name():
                r = message_reducer(name="chat_history")
                assert r.name == "chat_history"

        def describe_integration():

            def when_default_system_message():

                def it_accumulates_system_and_user_messages():
                    class UserMsgReceived(MessageEvent):
                        message: HumanMessage = None  # type: ignore[assignment]

                    class BotReplied(MessageEvent):
                        message: AIMessage = None  # type: ignore[assignment]

                    class Finished(Event):
                        answer: str = ""

                    r = message_reducer([SystemMessage(content="You are a test bot")])
                    received_messages: list[list[BaseMessage]] = []

                    @on(UserMsgReceived)
                    def respond(
                        event: UserMsgReceived,
                        messages: list[BaseMessage],
                    ) -> BotReplied:
                        received_messages.append(list(messages))
                        return BotReplied(
                            message=AIMessage(content="I got: " + event.message.content)
                        )

                    @on(BotReplied)
                    def finish(event: BotReplied) -> Finished:
                        return Finished(answer=event.message.content)

                    graph = EventGraph([respond, finish], reducers=[r])
                    log = graph.invoke(
                        UserMsgReceived(message=HumanMessage(content="hello"))
                    )
                    assert log.latest(Finished) == Finished(answer="I got: hello")
                    msgs = received_messages[0]
                    assert len(msgs) == 2
                    assert msgs[0].content == "You are a test bot"
                    assert msgs[1].content == "hello"

            def when_system_prompt_set_seed():

                def it_contributes_to_message_history():
                    class UserMsgReceived(MessageEvent):
                        message: HumanMessage = None  # type: ignore[assignment]

                    class Finished(Event):
                        answer: str = ""

                    r = message_reducer()
                    received_messages: list[list[BaseMessage]] = []

                    @on(UserMsgReceived)
                    def respond(
                        event: UserMsgReceived,
                        messages: list[BaseMessage],
                    ) -> Finished:
                        received_messages.append(list(messages))
                        return Finished(answer="ok")

                    graph = EventGraph([respond], reducers=[r])
                    log = graph.invoke(
                        [
                            SystemPromptSet.from_str("You are a test bot"),
                            UserMsgReceived(message=HumanMessage(content="hello")),
                        ]
                    )
                    assert log.latest(Finished) is not None
                    msgs = received_messages[0]
                    assert len(msgs) == 2
                    assert isinstance(msgs[0], SystemMessage)
                    assert msgs[0].content == "You are a test bot"
                    assert msgs[1].content == "hello"

                def it_is_queryable_in_event_log():
                    class UserMsgReceived(MessageEvent):
                        message: HumanMessage = None  # type: ignore[assignment]

                    class Finished(Event):
                        prompt_content: str = ""

                    r = message_reducer()

                    @on(UserMsgReceived)
                    def respond(event: UserMsgReceived, log: EventLog) -> Finished:
                        prompt = log.latest(SystemPromptSet)
                        return Finished(
                            prompt_content=(
                                prompt.message.content if prompt else "none"
                            )
                        )

                    graph = EventGraph([respond], reducers=[r])
                    log = graph.invoke(
                        [
                            SystemPromptSet.from_str("You are helpful"),
                            UserMsgReceived(message=HumanMessage(content="hi")),
                        ]
                    )
                    assert log.latest(Finished) == Finished(
                        prompt_content="You are helpful"
                    )

    def describe_compiled():

        def when_no_checkpointer():

            def it_returns_same_instance():
                @on(Started)
                def step(event: Started) -> Ended:
                    return Ended(result=event.data)

                graph = EventGraph([step])
                first = graph.compiled
                second = graph.compiled
                assert first is second

            def it_returns_cached_instance_on_second_call():
                @on(Started)
                def step(event: Started) -> Ended:
                    return Ended(result=event.data)

                graph = EventGraph([step])
                first = graph.compiled
                second = graph.compiled
                assert first is second

        def when_checkpointer():

            def it_persists_state():
                from langgraph.checkpoint.memory import MemorySaver

                @on(Started)
                def step(event: Started) -> Ended:
                    return Ended(result=event.data)

                graph = EventGraph([step], checkpointer=MemorySaver())

                config = {"configurable": {"thread_id": "test-1"}}
                log = graph.invoke(Started(data="hello"), config=config)
                assert log[-1] == Ended(result="hello")

                state = graph.get_state(config)
                assert len(state.events) == 2

            def it_only_processes_new_events_on_re_invoke():
                from langgraph.checkpoint.memory import MemorySaver

                seen: list[list[str]] = []

                @on(Started)
                def step(event: Started) -> Ended:
                    seen.append([event.data])
                    return Ended(result=event.data)

                graph = EventGraph([step], checkpointer=MemorySaver())
                config = {"configurable": {"thread_id": "re-invoke-1"}}

                # Run 1
                graph.invoke(Started(data="a"), config=config)
                assert len(seen) == 1
                assert seen[-1] == ["a"]

                # Run 2 — same thread, only Started("b") should be pending
                graph.invoke(Started(data="b"), config=config)
                assert len(seen) == 2
                assert seen[-1] == ["b"]

            def it_handles_three_sequential_re_invokes():
                from langgraph.checkpoint.memory import MemorySaver

                @on(Started)
                def step(event: Started) -> Ended:
                    return Ended(result=event.data)

                graph = EventGraph([step], checkpointer=MemorySaver())
                config = {"configurable": {"thread_id": "re-invoke-3"}}

                graph.invoke(Started(data="first"), config=config)
                graph.invoke(Started(data="second"), config=config)
                log = graph.invoke(Started(data="third"), config=config)

                # Finalized result only reflects third run's input
                assert log[-1] == Ended(result="third")

                # Full state has all 6 events (3 Started + 3 Ended)
                state = graph.get_state(config)
                assert len(state.events) == 6

    def describe_stream_events():

        def when_default():

            def it_yields_event_objects():
                @on(Started)
                def step1(event: Started) -> Processed:
                    return Processed(data=event.data)

                @on(Processed)
                def step2(event: Processed) -> Ended:
                    return Ended(result=event.data)

                graph = EventGraph([step1, step2])
                events = list(graph.stream_events(Started(data="hi")))
                assert all(isinstance(e, Event) for e in events)
                types = [type(e).__name__ for e in events]
                assert "Started" in types
                assert "Processed" in types
                assert "Ended" in types

            def it_yields_events_in_order():
                @on(Started)
                def step1(event: Started) -> Processed:
                    return Processed(data="mid")

                @on(Processed)
                def step2(event: Processed) -> Ended:
                    return Ended(result="done")

                graph = EventGraph([step1, step2])
                events = list(graph.stream_events(Started(data="go")))
                assert isinstance(events[0], Started)
                assert isinstance(events[-1], Ended)

        def when_multi_seed():

            def it_includes_all_seed_types():
                @on(Started)
                def step(event: Started) -> Ended:
                    return Ended(result=event.data)

                graph = EventGraph([step])
                events = list(graph.stream_events([Started(data="a")]))
                types = [type(e).__name__ for e in events]
                assert "Started" in types
                assert "Ended" in types

        def when_include_reducers_true():

            def it_yields_stream_frames():
                reducer = _data_reducer()

                @on(Started)
                def step1(event: Started) -> Processed:
                    return Processed(data=event.data)

                @on(Processed)
                def step2(event: Processed) -> Ended:
                    return Ended(result=event.data)

                graph = EventGraph([step1, step2], reducers=[reducer])
                frames = list(
                    graph.stream_events(
                        Started(data="hello"),
                        include_reducers=True,
                    )
                )
                assert all(isinstance(f, StreamFrame) for f in frames)
                # values-mode frames (sync API) do not track reducer deltas
                assert all(f.changed_reducers is None for f in frames)
                types = [type(f.event).__name__ for f in frames]
                assert "Started" in types
                assert "Processed" in types
                assert "Ended" in types
                seed_frame = next(f for f in frames if isinstance(f.event, Started))
                assert "data_items" in seed_frame.reducers
                assert "hello" in seed_frame.reducers["data_items"]

        def when_include_reducers_selective():

            def it_only_includes_named_reducers():
                reducer = _data_reducer()

                @on(Started)
                def step(event: Started) -> Ended:
                    return Ended(result=event.data)

                graph = EventGraph([step], reducers=[reducer])
                frames = list(
                    graph.stream_events(
                        Started(data="x"),
                        include_reducers=["data_items"],
                    )
                )
                assert all(isinstance(f, StreamFrame) for f in frames)
                assert "data_items" in frames[0].reducers

        def when_include_reducers_partial_overlap():

            def it_includes_only_valid_reducer_names():
                reducer = _data_reducer()

                @on(Started)
                def step(event: Started) -> Ended:
                    return Ended(result=event.data)

                graph = EventGraph([step], reducers=[reducer])
                frames = list(
                    graph.stream_events(
                        Started(data="x"),
                        include_reducers=["data_items", "nonexistent"],
                    )
                )
                assert all(isinstance(f, StreamFrame) for f in frames)
                # Only the valid reducer appears in snapshots
                assert "data_items" in frames[0].reducers
                assert "nonexistent" not in frames[0].reducers

        def when_include_reducers_unknown_name():

            def it_warns_about_unknown_reducer_names():
                reducer = _data_reducer()

                @on(Started)
                def step(event: Started) -> Ended:
                    return Ended(result=event.data)

                graph = EventGraph([step], reducers=[reducer])
                with pytest.warns(
                    UserWarning,
                    match="Unknown reducer name.*nonexistent",
                ):
                    list(
                        graph.stream_events(
                            Started(data="x"),
                            include_reducers=["nonexistent"],
                        )
                    )

            def it_falls_back_to_bare_events():
                reducer = _data_reducer()

                @on(Started)
                def step(event: Started) -> Ended:
                    return Ended(result=event.data)

                graph = EventGraph([step], reducers=[reducer])
                frames = list(
                    graph.stream_events(
                        Started(data="x"),
                        include_reducers=["nonexistent"],
                    )
                )
                assert all(isinstance(f, Event) for f in frames)

        def when_include_reducers_false():

            def it_yields_bare_event_objects():
                @on(Started)
                def step(event: Started) -> Ended:
                    return Ended(result=event.data)

                graph = EventGraph([step])
                events = list(graph.stream_events(Started(data="hi")))
                assert all(isinstance(e, Event) for e in events)
                assert not any(isinstance(e, StreamFrame) for e in events)

        def when_async():

            @pytest.mark.asyncio
            async def it_yields_stream_frames():
                reducer = _data_reducer()

                @on(Started)
                def step1(event: Started) -> Processed:
                    return Processed(data=event.data)

                @on(Processed)
                def step2(event: Processed) -> Ended:
                    return Ended(result=event.data)

                graph = EventGraph([step1, step2], reducers=[reducer])
                frames = [
                    f
                    async for f in graph.astream_events(
                        Started(data="async"),
                        include_reducers=True,
                    )
                ]
                assert all(isinstance(f, StreamFrame) for f in frames)
                types = [type(f.event).__name__ for f in frames]
                assert "Started" in types
                assert "Ended" in types
                seed_frame = next(f for f in frames if isinstance(f.event, Started))
                assert "async" in seed_frame.reducers["data_items"]

        def when_reducer_accumulation():

            def it_accumulates_reducer_values_across_events():
                reducer = _data_reducer()

                class StartedA(Started):
                    pass

                class StartedB(Started):
                    pass

                @on(StartedA)
                def step_a(event: StartedA) -> StartedB:
                    return StartedB(data=f"b_from_{event.data}")

                @on(StartedB)
                def step_b(event: StartedB) -> Ended:
                    return Ended(result=event.data)

                graph = EventGraph([step_a, step_b], reducers=[reducer])
                frames = list(
                    graph.stream_events(StartedA(data="a1"), include_reducers=True)
                )
                last_frame = frames[-1]
                data_items = last_frame.reducers["data_items"]
                assert "a1" in data_items
                assert "b_from_a1" in data_items

    def describe_stream_resume():

        def it_yields_resume_handler_events():
            from langgraph.checkpoint.memory import MemorySaver

            @on(Started)
            def need_input(event: Started) -> _StepInterrupted:
                return _StepInterrupted(step=1)

            @on(Completed)
            def finish(event: Completed) -> Ended:
                return Ended(result=event.result)

            graph = EventGraph(
                [need_input, finish],
                checkpointer=MemorySaver(),
            )
            config = {"configurable": {"thread_id": "sr-handler"}}
            graph.invoke(Started(data="go"), config=config)

            events = list(graph.stream_resume(Completed(result="done"), config=config))
            types = [type(e).__name__ for e in events]
            assert "Ended" in types

        def it_includes_stale_interrupted_in_raw_stream():
            from langgraph.checkpoint.memory import MemorySaver

            @on(Started)
            def need_input(event: Started) -> _StepInterrupted:
                return _StepInterrupted(step=1)

            @on(Completed)
            def finish(event: Completed) -> Ended:
                return Ended(result=event.result)

            graph = EventGraph(
                [need_input, finish],
                checkpointer=MemorySaver(),
            )
            config = {"configurable": {"thread_id": "sr-no-stale"}}
            graph.invoke(Started(data="go"), config=config)

            events = list(graph.stream_resume(Completed(result="done"), config=config))
            # Raw stream_resume is semantically complete — stale Interrupted appears
            assert any(isinstance(e, Interrupted) for e in events)

        def it_yields_reducer_stream_frames():
            from langgraph.checkpoint.memory import MemorySaver

            @on(Started)
            def need_input(event: Started) -> _StepInterrupted:
                return _StepInterrupted(step=1)

            @on(Completed)
            def finish(event: Completed) -> Ended:
                return Ended(result=event.result)

            reducer = _data_reducer()
            graph = EventGraph(
                [need_input, finish],
                checkpointer=MemorySaver(),
                reducers=[reducer],
            )
            config = {"configurable": {"thread_id": "sr-reducers"}}
            graph.invoke(Started(data="go"), config=config)

            frames = list(
                graph.stream_resume(
                    Completed(result="done"),
                    include_reducers=True,
                    config=config,
                )
            )
            assert all(isinstance(f, StreamFrame) for f in frames)
            assert any("data_items" in f.reducers for f in frames)

        @pytest.mark.asyncio
        async def it_yields_resume_events_async():
            from langgraph.checkpoint.memory import MemorySaver

            @on(Started)
            def need_input(event: Started) -> _StepInterrupted:
                return _StepInterrupted(step=1)

            @on(Completed)
            def finish(event: Completed) -> Ended:
                return Ended(result=event.result)

            graph = EventGraph(
                [need_input, finish],
                checkpointer=MemorySaver(),
            )
            config = {"configurable": {"thread_id": "sr-async"}}
            await graph.ainvoke(Started(data="go"), config=config)

            events = [
                e
                async for e in graph.astream_resume(
                    Completed(result="async-done"), config=config
                )
            ]
            types = [type(e).__name__ for e in events]
            assert "Ended" in types

        def it_yields_new_interrupted_during_resume():
            from langgraph.checkpoint.memory import MemorySaver

            @on(Started)
            def step_one(event: Started) -> _StepInterrupted:
                return _StepInterrupted(step=1)

            @on(Completed)
            def step_two(event: Completed) -> _StepInterrupted:
                return _StepInterrupted(step=2)

            graph = EventGraph(
                [step_one, step_two],
                checkpointer=MemorySaver(),
            )
            config = {"configurable": {"thread_id": "sr-new-interrupt"}}
            graph.invoke(Started(data="go"), config=config)

            events = list(graph.stream_resume(Completed(result="next"), config=config))
            # Stale step=1 interrupt appears in raw stream (not filtered)
            assert any(isinstance(e, _StepInterrupted) and e.step == 1 for e in events)
            # New interrupt (step=2) is in checkpoint tasks, detectable post-stream
            snapshot = graph.compiled.get_state(config)
            assert snapshot.next  # graph is still interrupted
            interrupt_values = [
                getattr(intr, "value", None)
                for task in snapshot.tasks
                for intr in getattr(task, "interrupts", ())
            ]
            assert any(
                isinstance(v, _StepInterrupted) and v.step == 2
                for v in interrupt_values
            )

        @pytest.mark.asyncio
        async def it_preserves_reducer_state_from_checkpoint_in_v2():
            from langgraph.checkpoint.memory import MemorySaver

            r = Reducer("texts", event_type=MessageReceived, fn=lambda e: [e.text])

            @on(Started)
            def need_input(event: Started) -> _StepInterrupted:
                return _StepInterrupted(step=1)

            @on(Completed)
            def finish(event: Completed) -> Ended:
                return Ended(result=event.result)

            graph = EventGraph(
                [need_input, finish],
                checkpointer=MemorySaver(),
                reducers=[r],
            )
            config = {"configurable": {"thread_id": "v2-resume-reducer"}}

            # First run — seed with MessageReceived to populate reducer, then
            # interrupt via Started → _StepInterrupted.
            graph.invoke(
                [MessageReceived(text="hello"), Started(data="go")], config=config
            )

            # Resume via _astream_v2 path (include_custom_events forces v2)
            frames = [
                item
                async for item in graph.astream_resume(
                    Completed(result="done"),
                    include_reducers=True,
                    include_custom_events=True,
                    config=config,
                )
            ]

            stream_frames = [f for f in frames if isinstance(f, StreamFrame)]
            assert len(stream_frames) > 0
            # Reducer must reflect checkpoint state ("hello" from first run)
            assert "hello" in stream_frames[0].reducers["texts"]

        @pytest.mark.asyncio
        async def it_accumulates_reducer_across_v2_astream_events_runs():
            from langgraph.checkpoint.memory import MemorySaver

            r = Reducer("texts", event_type=MessageReceived, fn=lambda e: [e.text])

            @on(MessageReceived)
            def step(event: MessageReceived) -> Completed:
                return Completed(result=event.text)

            graph = EventGraph(
                [step],
                checkpointer=MemorySaver(),
                reducers=[r],
            )
            config = {"configurable": {"thread_id": "v2-second-run"}}

            # First run via astream_events (v2 path)
            _ = [
                item
                async for item in graph.astream_events(
                    MessageReceived(text="first"),
                    include_reducers=True,
                    include_custom_events=True,
                    config=config,
                )
            ]

            # Second run on same thread — seed contributes on top of checkpoint
            frames = [
                item
                async for item in graph.astream_events(
                    MessageReceived(text="second"),
                    include_reducers=True,
                    include_custom_events=True,
                    config=config,
                )
            ]

            stream_frames = [f for f in frames if isinstance(f, StreamFrame)]
            assert len(stream_frames) > 0
            texts = stream_frames[0].reducers["texts"]
            assert "first" in texts
            assert "second" in texts

        @pytest.mark.asyncio
        async def it_preserves_scalar_reducer_from_checkpoint_in_v2():
            from langgraph.checkpoint.memory import MemorySaver

            sr = ScalarReducer(
                name="proposal",
                event_type=MessageReceived,
                fn=lambda e: e.text,
            )

            @on(Started)
            def need_input(event: Started) -> _StepInterrupted:
                return _StepInterrupted(step=1)

            @on(Completed)
            def finish(event: Completed) -> Ended:
                return Ended(result=event.result)

            graph = EventGraph(
                [need_input, finish],
                checkpointer=MemorySaver(),
                reducers=[sr],
            )
            config = {"configurable": {"thread_id": "v2-scalar-resume"}}

            # First run — MessageReceived populates scalar, Started interrupts
            graph.invoke(
                [MessageReceived(text="chosen"), Started(data="go")], config=config
            )

            # Resume via v2 path
            frames = [
                item
                async for item in graph.astream_resume(
                    Completed(result="done"),
                    include_reducers=True,
                    include_custom_events=True,
                    config=config,
                )
            ]

            stream_frames = [f for f in frames if isinstance(f, StreamFrame)]
            assert len(stream_frames) > 0
            assert stream_frames[0].reducers["proposal"] == "chosen"

    def describe_reflection_loop():

        class WriteRequested(Event):
            topic: str = ""
            max_revisions: int = 3

        class DraftProduced(Event):
            content: str = ""
            revision: int = 0

        class CritiqueReceived(Event):
            draft: str = ""
            feedback: str = ""
            revision: int = 0

        class FinalDraftProduced(Event):
            content: str = ""

        def it_terminates_at_max_revisions():
            @on(WriteRequested, CritiqueReceived)
            def generate(event: Event, log: EventLog) -> DraftProduced:
                if isinstance(event, CritiqueReceived):
                    return DraftProduced(
                        content=f"revised({event.draft})",
                        revision=event.revision + 1,
                    )
                return DraftProduced(content=f"first_draft({event.topic})")

            @on(DraftProduced)
            def evaluate(
                event: DraftProduced,
                log: EventLog,
            ) -> CritiqueReceived | FinalDraftProduced:
                request = log.latest(WriteRequested)
                if event.revision >= request.max_revisions:
                    return FinalDraftProduced(content=event.content)
                return CritiqueReceived(
                    draft=event.content,
                    feedback="needs work",
                    revision=event.revision,
                )

            graph = EventGraph([generate, evaluate])
            log = graph.invoke(WriteRequested(topic="AI", max_revisions=2))
            assert log.has(FinalDraftProduced)
            final = log.latest(FinalDraftProduced)
            assert "revised" in final.content
            drafts = log.filter(DraftProduced)
            assert len(drafts) == 3

        def it_exits_early_on_pass():
            @on(WriteRequested, CritiqueReceived)
            def generate(event: Event) -> DraftProduced:
                return DraftProduced(content="perfect", revision=0)

            @on(DraftProduced)
            def evaluate(event: DraftProduced) -> CritiqueReceived | FinalDraftProduced:
                return FinalDraftProduced(content=event.content)

            graph = EventGraph([generate, evaluate])
            log = graph.invoke(WriteRequested(topic="test"))
            assert log.latest(FinalDraftProduced) == (
                FinalDraftProduced(content="perfect")
            )
            assert len(log.filter(DraftProduced)) == 1

    def describe_safety():

        def describe_return_type_enforcement():

            def it_rejects_list_return():
                @on(Started)
                def bad_handler(event: Started):
                    return [Processed(data="a"), Processed(data="b")]

                graph = EventGraph([bad_handler])
                with pytest.raises(TypeError, match="never a list"):
                    graph.invoke(Started(data="test"))

        def describe_max_rounds():

            def it_detects_infinite_loop():
                class LoopDetected(Event):
                    n: int = 0

                @on(LoopDetected)
                def looper(event: LoopDetected) -> LoopDetected:
                    return LoopDetected(n=event.n + 1)

                graph = EventGraph([looper], max_rounds=5)
                log = graph.invoke(LoopDetected(n=0))
                assert log.latest(MaxRoundsExceeded) is not None

            def it_resets_round_counter_on_resume():
                from langgraph.checkpoint.memory import MemorySaver

                class ResumeConfirmed(Event):
                    pass

                @on(Started)
                def ask(event: Started) -> _StepInterrupted:
                    return _StepInterrupted()

                @on(Resumed)
                def after_resume(event: Resumed) -> _StepInterrupted | Ended:
                    prev = event.interrupted
                    step = prev.step if isinstance(prev, _StepInterrupted) else 0
                    if step >= 2:
                        return Ended(result="done")
                    return _StepInterrupted(step=step + 1)

                # max_rounds=2 would be exceeded without reset:
                # Run 1 uses round 1 (seed→ask), then pauses.
                # Run 2: resume resets to 1, after_resume uses round 2,
                #   then pauses again — OK with reset but would be
                #   round 3 without it.
                graph = EventGraph(
                    [ask, after_resume],
                    max_rounds=2,
                    checkpointer=MemorySaver(),
                )
                config = {"configurable": {"thread_id": "resume-rounds"}}

                # Run 1: Started → Interrupted (pause)
                graph.invoke(Started(data="go"), config=config)

                # Run 2: resume → Interrupted (round resets on Resumed)
                graph.resume(ResumeConfirmed(), config=config)

                # Run 3: resume → Interrupted (round resets again)
                graph.resume(ResumeConfirmed(), config=config)

                # Run 4: resume → Ended (round resets, step=2 → done)
                log = graph.resume(ResumeConfirmed(), config=config)
                assert log.latest(Ended) is not None

            def it_halts_on_max_rounds_not_recursion_error():
                """max_rounds fires before LangGraph's recursion_limit."""

                class PingSent(Event):
                    n: int = 0

                @on(PingSent)
                def pong(event: PingSent) -> PingSent:
                    return PingSent(n=event.n + 1)

                graph = EventGraph([pong], max_rounds=5)
                log = graph.invoke(PingSent())
                assert log.latest(MaxRoundsExceeded) is not None

            def it_halts_on_max_rounds_for_multiple_handlers():
                """recursion_limit accounts for multiple handlers per round."""

                class Ticked(Event):
                    n: int = 0

                class Tocked(Event):
                    n: int = 0

                @on(Ticked)
                def handle_tick(event: Ticked) -> Tocked:
                    return Tocked(n=event.n)

                @on(Tocked)
                def handle_tock(event: Tocked) -> Ticked:
                    return Ticked(n=event.n + 1)

                graph = EventGraph([handle_tick, handle_tock], max_rounds=5)
                log = graph.invoke(Ticked())
                assert log.latest(MaxRoundsExceeded) is not None

            def it_saves_clean_checkpoint_on_max_rounds():
                from langgraph.checkpoint.memory import MemorySaver

                class LoopEvent(Event):
                    n: int = 0

                @on(LoopEvent)
                def looper(event: LoopEvent) -> LoopEvent:
                    return LoopEvent(n=event.n + 1)

                graph = EventGraph([looper], max_rounds=3, checkpointer=MemorySaver())
                config = {"configurable": {"thread_id": "max-rounds-ckpt"}}
                log = graph.invoke(LoopEvent(n=0), config=config)
                assert log.latest(MaxRoundsExceeded) is not None

                state = graph.get_state(config)
                assert state.events.latest(MaxRoundsExceeded) is not None
                assert state.is_interrupted is False

            def it_streams_halted_on_max_rounds():
                class LoopEvent(Event):
                    n: int = 0

                @on(LoopEvent)
                def looper(event: LoopEvent) -> LoopEvent:
                    return LoopEvent(n=event.n + 1)

                graph = EventGraph([looper], max_rounds=3)
                events = list(graph.stream_events(LoopEvent(n=0)))
                assert any(isinstance(e, MaxRoundsExceeded) for e in events)

        def describe_cancellation():

            async def it_halts_on_cancelled_error():
                ready = asyncio.Event()

                @on(Started)
                async def slow(event: Started) -> Ended:
                    ready.set()
                    await asyncio.sleep(100)
                    return Ended(result="done")

                graph = EventGraph([slow])
                task = asyncio.ensure_future(graph.ainvoke(Started(data="go")))
                await ready.wait()
                task.cancel()
                log = await task
                assert log.latest(Cancelled) is not None
                assert not log.has(Ended)

            async def it_discards_partial_events_on_cancel():
                """Events collected before cancellation are not in the log."""
                call_count = 0
                ready = asyncio.Event()

                @on(Started)
                async def multi(event: Started) -> Scatter[Processed]:
                    return Scatter([Processed(data="a"), Processed(data="b")])

                @on(Processed)
                async def slow(event: Processed) -> Ended:
                    nonlocal call_count
                    call_count += 1
                    if call_count == 2:
                        ready.set()
                        await asyncio.sleep(100)
                    return Ended(result=event.data)

                graph = EventGraph([multi, slow])
                task = asyncio.ensure_future(graph.ainvoke(Started(data="go")))
                await ready.wait()
                task.cancel()
                log = await task
                assert log.latest(Cancelled) is not None
                # First invocation's Ended is discarded — partial events
                # within the same handler node are never committed.
                assert not log.has(Ended)

    def describe_mermaid():

        def it_shows_linear_chain_as_edges():
            @on(Started)
            def step1(event: Started) -> Processed:
                return Processed(data=event.data)

            @on(Processed)
            def step2(event: Processed) -> Ended:
                return Ended(result=event.data)

            graph = EventGraph([step1, step2])
            output = graph.mermaid()
            assert "graph LR" in output
            assert "Started -->|step1| Processed" in output
            assert "Processed -->|step2| Ended" in output

        def it_shows_branching_return_types():
            class Accepted(Event):
                pass

            class Rejected(Event):
                pass

            @on(Started)
            def classify(event: Started) -> Accepted | Rejected:
                return Accepted()

            graph = EventGraph([classify])
            output = graph.mermaid()
            assert "Started -->|classify| Accepted" in output
            assert "Started -->|classify| Rejected" in output

        def it_lists_side_effect_handlers_in_footer():
            @on(Started)
            def side_effect(event: Started) -> None:
                pass

            @on(Started)
            def producer(event: Started) -> Ended:
                return Ended(result="ok")

            graph = EventGraph([side_effect, producer])
            output = graph.mermaid()
            assert "%% Side-effect handlers: side_effect (Started)" in output
            assert "Started -->|producer| Ended" in output

        def it_shows_scatter_in_footer():
            @on(Started)
            def split(event: Started) -> Scatter:
                return Scatter([Processed(data="a")])

            graph = EventGraph([split])
            output = graph.mermaid()
            # No edge to a Scatter node
            assert "-->|split| Scatter" not in output
            assert "%% Scatter handlers: split (Started)" in output

        def it_dashes_raises_edge_to_handler_raised():
            from langgraph_events import HandlerRaised

            class _DemoError(Exception):
                pass

            @on(Started, raises=_DemoError)
            def flaky(event: Started) -> Ended:
                raise _DemoError

            @on(HandlerRaised, exception=_DemoError)
            def recover(event: HandlerRaised) -> Ended:
                return Ended(result="recovered")

            graph = EventGraph([flaky, recover])
            output = graph.mermaid()
            assert "Started -.->|flaky (raises)| HandlerRaised" in output
            assert "HandlerRaised -->|recover| Ended" in output
            # HandlerRaised must not appear as a seed entry
            assert "==> HandlerRaised" not in output

        def it_dashes_raises_edge_for_side_effect_handler():
            from langgraph_events import HandlerRaised

            class _DemoError(Exception):
                pass

            @on(Started, raises=_DemoError)
            def side_effect(event: Started) -> None:
                raise _DemoError

            @on(HandlerRaised, exception=_DemoError)
            def recover(event: HandlerRaised) -> None:
                return None

            graph = EventGraph([side_effect, recover])
            output = graph.mermaid()
            # Even though side_effect has no positive return type, the raises
            # edge must still be drawn so HandlerRaised is a real target and
            # the diagram reflects runtime behaviour.
            assert "Started -.->|side_effect (raises)| HandlerRaised" in output
            assert "==> HandlerRaised" not in output

        def it_dashes_interrupted_to_resumed_edge():
            @on(Started)
            def request_approval(event: Started) -> Interrupted:
                return Interrupted()

            @on(Resumed)
            def handle_review(event: Resumed) -> Ended:
                return Ended(result="ok")

            graph = EventGraph([request_approval, handle_review])
            output = graph.mermaid()
            assert "Interrupted -.-> Resumed" in output
            assert "Resumed -->|handle_review| Ended" in output

        def it_shows_question_mark_for_unannotated_handlers():
            @on(Started)
            def mystery(event: Started):
                return Ended(result="ok")

            graph = EventGraph([mystery])
            output = graph.mermaid()
            assert "Started -->|mystery| ?" in output

        def it_shows_multi_subscription_edges():
            @on(Started, Processed)
            def handle_both(event: Event) -> Ended:
                return Ended(result="ok")

            graph = EventGraph([handle_both])
            output = graph.mermaid()
            assert "Started -->|handle_both| Ended" in output
            assert "Processed -->|handle_both| Ended" in output

        def it_uses_thick_entry_edges_for_seeds():
            @on(Started)
            def step1(event: Started) -> Processed:
                return Processed(data=event.data)

            @on(Processed)
            def step2(event: Processed) -> Ended:
                return Ended(result=event.data)

            graph = EventGraph([step1, step2])
            output = graph.mermaid()
            assert "classDef entry fill:none,stroke:none,color:none" in output
            assert "_e0_[ ]:::entry ==> Started" in output
            # Processed is a target, not a seed
            assert "==> Processed" not in output

        def it_shows_typed_scatter_as_dashed_edge():
            @on(Started)
            def split(event: Started) -> Scatter[Processed]:
                return Scatter([Processed(data="a")])

            @on(Processed)
            def step2(event: Processed) -> Ended:
                return Ended(result=event.data)

            graph = EventGraph([split, step2])
            output = graph.mermaid()
            assert "Started -.->|split| Processed" in output
            assert "%% Scatter handlers" not in output

    def describe_construction_validation():

        def when_no_handlers():

            def it_raises_value_error():
                with pytest.raises(ValueError, match="at least one handler"):
                    EventGraph([])

        def when_duplicate_handler_names():

            def it_appends_numeric_suffix():
                @on(Started)
                def handler(event: Started) -> Processed:
                    return Processed(data=event.data)

                # Passing the same function twice triggers name dedup
                graph = EventGraph([handler, handler])
                names = [m.name for m in graph._handler_metas]
                assert len(names) == 2
                assert names[0] != names[1]
                assert "_2" in names[1]

            def it_uses_deduped_names_in_mermaid_labels():
                @on(Started)
                def handler(event: Started) -> Processed:
                    return Processed(data=event.data)

                graph = EventGraph([handler, handler])
                output = graph.mermaid()
                assert "-->|handler|" in output
                assert "-->|handler_2|" in output

            def it_preserves_raises_on_deduped_copies():
                from langgraph_events import HandlerRaised

                class _DedupError(Exception):
                    pass

                @on(Started, raises=_DedupError)
                def raiser(event: Started) -> Ended:
                    raise _DedupError("boom")

                @on(HandlerRaised, exception=_DedupError)
                def catcher(event: HandlerRaised) -> Ended:
                    return Ended(result="caught")

                graph = EventGraph([raiser, raiser, catcher])
                raisers = [
                    m for m in graph._handler_metas if m.name.startswith("raiser")
                ]
                assert len(raisers) == 2
                # Without the fix, the second copy's raises= is silently dropped
                for m in raisers:
                    assert m.raises == (_DedupError,)

            def it_preserves_field_matchers_on_deduped_copies():
                from langgraph_events import HandlerRaised

                class _DedupError(Exception):
                    pass

                @on(Started, raises=_DedupError)
                def raiser(event: Started) -> Ended:
                    raise _DedupError("boom")

                @on(HandlerRaised, exception=_DedupError)
                def catcher(event: HandlerRaised) -> Ended:
                    return Ended(result="caught")

                graph = EventGraph([raiser, catcher, catcher])
                catchers = [
                    m for m in graph._handler_metas if m.name.startswith("catcher")
                ]
                assert len(catchers) == 2
                # Without the fix, the second copy becomes a universal catcher
                for m in catchers:
                    matcher_names = [fn for fn, _ in m.field_matchers]
                    assert "exception" in matcher_names

        def when_base_event_return_type():

            def it_rejects_base_event_return_type():
                @on(Started)
                def handler(event: Started) -> Event:
                    return Processed(data=event.data)

                with pytest.raises(ValueError, match="base 'Event'"):
                    EventGraph([handler])

            def it_rejects_event_in_union_return_type():
                @on(Started)
                def handler(event: Started) -> Event | None:
                    return Processed(data=event.data)

                with pytest.raises(ValueError, match="base 'Event'"):
                    EventGraph([handler])

            def it_allows_event_subclass_return_type():
                class Audited(Event):
                    data: str = ""

                @on(Started)
                def handler(event: Started) -> Audited:
                    return Audited(data=event.data)

                # Should not raise
                EventGraph([handler])

    def describe_astream_llm_tokens():

        def describe_custom_event_helpers():

            @pytest.mark.asyncio
            async def it_emits_custom_frames_from_sync_handler():
                from langgraph_events import CustomEventFrame

                @on(Started)
                def step(event: Started) -> Ended:
                    emit_custom("tool.progress", {"pct": 25})
                    return Ended(result=event.data)

                graph = EventGraph([step])
                items = [
                    item
                    async for item in graph.astream_events(
                        Started(data="hello"),
                        include_custom_events=True,
                    )
                ]

                custom_frames = [i for i in items if isinstance(i, CustomEventFrame)]
                assert len(custom_frames) == 1
                assert custom_frames[0].name == "tool.progress"
                assert custom_frames[0].data == {"pct": 25}

            @pytest.mark.asyncio
            async def it_emits_custom_frames_from_async_handler():
                from langgraph_events import CustomEventFrame

                @on(Started)
                async def step(event: Started) -> Ended:
                    await aemit_custom("tool.progress", {"pct": 80})
                    return Ended(result=event.data)

                graph = EventGraph([step])
                items = [
                    item
                    async for item in graph.astream_events(
                        Started(data="hello"),
                        include_custom_events=True,
                    )
                ]

                custom_frames = [i for i in items if isinstance(i, CustomEventFrame)]
                assert len(custom_frames) == 1
                assert custom_frames[0].name == "tool.progress"
                assert custom_frames[0].data == {"pct": 80}

            @pytest.mark.asyncio
            async def it_emits_state_snapshot_frames_from_sync_handler():
                @on(Started)
                def step(event: Started) -> Ended:
                    emit_state_snapshot({"step": "draft"})
                    return Ended(result=event.data)

                graph = EventGraph([step])
                items = [
                    item
                    async for item in graph.astream_events(
                        Started(data="hello"),
                        include_custom_events=True,
                    )
                ]

                snapshots = [i for i in items if isinstance(i, StateSnapshotFrame)]
                assert len(snapshots) == 1
                assert snapshots[0].data == {"step": "draft"}

            @pytest.mark.asyncio
            async def it_emits_state_snapshot_frames_from_async_handler():
                @on(Started)
                async def step(event: Started) -> Ended:
                    await aemit_state_snapshot({"step": "review"})
                    return Ended(result=event.data)

                graph = EventGraph([step])
                items = [
                    item
                    async for item in graph.astream_events(
                        Started(data="hello"),
                        include_custom_events=True,
                    )
                ]

                snapshots = [i for i in items if isinstance(i, StateSnapshotFrame)]
                assert len(snapshots) == 1
                assert snapshots[0].data == {"step": "review"}

            def it_raises_for_emit_custom_outside_handler():
                with pytest.raises(RuntimeError, match="while an EventGraph handler"):
                    emit_custom("tool.progress", {"pct": 1})

            def it_raises_for_emit_state_snapshot_outside_handler():
                with pytest.raises(RuntimeError, match="while an EventGraph handler"):
                    emit_state_snapshot({"step": "x"})

            @pytest.mark.asyncio
            async def it_raises_for_aemit_custom_outside_handler():
                with pytest.raises(RuntimeError, match="while an EventGraph handler"):
                    await aemit_custom("tool.progress", {"pct": 1})

            @pytest.mark.asyncio
            async def it_raises_for_aemit_state_snapshot_outside_handler():
                with pytest.raises(RuntimeError, match="while an EventGraph handler"):
                    await aemit_state_snapshot({"step": "x"})

        @pytest.mark.asyncio
        async def it_yields_llm_token_and_stream_end_frames():
            from typing import Any

            from langchain_core.language_models.fake_chat_models import (
                FakeListChatModel,
            )

            from langgraph_events._graph import LLMStreamEnd, LLMToken

            llm = FakeListChatModel(responses=["hello world"], sleep=0)

            class UserSent(MessageEvent):
                message: HumanMessage = None  # type: ignore[assignment]

            class AgentReplied(MessageEvent):
                message: AIMessage = None  # type: ignore[assignment]

            @on(UserSent)
            async def reply(event: UserSent, messages: list[Any]) -> AgentReplied:
                response = await llm.ainvoke([*messages, HumanMessage(content="hi")])
                return AgentReplied(message=response)

            graph = EventGraph([reply], reducers=[message_reducer()])
            items = [
                item
                async for item in graph.astream_events(
                    UserSent(message=HumanMessage(content="hi")),
                    include_llm_tokens=True,
                )
            ]

            tokens = [i for i in items if isinstance(i, LLMToken)]
            ends = [i for i in items if isinstance(i, LLMStreamEnd)]

            # Should have at least one token and one end
            assert len(tokens) >= 1
            assert len(ends) >= 1
            # Token content should reconstruct the response
            assert "".join(t.content for t in tokens) == "hello world"
            # LLMStreamEnd should have a message_id (AIMessage.id)
            assert ends[0].message_id is not None

        @pytest.mark.asyncio
        async def it_yields_domain_events_alongside_llm_tokens():
            from typing import Any

            from langchain_core.language_models.fake_chat_models import (
                FakeListChatModel,
            )

            from langgraph_events._graph import LLMToken

            llm = FakeListChatModel(responses=["reply"], sleep=0)

            class UserSent(MessageEvent):
                message: HumanMessage = None  # type: ignore[assignment]

            class AgentReplied(MessageEvent):
                message: AIMessage = None  # type: ignore[assignment]

            @on(UserSent)
            async def reply(event: UserSent, messages: list[Any]) -> AgentReplied:
                response = await llm.ainvoke([*messages, HumanMessage(content="hi")])
                return AgentReplied(message=response)

            graph = EventGraph([reply], reducers=[message_reducer()])
            items = [
                item
                async for item in graph.astream_events(
                    UserSent(message=HumanMessage(content="hi")),
                    include_llm_tokens=True,
                )
            ]

            domain_events = [i for i in items if isinstance(i, Event)]
            tokens = [i for i in items if isinstance(i, LLMToken)]
            assert len(domain_events) >= 2  # at least seed + reply
            assert len(tokens) >= 1

        @pytest.mark.asyncio
        async def it_yields_reducer_frames_and_tokens():
            from typing import Any

            from langchain_core.language_models.fake_chat_models import (
                FakeListChatModel,
            )

            from langgraph_events._graph import LLMToken

            llm = FakeListChatModel(responses=["hi back"], sleep=0)

            class UserSent(MessageEvent):
                message: HumanMessage = None  # type: ignore[assignment]

            class AgentReplied(MessageEvent):
                message: AIMessage = None  # type: ignore[assignment]

            @on(UserSent)
            async def reply(event: UserSent, messages: list[Any]) -> AgentReplied:
                response = await llm.ainvoke([*messages, HumanMessage(content="go")])
                return AgentReplied(message=response)

            graph = EventGraph([reply], reducers=[message_reducer()])
            items = [
                item
                async for item in graph.astream_events(
                    UserSent(message=HumanMessage(content="go")),
                    include_reducers=True,
                    include_llm_tokens=True,
                )
            ]

            frames = [i for i in items if isinstance(i, StreamFrame)]
            tokens = [i for i in items if isinstance(i, LLMToken)]
            assert len(frames) >= 2  # seed + reply
            assert len(tokens) >= 1
            # Frames should have reducer data
            assert all("messages" in f.reducers for f in frames)
            # v2 reducer frames track which reducers changed per event
            assert all(f.changed_reducers is not None for f in frames)
            assert "messages" in frames[0].changed_reducers
            assert "messages" in frames[-1].changed_reducers

        @pytest.mark.asyncio
        async def it_reports_empty_changed_reducers_for_non_matching_events():
            reducer = _data_reducer()

            @on(Started)
            def step1(event: Started) -> Processed:
                return Processed(data=f"mid:{event.data}")

            @on(Processed)
            def step2(event: Processed) -> Ended:
                return Ended(result=event.data)

            graph = EventGraph([step1, step2], reducers=[reducer])
            items = [
                item
                async for item in graph.astream_events(
                    Started(data="x"),
                    include_reducers=True,
                    include_llm_tokens=True,
                )
            ]

            frames = [i for i in items if isinstance(i, StreamFrame)]
            assert len(frames) >= 3
            assert frames[0].changed_reducers == frozenset({"data_items"})
            # Processed/Ended are not Started events, so reducer is unchanged.
            assert all(f.changed_reducers == frozenset() for f in frames[1:])

        @pytest.mark.asyncio
        async def it_omits_tokens_by_default():
            """Without include_llm_tokens, no LLMToken/LLMStreamEnd are yielded."""
            from langgraph_events._graph import LLMStreamEnd, LLMToken

            @on(Started)
            def step(event: Started) -> Ended:
                return Ended(result=event.data)

            graph = EventGraph([step])
            items = [item async for item in graph.astream_events(Started(data="hi"))]
            assert all(isinstance(i, Event) for i in items)
            assert not any(isinstance(i, (LLMToken, LLMStreamEnd)) for i in items)

        @pytest.mark.asyncio
        async def it_yields_custom_event_frames_from_v2_custom_events(monkeypatch):
            from langgraph_events._graph import CustomEventFrame, StateSnapshotFrame

            @on(Started)
            def step(event: Started) -> Ended:
                return Ended(result=event.data)

            graph = EventGraph([step])

            async def fake_astream_events(*args, **kwargs):
                del args, kwargs
                yield {
                    "event": "on_custom_event",
                    "name": "progress",
                    "data": {"pct": 50},
                }
                yield {
                    "event": "on_custom_event",
                    "name": STATE_SNAPSHOT_EVENT_NAME,
                    "data": {"step": "draft"},
                }

            monkeypatch.setattr(graph.compiled, "astream_events", fake_astream_events)

            items = [
                item
                async for item in graph.astream_events(
                    Started(data="hi"),
                    include_llm_tokens=True,
                    include_custom_events=True,
                )
            ]

            custom_frames = [i for i in items if isinstance(i, CustomEventFrame)]
            assert len(custom_frames) == 1
            assert custom_frames[0].name == "progress"
            assert custom_frames[0].data == {"pct": 50}

            snapshots = [i for i in items if isinstance(i, StateSnapshotFrame)]
            assert len(snapshots) == 1
            assert snapshots[0].data == {"step": "draft"}

        @pytest.mark.asyncio
        async def it_does_not_yield_custom_event_frames_by_default(monkeypatch):
            from langgraph_events._graph import CustomEventFrame

            @on(Started)
            def step(event: Started) -> Ended:
                return Ended(result=event.data)

            graph = EventGraph([step])

            called = False

            async def fake_astream_events(*args, **kwargs):
                nonlocal called
                called = True
                del args, kwargs
                yield {
                    "event": "on_custom_event",
                    "name": "progress",
                    "data": {"pct": 50},
                }

            monkeypatch.setattr(graph.compiled, "astream_events", fake_astream_events)

            items = [item async for item in graph.astream_events(Started(data="hi"))]
            assert not any(isinstance(i, CustomEventFrame) for i in items)
            # Default flags route to _astream_core, not v2 — confirm the fake
            # was not called so the test's intent is clear.
            assert not called

        @pytest.mark.asyncio
        async def it_filters_custom_events_in_v2_path(monkeypatch):
            from langgraph_events._graph import CustomEventFrame

            @on(Started)
            def step(event: Started) -> Ended:
                return Ended(result=event.data)

            graph = EventGraph([step])

            async def fake_astream_events(*args, **kwargs):
                del args, kwargs
                yield {
                    "event": "on_custom_event",
                    "name": "progress",
                    "data": {"pct": 50},
                }

            monkeypatch.setattr(graph.compiled, "astream_events", fake_astream_events)

            # include_llm_tokens routes to _astream_v2 but custom events off
            items = [
                item
                async for item in graph.astream_events(
                    Started(data="hi"),
                    include_llm_tokens=True,
                    include_custom_events=False,
                )
            ]
            assert not any(isinstance(i, CustomEventFrame) for i in items)

        @pytest.mark.asyncio
        async def it_yields_custom_event_frames_on_opt_in(
            monkeypatch,
        ):
            from langgraph_events._graph import CustomEventFrame

            @on(Started)
            def step(event: Started) -> Ended:
                return Ended(result=event.data)

            graph = EventGraph([step])

            async def fake_astream_events(*args, **kwargs):
                del args, kwargs
                yield {
                    "event": "on_custom_event",
                    "name": "progress",
                    "data": {"pct": 50},
                }

            monkeypatch.setattr(graph.compiled, "astream_events", fake_astream_events)

            items = [
                item
                async for item in graph.astream_events(
                    Started(data="hi"),
                    include_custom_events=True,
                )
            ]
            custom_frames = [i for i in items if isinstance(i, CustomEventFrame)]
            assert len(custom_frames) == 1

        @pytest.mark.asyncio
        async def it_yields_custom_event_frames_in_astream_resume(
            monkeypatch,
        ):
            from langgraph.checkpoint.memory import MemorySaver

            from langgraph_events._graph import CustomEventFrame

            @on(Started)
            def step(event: Started) -> Ended:
                return Ended(result=event.data)

            graph = EventGraph([step], checkpointer=MemorySaver())

            async def fake_astream_events(*args, **kwargs):
                del args, kwargs
                yield {
                    "event": "on_custom_event",
                    "name": "resume.progress",
                    "data": {"pct": 90},
                }

            monkeypatch.setattr(graph.compiled, "astream_events", fake_astream_events)

            items = [
                item
                async for item in graph.astream_resume(
                    Started(data="resume"),
                    include_custom_events=True,
                )
            ]

            custom_frames = [i for i in items if isinstance(i, CustomEventFrame)]
            assert len(custom_frames) == 1
            assert custom_frames[0].name == "resume.progress"


def describe_OrphanedEventWarning():

    def when_orphaned():

        def it_warns_about_orphaned_event_types():
            class Orphan(Event):
                pass

            @on(Started)
            def produce_orphan(event: Started) -> Orphan:
                return Orphan()

            with pytest.warns(OrphanedEventWarning, match="Orphan"):
                EventGraph([produce_orphan])

        def it_warns_for_orphaned_scatter_types():
            class ScatterOrphan(Event):
                pass

            @on(Started)
            def scatter_producer(event: Started) -> Scatter[ScatterOrphan]:
                return Scatter([ScatterOrphan()])

            with pytest.warns(OrphanedEventWarning, match="ScatterOrphan"):
                EventGraph([scatter_producer])

    def when_not_orphaned():

        def it_does_not_warn_for_subscribed_via_inheritance():
            class Base(Event):
                pass

            class Sub(Base):
                pass

            @on(Started)
            def produce_sub(event: Started) -> Sub:
                return Sub()

            @on(Base)
            def consume_base(event: Base):
                pass

            # Sub is consumed by @on(Base) via isinstance — no warning
            with warnings.catch_warnings():
                warnings.simplefilter("error", OrphanedEventWarning)
                EventGraph([produce_sub, consume_base])

        def it_does_not_warn_for_halted_returns():
            @on(Started)
            def halter(event: Started) -> Halted:
                return Halted()

            with warnings.catch_warnings():
                warnings.simplefilter("error", OrphanedEventWarning)
                EventGraph([halter])

        def it_does_not_warn_for_interrupted_returns():
            class AskApproval(Interrupted):
                pass

            @on(Started)
            def asker(event: Started) -> AskApproval:
                return AskApproval()

            with warnings.catch_warnings():
                warnings.simplefilter("error", OrphanedEventWarning)
                EventGraph([asker])

        def it_does_not_warn_for_unannotated_handlers():
            @on(Started)
            def no_annotation(event: Started):
                return Ended(result="ok")

            with warnings.catch_warnings():
                warnings.simplefilter("error", OrphanedEventWarning)
                EventGraph([no_annotation])

        def it_does_not_warn_for_none_in_union():
            """Optional[Event] return type should not warn about NoneType."""

            class MaybeResult(Event):
                pass

            @on(Started)
            def maybe(event: Started) -> MaybeResult | None:
                return None

            @on(MaybeResult)
            def consumer(event: MaybeResult):
                pass

            with warnings.catch_warnings():
                warnings.simplefilter("error", OrphanedEventWarning)
                EventGraph([maybe, consumer])
