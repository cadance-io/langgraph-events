"""Integration tests for EventGraph — the full event-driven graph engine."""

import pytest
from conftest import Done, End, Middle, MsgIn, MsgOut, Start
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)

from langgraph_events import (
    Event,
    EventGraph,
    EventLog,
    Halt,
    Interrupted,
    MessageEvent,
    Reducer,
    Resumed,
    Scatter,
    StreamFrame,
    SystemPromptSet,
    message_reducer,
    on,
)

# ---------------------------------------------------------------------------
# Helpers (prefixed with _ to exclude from collection)
# ---------------------------------------------------------------------------


def _data_reducer() -> Reducer:
    """Simple reducer that accumulates Start.data values."""

    def fn(event: Event) -> list[str]:
        if isinstance(event, Start):
            return [event.data]
        return []

    return Reducer(name="data_items", fn=fn)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def describe_EventGraph():

    def describe_invoke():

        def describe_linear_chain():

            def it_processes_three_step_chain(linear_chain):
                log = linear_chain.invoke(Start(data="hello"))
                assert isinstance(log, EventLog)
                assert len(log) == 3
                assert log.latest(End) == End(result="done:processed:hello")

            async def it_works_with_async_handlers():
                @on(Start)
                async def step1(event: Start) -> Middle:
                    return Middle(data=event.data.upper())

                @on(Middle)
                async def step2(event: Middle) -> End:
                    return End(result=event.data)

                graph = EventGraph([step1, step2])
                log = await graph.ainvoke(Start(data="hello"))
                assert log.latest(End) == End(result="HELLO")

        def describe_branching():

            class Input(Event):
                kind: str = ""
                data: str = ""

            class FastPath(Event):
                data: str = ""

            class SlowPath(Event):
                data: str = ""

            class Output(Event):
                result: str = ""

            @pytest.fixture
            def branching_graph():
                @on(Input)
                def route(event: Input) -> Event | None:
                    if event.kind == "fast":
                        return FastPath(data=event.data)
                    return SlowPath(data=event.data)

                @on(FastPath)
                def handle_fast(event: FastPath) -> Output:
                    return Output(result=f"fast:{event.data}")

                @on(SlowPath)
                def handle_slow(event: SlowPath) -> Output:
                    return Output(result=f"slow:{event.data}")

                return EventGraph([route, handle_fast, handle_slow])

            def when_fast_path():

                def it_produces_fast_output(branching_graph):
                    log = branching_graph.invoke(Input(kind="fast", data="x"))
                    assert log.latest(Output) == Output(result="fast:x")

                def it_does_not_trigger_slow_handler(branching_graph):
                    log = branching_graph.invoke(Input(kind="fast", data="x"))
                    assert not log.has(SlowPath)

            def when_slow_path():

                def it_produces_slow_output(branching_graph):
                    log = branching_graph.invoke(Input(kind="slow", data="y"))
                    assert log.latest(Output) == Output(result="slow:y")

                def it_does_not_trigger_fast_handler(branching_graph):
                    log = branching_graph.invoke(Input(kind="slow", data="y"))
                    assert not log.has(FastPath)

        def describe_fan_out_via_inheritance():

            class Trackable(Event):
                action: str = ""

            class Processable(Event):
                item: str = ""

            class TrackableItem(Trackable, Processable):
                action: str = ""
                item: str = ""

            class AuditDone(Event):
                msg: str = ""

            class ProcessDone(Event):
                msg: str = ""

            def it_triggers_both_parent_handlers():
                @on(Trackable)
                def audit(event: Trackable) -> AuditDone:
                    return AuditDone(msg=f"audited:{event.action}")

                @on(Processable)
                def process(event: Processable) -> ProcessDone:
                    return ProcessDone(msg=f"processed:{event.item}")

                graph = EventGraph([audit, process])
                log = graph.invoke(TrackableItem(action="create", item="doc1"))
                assert log.has(AuditDone)
                assert log.has(ProcessDone)
                assert log.latest(AuditDone) == AuditDone(msg="audited:create")
                assert log.latest(ProcessDone) == ProcessDone(msg="processed:doc1")

            def it_fires_parent_handler_for_child_event():
                class Base(Event):
                    x: str = ""

                class Child(Base):
                    y: str = ""

                class Result(Event):
                    v: str = ""

                @on(Base)
                def handle_base(event: Base) -> Result:
                    return Result(v=event.x)

                graph = EventGraph([handle_base])
                log = graph.invoke(Child(x="hello", y="world"))
                assert log.latest(Result) == Result(v="hello")

        def describe_side_effect_handlers():

            def it_executes_side_effect_on_none_return():
                side_effects: list[str] = []

                @on(Start)
                def produce(event: Start) -> Middle:
                    return Middle(data=event.data)

                @on(Middle)
                def consume(event: Middle) -> None:
                    side_effects.append(event.data)

                graph = EventGraph([produce, consume])
                log = graph.invoke(Start(data="test"))
                assert len(log) == 2
                assert side_effects == ["test"]

        def describe_event_log_injection():

            def it_provides_full_log_to_handler():
                @on(Start)
                def step1(event: Start) -> Middle:
                    return Middle(data=event.data)

                @on(Middle)
                def step2(event: Middle, log: EventLog) -> End:
                    assert log.has(Start)
                    count = len(log.filter(Event))
                    return End(result=f"saw {count} events")

                graph = EventGraph([step1, step2])
                log = graph.invoke(Start(data="hello"))
                assert log.latest(End) == End(result="saw 2 events")

            def it_shows_snapshot_not_affected_by_later_events():
                log_lengths: list[int] = []

                @on(Start)
                def step1(event: Start) -> Middle:
                    return Middle(data="from_step1")

                @on(Middle)
                def step2(event: Middle, log: EventLog) -> End:
                    log_lengths.append(len(log))
                    assert not log.has(End)
                    return End(result="done")

                graph = EventGraph([step1, step2])
                final_log = graph.invoke(Start(data="test"))
                assert log_lengths == [2]
                assert len(final_log) == 3

            def it_prevents_mutation_from_corrupting_graph_state():
                @on(Start)
                def evil_handler(event: Start, log: EventLog) -> Middle:
                    log._events.append(End(result="INJECTED"))
                    log._events.clear()
                    return Middle(data="honest")

                @on(Middle)
                def step2(event: Middle, log: EventLog) -> End:
                    assert log.has(Start)
                    assert log.has(Middle)
                    injected = [
                        e for e in log if isinstance(e, End) and e.result == "INJECTED"
                    ]
                    assert injected == []
                    return End(result="clean")

                graph = EventGraph([evil_handler, step2])
                final_log = graph.invoke(Start(data="test"))
                assert len(final_log) == 3
                assert final_log.latest(End) == End(result="clean")
                injected = [
                    e
                    for e in final_log
                    if isinstance(e, End) and e.result == "INJECTED"
                ]
                assert injected == []

            def it_provides_independent_snapshots_to_parallel_handlers():
                class Trigger(Event):
                    value: str = ""

                class ResultA(Event):
                    saw_events: int = 0

                class ResultB(Event):
                    saw_events: int = 0

                class Collected(Event):
                    a_saw: int = 0
                    b_saw: int = 0

                @on(Trigger)
                def handler_a(event: Trigger, log: EventLog) -> ResultA:
                    log._events.append(End(result="from_a"))
                    return ResultA(saw_events=len(log))

                @on(Trigger)
                def handler_b(event: Trigger, log: EventLog) -> ResultB:
                    has_end = any(isinstance(e, End) for e in log)
                    assert not has_end
                    return ResultB(saw_events=len(log))

                @on(ResultA, ResultB)
                def collect(event: Event, log: EventLog) -> Collected | None:
                    if log.has(ResultA) and log.has(ResultB):
                        a = log.latest(ResultA)
                        b = log.latest(ResultB)
                        return Collected(a_saw=a.saw_events, b_saw=b.saw_events)
                    return None

                graph = EventGraph([handler_a, handler_b, collect])
                final_log = graph.invoke(Trigger(value="go"))
                result = final_log.latest(Collected)
                assert result is not None
                assert result.b_saw == 1

        def describe_multi_subscription():

            class Ping(Event):
                value: str = ""

            class Pong(Event):
                value: str = ""

            class Reply(Event):
                value: str = ""

            def it_fires_on_either_event_type():
                @on(Ping, Pong)
                def echo(event: Event) -> Reply:
                    if isinstance(event, Ping):
                        return Reply(value=f"ping:{event.value}")
                    return Reply(value=f"pong:{event.value}")

                @on(Reply)
                def finish(event: Reply) -> Done:
                    return Done(result=event.value)

                graph = EventGraph([echo, finish])
                log = graph.invoke(Ping(value="hello"))
                assert log.latest(Done) == Done(result="ping:hello")
                log = graph.invoke(Pong(value="world"))
                assert log.latest(Done) == Done(result="pong:world")

            def it_provides_log_to_multi_sub_handler():
                class MsgA(Event):
                    text: str = ""

                class MsgB(Event):
                    text: str = ""

                class Summary(Event):
                    count: int = 0

                @on(MsgA, MsgB)
                def summarize(event: Event, log: EventLog) -> Summary:
                    total = len(log.filter(Event))
                    return Summary(count=total)

                graph = EventGraph([summarize])
                log = graph.invoke(MsgA(text="hi"))
                assert log.latest(Summary) == Summary(count=1)

            def it_supports_react_loop_pattern():
                class UserMsg(Event):
                    content: str = ""

                class AssistantMsg(Event):
                    content: str = ""
                    needs_tool: bool = False

                class ToolResult(Event):
                    result: str = ""

                class FinalAnswer(Event):
                    answer: str = ""

                call_count = 0

                @on(UserMsg, ToolResult)
                def call_llm(event: Event, log: EventLog) -> AssistantMsg:
                    nonlocal call_count
                    call_count += 1
                    if isinstance(event, UserMsg):
                        return AssistantMsg(content="need tool", needs_tool=True)
                    return AssistantMsg(content=f"got:{event.result}", needs_tool=False)

                @on(AssistantMsg)
                def handle_response(
                    event: AssistantMsg,
                ) -> ToolResult | FinalAnswer:
                    if event.needs_tool:
                        return ToolResult(result="42")
                    return FinalAnswer(answer=event.content)

                graph = EventGraph([call_llm, handle_response])
                log = graph.invoke(UserMsg(content="what is 6*7?"))
                assert call_count == 2
                assert log.latest(FinalAnswer) == FinalAnswer(answer="got:42")
                assert log.has(ToolResult)
                assert log.has(AssistantMsg)

        def describe_multi_seed():

            def it_accepts_list_of_seed_events():
                @on(Start)
                def step1(event: Start) -> Middle:
                    return Middle(data=event.data)

                @on(Middle)
                def step2(event: Middle) -> End:
                    return End(result=event.data)

                graph = EventGraph([step1, step2])
                log = graph.invoke([Start(data="hello")])
                assert log.latest(End) == End(result="hello")

            def it_includes_all_seed_events_in_log():
                class Config(Event):
                    setting: str = ""

                @on(Start)
                def handle(event: Start, log: EventLog) -> End:
                    config = log.latest(Config)
                    return End(result=f"{config.setting}:{event.data}")

                graph = EventGraph([handle])
                log = graph.invoke([Config(setting="v1"), Start(data="go")])
                assert log.has(Config)
                assert log.has(Start)
                assert log.latest(End) == End(result="v1:go")

            def it_still_accepts_single_event():
                @on(Start)
                def step(event: Start) -> End:
                    return End(result=event.data)

                graph = EventGraph([step])
                log = graph.invoke(Start(data="solo"))
                assert log.latest(End) == End(result="solo")

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
                    @on(Start)
                    def handle(event: Start, log: EventLog) -> End:
                        has_prompt = log.has(SystemPromptSet)
                        return End(result=f"has_prompt={has_prompt}")

                    graph = EventGraph([handle])
                    log = graph.invoke(
                        [
                            SystemPromptSet.from_str("You are helpful"),
                            Start(data="go"),
                        ]
                    )
                    assert log.has(SystemPromptSet)
                    assert log.latest(End) == End(result="has_prompt=True")

                def it_contributes_to_message_reducer():
                    class UserMsg(MessageEvent):
                        message: HumanMessage = None  # type: ignore[assignment]

                    class Finished(Event):
                        answer: str = ""

                    r = message_reducer()

                    received_messages: list[list[BaseMessage]] = []

                    @on(UserMsg)
                    def respond(
                        event: UserMsg, messages: list[BaseMessage]
                    ) -> Finished:
                        received_messages.append(list(messages))
                        return Finished(answer="ok")

                    graph = EventGraph([respond], reducers=[r])
                    log = graph.invoke(
                        [
                            SystemPromptSet.from_str("You are a test bot"),
                            UserMsg(message=HumanMessage(content="hello")),
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
                @on(Start)
                async def step(event: Start) -> End:
                    return End(result=event.data)

                graph = EventGraph([step])
                log = await graph.ainvoke([Start(data="a"), Start(data="b")])
                ends = log.filter(End)
                assert len(ends) == 2
                assert {e.result for e in ends} == {"a", "b"}

            async def it_stops_on_halt():
                @on(Start)
                async def halter(event: Start) -> Halt:
                    return Halt(reason="stop")

                @on(Halt)
                async def unreachable(event: Halt) -> End:
                    return End(result="should not run")

                graph = EventGraph([halter, unreachable])
                log = await graph.ainvoke(Start(data="go"))
                assert log.has(Halt)
                assert not log.has(End)

            async def it_injects_reducer_values():
                reducer = _data_reducer()

                @on(Start)
                async def step(event: Start, data_items: list) -> End:
                    return End(result=",".join(data_items))

                graph = EventGraph([step], reducers=[reducer])
                log = await graph.ainvoke(Start(data="x"))
                assert log.latest(End) == End(result="x")

    def describe_halt():

        def it_stores_reason_and_is_Event_subclass():
            h = Halt(reason="done")
            assert h.reason == "done"
            assert isinstance(h, Event)
            assert isinstance(h, Halt)

        def it_stops_execution_immediately():
            @on(Start)
            def step1(event: Start) -> Halt:
                return Halt(reason="stopped early")

            @on(Halt)
            def should_not_run(event: Halt) -> End:
                return End(result="should not reach here")

            graph = EventGraph([step1, should_not_run])
            log = graph.invoke(Start(data="test"))
            assert log.has(Halt)
            assert not log.has(End)

    def describe_interrupt():

        def it_stores_prompt_and_payload():
            i = Interrupted(prompt="Approve?", payload={"doc_id": "123"})
            assert i.prompt == "Approve?"
            assert i.payload == {"doc_id": "123"}
            assert isinstance(i, Event)

        def it_stores_value_and_interrupted_reference():
            i = Interrupted(prompt="Approve?")
            r = Resumed(value="yes", interrupted=i)
            assert r.value == "yes"
            assert r.interrupted is i
            assert isinstance(r, Event)

        def it_pauses_and_resumes_with_human_input():
            from langgraph.checkpoint.memory import MemorySaver
            from langgraph.types import Command

            @on(Start)
            def need_input(event: Start) -> Interrupted:
                return Interrupted(
                    prompt="Please confirm",
                    payload={"data": event.data},
                )

            @on(Resumed)
            def handle_resume(event: Resumed) -> End:
                return End(result=f"confirmed:{event.value}")

            graph = EventGraph([need_input, handle_resume])
            checkpointer = MemorySaver()
            compiled = graph.compile(checkpointer=checkpointer)

            config = {"configurable": {"thread_id": "interrupt-test"}}
            compiled.invoke({"events": [Start(data="test")]}, config)
            state = compiled.get_state(config)
            assert state.next

            result = compiled.invoke(Command(resume="yes"), config)
            events = result["events"]
            end_events = [e for e in events if isinstance(e, End)]
            assert len(end_events) == 1
            assert end_events[0].result == "confirmed:yes"

    def describe_scatter():

        class Batch(Event):
            items: tuple = ()

        class WorkItem(Event):
            item: str = ""
            batch_size: int = 0

        class WorkDone(Event):
            item: str = ""
            result: str = ""

        class BatchResult(Event):
            results: tuple = ()

        def describe_construction():

            def it_wraps_a_list_of_events():
                class Item(Event):
                    v: int = 0

                s = Scatter([Item(v=1), Item(v=2), Item(v=3)])
                assert len(s.events) == 3
                assert s.events[0] == Item(v=1)

            def when_empty():

                def it_raises_value_error():
                    with pytest.raises(ValueError, match="at least one"):
                        Scatter([])

            def when_contains_non_events():

                def it_raises_type_error():
                    with pytest.raises(TypeError, match="Event instances"):
                        Scatter(["not an event"])  # type: ignore

        def it_fans_out_work_items_and_gathers_results():
            @on(Batch)
            def split(event: Batch) -> Scatter:
                return Scatter(
                    [
                        WorkItem(item=item, batch_size=len(event.items))
                        for item in event.items
                    ]
                )

            @on(WorkItem)
            def process(event: WorkItem) -> WorkDone:
                return WorkDone(item=event.item, result=f"done:{event.item}")

            @on(WorkDone)
            def gather(event: WorkDone, log: EventLog) -> BatchResult | None:
                all_done = log.filter(WorkDone)
                batch = log.latest(Batch)
                if len(all_done) >= len(batch.items):
                    return BatchResult(results=tuple(e.result for e in all_done))
                return None

            graph = EventGraph([split, process, gather])
            log = graph.invoke(Batch(items=("a", "b", "c")))
            assert log.has(BatchResult)
            result = log.latest(BatchResult)
            assert len(result.results) == 3
            assert set(result.results) == {"done:a", "done:b", "done:c"}

        def when_single_item():

            def it_still_produces_output():
                @on(Batch)
                def split(event: Batch) -> Scatter:
                    return Scatter([WorkItem(item=event.items[0])])

                @on(WorkItem)
                def process(event: WorkItem) -> WorkDone:
                    return WorkDone(item=event.item, result=f"ok:{event.item}")

                graph = EventGraph([split, process])
                log = graph.invoke(Batch(items=("only",)))
                assert log.latest(WorkDone) == WorkDone(item="only", result="ok:only")

    def describe_reducer():

        def describe_injection():

            def it_passes_accumulated_values_to_handler():
                def project(event: Event) -> list:
                    if isinstance(event, MsgIn):
                        return [f"in:{event.text}"]
                    if isinstance(event, MsgOut):
                        return [f"out:{event.text}"]
                    return []

                r = Reducer("history", fn=project, default=["start"])
                received_history = []

                @on(MsgIn)
                def respond(event: MsgIn, history: list) -> MsgOut:
                    received_history.extend(history)
                    return MsgOut(text=event.text.upper())

                @on(MsgOut)
                def finish(event: MsgOut) -> Done:
                    return Done(result=event.text)

                graph = EventGraph([respond, finish], reducers=[r])
                log = graph.invoke(MsgIn(text="hello"))
                assert received_history == ["start", "in:hello"]
                assert log.latest(Done) == Done(result="HELLO")

            def it_injects_default_plus_projected_seed():
                def project(event: Event) -> list:
                    if isinstance(event, MsgIn):
                        return [event.text]
                    return []

                r = Reducer("texts", fn=project)

                @on(MsgIn)
                def step(event: MsgIn, log: EventLog, texts: list) -> Done:
                    return Done(result=f"log={len(log)},texts={len(texts)}")

                graph = EventGraph([step], reducers=[r])
                log = graph.invoke(MsgIn(text="hi"))
                assert log.latest(Done) == Done(result="log=1,texts=1")

        def describe_accumulation():

            def it_grows_across_multiple_rounds():
                class ToolResult(Event):
                    result: str = ""

                def project_all(event: Event) -> list:
                    if isinstance(event, MsgIn):
                        return [f"in:{event.text}"]
                    if isinstance(event, MsgOut):
                        return [f"out:{event.text}"]
                    if isinstance(event, ToolResult):
                        return [f"tool:{event.result}"]
                    return []

                r = Reducer("history", fn=project_all)
                call_count = 0
                snapshots: list[list] = []

                @on(MsgIn, ToolResult)
                def call_llm(event: Event, history: list) -> MsgOut:
                    nonlocal call_count
                    call_count += 1
                    snapshots.append(list(history))
                    if isinstance(event, MsgIn):
                        return MsgOut(text="need_tool")
                    return MsgOut(text=f"final:{event.result}")

                @on(MsgOut)
                def handle_response(
                    event: MsgOut,
                ) -> ToolResult | Done:
                    if event.text == "need_tool":
                        return ToolResult(result="42")
                    return Done(result=event.text)

                graph = EventGraph([call_llm, handle_response], reducers=[r])
                graph.invoke(MsgIn(text="question"))
                assert call_count == 2
                assert snapshots[0] == ["in:question"]
                assert snapshots[1] == [
                    "in:question",
                    "out:need_tool",
                    "tool:42",
                ]

            def when_events_have_no_contribution():

                def it_does_not_change_reducer_value():
                    def project(event: Event) -> list:
                        if isinstance(event, MsgIn):
                            return [event.text]
                        return []

                    r = Reducer("texts", fn=project)
                    snapshots: list[list] = []

                    @on(MsgIn)
                    def respond(event: MsgIn, texts: list) -> MsgOut:
                        snapshots.append(list(texts))
                        return MsgOut(text=event.text)

                    @on(MsgOut)
                    def finish(event: MsgOut, texts: list) -> Done:
                        snapshots.append(list(texts))
                        return Done(result="ok")

                    graph = EventGraph([respond, finish], reducers=[r])
                    graph.invoke(MsgIn(text="a"))
                    assert snapshots[0] == ["a"]
                    assert snapshots[1] == ["a"]

        def describe_multiple_reducers():

            def it_accumulates_independently():
                def project_upper(event: Event) -> list:
                    if isinstance(event, MsgIn):
                        return [event.text.upper()]
                    return []

                def project_lower(event: Event) -> list:
                    if isinstance(event, MsgIn):
                        return [event.text.lower()]
                    return []

                upper = Reducer("upper", fn=project_upper, default=["INIT"])
                lower = Reducer("lower", fn=project_lower)

                @on(MsgIn)
                def step(event: MsgIn, upper: list, lower: list) -> Done:
                    return Done(result=f"upper={upper},lower={lower}")

                graph = EventGraph([step], reducers=[upper, lower])
                log = graph.invoke(MsgIn(text="Hello"))
                assert log.latest(Done) == Done(
                    result="upper=['INIT', 'HELLO'],lower=['hello']"
                )

        def describe_parallel_handlers():

            def it_accepts_contributions_from_both():
                class Trigger(Event):
                    value: str = ""

                class ResultA(Event):
                    value: str = ""

                class ResultB(Event):
                    value: str = ""

                class Collected(Event):
                    items: tuple = ()

                def project(event: Event) -> list:
                    if isinstance(event, Trigger):
                        return [f"trigger:{event.value}"]
                    if isinstance(event, ResultA):
                        return [f"a:{event.value}"]
                    if isinstance(event, ResultB):
                        return [f"b:{event.value}"]
                    return []

                r = Reducer("items", fn=project)

                @on(Trigger)
                def handle_a(event: Trigger) -> ResultA:
                    return ResultA(value=event.value)

                @on(Trigger)
                def handle_b(event: Trigger) -> ResultB:
                    return ResultB(value=event.value)

                @on(ResultA, ResultB)
                def collect(
                    event: Event, items: list, log: EventLog
                ) -> Collected | None:
                    if log.has(ResultA) and log.has(ResultB):
                        return Collected(items=tuple(items))
                    return None

                graph = EventGraph([handle_a, handle_b, collect], reducers=[r])
                log = graph.invoke(Trigger(value="x"))
                result = log.latest(Collected)
                assert result is not None
                assert "trigger:x" in result.items
                assert "a:x" in result.items
                assert "b:x" in result.items

        def describe_react_loop():

            def it_accumulates_system_user_assistant_tool_messages():
                class UserMsg(Event):
                    content: str = ""

                class AssistantMsg(Event):
                    content: str = ""
                    needs_tool: bool = False

                class ToolResult(Event):
                    result: str = ""

                class FinalAnswer(Event):
                    answer: str = ""

                def to_messages(event: Event) -> list:
                    if isinstance(event, UserMsg):
                        return [("user", event.content)]
                    if isinstance(event, AssistantMsg):
                        return [("assistant", event.content)]
                    if isinstance(event, ToolResult):
                        return [("tool", event.result)]
                    return []

                r = Reducer(
                    "messages",
                    fn=to_messages,
                    default=[("system", "You are helpful")],
                )
                message_snapshots: list[list] = []

                @on(UserMsg, ToolResult)
                def call_llm(event: Event, messages: list) -> AssistantMsg:
                    message_snapshots.append(list(messages))
                    if isinstance(event, UserMsg):
                        return AssistantMsg(content="need tool", needs_tool=True)
                    return AssistantMsg(content=f"got:{event.result}", needs_tool=False)

                @on(AssistantMsg)
                def handle_response(
                    event: AssistantMsg,
                ) -> ToolResult | FinalAnswer:
                    if event.needs_tool:
                        return ToolResult(result="42")
                    return FinalAnswer(answer=event.content)

                graph = EventGraph([call_llm, handle_response], reducers=[r])
                log = graph.invoke(UserMsg(content="what is 6*7?"))
                assert log.latest(FinalAnswer) == FinalAnswer(answer="got:42")
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

            def it_works_without_any_reducers():
                @on(MsgIn)
                def step(event: MsgIn) -> Done:
                    return Done(result=event.text)

                graph = EventGraph([step])
                log = graph.invoke(MsgIn(text="hello"))
                assert log.latest(Done) == Done(result="hello")

            def it_coexists_with_event_log_injection():
                def project(event: Event) -> list:
                    return [1]

                r = Reducer("counter", fn=project)

                @on(MsgIn)
                def step(event: MsgIn, log: EventLog) -> Done:
                    return Done(result=f"events={len(log)}")

                graph = EventGraph([step], reducers=[r])
                log = graph.invoke(MsgIn(text="hi"))
                assert log.latest(Done) == Done(result="events=1")

        def describe_edge_cases():

            def when_fn_returns_non_list():

                def it_raises_type_error():
                    def bad_project(event: Event) -> list:
                        return "not a list"  # type: ignore

                    r = Reducer("bad", fn=bad_project)

                    @on(MsgIn)
                    def step(event: MsgIn) -> Done:
                        return Done(result="ok")

                    graph = EventGraph([step], reducers=[r])
                    with pytest.raises(TypeError, match="must return a list"):
                        graph.invoke(MsgIn(text="hello"))

            def it_supports_custom_log_parameter_name():
                @on(MsgIn)
                def step(event: MsgIn, event_log: EventLog) -> Done:
                    return Done(result=f"events={len(event_log)}")

                graph = EventGraph([step])
                log = graph.invoke(MsgIn(text="hi"))
                assert log.latest(Done) == Done(result="events=1")

            def when_checkpointer():

                def it_does_not_double_values_on_re_invoke():
                    from langgraph.checkpoint.memory import MemorySaver

                    def project(event: Event) -> list:
                        if isinstance(event, MsgIn):
                            return [event.text]
                        return []

                    r = Reducer("texts", fn=project, default=["init"])

                    @on(MsgIn)
                    def step(event: MsgIn, texts: list) -> Done:
                        return Done(result=",".join(texts))

                    graph = EventGraph([step], reducers=[r])
                    checkpointer = MemorySaver()
                    compiled = graph.compile(checkpointer=checkpointer)

                    config = {"configurable": {"thread_id": "reducer-test"}}
                    result = compiled.invoke({"events": [MsgIn(text="a")]}, config)
                    end_events = [e for e in result["events"] if isinstance(e, Done)]
                    assert end_events[-1].result == "init,a"

                    result = compiled.invoke({"events": [MsgIn(text="b")]}, config)
                    end_events = [e for e in result["events"] if isinstance(e, Done)]
                    assert end_events[-1].result == "init,a,b"

            def it_supports_custom_reducer_function():
                def always_keep_last_n(left: list, right: list) -> list:
                    combined = left + right
                    return combined[-3:]

                class Continue(Event):
                    text: str = ""

                def project_all(event: Event) -> list:
                    if isinstance(event, MsgIn):
                        return [event.text]
                    if isinstance(event, Continue):
                        return [event.text]
                    return []

                r = Reducer(
                    "recent",
                    fn=project_all,
                    reducer=always_keep_last_n,
                    default=["x", "y", "z"],
                )
                snapshots: list[list] = []

                @on(MsgIn, Continue)
                def step(event: Event, recent: list) -> MsgOut | Continue:
                    snapshots.append(list(recent))
                    if isinstance(event, MsgIn):
                        return Continue(text="b")
                    return MsgOut(text="done")

                @on(MsgOut)
                def finish(event: MsgOut) -> Done:
                    return Done(result="ok")

                graph = EventGraph([step, finish], reducers=[r])
                graph.invoke(MsgIn(text="a"))
                assert snapshots[0] == ["y", "z", "a"]
                assert snapshots[1] == ["z", "a", "b"]

    def describe_message_reducer():

        def it_projects_message_events():
            class UserMsg(MessageEvent):
                message: HumanMessage = None  # type: ignore[assignment]

            class Reply(Event):
                text: str = ""

            r = message_reducer([SystemMessage(content="You are helpful")])
            msg = HumanMessage(content="hello")
            result = r.fn(UserMsg(message=msg))
            assert result == [msg]

        def it_returns_empty_for_non_message_events():
            class Reply(Event):
                text: str = ""

            r = message_reducer([SystemMessage(content="You are helpful")])
            result = r.fn(Reply(text="hi"))
            assert result == []

        def it_includes_default_messages():
            r = message_reducer([SystemMessage(content="Be nice")])
            assert len(r.default) == 1
            assert r.default[0].content == "Be nice"

        def it_has_empty_default_when_none_given():
            r = message_reducer()
            assert r.default == []

        def it_respects_custom_channel_name():
            r = message_reducer(name="chat_history")
            assert r.name == "chat_history"

        def describe_integration():

            def it_accumulates_system_and_user_messages():
                class UserMsg(MessageEvent):
                    message: HumanMessage = None  # type: ignore[assignment]

                class BotReply(MessageEvent):
                    message: AIMessage = None  # type: ignore[assignment]

                class Finished(Event):
                    answer: str = ""

                r = message_reducer([SystemMessage(content="You are a test bot")])
                received_messages: list[list[BaseMessage]] = []

                @on(UserMsg)
                def respond(event: UserMsg, messages: list[BaseMessage]) -> BotReply:
                    received_messages.append(list(messages))
                    return BotReply(
                        message=AIMessage(content="I got: " + event.message.content)
                    )

                @on(BotReply)
                def finish(event: BotReply) -> Finished:
                    return Finished(answer=event.message.content)

                graph = EventGraph([respond, finish], reducers=[r])
                log = graph.invoke(UserMsg(message=HumanMessage(content="hello")))
                assert log.latest(Finished) == Finished(answer="I got: hello")
                msgs = received_messages[0]
                assert len(msgs) == 2
                assert msgs[0].content == "You are a test bot"
                assert msgs[1].content == "hello"

            def when_system_prompt_set_seed():

                def it_contributes_to_message_history():
                    class UserMsg(MessageEvent):
                        message: HumanMessage = None  # type: ignore[assignment]

                    class Finished(Event):
                        answer: str = ""

                    r = message_reducer()
                    received_messages: list[list[BaseMessage]] = []

                    @on(UserMsg)
                    def respond(
                        event: UserMsg,
                        messages: list[BaseMessage],
                    ) -> Finished:
                        received_messages.append(list(messages))
                        return Finished(answer="ok")

                    graph = EventGraph([respond], reducers=[r])
                    log = graph.invoke(
                        [
                            SystemPromptSet.from_str("You are a test bot"),
                            UserMsg(message=HumanMessage(content="hello")),
                        ]
                    )
                    assert log.latest(Finished) is not None
                    msgs = received_messages[0]
                    assert len(msgs) == 2
                    assert isinstance(msgs[0], SystemMessage)
                    assert msgs[0].content == "You are a test bot"
                    assert msgs[1].content == "hello"

                def it_is_queryable_in_event_log():
                    class UserMsg(MessageEvent):
                        message: HumanMessage = None  # type: ignore[assignment]

                    class Finished(Event):
                        prompt_content: str = ""

                    r = message_reducer()

                    @on(UserMsg)
                    def respond(event: UserMsg, log: EventLog) -> Finished:
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
                            UserMsg(message=HumanMessage(content="hi")),
                        ]
                    )
                    assert log.latest(Finished) == Finished(
                        prompt_content="You are helpful"
                    )

    def describe_compile():

        def when_checkpointer():

            def it_persists_state():
                from langgraph.checkpoint.memory import MemorySaver

                @on(Start)
                def step(event: Start) -> End:
                    return End(result=event.data)

                graph = EventGraph([step])
                checkpointer = MemorySaver()
                compiled = graph.compile(checkpointer=checkpointer)

                config = {"configurable": {"thread_id": "test-1"}}
                result = compiled.invoke({"events": [Start(data="hello")]}, config)
                assert result["events"][-1] == End(result="hello")

                state = compiled.get_state(config)
                assert len(state.values["events"]) == 2

            def it_only_processes_new_events_on_re_invoke():
                from langgraph.checkpoint.memory import MemorySaver

                seen: list[list[str]] = []

                @on(Start)
                def step(event: Start) -> End:
                    seen.append([event.data])
                    return End(result=event.data)

                graph = EventGraph([step])
                checkpointer = MemorySaver()
                compiled = graph.compile(checkpointer=checkpointer)
                config = {"configurable": {"thread_id": "re-invoke-1"}}

                # Run 1
                compiled.invoke({"events": [Start(data="a")]}, config)
                assert len(seen) == 1
                assert seen[-1] == ["a"]

                # Run 2 — same thread, only Start("b") should be pending
                compiled.invoke({"events": [Start(data="b")]}, config)
                assert len(seen) == 2
                assert seen[-1] == ["b"]

            def it_handles_three_sequential_re_invokes():
                from langgraph.checkpoint.memory import MemorySaver

                @on(Start)
                def step(event: Start) -> End:
                    return End(result=event.data)

                graph = EventGraph([step])
                checkpointer = MemorySaver()
                compiled = graph.compile(checkpointer=checkpointer)
                config = {"configurable": {"thread_id": "re-invoke-3"}}

                compiled.invoke({"events": [Start(data="first")]}, config)
                compiled.invoke({"events": [Start(data="second")]}, config)
                result = compiled.invoke({"events": [Start(data="third")]}, config)

                # Final result only reflects third run's input
                assert result["events"][-1] == End(result="third")

                # Full state has all 6 events (3 Start + 3 End)
                state = compiled.get_state(config)
                assert len(state.values["events"]) == 6

        def it_returns_cached_instance_on_second_call():
            @on(Start)
            def step(event: Start) -> End:
                return End(result=event.data)

            graph = EventGraph([step])
            first = graph.compile()
            second = graph.compile()
            assert first is second

    def describe_stream():

        def it_yields_update_chunks():
            @on(Start)
            def step1(event: Start) -> Middle:
                return Middle(data=event.data)

            @on(Middle)
            def step2(event: Middle) -> End:
                return End(result=event.data)

            graph = EventGraph([step1, step2])
            chunks = list(graph.stream(Start(data="hi"), stream_mode="updates"))
            assert len(chunks) > 0

        async def it_yields_chunks_via_astream():
            @on(Start)
            def step1(event: Start) -> Middle:
                return Middle(data=event.data)

            @on(Middle)
            def step2(event: Middle) -> End:
                return End(result=event.data)

            graph = EventGraph([step1, step2])
            chunks = []
            async for chunk in graph.astream(Start(data="hi"), stream_mode="updates"):
                chunks.append(chunk)
            assert len(chunks) > 0

    def describe_stream_events():

        def it_yields_event_objects():
            @on(Start)
            def step1(event: Start) -> Middle:
                return Middle(data=event.data)

            @on(Middle)
            def step2(event: Middle) -> End:
                return End(result=event.data)

            graph = EventGraph([step1, step2])
            events = list(graph.stream_events(Start(data="hi")))
            assert all(isinstance(e, Event) for e in events)
            types = [type(e).__name__ for e in events]
            assert "Start" in types
            assert "Middle" in types
            assert "End" in types

        def it_yields_events_in_order():
            @on(Start)
            def step1(event: Start) -> Middle:
                return Middle(data="mid")

            @on(Middle)
            def step2(event: Middle) -> End:
                return End(result="done")

            graph = EventGraph([step1, step2])
            events = list(graph.stream_events(Start(data="go")))
            assert isinstance(events[0], Start)
            assert isinstance(events[-1], End)

        def when_multi_seed():

            def it_includes_all_seed_types():
                @on(Start)
                def step(event: Start) -> End:
                    return End(result=event.data)

                graph = EventGraph([step])
                events = list(graph.stream_events([Start(data="a")]))
                types = [type(e).__name__ for e in events]
                assert "Start" in types
                assert "End" in types

        def when_include_reducers_true():

            def it_yields_stream_frames_with_reducer_snapshots():
                reducer = _data_reducer()

                @on(Start)
                def step1(event: Start) -> Middle:
                    return Middle(data=event.data)

                @on(Middle)
                def step2(event: Middle) -> End:
                    return End(result=event.data)

                graph = EventGraph([step1, step2], reducers=[reducer])
                frames = list(
                    graph.stream_events(
                        Start(data="hello"),
                        include_reducers=True,
                    )
                )
                assert all(isinstance(f, StreamFrame) for f in frames)
                types = [type(f.event).__name__ for f in frames]
                assert "Start" in types
                assert "Middle" in types
                assert "End" in types
                seed_frame = next(f for f in frames if isinstance(f.event, Start))
                assert "data_items" in seed_frame.reducers
                assert "hello" in seed_frame.reducers["data_items"]

        def when_include_reducers_selective():

            def it_only_includes_named_reducers():
                reducer = _data_reducer()

                @on(Start)
                def step(event: Start) -> End:
                    return End(result=event.data)

                graph = EventGraph([step], reducers=[reducer])
                frames = list(
                    graph.stream_events(
                        Start(data="x"),
                        include_reducers=["data_items"],
                    )
                )
                assert all(isinstance(f, StreamFrame) for f in frames)
                assert "data_items" in frames[0].reducers

        def when_include_reducers_partial_overlap():

            def it_includes_only_valid_reducer_names():
                reducer = _data_reducer()

                @on(Start)
                def step(event: Start) -> End:
                    return End(result=event.data)

                graph = EventGraph([step], reducers=[reducer])
                frames = list(
                    graph.stream_events(
                        Start(data="x"),
                        include_reducers=["data_items", "nonexistent"],
                    )
                )
                assert all(isinstance(f, StreamFrame) for f in frames)
                # Only the valid reducer appears in snapshots
                assert "data_items" in frames[0].reducers
                assert "nonexistent" not in frames[0].reducers

        def when_include_reducers_unknown_name():

            def it_falls_back_to_bare_events():
                reducer = _data_reducer()

                @on(Start)
                def step(event: Start) -> End:
                    return End(result=event.data)

                graph = EventGraph([step], reducers=[reducer])
                frames = list(
                    graph.stream_events(
                        Start(data="x"),
                        include_reducers=["nonexistent"],
                    )
                )
                assert all(isinstance(f, Event) for f in frames)

        def when_include_reducers_false():

            def it_yields_bare_event_objects():
                @on(Start)
                def step(event: Start) -> End:
                    return End(result=event.data)

                graph = EventGraph([step])
                events = list(graph.stream_events(Start(data="hi")))
                assert all(isinstance(e, Event) for e in events)
                assert not any(isinstance(e, StreamFrame) for e in events)

        @pytest.mark.asyncio
        async def it_works_with_astream_events():
            reducer = _data_reducer()

            @on(Start)
            def step1(event: Start) -> Middle:
                return Middle(data=event.data)

            @on(Middle)
            def step2(event: Middle) -> End:
                return End(result=event.data)

            graph = EventGraph([step1, step2], reducers=[reducer])
            frames = [
                f
                async for f in graph.astream_events(
                    Start(data="async"),
                    include_reducers=True,
                )
            ]
            assert all(isinstance(f, StreamFrame) for f in frames)
            types = [type(f.event).__name__ for f in frames]
            assert "Start" in types
            assert "End" in types
            seed_frame = next(f for f in frames if isinstance(f.event, Start))
            assert "async" in seed_frame.reducers["data_items"]

        def it_accumulates_reducer_values_across_events():
            reducer = _data_reducer()

            class StartA(Start):
                pass

            class StartB(Start):
                pass

            @on(StartA)
            def step_a(event: StartA) -> StartB:
                return StartB(data=f"b_from_{event.data}")

            @on(StartB)
            def step_b(event: StartB) -> End:
                return End(result=event.data)

            graph = EventGraph([step_a, step_b], reducers=[reducer])
            frames = list(graph.stream_events(StartA(data="a1"), include_reducers=True))
            last_frame = frames[-1]
            data_items = last_frame.reducers["data_items"]
            assert "a1" in data_items
            assert "b_from_a1" in data_items

    def describe_reflection_loop():

        class WriteRequest(Event):
            topic: str = ""
            max_revisions: int = 3

        class Draft(Event):
            content: str = ""
            revision: int = 0

        class Critique(Event):
            draft: str = ""
            feedback: str = ""
            revision: int = 0

        class FinalDraft(Event):
            content: str = ""

        def it_terminates_at_max_revisions():
            @on(WriteRequest, Critique)
            def generate(event: Event, log: EventLog) -> Draft:
                if isinstance(event, Critique):
                    return Draft(
                        content=f"revised({event.draft})",
                        revision=event.revision + 1,
                    )
                return Draft(content=f"first_draft({event.topic})")

            @on(Draft)
            def evaluate(event: Draft, log: EventLog) -> Critique | FinalDraft:
                request = log.latest(WriteRequest)
                if event.revision >= request.max_revisions:
                    return FinalDraft(content=event.content)
                return Critique(
                    draft=event.content,
                    feedback="needs work",
                    revision=event.revision,
                )

            graph = EventGraph([generate, evaluate])
            log = graph.invoke(WriteRequest(topic="AI", max_revisions=2))
            assert log.has(FinalDraft)
            final = log.latest(FinalDraft)
            assert "revised" in final.content
            drafts = log.filter(Draft)
            assert len(drafts) == 3

        def it_exits_early_on_pass():
            @on(WriteRequest, Critique)
            def generate(event: Event) -> Draft:
                return Draft(content="perfect", revision=0)

            @on(Draft)
            def evaluate(event: Draft) -> Critique | FinalDraft:
                return FinalDraft(content=event.content)

            graph = EventGraph([generate, evaluate])
            log = graph.invoke(WriteRequest(topic="test"))
            assert log.latest(FinalDraft) == FinalDraft(content="perfect")
            assert len(log.filter(Draft)) == 1

    def describe_safety():

        def describe_return_type_enforcement():

            def it_rejects_list_return():
                @on(Start)
                def bad_handler(event: Start):
                    return [Middle(data="a"), Middle(data="b")]

                graph = EventGraph([bad_handler])
                with pytest.raises(TypeError, match="never a list"):
                    graph.invoke(Start(data="test"))

        def describe_max_rounds():

            def it_detects_infinite_loop():
                class LoopEvent(Event):
                    n: int = 0

                @on(LoopEvent)
                def looper(event: LoopEvent) -> Event:
                    return LoopEvent(n=event.n + 1)

                graph = EventGraph([looper], max_rounds=5)
                with pytest.raises(RuntimeError, match="max_rounds"):
                    graph.invoke(LoopEvent(n=0))

    def describe_mermaid():

        def it_shows_linear_chain_as_edges():
            @on(Start)
            def step1(event: Start) -> Middle:
                return Middle(data=event.data)

            @on(Middle)
            def step2(event: Middle) -> End:
                return End(result=event.data)

            graph = EventGraph([step1, step2])
            output = graph.mermaid()
            assert "graph LR" in output
            assert "Start -->|step1| Middle" in output
            assert "Middle -->|step2| End" in output

        def it_shows_branching_return_types():
            class Good(Event):
                pass

            class Bad(Event):
                pass

            @on(Start)
            def classify(event: Start) -> Good | Bad:
                return Good()

            graph = EventGraph([classify])
            output = graph.mermaid()
            assert "Start -->|classify| Good" in output
            assert "Start -->|classify| Bad" in output

        def it_lists_side_effect_handlers_in_footer():
            @on(Start)
            def side_effect(event: Start) -> None:
                pass

            @on(Start)
            def producer(event: Start) -> End:
                return End(result="ok")

            graph = EventGraph([side_effect, producer])
            output = graph.mermaid()
            assert "%% Side-effect handlers: side_effect (Start)" in output
            assert "Start -->|producer| End" in output

        def it_shows_scatter_in_footer():
            @on(Start)
            def split(event: Start) -> Scatter:
                return Scatter([Middle(data="a")])

            graph = EventGraph([split])
            output = graph.mermaid()
            # No edge to a Scatter node
            assert "-->|split| Scatter" not in output
            assert "%% Scatter handlers: split (Start)" in output

        def it_connects_interrupted_to_resumed_with_dashed_edge():
            @on(Start)
            def request_approval(event: Start) -> Interrupted:
                return Interrupted(prompt="approve?")

            @on(Resumed)
            def handle_review(event: Resumed) -> End:
                return End(result="ok")

            graph = EventGraph([request_approval, handle_review])
            output = graph.mermaid()
            assert "Interrupted -.-> Resumed" in output
            assert "Resumed -->|handle_review| End" in output

        def it_shows_question_mark_for_unannotated_handlers():
            @on(Start)
            def mystery(event: Start):
                return End(result="ok")

            graph = EventGraph([mystery])
            output = graph.mermaid()
            assert "Start -->|mystery| ?" in output

        def it_shows_multi_subscription_edges():
            @on(Start, Middle)
            def handle_both(event: Event) -> End:
                return End(result="ok")

            graph = EventGraph([handle_both])
            output = graph.mermaid()
            assert "Start -->|handle_both| End" in output
            assert "Middle -->|handle_both| End" in output

    def describe_construction_validation():

        def when_no_handlers():

            def it_raises_value_error():
                with pytest.raises(ValueError, match="at least one handler"):
                    EventGraph([])

        def when_duplicate_handler_names():

            def it_deduplicates_with_suffix():
                @on(Start)
                def handler(event: Start) -> Middle:
                    return Middle(data=event.data)

                # Passing the same function twice triggers name dedup
                graph = EventGraph([handler, handler])
                names = [m.name for m in graph._handler_metas]
                assert len(names) == 2
                assert names[0] != names[1]
                assert "_2" in names[1]
