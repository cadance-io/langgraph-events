"""Integration tests for EventGraph — the full event-driven graph engine."""

import pytest

from langgraph_events import (
    Event,
    EventGraph,
    EventLog,
    Halt,
    Interrupted,
    Resumed,
    Scatter,
    SystemPromptSet,
    on,
)

# ---------------------------------------------------------------------------
# Test events
# ---------------------------------------------------------------------------


class Start(Event):
    data: str = ""


class Middle(Event):
    data: str = ""


class End(Event):
    result: str = ""


# ---------------------------------------------------------------------------
# Test: linear chain  A → B → C
# ---------------------------------------------------------------------------


class TestLinearChain:
    def test_three_step_chain(self):
        @on(Start)
        def step1(event: Start) -> Middle:
            return Middle(data=f"processed:{event.data}")

        @on(Middle)
        def step2(event: Middle) -> End:
            return End(result=f"done:{event.data}")

        graph = EventGraph([step1, step2])
        log = graph.invoke(Start(data="hello"))

        assert isinstance(log, EventLog)
        assert len(log) == 3  # Start, Middle, End
        assert log.latest(End) == End(result="done:processed:hello")

    def test_async_handlers(self):
        @on(Start)
        async def step1(event: Start) -> Middle:
            return Middle(data=event.data.upper())

        @on(Middle)
        async def step2(event: Middle) -> End:
            return End(result=event.data)

        graph = EventGraph([step1, step2])
        log = graph.invoke(Start(data="hello"))
        assert log.latest(End) == End(result="HELLO")


# ---------------------------------------------------------------------------
# Test: branching  A → B or C
# ---------------------------------------------------------------------------


class Input(Event):
    kind: str = ""
    data: str = ""


class FastPath(Event):
    data: str = ""


class SlowPath(Event):
    data: str = ""


class Output(Event):
    result: str = ""


class TestBranching:
    def test_conditional_routing(self):
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

        graph = EventGraph([route, handle_fast, handle_slow])

        # Fast path
        log = graph.invoke(Input(kind="fast", data="x"))
        assert log.latest(Output) == Output(result="fast:x")
        assert not log.has(SlowPath)

        # Slow path
        log = graph.invoke(Input(kind="slow", data="y"))
        assert log.latest(Output) == Output(result="slow:y")
        assert not log.has(FastPath)


# ---------------------------------------------------------------------------
# Test: fan-out via inheritance
# ---------------------------------------------------------------------------


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


class TestFanOut:
    def test_multiple_inheritance_triggers_both_handlers(self):
        @on(Trackable)
        def audit(event: Trackable) -> AuditDone:
            return AuditDone(msg=f"audited:{event.action}")

        @on(Processable)
        def process(event: Processable) -> ProcessDone:
            return ProcessDone(msg=f"processed:{event.item}")

        graph = EventGraph([audit, process])
        log = graph.invoke(TrackableItem(action="create", item="doc1"))

        # Both handlers should have fired
        assert log.has(AuditDone)
        assert log.has(ProcessDone)
        assert log.latest(AuditDone) == AuditDone(msg="audited:create")
        assert log.latest(ProcessDone) == ProcessDone(msg="processed:doc1")

    def test_single_inheritance_parent_handler(self):
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

        # Parent handler should fire for child event
        assert log.latest(Result) == Result(v="hello")


# ---------------------------------------------------------------------------
# Test: side-effect handler (returns None)
# ---------------------------------------------------------------------------


class TestSideEffect:
    def test_none_return(self):
        side_effects: list[str] = []

        @on(Start)
        def produce(event: Start) -> Middle:
            return Middle(data=event.data)

        @on(Middle)
        def consume(event: Middle) -> None:
            side_effects.append(event.data)

        graph = EventGraph([produce, consume])
        log = graph.invoke(Start(data="test"))

        assert len(log) == 2  # Start, Middle
        assert side_effects == ["test"]


# ---------------------------------------------------------------------------
# Test: Halt
# ---------------------------------------------------------------------------


class TestHalt:
    def test_halt_stops_execution(self):
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


# ---------------------------------------------------------------------------
# Test: EventLog injection
# ---------------------------------------------------------------------------


class TestEventLogInjection:
    def test_handler_receives_full_log(self):
        @on(Start)
        def step1(event: Start) -> Middle:
            return Middle(data=event.data)

        @on(Middle)
        def step2(event: Middle, log: EventLog) -> End:
            # Should see Start and Middle in the log
            assert log.has(Start)
            count = len(log.filter(Event))
            return End(result=f"saw {count} events")

        graph = EventGraph([step1, step2])
        log = graph.invoke(Start(data="hello"))
        # Log at time of step2: [Start, Middle] = 2 events
        assert log.latest(End) == End(result="saw 2 events")


# ---------------------------------------------------------------------------
# Test: handler return type enforcement
# ---------------------------------------------------------------------------


class TestReturnTypeEnforcement:
    def test_list_return_raises_error(self):
        @on(Start)
        def bad_handler(event: Start):
            return [Middle(data="a"), Middle(data="b")]  # NOT allowed

        graph = EventGraph([bad_handler])
        with pytest.raises(TypeError, match="never a list"):
            graph.invoke(Start(data="test"))


# ---------------------------------------------------------------------------
# Test: max_rounds safety
# ---------------------------------------------------------------------------


class TestMaxRounds:
    def test_infinite_loop_detected(self):
        class LoopEvent(Event):
            n: int = 0

        @on(LoopEvent)
        def looper(event: LoopEvent) -> Event:
            return LoopEvent(n=event.n + 1)  # infinite loop

        graph = EventGraph([looper], max_rounds=5)
        with pytest.raises(RuntimeError, match="max_rounds"):
            graph.invoke(LoopEvent(n=0))


# ---------------------------------------------------------------------------
# Test: checkpointer pass-through
# ---------------------------------------------------------------------------


class TestCheckpointer:
    def test_checkpointer_works(self):
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

        # Verify checkpoint was saved
        state = compiled.get_state(config)
        assert len(state.values["events"]) == 2


# ---------------------------------------------------------------------------
# Test: streaming
# ---------------------------------------------------------------------------


class TestStreaming:
    def test_stream_mode_updates(self):
        @on(Start)
        def step1(event: Start) -> Middle:
            return Middle(data=event.data)

        @on(Middle)
        def step2(event: Middle) -> End:
            return End(result=event.data)

        graph = EventGraph([step1, step2])
        chunks = list(graph.stream(Start(data="hi"), stream_mode="updates"))

        # Should have multiple update chunks
        assert len(chunks) > 0

    def test_stream_events_yields_events(self):
        @on(Start)
        def step1(event: Start) -> Middle:
            return Middle(data=event.data)

        @on(Middle)
        def step2(event: Middle) -> End:
            return End(result=event.data)

        graph = EventGraph([step1, step2])
        events = list(graph.stream_events(Start(data="hi")))

        # Should yield Event objects, not dicts
        from langgraph_events import Event

        assert all(isinstance(e, Event) for e in events)
        # Should contain the seed, middle, and end events
        types = [type(e).__name__ for e in events]
        assert "Start" in types
        assert "Middle" in types
        assert "End" in types

    def test_stream_events_order(self):
        @on(Start)
        def step1(event: Start) -> Middle:
            return Middle(data="mid")

        @on(Middle)
        def step2(event: Middle) -> End:
            return End(result="done")

        graph = EventGraph([step1, step2])
        events = list(graph.stream_events(Start(data="go")))

        # Seed event should come first
        assert isinstance(events[0], Start)
        # End event should come last
        assert isinstance(events[-1], End)

    def test_stream_events_with_multi_seed(self):
        @on(Start)
        def step(event: Start) -> End:
            return End(result=event.data)

        graph = EventGraph([step])
        events = list(graph.stream_events([Start(data="a")]))

        types = [type(e).__name__ for e in events]
        assert "Start" in types
        assert "End" in types


# ---------------------------------------------------------------------------
# Test: interrupt / resume
# ---------------------------------------------------------------------------


class TestInterrupt:
    def test_interrupted_and_resumed(self):
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

        # First invoke — should pause
        result = compiled.invoke({"events": [Start(data="test")]}, config)
        state = compiled.get_state(config)
        assert state.next  # graph is paused

        # Resume with human input
        result = compiled.invoke(
            Command(resume="yes"),
            config,
        )
        events = result["events"]
        end_events = [e for e in events if isinstance(e, End)]
        assert len(end_events) == 1
        assert end_events[0].result == "confirmed:yes"


# ---------------------------------------------------------------------------
# Test: multi-subscription @on(A, B)
# ---------------------------------------------------------------------------


class Ping(Event):
    value: str = ""


class Pong(Event):
    value: str = ""


class Reply(Event):
    value: str = ""


class Done(Event):
    result: str = ""


class TestMultiSubscription:
    def test_handler_fires_on_either_event_type(self):
        """ReAct-style loop: one handler triggered by two different event types."""

        @on(Ping, Pong)
        def echo(event: Event) -> Reply:
            if isinstance(event, Ping):
                return Reply(value=f"ping:{event.value}")
            return Reply(value=f"pong:{event.value}")

        @on(Reply)
        def finish(event: Reply) -> Done:
            return Done(result=event.value)

        graph = EventGraph([echo, finish])

        # Trigger via Ping
        log = graph.invoke(Ping(value="hello"))
        assert log.latest(Done) == Done(result="ping:hello")

        # Trigger via Pong
        log = graph.invoke(Pong(value="world"))
        assert log.latest(Done) == Done(result="pong:world")

    def test_multi_sub_with_log(self):
        """Multi-subscription handler that uses EventLog."""

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

    def test_react_loop_pattern(self):
        """Simulated ReAct loop: call_llm fires on UserMsg and ToolResult."""

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
            # Second call after tool result
            return AssistantMsg(content=f"got:{event.result}", needs_tool=False)

        @on(AssistantMsg)
        def handle_response(event: AssistantMsg) -> ToolResult | FinalAnswer:
            if event.needs_tool:
                return ToolResult(result="42")
            return FinalAnswer(answer=event.content)

        graph = EventGraph([call_llm, handle_response])
        log = graph.invoke(UserMsg(content="what is 6*7?"))

        assert call_count == 2
        assert log.latest(FinalAnswer) == FinalAnswer(answer="got:42")
        assert log.has(ToolResult)
        assert log.has(AssistantMsg)


# ---------------------------------------------------------------------------
# Test: Scatter + gather (map-reduce)
# ---------------------------------------------------------------------------


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


class TestScatter:
    def test_scatter_fan_out_and_gather(self):
        """Map-reduce: Scatter fan-out → parallel processing → gather."""

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

    def test_scatter_single_item(self):
        """Scatter with a single item still works."""

        @on(Batch)
        def split(event: Batch) -> Scatter:
            return Scatter([WorkItem(item=event.items[0])])

        @on(WorkItem)
        def process(event: WorkItem) -> WorkDone:
            return WorkDone(item=event.item, result=f"ok:{event.item}")

        graph = EventGraph([split, process])
        log = graph.invoke(Batch(items=("only",)))

        assert log.latest(WorkDone) == WorkDone(item="only", result="ok:only")


# ---------------------------------------------------------------------------
# Test: reflection loop (generate → critique → revise cycle)
# ---------------------------------------------------------------------------


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


class TestReflectionLoop:
    def test_generate_critique_revise_terminates(self):
        """Reflection loop: generate/critique/revise, terminates at max_revisions."""

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
        # first_draft → critique → revised → critique → revised → final
        assert "revised" in final.content

        # Verify the number of drafts
        drafts = log.filter(Draft)
        assert len(drafts) == 3  # initial + 2 revisions

    def test_reflection_early_pass(self):
        """Reflection loop exits early when critique passes."""

        @on(WriteRequest, Critique)
        def generate(event: Event) -> Draft:
            return Draft(content="perfect", revision=0)

        @on(Draft)
        def evaluate(event: Draft) -> Critique | FinalDraft:
            # Always passes on first try
            return FinalDraft(content=event.content)

        graph = EventGraph([generate, evaluate])
        log = graph.invoke(WriteRequest(topic="test"))

        assert log.latest(FinalDraft) == FinalDraft(content="perfect")
        assert len(log.filter(Draft)) == 1  # only one draft, no revisions


# ---------------------------------------------------------------------------
# Test: EventLog isolation guarantees
# ---------------------------------------------------------------------------


class TestEventLogIsolation:
    def test_log_snapshot_not_affected_by_later_events(self):
        """Handler's log snapshot reflects only events that existed before it ran."""

        log_lengths: list[int] = []

        @on(Start)
        def step1(event: Start) -> Middle:
            return Middle(data="from_step1")

        @on(Middle)
        def step2(event: Middle, log: EventLog) -> End:
            log_lengths.append(len(log))
            # At this point: [Start, Middle] = 2 events
            # End has NOT been produced yet, so it must not appear
            assert not log.has(End)
            return End(result="done")

        graph = EventGraph([step1, step2])
        final_log = graph.invoke(Start(data="test"))

        # step2 saw exactly 2 events (Start + Middle), not 3
        assert log_lengths == [2]
        # Final log has all 3
        assert len(final_log) == 3

    def test_handler_mutating_log_does_not_affect_graph(self):
        """Mutating an injected log's internal list cannot corrupt graph state."""

        @on(Start)
        def evil_handler(event: Start, log: EventLog) -> Middle:
            # Attempt to corrupt the log by mutating internal list
            log._events.append(End(result="INJECTED"))
            log._events.clear()
            return Middle(data="honest")

        @on(Middle)
        def step2(event: Middle, log: EventLog) -> End:
            # step2's log should be clean — no INJECTED event, Start is present
            assert log.has(Start)
            assert log.has(Middle)
            injected = [e for e in log if isinstance(e, End) and e.result == "INJECTED"]
            assert injected == []
            return End(result="clean")

        graph = EventGraph([evil_handler, step2])
        final_log = graph.invoke(Start(data="test"))

        # Final log should have Start, Middle, End(clean) — no INJECTED
        assert len(final_log) == 3
        assert final_log.latest(End) == End(result="clean")
        injected = [
            e for e in final_log if isinstance(e, End) and e.result == "INJECTED"
        ]
        assert injected == []

    def test_parallel_handlers_get_independent_log_snapshots(self):
        """Fan-out handlers each get their own independent EventLog snapshot."""

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
            # Attempt mutation — should NOT affect handler_b's log
            log._events.append(End(result="from_a"))
            return ResultA(saw_events=len(log))

        @on(Trigger)
        def handler_b(event: Trigger, log: EventLog) -> ResultB:
            # Should not see the End event that handler_a injected
            has_end = any(isinstance(e, End) for e in log)
            assert not has_end, "handler_b should not see handler_a's mutation"
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
        # handler_a: log [Trigger] + mutated End → len 2
        # handler_b: independent snapshot [Trigger] → len 1
        # Key: handler_b did NOT see handler_a's mutation
        assert result.b_saw == 1  # handler_b saw only [Trigger]


# ---------------------------------------------------------------------------
# Test: multi-seed events
# ---------------------------------------------------------------------------


class TestMultiSeed:
    def test_list_of_seed_events(self):
        """invoke() accepts a list of seed events."""

        @on(Start)
        def step1(event: Start) -> Middle:
            return Middle(data=event.data)

        @on(Middle)
        def step2(event: Middle) -> End:
            return End(result=event.data)

        graph = EventGraph([step1, step2])
        log = graph.invoke([Start(data="hello")])

        assert log.latest(End) == End(result="hello")

    def test_multiple_seed_events_all_in_log(self):
        """Multiple seed events all appear in the event log."""

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

    def test_system_prompt_set_as_seed(self):
        """SystemPromptSet can be used as a seed event alongside other events."""

        @on(Start)
        def handle(event: Start, log: EventLog) -> End:
            has_prompt = log.has(SystemPromptSet)
            return End(result=f"has_prompt={has_prompt}")

        graph = EventGraph([handle])
        log = graph.invoke([
            SystemPromptSet.from_str("You are helpful"),
            Start(data="go"),
        ])

        assert log.has(SystemPromptSet)
        assert log.latest(End) == End(result="has_prompt=True")

    def test_single_event_still_works(self):
        """Single event (not wrapped in list) still works as before."""

        @on(Start)
        def step(event: Start) -> End:
            return End(result=event.data)

        graph = EventGraph([step])
        log = graph.invoke(Start(data="solo"))

        assert log.latest(End) == End(result="solo")
