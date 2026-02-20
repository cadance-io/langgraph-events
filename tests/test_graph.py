"""Integration tests for EventGraph — the full event-driven graph engine."""

from dataclasses import dataclass

import pytest

from langgraph_events import Event, EventGraph, EventLog, Halt, Interrupted, Resumed, on

# ---------------------------------------------------------------------------
# Test events
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Start(Event):
    data: str = ""


@dataclass(frozen=True)
class Middle(Event):
    data: str = ""


@dataclass(frozen=True)
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


@dataclass(frozen=True)
class Input(Event):
    kind: str = ""
    data: str = ""


@dataclass(frozen=True)
class FastPath(Event):
    data: str = ""


@dataclass(frozen=True)
class SlowPath(Event):
    data: str = ""


@dataclass(frozen=True)
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


@dataclass(frozen=True)
class Auditable(Event):
    action: str = ""


@dataclass(frozen=True)
class Processable(Event):
    item: str = ""


@dataclass(frozen=True)
class AuditableItem(Auditable, Processable):
    action: str = ""
    item: str = ""


@dataclass(frozen=True)
class AuditDone(Event):
    msg: str = ""


@dataclass(frozen=True)
class ProcessDone(Event):
    msg: str = ""


class TestFanOut:
    def test_multiple_inheritance_triggers_both_handlers(self):
        @on(Auditable)
        def audit(event: Auditable) -> AuditDone:
            return AuditDone(msg=f"audited:{event.action}")

        @on(Processable)
        def process(event: Processable) -> ProcessDone:
            return ProcessDone(msg=f"processed:{event.item}")

        graph = EventGraph([audit, process])
        log = graph.invoke(AuditableItem(action="create", item="doc1"))

        # Both handlers should have fired
        assert log.has(AuditDone)
        assert log.has(ProcessDone)
        assert log.latest(AuditDone) == AuditDone(msg="audited:create")
        assert log.latest(ProcessDone) == ProcessDone(msg="processed:doc1")

    def test_single_inheritance_parent_handler(self):
        @dataclass(frozen=True)
        class Base(Event):
            x: str = ""

        @dataclass(frozen=True)
        class Child(Base):
            y: str = ""

        @dataclass(frozen=True)
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
        @dataclass(frozen=True)
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
