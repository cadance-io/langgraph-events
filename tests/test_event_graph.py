"""Integration tests for EventGraph — the full event-driven graph engine."""

import asyncio
import time
import typing
import warnings

import pytest
from conftest import (
    Completed,
    Ended,
    MessageReceived,
    MessageSent,
    Order,
    Processed,
    Started,
)
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END
from langgraph.types import StateUpdate

from langgraph_events import (
    STATE_SNAPSHOT_EVENT_NAME,
    Abandoned,
    Cancelled,
    Command,
    DomainEvent,
    Event,
    EventGraph,
    EventLog,
    Halted,
    HandlerRaised,
    IntegrationEvent,
    Interrupted,
    MaxRoundsExceeded,
    MessageEvent,
    Namespace,
    OrphanedEventWarning,
    Reducer,
    Resumed,
    RunPaused,
    ScalarReducer,
    Scatter,
    SystemPromptSet,
    Unresumable,
    UnresumableError,
    aemit_custom,
    aemit_state_snapshot,
    emit_custom,
    emit_state_snapshot,
    message_reducer,
    on,
)
from langgraph_events.stream import (
    CustomEventFrame,
    LLMStreamEnd,
    LLMToken,
    LLMToolCallChunk,
    StateSnapshotFrame,
    StreamFrame,
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


class _OrphanSuite(Namespace):
    """Namespace with a free-standing DomainEvent — terminal by design."""

    class Analyze(Command):
        text: str = ""

    class Analyzed(DomainEvent):
        label: str = ""


class _ApprovalRequested(Interrupted):
    """Interrupt narrowed on by the field-matcher suites."""

    draft: str = ""


class _OtherInterrupted(Interrupted):
    """Interrupt that must *not* satisfy an ``_ApprovalRequested`` matcher."""

    reason: str = ""


class _ReviewApproved(IntegrationEvent):
    """Resume value for the field-matcher suites."""


class _Acknowledge(IntegrationEvent):
    """Alternative resume value for the field-matcher suites."""


class _OtherEvent(IntegrationEvent):
    """Resume value that must *not* satisfy a ``_ReviewApproved`` matcher."""


class _UserMsgReceived(IntegrationEvent):
    """User turn in the shared ReAct-loop scenario."""

    content: str = ""


class _AssistantMsgSent(IntegrationEvent):
    """Assistant turn in the shared ReAct-loop scenario."""

    content: str = ""
    needs_tool: bool = False


class _ToolResultReturned(IntegrationEvent):
    """Tool result in the shared ReAct-loop scenario."""

    result: str = ""


class _FinalAnswerProduced(IntegrationEvent):
    """Terminal answer in the shared ReAct-loop scenario."""

    answer: str = ""


@on(_AssistantMsgSent)
def _handle_response(
    event: _AssistantMsgSent,
) -> _ToolResultReturned | _FinalAnswerProduced:
    """ReAct branch: run the tool when asked, otherwise finalize."""
    if event.needs_tool:
        return _ToolResultReturned(result="42")
    return _FinalAnswerProduced(answer=event.content)


@on(Started)
def _to_processed(event: Started) -> Processed:
    """``Started`` leg of the deadline / RunPaused scenarios."""
    return Processed(data=event.data)


@on(MessageReceived)
def _to_ended(event: MessageReceived) -> Ended:
    """``MessageReceived`` leg of the deadline / RunPaused scenarios."""
    return Ended(result=event.text)


def _deadline_graph(thread_id: str) -> tuple[EventGraph, dict[str, typing.Any]]:
    """Checkpointed two-handler graph for the deadline suites, plus its config."""
    graph = EventGraph([_to_processed, _to_ended], checkpointer=MemorySaver())
    return graph, {"configurable": {"thread_id": thread_id}}


class _DedupError(Exception):
    """Declared raise used by the handler-dedup suites."""


@on(Started, raises=_DedupError)
def _raiser(event: Started) -> Ended:
    """Handler passed twice so name dedup has to copy its ``raises=``."""
    raise _DedupError("boom")


@on(HandlerRaised, exception=_DedupError)
def _catcher(event: HandlerRaised) -> Ended:
    """Catcher passed twice so name dedup has to copy its field matchers."""
    return Ended(result="caught")


class _Triggered(IntegrationEvent):
    """Seed for the scalar-reducer suites."""

    value: str = ""
    tag: str = ""


class _Unmatched(IntegrationEvent):
    """Event type the scalar-reducer suites reduce over but never emit."""


class _ResultProduced(IntegrationEvent):
    """Terminal event carrying an injected reducer value as text."""

    got: str = ""
    summary: str = ""


class _WidgetSuite(Namespace):
    """Command with a nested DomainEvent outcome — no subscriber."""

    class Place(Command):
        customer_id: str = ""

        class Placed(DomainEvent):
            order_id: str = ""

        def handle(self) -> "_WidgetSuite.Place.Placed":
            return _WidgetSuite.Place.Placed(order_id="o1")


def _data_reducer() -> Reducer:
    """Simple reducer that accumulates Started.data values."""
    return Reducer(name="data_items", event_type=Started, fn=lambda e: [e.data])


class _UserSent(IntegrationEvent, MessageEvent):
    """Inbound chat turn for the LLM-streaming suites."""

    message: HumanMessage = None  # type: ignore[assignment]


class _AgentReplied(IntegrationEvent, MessageEvent):
    """Outbound chat turn for the LLM-streaming suites."""

    message: AIMessage = None  # type: ignore[assignment]


@on(Started)
def _echo(event: Started) -> Ended:
    """Trivial ``Started -> Ended`` step for suites that just need a graph."""
    return Ended(result=event.data)


@on(Started)
def _relay(event: Started) -> Processed:
    """First leg of the shared ``Started -> Processed -> Ended`` chain."""
    return Processed(data=event.data)


@on(Processed)
def _finish(event: Processed) -> Ended:
    """Second leg of the shared ``Started -> Processed -> Ended`` chain."""
    return Ended(result=event.data)


@on(Started)
def _pause_at_step_one(event: Started) -> _StepInterrupted:
    """Interrupts the run so resume-oriented suites have a paused thread."""
    return _StepInterrupted(step=1)


@on(Completed)
def _finish_completed(event: Completed) -> Ended:
    """Resume-side step that turns a ``Completed`` resume value into ``Ended``."""
    return Ended(result=event.result)


def _echo_graph(**kwargs: typing.Any) -> EventGraph:
    """An ``EventGraph`` whose sole handler echoes ``Started`` into ``Ended``."""
    return EventGraph([_echo], **kwargs)


def _chain_graph(**kwargs: typing.Any) -> EventGraph:
    """An ``EventGraph`` running ``Started -> Processed -> Ended``."""
    return EventGraph([_relay, _finish], **kwargs)


def _interruptible_graph(
    thread_id: str, **kwargs: typing.Any
) -> tuple[EventGraph, dict[str, typing.Any]]:
    """A checkpointed graph that pauses on ``Started``, plus its thread config."""
    graph = EventGraph(
        [_pause_at_step_one, _finish_completed],
        checkpointer=MemorySaver(),
        **kwargs,
    )
    return graph, {"configurable": {"thread_id": thread_id}}


def _llm_graph(response: str, prompt: str = "hi") -> EventGraph:
    """A message-reducing graph whose handler replies from a canned LLM."""
    from langchain_core.language_models.fake_chat_models import FakeListChatModel

    llm = FakeListChatModel(responses=[response], sleep=0)

    @on(_UserSent)
    async def reply(event: _UserSent, messages: list[typing.Any]) -> _AgentReplied:
        reply_message = await llm.ainvoke([*messages, HumanMessage(content=prompt)])
        return _AgentReplied(message=reply_message)

    return EventGraph([reply], reducers=[message_reducer()])


async def _adrain(stream: typing.AsyncIterator[typing.Any]) -> list[typing.Any]:
    """Collect an async stream into a list."""
    return [item async for item in stream]


def _fake_astream_events(
    *payloads: dict[str, typing.Any],
) -> typing.Callable[..., typing.AsyncIterator[dict[str, typing.Any]]]:
    """Stand-in for ``compiled.astream_events`` yielding fixed v2 payloads."""

    async def fake(
        *args: typing.Any, **kwargs: typing.Any
    ) -> typing.AsyncIterator[dict[str, typing.Any]]:
        del args, kwargs
        for payload in payloads:
            yield payload

    return fake


def _custom_payload(name: str, data: dict[str, typing.Any]) -> dict[str, typing.Any]:
    """A raw ``on_custom_event`` v2 payload."""
    return {"event": "on_custom_event", "name": name, "data": data}


def _chat_payload(
    chunk: AIMessageChunk, run_id: str = "run-x"
) -> dict[str, typing.Any]:
    """A raw ``on_chat_model_stream`` v2 payload."""
    return {"event": "on_chat_model_stream", "run_id": run_id, "data": {"chunk": chunk}}


def _tool_call_chunk_message(
    content: str = "",
    *,
    name: str = "search",
    args: str = "",
    tool_call_id: str = "tc-1",
    index: int | None = 0,
) -> AIMessageChunk:
    """An ``AIMessageChunk`` with one tool-call chunk; ``index=None`` omits the key."""
    chunk: dict[str, typing.Any] = {
        "name": name,
        "args": args,
        "id": tool_call_id,
        "type": "tool_call_chunk",
    }
    if index is not None:
        chunk["index"] = index
    return AIMessageChunk(content=content, tool_call_chunks=[chunk])


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

            class InputReceived(IntegrationEvent):
                kind: str = ""
                data: str = ""

            class FastPathChosen(IntegrationEvent):
                data: str = ""

            class SlowPathChosen(IntegrationEvent):
                data: str = ""

            class OutputProduced(IntegrationEvent):
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

            class Tracked(IntegrationEvent):
                action: str = ""

            class ProcessCompleted(IntegrationEvent):
                item: str = ""

            class TrackedItem(Tracked, ProcessCompleted):
                action: str = ""
                item: str = ""

            class AuditDone(IntegrationEvent):
                msg: str = ""

            class ProcessDone(IntegrationEvent):
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
                class BaseReceived(IntegrationEvent):
                    x: str = ""

                class ChildReceived(BaseReceived):
                    y: str = ""

                class ResultProduced(IntegrationEvent):
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
                class Triggered(IntegrationEvent):
                    value: str = ""

                class ResultAProduced(IntegrationEvent):
                    saw_events: int = 0

                class ResultBProduced(IntegrationEvent):
                    saw_events: int = 0

                class Collected(IntegrationEvent):
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

            class PingSent(IntegrationEvent):
                value: str = ""

            class PongReceived(IntegrationEvent):
                value: str = ""

            class Replied(IntegrationEvent):
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
                    class MsgAReceived(IntegrationEvent):
                        text: str = ""

                    class MsgBReceived(IntegrationEvent):
                        text: str = ""

                    class Summarized(IntegrationEvent):
                        count: int = 0

                    @on(MsgAReceived, MsgBReceived)
                    def summarize(event: Event, log: EventLog) -> Summarized:
                        total = len(log.filter(Event))
                        return Summarized(count=total)

                    graph = EventGraph([summarize])
                    log = graph.invoke(MsgAReceived(text="hi"))
                    assert log.latest(Summarized) == Summarized(count=1)

                def it_supports_react_loop_pattern():
                    call_count = 0

                    @on(_UserMsgReceived, _ToolResultReturned)
                    def call_llm(event: Event, log: EventLog) -> _AssistantMsgSent:
                        nonlocal call_count
                        call_count += 1
                        if isinstance(event, _UserMsgReceived):
                            return _AssistantMsgSent(
                                content="need tool", needs_tool=True
                            )
                        return _AssistantMsgSent(
                            content=f"got:{event.result}",
                            needs_tool=False,
                        )

                    graph = EventGraph([call_llm, _handle_response])
                    log = graph.invoke(_UserMsgReceived(content="what is 6*7?"))
                    assert call_count == 2
                    assert log.latest(_FinalAnswerProduced) == (
                        _FinalAnswerProduced(answer="got:42")
                    )
                    assert log.has(_ToolResultReturned)
                    assert log.has(_AssistantMsgSent)

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
                class ConfigSet(IntegrationEvent):
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
            class Confirmed(IntegrationEvent):
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

            class Confirmed(IntegrationEvent):
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
            class Confirmed(IntegrationEvent):
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

            class ApprovalSubmitted(IntegrationEvent):
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

            class UserMsgReceived(IntegrationEvent, MessageEvent):
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

        def _resume_after_interrupt(handlers, resume_with, thread_id: str) -> EventLog:
            """Interrupt a checkpointed graph, then resume it with ``resume_with``."""
            graph = EventGraph(handlers, checkpointer=MemorySaver())
            config = {"configurable": {"thread_id": thread_id}}
            graph.invoke(Started(data="test"), config=config)
            return graph.resume(resume_with, config=config)

        def when_field_matches():

            def it_dispatches_the_handler():
                captured = []

                @on(Started)
                def need_input(event: Started) -> _ApprovalRequested:
                    return _ApprovalRequested(draft="hello")

                @on(Resumed, interrupted=_ApprovalRequested)
                def handle_approval(event: Resumed) -> Ended:
                    captured.append(event.interrupted)
                    return Ended(result="approved")

                log = _resume_after_interrupt(
                    [need_input, handle_approval],
                    _ReviewApproved(),
                    "field-match-test",
                )

                assert log.latest(Ended) == Ended(result="approved")
                assert len(captured) == 1
                assert isinstance(captured[0], _ApprovalRequested)

        def when_field_does_not_match():

            def it_skips_the_handler():
                captured = []

                @on(Started)
                def need_input(event: Started) -> _OtherInterrupted:
                    return _OtherInterrupted(reason="different")

                @on(Resumed, interrupted=_ApprovalRequested)
                def handle_approval(event: Resumed) -> Ended:
                    captured.append("should not fire")
                    return Ended(result="approved")

                @on(_ReviewApproved)
                def fallback(event: _ReviewApproved) -> Ended:
                    return Ended(result="fallback")

                log = _resume_after_interrupt(
                    [need_input, handle_approval, fallback],
                    _ReviewApproved(),
                    "field-no-match-test",
                )

                assert len(captured) == 0
                assert log.latest(Ended) == Ended(result="fallback")

        def when_field_is_none():

            def it_skips_the_handler():
                """A None field value does not match a field matcher."""
                captured = []

                @on(Started)
                def need_input(event: Started) -> _OtherInterrupted:
                    return _OtherInterrupted(reason="test")

                @on(Resumed, interrupted=_ApprovalRequested)
                def approval_handler(event: Resumed) -> Ended:
                    captured.append("should not fire")
                    return Ended(result="approval")

                @on(_Acknowledge)
                def fallback(event: _Acknowledge) -> Ended:
                    return Ended(result="fallback")

                log = _resume_after_interrupt(
                    [need_input, approval_handler, fallback],
                    _Acknowledge(),
                    "field-none-test",
                )

                assert len(captured) == 0
                assert log.latest(Ended) == Ended(result="fallback")

        def when_handler_requests_field_injection():

            def it_injects_the_narrowed_field():
                injected_values = []

                @on(Started)
                def need_input(event: Started) -> _ApprovalRequested:
                    return _ApprovalRequested(draft="my draft")

                @on(Resumed, interrupted=_ApprovalRequested)
                def handle_approval(
                    event: Resumed, interrupted: _ApprovalRequested
                ) -> Ended:
                    injected_values.append(interrupted)
                    return Ended(result=interrupted.draft)

                log = _resume_after_interrupt(
                    [need_input, handle_approval],
                    _ReviewApproved(),
                    "field-inject-test",
                )

                assert log.latest(Ended) == Ended(result="my draft")
                assert len(injected_values) == 1
                assert isinstance(injected_values[0], _ApprovalRequested)
                assert injected_values[0].draft == "my draft"

        def when_multiple_field_matchers():

            def it_requires_all_fields_to_match():
                captured = []

                @on(Started)
                def need_input(event: Started) -> _ApprovalRequested:
                    return _ApprovalRequested(draft="hello")

                @on(Resumed, interrupted=_ApprovalRequested, value=_ReviewApproved)
                def strict_handler(
                    event: Resumed,
                    interrupted: _ApprovalRequested,
                    value: _ReviewApproved,
                ) -> Ended:
                    captured.append((interrupted, value))
                    return Ended(result="strict")

                log = _resume_after_interrupt(
                    [need_input, strict_handler],
                    _ReviewApproved(),
                    "multi-field-test",
                )

                assert log.latest(Ended) == Ended(result="strict")
                assert len(captured) == 1
                assert isinstance(captured[0][0], _ApprovalRequested)
                assert isinstance(captured[0][1], _ReviewApproved)

            def when_one_field_does_not_match():

                def it_skips_the_handler():
                    captured = []

                    @on(Started)
                    def need_input(event: Started) -> _ApprovalRequested:
                        return _ApprovalRequested(draft="hello")

                    # value=_OtherEvent won't match _ReviewApproved
                    @on(Resumed, interrupted=_ApprovalRequested, value=_OtherEvent)
                    def strict_handler(event: Resumed) -> Ended:
                        captured.append("should not fire")
                        return Ended(result="strict")

                    @on(_ReviewApproved)
                    def fallback(event: _ReviewApproved) -> Ended:
                        return Ended(result="fallback")

                    log = _resume_after_interrupt(
                        [need_input, strict_handler, fallback],
                        _ReviewApproved(),
                        "multi-field-skip",
                    )

                    assert len(captured) == 0
                    assert log.latest(Ended) == Ended(result="fallback")

    def describe_scatter():

        class BatchReceived(IntegrationEvent):
            items: tuple = ()

        class WorkItemDispatched(IntegrationEvent):
            item: str = ""
            batch_size: int = 0

        class WorkDone(IntegrationEvent):
            item: str = ""
            result: str = ""

        class BatchResultCollected(IntegrationEvent):
            results: tuple = ()

        def describe_construction():

            def when_valid_events():

                def it_wraps_a_list_of_events():
                    class ItemDispatched(IntegrationEvent):
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
                def split(event: BatchReceived) -> Scatter[WorkItemDispatched]:
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
                def split(event: BatchReceived) -> Scatter[WorkItemDispatched]:
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
                    class ToolResultReturned(IntegrationEvent):
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
                class Triggered(IntegrationEvent):
                    value: str = ""

                class ResultAProduced(IntegrationEvent):
                    value: str = ""

                class ResultBProduced(IntegrationEvent):
                    value: str = ""

                class Collected(IntegrationEvent):
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
                def to_messages(event: Event) -> list:
                    if isinstance(event, _UserMsgReceived):
                        return [("user", event.content)]
                    if isinstance(event, _AssistantMsgSent):
                        return [("assistant", event.content)]
                    if isinstance(event, _ToolResultReturned):
                        return [("tool", event.result)]
                    return []

                r = Reducer(
                    "messages",
                    event_type=Event,
                    fn=to_messages,
                    default=[("system", "You are helpful")],
                )
                message_snapshots: list[list] = []

                @on(_UserMsgReceived, _ToolResultReturned)
                def call_llm(event: Event, messages: list) -> _AssistantMsgSent:
                    message_snapshots.append(list(messages))
                    if isinstance(event, _UserMsgReceived):
                        return _AssistantMsgSent(content="need tool", needs_tool=True)
                    return _AssistantMsgSent(
                        content=f"got:{event.result}",
                        needs_tool=False,
                    )

                graph = EventGraph([call_llm, _handle_response], reducers=[r])
                log = graph.invoke(_UserMsgReceived(content="what is 6*7?"))
                assert log.latest(_FinalAnswerProduced) == (
                    _FinalAnswerProduced(answer="got:42")
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

                    class Continued(IntegrationEvent):
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
                class StrategyChosen(IntegrationEvent):
                    strategy: str = ""

                class TaskDone(IntegrationEvent):
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
                class StepCompleted(IntegrationEvent):
                    value: str = ""

                class Finalized(IntegrationEvent):
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
                class StepCompleted(IntegrationEvent):
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
                sr = ScalarReducer(
                    name="mode",
                    event_type=_Unmatched,
                    fn=lambda e: "irrelevant",
                )

                @on(_Triggered)
                def handle(event: _Triggered, mode: object) -> _ResultProduced:
                    return _ResultProduced(got=str(mode))

                graph = EventGraph([handle], reducers=[sr])
                log = graph.invoke(_Triggered())
                assert log.latest(_ResultProduced) == _ResultProduced(got="None")

            def it_returns_skip():
                class StepCompleted(IntegrationEvent):
                    pass

                class OtherReceived(IntegrationEvent):
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
                from langgraph_events import SKIP

                sr = ScalarReducer(
                    name="mode",
                    event_type=_Triggered,
                    fn=lambda e: SKIP,
                    default="fallback",
                )

                result = sr.collect([_Triggered()])
                assert result is SKIP
                assert sr.has_contributions(result) is False
                assert sr.seed([_Triggered()]) == "fallback"

            def it_uses_custom_default():
                sr = ScalarReducer(
                    name="mode",
                    event_type=_Unmatched,
                    fn=lambda e: "irrelevant",
                    default="fallback",
                )

                @on(_Triggered)
                def handle(event: _Triggered, mode: str) -> _ResultProduced:
                    return _ResultProduced(got=mode)

                graph = EventGraph([handle], reducers=[sr])
                log = graph.invoke(_Triggered())
                assert log.latest(_ResultProduced) == _ResultProduced(got="fallback")

        def when_mixed_list_reducers():

            def it_works_alongside_list_reducers():
                list_r = Reducer(
                    name="tags",
                    event_type=_Triggered,
                    fn=lambda e: [e.tag] if e.tag else [],
                )
                scalar_r = ScalarReducer(
                    name="last_tag",
                    event_type=_Triggered,
                    fn=lambda e: e.tag,
                )

                @on(_Triggered)
                def handle(
                    event: _Triggered,
                    tags: list,
                    last_tag: object,
                ) -> _ResultProduced:
                    return _ResultProduced(summary=f"tags={tags},last={last_tag}")

                graph = EventGraph([handle], reducers=[list_r, scalar_r])
                log = graph.invoke(_Triggered(tag="x"))
                assert log.latest(_ResultProduced) == (
                    _ResultProduced(summary="tags=['x'],last=x")
                )

        def when_parallel_handlers():

            def it_handles_parallel_handler_contributions():
                class ResultAProduced(IntegrationEvent):
                    data: str = ""

                class ResultBProduced(IntegrationEvent):
                    data: str = ""

                sr = ScalarReducer(
                    name="latest",
                    event_type=Event,
                    fn=lambda e: (
                        e.value
                        if isinstance(e, _Triggered)
                        else (
                            e.data
                            if isinstance(e, (ResultAProduced, ResultBProduced))
                            else None
                        )
                    ),
                )

                @on(_Triggered)
                def handler_a(event: _Triggered, latest: object) -> ResultAProduced:
                    return ResultAProduced(data=f"a:{event.value}")

                @on(_Triggered)
                def handler_b(event: _Triggered, latest: object) -> ResultBProduced:
                    return ResultBProduced(data=f"b:{event.value}")

                graph = EventGraph([handler_a, handler_b], reducers=[sr])
                log = graph.invoke(_Triggered(value="x"))
                # Both handlers run in parallel — should not crash
                assert log.has(ResultAProduced)
                assert log.has(ResultBProduced)

        def when_subsequent_round_has_no_contribution():

            def it_persists_value():
                class ValueSet(IntegrationEvent):
                    value: str = ""

                class UnrelatedReceived(IntegrationEvent):
                    pass

                sr = ScalarReducer(
                    name="kept",
                    event_type=ValueSet,
                    fn=lambda e: e.value,
                )

                @on(ValueSet)
                def step1(event: ValueSet) -> UnrelatedReceived:
                    return UnrelatedReceived()

                @on(UnrelatedReceived)
                def step2(event: UnrelatedReceived, kept: object) -> _ResultProduced:
                    return _ResultProduced(got=str(kept))

                graph = EventGraph([step1, step2], reducers=[sr])
                log = graph.invoke(ValueSet(value="hello"))
                # Round 2 produces UnrelatedReceived (doesn't match event_type) —
                # scalar must still be "hello", not reverted.
                assert log.latest(_ResultProduced) == _ResultProduced(got="hello")

        def when_fn_returns_none():

            def it_stores_none_as_valid_contribution():
                class ClearSignaled(IntegrationEvent):
                    pass

                sr = ScalarReducer(
                    name="value",
                    event_type=ClearSignaled,
                    fn=lambda e: None,
                    default="initial",
                )

                @on(ClearSignaled)
                def handle(event: ClearSignaled, value: object) -> _ResultProduced:
                    return _ResultProduced(got=repr(value))

                graph = EventGraph([handle], reducers=[sr])
                log = graph.invoke(ClearSignaled())
                # fn returns None — this is a real contribution, not "no contribution"
                assert log.latest(_ResultProduced) == _ResultProduced(got="None")

        def when_protocol_event_type():

            def it_supports_protocol_event_type():
                from typing import Protocol, runtime_checkable

                @runtime_checkable
                class HasScore(Protocol):
                    score: int

                class ScoreARecorded(IntegrationEvent):
                    score: int = 0

                class ScoreBRecorded(IntegrationEvent):
                    score: int = 0

                sr = ScalarReducer(
                    name="last_score",
                    event_type=HasScore,
                    fn=lambda e: e.score,
                )

                @on(ScoreARecorded)
                def step_a(event: ScoreARecorded, last_score: object) -> ScoreBRecorded:
                    return ScoreBRecorded(score=event.score + 10)

                @on(ScoreBRecorded)
                def step_b(
                    event: ScoreBRecorded, last_score: object
                ) -> _ResultProduced:
                    return _ResultProduced(got=str(last_score))

                graph = EventGraph([step_a, step_b], reducers=[sr])
                log = graph.invoke(ScoreARecorded(score=5))
                # ScoreA(5) → 5, then ScoreB(15) → 15
                assert log.latest(_ResultProduced) == _ResultProduced(got="15")

        def when_checkpointer():

            def it_does_not_lose_scalar_on_re_invoke():
                from langgraph.checkpoint.memory import MemorySaver

                sr = ScalarReducer(
                    name="latest",
                    event_type=_Triggered,
                    fn=lambda e: e.value,
                )

                @on(_Triggered)
                def handle(event: _Triggered, latest: object) -> _ResultProduced:
                    return _ResultProduced(got=str(latest))

                graph = EventGraph([handle], reducers=[sr], checkpointer=MemorySaver())
                config = {"configurable": {"thread_id": "scalar-re-invoke"}}

                # Run 1
                log1 = graph.invoke(_Triggered(value="first"), config=config)
                assert log1.latest(_ResultProduced) == _ResultProduced(got="first")

                # Run 2 — re-invoke on same thread
                log2 = graph.invoke(_Triggered(value="second"), config=config)
                assert log2.latest(_ResultProduced) == _ResultProduced(got="second")

    def describe_message_reducer():

        def when_defaults_provided():

            def it_projects_message_events():
                class UserMsgReceived(IntegrationEvent, MessageEvent):
                    message: HumanMessage = None  # type: ignore[assignment]

                class Replied(IntegrationEvent):
                    text: str = ""

                r = message_reducer([SystemMessage(content="You are helpful")])
                msg = HumanMessage(content="hello")
                result = r.fn(UserMsgReceived(message=msg))
                assert result == [msg]

            def it_skips_non_message_events_at_collect_level():
                class Replied(IntegrationEvent):
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
                    class UserMsgReceived(IntegrationEvent, MessageEvent):
                        message: HumanMessage = None  # type: ignore[assignment]

                    class BotReplied(IntegrationEvent, MessageEvent):
                        message: AIMessage = None  # type: ignore[assignment]

                    class Finished(IntegrationEvent):
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
                    class UserMsgReceived(IntegrationEvent, MessageEvent):
                        message: HumanMessage = None  # type: ignore[assignment]

                    class Finished(IntegrationEvent):
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
                    class UserMsgReceived(IntegrationEvent, MessageEvent):
                        message: HumanMessage = None  # type: ignore[assignment]

                    class Finished(IntegrationEvent):
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
                graph = _echo_graph()
                first = graph.compiled
                second = graph.compiled
                assert first is second

        def when_checkpointer():

            def it_persists_state():
                graph = _echo_graph(checkpointer=MemorySaver())

                config = {"configurable": {"thread_id": "test-1"}}
                log = graph.invoke(Started(data="hello"), config=config)
                assert log[-1] == Ended(result="hello")

                state = graph.get_state(config)
                assert len(state.events) == 2

            def it_only_processes_new_events_on_re_invoke():
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
                graph = _echo_graph(checkpointer=MemorySaver())
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
                graph = _chain_graph()
                events = list(graph.stream_events(Started(data="hi")))
                assert all(isinstance(e, Event) for e in events)
                types = [type(e).__name__ for e in events]
                assert "Started" in types
                assert "Processed" in types
                assert "Ended" in types

            def it_yields_events_in_order():
                graph = _chain_graph()
                events = list(graph.stream_events(Started(data="go")))
                assert isinstance(events[0], Started)
                assert isinstance(events[-1], Ended)

        def when_multi_seed():

            def it_includes_all_seed_types():
                graph = _echo_graph()
                events = list(graph.stream_events([Started(data="a")]))
                types = [type(e).__name__ for e in events]
                assert "Started" in types
                assert "Ended" in types

        def when_include_reducers_true():

            def it_yields_stream_frames():
                graph = _chain_graph(reducers=[_data_reducer()])
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
                graph = _echo_graph(reducers=[_data_reducer()])
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
                graph = _echo_graph(reducers=[_data_reducer()])
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
                graph = _echo_graph(reducers=[_data_reducer()])
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
                graph = _echo_graph(reducers=[_data_reducer()])
                frames = list(
                    graph.stream_events(
                        Started(data="x"),
                        include_reducers=["nonexistent"],
                    )
                )
                assert all(isinstance(f, Event) for f in frames)

        def when_include_reducers_false():

            def it_yields_bare_event_objects():
                graph = _echo_graph()
                events = list(graph.stream_events(Started(data="hi")))
                assert all(isinstance(e, Event) for e in events)
                assert not any(isinstance(e, StreamFrame) for e in events)

        def when_async():

            @pytest.mark.asyncio
            async def it_yields_stream_frames():
                graph = _chain_graph(reducers=[_data_reducer()])
                frames = await _adrain(
                    graph.astream_events(
                        Started(data="async"),
                        include_reducers=True,
                    )
                )
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

        async def _v2_resume_frames(graph, config):
            """Resume through the v2 path; return the ``StreamFrame``s yielded."""
            frames = await _adrain(
                graph.astream_resume(
                    Completed(result="done"),
                    include_reducers=True,
                    include_custom_events=True,
                    config=config,
                )
            )
            return [f for f in frames if isinstance(f, StreamFrame)]

        def it_yields_resume_handler_events():
            graph, config = _interruptible_graph("sr-handler")
            graph.invoke(Started(data="go"), config=config)

            events = list(graph.stream_resume(Completed(result="done"), config=config))
            types = [type(e).__name__ for e in events]
            assert "Ended" in types

        def it_includes_stale_interrupted_in_raw_stream():
            graph, config = _interruptible_graph("sr-no-stale")
            graph.invoke(Started(data="go"), config=config)

            events = list(graph.stream_resume(Completed(result="done"), config=config))
            # Raw stream_resume is semantically complete — stale Interrupted appears
            assert any(isinstance(e, Interrupted) for e in events)

        def it_yields_reducer_stream_frames():
            graph, config = _interruptible_graph(
                "sr-reducers", reducers=[_data_reducer()]
            )
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
            graph, config = _interruptible_graph("sr-async")
            await graph.ainvoke(Started(data="go"), config=config)

            events = await _adrain(
                graph.astream_resume(Completed(result="async-done"), config=config)
            )
            types = [type(e).__name__ for e in events]
            assert "Ended" in types

        def it_yields_new_interrupted_during_resume():
            @on(Completed)
            def step_two(event: Completed) -> _StepInterrupted:
                return _StepInterrupted(step=2)

            graph = EventGraph(
                [_pause_at_step_one, step_two],
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
            r = Reducer("texts", event_type=MessageReceived, fn=lambda e: [e.text])
            graph, config = _interruptible_graph("v2-resume-reducer", reducers=[r])

            # First run — seed with MessageReceived to populate reducer, then
            # interrupt via Started → _StepInterrupted.
            graph.invoke(
                [MessageReceived(text="hello"), Started(data="go")], config=config
            )

            # Resume via _astream_v2 path (include_custom_events forces v2)
            stream_frames = await _v2_resume_frames(graph, config)
            assert len(stream_frames) > 0
            # Reducer must reflect checkpoint state ("hello" from first run)
            assert "hello" in stream_frames[0].reducers["texts"]

        @pytest.mark.asyncio
        async def it_accumulates_reducer_across_v2_astream_events_runs():
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

            def _v2_run(text: str):
                return graph.astream_events(
                    MessageReceived(text=text),
                    include_reducers=True,
                    include_custom_events=True,
                    config=config,
                )

            # First run via astream_events (v2 path)
            _ = await _adrain(_v2_run("first"))

            # Second run on same thread — seed contributes on top of checkpoint
            frames = await _adrain(_v2_run("second"))

            stream_frames = [f for f in frames if isinstance(f, StreamFrame)]
            assert len(stream_frames) > 0
            texts = stream_frames[0].reducers["texts"]
            assert "first" in texts
            assert "second" in texts

        @pytest.mark.asyncio
        async def it_preserves_scalar_reducer_from_checkpoint_in_v2():
            sr = ScalarReducer(
                name="proposal",
                event_type=MessageReceived,
                fn=lambda e: e.text,
            )
            graph, config = _interruptible_graph("v2-scalar-resume", reducers=[sr])

            # First run — MessageReceived populates scalar, Started interrupts
            graph.invoke(
                [MessageReceived(text="chosen"), Started(data="go")], config=config
            )

            # Resume via v2 path
            stream_frames = await _v2_resume_frames(graph, config)
            assert len(stream_frames) > 0
            assert stream_frames[0].reducers["proposal"] == "chosen"

    def describe_reflection_loop():

        class WriteRequested(IntegrationEvent):
            topic: str = ""
            max_revisions: int = 3

        class DraftProduced(IntegrationEvent):
            content: str = ""
            revision: int = 0

        class CritiqueReceived(IntegrationEvent):
            draft: str = ""
            feedback: str = ""
            revision: int = 0

        class FinalDraftProduced(IntegrationEvent):
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
                class LoopDetected(IntegrationEvent):
                    n: int = 0

                @on(LoopDetected)
                def looper(event: LoopDetected) -> LoopDetected:
                    return LoopDetected(n=event.n + 1)

                graph = EventGraph([looper], max_rounds=5)
                log = graph.invoke(LoopDetected(n=0))
                assert log.latest(MaxRoundsExceeded) is not None

            def it_resets_round_counter_on_resume():
                from langgraph.checkpoint.memory import MemorySaver

                class ResumeConfirmed(IntegrationEvent):
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

                class PingSent(IntegrationEvent):
                    n: int = 0

                @on(PingSent)
                def pong(event: PingSent) -> PingSent:
                    return PingSent(n=event.n + 1)

                graph = EventGraph([pong], max_rounds=5)
                log = graph.invoke(PingSent())
                assert log.latest(MaxRoundsExceeded) is not None

            def it_halts_on_max_rounds_for_multiple_handlers():
                """recursion_limit accounts for multiple handlers per round."""

                class Ticked(IntegrationEvent):
                    n: int = 0

                class Tocked(IntegrationEvent):
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

            def it_respects_explicit_recursion_limit_kwarg():
                """Explicit recursion_limit kwarg wins over auto-computed."""

                class Tick(IntegrationEvent):
                    pass

                @on(Tick)
                def noop(event: Tick) -> None:
                    return None

                graph = EventGraph([noop], max_rounds=10, recursion_limit=12345)
                assert graph.compiled.config["recursion_limit"] == 12345

            def it_saves_clean_checkpoint_on_max_rounds():
                from langgraph.checkpoint.memory import MemorySaver

                class LoopEvent(IntegrationEvent):
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
                class LoopEvent(IntegrationEvent):
                    n: int = 0

                @on(LoopEvent)
                def looper(event: LoopEvent) -> LoopEvent:
                    return LoopEvent(n=event.n + 1)

                graph = EventGraph([looper], max_rounds=3)
                events = list(graph.stream_events(LoopEvent(n=0)))
                assert any(isinstance(e, MaxRoundsExceeded) for e in events)

        def describe_deadline():

            def when_deadline_is_already_expired():

                def it_emits_RunPaused():
                    @on(Started)
                    def handler(event: Started) -> Ended:
                        return Ended(result="should-not-matter")

                    graph = EventGraph([handler])
                    log = graph.invoke(Started(data="go"), deadline=0.0)
                    assert log.latest(RunPaused) is not None

            def when_using_astream_events():

                async def it_emits_RunPaused_in_the_async_stream():
                    @on(Started)
                    def handler(event: Started) -> Ended:
                        return Ended(result="never")

                    graph = EventGraph([handler])
                    events: list[Event] = []
                    async for event in graph.astream_events(
                        Started(data="go"), deadline=0.0
                    ):
                        events.append(event)
                    assert any(isinstance(e, RunPaused) for e in events)

            def when_resuming_an_interrupted_run():

                def it_respects_deadline_on_the_resume_path():
                    """``deadline=`` flows through the ``resume()`` entry
                    point the same way it flows through ``invoke()`` — the
                    router emits ``RunPaused`` between dispatch rounds.
                    Pins the contract that the shared
                    ``_apply_deadline_kwarg`` injector covers the resume
                    path too.
                    """
                    from langgraph.checkpoint.memory import MemorySaver

                    class ConfirmationRequested(Interrupted):
                        pass

                    class Confirmed(IntegrationEvent):
                        pass

                    @on(Started)
                    def need_input(event: Started) -> ConfirmationRequested:
                        return ConfirmationRequested()

                    @on(Confirmed)
                    def handle_confirm(event: Confirmed) -> Ended:
                        return Ended(result="confirmed")

                    graph = EventGraph(
                        [need_input, handle_confirm],
                        checkpointer=MemorySaver(),
                    )
                    config = {"configurable": {"thread_id": "resume-deadline"}}
                    graph.invoke(Started(data="test"), config=config)

                    log = graph.resume(Confirmed(), config=config, deadline=0.0)
                    assert log.latest(RunPaused) is not None
                    assert log.latest(Ended) is None

            def when_a_fresh_run_follows_a_paused_run():

                def it_continues_past_the_old_RunPaused():
                    """Fresh /run on the same thread excludes the old
                    RunPaused from new_events (cursor advanced past it)
                    and dispatches the new seeds normally.

                    This is the resumable semantic that justifies
                    RunPaused being a SystemEvent rather than a Halted
                    subclass — MaxRoundsExceeded would re-terminate here.
                    """
                    graph, config = _deadline_graph("deadline-resume")

                    # Run 1: deadline already expired → RunPaused emitted.
                    log1 = graph.invoke(
                        Started(data="first"),
                        deadline=0.0,
                        config=config,
                    )
                    assert log1.latest(RunPaused) is not None

                    # Run 2: same thread, no deadline → must continue
                    # normally. The to_ended handler should fire on the
                    # new MessageReceived seed.
                    log2 = graph.invoke(
                        MessageReceived(text="second"),
                        config=config,
                    )
                    assert log2.latest(Ended) is not None
                    assert log2.latest(Ended).result == "second"

            def when_many_parallel_branches_finish_after_the_deadline():

                async def it_emits_RunPaused_exactly_once_per_run():
                    """End-to-end pin: parallel work past the deadline
                    yields exactly one ``RunPaused``. The state-machine
                    discriminator is :func:`it_routes_late_fan_in_to_END`.
                    """

                    class FanOut(IntegrationEvent):
                        pass

                    class Work(IntegrationEvent):
                        n: int = 0

                    class WorkDone(IntegrationEvent):
                        n: int = 0

                    proceed = asyncio.Event()

                    @on(FanOut)
                    def split(event: FanOut) -> Scatter[Work]:
                        return Scatter([Work(n=i) for i in range(8)])

                    @on(Work)
                    async def do_work(event: Work) -> WorkDone:
                        await proceed.wait()
                        return WorkDone(n=event.n)

                    graph = EventGraph([split, do_work])
                    deadline = time.monotonic() + 0.005

                    async def release_after_deadline() -> None:
                        remaining = deadline - time.monotonic()
                        if remaining > 0:
                            await asyncio.sleep(remaining + 0.002)
                        assert time.monotonic() >= deadline
                        proceed.set()

                    releaser = asyncio.create_task(release_after_deadline())
                    try:
                        log = await graph.ainvoke(FanOut(), deadline=deadline)
                    finally:
                        await releaser
                    count = log.count(RunPaused)
                    assert count == 1, (
                        f"expected exactly one RunPaused per /run, got "
                        f"{count}: {log.filter(RunPaused)!r}"
                    )

            def when_the_router_is_re_entered_after_the_deadline():

                def it_routes_late_fan_in_to_END():
                    """Direct contract test for the once-per-pause
                    state machine in ``make_router_node``. Once the
                    router has emitted ``RunPaused`` in a run, any
                    subsequent invocation while the deadline is still
                    expired must return an empty ``_pending`` (so
                    ``dispatch`` returns ``END``) and must not append
                    another ``RunPaused`` to the events channel.
                    """
                    from langgraph_events._internal import (
                        _DEADLINE_KEY,
                        _DEADLINE_STARTED_AT_KEY,
                        make_router_node,
                    )

                    router = make_router_node(max_rounds=100)
                    now = time.monotonic()
                    config: dict[str, typing.Any] = {
                        "configurable": {
                            _DEADLINE_KEY: now - 1,
                            _DEADLINE_STARTED_AT_KEY: now - 2,
                        }
                    }
                    state: dict[str, typing.Any] = {
                        "events": [Started(data="seed")],
                        "_cursor": 1,
                        "_round": 1,
                    }

                    first = router(state, config)
                    assert [e for e in first["events"] if isinstance(e, RunPaused)], (
                        "first invocation past the deadline must emit"
                    )

                    late = router(
                        {
                            **state,
                            "events": [
                                *state["events"],
                                *first["events"],
                                Processed(data="late-fan-in"),
                            ],
                            "_cursor": first["_cursor"],
                            "_round": first["_round"],
                            "_run_paused_emitted": first["_run_paused_emitted"],
                        },
                        config,
                    )
                    assert late.get("_pending") == []
                    late_pauses = [
                        e for e in late.get("events", []) if isinstance(e, RunPaused)
                    ]
                    assert late_pauses == [], (
                        f"late fan-in must not append another RunPaused; "
                        f"got {late_pauses!r}"
                    )

            def when_two_consecutive_runs_each_expire_their_deadline():

                def it_pauses_again_on_the_second_run():
                    """The once-per-pause gate is scoped to a single
                    ``/run``, not to the lifetime of the thread.  Seed
                    must reset ``_run_paused_emitted`` so a subsequent
                    ``/run`` on the same ``thread_id`` can pause again
                    if its own deadline expires.  Without the reset,
                    the second run would inherit a stuck flag and
                    drain silently with no ``RunPaused`` event.
                    """
                    graph, config = _deadline_graph("double-pause")

                    log1 = graph.invoke(
                        Started(data="first"), deadline=0.0, config=config
                    )
                    assert log1.count(RunPaused) == 1

                    log2 = graph.invoke(
                        MessageReceived(text="second"),
                        deadline=0.0,
                        config=config,
                    )
                    # log2 includes the events from log1 (checkpoint
                    # restore), so filter to the new pause.
                    pauses = log2.filter(RunPaused)
                    assert len(pauses) == 2, (
                        f"second /run must produce its own RunPaused; "
                        f"got {len(pauses)} pause(s) in log2"
                    )

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
            output = graph.namespaces().mermaid()
            assert "graph LR" in output
            assert "Started -->|step1| Processed" in output
            assert "Processed -->|step2| Ended" in output

        def it_shows_branching_return_types():
            class Accepted(IntegrationEvent):
                pass

            class Rejected(IntegrationEvent):
                pass

            @on(Started)
            def classify(event: Started) -> Accepted | Rejected:
                return Accepted()

            graph = EventGraph([classify])
            output = graph.namespaces().mermaid()
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
            output = graph.namespaces().mermaid()
            assert "%% Side-effect handlers: side_effect (Started)" in output
            assert "Started -->|producer| Ended" in output

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
            output = graph.namespaces().mermaid()
            assert 'Started -.->|"flaky (raises)"| HandlerRaised' in output
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
            output = graph.namespaces().mermaid()
            # Even though side_effect has no positive return type, the raises
            # edge must still be drawn so HandlerRaised is a real target and
            # the diagram reflects runtime behaviour.
            assert 'Started -.->|"side_effect (raises)"| HandlerRaised' in output
            assert "==> HandlerRaised" not in output

        def it_dashes_interrupted_to_resumed_edge():
            @on(Started)
            def request_approval(event: Started) -> Interrupted:
                return Interrupted()

            @on(Resumed)
            def handle_review(event: Resumed) -> Ended:
                return Ended(result="ok")

            graph = EventGraph([request_approval, handle_review])
            output = graph.namespaces().mermaid()
            assert "Interrupted -.-> Resumed" in output
            assert "Resumed -->|handle_review| Ended" in output

        def it_shows_question_mark_for_unannotated_handlers():
            @on(Started)
            def mystery(event: Started):
                return Ended(result="ok")

            graph = EventGraph([mystery])
            output = graph.namespaces().mermaid()
            assert "Started -->|mystery| ?" in output

        def it_shows_multi_subscription_edges():
            @on(Started, Processed)
            def handle_both(event: Event) -> Ended:
                return Ended(result="ok")

            graph = EventGraph([handle_both])
            output = graph.namespaces().mermaid()
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
            output = graph.namespaces().mermaid()
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
            output = graph.namespaces().mermaid()
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
                output = graph.namespaces().mermaid()
                assert "-->|handler|" in output
                assert "-->|handler_2|" in output

            def _deduped_metas(handlers, prefix: str):
                """Handler metas whose name starts with ``prefix`` after dedup."""
                graph = EventGraph(handlers)
                return [m for m in graph._handler_metas if m.name.startswith(prefix)]

            def it_preserves_raises_on_deduped_copies():
                raisers = _deduped_metas([_raiser, _raiser, _catcher], "_raiser")
                assert len(raisers) == 2
                # Without the fix, the second copy's raises= is silently dropped
                for m in raisers:
                    assert m.raises == (_DedupError,)

            def it_preserves_field_matchers_on_deduped_copies():
                catchers = _deduped_metas([_raiser, _catcher, _catcher], "_catcher")
                assert len(catchers) == 2
                # Without the fix, the second copy becomes a universal catcher
                for m in catchers:
                    matcher_names = [fn for fn, *_ in m.field_matchers]
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
                class Audited(IntegrationEvent):
                    data: str = ""

                @on(Started)
                def handler(event: Started) -> Audited:
                    return Audited(data=event.data)

                # Should not raise
                EventGraph([handler])

    def describe_astream_llm_tokens():

        def describe_custom_event_helpers():

            @on(Started)
            def _emit_custom_sync(event: Started) -> Ended:
                emit_custom("tool.progress", {"pct": 25})
                return Ended(result=event.data)

            @on(Started)
            async def _emit_custom_async(event: Started) -> Ended:
                await aemit_custom("tool.progress", {"pct": 80})
                return Ended(result=event.data)

            @on(Started)
            def _emit_snapshot_sync(event: Started) -> Ended:
                emit_state_snapshot({"step": "draft"})
                return Ended(result=event.data)

            @on(Started)
            async def _emit_snapshot_async(event: Started) -> Ended:
                await aemit_state_snapshot({"step": "review"})
                return Ended(result=event.data)

            async def _frames_from(step):
                """Stream a one-handler graph with custom events switched on."""
                return await _adrain(
                    EventGraph([step]).astream_events(
                        Started(data="hello"),
                        include_custom_events=True,
                    )
                )

            @pytest.mark.asyncio
            @pytest.mark.parametrize(
                "step, pct",
                [(_emit_custom_sync, 25), (_emit_custom_async, 80)],
                ids=["sync_handler", "async_handler"],
            )
            async def it_emits_custom_frames(step, pct):
                items = await _frames_from(step)

                custom_frames = [i for i in items if isinstance(i, CustomEventFrame)]
                assert len(custom_frames) == 1
                assert custom_frames[0].name == "tool.progress"
                assert custom_frames[0].data == {"pct": pct}

            @pytest.mark.asyncio
            @pytest.mark.parametrize(
                "step, stage",
                [(_emit_snapshot_sync, "draft"), (_emit_snapshot_async, "review")],
                ids=["sync_handler", "async_handler"],
            )
            async def it_emits_state_snapshot_frames(step, stage):
                items = await _frames_from(step)

                snapshots = [i for i in items if isinstance(i, StateSnapshotFrame)]
                assert len(snapshots) == 1
                assert snapshots[0].data == {"step": stage}

            @pytest.mark.parametrize(
                "emit",
                [
                    lambda: emit_custom("tool.progress", {"pct": 1}),
                    lambda: emit_state_snapshot({"step": "x"}),
                ],
                ids=["emit_custom", "emit_state_snapshot"],
            )
            def it_raises_for_a_sync_emit_outside_a_handler(emit):
                with pytest.raises(RuntimeError, match="while an EventGraph handler"):
                    emit()

            @pytest.mark.asyncio
            @pytest.mark.parametrize(
                "aemit",
                [
                    lambda: aemit_custom("tool.progress", {"pct": 1}),
                    lambda: aemit_state_snapshot({"step": "x"}),
                ],
                ids=["aemit_custom", "aemit_state_snapshot"],
            )
            async def it_raises_for_an_async_emit_outside_a_handler(aemit):
                with pytest.raises(RuntimeError, match="while an EventGraph handler"):
                    await aemit()

        @pytest.mark.asyncio
        async def it_yields_llm_token_and_stream_end_frames():
            graph = _llm_graph("hello world")
            items = await _adrain(
                graph.astream_events(
                    _UserSent(message=HumanMessage(content="hi")),
                    include_llm_tokens=True,
                )
            )

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
            graph = _llm_graph("reply")
            items = await _adrain(
                graph.astream_events(
                    _UserSent(message=HumanMessage(content="hi")),
                    include_llm_tokens=True,
                )
            )

            domain_events = [i for i in items if isinstance(i, Event)]
            tokens = [i for i in items if isinstance(i, LLMToken)]
            assert len(domain_events) >= 2  # at least seed + reply
            assert len(tokens) >= 1

        @pytest.mark.asyncio
        async def it_yields_reducer_frames_and_tokens():
            graph = _llm_graph("hi back", prompt="go")
            items = await _adrain(
                graph.astream_events(
                    _UserSent(message=HumanMessage(content="go")),
                    include_reducers=True,
                    include_llm_tokens=True,
                )
            )

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
            graph = _chain_graph(reducers=[_data_reducer()])
            items = await _adrain(
                graph.astream_events(
                    Started(data="x"),
                    include_reducers=True,
                    include_llm_tokens=True,
                )
            )

            frames = [i for i in items if isinstance(i, StreamFrame)]
            assert len(frames) >= 3
            assert frames[0].changed_reducers == frozenset({"data_items"})
            # Processed/Ended are not Started events, so reducer is unchanged.
            assert all(f.changed_reducers == frozenset() for f in frames[1:])

        @pytest.mark.asyncio
        async def it_omits_tokens_by_default():
            """Without include_llm_tokens, no LLMToken/LLMStreamEnd are yielded."""
            graph = _echo_graph()
            items = await _adrain(graph.astream_events(Started(data="hi")))
            assert all(isinstance(i, Event) for i in items)
            assert not any(isinstance(i, (LLMToken, LLMStreamEnd)) for i in items)

        @pytest.mark.asyncio
        async def it_yields_custom_event_frames_from_v2_custom_events(monkeypatch):
            graph = _echo_graph()
            monkeypatch.setattr(
                graph.compiled,
                "astream_events",
                _fake_astream_events(
                    _custom_payload("progress", {"pct": 50}),
                    _custom_payload(STATE_SNAPSHOT_EVENT_NAME, {"step": "draft"}),
                ),
            )

            items = await _adrain(
                graph.astream_events(
                    Started(data="hi"),
                    include_llm_tokens=True,
                    include_custom_events=True,
                )
            )

            custom_frames = [i for i in items if isinstance(i, CustomEventFrame)]
            assert len(custom_frames) == 1
            assert custom_frames[0].name == "progress"
            assert custom_frames[0].data == {"pct": 50}

            snapshots = [i for i in items if isinstance(i, StateSnapshotFrame)]
            assert len(snapshots) == 1
            assert snapshots[0].data == {"step": "draft"}

        @pytest.mark.asyncio
        async def it_does_not_yield_custom_event_frames_by_default(monkeypatch):
            graph = _echo_graph()

            called = False
            fake = _fake_astream_events(_custom_payload("progress", {"pct": 50}))

            async def tracking_fake(*args, **kwargs):
                nonlocal called
                called = True
                async for payload in fake(*args, **kwargs):
                    yield payload

            monkeypatch.setattr(graph.compiled, "astream_events", tracking_fake)

            items = await _adrain(graph.astream_events(Started(data="hi")))
            assert not any(isinstance(i, CustomEventFrame) for i in items)
            # Default flags route to _astream_core, not v2 — confirm the fake
            # was not called so the test's intent is clear.
            assert not called

        @pytest.mark.asyncio
        async def it_filters_custom_events_in_v2_path(monkeypatch):
            graph = _echo_graph()
            monkeypatch.setattr(
                graph.compiled,
                "astream_events",
                _fake_astream_events(_custom_payload("progress", {"pct": 50})),
            )

            # include_llm_tokens routes to _astream_v2 but custom events off
            items = await _adrain(
                graph.astream_events(
                    Started(data="hi"),
                    include_llm_tokens=True,
                    include_custom_events=False,
                )
            )
            assert not any(isinstance(i, CustomEventFrame) for i in items)

        @pytest.mark.asyncio
        async def it_yields_custom_event_frames_on_opt_in(monkeypatch):
            graph = _echo_graph()
            monkeypatch.setattr(
                graph.compiled,
                "astream_events",
                _fake_astream_events(_custom_payload("progress", {"pct": 50})),
            )

            items = await _adrain(
                graph.astream_events(
                    Started(data="hi"),
                    include_custom_events=True,
                )
            )
            custom_frames = [i for i in items if isinstance(i, CustomEventFrame)]
            assert len(custom_frames) == 1

        @pytest.mark.asyncio
        async def it_yields_custom_event_frames_in_astream_resume(monkeypatch):
            graph = _echo_graph(checkpointer=MemorySaver())
            monkeypatch.setattr(
                graph.compiled,
                "astream_events",
                _fake_astream_events(_custom_payload("resume.progress", {"pct": 90})),
            )

            items = await _adrain(
                graph.astream_resume(
                    Started(data="resume"),
                    include_custom_events=True,
                )
            )

            custom_frames = [i for i in items if isinstance(i, CustomEventFrame)]
            assert len(custom_frames) == 1
            assert custom_frames[0].name == "resume.progress"

        def describe_llm_tool_call_chunks():

            def when_tokens_enabled():

                @pytest.mark.asyncio
                async def it_yields_frames_per_tool_call_chunk(monkeypatch):
                    graph = _echo_graph()
                    monkeypatch.setattr(
                        graph.compiled,
                        "astream_events",
                        _fake_astream_events(
                            _chat_payload(_tool_call_chunk_message()),
                            _chat_payload(
                                _tool_call_chunk_message(
                                    name="", args='{"q":"hi"}', tool_call_id=""
                                )
                            ),
                        ),
                    )

                    items = await _adrain(
                        graph.astream_events(
                            Started(data="hi"),
                            include_llm_tokens=True,
                        )
                    )

                    chunks = [i for i in items if isinstance(i, LLMToolCallChunk)]
                    assert len(chunks) == 2
                    assert chunks[0].run_id == "run-x"
                    assert chunks[0].call_index == 0
                    assert chunks[0].tool_call_id == "tc-1"
                    assert chunks[0].name == "search"
                    assert chunks[0].args_delta == ""
                    assert chunks[1].args_delta == '{"q":"hi"}'

                @pytest.mark.asyncio
                async def it_yields_both_text_and_tool_call_from_one_chunk(monkeypatch):
                    graph = _echo_graph()
                    monkeypatch.setattr(
                        graph.compiled,
                        "astream_events",
                        _fake_astream_events(
                            _chat_payload(_tool_call_chunk_message("thinking…"))
                        ),
                    )

                    items = await _adrain(
                        graph.astream_events(
                            Started(data="hi"),
                            include_llm_tokens=True,
                        )
                    )

                    tokens = [i for i in items if isinstance(i, LLMToken)]
                    chunks = [i for i in items if isinstance(i, LLMToolCallChunk)]
                    assert len(tokens) == 1
                    assert tokens[0].content == "thinking…"
                    assert len(chunks) == 1
                    assert chunks[0].tool_call_id == "tc-1"

            def when_tokens_disabled():

                @pytest.mark.asyncio
                async def it_suppresses_chunks(monkeypatch):
                    graph = _echo_graph()
                    monkeypatch.setattr(
                        graph.compiled,
                        "astream_events",
                        _fake_astream_events(_chat_payload(_tool_call_chunk_message())),
                    )

                    items = await _adrain(
                        graph.astream_events(
                            Started(data="hi"),
                            include_custom_events=True,
                        )
                    )
                    assert not any(isinstance(i, LLMToolCallChunk) for i in items)

            def when_chunk_missing_index():

                @pytest.mark.asyncio
                async def it_raises(monkeypatch):
                    graph = _echo_graph()
                    monkeypatch.setattr(
                        graph.compiled,
                        "astream_events",
                        _fake_astream_events(
                            _chat_payload(_tool_call_chunk_message(index=None))
                        ),
                    )

                    with pytest.raises(ValueError, match=r"missing 'index'"):
                        async for _ in graph.astream_events(
                            Started(data="hi"),
                            include_llm_tokens=True,
                        ):
                            pass


def describe_OrphanedEventWarning():

    def when_orphaned():

        def it_warns_about_orphaned_event_types():
            class Orphan(IntegrationEvent):
                pass

            @on(Started)
            def produce_orphan(event: Started) -> Orphan:
                return Orphan()

            with pytest.warns(OrphanedEventWarning, match="Orphan"):
                EventGraph([produce_orphan])

        def it_points_at_user_code_not_library_internals():
            # Regression guard: stacklevel must walk past the library frames
            # so the warning surfaces the user's `EventGraph(...)` line. Went
            # unpinned until the emit moved out of __init__ into a helper and
            # the stale stacklevel anchored the warning at `sys:1`.
            class AnchorOrphan(IntegrationEvent):
                pass

            @on(Started)
            def produce_orphan(event: Started) -> AnchorOrphan:
                return AnchorOrphan()

            with pytest.warns(OrphanedEventWarning) as captured:
                EventGraph([produce_orphan])

            filename = captured[0].filename
            assert "langgraph_events" not in filename, (
                f"warning anchored to library file {filename!r}; expected user "
                f"code. Check stacklevel in _register_produced_types."
            )
            assert filename.endswith("test_event_graph.py")

        def it_warns_for_orphaned_scatter_types():
            class ScatterOrphan(IntegrationEvent):
                pass

            @on(Started)
            def scatter_producer(event: Started) -> Scatter[ScatterOrphan]:
                return Scatter([ScatterOrphan()])

            with pytest.warns(OrphanedEventWarning, match="ScatterOrphan"):
                EventGraph([scatter_producer])

        def it_warns_for_each_orphan_in_scatter_union():
            class UnionScatterA(IntegrationEvent):
                pass

            class UnionScatterB(IntegrationEvent):
                pass

            @on(Started)
            def scatter_producer(
                event: Started,
            ) -> Scatter[UnionScatterA | UnionScatterB]:
                return Scatter([UnionScatterA()])

            with pytest.warns(
                OrphanedEventWarning, match=r"UnionScatterA.*UnionScatterB"
            ):
                EventGraph([scatter_producer])

        def it_warns_for_each_orphan_in_scatter_typing_union():
            # Same behavior as ``Scatter[A | B]`` for the legacy ``typing.Union``
            # spelling — exercises the helper's ``typing.Union`` branch.
            class TypingUnionScatterA(IntegrationEvent):
                pass

            class TypingUnionScatterB(IntegrationEvent):
                pass

            @on(Started)
            def scatter_producer(
                event: Started,
            ) -> Scatter[typing.Union[TypingUnionScatterA, TypingUnionScatterB]]:  # noqa: UP007
                return Scatter([TypingUnionScatterA()])

            with pytest.warns(
                OrphanedEventWarning,
                match=r"TypingUnionScatterA.*TypingUnionScatterB",
            ):
                EventGraph([scatter_producer])

        def it_warns_for_each_orphan_in_union_of_scatters():
            # ``Scatter[A] | Scatter[B]`` — the original workaround form, parsed
            # at the outer ``_parse_return_types`` loop rather than the helper.
            class OuterScatterA(IntegrationEvent):
                pass

            class OuterScatterB(IntegrationEvent):
                pass

            @on(Started)
            def scatter_producer(
                event: Started,
            ) -> Scatter[OuterScatterA] | Scatter[OuterScatterB]:
                return Scatter([OuterScatterA()])

            with pytest.warns(
                OrphanedEventWarning, match=r"OuterScatterA.*OuterScatterB"
            ):
                EventGraph([scatter_producer])

    def when_not_orphaned():

        def when_all_scatter_union_members_subscribed():
            def it_does_not_warn():
                class SubscribedScatterA(IntegrationEvent):
                    pass

                class SubscribedScatterB(IntegrationEvent):
                    pass

                @on(Started)
                def scatter_producer(
                    event: Started,
                ) -> Scatter[SubscribedScatterA | SubscribedScatterB]:
                    return Scatter([SubscribedScatterA()])

                @on(SubscribedScatterA)
                def consume_a(event: SubscribedScatterA):
                    pass

                @on(SubscribedScatterB)
                def consume_b(event: SubscribedScatterB):
                    pass

                with warnings.catch_warnings():
                    warnings.simplefilter("error", OrphanedEventWarning)
                    EventGraph([scatter_producer, consume_a, consume_b])

        def it_does_not_warn_for_subscribed_via_inheritance():
            class Base(IntegrationEvent):
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

            class MaybeResult(IntegrationEvent):
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

        def it_does_not_warn_for_terminal_domain_event_outcomes():
            # DomainEvents nested inside a Command are terminal outcomes —
            # having no subscriber is idiomatic DDD, not an orphan.
            with warnings.catch_warnings():
                warnings.simplefilter("error", OrphanedEventWarning)
                EventGraph([_WidgetSuite.Place])

        def it_does_not_warn_for_free_standing_domain_events():
            # A DomainEvent nested directly under a Namespace (not inside a
            # Command) — like a "fact" emitted by a policy — is terminal
            # by design too. Having no subscriber must not warn.

            @on(_OrphanSuite.Analyze)
            def analyze(event: _OrphanSuite.Analyze) -> _OrphanSuite.Analyzed:
                return _OrphanSuite.Analyzed(label="ok")

            with warnings.catch_warnings():
                warnings.simplefilter("error", OrphanedEventWarning)
                EventGraph([analyze])


def describe_handler_evolution():
    # @on(previously=) keeps an interrupted checkpoint resumable after the
    # handler that paused it is renamed: the graph registers an alias node
    # under the old name so LangGraph re-enters it on resume.

    def when_previously_is_declared():
        def it_registers_an_alias_node():
            @on(Started, previously="old_name")
            def newer(event: Started) -> Ended:
                return Ended(result="x")

            nodes = set(EventGraph([newer])._compile().get_graph().nodes)
            assert "newer" in nodes
            assert "old_name" in nodes

        def it_does_not_route_new_events_into_the_alias():
            # The load-bearing invariant: the dispatcher only ever returns
            # canonical node names, so an alias never fires for fresh work — it
            # exists purely to catch resumes of in-flight checkpoints. A fresh
            # invoke must run the handler exactly once, not once per alias.
            @on(Started, previously="old_name")
            def newer(event: Started) -> Ended:
                return Ended(result="once")

            log = EventGraph([newer]).invoke(Started(data="x"))
            assert [e for e in log if isinstance(e, Ended)] == [Ended(result="once")]

    def when_an_alias_collides():
        def with_a_live_handler_name():
            def it_raises_at_build():
                @on(Started)
                def live(event: Started) -> Ended:
                    return Ended(result="x")

                @on(Ended, previously="live")
                def other(event: Ended) -> None:
                    return None

                with pytest.raises(ValueError, match="collides"):
                    EventGraph([live, other])

    def when_a_paused_handler_is_renamed():
        def with_previously_declared():
            def it_resumes_the_old_checkpoint_via_the_alias():
                # Expressed through the assert_resume_recovers helper, which
                # collapses the invoke -> assert-interrupted -> resume dance.
                from langgraph.checkpoint.memory import MemorySaver

                from langgraph_events.serde import assert_resume_recovers

                class ConfirmationRequested(Interrupted):
                    data: str

                class Confirmed(IntegrationEvent):
                    pass

                @on(Started)
                def await_input(event: Started) -> ConfirmationRequested:
                    return ConfirmationRequested(data=event.data)

                @on(Confirmed)
                def handle_confirm(event: Confirmed) -> Ended:
                    return Ended(result="confirmed")

                saver = MemorySaver()
                before = EventGraph([await_input, handle_confirm], checkpointer=saver)

                # Rename the paused handler; declare its old node name as alias.
                @on(Started, previously="await_input")
                def gather_input(event: Started) -> ConfirmationRequested:
                    return ConfirmationRequested(data=event.data)

                after = EventGraph([gather_input, handle_confirm], checkpointer=saver)
                log = assert_resume_recovers(
                    before, after, seed=Started(data="test"), resume_with=Confirmed()
                )
                assert log.latest(Ended) == Ended(result="confirmed")


class _Pause(Interrupted):
    pass


class _Go(IntegrationEvent):
    pass


class _SideDone(IntegrationEvent):
    pass


@on(Started)
def _waiter(event: Started) -> _Pause:
    """Pauses the run so resume-policy suites have an interrupted thread."""
    return _Pause()


@on(Started)
def _side_effect(event: Started) -> _SideDone:
    """Fan-out sibling of ``_waiter``. Completes normally in the same
    superstep."""
    return _SideDone()


@on(_Go)
def _go_noop(event: _Go) -> None:
    """Resume-side handler that produces nothing."""
    return None


@on(_Go)
def _go_ends(event: _Go) -> Ended:
    """Resume-side handler that completes the run."""
    return Ended(result="went")


def _resumable_pair(saver, tid: str, **kwargs: typing.Any):
    """A graph whose paused handler was removed, plus the paused thread config."""
    cfg = {"configurable": {"thread_id": tid}}
    EventGraph([_waiter, _go_noop], checkpointer=saver).invoke(
        Started(data="x"), config=cfg
    )
    return EventGraph([_go_noop], checkpointer=saver, **kwargs), cfg


def _paused_pair(saver, tid: str, **kwargs: typing.Any):
    """A genuinely-interrupted thread, ``_waiter`` still registered — the
    "live" state ``abandon()`` itself is meant to settle, before anything
    has cleared its pending task.
    """
    cfg = {"configurable": {"thread_id": tid}}
    graph = EventGraph([_waiter, _go_noop], checkpointer=saver, **kwargs)
    graph.invoke(Started(data="x"), config=cfg)
    return graph, cfg


def _abandoned_pair(saver, tid: str, **kwargs: typing.Any):
    """A genuinely-interrupted thread whose pending task was cleared out
    from under it.

    This is the shape a future ``abandon()`` will leave behind.
    ``_pending`` still stale-references the still-registered paused node,
    ``_waiter``. The checkpoint's own ``next`` reports empty. A bare
    "clear all tasks" ``bulk_update_state`` call produces this state. The
    call runs through LangGraph's public ``compiled`` property.

    Returns the graph, still with ``_waiter`` registered, and the paused
    thread config.
    """
    graph, cfg = _paused_pair(saver, tid, **kwargs)
    graph.compiled.bulk_update_state(cfg, [[StateUpdate(None, END)]])
    return graph, cfg


def describe_on_unresumable():
    # resume() on a thread that is not awaiting input (paused handler removed,
    # already-finished, or double-resume) is governed by
    # EventGraph(on_unresumable=...). Default `raise` turns the old silent
    # no-op into a clear error; `warn`/`halt` opt into non-fatal handling.

    def when_the_paused_handler_was_removed():
        def with_default_policy():
            def it_raises_unresumable_error():
                v2, cfg = _resumable_pair(MemorySaver(), "unres-raise")
                with pytest.raises(UnresumableError):
                    v2.resume(_Go(), config=cfg)

    def when_the_thread_is_genuinely_interrupted():
        def it_resumes_normally():
            cfg = {"configurable": {"thread_id": "unres-live"}}
            graph = EventGraph([_waiter, _go_ends], checkpointer=MemorySaver())
            graph.invoke(Started(data="x"), config=cfg)
            assert graph.get_state(cfg).is_interrupted

            log = graph.resume(_Go(), config=cfg)
            assert log.latest(Ended) == Ended(result="went")

    def when_the_policy_value_is_invalid():
        def it_raises_at_construction():
            with pytest.raises(ValueError, match="on_unresumable"):
                EventGraph([_echo], on_unresumable="nope")  # type: ignore[arg-type]

    def when_warn_policy():
        def it_warns_and_leaves_the_log_unchanged():
            v2, cfg = _resumable_pair(
                MemorySaver(), "unres-warn", on_unresumable="warn"
            )
            with pytest.warns(UserWarning, match="not awaiting input"):
                log = v2.resume(_Go(), config=cfg)

            assert not any(isinstance(e, Halted) for e in log)

    def when_halt_policy():
        def it_finalizes_the_thread_terminally():
            v2, cfg = _resumable_pair(
                MemorySaver(), "unres-halt", on_unresumable="halt"
            )
            log = v2.resume(_Go(), config=cfg)

            assert isinstance(log.latest(Unresumable), Unresumable)
            assert isinstance(log.latest(Unresumable), Halted)
            assert not v2.get_state(cfg).is_interrupted

        def it_leaves_nothing_scheduled():
            # `_abandoned_pair` sets up a thread where `_pending` still
            # stale-references the still-registered paused node. A
            # single-superstep write reschedules that node.
            graph, cfg = _abandoned_pair(
                MemorySaver(), "unres-halt-next", on_unresumable="halt"
            )

            graph.resume(_Go(), config=cfg)

            assert graph.compiled.get_state(cfg).next == ()

        def it_leaves_the_thread_usable():
            saver = MemorySaver()
            cfg = {"configurable": {"thread_id": "unres-halt-usable"}}
            EventGraph([_waiter, _go_noop], checkpointer=saver).invoke(
                Started(data="x"), config=cfg
            )
            v2 = EventGraph([_go_ends], checkpointer=saver, on_unresumable="halt")
            v2.resume(_Go(), config=cfg)

            log = v2.invoke(_Go(), config=cfg)

            assert log.latest(Ended) == Ended(result="went")

        def it_does_not_resurrect_the_retired_identity():
            # Same `_abandoned_pair` stale-scheduling setup as
            # `it_leaves_nothing_scheduled`. If the halt policy re-arms
            # the paused node, a second `resume()` call passes
            # `_resume_is_pending`. It then runs `_waiter` for real,
            # writing the retired `_Pause` identity back into the log.
            graph, cfg = _abandoned_pair(
                MemorySaver(), "unres-halt-resurrect", on_unresumable="halt"
            )
            graph.resume(_Go(), config=cfg)

            log = graph.resume(_Go(), config=cfg)

            assert not any(isinstance(e, _Pause) for e in log)

        def it_preserves_completed_sibling_writes():
            saver = MemorySaver()
            cfg = {"configurable": {"thread_id": "unres-halt-sibling"}}
            EventGraph([_waiter, _side_effect, _go_noop], checkpointer=saver).invoke(
                Started(data="x"), config=cfg
            )
            v2 = EventGraph(
                [_side_effect, _go_noop], checkpointer=saver, on_unresumable="halt"
            )

            log = v2.resume(_Go(), config=cfg)

            assert log.has(_SideDone)

    def when_resumed_via_another_entrypoint():
        # Every resume entrypoint consults the same on_unresumable policy.
        @pytest.mark.parametrize(
            "tid, drive",
            [
                (
                    "unres-aresume",
                    lambda graph, cfg: asyncio.run(graph.aresume(_Go(), config=cfg)),
                ),
                (
                    "unres-stream",
                    lambda graph, cfg: list(graph.stream_resume(_Go(), config=cfg)),
                ),
                (
                    "unres-astream",
                    lambda graph, cfg: asyncio.run(
                        _adrain(graph.astream_resume(_Go(), config=cfg))
                    ),
                ),
            ],
            ids=["aresume", "stream_resume", "astream_resume"],
        )
        def it_honors_the_policy(tid, drive):
            v2, cfg = _resumable_pair(MemorySaver(), tid)
            with pytest.raises(UnresumableError):
                drive(v2, cfg)


def describe_abandon():
    # abandon() settles a genuinely-paused thread without ever answering
    # its Interrupted — the tool for retiring an Interrupted subclass
    # (#162). Contrast with on_unresumable="halt", which settles a thread
    # that *shouldn't* still be paused (removed/renamed handler); abandon()
    # acts deliberately on a thread that legitimately still is.

    def when_a_thread_is_genuinely_paused():
        def it_leaves_nothing_scheduled():
            graph, cfg = _paused_pair(MemorySaver(), "abandon-next")

            result = graph.abandon(cfg)

            assert result is None
            assert graph.compiled.get_state(cfg).next == ()

        def it_leaves_no_pending_interrupt_write():
            saver = MemorySaver()
            graph, cfg = _paused_pair(saver, "abandon-pending-write")

            graph.abandon(cfg)

            assert saver.get_tuple(cfg).pending_writes == []

        def it_does_not_record_the_interrupt():
            graph, cfg = _paused_pair(MemorySaver(), "abandon-no-interrupt")

            graph.abandon(cfg)

            log = graph.get_state(cfg).events
            assert not any(isinstance(e, _Pause) for e in log)

        def it_records_the_discarded_type_name():
            graph, cfg = _paused_pair(MemorySaver(), "abandon-discarded")

            graph.abandon(cfg)

            log = graph.get_state(cfg).events
            assert log.latest(Abandoned).discarded == "_Pause"

        def it_records_the_reason():
            graph, cfg = _paused_pair(MemorySaver(), "abandon-reason")

            graph.abandon(cfg, reason="retiring _Pause")

            log = graph.get_state(cfg).events
            assert log.latest(Abandoned).reason == "retiring _Pause"

        def it_preserves_completed_sibling_writes():
            # A fan-out where one handler (_waiter) interrupts and a
            # sibling (_side_effect) completes in the same superstep.
            # `_settle`'s leading clear must commit the sibling's already-
            # written event rather than discard it along with the stale
            # pending task.
            saver = MemorySaver()
            cfg = {"configurable": {"thread_id": "abandon-sibling"}}
            graph = EventGraph([_waiter, _side_effect, _go_noop], checkpointer=saver)
            graph.invoke(Started(data="x"), config=cfg)

            graph.abandon(cfg)

            assert graph.get_state(cfg).events.has(_SideDone)

        def it_leaves_the_thread_usable():
            saver = MemorySaver()
            cfg = {"configurable": {"thread_id": "abandon-usable"}}
            EventGraph([_waiter, _go_noop], checkpointer=saver).invoke(
                Started(data="x"), config=cfg
            )
            v2 = EventGraph([_go_ends], checkpointer=saver)
            v2.abandon(cfg)

            log = v2.invoke(_Go(), config=cfg)

            assert log.latest(Ended) == Ended(result="went")

    def when_resumed_after_abandoning():
        def with_default_policy():
            def it_raises_naming_the_abandonment():
                graph, cfg = _paused_pair(MemorySaver(), "abandon-resume-raise")
                graph.abandon(cfg, reason="retiring _Pause")

                with pytest.raises(UnresumableError, match="abandon"):
                    graph.resume(_Go(), config=cfg)

        def with_halt_policy():
            def it_does_not_resurrect_the_retired_identity():
                graph, cfg = _paused_pair(
                    MemorySaver(), "abandon-resume-halt", on_unresumable="halt"
                )
                graph.abandon(cfg)

                graph.resume(_Go(), config=cfg)
                log = graph.resume(_Go(), config=cfg)

                assert not any(isinstance(e, _Pause) for e in log)

    def when_the_thread_was_never_run():
        def it_raises():
            graph = EventGraph([_waiter, _go_noop], checkpointer=MemorySaver())
            cfg = {"configurable": {"thread_id": "abandon-never-run"}}

            with pytest.raises(ValueError, match=r"abandon"):
                graph.abandon(cfg)

    def when_only_pre_seeded():
        def it_raises():
            saver = MemorySaver()
            graph = EventGraph([_waiter, _go_noop], checkpointer=saver)
            cfg = {"configurable": {"thread_id": "abandon-pre-seeded"}}
            graph.pre_seed(cfg, {})

            with pytest.raises(ValueError, match=r"abandon"):
                graph.abandon(cfg)

    def when_there_is_no_checkpointer():
        def it_raises():
            graph = EventGraph([_waiter, _go_noop])
            cfg = {"configurable": {"thread_id": "abandon-no-checkpointer"}}

            with pytest.raises(ValueError, match=r"abandon.*requires a checkpointer"):
                graph.abandon(cfg)


class _AsyncOnlySaver(MemorySaver):
    """Mimics ``AsyncPostgresSaver``: a synchronous checkpoint read from a
    running event loop is rejected.

    Async-only checkpointers raise ``InvalidStateError`` on sync checkpointer
    access issued from the running loop; ``MemorySaver`` allows it, which is
    why ``MemorySaver``-only suites miss the async-resume regression (#95).
    The sync ``get_tuple`` is guarded so any sync ``get_state`` on the async
    path trips it; both ``get_tuple`` and ``aget_tuple`` reach the store through
    a shared ``_unguarded_get_tuple`` so the async path never routes through the
    guarded sync sibling — a real async saver has a genuinely separate async
    implementation (``MemorySaver``'s merely delegates to the sync method).
    """

    @staticmethod
    def _reject_in_loop() -> None:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return  # no running loop (sync setup) — allow
        raise asyncio.InvalidStateError(
            "Synchronous calls to AsyncPostgresSaver are only allowed from a "
            "different thread."
        )

    def _unguarded_get_tuple(self, config):
        return super().get_tuple(config)

    def get_tuple(self, config):  # type: ignore[override]
        self._reject_in_loop()
        return self._unguarded_get_tuple(config)

    async def aget_tuple(self, config):  # type: ignore[override]
        return self._unguarded_get_tuple(config)


def describe_async_only_checkpointer():
    # #95: aresume()/astream_resume() must not drive an async-only checkpointer
    # (e.g. AsyncPostgresSaver) synchronously. The resume-pending gate and the
    # on_unresumable policy reads previously went through sync get_state, which
    # such a checkpointer rejects from the running event loop.

    async def _apaused_pair(tid: str, **kwargs: typing.Any):
        """Pause a thread via ainvoke, then rebuild it without ``_waiter``."""
        saver = _AsyncOnlySaver()
        cfg = {"configurable": {"thread_id": tid}}
        await EventGraph([_waiter, _go_noop], checkpointer=saver).ainvoke(
            Started(data="x"), config=cfg
        )
        return EventGraph([_go_noop], checkpointer=saver, **kwargs), cfg

    async def _aabandoned_pair(tid: str, **kwargs: typing.Any):
        """Async mirror of ``_abandoned_pair``.

        Builds a genuinely-interrupted thread whose pending task was
        cleared out from under it. ``_waiter`` is still registered. The
        setup runs through ``_AsyncOnlySaver``.
        """
        saver = _AsyncOnlySaver()
        cfg = {"configurable": {"thread_id": tid}}
        graph = EventGraph([_waiter, _go_noop], checkpointer=saver, **kwargs)
        await graph.ainvoke(Started(data="x"), config=cfg)
        await graph.compiled.abulk_update_state(cfg, [[StateUpdate(None, END)]])
        return graph, cfg

    def when_the_thread_is_genuinely_pending():
        def without_sync_checkpointer_access():

            async def _paused_graph(tid: str):
                cfg = {"configurable": {"thread_id": tid}}
                graph = EventGraph([_waiter, _go_ends], checkpointer=_AsyncOnlySaver())
                await graph.ainvoke(Started(data="x"), config=cfg)
                return graph, cfg

            @pytest.mark.asyncio
            async def it_aresumes():
                graph, cfg = await _paused_graph("async-only-aresume")

                log = await graph.aresume(_Go(), config=cfg)
                assert log.latest(Resumed) is not None
                assert log.latest(Ended) == Ended(result="went")

            @pytest.mark.asyncio
            async def it_astream_resumes():
                graph, cfg = await _paused_graph("async-only-astream")

                events = await _adrain(graph.astream_resume(_Go(), config=cfg))
                assert any(isinstance(e, Ended) for e in events)

    def when_the_thread_is_not_pending():
        @pytest.mark.asyncio
        async def it_aresume_raises_under_default_policy():
            v2, cfg = await _apaused_pair("async-only-raise")
            with pytest.raises(UnresumableError):
                await v2.aresume(_Go(), config=cfg)

        @pytest.mark.asyncio
        async def it_aresume_warns_and_leaves_the_log_unchanged():
            v2, cfg = await _apaused_pair("async-only-warn", on_unresumable="warn")
            with pytest.warns(UserWarning, match="not awaiting input"):
                log = await v2.aresume(_Go(), config=cfg)

            assert not any(isinstance(e, Halted) for e in log)

        @pytest.mark.asyncio
        async def it_aresume_halt_finalizes_the_thread_terminally():
            v2, cfg = await _apaused_pair("async-only-halt", on_unresumable="halt")
            log = await v2.aresume(_Go(), config=cfg)

            assert log.latest(Unresumable) is not None

        @pytest.mark.asyncio
        async def it_aresume_halt_leaves_nothing_scheduled():
            # Async mirror of the sync `it_leaves_nothing_scheduled`. It
            # uses the same `_aabandoned_pair` stale-scheduling setup.
            # The test runs through the async-only checkpointer to
            # exercise `_asettle`.
            graph, cfg = await _aabandoned_pair(
                "async-only-halt-next", on_unresumable="halt"
            )

            await graph.aresume(_Go(), config=cfg)

            state = await graph.compiled.aget_state(cfg)
            assert state.next == ()

        @pytest.mark.asyncio
        async def it_astream_resume_halt_yields_the_terminal_event():
            v2, cfg = await _apaused_pair(
                "async-only-astream-halt", on_unresumable="halt"
            )
            events = await _adrain(v2.astream_resume(_Go(), config=cfg))

            assert any(isinstance(e, Unresumable) for e in events)

    def describe_aabandon():
        # Async mirror of describe_abandon() (module scope), driven through
        # _AsyncOnlySaver — the #95 contract: aabandon() must read/write
        # exclusively via aget_state/_asettle, never falling back to a sync
        # checkpoint read from the running event loop.

        async def _apaused_live_pair(tid: str, **kwargs: typing.Any):
            """Genuinely-interrupted async mirror of the module-level
            `_paused_pair`, still registering `_waiter`, driven through
            `_AsyncOnlySaver`."""
            saver = _AsyncOnlySaver()
            cfg = {"configurable": {"thread_id": tid}}
            graph = EventGraph([_waiter, _go_noop], checkpointer=saver, **kwargs)
            await graph.ainvoke(Started(data="x"), config=cfg)
            return graph, cfg

        def when_a_thread_is_genuinely_paused():
            @pytest.mark.asyncio
            async def it_leaves_nothing_scheduled():
                graph, cfg = await _apaused_live_pair("aabandon-next")

                result = await graph.aabandon(cfg)

                assert result is None
                state = await graph.compiled.aget_state(cfg)
                assert state.next == ()

            @pytest.mark.asyncio
            async def it_records_the_discarded_type_name_and_reason():
                graph, cfg = await _apaused_live_pair("aabandon-discarded")

                await graph.aabandon(cfg, reason="retiring _Pause")

                log = (await graph.aget_state(cfg)).events
                event = log.latest(Abandoned)
                assert event.discarded == "_Pause"
                assert event.reason == "retiring _Pause"

            @pytest.mark.asyncio
            async def it_leaves_the_thread_usable():
                saver = _AsyncOnlySaver()
                cfg = {"configurable": {"thread_id": "aabandon-usable"}}
                await EventGraph([_waiter, _go_noop], checkpointer=saver).ainvoke(
                    Started(data="x"), config=cfg
                )
                v2 = EventGraph([_go_ends], checkpointer=saver)
                await v2.aabandon(cfg)

                log = await v2.ainvoke(_Go(), config=cfg)

                assert log.latest(Ended) == Ended(result="went")

        def when_resumed_after_abandoning():
            @pytest.mark.asyncio
            async def it_raises_naming_the_abandonment():
                graph, cfg = await _apaused_live_pair("aabandon-resume-raise")
                await graph.aabandon(cfg)

                with pytest.raises(UnresumableError, match="abandon"):
                    await graph.aresume(_Go(), config=cfg)

        def when_the_thread_was_never_run():
            @pytest.mark.asyncio
            async def it_raises():
                graph = EventGraph([_waiter, _go_noop], checkpointer=_AsyncOnlySaver())
                cfg = {"configurable": {"thread_id": "aabandon-never-run"}}

                with pytest.raises(ValueError, match=r"aabandon"):
                    await graph.aabandon(cfg)

        def when_there_is_no_checkpointer():
            @pytest.mark.asyncio
            async def it_raises():
                graph = EventGraph([_waiter, _go_noop])
                cfg = {"configurable": {"thread_id": "aabandon-no-checkpointer"}}

                with pytest.raises(
                    ValueError, match=r"aabandon.*requires a checkpointer"
                ):
                    await graph.aabandon(cfg)


def describe_assert_resume_recovers():
    # Convenience helper: collapses the interrupt -> rebuild -> resume recovery
    # proof for an @on(previously=) rename into one assertion — the handler
    # analog of assert_all_baselined_revive.

    def _recovers(before: EventGraph, after: EventGraph):
        from langgraph_events.serde import assert_resume_recovers

        return assert_resume_recovers(
            before, after, seed=Started(data="x"), resume_with=_Go()
        )

    def when_a_renamed_handler_declares_previously():
        def it_returns_the_post_resume_log():
            saver = MemorySaver()

            @on(Started)
            def await_input(event: Started) -> _Pause:
                return _Pause()

            before = EventGraph([await_input, _go_ends], checkpointer=saver)

            @on(Started, previously="await_input")
            def gather(event: Started) -> _Pause:
                return _Pause()

            after = EventGraph([gather, _go_ends], checkpointer=saver)

            log = _recovers(before, after)
            assert log.latest(Ended) == Ended(result="went")

    def when_the_rename_is_undeclared():
        def it_propagates_unresumable_error():
            saver = MemorySaver()

            @on(Started)
            def await_input(event: Started) -> _Pause:
                return _Pause()

            before = EventGraph([await_input, _go_noop], checkpointer=saver)

            @on(Started)  # renamed, NO previously=
            def gather(event: Started) -> _Pause:
                return _Pause()

            after = EventGraph([gather, _go_noop], checkpointer=saver)

            with pytest.raises(UnresumableError):
                _recovers(before, after)

    def when_the_seed_does_not_interrupt():
        def it_raises_a_precondition_error():
            saver = MemorySaver()

            @on(Started)
            def noop(event: Started) -> None:
                return None

            before = EventGraph([noop, _go_noop], checkpointer=saver)
            after = EventGraph([noop, _go_noop], checkpointer=saver)

            with pytest.raises(AssertionError, match="did not pause"):
                _recovers(before, after)

    def when_the_graphs_do_not_share_a_checkpointer():
        def it_raises_value_error():
            before = EventGraph([_waiter], checkpointer=MemorySaver())
            after = EventGraph([_waiter], checkpointer=MemorySaver())

            with pytest.raises(ValueError, match="checkpointer"):
                _recovers(before, after)


def describe_namespaces_cache():
    def it_reuses_a_cached_namespace_model_across_calls():
        graph = EventGraph([Order.Place])

        assert graph.namespaces() is graph.namespaces()
