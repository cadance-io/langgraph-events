"""AG-UI adapter tests — BDD-style with real EventGraph, no HTTP."""

from __future__ import annotations

import json
import warnings
from typing import Any

import pytest
from ag_ui.core import (
    BaseEvent,
    CustomEvent,
    EventType,
    RunAgentInput,
)
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import MemorySaver

from langgraph_events import (
    Event,
    EventGraph,
    Interrupted,
    MessageEvent,
    message_reducer,
    on,
)
from langgraph_events._reducer import ScalarReducer
from langgraph_events.agui import (
    AGUIAdapter,
    MapperContext,
)
from langgraph_events.agui._mappers import _warned_classes

# ---------------------------------------------------------------------------
# Test event classes
# ---------------------------------------------------------------------------


class UserAsked(Event):
    question: str = ""

    def agui_dict(self) -> dict[str, Any]:
        return {"question": self.question}


class AgentReplied(MessageEvent):
    message: AIMessage = None  # type: ignore[assignment]


class AgentCalledTools(MessageEvent):
    message: AIMessage = None  # type: ignore[assignment]


class ToolsExecuted(MessageEvent):
    messages: tuple[ToolMessage, ...] = ()


class AgentAndToolMessages(MessageEvent):
    messages: tuple[Any, ...] = ()


class UserSent(MessageEvent):
    message: HumanMessage = None  # type: ignore[assignment]

    def agui_dict(self) -> dict[str, Any]:
        return {"content": self.message.content if self.message else ""}


class FollowUpReply(MessageEvent):
    message: AIMessage = None  # type: ignore[assignment]


class TaskCreated(Event):
    title: str = ""

    def agui_dict(self) -> dict[str, Any]:
        return {"title": self.title}


class ApprovalRequested(Interrupted):
    draft: str = ""

    def agui_dict(self) -> dict[str, Any]:
        return {"draft": self.draft}


class ApprovalGiven(Event):
    approved: bool = True

    def agui_dict(self) -> dict[str, Any]:
        return {"approved": self.approved}


class PhaseA(MessageEvent):
    messages: tuple[Any, ...] = ()


class PhaseB(MessageEvent):
    messages: tuple[Any, ...] = ()


class ErrorTrigger(Event):
    def agui_dict(self) -> dict[str, Any]:
        return {}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_input(**overrides: Any) -> RunAgentInput:
    defaults: dict[str, Any] = {
        "thread_id": "thread-1",
        "run_id": "run-1",
        "state": {},
        "messages": [],
        "tools": [],
        "context": [],
        "forwarded_props": {},
    }
    defaults.update(overrides)
    return RunAgentInput(**defaults)


async def _collect(adapter: AGUIAdapter, input_data: RunAgentInput) -> list[BaseEvent]:
    events: list[BaseEvent] = []
    async for event in adapter.stream(input_data):
        events.append(event)
    return events


def _types(events: list[BaseEvent]) -> list[EventType]:
    return [e.type for e in events]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def describe_AGUIAdapter():
    def describe_stream():
        @pytest.fixture
        def simple_graph():
            @on(UserAsked)
            def reply(event: UserAsked) -> AgentReplied:
                return AgentReplied(
                    message=AIMessage(content=f"Answer to: {event.question}")
                )

            return EventGraph([reply])

        def when_default_configuration():
            async def it_emits_run_started_and_finished(simple_graph):
                adapter = AGUIAdapter(
                    graph=simple_graph,
                    seed_factory=lambda inp: UserAsked(question="hi"),
                )
                events = await _collect(adapter, _make_input())

                assert events[0].type == EventType.RUN_STARTED
                assert events[0].thread_id == "thread-1"
                assert events[0].run_id == "run-1"
                assert events[-1].type == EventType.RUN_FINISHED
                assert events[-1].thread_id == "thread-1"

            async def it_maps_ai_message_to_text_events(simple_graph):
                adapter = AGUIAdapter(
                    graph=simple_graph,
                    seed_factory=lambda inp: UserAsked(question="hello"),
                )
                events = await _collect(adapter, _make_input())

                text_events = [
                    e
                    for e in events
                    if e.type
                    in (
                        EventType.TEXT_MESSAGE_START,
                        EventType.TEXT_MESSAGE_CONTENT,
                        EventType.TEXT_MESSAGE_END,
                    )
                ]
                assert len(text_events) == 3
                assert text_events[0].type == EventType.TEXT_MESSAGE_START
                assert text_events[0].role == "assistant"
                assert text_events[1].type == EventType.TEXT_MESSAGE_CONTENT
                assert "Answer to: hello" in text_events[1].delta
                assert text_events[2].type == EventType.TEXT_MESSAGE_END
                # All share the same message_id
                msg_id = text_events[0].message_id
                assert all(e.message_id == msg_id for e in text_events)

            async def it_maps_tool_calls_to_tool_events():
                @on(UserAsked)
                def call_tool(event: UserAsked) -> AgentCalledTools:
                    return AgentCalledTools(
                        message=AIMessage(
                            content="",
                            tool_calls=[
                                {
                                    "id": "tc-1",
                                    "name": "search",
                                    "args": {"query": "test"},
                                }
                            ],
                        )
                    )

                graph = EventGraph([call_tool])
                adapter = AGUIAdapter(
                    graph=graph,
                    seed_factory=lambda inp: UserAsked(question="search"),
                )
                events = await _collect(adapter, _make_input())

                tool_events = [
                    e
                    for e in events
                    if e.type
                    in (
                        EventType.TOOL_CALL_START,
                        EventType.TOOL_CALL_ARGS,
                        EventType.TOOL_CALL_END,
                    )
                ]
                assert len(tool_events) == 3
                assert tool_events[0].type == EventType.TOOL_CALL_START
                assert tool_events[0].tool_call_id == "tc-1"
                assert tool_events[0].tool_call_name == "search"
                assert tool_events[1].type == EventType.TOOL_CALL_ARGS
                assert json.loads(tool_events[1].delta) == {"query": "test"}
                assert tool_events[2].type == EventType.TOOL_CALL_END

            async def it_maps_tool_results_to_tool_call_result_events():
                @on(UserAsked)
                def run_tools(event: UserAsked) -> ToolsExecuted:
                    return ToolsExecuted(
                        messages=(
                            ToolMessage(
                                content="result-1",
                                tool_call_id="tc-1",
                            ),
                        )
                    )

                graph = EventGraph([run_tools])
                adapter = AGUIAdapter(
                    graph=graph,
                    seed_factory=lambda inp: UserAsked(question="run"),
                )
                events = await _collect(adapter, _make_input())

                result_events = [
                    e for e in events if e.type == EventType.TOOL_CALL_RESULT
                ]
                assert len(result_events) == 1
                assert result_events[0].tool_call_id == "tc-1"
                assert result_events[0].content == "result-1"

            async def it_maps_mixed_ai_and_tool_messages():
                @on(UserAsked)
                def reply_tool_result(event: UserAsked) -> AgentAndToolMessages:
                    return AgentAndToolMessages(
                        messages=(
                            AIMessage(content="I used a tool"),
                            ToolMessage(content="tool output", tool_call_id="tc-mixed"),
                        )
                    )

                graph = EventGraph([reply_tool_result])
                adapter = AGUIAdapter(
                    graph=graph,
                    seed_factory=lambda inp: UserAsked(question="mixed"),
                )
                events = await _collect(adapter, _make_input())

                text_events = [
                    e for e in events if e.type == EventType.TEXT_MESSAGE_CONTENT
                ]
                result_events = [
                    e for e in events if e.type == EventType.TOOL_CALL_RESULT
                ]
                assert any("I used a tool" in e.delta for e in text_events)
                assert len(result_events) == 1
                assert result_events[0].tool_call_id == "tc-mixed"
                assert result_events[0].content == "tool output"

            async def it_maps_interrupted_to_custom_event():
                @on(UserAsked)
                def ask_approval(event: UserAsked) -> ApprovalRequested:
                    return ApprovalRequested(draft="draft text")

                graph = EventGraph([ask_approval], checkpointer=MemorySaver())
                adapter = AGUIAdapter(
                    graph=graph,
                    seed_factory=lambda inp: UserAsked(question="approve?"),
                )
                events = await _collect(adapter, _make_input())

                custom_events = [
                    e
                    for e in events
                    if e.type == EventType.CUSTOM and e.name == "interrupted"
                ]
                assert len(custom_events) == 1
                assert custom_events[0].value["draft"] == "draft text"

            async def it_suppresses_resumed_events():
                @on(UserAsked)
                def ask_approval(event: UserAsked) -> ApprovalRequested:
                    return ApprovalRequested(draft="draft")

                @on(ApprovalGiven)
                def handle_approval(event: ApprovalGiven) -> AgentReplied:
                    return AgentReplied(message=AIMessage(content="Approved!"))

                graph = EventGraph(
                    [ask_approval, handle_approval],
                    checkpointer=MemorySaver(),
                )

                # First run — triggers interrupt
                config = {"configurable": {"thread_id": "thread-resume"}}
                # Run directly to create interrupt checkpoint
                await graph.ainvoke(UserAsked(question="approve?"), config=config)

                # Resume run
                adapter_resume = AGUIAdapter(
                    graph=graph,
                    seed_factory=lambda inp: UserAsked(question="unused"),
                    resume_factory=lambda inp: ApprovalGiven(approved=True),
                )
                events = await _collect(
                    adapter_resume,
                    _make_input(thread_id="thread-resume"),
                )

                # Resumed events should be suppressed
                resumed_custom = [
                    e
                    for e in events
                    if e.type == EventType.CUSTOM and e.name == "Resumed"
                ]
                assert len(resumed_custom) == 0

            async def it_maps_unknown_events_to_custom_events():
                @on(UserAsked)
                def create_task(event: UserAsked) -> TaskCreated:
                    return TaskCreated(title="new task")

                graph = EventGraph([create_task])
                adapter = AGUIAdapter(
                    graph=graph,
                    seed_factory=lambda inp: UserAsked(question="task"),
                )
                events = await _collect(adapter, _make_input())

                custom_events = [
                    e
                    for e in events
                    if e.type == EventType.CUSTOM and e.name == "TaskCreated"
                ]
                assert len(custom_events) == 1
                assert custom_events[0].value["title"] == "new task"

            async def it_emits_run_error_on_exception():
                @on(ErrorTrigger)
                def blow_up(event: ErrorTrigger) -> None:
                    raise RuntimeError("boom")

                graph = EventGraph([blow_up])
                adapter = AGUIAdapter(
                    graph=graph,
                    seed_factory=lambda inp: ErrorTrigger(),
                )
                events = await _collect(adapter, _make_input())

                assert events[0].type == EventType.RUN_STARTED
                error_events = [e for e in events if e.type == EventType.RUN_ERROR]
                assert len(error_events) == 1
                assert "boom" in error_events[0].message
                # RunFinished should NOT appear after error
                assert events[-1].type == EventType.RUN_ERROR

            async def it_streams_text_message_content_during_llm_calls():
                llm = FakeListChatModel(responses=["stream me"], sleep=0)

                @on(UserAsked)
                async def stream_reply(
                    event: UserAsked,
                    messages: list[Any],
                ) -> AgentReplied:
                    response = await llm.ainvoke(
                        [*messages, HumanMessage(content=event.question)]
                    )
                    return AgentReplied(message=response)

                graph = EventGraph([stream_reply], reducers=[message_reducer()])
                adapter = AGUIAdapter(
                    graph=graph,
                    seed_factory=lambda inp: UserAsked(question="go"),
                    include_reducers=True,
                )
                events = await _collect(adapter, _make_input())

                text_events = [
                    e
                    for e in events
                    if e.type
                    in (
                        EventType.TEXT_MESSAGE_START,
                        EventType.TEXT_MESSAGE_CONTENT,
                        EventType.TEXT_MESSAGE_END,
                    )
                ]
                assert text_events[0].type == EventType.TEXT_MESSAGE_START
                deltas = [
                    e.delta
                    for e in text_events
                    if e.type == EventType.TEXT_MESSAGE_CONTENT
                ]
                assert len(deltas) > 1
                assert "".join(deltas) == "stream me"
                assert text_events[-1].type == EventType.TEXT_MESSAGE_END

            async def it_skips_events_lacking_agui_dict():
                class PlainEvent(Event):
                    value: str = "no-dict"

                @on(UserAsked)
                def emit_plain(event: UserAsked) -> PlainEvent:
                    return PlainEvent(value="hello")

                graph = EventGraph([emit_plain])
                adapter = AGUIAdapter(
                    graph=graph,
                    seed_factory=lambda inp: UserAsked(question="go"),
                )
                _warned_classes.discard(PlainEvent)
                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter("always")
                    events = await _collect(adapter, _make_input())

                custom_events = [
                    e
                    for e in events
                    if e.type == EventType.CUSTOM and e.name == "PlainEvent"
                ]
                assert len(custom_events) == 0
                assert any("PlainEvent" in str(warning.message) for warning in w)

            async def it_emits_events_having_agui_dict():
                @on(UserAsked)
                def create_task(event: UserAsked) -> TaskCreated:
                    return TaskCreated(title="with dict")

                graph = EventGraph([create_task])
                adapter = AGUIAdapter(
                    graph=graph,
                    seed_factory=lambda inp: UserAsked(question="go"),
                )
                events = await _collect(adapter, _make_input())

                custom_events = [
                    e
                    for e in events
                    if e.type == EventType.CUSTOM and e.name == "TaskCreated"
                ]
                assert len(custom_events) == 1
                assert custom_events[0].value == {"title": "with dict"}

            async def it_warns_once_per_class():
                class NoDict1(Event):
                    x: int = 0

                class NoDict2(Event):
                    x: int = 0

                @on(UserAsked)
                def step1(event: UserAsked) -> NoDict1:
                    return NoDict1(x=1)

                @on(NoDict1)
                def step2(event: NoDict1) -> NoDict2:
                    return NoDict2(x=2)

                graph = EventGraph([step1, step2])
                adapter = AGUIAdapter(
                    graph=graph,
                    seed_factory=lambda inp: UserAsked(question="go"),
                )
                _warned_classes.discard(NoDict1)
                _warned_classes.discard(NoDict2)
                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter("always")
                    await _collect(adapter, _make_input())

                nodict1_warnings = [x for x in w if "NoDict1" in str(x.message)]
                nodict2_warnings = [x for x in w if "NoDict2" in str(x.message)]
                # Each class warned exactly once
                assert len(nodict1_warnings) == 1
                assert len(nodict2_warnings) == 1

        def when_include_reducers():
            async def it_emits_state_snapshot():
                @on(UserAsked)
                def reply(event: UserAsked) -> AgentReplied:
                    return AgentReplied(message=AIMessage(content="hello"))

                graph = EventGraph(
                    [reply],
                    reducers=[message_reducer()],
                )
                adapter = AGUIAdapter(
                    graph=graph,
                    seed_factory=lambda inp: UserAsked(question="hi"),
                    include_reducers=True,
                )
                events = await _collect(adapter, _make_input())

                snapshots = [e for e in events if e.type == EventType.STATE_SNAPSHOT]
                assert len(snapshots) >= 1
                # Snapshot should not duplicate dedicated messages channel
                assert "messages" not in snapshots[-1].snapshot

            async def it_excludes_messages_from_state_snapshot():
                @on(UserAsked)
                def reply(event: UserAsked) -> AgentReplied:
                    return AgentReplied(message=AIMessage(content="hello"))

                graph = EventGraph(
                    [reply],
                    reducers=[message_reducer()],
                )
                adapter = AGUIAdapter(
                    graph=graph,
                    seed_factory=lambda inp: UserAsked(question="hi"),
                    include_reducers=True,
                )
                events = await _collect(adapter, _make_input())

                state_snapshots = [
                    e for e in events if e.type == EventType.STATE_SNAPSHOT
                ]
                message_snapshots = [
                    e for e in events if e.type == EventType.MESSAGES_SNAPSHOT
                ]

                assert len(state_snapshots) >= 1
                assert all(
                    "messages" not in snapshot.snapshot for snapshot in state_snapshots
                )
                assert len(message_snapshots) >= 1
                assert isinstance(message_snapshots[-1].messages, list)
                assert len(message_snapshots[-1].messages) >= 1

        def when_messages_reducer_present():
            async def it_emits_messages_snapshot():
                @on(UserAsked)
                def reply(event: UserAsked) -> AgentReplied:
                    return AgentReplied(message=AIMessage(content="hello"))

                graph = EventGraph(
                    [reply],
                    reducers=[message_reducer()],
                )
                adapter = AGUIAdapter(
                    graph=graph,
                    seed_factory=lambda inp: UserAsked(question="hi"),
                    include_reducers=True,
                )
                events = await _collect(adapter, _make_input())

                msg_snapshots = [
                    e for e in events if e.type == EventType.MESSAGES_SNAPSHOT
                ]
                assert len(msg_snapshots) >= 1
                # Should contain converted messages
                last_snap = msg_snapshots[-1]
                assert isinstance(last_snap.messages, list)
                assert len(last_snap.messages) >= 1

        def when_include_reducers_has_no_messages_reducer():
            async def it_still_emits_text_message_events():
                """Mapper must not be suppressed without messages reducer."""
                counter = ScalarReducer(
                    name="counter",
                    event_type=AgentReplied,
                    fn=lambda e: 1,
                    default=0,
                )

                @on(UserAsked)
                def reply(event: UserAsked) -> AgentReplied:
                    return AgentReplied(message=AIMessage(content="hello"))

                graph = EventGraph([reply], reducers=[counter])
                adapter = AGUIAdapter(
                    graph=graph,
                    seed_factory=lambda inp: UserAsked(question="hi"),
                    include_reducers=True,
                )
                events = await _collect(adapter, _make_input())

                starts = [e for e in events if e.type == EventType.TEXT_MESSAGE_START]
                assert len(starts) == 1

                msg_snapshots = [
                    e for e in events if e.type == EventType.MESSAGES_SNAPSHOT
                ]
                assert len(msg_snapshots) == 0

        def when_error_message_provided():
            async def it_uses_custom_error_message():
                @on(ErrorTrigger)
                def blow_up(event: ErrorTrigger) -> None:
                    raise RuntimeError("boom")

                graph = EventGraph([blow_up])
                adapter = AGUIAdapter(
                    graph=graph,
                    seed_factory=lambda inp: ErrorTrigger(),
                    error_message="Something went wrong. Please try again.",
                )
                events = await _collect(adapter, _make_input())

                error_events = [e for e in events if e.type == EventType.RUN_ERROR]
                assert len(error_events) == 1
                assert (
                    error_events[0].message == "Something went wrong. Please try again."
                )

        def when_messages_unchanged():
            async def it_skips_redundant_messages_snapshots():
                @on(UserAsked)
                def reply(event: UserAsked) -> AgentReplied:
                    return AgentReplied(message=AIMessage(content="hello"))

                @on(AgentReplied)
                def followup(event: AgentReplied) -> TaskCreated:
                    return TaskCreated(title="no new messages")

                graph = EventGraph(
                    [reply, followup],
                    reducers=[message_reducer()],
                )
                adapter = AGUIAdapter(
                    graph=graph,
                    seed_factory=lambda inp: UserAsked(question="hi"),
                    include_reducers=True,
                )
                events = await _collect(adapter, _make_input())

                state_count = sum(
                    1 for e in events if e.type == EventType.STATE_SNAPSHOT
                )
                msg_count = sum(
                    1 for e in events if e.type == EventType.MESSAGES_SNAPSHOT
                )
                assert state_count > msg_count

            async def it_detects_message_content_changes():
                """MessagesSnapshot emits when add_messages replaces in-place."""

                class UserSent(MessageEvent):
                    message: HumanMessage = None  # type: ignore[assignment]

                class AgentDrafted(MessageEvent):
                    message: AIMessage = None  # type: ignore[assignment]

                class AgentRevised(MessageEvent):
                    message: AIMessage = None  # type: ignore[assignment]

                @on(UserSent)
                def draft(event: UserSent) -> AgentDrafted:
                    return AgentDrafted(
                        message=AIMessage(content="draft v1", id="msg-ai")
                    )

                @on(AgentDrafted)
                def revise(event: AgentDrafted) -> AgentRevised:
                    # Same ID, different content — add_messages replaces in-place
                    return AgentRevised(
                        message=AIMessage(content="draft v2", id="msg-ai")
                    )

                graph = EventGraph(
                    [draft, revise],
                    reducers=[message_reducer()],
                )
                adapter = AGUIAdapter(
                    graph=graph,
                    seed_factory=lambda inp: UserSent(
                        message=HumanMessage(content="go")
                    ),
                    include_reducers=True,
                )
                events = await _collect(adapter, _make_input())

                msg_snapshots = [
                    e for e in events if e.type == EventType.MESSAGES_SNAPSHOT
                ]
                # Must have at least 2: one after draft, one after revise
                assert len(msg_snapshots) >= 2
                # The last snapshot must contain the REVISED content
                last_snap = msg_snapshots[-1]
                ai_msgs = [m for m in last_snap.messages if m.role == "assistant"]
                assert len(ai_msgs) == 1
                assert ai_msgs[0].content == "draft v2"

            async def it_detects_message_tool_call_changes():
                """MessagesSnapshot emits when AI tool_calls change in-place."""

                @on(UserAsked)
                def draft(event: UserAsked) -> AgentCalledTools:
                    return AgentCalledTools(
                        message=AIMessage(
                            id="msg-ai",
                            content="working",
                            tool_calls=[
                                {
                                    "id": "tc-1",
                                    "name": "lookup",
                                    "args": {"query": "v1"},
                                }
                            ],
                        )
                    )

                @on(AgentCalledTools)
                def revise(event: AgentCalledTools) -> AgentReplied:
                    # Same ID/content, different tool_calls
                    return AgentReplied(
                        message=AIMessage(
                            id="msg-ai",
                            content="working",
                            tool_calls=[
                                {
                                    "id": "tc-2",
                                    "name": "lookup",
                                    "args": {"query": "v2"},
                                }
                            ],
                        )
                    )

                graph = EventGraph(
                    [draft, revise],
                    reducers=[message_reducer()],
                )
                adapter = AGUIAdapter(
                    graph=graph,
                    seed_factory=lambda inp: UserAsked(question="go"),
                    include_reducers=True,
                )
                events = await _collect(adapter, _make_input())

                msg_snapshots = [
                    e for e in events if e.type == EventType.MESSAGES_SNAPSHOT
                ]
                # Must have at least 2: one after draft, one after revise
                assert len(msg_snapshots) >= 2

                first_ai = next(
                    m for m in msg_snapshots[0].messages if m.role == "assistant"
                )
                last_ai = next(
                    m for m in msg_snapshots[-1].messages if m.role == "assistant"
                )

                assert first_ai.tool_calls is not None
                assert last_ai.tool_calls is not None
                assert first_ai.tool_calls[0].id == "tc-1"
                assert last_ai.tool_calls[0].id == "tc-2"
                assert json.loads(last_ai.tool_calls[0].function.arguments) == {
                    "query": "v2"
                }

        def when_multimodal_ai_message():
            async def it_handles_list_content_in_snapshot():
                """Multimodal AIMessage.content (list) must not crash snapshot."""

                class MultimodalReply(MessageEvent):
                    message: AIMessage = None  # type: ignore[assignment]

                @on(UserAsked)
                def reply(event: UserAsked) -> MultimodalReply:
                    return MultimodalReply(
                        message=AIMessage(
                            content=[
                                {"type": "text", "text": "hello"},
                                {
                                    "type": "image_url",
                                    "image_url": {"url": "data:image/png;base64,abc"},
                                },
                            ],
                            id="multi-1",
                        )
                    )

                graph = EventGraph(
                    [reply],
                    reducers=[message_reducer()],
                )
                adapter = AGUIAdapter(
                    graph=graph,
                    seed_factory=lambda inp: UserAsked(question="go"),
                    include_reducers=True,
                )
                events = await _collect(adapter, _make_input())

                # Must not crash — no RUN_ERROR
                error_events = [e for e in events if e.type == EventType.RUN_ERROR]
                assert len(error_events) == 0

                msg_snapshots = [
                    e for e in events if e.type == EventType.MESSAGES_SNAPSHOT
                ]
                assert len(msg_snapshots) >= 1

                last_snap = msg_snapshots[-1]
                ai_msgs = [m for m in last_snap.messages if m.role == "assistant"]
                assert len(ai_msgs) == 1
                # List content not representable as str → None
                assert ai_msgs[0].content is None

    def describe_custom_mappers():
        async def it_allows_user_mapper_to_claim_events():
            class TaskMapper:
                def map(self, event, ctx):
                    if isinstance(event, TaskCreated):
                        return [
                            CustomEvent(
                                type=EventType.CUSTOM,
                                name="task.created",
                                value={"title": event.title},
                            )
                        ]
                    return None

            @on(UserAsked)
            def create_task(event: UserAsked) -> TaskCreated:
                return TaskCreated(title="my task")

            graph = EventGraph([create_task])
            adapter = AGUIAdapter(
                graph=graph,
                seed_factory=lambda inp: UserAsked(question="create"),
                mappers=[TaskMapper()],
            )
            events = await _collect(adapter, _make_input())

            custom = [
                e
                for e in events
                if e.type == EventType.CUSTOM and e.name == "task.created"
            ]
            assert len(custom) == 1
            assert custom[0].value["title"] == "my task"

        async def it_user_mapper_before_fallback():
            """User mappers run before FallbackMapper, claiming events first."""

            class TaskMapper:
                def map(self, event, ctx):
                    if isinstance(event, TaskCreated):
                        return [
                            CustomEvent(
                                type=EventType.CUSTOM,
                                name="custom.task",
                                value={"t": event.title},
                            )
                        ]
                    return None

            @on(UserAsked)
            def create_task(event: UserAsked) -> TaskCreated:
                return TaskCreated(title="test")

            graph = EventGraph([create_task])
            adapter = AGUIAdapter(
                graph=graph,
                seed_factory=lambda inp: UserAsked(question="go"),
                mappers=[TaskMapper()],
            )
            events = await _collect(adapter, _make_input())

            # Should see custom.task (user mapper), not TaskCreated (fallback)
            task_events = [
                e
                for e in events
                if e.type == EventType.CUSTOM
                and e.name in ("custom.task", "TaskCreated")
            ]
            assert len(task_events) == 1
            assert task_events[0].name == "custom.task"

    def describe_resume():
        def when_resume_factory_available():
            async def it_uses_resume_factory():
                @on(UserAsked)
                def ask(event: UserAsked) -> ApprovalRequested:
                    return ApprovalRequested(draft="draft")

                @on(ApprovalGiven)
                def approve(event: ApprovalGiven) -> AgentReplied:
                    return AgentReplied(message=AIMessage(content="Done!"))

                graph = EventGraph([ask, approve], checkpointer=MemorySaver())

                # Create the interrupt
                config = {"configurable": {"thread_id": "t-resume-factory"}}
                await graph.ainvoke(UserAsked(question="go"), config=config)

                # Resume via adapter
                adapter = AGUIAdapter(
                    graph=graph,
                    seed_factory=lambda inp: UserAsked(question="unused"),
                    resume_factory=lambda inp: ApprovalGiven(approved=True),
                )
                events = await _collect(
                    adapter, _make_input(thread_id="t-resume-factory")
                )

                assert events[0].type == EventType.RUN_STARTED
                assert events[-1].type == EventType.RUN_FINISHED

                # Should contain text from the approve handler
                text_events = [
                    e for e in events if e.type == EventType.TEXT_MESSAGE_CONTENT
                ]
                assert any("Done!" in e.delta for e in text_events)

            async def it_streams_resume_events():
                @on(UserAsked)
                def ask(event: UserAsked) -> ApprovalRequested:
                    return ApprovalRequested(draft="check this")

                @on(ApprovalGiven)
                def approve(event: ApprovalGiven) -> TaskCreated:
                    return TaskCreated(title="approved task")

                graph = EventGraph([ask, approve], checkpointer=MemorySaver())

                config = {"configurable": {"thread_id": "t-stream-resume"}}
                await graph.ainvoke(UserAsked(question="go"), config=config)

                adapter = AGUIAdapter(
                    graph=graph,
                    seed_factory=lambda inp: UserAsked(question="unused"),
                    resume_factory=lambda inp: ApprovalGiven(approved=True),
                )
                events = await _collect(
                    adapter, _make_input(thread_id="t-stream-resume")
                )

                # Should have TaskCreated as a custom event
                custom = [
                    e
                    for e in events
                    if e.type == EventType.CUSTOM and e.name == "TaskCreated"
                ]
                assert len(custom) == 1
                assert custom[0].value["title"] == "approved task"

            async def it_does_not_emit_interrupted_during_successful_resume():
                @on(UserAsked)
                def ask(event: UserAsked) -> ApprovalRequested:
                    return ApprovalRequested(draft="needs approval")

                @on(ApprovalGiven)
                def approve(event: ApprovalGiven) -> AgentReplied:
                    return AgentReplied(message=AIMessage(content="Approved!"))

                graph = EventGraph([ask, approve], checkpointer=MemorySaver())

                # Create the interrupt
                config = {"configurable": {"thread_id": "t-no-stale-interrupt"}}
                await graph.ainvoke(UserAsked(question="go"), config=config)

                # Resume — should complete successfully
                adapter = AGUIAdapter(
                    graph=graph,
                    seed_factory=lambda inp: UserAsked(question="unused"),
                    resume_factory=lambda inp: ApprovalGiven(approved=True),
                )
                events = await _collect(
                    adapter, _make_input(thread_id="t-no-stale-interrupt")
                )

                assert events[-1].type == EventType.RUN_FINISHED

                # No "interrupted" CustomEvents should appear during successful resume
                interrupted_events = [
                    e
                    for e in events
                    if e.type == EventType.CUSTOM and e.name == "interrupted"
                ]
                assert len(interrupted_events) == 0

            async def it_emits_new_interrupted_event_created_during_resume():
                @on(UserAsked)
                def ask(event: UserAsked) -> ApprovalRequested:
                    return ApprovalRequested(draft="walkthrough preference")

                @on(ApprovalGiven)
                def request_persona_review(event: ApprovalGiven) -> ApprovalRequested:
                    return ApprovalRequested(draft="persona review")

                graph = EventGraph(
                    [ask, request_persona_review],
                    checkpointer=MemorySaver(),
                )

                # First run creates an interrupt.
                config = {"configurable": {"thread_id": "t-resume-new-interrupt"}}
                await graph.ainvoke(UserAsked(question="go"), config=config)

                # Resume creates a new interrupt, which is suppressed
                # in-stream and must be detected from the post-stream
                # checkpoint read.
                adapter = AGUIAdapter(
                    graph=graph,
                    seed_factory=lambda inp: UserAsked(question="unused"),
                    resume_factory=lambda inp: ApprovalGiven(approved=True),
                )
                events = await _collect(
                    adapter, _make_input(thread_id="t-resume-new-interrupt")
                )

                interrupted_events = [
                    e
                    for e in events
                    if e.type == EventType.CUSTOM and e.name == "interrupted"
                ]
                assert len(interrupted_events) == 1
                assert interrupted_events[0].value["draft"] == "persona review"
                assert events[-1].type == EventType.RUN_FINISHED

            def when_stream_emits_interrupt():

                async def it_skips_post_stream_checkpoint_read(
                    monkeypatch,
                ):
                    @on(UserAsked)
                    def reply(event: UserAsked) -> AgentReplied:
                        return AgentReplied(message=AIMessage(content="ok"))

                    graph = EventGraph([reply], checkpointer=MemorySaver())
                    adapter = AGUIAdapter(
                        graph=graph,
                        seed_factory=lambda inp: UserAsked(question="go"),
                    )

                    called = {"checkpoint": 0}

                    async def fake_astream_events(
                        seed,
                        *,
                        include_reducers,
                        include_llm_tokens,
                        include_custom_events,
                        config,
                    ):
                        del (
                            seed,
                            include_reducers,
                            include_llm_tokens,
                            include_custom_events,
                            config,
                        )
                        yield ApprovalRequested(draft="stream-first")

                    original_aget_state = graph.compiled.aget_state

                    async def count_aget_state(config):
                        called["checkpoint"] += 1
                        return await original_aget_state(config)

                    monkeypatch.setattr(
                        adapter._graph,
                        "astream_events",
                        fake_astream_events,
                    )
                    monkeypatch.setattr(graph.compiled, "aget_state", count_aget_state)
                    events = await _collect(
                        adapter, _make_input(thread_id="t-stream-int")
                    )

                    interrupted_events = [
                        e
                        for e in events
                        if e.type == EventType.CUSTOM and e.name == "interrupted"
                    ]
                    assert len(interrupted_events) == 1
                    assert called["checkpoint"] == 1

            async def it_passes_checkpoint_state_to_resume_factory():
                seen_state: dict[str, Any] = {}

                @on(UserAsked)
                def ask(event: UserAsked) -> ApprovalRequested:
                    return ApprovalRequested(draft="needs approval")

                @on(ApprovalGiven)
                def approve(event: ApprovalGiven) -> AgentReplied:
                    return AgentReplied(message=AIMessage(content="Approved!"))

                graph = EventGraph(
                    [ask, approve],
                    checkpointer=MemorySaver(),
                    reducers=[message_reducer()],
                )

                config = {"configurable": {"thread_id": "t-resume-state"}}
                await graph.ainvoke(UserAsked(question="go"), config=config)

                def resume_state(
                    input_data: RunAgentInput,
                    checkpoint_state: dict[str, Any] | None,
                ) -> Event | None:
                    if checkpoint_state is not None:
                        seen_state.update(checkpoint_state)
                    return ApprovalGiven(approved=True)

                adapter = AGUIAdapter(
                    graph=graph,
                    seed_factory=lambda inp: UserAsked(question="unused"),
                    resume_factory=resume_state,
                )
                events = await _collect(
                    adapter, _make_input(thread_id="t-resume-state")
                )

                assert events[-1].type == EventType.RUN_FINISHED
                assert seen_state["is_interrupted"] is True
                assert isinstance(seen_state["reducers"], dict)
                assert "events" in seen_state
                assert "messages" in seen_state
                assert len(seen_state["pending_interrupts"]) == 1

    def describe_init():
        def when_resume_factory_missing_checkpointer():
            def it_raises():
                @on(UserAsked)
                def reply(event: UserAsked) -> AgentReplied:
                    return AgentReplied(message=AIMessage(content="hi"))

                graph = EventGraph([reply])  # no checkpointer
                with pytest.raises(
                    ValueError, match="resume_factory requires a checkpointer"
                ):
                    AGUIAdapter(
                        graph=graph,
                        seed_factory=lambda inp: UserAsked(question="hi"),
                        resume_factory=lambda inp: ApprovalGiven(approved=True),
                    )

    def describe_seed_factory():
        async def it_passes_input_to_factory():
            received_inputs: list[Any] = []

            def tracking_factory(inp: RunAgentInput) -> UserAsked:
                received_inputs.append(inp)
                return UserAsked(question="from factory")

            @on(UserAsked)
            def reply(event: UserAsked) -> AgentReplied:
                return AgentReplied(message=AIMessage(content=event.question))

            graph = EventGraph([reply])
            adapter = AGUIAdapter(
                graph=graph,
                seed_factory=tracking_factory,
            )
            input_data = _make_input(thread_id="t-seed")
            events = await _collect(adapter, input_data)

            assert len(received_inputs) == 1
            assert received_inputs[0].thread_id == "t-seed"

            text_events = [
                e for e in events if e.type == EventType.TEXT_MESSAGE_CONTENT
            ]
            assert any("from factory" in e.delta for e in text_events)


def describe_transport():
    async def it_encodes_events_as_sse():
        from ag_ui.core import RunStartedEvent

        from langgraph_events.agui import encode_sse_stream

        async def _events():
            yield RunStartedEvent(
                type=EventType.RUN_STARTED,
                thread_id="t1",
                run_id="r1",
            )

        lines = [line async for line in encode_sse_stream(_events())]
        assert len(lines) == 1
        assert lines[0].startswith("data: ")
        assert lines[0].endswith("\n\n")

    async def it_creates_starlette_response():
        pytest.importorskip("starlette")
        from ag_ui.core import RunStartedEvent
        from starlette.responses import StreamingResponse

        from langgraph_events.agui import create_starlette_response

        async def _events():
            yield RunStartedEvent(
                type=EventType.RUN_STARTED,
                thread_id="t1",
                run_id="r1",
            )

        response = create_starlette_response(_events())
        assert isinstance(response, StreamingResponse)


def describe_interrupt_detection():
    async def it_detects_interrupts_from_checkpoint():
        """Adapter emits interrupted CustomEvent from checkpoint."""

        @on(UserAsked)
        def ask_approval(event: UserAsked) -> ApprovalRequested:
            return ApprovalRequested(draft="needs review")

        graph = EventGraph([ask_approval], checkpointer=MemorySaver())
        adapter = AGUIAdapter(
            graph=graph,
            seed_factory=lambda inp: UserAsked(question="check"),
        )
        events = await _collect(adapter, _make_input())

        interrupted_events = [
            e for e in events if e.type == EventType.CUSTOM and e.name == "interrupted"
        ]
        assert len(interrupted_events) == 1
        assert interrupted_events[0].value["draft"] == "needs review"


def describe_resume_reducers():
    async def it_emits_state_snapshot_during_resume():
        @on(UserAsked)
        def ask(event: UserAsked) -> ApprovalRequested:
            return ApprovalRequested(draft="draft")

        @on(ApprovalGiven)
        def approve(event: ApprovalGiven) -> AgentReplied:
            return AgentReplied(message=AIMessage(content="Approved!"))

        graph = EventGraph(
            [ask, approve],
            checkpointer=MemorySaver(),
            reducers=[message_reducer()],
        )

        config = {"configurable": {"thread_id": "t-resume-reducers"}}
        await graph.ainvoke(UserAsked(question="go"), config=config)

        adapter = AGUIAdapter(
            graph=graph,
            seed_factory=lambda inp: UserAsked(question="unused"),
            resume_factory=lambda inp: ApprovalGiven(approved=True),
            include_reducers=True,
        )
        events = await _collect(adapter, _make_input(thread_id="t-resume-reducers"))

        snapshots = [e for e in events if e.type == EventType.STATE_SNAPSHOT]
        assert len(snapshots) >= 1


def describe_connect():
    def when_reconnect():

        def it_is_alias_for_connect():
            assert AGUIAdapter.reconnect is AGUIAdapter.connect

    def when_checkpointer_present():

        def when_existing_thread():

            async def it_replays_state_from_checkpoint():
                @on(UserAsked)
                def ask(event: UserAsked) -> ApprovalRequested:
                    return ApprovalRequested(draft="pending")

                graph = EventGraph(
                    [ask],
                    checkpointer=MemorySaver(),
                    reducers=[message_reducer()],
                )
                await graph.ainvoke(
                    UserAsked(question="go"),
                    config={"configurable": {"thread_id": "t-connect"}},
                )

                adapter = AGUIAdapter(
                    graph=graph,
                    seed_factory=lambda inp: UserAsked(question="unused"),
                )
                events = [
                    event
                    async for event in adapter.connect(
                        _make_input(thread_id="t-connect")
                    )
                ]

                assert any(e.type == EventType.STATE_SNAPSHOT for e in events)
                assert any(e.type == EventType.MESSAGES_SNAPSHOT for e in events)
                interrupted = [
                    e
                    for e in events
                    if e.type == EventType.CUSTOM and e.name == "interrupted"
                ]
                assert len(interrupted) == 1
                assert interrupted[0].value["draft"] == "pending"

        def when_new_thread():

            async def it_emits_empty_snapshots():
                @on(UserAsked)
                def reply(event: UserAsked) -> AgentReplied:
                    return AgentReplied(message=AIMessage(content="ok"))

                graph = EventGraph(
                    [reply],
                    checkpointer=MemorySaver(),
                    reducers=[message_reducer()],
                )
                adapter = AGUIAdapter(
                    graph=graph,
                    seed_factory=lambda inp: UserAsked(question="unused"),
                )

                events = [
                    event
                    async for event in adapter.connect(
                        _make_input(thread_id="brand-new-thread")
                    )
                ]
                assert len(events) == 2
                assert events[0].type == EventType.STATE_SNAPSHOT
                assert events[0].snapshot == {}
                assert events[1].type == EventType.MESSAGES_SNAPSHOT
                assert events[1].messages == []

    def when_no_checkpointer():

        async def it_emits_empty_snapshots():
            @on(UserAsked)
            def reply(event: UserAsked) -> AgentReplied:
                return AgentReplied(message=AIMessage(content="ok"))

            graph = EventGraph([reply])
            adapter = AGUIAdapter(
                graph=graph,
                seed_factory=lambda inp: UserAsked(question="unused"),
            )

            events = [event async for event in adapter.connect(_make_input())]
            assert len(events) == 2
            assert events[0].type == EventType.STATE_SNAPSHOT
            assert events[0].snapshot == {}
            assert events[1].type == EventType.MESSAGES_SNAPSHOT
            assert events[1].messages == []

    def when_forwarded_props():

        async def it_passes_config_to_aget_state(monkeypatch):
            @on(UserAsked)
            def ask(event: UserAsked) -> ApprovalRequested:
                return ApprovalRequested(draft="pending")

            graph = EventGraph(
                [ask],
                checkpointer=MemorySaver(),
                reducers=[message_reducer()],
            )
            await graph.ainvoke(
                UserAsked(question="go"),
                config={"configurable": {"thread_id": "t-connect-config"}},
            )

            captured: list[dict[str, Any]] = []
            original_aget_state = graph.compiled.aget_state

            async def capture_aget_state(config):
                captured.append(config)
                return await original_aget_state(config)

            monkeypatch.setattr(graph.compiled, "aget_state", capture_aget_state)

            adapter = AGUIAdapter(
                graph=graph,
                seed_factory=lambda inp: UserAsked(question="unused"),
            )
            events = [
                event
                async for event in adapter.connect(
                    _make_input(
                        thread_id="t-connect-config",
                        forwarded_props={
                            "langgraph_config": {
                                "recursion_limit": 13,
                                "configurable": {"tenant_id": "acme"},
                            }
                        },
                    )
                )
            ]

            assert len(captured) == 1
            assert captured[0]["recursion_limit"] == 13
            assert captured[0]["configurable"]["tenant_id"] == "acme"
            assert captured[0]["configurable"]["thread_id"] == "t-connect-config"
            assert any(e.type == EventType.STATE_SNAPSHOT for e in events)


def describe_config_passthrough():
    def _setup_stream_capture(monkeypatch):
        @on(UserAsked)
        def reply(event: UserAsked) -> AgentReplied:
            return AgentReplied(message=AIMessage(content="ok"))

        graph = EventGraph([reply])
        captured: list[dict[str, Any]] = []

        async def fake_astream_events(
            seed,
            *,
            include_reducers,
            include_llm_tokens,
            include_custom_events,
            config,
        ):
            del seed, include_reducers, include_llm_tokens, include_custom_events
            captured.append(config)
            if False:  # pragma: no cover
                yield None

        monkeypatch.setattr(graph, "astream_events", fake_astream_events)

        adapter = AGUIAdapter(
            graph=graph,
            seed_factory=lambda inp: UserAsked(question="go"),
        )
        return adapter, captured

    async def it_forwards_langgraph_config_from_forwarded_props(monkeypatch):
        adapter, captured = _setup_stream_capture(monkeypatch)
        input_data = _make_input(
            thread_id="t-config",
            forwarded_props={
                "langgraph_config": {
                    "recursion_limit": 7,
                    "configurable": {"tenant_id": "acme"},
                }
            },
        )

        events = await _collect(adapter, input_data)

        assert events[0].type == EventType.RUN_STARTED
        assert events[-1].type == EventType.RUN_FINISHED
        assert len(captured) == 1
        assert captured[0]["recursion_limit"] == 7
        assert captured[0]["configurable"]["tenant_id"] == "acme"
        assert captured[0]["configurable"]["thread_id"] == "t-config"

    async def it_forwards_config_key_from_forwarded_props(monkeypatch):
        adapter, captured = _setup_stream_capture(monkeypatch)
        input_data = _make_input(
            thread_id="t-config-key",
            forwarded_props={
                "config": {
                    "recursion_limit": 9,
                    "configurable": {
                        "tenant_id": "acme",
                        "thread_id": "wrong-thread",
                    },
                }
            },
        )

        events = await _collect(adapter, input_data)

        assert events[0].type == EventType.RUN_STARTED
        assert events[-1].type == EventType.RUN_FINISHED
        assert len(captured) == 1
        assert captured[0]["recursion_limit"] == 9
        assert captured[0]["configurable"]["tenant_id"] == "acme"
        # Request thread_id wins over forwarded config thread_id.
        assert captured[0]["configurable"]["thread_id"] == "t-config-key"

    async def it_accepts_top_level_langgraph_config_shape(monkeypatch):
        adapter, captured = _setup_stream_capture(monkeypatch)
        input_data = _make_input(
            thread_id="t-top-level",
            forwarded_props={
                "recursion_limit": 11,
                "configurable": {
                    "tenant_id": "top-level-tenant",
                    "thread_id": "ignored-thread",
                },
            },
        )

        events = await _collect(adapter, input_data)

        assert events[0].type == EventType.RUN_STARTED
        assert events[-1].type == EventType.RUN_FINISHED
        assert len(captured) == 1
        assert captured[0]["recursion_limit"] == 11
        assert captured[0]["configurable"]["tenant_id"] == "top-level-tenant"
        assert captured[0]["configurable"]["thread_id"] == "t-top-level"

    async def it_ignores_unrecognized_forwarded_props(monkeypatch):
        adapter, captured = _setup_stream_capture(monkeypatch)
        input_data = _make_input(
            thread_id="t-default-config",
            forwarded_props={"foo": "bar"},
        )

        events = await _collect(adapter, input_data)

        assert events[0].type == EventType.RUN_STARTED
        assert events[-1].type == EventType.RUN_FINISHED
        assert len(captured) == 1
        assert captured[0]["configurable"]["thread_id"] == "t-default-config"
        assert "foo" not in captured[0]


def describe_custom_event_passthrough():
    async def it_maps_state_snapshot_frame_to_state_snapshot(monkeypatch):
        from langgraph_events._graph import StateSnapshotFrame

        @on(UserAsked)
        def reply(event: UserAsked) -> AgentReplied:
            return AgentReplied(message=AIMessage(content="ok"))

        graph = EventGraph([reply])
        calls: list[dict[str, Any]] = []

        async def fake_astream_events(
            seed,
            *,
            include_reducers,
            include_llm_tokens,
            include_custom_events,
            config,
        ):
            calls.append(
                {
                    "include_reducers": include_reducers,
                    "include_llm_tokens": include_llm_tokens,
                    "include_custom_events": include_custom_events,
                }
            )
            del (
                seed,
                include_reducers,
                include_llm_tokens,
                include_custom_events,
                config,
            )
            yield StateSnapshotFrame(data={"messages": [], "step": "draft"})

        monkeypatch.setattr(graph, "astream_events", fake_astream_events)

        adapter = AGUIAdapter(
            graph=graph,
            seed_factory=lambda inp: UserAsked(question="go"),
        )

        events = await _collect(adapter, _make_input())
        snapshots = [e for e in events if e.type == EventType.STATE_SNAPSHOT]

        assert len(calls) == 1
        assert calls[0]["include_llm_tokens"] is True
        assert calls[0]["include_custom_events"] is True
        assert len(snapshots) == 1
        assert snapshots[0].snapshot == {"messages": [], "step": "draft"}

    async def it_passes_intermediate_state_custom_event_frame_through(monkeypatch):
        from langgraph_events._custom_event import STATE_SNAPSHOT_EVENT_NAME
        from langgraph_events._graph import CustomEventFrame

        @on(UserAsked)
        def reply(event: UserAsked) -> AgentReplied:
            return AgentReplied(message=AIMessage(content="ok"))

        graph = EventGraph([reply])

        async def fake_astream_events(
            seed,
            *,
            include_reducers,
            include_llm_tokens,
            include_custom_events,
            config,
        ):
            del (
                seed,
                include_reducers,
                include_llm_tokens,
                include_custom_events,
                config,
            )
            yield CustomEventFrame(
                name=STATE_SNAPSHOT_EVENT_NAME,
                data={"messages": [], "step": "draft"},
            )

        monkeypatch.setattr(graph, "astream_events", fake_astream_events)

        adapter = AGUIAdapter(
            graph=graph,
            seed_factory=lambda inp: UserAsked(question="go"),
        )

        events = await _collect(adapter, _make_input())
        custom_events = [
            e
            for e in events
            if e.type == EventType.CUSTOM and e.name == STATE_SNAPSHOT_EVENT_NAME
        ]

        assert len(custom_events) == 1
        assert custom_events[0].value == {"messages": [], "step": "draft"}

    async def it_maps_custom_event_frame_to_custom_event(monkeypatch):
        from langgraph_events._graph import CustomEventFrame

        @on(UserAsked)
        def reply(event: UserAsked) -> AgentReplied:
            return AgentReplied(message=AIMessage(content="ok"))

        graph = EventGraph([reply])

        async def fake_astream_events(
            seed,
            *,
            include_reducers,
            include_llm_tokens,
            include_custom_events,
            config,
        ):
            del (
                seed,
                include_reducers,
                include_llm_tokens,
                include_custom_events,
                config,
            )
            yield CustomEventFrame(name="tool.progress", data={"pct": 80})

        monkeypatch.setattr(graph, "astream_events", fake_astream_events)

        adapter = AGUIAdapter(
            graph=graph,
            seed_factory=lambda inp: UserAsked(question="go"),
        )

        events = await _collect(adapter, _make_input())
        custom_events = [
            e
            for e in events
            if e.type == EventType.CUSTOM and e.name == "tool.progress"
        ]

        assert len(custom_events) == 1
        assert custom_events[0].value == {"pct": 80}


def describe_async_checkpoint_reads():
    async def it_uses_aget_state_for_interrupt_detection(monkeypatch):
        @on(UserAsked)
        def ask(event: UserAsked) -> ApprovalRequested:
            return ApprovalRequested(draft="needs review")

        graph = EventGraph([ask], checkpointer=MemorySaver())
        adapter = AGUIAdapter(
            graph=graph,
            seed_factory=lambda inp: UserAsked(question="go"),
        )

        called = {"sync": False}

        def fail_get_state(config):
            called["sync"] = True
            raise AssertionError("sync get_state should not be called")

        monkeypatch.setattr(graph.compiled, "get_state", fail_get_state)
        events = await _collect(adapter, _make_input(thread_id="t-async-state"))

        assert called["sync"] is False
        interrupted = [
            e for e in events if e.type == EventType.CUSTOM and e.name == "interrupted"
        ]
        assert len(interrupted) == 1


def describe_seed_factory_state():
    def when_factory_accepts_state():
        async def it_passes_checkpoint_state_to_seed_factory():
            received_states: list[Any] = []

            def stateful_factory(inp: RunAgentInput, state: Any) -> UserAsked:
                received_states.append(state)
                return UserAsked(question="from stateful")

            @on(UserAsked)
            def reply(event: UserAsked) -> AgentReplied:
                return AgentReplied(message=AIMessage(content="ok"))

            graph = EventGraph([reply], checkpointer=MemorySaver())
            # Create a thread with state
            await graph.ainvoke(
                UserAsked(question="first"),
                config={"configurable": {"thread_id": "t-seed-state"}},
            )

            adapter = AGUIAdapter(
                graph=graph,
                seed_factory=stateful_factory,
            )
            await _collect(adapter, _make_input(thread_id="t-seed-state"))

            assert len(received_states) == 1
            state = received_states[0]
            assert state is not None
            assert "events" in state
            # Raw snapshot passthrough for advanced access
            assert state["snapshot"] is not None
            assert hasattr(state["snapshot"], "values")
            assert hasattr(state["snapshot"], "next")
            # Prove identity — snapshot.values IS the reducers dict
            snapshot_vals = state["snapshot"].values
            assert isinstance(snapshot_vals, dict)
            assert snapshot_vals == state["reducers"]

    def when_single_arg_seed_factory():
        async def it_accepts_single_arg_factory():
            @on(UserAsked)
            def reply(event: UserAsked) -> AgentReplied:
                return AgentReplied(message=AIMessage(content="ok"))

            graph = EventGraph([reply])
            adapter = AGUIAdapter(
                graph=graph,
                seed_factory=lambda inp: UserAsked(question="simple"),
            )
            events = await _collect(adapter, _make_input())

            assert events[0].type == EventType.RUN_STARTED
            assert events[-1].type == EventType.RUN_FINISHED

    def when_no_checkpointer():
        async def it_seed_factory_receives_none():
            received_states: list[Any] = []

            def stateful_factory(inp: RunAgentInput, state: Any) -> UserAsked:
                received_states.append(state)
                return UserAsked(question="no cp")

            @on(UserAsked)
            def reply(event: UserAsked) -> AgentReplied:
                return AgentReplied(message=AIMessage(content="ok"))

            graph = EventGraph([reply])  # no checkpointer
            adapter = AGUIAdapter(
                graph=graph,
                seed_factory=stateful_factory,
            )
            await _collect(adapter, _make_input())

            assert len(received_states) == 1
            assert received_states[0] is None


def describe_interrupt_gate():
    def when_gated():
        async def it_reemits_interrupt():
            call_count = {"n": 0}

            @on(UserAsked)
            def ask(event: UserAsked) -> ApprovalRequested:
                call_count["n"] += 1
                return ApprovalRequested(draft="pending")

            graph = EventGraph(
                [ask],
                checkpointer=MemorySaver(),
                reducers=[message_reducer()],
            )
            # Create interrupted thread
            await graph.ainvoke(
                UserAsked(question="go"),
                config={"configurable": {"thread_id": "t-gate"}},
            )
            initial_calls = call_count["n"]

            adapter = AGUIAdapter(
                graph=graph,
                seed_factory=lambda inp: UserAsked(question="retry"),
                interrupt_gate=True,
            )
            events = await _collect(adapter, _make_input(thread_id="t-gate"))

            # Handler should NOT have been called again
            assert call_count["n"] == initial_calls
            # Should have state + interrupt, wrapped in run lifecycle
            assert events[0].type == EventType.RUN_STARTED
            assert events[-1].type == EventType.RUN_FINISHED
            assert any(e.type == EventType.STATE_SNAPSHOT for e in events)
            interrupted = [
                e
                for e in events
                if e.type == EventType.CUSTOM and e.name == "interrupted"
            ]
            assert len(interrupted) == 1
            assert interrupted[0].value["draft"] == "pending"

    def when_gate_disabled():
        async def it_executes_normally():
            @on(UserAsked)
            def ask(event: UserAsked) -> ApprovalRequested:
                return ApprovalRequested(draft="pending")

            graph = EventGraph([ask], checkpointer=MemorySaver())
            await graph.ainvoke(
                UserAsked(question="go"),
                config={"configurable": {"thread_id": "t-gate-off"}},
            )

            adapter = AGUIAdapter(
                graph=graph,
                seed_factory=lambda inp: UserAsked(question="new seed"),
                interrupt_gate=False,
            )
            events = await _collect(adapter, _make_input(thread_id="t-gate-off"))

            assert events[0].type == EventType.RUN_STARTED
            assert events[-1].type == EventType.RUN_FINISHED
            # Should have executed — UserAsked custom event appears
            custom = [
                e
                for e in events
                if e.type == EventType.CUSTOM and e.name == "UserAsked"
            ]
            assert len(custom) >= 1

    def when_not_interrupted():
        async def it_executes_normally():
            @on(UserAsked)
            def reply(event: UserAsked) -> AgentReplied:
                return AgentReplied(message=AIMessage(content="done"))

            graph = EventGraph([reply], checkpointer=MemorySaver())
            adapter = AGUIAdapter(
                graph=graph,
                seed_factory=lambda inp: UserAsked(question="go"),
                interrupt_gate=True,  # gate on, but no interrupt
            )
            events = await _collect(adapter, _make_input(thread_id="t-gate-clean"))

            assert events[0].type == EventType.RUN_STARTED
            assert events[-1].type == EventType.RUN_FINISHED
            text = [e for e in events if e.type == EventType.TEXT_MESSAGE_CONTENT]
            assert any("done" in e.delta for e in text)


def describe_AGUICustomEvent():
    def when_agui_event_name_implemented():
        async def it_uses_agui_event_name():
            class NamedEvent(Event):
                data: str = ""

                @property
                def agui_event_name(self) -> str:
                    return "custom.named"

                def agui_dict(self) -> dict[str, Any]:
                    return {"data": self.data}

            @on(UserAsked)
            def emit(event: UserAsked) -> NamedEvent:
                return NamedEvent(data="hello")

            graph = EventGraph([emit])
            adapter = AGUIAdapter(
                graph=graph,
                seed_factory=lambda inp: UserAsked(question="go"),
            )
            events = await _collect(adapter, _make_input())

            custom = [
                e
                for e in events
                if e.type == EventType.CUSTOM and e.name == "custom.named"
            ]
            assert len(custom) == 1
            assert custom[0].value == {"data": "hello"}

    def when_agui_event_name_not_implemented():
        async def it_falls_back_to_class_name():
            @on(UserAsked)
            def emit(event: UserAsked) -> TaskCreated:
                return TaskCreated(title="fallback")

            graph = EventGraph([emit])
            adapter = AGUIAdapter(
                graph=graph,
                seed_factory=lambda inp: UserAsked(question="go"),
            )
            events = await _collect(adapter, _make_input())

            # Should use class name, not agui_event_name
            custom = [
                e
                for e in events
                if e.type == EventType.CUSTOM and e.name == "TaskCreated"
            ]
            assert len(custom) == 1


def describe_connect_completed_thread():
    async def it_emits_state_and_messages():
        @on(UserAsked)
        def reply(event: UserAsked) -> AgentReplied:
            return AgentReplied(message=AIMessage(content="done"))

        graph = EventGraph(
            [reply],
            checkpointer=MemorySaver(),
            reducers=[message_reducer()],
        )
        await graph.ainvoke(
            UserAsked(question="go"),
            config={"configurable": {"thread_id": "t-completed"}},
        )

        adapter = AGUIAdapter(
            graph=graph,
            seed_factory=lambda inp: UserAsked(question="unused"),
        )
        events = [
            e async for e in adapter.connect(_make_input(thread_id="t-completed"))
        ]

        assert any(e.type == EventType.STATE_SNAPSHOT for e in events)
        assert any(e.type == EventType.MESSAGES_SNAPSHOT for e in events)
        # No interrupt — thread completed successfully
        interrupted = [
            e for e in events if e.type == EventType.CUSTOM and e.name == "interrupted"
        ]
        assert len(interrupted) == 0


def describe_agui_event_name_edge_cases():
    def when_event_has_name_but_no_agui_dict():

        async def it_warns_and_suppresses():
            class NameOnlyEvent(Event):
                data: str = ""

                @property
                def agui_event_name(self) -> str:
                    return "custom.name"

                # no agui_dict — not AGUISerializable

            @on(UserAsked)
            def emit(event: UserAsked) -> NameOnlyEvent:
                return NameOnlyEvent(data="test")

            graph = EventGraph([emit])
            adapter = AGUIAdapter(
                graph=graph,
                seed_factory=lambda inp: UserAsked(question="go"),
            )
            _warned_classes.discard(NameOnlyEvent)
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                events = await _collect(adapter, _make_input())

            # Not AGUISerializable → suppressed with warning
            custom = [
                e
                for e in events
                if e.type == EventType.CUSTOM and e.name == "custom.name"
            ]
            assert len(custom) == 0
            assert any("NameOnlyEvent" in str(x.message) for x in w)


def describe_multi_seed_factory():
    async def it_handles_list_of_seeds():
        @on(UserAsked)
        def reply(event: UserAsked) -> AgentReplied:
            return AgentReplied(message=AIMessage(content=f"Re: {event.question}"))

        graph = EventGraph([reply])
        adapter = AGUIAdapter(
            graph=graph,
            seed_factory=lambda inp: [
                UserAsked(question="first"),
                UserAsked(question="second"),
            ],
        )
        events = await _collect(adapter, _make_input())

        assert events[0].type == EventType.RUN_STARTED
        assert events[-1].type == EventType.RUN_FINISHED
        # Both seeds should produce replies
        text_events = [e for e in events if e.type == EventType.TEXT_MESSAGE_CONTENT]
        deltas = [e.delta for e in text_events]
        assert any("first" in d for d in deltas)
        assert any("second" in d for d in deltas)


def describe_ai_message_dedup():
    async def it_emits_text_message_start_exactly_once():
        """Streamed tokens should not be doubled by MessageEventMapper."""
        llm = FakeListChatModel(responses=["dedup me"], sleep=0)

        @on(UserAsked)
        async def stream_reply(
            event: UserAsked,
            messages: list[Any],
        ) -> AgentReplied:
            response = await llm.ainvoke(
                [*messages, HumanMessage(content=event.question)]
            )
            return AgentReplied(message=response)

        graph = EventGraph([stream_reply], reducers=[message_reducer()])
        adapter = AGUIAdapter(
            graph=graph,
            seed_factory=lambda inp: UserAsked(question="go"),
            include_reducers=True,
        )
        events = await _collect(adapter, _make_input())

        starts = [e for e in events if e.type == EventType.TEXT_MESSAGE_START]
        # Exactly one start — from streaming, not doubled by mapper
        assert len(starts) == 1


def describe_message_event_dedup():
    """Multiple MessageEvents carrying the same AI message must not duplicate."""

    def when_same_ai_id_across_events():
        async def it_emits_text_message_start_once():
            shared_ai = AIMessage(content="hello", id="ai-shared")

            @on(UserAsked)
            def step1(event: UserAsked) -> PhaseA:
                return PhaseA(messages=(shared_ai,))

            @on(PhaseA)
            def step2(event: PhaseA) -> PhaseB:
                return PhaseB(messages=(shared_ai,))

            graph = EventGraph([step1, step2])
            adapter = AGUIAdapter(
                graph=graph,
                seed_factory=lambda inp: UserAsked(question="go"),
            )
            events = await _collect(adapter, _make_input())

            starts = [e for e in events if e.type == EventType.TEXT_MESSAGE_START]
            assert len(starts) == 1

    def when_streamed_ai_reappears_via_message_event():
        async def it_does_not_reemit():
            llm = FakeListChatModel(responses=["streamed"], sleep=0)

            @on(UserAsked)
            async def stream_then_wrap(
                event: UserAsked,
                messages: list[Any],
            ) -> PhaseA:
                response = await llm.ainvoke(
                    [*messages, HumanMessage(content=event.question)]
                )
                return PhaseA(messages=(response,))

            graph = EventGraph([stream_then_wrap], reducers=[message_reducer()])
            adapter = AGUIAdapter(
                graph=graph,
                seed_factory=lambda inp: UserAsked(question="go"),
                include_reducers=True,
            )
            events = await _collect(adapter, _make_input())

            starts = [e for e in events if e.type == EventType.TEXT_MESSAGE_START]
            assert len(starts) == 1

    def without_message_ids():
        async def it_emits_both_because_dedup_cannot_apply():
            no_id = AIMessage(content="no id", id=None)

            @on(UserAsked)
            def step1(event: UserAsked) -> PhaseA:
                return PhaseA(messages=(no_id,))

            @on(PhaseA)
            def step2(event: PhaseA) -> PhaseB:
                return PhaseB(messages=(no_id,))

            graph = EventGraph([step1, step2])
            adapter = AGUIAdapter(
                graph=graph,
                seed_factory=lambda inp: UserAsked(question="go"),
            )
            events = await _collect(adapter, _make_input())

            starts = [e for e in events if e.type == EventType.TEXT_MESSAGE_START]
            assert len(starts) == 2

    def when_distinct_ai_ids_across_events():
        async def it_emits_both():
            ai1 = AIMessage(content="first", id="ai-1")
            ai2 = AIMessage(content="second", id="ai-2")

            @on(UserAsked)
            def step1(event: UserAsked) -> PhaseA:
                return PhaseA(messages=(ai1,))

            @on(PhaseA)
            def step2(event: PhaseA) -> PhaseB:
                return PhaseB(messages=(ai2,))

            graph = EventGraph([step1, step2])
            adapter = AGUIAdapter(
                graph=graph,
                seed_factory=lambda inp: UserAsked(question="go"),
            )
            events = await _collect(adapter, _make_input())

            starts = [e for e in events if e.type == EventType.TEXT_MESSAGE_START]
            assert len(starts) == 2

    def when_same_tool_message_id_across_events():
        async def it_emits_tool_call_result_once():
            shared_tool = ToolMessage(
                content="result", tool_call_id="tc-1", id="tool-shared"
            )

            @on(UserAsked)
            def step1(event: UserAsked) -> PhaseA:
                return PhaseA(messages=(shared_tool,))

            @on(PhaseA)
            def step2(event: PhaseA) -> PhaseB:
                return PhaseB(messages=(shared_tool,))

            graph = EventGraph([step1, step2])
            adapter = AGUIAdapter(
                graph=graph,
                seed_factory=lambda inp: UserAsked(question="go"),
            )
            events = await _collect(adapter, _make_input())

            results = [e for e in events if e.type == EventType.TOOL_CALL_RESULT]
            assert len(results) == 1


def describe_llm_streaming_on_resume():
    async def it_streams_tokens_during_resume():
        llm = FakeListChatModel(responses=["resumed reply"], sleep=0)

        @on(UserAsked)
        def ask(event: UserAsked) -> ApprovalRequested:
            return ApprovalRequested(draft="needs approval")

        @on(ApprovalGiven)
        async def approve(
            event: ApprovalGiven,
            messages: list[Any],
        ) -> AgentReplied:
            response = await llm.ainvoke([*messages, HumanMessage(content="approved")])
            return AgentReplied(message=response)

        graph = EventGraph(
            [ask, approve],
            checkpointer=MemorySaver(),
            reducers=[message_reducer()],
        )
        await graph.ainvoke(
            UserAsked(question="go"),
            config={"configurable": {"thread_id": "t-llm-resume"}},
        )

        adapter = AGUIAdapter(
            graph=graph,
            seed_factory=lambda inp: UserAsked(question="unused"),
            resume_factory=lambda inp: ApprovalGiven(approved=True),
            include_reducers=True,
        )
        events = await _collect(adapter, _make_input(thread_id="t-llm-resume"))

        assert events[0].type == EventType.RUN_STARTED
        assert events[-1].type == EventType.RUN_FINISHED
        # Should have streamed tokens
        deltas = [e.delta for e in events if e.type == EventType.TEXT_MESSAGE_CONTENT]
        assert len(deltas) >= 1
        assert "".join(deltas) == "resumed reply"


def describe_MapperContext():
    def it_generates_incrementing_message_ids():
        ctx = MapperContext(
            run_id="r1",
            thread_id="t1",
            input_data=_make_input(),
        )
        assert ctx.next_message_id() == "msg-1"
        assert ctx.next_message_id() == "msg-2"
        assert ctx.next_message_id() == "msg-3"

    def it_allows_reused_llm_run_id_after_close():
        ctx = MapperContext(
            run_id="r1",
            thread_id="t1",
            input_data=_make_input(),
        )

        first_id, is_new_first = ctx.ensure_stream_message_id("llm-run")
        assert is_new_first is True
        assert first_id == "msg-1"

        assert ctx.close_stream_message_id("llm-run") == "msg-1"

        second_id, is_new_second = ctx.ensure_stream_message_id("llm-run")
        assert is_new_second is True
        assert second_id == "msg-2"


def describe_snapshot_uses_langchain_ids():
    """Snapshots use original LangChain IDs for cross-run consistency."""

    async def it_uses_langchain_ids_not_streaming_ids():
        """Snapshot AI message keeps its LangChain ID, not the streaming msg-N ID."""
        llm = FakeListChatModel(responses=["hello"], sleep=0)

        @on(UserAsked)
        async def stream_reply(
            event: UserAsked,
            messages: list[Any],
        ) -> AgentReplied:
            response = await llm.ainvoke(
                [*messages, HumanMessage(content=event.question)]
            )
            return AgentReplied(message=response)

        graph = EventGraph([stream_reply], reducers=[message_reducer()])
        adapter = AGUIAdapter(
            graph=graph,
            seed_factory=lambda inp: UserAsked(question="go"),
            include_reducers=True,
        )
        events = await _collect(adapter, _make_input())

        starts = [e for e in events if e.type == EventType.TEXT_MESSAGE_START]
        assert len(starts) == 1
        stream_msg_id = starts[0].message_id

        msg_snapshots = [e for e in events if e.type == EventType.MESSAGES_SNAPSHOT]
        assert len(msg_snapshots) >= 1
        last_snap = msg_snapshots[-1]
        ai_msgs = [m for m in last_snap.messages if m.role == "assistant"]
        assert len(ai_msgs) >= 1
        # Snapshot uses LangChain ID, NOT the streaming ID
        assert ai_msgs[-1].id != stream_msg_id
        assert ai_msgs[-1].id.startswith("lc_run--")

    async def it_preserves_all_original_ids():
        """All messages in snapshot keep their LangChain IDs."""
        llm = FakeListChatModel(responses=["reply"], sleep=0)

        @on(UserSent)
        async def stream_reply(
            event: UserSent,
            messages: list[Any],
        ) -> AgentReplied:
            response = await llm.ainvoke(messages)
            return AgentReplied(message=response)

        graph = EventGraph([stream_reply], reducers=[message_reducer()])
        adapter = AGUIAdapter(
            graph=graph,
            seed_factory=lambda inp: UserSent(
                message=HumanMessage(content="hello", id="human-1")
            ),
            include_reducers=True,
        )
        events = await _collect(adapter, _make_input())

        msg_snapshots = [e for e in events if e.type == EventType.MESSAGES_SNAPSHOT]
        assert len(msg_snapshots) >= 1
        last_snap = msg_snapshots[-1]

        human_msgs = [m for m in last_snap.messages if m.role == "user"]
        assert len(human_msgs) >= 1
        assert human_msgs[0].id == "human-1"

        ai_msgs = [m for m in last_snap.messages if m.role == "assistant"]
        assert len(ai_msgs) >= 1
        # AI message uses LangChain ID, not streaming ID
        assert not ai_msgs[-1].id.startswith("msg-")

    async def it_uses_langchain_ids_across_multiple_llm_calls():
        """Both AI messages keep LangChain IDs after two LLM streams."""
        llm = FakeListChatModel(responses=["first reply", "second reply"], sleep=0)

        @on(UserAsked)
        async def first_reply(
            event: UserAsked,
            messages: list[Any],
        ) -> AgentReplied:
            response = await llm.ainvoke(
                [*messages, HumanMessage(content=event.question)]
            )
            return AgentReplied(message=response)

        @on(AgentReplied)
        async def second_reply(
            event: AgentReplied,
            messages: list[Any],
        ) -> FollowUpReply:
            response = await llm.ainvoke([*messages, HumanMessage(content="follow up")])
            return FollowUpReply(message=response)

        graph = EventGraph([first_reply, second_reply], reducers=[message_reducer()])
        adapter = AGUIAdapter(
            graph=graph,
            seed_factory=lambda inp: UserAsked(question="go"),
            include_reducers=True,
        )
        events = await _collect(adapter, _make_input())

        starts = [e for e in events if e.type == EventType.TEXT_MESSAGE_START]
        assert len(starts) == 2

        msg_snapshots = [e for e in events if e.type == EventType.MESSAGES_SNAPSHOT]
        last_snap = msg_snapshots[-1]
        ai_msgs = [m for m in last_snap.messages if m.role == "assistant"]
        assert len(ai_msgs) == 2
        for m in ai_msgs:
            assert m.id.startswith("lc_run--"), f"Expected LangChain ID, got {m.id}"
            assert not m.id.startswith("msg-")


def describe_non_streamed_id_reconciliation():
    """Non-streamed AI messages are delivered only via MESSAGES_SNAPSHOT."""

    def when_non_streamed_ai_message():
        async def it_delivers_via_snapshot_only():
            """No TextMessageStart for non-streamed AI; snapshot keeps LC ID."""
            ai = AIMessage(content="from history", id="lc-ai-1")

            @on(UserAsked)
            def emit(event: UserAsked) -> PhaseA:
                return PhaseA(messages=(ai,))

            graph = EventGraph([emit], reducers=[message_reducer()])
            adapter = AGUIAdapter(
                graph=graph,
                seed_factory=lambda inp: UserAsked(question="go"),
                include_reducers=True,
            )
            events = await _collect(adapter, _make_input())

            starts = [e for e in events if e.type == EventType.TEXT_MESSAGE_START]
            assert len(starts) == 0

            snapshots = [e for e in events if e.type == EventType.MESSAGES_SNAPSHOT]
            assert len(snapshots) >= 1
            last_snap = snapshots[-1]
            ai_msgs = [m for m in last_snap.messages if m.role == "assistant"]
            assert len(ai_msgs) == 1
            assert ai_msgs[0].id == "lc-ai-1"

    def when_multiple_non_streamed_ai_messages():
        async def it_delivers_all_via_snapshot_using_original_ids():
            ai1 = AIMessage(content="first", id="lc-ai-1")
            ai2 = AIMessage(content="second", id="lc-ai-2")

            @on(UserAsked)
            def step1(event: UserAsked) -> PhaseA:
                return PhaseA(messages=(ai1,))

            @on(PhaseA)
            def step2(event: PhaseA) -> PhaseB:
                return PhaseB(messages=(ai2,))

            graph = EventGraph([step1, step2], reducers=[message_reducer()])
            adapter = AGUIAdapter(
                graph=graph,
                seed_factory=lambda inp: UserAsked(question="go"),
                include_reducers=True,
            )
            events = await _collect(adapter, _make_input())

            starts = [e for e in events if e.type == EventType.TEXT_MESSAGE_START]
            assert len(starts) == 0

            snapshots = [e for e in events if e.type == EventType.MESSAGES_SNAPSHOT]
            last_snap = snapshots[-1]
            ai_msgs = [m for m in last_snap.messages if m.role == "assistant"]
            snapshot_ids = {m.id for m in ai_msgs}
            assert snapshot_ids == {"lc-ai-1", "lc-ai-2"}

    def when_ai_message_has_no_content_or_tool_calls():
        async def it_does_not_create_orphan_id_override():
            """Empty AI message must not register an override."""
            empty_ai = AIMessage(content="", id="lc-empty")

            @on(UserAsked)
            def emit(event: UserAsked) -> PhaseA:
                return PhaseA(messages=(empty_ai,))

            graph = EventGraph([emit], reducers=[message_reducer()])
            adapter = AGUIAdapter(
                graph=graph,
                seed_factory=lambda inp: UserAsked(question="go"),
                include_reducers=True,
            )
            events = await _collect(adapter, _make_input())

            # No TextMessageStart should have been emitted
            starts = [e for e in events if e.type == EventType.TEXT_MESSAGE_START]
            assert len(starts) == 0

            # Snapshot should keep the original LC ID, not rewrite to an orphan msg-*
            snapshots = [e for e in events if e.type == EventType.MESSAGES_SNAPSHOT]
            assert len(snapshots) >= 1
            last_snap = snapshots[-1]
            ai_msgs = [m for m in last_snap.messages if m.role == "assistant"]
            for m in ai_msgs:
                assert not m.id.startswith("msg-"), (
                    f"Orphan override: snapshot rewrote to {m.id} with no "
                    f"corresponding TextMessageStart"
                )


def describe_unclosed_stream_on_error():
    async def it_emits_text_message_end_before_run_error(monkeypatch):
        """TEXT_MESSAGE_END must be emitted for open streams before RUN_ERROR."""
        from langgraph_events._graph import LLMToken

        @on(UserAsked)
        def reply(event: UserAsked) -> AgentReplied:
            return AgentReplied(message=AIMessage(content="ok"))

        graph = EventGraph([reply])

        async def exploding_stream(
            seed,
            *,
            include_reducers,
            include_llm_tokens,
            include_custom_events,
            config,
        ):
            del seed, include_reducers, include_llm_tokens
            del include_custom_events, config
            yield LLMToken(run_id="llm-run-1", content="partial ")
            yield LLMToken(run_id="llm-run-1", content="content")
            raise RuntimeError("mid-stream failure")

        monkeypatch.setattr(graph, "astream_events", exploding_stream)

        adapter = AGUIAdapter(
            graph=graph,
            seed_factory=lambda inp: UserAsked(question="go"),
        )
        events = await _collect(adapter, _make_input())

        types = _types(events)
        assert EventType.RUN_STARTED in types
        assert EventType.TEXT_MESSAGE_START in types
        assert EventType.TEXT_MESSAGE_CONTENT in types
        assert EventType.RUN_ERROR in types

        # TEXT_MESSAGE_END must appear before RUN_ERROR
        assert EventType.TEXT_MESSAGE_END in types
        end_idx = types.index(EventType.TEXT_MESSAGE_END)
        error_idx = types.index(EventType.RUN_ERROR)
        assert end_idx < error_idx, (
            f"TEXT_MESSAGE_END (idx={end_idx}) must come before "
            f"RUN_ERROR (idx={error_idx})"
        )
