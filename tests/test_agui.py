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
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.checkpoint.memory import MemorySaver

from langgraph_events import (
    Event,
    EventGraph,
    IntegrationEvent,
    Interrupted,
    MessageEvent,
    message_reducer,
    on,
)
from langgraph_events.agui import (
    AGUIAdapter,
    FrontendToolCallRequested,
    InterruptedWithPayload,
    MapperContext,
    build_langchain_tools,
    detect_new_tool_results,
)
from langgraph_events.agui._mappers import _warned_classes

# ---------------------------------------------------------------------------
# Test event classes
# ---------------------------------------------------------------------------


class UserAsked(IntegrationEvent):
    question: str = ""

    def agui_dict(self) -> dict[str, Any]:
        return {"question": self.question}


class AgentReplied(IntegrationEvent, MessageEvent):
    message: AIMessage = None  # type: ignore[assignment]


class AgentCalledTools(IntegrationEvent, MessageEvent):
    message: AIMessage = None  # type: ignore[assignment]


class ToolsExecuted(IntegrationEvent, MessageEvent):
    messages: tuple[ToolMessage, ...] = ()


class AgentAndToolMessages(IntegrationEvent, MessageEvent):
    messages: tuple[Any, ...] = ()


class UserSent(IntegrationEvent, MessageEvent):
    message: HumanMessage = None  # type: ignore[assignment]

    def agui_dict(self) -> dict[str, Any]:
        return {"content": self.message.content if self.message else ""}


class FollowUpReply(IntegrationEvent, MessageEvent):
    message: AIMessage = None  # type: ignore[assignment]


class TaskCreated(IntegrationEvent):
    title: str = ""

    def agui_dict(self) -> dict[str, Any]:
        return {"title": self.title}


class ApprovalRequested(Interrupted):
    draft: str = ""

    def agui_dict(self) -> dict[str, Any]:
        return {"draft": self.draft}


class _ReviewPayload(dict):
    pass


class ReviewWithPayload(InterruptedWithPayload[_ReviewPayload]):
    """Payload-typed interrupt without an explicit agui_dict() override."""

    draft: str = ""

    def interrupt_payload(self) -> _ReviewPayload:
        return _ReviewPayload(kind="review", draft=self.draft)


class ApprovalGiven(IntegrationEvent):
    approved: bool = True

    def agui_dict(self) -> dict[str, Any]:
        return {"approved": self.approved}


class PhaseA(IntegrationEvent, MessageEvent):
    messages: tuple[Any, ...] = ()


class PhaseB(IntegrationEvent, MessageEvent):
    messages: tuple[Any, ...] = ()


class SystemPromptDelivered(IntegrationEvent, MessageEvent):
    message: SystemMessage = None  # type: ignore[assignment]


class ErrorTrigger(IntegrationEvent):
    def agui_dict(self) -> dict[str, Any]:
        return {}


# Module-level event classes used as handler return annotations in the
# test bodies below. Defining them inline would break get_type_hints()
# resolution (see CLAUDE.md forward-reference convention).


class PlainEvent(IntegrationEvent):
    value: str = "no-dict"


class NoDict1(IntegrationEvent):
    x: int = 0


class NoDict2(IntegrationEvent):
    x: int = 0


class DraftUserSent(IntegrationEvent, MessageEvent):
    message: HumanMessage = None  # type: ignore[assignment]


class AgentDrafted(IntegrationEvent, MessageEvent):
    message: AIMessage = None  # type: ignore[assignment]


class AgentRevised(IntegrationEvent, MessageEvent):
    message: AIMessage = None  # type: ignore[assignment]


class MultimodalReply(IntegrationEvent, MessageEvent):
    message: AIMessage = None  # type: ignore[assignment]


class NamedEvent(IntegrationEvent):
    data: str = ""

    @property
    def agui_event_name(self) -> str:
        return "custom.named"

    def agui_dict(self) -> dict[str, Any]:
        return {"data": self.data}


class NameOnlyEvent(IntegrationEvent):
    data: str = ""

    @property
    def agui_event_name(self) -> str:
        return "custom.name"


class StepA(IntegrationEvent):
    value: str = ""

    def agui_dict(self) -> dict[str, Any]:
        return {"value": self.value}


class StepB(IntegrationEvent):
    value: str = ""

    def agui_dict(self) -> dict[str, Any]:
        return {"value": self.value}


class FocusLogged(IntegrationEvent):
    value: str = ""

    def agui_dict(self) -> dict[str, Any]:
        return {"value": self.value}


class FocusEcho(IntegrationEvent):
    """Module-level for handler-annotation type resolution (per CLAUDE.md).

    Used by the C1 regression test that primes an accumulator reducer
    before a resume.
    """

    value: str = ""


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


def describe_default_state_projection():
    """Direct coverage of the framework-strip layer.

    Module-internal but load-bearing: every snapshot (outbound + inbound)
    flows through it.  A regression here would silently leak framework
    channels (``events``, ``_cursor``, …) or dedicated AG-UI keys
    (``messages``) to clients — so we test it directly, independent of
    the AGUIAdapter integration that also exercises it.
    """

    def when_input_has_framework_keys():
        def it_strips_them():
            from langgraph_events.agui._state import _default_state_projection

            result = _default_state_projection(
                {
                    "events": ["audit-log"],
                    "_cursor": 7,
                    "_pending": [],
                    "_round": 3,
                    "focus": "scene-1",
                }
            )
            assert result == {"focus": "scene-1"}

    def when_input_has_messages_key():
        def it_strips_the_dedicated_channel():
            from langgraph_events.agui._state import _default_state_projection

            result = _default_state_projection({"messages": ["m1"], "focus": "scene-2"})
            assert result == {"focus": "scene-2"}

    def when_input_has_only_user_keys():
        def it_passes_them_through_unchanged():
            from langgraph_events.agui._state import _default_state_projection

            payload = {"focus": "scene-3", "scene": "@scene-3", "count": 7}
            assert _default_state_projection(payload) == payload

    def when_input_is_empty():
        def it_returns_empty():
            from langgraph_events.agui._state import _default_state_projection

            assert _default_state_projection({}) == {}


def describe_AGUIAdapter():
    def describe_stream():
        @pytest.fixture
        def simple_graph():
            @on(UserAsked)
            def reply(event: UserAsked) -> AgentReplied:
                return AgentReplied(
                    message=AIMessage(content=f"Answer to: {event.question}")
                )

            return EventGraph([reply], reducers=[message_reducer()])

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

            async def it_delivers_ai_messages_via_snapshot(simple_graph):
                adapter = AGUIAdapter(
                    graph=simple_graph,
                    seed_factory=lambda inp: UserAsked(question="hello"),
                )
                events = await _collect(adapter, _make_input())

                msg_snapshots = [
                    e for e in events if e.type == EventType.MESSAGES_SNAPSHOT
                ]
                assert len(msg_snapshots) >= 1
                last_snap = msg_snapshots[-1]
                ai_msgs = [m for m in last_snap.messages if m.role == "assistant"]
                assert len(ai_msgs) >= 1
                assert "Answer to: hello" in ai_msgs[-1].content

            async def it_delivers_tool_calls_via_snapshot():
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

                graph = EventGraph([call_tool], reducers=[message_reducer()])
                adapter = AGUIAdapter(
                    graph=graph,
                    seed_factory=lambda inp: UserAsked(question="search"),
                )
                events = await _collect(adapter, _make_input())

                msg_snapshots = [
                    e for e in events if e.type == EventType.MESSAGES_SNAPSHOT
                ]
                assert len(msg_snapshots) >= 1
                last_snap = msg_snapshots[-1]
                ai_msgs = [m for m in last_snap.messages if m.role == "assistant"]
                assert len(ai_msgs) >= 1
                assert len(ai_msgs[-1].tool_calls) == 1
                assert ai_msgs[-1].tool_calls[0].id == "tc-1"
                assert ai_msgs[-1].tool_calls[0].function.name == "search"

            async def it_delivers_tool_results_via_snapshot():
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

                graph = EventGraph([run_tools], reducers=[message_reducer()])
                adapter = AGUIAdapter(
                    graph=graph,
                    seed_factory=lambda inp: UserAsked(question="run"),
                )
                events = await _collect(adapter, _make_input())

                msg_snapshots = [
                    e for e in events if e.type == EventType.MESSAGES_SNAPSHOT
                ]
                assert len(msg_snapshots) >= 1
                last_snap = msg_snapshots[-1]
                tool_msgs = [m for m in last_snap.messages if m.role == "tool"]
                assert len(tool_msgs) >= 1
                assert tool_msgs[0].content == "result-1"

            async def it_delivers_mixed_messages_via_snapshot():
                @on(UserAsked)
                def reply_tool_result(event: UserAsked) -> AgentAndToolMessages:
                    return AgentAndToolMessages(
                        messages=(
                            AIMessage(content="I used a tool"),
                            ToolMessage(content="tool output", tool_call_id="tc-mixed"),
                        )
                    )

                graph = EventGraph([reply_tool_result], reducers=[message_reducer()])
                adapter = AGUIAdapter(
                    graph=graph,
                    seed_factory=lambda inp: UserAsked(question="mixed"),
                )
                events = await _collect(adapter, _make_input())

                msg_snapshots = [
                    e for e in events if e.type == EventType.MESSAGES_SNAPSHOT
                ]
                assert len(msg_snapshots) >= 1
                last_snap = msg_snapshots[-1]
                ai_msgs = [m for m in last_snap.messages if m.role == "assistant"]
                tool_msgs = [m for m in last_snap.messages if m.role == "tool"]
                assert any("I used a tool" in m.content for m in ai_msgs)
                assert len(tool_msgs) >= 1
                assert tool_msgs[0].content == "tool output"

            async def it_maps_interrupted_to_custom_event():
                @on(UserAsked)
                def ask_approval(event: UserAsked) -> ApprovalRequested:
                    return ApprovalRequested(draft="draft text")

                graph = EventGraph(
                    [ask_approval],
                    checkpointer=MemorySaver(),
                    reducers=[message_reducer()],
                )
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
                    reducers=[message_reducer()],
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

                graph = EventGraph([create_task], reducers=[message_reducer()])
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

                graph = EventGraph([blow_up], reducers=[message_reducer()])
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
                @on(UserAsked)
                def emit_plain(event: UserAsked) -> PlainEvent:
                    return PlainEvent(value="hello")

                graph = EventGraph([emit_plain], reducers=[message_reducer()])
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

                graph = EventGraph([create_task], reducers=[message_reducer()])
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
                @on(UserAsked)
                def step1(event: UserAsked) -> NoDict1:
                    return NoDict1(x=1)

                @on(NoDict1)
                def step2(event: NoDict1) -> NoDict2:
                    return NoDict2(x=2)

                graph = EventGraph([step1, step2], reducers=[message_reducer()])
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

        def when_error_message_provided():
            async def it_uses_custom_error_message():
                @on(ErrorTrigger)
                def blow_up(event: ErrorTrigger) -> None:
                    raise RuntimeError("boom")

                graph = EventGraph([blow_up], reducers=[message_reducer()])
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
                )
                events = await _collect(adapter, _make_input())

                state_count = sum(
                    1 for e in events if e.type == EventType.STATE_SNAPSHOT
                )
                msg_count = sum(
                    1 for e in events if e.type == EventType.MESSAGES_SNAPSHOT
                )
                # With StreamFrame.changed_reducers, state snapshots are only
                # emitted initially (or when non-dedicated reducers change).
                assert state_count == 1
                assert msg_count == 1

            async def it_detects_message_content_changes():
                """MessagesSnapshot emits when add_messages replaces in-place."""

                @on(DraftUserSent)
                def draft(event: DraftUserSent) -> AgentDrafted:
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
                    seed_factory=lambda inp: DraftUserSent(
                        message=HumanMessage(content="go")
                    ),
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

            graph = EventGraph([create_task], reducers=[message_reducer()])
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

            graph = EventGraph([create_task], reducers=[message_reducer()])
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

                graph = EventGraph(
                    [ask, approve],
                    checkpointer=MemorySaver(),
                    reducers=[message_reducer()],
                )

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

                # Should contain message from the approve handler
                msg_snapshots = [
                    e for e in events if e.type == EventType.MESSAGES_SNAPSHOT
                ]
                assert len(msg_snapshots) >= 1
                last_snap = msg_snapshots[-1]
                ai_msgs = [m for m in last_snap.messages if m.role == "assistant"]
                assert any("Done!" in m.content for m in ai_msgs)

            async def it_streams_resume_events():
                @on(UserAsked)
                def ask(event: UserAsked) -> ApprovalRequested:
                    return ApprovalRequested(draft="check this")

                @on(ApprovalGiven)
                def approve(event: ApprovalGiven) -> TaskCreated:
                    return TaskCreated(title="approved task")

                graph = EventGraph(
                    [ask, approve],
                    checkpointer=MemorySaver(),
                    reducers=[message_reducer()],
                )

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

                graph = EventGraph(
                    [ask, approve],
                    checkpointer=MemorySaver(),
                    reducers=[message_reducer()],
                )

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
                    reducers=[message_reducer()],
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

                    graph = EventGraph(
                        [reply],
                        checkpointer=MemorySaver(),
                        reducers=[message_reducer()],
                    )
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

                graph = EventGraph(
                    [reply], reducers=[message_reducer()]
                )  # no checkpointer
                with pytest.raises(
                    ValueError, match="resume_factory requires a checkpointer"
                ):
                    AGUIAdapter(
                        graph=graph,
                        seed_factory=lambda inp: UserAsked(question="hi"),
                        resume_factory=lambda inp: ApprovalGiven(approved=True),
                    )

        def when_graph_has_no_message_reducer():
            async def it_raises():
                @on(UserAsked)
                def reply(event: UserAsked) -> AgentReplied:
                    return AgentReplied(message=AIMessage(content="hi"))

                graph = EventGraph([reply])
                with pytest.raises(ValueError, match="message_reducer"):
                    AGUIAdapter(
                        graph=graph,
                        seed_factory=lambda inp: UserAsked(question="hi"),
                    )

        def when_include_reducers_false():
            @pytest.mark.asyncio
            async def it_auto_includes_messages():
                """include_reducers=False is overridden to include messages."""

                @on(UserAsked)
                def reply(event: UserAsked) -> AgentReplied:
                    return AgentReplied(message=AIMessage(content="auto-included"))

                graph = EventGraph([reply], reducers=[message_reducer()])
                adapter = AGUIAdapter(
                    graph=graph,
                    seed_factory=lambda inp: UserAsked(question="hi"),
                    include_reducers=False,
                )
                events = await _collect(adapter, _make_input())
                snapshots = [e for e in events if e.type == EventType.MESSAGES_SNAPSHOT]
                assert len(snapshots) >= 1

        def when_include_reducers_list_excludes_messages():
            @pytest.mark.asyncio
            async def it_auto_adds_messages():
                """include_reducers=['other'] auto-adds 'messages'."""

                @on(UserAsked)
                def reply(event: UserAsked) -> AgentReplied:
                    return AgentReplied(message=AIMessage(content="list-fix"))

                graph = EventGraph([reply], reducers=[message_reducer()])
                adapter = AGUIAdapter(
                    graph=graph,
                    seed_factory=lambda inp: UserAsked(question="hi"),
                    include_reducers=["nonexistent"],
                )
                events = await _collect(adapter, _make_input())
                snapshots = [e for e in events if e.type == EventType.MESSAGES_SNAPSHOT]
                assert len(snapshots) >= 1

        def when_include_reducers_is_malformed():
            def it_raises_typeerror_at_construction():
                """Garbage values (int, dict, callable, etc.) fail loudly at
                init, not silently as empty snapshots at runtime."""

                @on(UserAsked)
                def reply(event: UserAsked) -> AgentReplied:
                    return AgentReplied(message=AIMessage(content="ok"))

                graph = EventGraph([reply], reducers=[message_reducer()])
                with pytest.raises(TypeError, match="include_reducers"):
                    AGUIAdapter(
                        graph=graph,
                        seed_factory=lambda inp: UserAsked(question="hi"),
                        include_reducers=42,  # type: ignore[arg-type]
                    )

            def it_rejects_a_callable():
                """Bare callables aren't a supported include_reducers shape —
                use the list[str] allow-list form."""

                @on(UserAsked)
                def reply(event: UserAsked) -> AgentReplied:
                    return AgentReplied(message=AIMessage(content="ok"))

                graph = EventGraph([reply], reducers=[message_reducer()])
                with pytest.raises(TypeError, match="include_reducers"):
                    AGUIAdapter(
                        graph=graph,
                        seed_factory=lambda inp: UserAsked(question="hi"),
                        include_reducers=lambda r: r,  # type: ignore[arg-type]
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

            graph = EventGraph([reply], reducers=[message_reducer()])
            adapter = AGUIAdapter(
                graph=graph,
                seed_factory=tracking_factory,
            )
            input_data = _make_input(thread_id="t-seed")
            events = await _collect(adapter, input_data)

            assert len(received_inputs) == 1
            assert received_inputs[0].thread_id == "t-seed"

            msg_snapshots = [e for e in events if e.type == EventType.MESSAGES_SNAPSHOT]
            assert len(msg_snapshots) >= 1
            last_snap = msg_snapshots[-1]
            ai_msgs = [m for m in last_snap.messages if m.role == "assistant"]
            assert any("from factory" in m.content for m in ai_msgs)


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

        graph = EventGraph(
            [ask_approval],
            checkpointer=MemorySaver(),
            reducers=[message_reducer()],
        )
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

            graph = EventGraph([reply], reducers=[message_reducer()])
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

    def when_internal_audit_log_present():
        async def it_strips_events_audit_log_from_state_snapshot():
            """`events` is graph-internal and must not leak into client state.

            The EventGraph auto-injects an `events` reducer for the cumulative
            audit log.  AG-UI clients echo any `state.*` key back via
            `RunAgentInput.state` on every Send — round-tripping the entire
            audit log every run is O(history) wire bloat and the log itself is
            never a client concern.
            """
            from langgraph_events import SKIP, ScalarReducer

            focus = ScalarReducer(
                name="focus",
                event_type=UserAsked,
                fn=lambda e: e.question or SKIP,
            )

            @on(UserAsked)
            def reply(event: UserAsked) -> AgentReplied:
                return AgentReplied(message=AIMessage(content="hi"))

            graph = EventGraph(
                [reply],
                checkpointer=MemorySaver(),
                reducers=[message_reducer(), focus],
            )
            config = {"configurable": {"thread_id": "t-connect-no-events-leak"}}
            await graph.ainvoke(UserAsked(question="scene-7"), config=config)

            # Sanity: the checkpoint really does carry a non-empty audit log.
            checkpoint_events = graph.get_state(config).events
            assert len(checkpoint_events) >= 1

            adapter = AGUIAdapter(
                graph=graph,
                seed_factory=lambda inp: UserAsked(question="unused"),
            )
            events = [
                event
                async for event in adapter.connect(
                    _make_input(thread_id="t-connect-no-events-leak")
                )
            ]

            state_snapshots = [e for e in events if e.type == EventType.STATE_SNAPSHOT]
            assert len(state_snapshots) == 1
            snapshot = state_snapshots[0].snapshot
            # The internal audit log must NOT leak to the client.
            assert "events" not in snapshot
            # Dedicated channels remain stripped.
            assert "messages" not in snapshot
            # User-defined reducers must STILL be present (no over-filtering).
            assert snapshot.get("focus") == "scene-7"


def describe_config_passthrough():
    def _setup_stream_capture(monkeypatch):
        @on(UserAsked)
        def reply(event: UserAsked) -> AgentReplied:
            return AgentReplied(message=AIMessage(content="ok"))

        graph = EventGraph([reply], reducers=[message_reducer()])
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
        from langgraph_events.stream import StateSnapshotFrame

        @on(UserAsked)
        def reply(event: UserAsked) -> AgentReplied:
            return AgentReplied(message=AIMessage(content="ok"))

        graph = EventGraph([reply], reducers=[message_reducer()])
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
        from langgraph_events.stream import CustomEventFrame

        @on(UserAsked)
        def reply(event: UserAsked) -> AgentReplied:
            return AgentReplied(message=AIMessage(content="ok"))

        graph = EventGraph([reply], reducers=[message_reducer()])

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
        from langgraph_events.stream import CustomEventFrame

        @on(UserAsked)
        def reply(event: UserAsked) -> AgentReplied:
            return AgentReplied(message=AIMessage(content="ok"))

        graph = EventGraph([reply], reducers=[message_reducer()])

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

        graph = EventGraph(
            [ask], checkpointer=MemorySaver(), reducers=[message_reducer()]
        )
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

            graph = EventGraph(
                [reply], checkpointer=MemorySaver(), reducers=[message_reducer()]
            )
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

            graph = EventGraph([reply], reducers=[message_reducer()])
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

            graph = EventGraph([reply], reducers=[message_reducer()])  # no checkpointer
            adapter = AGUIAdapter(
                graph=graph,
                seed_factory=stateful_factory,
            )
            await _collect(adapter, _make_input())

            assert len(received_states) == 1
            assert received_states[0] is None


def describe_interrupt_replay():
    def when_interrupted():
        async def it_replays_interrupt_on_reconnect():
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

    def when_not_interrupted():
        async def it_executes_normally():
            @on(UserAsked)
            def reply(event: UserAsked) -> AgentReplied:
                return AgentReplied(message=AIMessage(content="done"))

            graph = EventGraph(
                [reply],
                checkpointer=MemorySaver(),
                reducers=[message_reducer()],
            )
            adapter = AGUIAdapter(
                graph=graph,
                seed_factory=lambda inp: UserAsked(question="go"),
            )
            events = await _collect(adapter, _make_input(thread_id="t-gate-clean"))

            assert events[0].type == EventType.RUN_STARTED
            assert events[-1].type == EventType.RUN_FINISHED
            msg_snapshots = [e for e in events if e.type == EventType.MESSAGES_SNAPSHOT]
            assert len(msg_snapshots) >= 1
            last_snap = msg_snapshots[-1]
            ai_msgs = [m for m in last_snap.messages if m.role == "assistant"]
            assert any("done" in m.content for m in ai_msgs)


def describe_AGUICustomEvent():
    def when_agui_event_name_implemented():
        async def it_uses_agui_event_name():
            @on(UserAsked)
            def emit(event: UserAsked) -> NamedEvent:
                return NamedEvent(data="hello")

            graph = EventGraph([emit], reducers=[message_reducer()])
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

            graph = EventGraph([emit], reducers=[message_reducer()])
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


def describe_connect_frontend_tool_interrupt():
    async def it_replays_tool_call_triple():
        @on(AskConfirm)
        def request(event: AskConfirm) -> FrontendToolCallRequested:
            return FrontendToolCallRequested(
                name="confirm",
                args={"prompt": event.prompt},
                tool_call_id="tc-connect-1",
            )

        graph = EventGraph(
            [request],
            reducers=[message_reducer()],
            checkpointer=MemorySaver(),
        )
        await graph.ainvoke(
            AskConfirm(prompt="Ship v1?"),
            config={"configurable": {"thread_id": "t-connect-fe"}},
        )

        adapter = AGUIAdapter(
            graph=graph,
            seed_factory=lambda inp: AskConfirm(prompt="unused"),
        )
        events = [
            e async for e in adapter.connect(_make_input(thread_id="t-connect-fe"))
        ]

        triple = [
            e.type
            for e in events
            if e.type
            in (
                EventType.TOOL_CALL_START,
                EventType.TOOL_CALL_ARGS,
                EventType.TOOL_CALL_END,
            )
        ]
        assert triple == [
            EventType.TOOL_CALL_START,
            EventType.TOOL_CALL_ARGS,
            EventType.TOOL_CALL_END,
        ]
        start = next(e for e in events if e.type == EventType.TOOL_CALL_START)
        args_ev = next(e for e in events if e.type == EventType.TOOL_CALL_ARGS)
        assert start.tool_call_id == "tc-connect-1"
        assert start.tool_call_name == "confirm"
        assert json.loads(args_ev.delta) == {"prompt": "Ship v1?"}

        # No generic "interrupted" CustomEvent — the frontend-tool mapper preempted it
        assert not any(
            e.type == EventType.CUSTOM and e.name == "interrupted" for e in events
        )


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
            @on(UserAsked)
            def emit(event: UserAsked) -> NameOnlyEvent:
                return NameOnlyEvent(data="test")

            graph = EventGraph([emit], reducers=[message_reducer()])
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

        graph = EventGraph([reply], reducers=[message_reducer()])
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
        msg_snapshots = [e for e in events if e.type == EventType.MESSAGES_SNAPSHOT]
        assert len(msg_snapshots) >= 1
        last_snap = msg_snapshots[-1]
        all_content = " ".join(
            m.content for m in last_snap.messages if m.role == "assistant"
        )
        assert "first" in all_content
        assert "second" in all_content


def describe_llm_streaming_single_start():
    async def it_emits_text_message_start_exactly_once():
        """LLM streaming emits exactly one TEXT_MESSAGE_START per call."""
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
        )
        events = await _collect(adapter, _make_input())

        starts = [e for e in events if e.type == EventType.TEXT_MESSAGE_START]
        # Exactly one start — from LLM streaming
        assert len(starts) == 1


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
    """Non-streamed AI messages appear only in MESSAGES_SNAPSHOT."""

    def when_non_streamed_ai_message():
        async def it_delivers_via_snapshot_only():
            """Snapshot delivers non-streamed AI with original LC ID."""
            ai = AIMessage(content="from history", id="lc-ai-1")

            @on(UserAsked)
            def emit(event: UserAsked) -> PhaseA:
                return PhaseA(messages=(ai,))

            graph = EventGraph([emit], reducers=[message_reducer()])
            adapter = AGUIAdapter(
                graph=graph,
                seed_factory=lambda inp: UserAsked(question="go"),
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
            """Empty AI message appears in snapshot with original LC ID."""
            empty_ai = AIMessage(content="", id="lc-empty")

            @on(UserAsked)
            def emit(event: UserAsked) -> PhaseA:
                return PhaseA(messages=(empty_ai,))

            graph = EventGraph([emit], reducers=[message_reducer()])
            adapter = AGUIAdapter(
                graph=graph,
                seed_factory=lambda inp: UserAsked(question="go"),
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
        from langgraph_events.stream import LLMToken

        @on(UserAsked)
        def reply(event: UserAsked) -> AgentReplied:
            return AgentReplied(message=AIMessage(content="ok"))

        graph = EventGraph([reply], reducers=[message_reducer()])

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


def describe_changed_reducers_none_fallback():
    """Adapter handles StreamFrame.changed_reducers=None (v1/sync path)."""

    async def it_emits_state_and_messages_on_first_frame(monkeypatch):
        """When changed_reducers is None, first frame emits both snapshots."""
        from langgraph_events.stream import StreamFrame

        @on(UserAsked)
        def reply(event: UserAsked) -> AgentReplied:
            return AgentReplied(message=AIMessage(content="ok"))

        graph = EventGraph([reply], reducers=[message_reducer()])

        async def fake_astream_events(
            seed,
            *,
            include_reducers,
            include_llm_tokens,
            include_custom_events,
            config,
        ):
            del seed, include_reducers, include_llm_tokens
            del include_custom_events, config
            # Simulate v1 path: changed_reducers=None
            yield StreamFrame(
                event=UserAsked(question="go"),
                reducers={"messages": [HumanMessage(content="go")]},
                changed_reducers=None,
            )
            yield StreamFrame(
                event=AgentReplied(message=AIMessage(content="ok")),
                reducers={
                    "messages": [
                        HumanMessage(content="go"),
                        AIMessage(content="ok"),
                    ]
                },
                changed_reducers=None,
            )

        monkeypatch.setattr(graph, "astream_events", fake_astream_events)

        adapter = AGUIAdapter(
            graph=graph,
            seed_factory=lambda inp: UserAsked(question="go"),
        )
        events = await _collect(adapter, _make_input())

        state_snapshots = [e for e in events if e.type == EventType.STATE_SNAPSHOT]
        msg_snapshots = [e for e in events if e.type == EventType.MESSAGES_SNAPSHOT]

        # First frame emits state snapshot; second does not (since
        # changed_reducers=None falls through to "already emitted" check)
        assert len(state_snapshots) == 1
        # Messages are present in reducers but changed_reducers is None,
        # so MessagesSnapshot is NOT emitted (requires explicit "messages"
        # in changed_reducers).
        assert len(msg_snapshots) == 0


def describe_system_message_conversion():
    """SystemMessage in message_reducer is converted to AG-UI SystemMessage."""

    async def it_includes_system_message_in_snapshot():
        @on(UserAsked)
        def set_system(event: UserAsked) -> SystemPromptDelivered:
            return SystemPromptDelivered(
                message=SystemMessage(content="You are a helpful assistant")
            )

        graph = EventGraph([set_system], reducers=[message_reducer()])
        adapter = AGUIAdapter(
            graph=graph,
            seed_factory=lambda inp: UserAsked(question="go"),
        )
        events = await _collect(adapter, _make_input())

        msg_snapshots = [e for e in events if e.type == EventType.MESSAGES_SNAPSHOT]
        assert len(msg_snapshots) >= 1
        last_snap = msg_snapshots[-1]
        sys_msgs = [m for m in last_snap.messages if m.role == "system"]
        assert len(sys_msgs) >= 1
        assert sys_msgs[0].content == "You are a helpful assistant"


def describe_multiple_custom_reducers():
    """StateSnapshot tracks changes across multiple non-message reducers."""

    def when_custom_reducers_change():
        async def it_emits_state_snapshot_for_each_change():
            from langgraph_events._reducer import ScalarReducer

            reducer_a = ScalarReducer(
                name="counter_a",
                event_type=StepA,
                fn=lambda e: e.value,
            )
            reducer_b = ScalarReducer(
                name="counter_b",
                event_type=StepB,
                fn=lambda e: e.value,
            )

            @on(StepA)
            def handle_a(event: StepA) -> StepB:
                return StepB(value="from_a")

            graph = EventGraph(
                [handle_a],
                reducers=[message_reducer(), reducer_a, reducer_b],
            )
            adapter = AGUIAdapter(
                graph=graph,
                seed_factory=lambda inp: StepA(value="first"),
            )
            events = await _collect(adapter, _make_input())

            state_snapshots = [e for e in events if e.type == EventType.STATE_SNAPSHOT]
            msg_snapshots = [e for e in events if e.type == EventType.MESSAGES_SNAPSHOT]

            # StateSnapshot should appear: initial + when custom reducers change
            assert len(state_snapshots) >= 2
            # Messages reducer never changes (no MessageEvent), so no msg snapshot
            assert len(msg_snapshots) == 0
            # State should contain custom reducer keys, not "messages"
            for snap in state_snapshots:
                assert "messages" not in snap.snapshot
            # Final state snapshot should contain both reducer values
            last_state = state_snapshots[-1].snapshot
            assert "counter_a" in last_state
            assert "counter_b" in last_state
            assert last_state["counter_a"] == "first"
            assert last_state["counter_b"] == "from_a"


# ---------------------------------------------------------------------------
# Tool-call streaming — LLM-initiated outbound path
# ---------------------------------------------------------------------------


def _tool_call_chunk_event(
    run_id: str,
    *chunks: dict[str, Any],
    content: str = "",
) -> dict[str, Any]:
    from langchain_core.messages import AIMessageChunk

    return {
        "event": "on_chat_model_stream",
        "run_id": run_id,
        "data": {
            "chunk": AIMessageChunk(
                content=content,
                tool_call_chunks=[{**c, "type": "tool_call_chunk"} for c in chunks],
            ),
        },
    }


def describe_tool_call_streaming():

    def when_single_tool_call():
        async def it_emits_start_args_end_in_order(monkeypatch):
            @on(UserAsked)
            def ask(event: UserAsked) -> AgentReplied:
                return AgentReplied(message=AIMessage(content="ok"))

            graph = EventGraph([ask], reducers=[message_reducer()])

            async def fake_astream(*args, **kwargs):
                del args, kwargs
                yield _tool_call_chunk_event(
                    "run-a",
                    {"name": "search", "args": "", "id": "tc-1", "index": 0},
                )
                yield _tool_call_chunk_event(
                    "run-a",
                    {"name": "", "args": '{"q":"hi"}', "id": "", "index": 0},
                )
                yield {
                    "event": "on_chat_model_end",
                    "run_id": "run-a",
                    "data": {"output": AIMessage(id="msg-x", content="ok")},
                }

            monkeypatch.setattr(graph.compiled, "astream_events", fake_astream)

            adapter = AGUIAdapter(
                graph=graph,
                seed_factory=lambda inp: UserAsked(question="q"),
            )
            events = await _collect(adapter, _make_input())

            tc_types = [
                e.type
                for e in events
                if e.type
                in (
                    EventType.TOOL_CALL_START,
                    EventType.TOOL_CALL_ARGS,
                    EventType.TOOL_CALL_END,
                )
            ]
            assert tc_types == [
                EventType.TOOL_CALL_START,
                EventType.TOOL_CALL_ARGS,
                EventType.TOOL_CALL_END,
            ]
            start = next(e for e in events if e.type == EventType.TOOL_CALL_START)
            args_ev = next(e for e in events if e.type == EventType.TOOL_CALL_ARGS)
            end = next(e for e in events if e.type == EventType.TOOL_CALL_END)
            assert start.tool_call_id == "tc-1"
            assert start.tool_call_name == "search"
            assert args_ev.tool_call_id == "tc-1"
            assert args_ev.delta == '{"q":"hi"}'
            assert end.tool_call_id == "tc-1"

    def when_parallel_tool_calls():
        async def it_gives_each_index_its_own_triple(monkeypatch):
            @on(UserAsked)
            def ask(event: UserAsked) -> AgentReplied:
                return AgentReplied(message=AIMessage(content="ok"))

            graph = EventGraph([ask], reducers=[message_reducer()])

            async def fake_astream(*args, **kwargs):
                del args, kwargs
                yield _tool_call_chunk_event(
                    "run-p",
                    {"name": "search", "args": "", "id": "tc-1", "index": 0},
                    {"name": "lookup", "args": "", "id": "tc-2", "index": 1},
                )
                yield _tool_call_chunk_event(
                    "run-p",
                    {"name": "", "args": '{"q":"a"}', "id": "", "index": 0},
                    {"name": "", "args": '{"k":"b"}', "id": "", "index": 1},
                )
                yield {
                    "event": "on_chat_model_end",
                    "run_id": "run-p",
                    "data": {"output": AIMessage(id="msg-y", content="ok")},
                }

            monkeypatch.setattr(graph.compiled, "astream_events", fake_astream)

            adapter = AGUIAdapter(
                graph=graph,
                seed_factory=lambda inp: UserAsked(question="q"),
            )
            events = await _collect(adapter, _make_input())

            starts = [e for e in events if e.type == EventType.TOOL_CALL_START]
            args_evs = [e for e in events if e.type == EventType.TOOL_CALL_ARGS]
            ends = [e for e in events if e.type == EventType.TOOL_CALL_END]
            assert {s.tool_call_id for s in starts} == {"tc-1", "tc-2"}
            assert len(args_evs) == 2
            assert {e.tool_call_id for e in ends} == {"tc-1", "tc-2"}

    def when_chunk_carries_text_and_tool_call():
        async def it_emits_text_message_and_uses_its_id_as_parent(monkeypatch):
            @on(UserAsked)
            def ask(event: UserAsked) -> AgentReplied:
                return AgentReplied(message=AIMessage(content="ok"))

            graph = EventGraph([ask], reducers=[message_reducer()])

            async def fake_astream(*args, **kwargs):
                del args, kwargs
                yield _tool_call_chunk_event(
                    "run-m",
                    {"name": "search", "args": "", "id": "tc-9", "index": 0},
                    content="thinking…",
                )
                yield {
                    "event": "on_chat_model_end",
                    "run_id": "run-m",
                    "data": {"output": AIMessage(id="msg-z", content="thinking…")},
                }

            monkeypatch.setattr(graph.compiled, "astream_events", fake_astream)

            adapter = AGUIAdapter(
                graph=graph,
                seed_factory=lambda inp: UserAsked(question="q"),
            )
            events = await _collect(adapter, _make_input())

            text_start = next(
                e for e in events if e.type == EventType.TEXT_MESSAGE_START
            )
            tc_start = next(e for e in events if e.type == EventType.TOOL_CALL_START)
            assert tc_start.parent_message_id == text_start.message_id

    def when_no_text_message_in_stream():
        async def it_emits_none_parent_message_id(monkeypatch):
            @on(UserAsked)
            def ask(event: UserAsked) -> AgentReplied:
                return AgentReplied(message=AIMessage(content="ok"))

            graph = EventGraph([ask], reducers=[message_reducer()])

            async def fake_astream(*args, **kwargs):
                del args, kwargs
                yield _tool_call_chunk_event(
                    "run-n",
                    {"name": "search", "args": "", "id": "tc-n", "index": 0},
                )
                yield {
                    "event": "on_chat_model_end",
                    "run_id": "run-n",
                    "data": {"output": AIMessage(id="msg-n", content="")},
                }

            monkeypatch.setattr(graph.compiled, "astream_events", fake_astream)

            adapter = AGUIAdapter(
                graph=graph,
                seed_factory=lambda inp: UserAsked(question="q"),
            )
            events = await _collect(adapter, _make_input())

            tc_start = next(e for e in events if e.type == EventType.TOOL_CALL_START)
            assert tc_start.parent_message_id is None

    def when_first_chunk_has_no_id():
        async def it_emits_run_error(monkeypatch):
            @on(UserAsked)
            def ask(event: UserAsked) -> AgentReplied:
                return AgentReplied(message=AIMessage(content="ok"))

            graph = EventGraph([ask], reducers=[message_reducer()])

            async def fake_astream(*args, **kwargs):
                del args, kwargs
                yield _tool_call_chunk_event(
                    "run-no-id",
                    {"name": "search", "args": "", "id": "", "index": 0},
                )

            monkeypatch.setattr(graph.compiled, "astream_events", fake_astream)

            adapter = AGUIAdapter(
                graph=graph,
                seed_factory=lambda inp: UserAsked(question="q"),
            )
            events = await _collect(adapter, _make_input())

            run_err = next((e for e in events if e.type == EventType.RUN_ERROR), None)
            assert run_err is not None
            assert "no 'id'" in run_err.message
            # No TOOL_CALL_START should have been emitted
            assert not any(e.type == EventType.TOOL_CALL_START for e in events)

    def when_first_chunk_has_no_name():
        async def it_emits_run_error(monkeypatch):
            @on(UserAsked)
            def ask(event: UserAsked) -> AgentReplied:
                return AgentReplied(message=AIMessage(content="ok"))

            graph = EventGraph([ask], reducers=[message_reducer()])

            async def fake_astream(*args, **kwargs):
                del args, kwargs
                yield _tool_call_chunk_event(
                    "run-no-name",
                    {"name": "", "args": "", "id": "tc-x", "index": 0},
                )

            monkeypatch.setattr(graph.compiled, "astream_events", fake_astream)

            adapter = AGUIAdapter(
                graph=graph,
                seed_factory=lambda inp: UserAsked(question="q"),
            )
            events = await _collect(adapter, _make_input())

            run_err = next((e for e in events if e.type == EventType.RUN_ERROR), None)
            assert run_err is not None
            assert "no 'name'" in run_err.message
            assert not any(e.type == EventType.TOOL_CALL_START for e in events)

    def when_stream_errors_mid_call():
        async def it_drains_tool_call_end_before_run_error(monkeypatch):
            @on(UserAsked)
            def ask(event: UserAsked) -> AgentReplied:
                return AgentReplied(message=AIMessage(content="ok"))

            graph = EventGraph([ask], reducers=[message_reducer()])

            async def fake_astream(*args, **kwargs):
                del args, kwargs
                yield _tool_call_chunk_event(
                    "run-e",
                    {"name": "search", "args": "", "id": "tc-err", "index": 0},
                )
                raise RuntimeError("LLM blew up")

            monkeypatch.setattr(graph.compiled, "astream_events", fake_astream)

            adapter = AGUIAdapter(
                graph=graph,
                seed_factory=lambda inp: UserAsked(question="q"),
            )
            events = await _collect(adapter, _make_input())

            tc_end = next(
                (e for e in events if e.type == EventType.TOOL_CALL_END), None
            )
            run_err = next((e for e in events if e.type == EventType.RUN_ERROR), None)
            assert tc_end is not None
            assert tc_end.tool_call_id == "tc-err"
            assert run_err is not None
            # End before RunError
            assert events.index(tc_end) < events.index(run_err)


# ---------------------------------------------------------------------------
# FrontendToolCallRequested — handler-initiated outbound path
# ---------------------------------------------------------------------------


class AskConfirm(IntegrationEvent):
    prompt: str = ""


class AskShip(IntegrationEvent):
    release: str = ""


class Shipped(IntegrationEvent):
    release: str = ""
    approved: bool = False

    def agui_dict(self) -> dict[str, Any]:
        return {"release": self.release, "approved": self.approved}


def describe_FrontendToolCallRequested_mapping():

    def when_handler_returns_event():
        async def it_emits_tool_call_triple_and_no_custom_interrupted():
            @on(AskConfirm)
            def request(event: AskConfirm) -> FrontendToolCallRequested:
                return FrontendToolCallRequested(
                    name="confirm",
                    args={"prompt": event.prompt},
                    tool_call_id="tc-c1",
                )

            graph = EventGraph(
                [request],
                reducers=[message_reducer()],
                checkpointer=MemorySaver(),
            )
            adapter = AGUIAdapter(
                graph=graph,
                seed_factory=lambda inp: AskConfirm(prompt="Ship?"),
            )
            events = await _collect(adapter, _make_input(thread_id="t-fe-1"))

            triple = [
                e.type
                for e in events
                if e.type
                in (
                    EventType.TOOL_CALL_START,
                    EventType.TOOL_CALL_ARGS,
                    EventType.TOOL_CALL_END,
                )
            ]
            assert triple == [
                EventType.TOOL_CALL_START,
                EventType.TOOL_CALL_ARGS,
                EventType.TOOL_CALL_END,
            ]

            start = next(e for e in events if e.type == EventType.TOOL_CALL_START)
            args_ev = next(e for e in events if e.type == EventType.TOOL_CALL_ARGS)
            assert start.tool_call_name == "confirm"
            assert start.tool_call_id == "tc-c1"
            assert json.loads(args_ev.delta) == {"prompt": "Ship?"}

            # No generic CustomEvent("interrupted", …) — the mapper preempted it
            custom_names = [e.name for e in events if e.type == EventType.CUSTOM]
            assert "interrupted" not in custom_names

    def when_args_empty():
        async def it_serializes_empty_object():
            @on(AskConfirm)
            def request(event: AskConfirm) -> FrontendToolCallRequested:
                return FrontendToolCallRequested(name="ping", tool_call_id="tc-ping")

            graph = EventGraph(
                [request],
                reducers=[message_reducer()],
                checkpointer=MemorySaver(),
            )
            adapter = AGUIAdapter(
                graph=graph,
                seed_factory=lambda inp: AskConfirm(prompt=""),
            )
            events = await _collect(adapter, _make_input(thread_id="t-fe-2"))
            args_ev = next(e for e in events if e.type == EventType.TOOL_CALL_ARGS)
            assert args_ev.delta == "{}"

    def when_tool_result_received():
        async def it_continues_graph_after_resume():
            from ag_ui.core import ToolMessage as AguiToolMessage

            @on(AskShip)
            def request(event: AskShip) -> FrontendToolCallRequested:
                return FrontendToolCallRequested(
                    name="confirm",
                    args={"release": event.release},
                    tool_call_id="tc-ship-1",
                )

            @on(ToolsExecuted)
            def ship(event: ToolsExecuted) -> Shipped:
                approved = False
                for m in event.messages:
                    approved = bool(json.loads(m.content or "{}").get("approved"))
                return Shipped(release="v1", approved=approved)

            graph = EventGraph(
                [request, ship],
                reducers=[message_reducer()],
                checkpointer=MemorySaver(),
            )

            def resume_factory(input_data, checkpoint_state=None):
                results = detect_new_tool_results(input_data, checkpoint_state)
                return ToolsExecuted(messages=tuple(results)) if results else None

            adapter = AGUIAdapter(
                graph=graph,
                seed_factory=lambda inp: AskShip(release="v1"),
                resume_factory=resume_factory,
            )

            # Step 1 — initial run pauses on FrontendToolCallRequested
            first = await _collect(adapter, _make_input(thread_id="t-roundtrip"))
            triple = [
                e.type
                for e in first
                if e.type
                in (
                    EventType.TOOL_CALL_START,
                    EventType.TOOL_CALL_ARGS,
                    EventType.TOOL_CALL_END,
                )
            ]
            assert triple == [
                EventType.TOOL_CALL_START,
                EventType.TOOL_CALL_ARGS,
                EventType.TOOL_CALL_END,
            ]
            # ship handler has not run yet
            assert not any(
                e.type == EventType.CUSTOM and e.name == "Shipped" for e in first
            )

            # Step 2 — frontend echoes tool message back; graph resumes, ship runs
            second_input = _make_input(
                thread_id="t-roundtrip",
                messages=[
                    AguiToolMessage(
                        id="fe-reply-1",
                        role="tool",
                        content='{"approved": true}',
                        tool_call_id="tc-ship-1",
                    ),
                ],
            )
            second = await _collect(adapter, second_input)

            shipped = [
                e for e in second if e.type == EventType.CUSTOM and e.name == "Shipped"
            ]
            assert len(shipped) == 1
            assert shipped[0].value == {"release": "v1", "approved": True}
            assert second[-1].type == EventType.RUN_FINISHED

    def when_args_not_json_serializable():
        async def it_surfaces_typeerror_as_run_error():
            from datetime import UTC, datetime

            @on(AskConfirm)
            def request(event: AskConfirm) -> FrontendToolCallRequested:
                return FrontendToolCallRequested(
                    name="confirm",
                    args={"when": datetime(2026, 4, 19, tzinfo=UTC)},
                    tool_call_id="tc-bad-args",
                )

            graph = EventGraph(
                [request],
                reducers=[message_reducer()],
                checkpointer=MemorySaver(),
            )
            adapter = AGUIAdapter(
                graph=graph,
                seed_factory=lambda inp: AskConfirm(prompt="x"),
            )
            events = await _collect(adapter, _make_input(thread_id="t-fe-bad-args"))

            run_err = next((e for e in events if e.type == EventType.RUN_ERROR), None)
            assert run_err is not None
            assert not any(e.type == EventType.TOOL_CALL_ARGS for e in events)


# Regression coverage: the existing `describe_AGUIAdapter.describe_stream
# .it_delivers_tool_calls_via_snapshot` test (above) validates that the
# MessagesSnapshot path still carries AIMessage.tool_calls unchanged when a
# handler emits an AIMessage with tool_calls. Streaming ToolCall* events and
# the post-stream MessagesSnapshot coexist: the frontend reconciles by
# tool_call_id (CopilotKit's useFrontendTool is idempotent), so the SDK does
# not deduplicate between the two paths.


# ---------------------------------------------------------------------------
# Tool-def bridging — build_langchain_tools
# ---------------------------------------------------------------------------


def describe_build_langchain_tools():

    def when_tools_present():
        def it_builds_openai_function_shape():
            from ag_ui.core import Tool

            tools = [
                Tool(
                    name="confirm",
                    description="Ask the user to confirm",
                    parameters={"type": "object", "properties": {}},
                ),
            ]
            out = build_langchain_tools(tools)
            assert out == [
                {
                    "type": "function",
                    "function": {
                        "name": "confirm",
                        "description": "Ask the user to confirm",
                        "parameters": {"type": "object", "properties": {}},
                    },
                }
            ]

    def when_empty_or_none():
        def it_returns_empty_list_for_empty():
            assert build_langchain_tools([]) == []

        def it_returns_empty_list_for_none():
            assert build_langchain_tools(None) == []


# ---------------------------------------------------------------------------
# Inbound routing — detect_new_tool_results
# ---------------------------------------------------------------------------


def describe_detect_new_tool_results():

    def when_fresh_run():
        def it_returns_empty():
            from ag_ui.core import ToolMessage as AguiToolMessage

            inp = _make_input(
                messages=[
                    AguiToolMessage(
                        id="m-1",
                        role="tool",
                        content="42",
                        tool_call_id="tc-1",
                    ),
                ],
            )
            assert detect_new_tool_results(inp, None) == []

    def when_all_tool_ids_known():
        def it_returns_empty():
            from ag_ui.core import ToolMessage as AguiToolMessage

            inp = _make_input(
                messages=[
                    AguiToolMessage(
                        id="m-1",
                        role="tool",
                        content="42",
                        tool_call_id="tc-1",
                    ),
                ],
            )
            checkpoint = {
                "messages": [
                    ToolMessage(content="42", tool_call_id="tc-1"),
                ],
            }
            assert detect_new_tool_results(inp, checkpoint) == []

    def when_some_new():
        def it_returns_only_unknown_ids():
            from ag_ui.core import ToolMessage as AguiToolMessage

            inp = _make_input(
                messages=[
                    AguiToolMessage(
                        id="m-1",
                        role="tool",
                        content="old",
                        tool_call_id="tc-old",
                    ),
                    AguiToolMessage(
                        id="m-2",
                        role="tool",
                        content="new",
                        tool_call_id="tc-new",
                    ),
                ],
            )
            checkpoint = {
                "messages": [ToolMessage(content="old", tool_call_id="tc-old")],
            }
            out = detect_new_tool_results(inp, checkpoint)
            assert len(out) == 1
            assert out[0].tool_call_id == "tc-new"
            assert out[0].content == "new"

    def when_checkpoint_messages_none():
        def it_treats_as_no_history():
            from ag_ui.core import ToolMessage as AguiToolMessage

            inp = _make_input(
                messages=[
                    AguiToolMessage(
                        id="m-1",
                        role="tool",
                        content="hello",
                        tool_call_id="tc-1",
                    ),
                ],
            )
            checkpoint = {"messages": None}
            out = detect_new_tool_results(inp, checkpoint)
            assert len(out) == 1
            assert out[0].tool_call_id == "tc-1"

    def when_tool_message_lacks_id():
        def it_raises():
            from types import SimpleNamespace

            # ag_ui's ToolMessage validates tool_call_id at construction,
            # so simulate a non-conformant inbound message via a duck type.
            bad_msg = SimpleNamespace(
                id="m-1", role="tool", content="42", tool_call_id=""
            )
            inp = _make_input(messages=[])
            inp.messages = [bad_msg]  # type: ignore[assignment]

            with pytest.raises(ValueError, match=r"tool_call_id"):
                detect_new_tool_results(inp, {"messages": []})

    def when_tool_message_id_field_absent():
        def it_raises():
            from types import SimpleNamespace

            bad_msg = SimpleNamespace(id="m-1", role="tool", content="42")
            inp = _make_input(messages=[])
            inp.messages = [bad_msg]  # type: ignore[assignment]

            with pytest.raises(ValueError, match=r"tool_call_id"):
                detect_new_tool_results(inp, {"messages": []})


def describe_frontend_state_mutated_event_class():
    """Shape checks for the new FrontendStateMutated event."""

    def when_instantiated():
        def it_is_an_integration_event():
            from langgraph_events.agui import FrontendStateMutated

            event = FrontendStateMutated(state={"focus": "scene-1"})
            assert isinstance(event, IntegrationEvent)
            assert event.state == {"focus": "scene-1"}

        def it_defaults_to_empty_state():
            from langgraph_events.agui import FrontendStateMutated

            event = FrontendStateMutated()
            assert event.state == {}


def describe_frontend_state_mutated():
    """AG-UI adapter emits FrontendStateMutated before the seed event."""

    def when_state_has_non_dedicated_keys():
        async def it_applies_scalar_reducer_before_seed_handler_runs():
            from langgraph_events import SKIP, ScalarReducer
            from langgraph_events.agui import FrontendStateMutated

            focus = ScalarReducer(
                name="focus",
                event_type=FrontendStateMutated,
                fn=lambda e: e.state.get("focus", SKIP),
            )
            seen: list[str | None] = []

            @on(UserAsked)
            def reply(event: UserAsked, focus: str | None = None) -> AgentReplied:
                seen.append(focus)
                return AgentReplied(message=AIMessage(content="ok"))

            graph = EventGraph(
                [reply],
                reducers=[message_reducer(), focus],
                checkpointer=MemorySaver(),
            )
            adapter = AGUIAdapter(
                graph=graph,
                seed_factory=lambda inp: UserAsked(question="go"),
            )
            await _collect(adapter, _make_input(state={"focus": "scene-42"}))

            assert seen == ["scene-42"]

        async def it_records_frontend_state_mutated_in_event_log():
            from langgraph_events.agui import FrontendStateMutated

            @on(UserAsked)
            def reply(event: UserAsked) -> AgentReplied:
                return AgentReplied(message=AIMessage(content="ok"))

            graph = EventGraph(
                [reply],
                reducers=[message_reducer()],
                checkpointer=MemorySaver(),
            )
            adapter = AGUIAdapter(
                graph=graph,
                seed_factory=lambda inp: UserAsked(question="go"),
            )
            config = {"configurable": {"thread_id": "thread-fsm-1"}}
            await _collect(
                adapter,
                _make_input(thread_id="thread-fsm-1", state={"focus": "scene-42"}),
            )

            log = graph.get_state(config).events
            fsm_events = [e for e in log if isinstance(e, FrontendStateMutated)]
            assert len(fsm_events) == 1
            assert fsm_events[0].state == {"focus": "scene-42"}

            # Ordering: FrontendStateMutated must precede UserAsked in the log.
            indices = {type(e).__name__: i for i, e in enumerate(log)}
            assert indices["FrontendStateMutated"] < indices["UserAsked"]

    def when_state_is_empty():
        async def it_does_not_emit_frontend_state_mutated():
            from langgraph_events.agui import FrontendStateMutated

            @on(UserAsked)
            def reply(event: UserAsked) -> AgentReplied:
                return AgentReplied(message=AIMessage(content="ok"))

            graph = EventGraph(
                [reply],
                reducers=[message_reducer()],
                checkpointer=MemorySaver(),
            )
            adapter = AGUIAdapter(
                graph=graph,
                seed_factory=lambda inp: UserAsked(question="go"),
            )
            config = {"configurable": {"thread_id": "thread-fsm-empty"}}
            await _collect(
                adapter,
                _make_input(thread_id="thread-fsm-empty", state={}),
            )

            log = graph.get_state(config).events
            assert not any(isinstance(e, FrontendStateMutated) for e in log)

    def when_state_is_none():
        async def it_does_not_emit_frontend_state_mutated():
            from langgraph_events.agui import FrontendStateMutated

            @on(UserAsked)
            def reply(event: UserAsked) -> AgentReplied:
                return AgentReplied(message=AIMessage(content="ok"))

            graph = EventGraph(
                [reply],
                reducers=[message_reducer()],
                checkpointer=MemorySaver(),
            )
            adapter = AGUIAdapter(
                graph=graph,
                seed_factory=lambda inp: UserAsked(question="go"),
            )
            config = {"configurable": {"thread_id": "thread-fsm-none"}}
            await _collect(
                adapter,
                _make_input(thread_id="thread-fsm-none", state=None),
            )

            log = graph.get_state(config).events
            assert not any(isinstance(e, FrontendStateMutated) for e in log)

    def when_state_contains_only_dedicated_keys():
        async def it_does_not_emit_frontend_state_mutated():
            from langgraph_events.agui import FrontendStateMutated

            @on(UserAsked)
            def reply(event: UserAsked) -> AgentReplied:
                return AgentReplied(message=AIMessage(content="ok"))

            graph = EventGraph(
                [reply],
                reducers=[message_reducer()],
                checkpointer=MemorySaver(),
            )
            adapter = AGUIAdapter(
                graph=graph,
                seed_factory=lambda inp: UserAsked(question="go"),
            )
            config = {"configurable": {"thread_id": "thread-fsm-msgs-only"}}
            await _collect(
                adapter,
                _make_input(
                    thread_id="thread-fsm-msgs-only",
                    state={"messages": [{"role": "user", "content": "forged"}]},
                ),
            )

            log = graph.get_state(config).events
            assert not any(isinstance(e, FrontendStateMutated) for e in log)

    def when_state_has_dedicated_plus_other_keys():
        async def it_strips_dedicated_keys_from_the_emitted_event():
            from langgraph_events.agui import FrontendStateMutated

            @on(UserAsked)
            def reply(event: UserAsked) -> AgentReplied:
                return AgentReplied(message=AIMessage(content="ok"))

            graph = EventGraph(
                [reply],
                reducers=[message_reducer()],
                checkpointer=MemorySaver(),
            )
            adapter = AGUIAdapter(
                graph=graph,
                seed_factory=lambda inp: UserAsked(question="go"),
            )
            config = {"configurable": {"thread_id": "thread-fsm-mixed"}}
            await _collect(
                adapter,
                _make_input(
                    thread_id="thread-fsm-mixed",
                    state={
                        "messages": [{"role": "user", "content": "forged"}],
                        "focus": "scene-7",
                    },
                ),
            )

            log = graph.get_state(config).events
            fsm_events = [e for e in log if isinstance(e, FrontendStateMutated)]
            assert len(fsm_events) == 1
            assert fsm_events[0].state == {"focus": "scene-7"}
            assert "messages" not in fsm_events[0].state

    def when_state_echoes_internal_events():
        def when_state_also_has_user_keys():
            async def it_strips_internal_keys_from_the_emitted_event():
                """Defense-in-depth: a stale client echoing `state.events` must
                never inject the EventGraph audit log back into the graph.
                """
                from langgraph_events.agui import FrontendStateMutated

                @on(UserAsked)
                def reply(event: UserAsked) -> AgentReplied:
                    return AgentReplied(message=AIMessage(content="ok"))

                graph = EventGraph(
                    [reply],
                    reducers=[message_reducer()],
                    checkpointer=MemorySaver(),
                )
                adapter = AGUIAdapter(
                    graph=graph,
                    seed_factory=lambda inp: UserAsked(question="go"),
                )
                config = {"configurable": {"thread_id": "thread-fsm-events-echo"}}
                await _collect(
                    adapter,
                    _make_input(
                        thread_id="thread-fsm-events-echo",
                        state={
                            "events": [
                                {"type": "stale", "payload": "from-client"},
                            ],
                            "user_key": "v",
                        },
                    ),
                )

                log = graph.get_state(config).events
                fsm_events = [e for e in log if isinstance(e, FrontendStateMutated)]
                assert len(fsm_events) == 1
                assert fsm_events[0].state == {"user_key": "v"}
                assert "events" not in fsm_events[0].state

        def when_state_only_has_internal_keys():
            async def it_drops_the_event_entirely():
                """If the only key is the internal `events`, no FSM event fires."""
                from langgraph_events.agui import FrontendStateMutated

                @on(UserAsked)
                def reply(event: UserAsked) -> AgentReplied:
                    return AgentReplied(message=AIMessage(content="ok"))

                graph = EventGraph(
                    [reply],
                    reducers=[message_reducer()],
                    checkpointer=MemorySaver(),
                )
                adapter = AGUIAdapter(
                    graph=graph,
                    seed_factory=lambda inp: UserAsked(question="go"),
                )
                config = {"configurable": {"thread_id": "thread-fsm-events-only"}}
                await _collect(
                    adapter,
                    _make_input(
                        thread_id="thread-fsm-events-only",
                        state={"events": [{"type": "stale"}]},
                    ),
                )

                log = graph.get_state(config).events
                assert not any(isinstance(e, FrontendStateMutated) for e in log)

    def when_no_checkpointer():
        async def it_still_applies_state_within_the_run():
            from langgraph_events import SKIP, ScalarReducer
            from langgraph_events.agui import FrontendStateMutated

            focus = ScalarReducer(
                name="focus",
                event_type=FrontendStateMutated,
                fn=lambda e: e.state.get("focus", SKIP),
            )
            seen: list[str | None] = []

            @on(UserAsked)
            def reply(event: UserAsked, focus: str | None = None) -> AgentReplied:
                seen.append(focus)
                return AgentReplied(message=AIMessage(content="ok"))

            graph = EventGraph(
                [reply],
                reducers=[message_reducer(), focus],
                # No checkpointer — used to be a hard skip under apre_seed.
            )
            adapter = AGUIAdapter(
                graph=graph,
                seed_factory=lambda inp: UserAsked(question="go"),
            )
            await _collect(adapter, _make_input(state={"focus": "no-ckpt"}))

            assert seen == ["no-ckpt"]

    def when_reducer_fn_transforms_the_value():
        """The reducer's ``fn`` runs on both the non-resume and resume paths.

        On resume the adapter computes per-reducer contributions from the
        ``FrontendStateMutated`` event and writes them to channels via
        ``apre_seed`` *before* the resume's domain dispatch — preserving
        ``fn`` semantics (transformations, ``SKIP``) symmetrically with the
        non-resume path.
        """

        async def it_applies_the_transformation_on_non_resume():
            from langgraph_events import SKIP, ScalarReducer
            from langgraph_events.agui import FrontendStateMutated

            focus = ScalarReducer(
                name="focus",
                event_type=FrontendStateMutated,
                fn=lambda e: e.state["focus"].upper() if "focus" in e.state else SKIP,
            )
            seen: list[str | None] = []

            @on(UserAsked)
            def reply(event: UserAsked, focus: str | None = None) -> AgentReplied:
                seen.append(focus)
                return AgentReplied(message=AIMessage(content="ok"))

            graph = EventGraph(
                [reply],
                reducers=[message_reducer(), focus],
                checkpointer=MemorySaver(),
            )
            adapter = AGUIAdapter(
                graph=graph,
                seed_factory=lambda inp: UserAsked(question="go"),
            )
            await _collect(adapter, _make_input(state={"focus": "scene-lc"}))

            assert seen == ["SCENE-LC"]

        async def it_applies_the_transformation_on_resume():
            from langgraph_events import SKIP, ScalarReducer
            from langgraph_events.agui import FrontendStateMutated

            focus = ScalarReducer(
                name="focus",
                event_type=FrontendStateMutated,
                fn=lambda e: e.state["focus"].upper() if "focus" in e.state else SKIP,
            )
            seen: list[str | None] = []

            @on(UserAsked)
            def ask(event: UserAsked) -> ApprovalRequested:
                return ApprovalRequested(draft="ship it?")

            @on(ApprovalGiven)
            def finish(
                event: ApprovalGiven,
                focus: str | None = None,
            ) -> AgentReplied:
                seen.append(focus)
                return AgentReplied(message=AIMessage(content="ok"))

            graph = EventGraph(
                [ask, finish],
                reducers=[message_reducer(), focus],
                checkpointer=MemorySaver(),
            )

            thread_id = "thread-fsm-fn-resume"
            config = {"configurable": {"thread_id": thread_id}}

            await graph.ainvoke(UserAsked(question="approve?"), config=config)

            adapter = AGUIAdapter(
                graph=graph,
                seed_factory=lambda inp: UserAsked(question="unused"),
                resume_factory=lambda inp: ApprovalGiven(approved=True),
            )
            await _collect(
                adapter,
                _make_input(
                    thread_id=thread_id,
                    state={"focus": "scene-lc"},
                ),
            )

            # Transformed value — fn's `.upper()` ran on resume too.
            assert seen == ["SCENE-LC"]

    def when_handler_subscribes_to_frontend_state_mutated():
        async def it_fires_and_its_output_event_appears_in_the_log():
            from langgraph_events.agui import FrontendStateMutated

            @on(FrontendStateMutated)
            def log_focus(event: FrontendStateMutated) -> FocusLogged | None:
                focus = event.state.get("focus")
                if focus is None:
                    return None
                return FocusLogged(value=focus)

            @on(UserAsked)
            def reply(event: UserAsked) -> AgentReplied:
                return AgentReplied(message=AIMessage(content="ok"))

            graph = EventGraph(
                [log_focus, reply],
                reducers=[message_reducer()],
                checkpointer=MemorySaver(),
            )
            adapter = AGUIAdapter(
                graph=graph,
                seed_factory=lambda inp: UserAsked(question="go"),
            )
            config = {"configurable": {"thread_id": "thread-fsm-handler"}}
            await _collect(
                adapter,
                _make_input(
                    thread_id="thread-fsm-handler",
                    state={"focus": "handler-case"},
                ),
            )

            log = graph.get_state(config).events
            focus_events = [e for e in log if isinstance(e, FocusLogged)]
            assert len(focus_events) == 1
            assert focus_events[0].value == "handler-case"

    def when_resuming():
        def with_state_change_from_client():
            async def it_applies_state_before_resume_event_handler_runs():
                from langgraph_events import SKIP, ScalarReducer
                from langgraph_events.agui import FrontendStateMutated

                focus = ScalarReducer(
                    name="focus",
                    event_type=FrontendStateMutated,
                    fn=lambda e: e.state.get("focus", SKIP),
                )
                seen: list[str | None] = []

                @on(UserAsked)
                def ask(event: UserAsked) -> ApprovalRequested:
                    return ApprovalRequested(draft="ship it?")

                @on(ApprovalGiven)
                def finish(
                    event: ApprovalGiven,
                    focus: str | None = None,
                ) -> AgentReplied:
                    seen.append(focus)
                    return AgentReplied(message=AIMessage(content="ok"))

                graph = EventGraph(
                    [ask, finish],
                    reducers=[message_reducer(), focus],
                    checkpointer=MemorySaver(),
                )

                thread_id = "thread-fsm-resume"
                config = {"configurable": {"thread_id": thread_id}}

                # Drive the graph to an interrupt (bypassing the adapter —
                # mirrors the priming pattern in
                # `it_suppresses_resumed_events`).
                await graph.ainvoke(UserAsked(question="approve?"), config=config)

                # Now resume via the adapter, shipping client state along
                # with the resume input.  The checkpoint's pending
                # interrupt causes the adapter to take the resume branch.
                adapter = AGUIAdapter(
                    graph=graph,
                    seed_factory=lambda inp: UserAsked(question="unused"),
                    resume_factory=lambda inp: ApprovalGiven(approved=True),
                )
                await _collect(
                    adapter,
                    _make_input(
                        thread_id=thread_id,
                        state={"focus": "resume-case"},
                    ),
                )

                assert seen == ["resume-case"]

        def with_backend_authoritative_channel():
            async def it_lets_the_resume_domain_event_win():
                """Cadance reproduction (`d1b7d7cf-…`, `560203cc-…`).

                A channel driven by a backend domain event must not be
                overwritten by a stale frontend snapshot key on resume.
                The reducer subscribes to the backend event only — FSM
                dispatch is a no-op for that channel, so the resume's
                domain dispatch wins.
                """
                from langgraph_events import ScalarReducer

                # Backend-authoritative: channel only updates from the
                # domain event, NOT from FrontendStateMutated.
                strategy = ScalarReducer(
                    name="walkthrough_strategy",
                    event_type=ApprovalGiven,
                    fn=lambda e: "guided" if e.approved else "skipped",
                )
                # Backend reads strategy after resume.
                seen_strategy: list[str | None] = []

                @on(UserAsked)
                def ask(event: UserAsked) -> ApprovalRequested:
                    return ApprovalRequested(draft="proceed?")

                @on(ApprovalGiven)
                def finish(
                    event: ApprovalGiven,
                    walkthrough_strategy: str | None = None,
                ) -> AgentReplied:
                    seen_strategy.append(walkthrough_strategy)
                    return AgentReplied(message=AIMessage(content="ok"))

                graph = EventGraph(
                    [ask, finish],
                    reducers=[message_reducer(), strategy],
                    checkpointer=MemorySaver(),
                )

                thread_id = "thread-backend-authoritative"
                config = {"configurable": {"thread_id": thread_id}}
                await graph.ainvoke(UserAsked(question="?"), config=config)

                adapter = AGUIAdapter(
                    graph=graph,
                    seed_factory=lambda inp: UserAsked(question="unused"),
                    resume_factory=lambda inp: ApprovalGiven(approved=True),
                )
                # Stale frontend snapshot tries to inject a NULL strategy.
                # Under the old apre_seed bypass this would clobber the
                # channel; under FSM dispatch the reducer doesn't subscribe
                # to FSM, so it's a no-op and the domain event wins.
                await _collect(
                    adapter,
                    _make_input(
                        thread_id=thread_id,
                        state={"walkthrough_strategy": None},
                    ),
                )

                assert seen_strategy == ["guided"]

        def with_accumulator_reducer_subscribed_to_fsm():
            async def it_does_not_double_count_the_contribution():
                """Regression for C1 (PR #56 final review): an
                ``operator.add``-style accumulator reducer wired to
                ``FrontendStateMutated`` must not have its FSM contribution
                applied twice on the resume path — once via the adapter's
                ``apre_seed`` (which writes the channel), then a second time
                via ``_astream_v2``'s seed-merge loop on top of the
                already-written checkpoint state.

                Symptom (pre-fix): the streamed STATE_SNAPSHOT shows the
                FSM contribution doubled (``["X", "X"]``) while the
                persisted checkpoint correctly shows ``["X"]``.
                """
                import operator

                from langgraph_events import Reducer
                from langgraph_events.agui import FrontendStateMutated

                # Append-style reducer: each FSM event contributes
                # [state["focus"]], merged via operator.add on the channel.
                focus_log = Reducer(
                    name="focus_log",
                    event_type=FrontendStateMutated,
                    fn=lambda e: [e.state["focus"]] if "focus" in e.state else [],
                    reducer=operator.add,
                )

                @on(UserAsked)
                def ask(event: UserAsked) -> ApprovalRequested:
                    return ApprovalRequested(draft="ok?")

                @on(ApprovalGiven)
                def finish(event: ApprovalGiven) -> AgentReplied:
                    return AgentReplied(message=AIMessage(content="done"))

                graph = EventGraph(
                    [ask, finish],
                    reducers=[message_reducer(), focus_log],
                    checkpointer=MemorySaver(),
                )

                thread_id = "thread-fsm-no-double-count"
                config = {"configurable": {"thread_id": thread_id}}
                await graph.ainvoke(UserAsked(question="?"), config=config)

                adapter = AGUIAdapter(
                    graph=graph,
                    seed_factory=lambda inp: UserAsked(question="unused"),
                    resume_factory=lambda inp: ApprovalGiven(approved=True),
                )
                events = await _collect(
                    adapter,
                    _make_input(
                        thread_id=thread_id,
                        state={"focus": "X"},
                    ),
                )

                # Persisted checkpoint: single contribution (correct).
                checkpoint_values = (await graph.compiled.aget_state(config)).values
                assert checkpoint_values.get("focus_log") == ["X"]

                # Streamed STATE_SNAPSHOT: must also be single contribution.
                snapshots = [e for e in events if e.type == EventType.STATE_SNAPSHOT]
                assert snapshots, "expected at least one STATE_SNAPSHOT"
                last = snapshots[-1].snapshot
                assert last.get("focus_log") == ["X"], (
                    f"expected ['X'], got {last.get('focus_log')!r} — FSM "
                    "contribution was double-counted in the streamed snapshot"
                )

            async def it_layers_one_new_entry_onto_existing_accumulator_state():
                """Stronger C1 regression: when the channel already holds
                accumulated values from a prior run, the resume's FSM
                contribution must add exactly one new entry, not two."""
                import operator

                from langgraph_events import Reducer
                from langgraph_events.agui import FrontendStateMutated

                # Reducer accumulates from BOTH a domain event (priming) and
                # FSM (resume payload), so we can prime with real prior
                # state then layer the FSM contribution on top.  FocusEcho
                # is defined at module level so handler annotations resolve.
                focus_log = Reducer(
                    name="focus_log",
                    event_type=(FocusEcho, FrontendStateMutated),
                    fn=lambda e: (
                        [e.value]
                        if isinstance(e, FocusEcho)
                        else ([e.state["focus"]] if "focus" in e.state else [])
                    ),
                    reducer=operator.add,
                )

                @on(FocusEcho)
                def prime(event: FocusEcho) -> ApprovalRequested:
                    return ApprovalRequested(draft="ok?")

                @on(ApprovalGiven)
                def finish(event: ApprovalGiven) -> AgentReplied:
                    return AgentReplied(message=AIMessage(content="done"))

                graph = EventGraph(
                    [prime, finish],
                    reducers=[message_reducer(), focus_log],
                    checkpointer=MemorySaver(),
                )

                thread_id = "thread-fsm-layered"
                config = {"configurable": {"thread_id": thread_id}}
                # Prime the channel: two prior FocusEcho events accumulate
                # ["a", "b"] before the interrupt.
                await graph.ainvoke(
                    [FocusEcho(value="a"), FocusEcho(value="b")], config=config
                )
                checkpoint_before = (await graph.compiled.aget_state(config)).values
                assert checkpoint_before.get("focus_log") == ["a", "b"]

                adapter = AGUIAdapter(
                    graph=graph,
                    seed_factory=lambda inp: FocusEcho(value="unused"),
                    resume_factory=lambda inp: ApprovalGiven(approved=True),
                )
                events = await _collect(
                    adapter,
                    _make_input(
                        thread_id=thread_id,
                        state={"focus": "X"},
                    ),
                )

                # Persisted checkpoint: exactly one new entry layered on top.
                checkpoint_after = (await graph.compiled.aget_state(config)).values
                assert checkpoint_after.get("focus_log") == ["a", "b", "X"]

                # Streamed STATE_SNAPSHOT: same — no double-count regression.
                snapshots = [e for e in events if e.type == EventType.STATE_SNAPSHOT]
                assert snapshots, "expected at least one STATE_SNAPSHOT"
                last = snapshots[-1].snapshot
                assert last.get("focus_log") == ["a", "b", "X"], (
                    f"expected ['a', 'b', 'X'], got {last.get('focus_log')!r}"
                    " — FSM contribution was double-counted on top of "
                    "pre-existing accumulator state"
                )

        def with_state_yielded_in_output_stream():
            async def it_emits_a_frontend_state_mutated_event_on_resume():
                """FSM appears in the audit log on the resume path,
                exactly like the non-resume path."""
                from langgraph_events.agui import FrontendStateMutated

                @on(UserAsked)
                def ask(event: UserAsked) -> ApprovalRequested:
                    return ApprovalRequested(draft="ok?")

                @on(ApprovalGiven)
                def finish(event: ApprovalGiven) -> AgentReplied:
                    return AgentReplied(message=AIMessage(content="done"))

                graph = EventGraph(
                    [ask, finish],
                    reducers=[message_reducer()],
                    checkpointer=MemorySaver(),
                )

                thread_id = "thread-fsm-resume-audit"
                config = {"configurable": {"thread_id": thread_id}}
                await graph.ainvoke(UserAsked(question="?"), config=config)

                adapter = AGUIAdapter(
                    graph=graph,
                    seed_factory=lambda inp: UserAsked(question="unused"),
                    resume_factory=lambda inp: ApprovalGiven(approved=True),
                )
                await _collect(
                    adapter,
                    _make_input(
                        thread_id=thread_id,
                        state={"focus": "resume-audit"},
                    ),
                )

                log = graph.get_state(config).events
                fsm_events = [e for e in log if isinstance(e, FrontendStateMutated)]
                assert len(fsm_events) == 1
                assert fsm_events[0].state == {"focus": "resume-audit"}

        def with_dispatch_ordering():
            async def it_records_fsm_before_the_resume_domain_event():
                """Plan acceptance criterion 4(a): FSM lands in the audit log
                before the resume's domain event, so reducers/observers see
                client-state contributions first and the resume handler's
                output reduces on top."""
                from langgraph_events.agui import FrontendStateMutated

                @on(UserAsked)
                def ask(event: UserAsked) -> ApprovalRequested:
                    return ApprovalRequested(draft="ok?")

                @on(ApprovalGiven)
                def finish(event: ApprovalGiven) -> AgentReplied:
                    return AgentReplied(message=AIMessage(content="done"))

                graph = EventGraph(
                    [ask, finish],
                    reducers=[message_reducer()],
                    checkpointer=MemorySaver(),
                )

                thread_id = "thread-fsm-resume-order"
                config = {"configurable": {"thread_id": thread_id}}
                await graph.ainvoke(UserAsked(question="?"), config=config)

                adapter = AGUIAdapter(
                    graph=graph,
                    seed_factory=lambda inp: UserAsked(question="unused"),
                    resume_factory=lambda inp: ApprovalGiven(approved=True),
                )
                await _collect(
                    adapter,
                    _make_input(
                        thread_id=thread_id,
                        state={"focus": "ordering"},
                    ),
                )

                log = list(graph.get_state(config).events)
                idx_fsm = next(
                    i for i, e in enumerate(log) if isinstance(e, FrontendStateMutated)
                )
                idx_resume_event = next(
                    i for i, e in enumerate(log) if isinstance(e, ApprovalGiven)
                )
                assert idx_fsm < idx_resume_event, (
                    f"FSM must precede the resume domain event "
                    f"(idx_fsm={idx_fsm}, idx_resume={idx_resume_event})"
                )

    def when_mapping_to_agui_output():
        async def it_does_not_warn_about_missing_agui_dict():
            from langgraph_events.agui import FrontendStateMutated
            from langgraph_events.agui._mappers import _warned_classes

            # Ensure the warning can fire for this class in this test run.
            _warned_classes.discard(FrontendStateMutated)

            @on(UserAsked)
            def reply(event: UserAsked) -> AgentReplied:
                return AgentReplied(message=AIMessage(content="ok"))

            graph = EventGraph(
                [reply],
                reducers=[message_reducer()],
                checkpointer=MemorySaver(),
            )
            adapter = AGUIAdapter(
                graph=graph,
                seed_factory=lambda inp: UserAsked(question="go"),
            )

            with warnings.catch_warnings(record=True) as captured:
                warnings.simplefilter("always")
                events = await _collect(
                    adapter,
                    _make_input(state={"focus": "wire-check"}),
                )

            # No missing-agui_dict warning for FrontendStateMutated.
            assert not any("FrontendStateMutated" in str(w.message) for w in captured)
            # And no CustomEvent echoing the event to the client.
            custom_names = [e.name for e in events if isinstance(e, CustomEvent)]
            assert "FrontendStateMutated" not in custom_names


def describe_InterruptedMapper():
    def when_event_subclasses_InterruptedWithPayload():
        def it_emits_a_CustomEvent_using_interrupt_payload():
            from langgraph_events.agui._mappers import InterruptedMapper

            event = ReviewWithPayload(draft="hello")
            ctx = MapperContext(
                run_id="r-1",
                thread_id="t-1",
                input_data=_make_input(),
            )

            result = InterruptedMapper().map(event, ctx)

            assert result is not None
            assert len(result) == 1
            emitted = result[0]
            assert isinstance(emitted, CustomEvent)
            assert emitted.type == EventType.CUSTOM
            assert emitted.name == "interrupted"
            assert emitted.value == {"kind": "review", "draft": "hello"}


def describe_FrontendToolCallRequested():
    def when_only_name_provided():
        def it_is_an_interrupted_subclass():
            e = FrontendToolCallRequested(name="confirm")
            assert isinstance(e, Interrupted)
            assert isinstance(e, Event)

        def it_defaults_args_to_empty_dict():
            e = FrontendToolCallRequested(name="confirm")
            assert e.args == {}

        def it_auto_generates_tool_call_id():
            a = FrontendToolCallRequested(name="confirm")
            b = FrontendToolCallRequested(name="confirm")
            assert a.tool_call_id
            assert b.tool_call_id
            assert a.tool_call_id != b.tool_call_id

    def when_explicit_fields():
        def it_preserves_all_fields():
            e = FrontendToolCallRequested(
                name="run_scenario",
                args={"scenario_id": "s-1"},
                tool_call_id="tc-fixed",
            )
            assert e.name == "run_scenario"
            assert e.args == {"scenario_id": "s-1"}
            assert e.tool_call_id == "tc-fixed"

    def when_agui_dict_called():
        def it_returns_name_args_and_id():
            e = FrontendToolCallRequested(
                name="confirm",
                args={"message": "Ship?"},
                tool_call_id="tc-1",
            )
            d = e.agui_dict()
            assert d == {
                "name": "confirm",
                "args": {"message": "Ship?"},
                "tool_call_id": "tc-1",
            }

    def when_name_is_empty():
        def it_raises_on_construction():
            with pytest.raises(ValueError, match=r"non-empty tool name"):
                FrontendToolCallRequested(name="")

    def when_name_is_whitespace():
        def it_raises_on_construction():
            with pytest.raises(ValueError, match=r"non-empty tool name"):
                FrontendToolCallRequested(name="   ")

    def when_no_args():
        def it_raises_missing_name():
            with pytest.raises(TypeError, match=r"missing.*name"):
                FrontendToolCallRequested()  # type: ignore[call-arg]


def describe_InterruptedWithPayload():
    def when_subclassed():
        def with_a_typed_payload():
            def it_returns_the_payload_from_interrupt_payload():
                class ReviewPayload(dict):
                    pass

                class ReviewInterrupted(InterruptedWithPayload[ReviewPayload]):
                    draft: str
                    revision: int

                    def interrupt_payload(self) -> ReviewPayload:
                        return ReviewPayload(
                            kind="review",
                            draft=self.draft,
                            revision=self.revision,
                        )

                event = ReviewInterrupted(draft="hello", revision=2)
                assert event.interrupt_payload() == {
                    "kind": "review",
                    "draft": "hello",
                    "revision": 2,
                }

            def it_remains_substitutable_for_Interrupted():
                class P(dict):
                    pass

                class MyInterrupt(InterruptedWithPayload[P]):
                    value: str

                    def interrupt_payload(self) -> P:
                        return P(value=self.value)

                assert issubclass(MyInterrupt, Interrupted)
                assert isinstance(MyInterrupt(value="x"), Interrupted)

        def without_overriding_interrupt_payload():
            def it_raises_NotImplementedError_naming_the_method():
                class P(dict):
                    pass

                class Forgotten(InterruptedWithPayload[P]):
                    pass

                with pytest.raises(NotImplementedError, match="interrupt_payload"):
                    Forgotten().interrupt_payload()


def describe_FrontendToolCallRequested_deprecated_top_level_alias():
    def when_imported_from_top_level_langgraph_events():
        def it_resolves_to_the_agui_class_and_warns():
            import langgraph_events
            from langgraph_events.agui import FrontendToolCallRequested as Canonical

            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                shim_cls = langgraph_events.FrontendToolCallRequested

            assert shim_cls is Canonical
            depr = [w for w in caught if issubclass(w.category, DeprecationWarning)]
            assert depr, (
                "expected DeprecationWarning when accessing the top-level "
                "FrontendToolCallRequested alias"
            )
            assert any("langgraph_events.agui" in str(w.message) for w in depr), (
                "deprecation message should point users at the new import path"
            )

        def it_appears_neither_in___all___nor_at_module_attribute_dir():
            # The deprecated alias is reachable via attribute access for back
            # compat, but tooling (autoimport, * imports) should not surface it.
            import langgraph_events

            assert "FrontendToolCallRequested" not in langgraph_events.__all__
