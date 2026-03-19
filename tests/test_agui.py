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
from langgraph_events.agui import (
    AGUIAdapter,
    MapperContext,
)
from langgraph_events.agui._mappers import _serialize_event, _warned_classes

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

            result_events = [e for e in events if e.type == EventType.TOOL_CALL_RESULT]
            assert len(result_events) == 1
            assert result_events[0].tool_call_id == "tc-1"
            assert result_events[0].content == "result-1"

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
                e for e in events if e.type == EventType.CUSTOM and e.name == "Resumed"
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

        async def it_emits_state_snapshot_when_include_reducers():
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
            # Snapshot should contain messages key
            assert "messages" in snapshots[-1].snapshot

        async def it_emits_messages_snapshot_when_messages_reducer_present():
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

            msg_snapshots = [e for e in events if e.type == EventType.MESSAGES_SNAPSHOT]
            assert len(msg_snapshots) >= 1
            # Should contain converted messages
            last_snap = msg_snapshots[-1]
            assert isinstance(last_snap.messages, list)
            assert len(last_snap.messages) >= 1

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

        async def it_skips_events_without_agui_dict():
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

        async def it_emits_events_with_agui_dict():
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

        def describe_when_messages_unchanged():
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
        async def it_uses_resume_factory_when_available():
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
            events = await _collect(adapter, _make_input(thread_id="t-resume-factory"))

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
            events = await _collect(adapter, _make_input(thread_id="t-stream-resume"))

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

    def describe_init():
        def it_raises_when_resume_factory_without_checkpointer():
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
        async def it_calls_seed_factory_with_input():
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


def describe_serialize_event():
    def it_serializes_simple_events():
        event = TaskCreated(title="test")
        result = _serialize_event(event)
        assert result == {"title": "test"}

    def it_handles_non_serializable_fields():
        class WithObj(Event):
            data: object = object()

        event = WithObj()
        result = _serialize_event(event)
        assert "data" in result
        # Result must be JSON-round-trip safe
        json.dumps(result)

    def it_uses_agui_dict_when_available():
        from langgraph_events.agui import AGUISerializable

        class CustomSerialized(Event):
            raw: str = "raw-value"

            def agui_dict(self) -> dict[str, Any]:
                return {"custom_key": self.raw.upper()}

        assert isinstance(CustomSerialized(), AGUISerializable)
        result = _serialize_event(CustomSerialized())
        assert result == {"custom_key": "RAW-VALUE"}

    def it_handles_asdict_success_with_non_json_values():
        from datetime import datetime

        class WithDatetime(Event):
            created_at: datetime = datetime(2025, 1, 15, 12, 0, 0)

        result = _serialize_event(WithDatetime())
        assert "created_at" in result
        # datetime should be coerced to string, and result must be JSON-safe
        json.dumps(result)
        assert isinstance(result["created_at"], str)


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


def describe_resume_with_reducers():
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
