"""Tests for the Reducer primitive."""

from dataclasses import dataclass

import pytest
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

from langgraph_events import (
    Event,
    EventGraph,
    EventLog,
    MessageEvent,
    Reducer,
    message_reducer,
    on,
)

# ---------------------------------------------------------------------------
# Test events
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MsgIn(Event):
    text: str = ""


@dataclass(frozen=True)
class MsgOut(Event):
    text: str = ""


@dataclass(frozen=True)
class Done(Event):
    result: str = ""


# ---------------------------------------------------------------------------
# Test: basic reducer
# ---------------------------------------------------------------------------


class TestReducerBasic:
    def test_reducer_injected_into_handler(self):
        """Handler receives accumulated values via parameter name."""

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

        # At time respond() runs, history = default + project(MsgIn)
        assert received_history == ["start", "in:hello"]
        assert log.latest(Done) == Done(result="HELLO")

    def test_reducer_grows_across_rounds(self):
        """Reducer values grow as events flow through multiple rounds."""

        def project(event: Event) -> list:
            if isinstance(event, MsgIn):
                return [f"in:{event.text}"]
            if isinstance(event, MsgOut):
                return [f"out:{event.text}"]
            return []

        r = Reducer("history", fn=project)

        call_count = 0
        snapshots: list[list] = []

        @dataclass(frozen=True)
        class ToolResult(Event):
            result: str = ""

        @on(MsgIn, ToolResult)
        def call_llm(event: Event, history: list) -> MsgOut:
            nonlocal call_count
            call_count += 1
            snapshots.append(list(history))
            if isinstance(event, MsgIn):
                return MsgOut(text="need_tool")
            return MsgOut(text=f"final:{event.result}")

        @on(MsgOut)
        def handle_response(event: MsgOut) -> ToolResult | Done:
            if event.text == "need_tool":
                return ToolResult(result="42")
            return Done(result=event.text)

        # Also register project for ToolResult
        def project_all(event: Event) -> list:
            if isinstance(event, MsgIn):
                return [f"in:{event.text}"]
            if isinstance(event, MsgOut):
                return [f"out:{event.text}"]
            if isinstance(event, ToolResult):
                return [f"tool:{event.result}"]
            return []

        r = Reducer("history", fn=project_all)

        graph = EventGraph([call_llm, handle_response], reducers=[r])
        graph.invoke(MsgIn(text="question"))

        assert call_count == 2
        # First call: sees [in:question]
        assert snapshots[0] == ["in:question"]
        # Second call: sees [in:question, out:need_tool, tool:42]
        assert snapshots[1] == ["in:question", "out:need_tool", "tool:42"]


class TestReducerWithLog:
    def test_handler_receives_both_log_and_reducer(self):
        """Handler can use both EventLog and reducer simultaneously."""

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

        # log has 1 event (MsgIn), texts has 1 item ("hi")
        assert log.latest(Done) == Done(result="log=1,texts=1")


class TestReducerNoContribution:
    def test_events_without_contribution_dont_affect_reducer(self):
        """Events that return [] from the projection fn don't add to reducer."""

        def project(event: Event) -> list:
            if isinstance(event, MsgIn):
                return [event.text]
            return []  # MsgOut and Done don't contribute

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

        # respond sees ["a"], finish also sees ["a"] (MsgOut didn't add)
        assert snapshots[0] == ["a"]
        assert snapshots[1] == ["a"]


class TestReducerMultiple:
    def test_two_reducers_on_same_graph(self):
        """Multiple reducers accumulate independently."""

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


class TestReducerParallelHandlers:
    def test_fan_out_both_contribute(self):
        """Two parallel handlers both contribute to the same reducer."""

        @dataclass(frozen=True)
        class Trigger(Event):
            value: str = ""

        @dataclass(frozen=True)
        class ResultA(Event):
            value: str = ""

        @dataclass(frozen=True)
        class ResultB(Event):
            value: str = ""

        @dataclass(frozen=True)
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
        def collect(event: Event, items: list, log: EventLog) -> Collected | None:
            if log.has(ResultA) and log.has(ResultB):
                return Collected(items=tuple(items))
            return None

        graph = EventGraph([handle_a, handle_b, collect], reducers=[r])
        log = graph.invoke(Trigger(value="x"))

        result = log.latest(Collected)
        assert result is not None
        # Should have trigger + both results
        assert "trigger:x" in result.items
        assert "a:x" in result.items
        assert "b:x" in result.items


class TestReducerReactLoop:
    def test_react_loop_with_reducer(self):
        """Full ReAct-style loop using reducer instead of rebuild."""

        @dataclass(frozen=True)
        class UserMsg(Event):
            content: str = ""

        @dataclass(frozen=True)
        class AssistantMsg(Event):
            content: str = ""
            needs_tool: bool = False

        @dataclass(frozen=True)
        class ToolResult(Event):
            result: str = ""

        @dataclass(frozen=True)
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
        def handle_response(event: AssistantMsg) -> ToolResult | FinalAnswer:
            if event.needs_tool:
                return ToolResult(result="42")
            return FinalAnswer(answer=event.content)

        graph = EventGraph([call_llm, handle_response], reducers=[r])
        log = graph.invoke(UserMsg(content="what is 6*7?"))

        assert log.latest(FinalAnswer) == FinalAnswer(answer="got:42")

        # First LLM call: system + user message
        assert message_snapshots[0] == [
            ("system", "You are helpful"),
            ("user", "what is 6*7?"),
        ]

        # Second LLM call: system + user + assistant + tool
        assert message_snapshots[1] == [
            ("system", "You are helpful"),
            ("user", "what is 6*7?"),
            ("assistant", "need tool"),
            ("tool", "42"),
        ]


class TestReducerBackwardCompatible:
    def test_graph_without_reducers_still_works(self):
        """Existing graphs without reducers are unaffected."""

        @on(MsgIn)
        def step(event: MsgIn) -> Done:
            return Done(result=event.text)

        graph = EventGraph([step])
        log = graph.invoke(MsgIn(text="hello"))

        assert log.latest(Done) == Done(result="hello")

    def test_handler_with_log_still_works(self):
        """EventLog injection still works with reducers present."""

        def project(event: Event) -> list:
            return [1]

        r = Reducer("counter", fn=project)

        @on(MsgIn)
        def step(event: MsgIn, log: EventLog) -> Done:
            return Done(result=f"events={len(log)}")

        graph = EventGraph([step], reducers=[r])
        log = graph.invoke(MsgIn(text="hi"))

        assert log.latest(Done) == Done(result="events=1")


class TestReducerEdgeCases:
    def test_non_list_return_raises_type_error(self):
        """Reducer fn that returns a non-list raises TypeError."""

        def bad_project(event: Event) -> list:
            return "not a list"  # type: ignore

        r = Reducer("bad", fn=bad_project)

        @on(MsgIn)
        def step(event: MsgIn) -> Done:
            return Done(result="ok")

        graph = EventGraph([step], reducers=[r])
        with pytest.raises(TypeError, match="must return a list"):
            graph.invoke(MsgIn(text="hello"))

    def test_custom_log_parameter_name(self):
        """EventLog can be injected with any parameter name, not just 'log'."""

        @on(MsgIn)
        def step(event: MsgIn, event_log: EventLog) -> Done:
            return Done(result=f"events={len(event_log)}")

        graph = EventGraph([step])
        log = graph.invoke(MsgIn(text="hi"))

        assert log.latest(Done) == Done(result="events=1")

    def test_checkpointer_no_reducer_doubling(self):
        """Reducers don't double on multi-invocation with checkpointer."""
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

        # First invocation
        result = compiled.invoke({"events": [MsgIn(text="a")]}, config)
        end_events = [e for e in result["events"] if isinstance(e, Done)]
        assert end_events[-1].result == "init,a"

        # Second invocation on same thread — should NOT double
        result = compiled.invoke({"events": [MsgIn(text="b")]}, config)
        end_events = [e for e in result["events"] if isinstance(e, Done)]
        # Should see init + a (from prior) + b (new), not init + a + init + a + b
        assert end_events[-1].result == "init,a,b"

    def test_custom_reducer_function(self):
        """Custom reducer function is used instead of operator.add."""

        def always_keep_last_n(left: list, right: list) -> list:
            """Keep only the last 3 items."""
            combined = left + right
            return combined[-3:]

        def project(event: Event) -> list:
            if isinstance(event, MsgIn):
                return [event.text]
            return []

        r = Reducer("recent", fn=project, reducer=always_keep_last_n)

        snapshots: list[list] = []

        @dataclass(frozen=True)
        class Continue(Event):
            text: str = ""

        @on(MsgIn, Continue)
        def step(event: Event, recent: list) -> MsgOut | Continue:
            snapshots.append(list(recent))
            if isinstance(event, MsgIn):
                return Continue(text="b")
            return MsgOut(text="done")

        # Also project Continue events
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

        @on(MsgOut)
        def finish(event: MsgOut) -> Done:
            return Done(result="ok")

        graph = EventGraph([step, finish], reducers=[r])
        graph.invoke(MsgIn(text="a"))

        # First call: default=[x,y,z] + project(MsgIn)=[a] → keep last 3 → [y,z,a]
        assert snapshots[0] == ["y", "z", "a"]
        # Second call: [y,z,a] + project(Continue)=[b] → keep last 3 → [z,a,b]
        assert snapshots[1] == ["z", "a", "b"]


# ---------------------------------------------------------------------------
# MessageEvent tests
# ---------------------------------------------------------------------------


class TestMessageEventConvention:
    def test_single_message_field(self):
        """MessageEvent with a ``message`` field auto-wraps in a list."""

        @dataclass(frozen=True)
        class UserMsg(MessageEvent):
            message: HumanMessage = None  # type: ignore[assignment]

        msg = HumanMessage(content="hello")
        event = UserMsg(message=msg)
        assert event.as_messages() == [msg]

    def test_messages_field(self):
        """MessageEvent with a ``messages`` field auto-converts tuple to list."""

        @dataclass(frozen=True)
        class ToolResults(MessageEvent):
            messages: tuple[ToolMessage, ...] = ()

        t1 = ToolMessage(content="42", tool_call_id="tc1")
        t2 = ToolMessage(content="7", tool_call_id="tc2")
        event = ToolResults(messages=(t1, t2))
        assert event.as_messages() == [t1, t2]

    def test_empty_messages_field(self):
        """MessageEvent with empty ``messages`` tuple returns empty list."""

        @dataclass(frozen=True)
        class Empty(MessageEvent):
            messages: tuple[ToolMessage, ...] = ()

        event = Empty()
        assert event.as_messages() == []

    def test_neither_field_raises(self):
        """MessageEvent without message/messages raises NotImplementedError."""

        @dataclass(frozen=True)
        class BadEvent(MessageEvent):
            text: str = ""

        event = BadEvent(text="hi")
        with pytest.raises(NotImplementedError, match="must declare"):
            event.as_messages()

    def test_custom_override(self):
        """Subclass can override as_messages() for custom behavior."""

        @dataclass(frozen=True)
        class Custom(MessageEvent):
            text: str = ""

            def as_messages(self) -> list[BaseMessage]:
                return [HumanMessage(content=self.text)]

        event = Custom(text="hello")
        result = event.as_messages()
        assert len(result) == 1
        assert result[0].content == "hello"

    def test_ai_message_field(self):
        """MessageEvent works with AIMessage including tool_calls."""

        @dataclass(frozen=True)
        class LLMResponse(MessageEvent):
            message: AIMessage = None  # type: ignore[assignment]

        ai_msg = AIMessage(
            content="Let me check",
            tool_calls=[{"id": "tc1", "name": "search", "args": {"q": "test"}}],
        )
        event = LLMResponse(message=ai_msg)
        result = event.as_messages()
        assert len(result) == 1
        assert result[0] is ai_msg
        assert result[0].tool_calls == ai_msg.tool_calls


# ---------------------------------------------------------------------------
# message_reducer tests
# ---------------------------------------------------------------------------


class TestMessageReducer:
    def test_projects_message_events(self):
        """message_reducer auto-projects MessageEvent instances."""

        @dataclass(frozen=True)
        class UserMsg(MessageEvent):
            message: HumanMessage = None  # type: ignore[assignment]

        @dataclass(frozen=True)
        class Reply(Event):
            text: str = ""

        r = message_reducer([SystemMessage(content="You are helpful")])

        # MessageEvent is projected
        msg = HumanMessage(content="hello")
        result = r.fn(UserMsg(message=msg))
        assert result == [msg]

        # Non-MessageEvent returns empty
        result = r.fn(Reply(text="hi"))
        assert result == []

    def test_default_messages(self):
        """message_reducer includes default messages."""
        r = message_reducer([SystemMessage(content="Be nice")])
        assert len(r.default) == 1
        assert r.default[0].content == "Be nice"

    def test_no_default(self):
        """message_reducer with no default has empty default."""
        r = message_reducer()
        assert r.default == []

    def test_custom_channel_name(self):
        """message_reducer respects custom channel name."""
        r = message_reducer(name="chat_history")
        assert r.name == "chat_history"

    def test_integration_with_event_graph(self):
        """MessageEvent + message_reducer work together in an EventGraph."""

        @dataclass(frozen=True)
        class UserMsg(MessageEvent):
            message: HumanMessage = None  # type: ignore[assignment]

        @dataclass(frozen=True)
        class BotReply(MessageEvent):
            message: AIMessage = None  # type: ignore[assignment]

        @dataclass(frozen=True)
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

        # Handler received: system + user message
        msgs = received_messages[0]
        assert len(msgs) == 2
        assert msgs[0].content == "You are a test bot"
        assert msgs[1].content == "hello"
