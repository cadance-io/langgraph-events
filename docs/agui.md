# AG-UI Protocol Adapter

[AG-UI](https://docs.ag-ui.com) is an open protocol (by CopilotKit) for streaming agent events to frontends. The `langgraph_events.agui` subpackage maps EventGraph streams to AG-UI SSE events, so any AG-UI-compatible frontend (CopilotKit, custom UIs) can consume your event-driven agents without custom wiring.

```
RunAgentInput -> SeedFactory -> EventGraph.astream_events -> [Mapper Chain] -> SSE
                                                  ^
                              ResumeFactory -> astream_resume (if resuming)

RunAgentInput -> AGUIAdapter.connect/reconnect -> checkpoint snapshots + interrupts
```

`AGUIAdapter` enables real-time assistant token streaming by default (it internally uses `include_llm_tokens=True`) and custom-event passthrough (it internally uses `include_custom_events=True`), emitting `TextMessageStart`/`TextMessageContent`/`TextMessageEnd` while the model is generating.

## Install

```bash
pip install "langgraph-events[agui] @ git+https://github.com/cadance-io/langgraph-events.git"
```

You'll also need `starlette` or `fastapi` for the HTTP layer.

## Quick Start

```python
from fastapi import FastAPI, Request
from ag_ui.core import RunAgentInput
from langgraph.checkpoint.memory import MemorySaver

from langgraph_events import Event, EventGraph, on, MessageEvent
from langgraph_events.agui import AGUIAdapter, create_starlette_response
from langchain_core.messages import HumanMessage, AIMessage


class UserMessageReceived(MessageEvent):
    message: HumanMessage


class AssistantReplied(MessageEvent):
    message: AIMessage


@on(UserMessageReceived)
async def reply(event: UserMessageReceived) -> AssistantReplied:
    return AssistantReplied(message=AIMessage(content="Hello from the agent!"))


graph = EventGraph([reply], checkpointer=MemorySaver())


def seed_factory(input_data: RunAgentInput) -> list[Event]:
    last_msg = input_data.messages[-1]
    return [UserMessageReceived(message=HumanMessage(content=last_msg.content))]


adapter = AGUIAdapter(graph, seed_factory=seed_factory)

app = FastAPI()


@app.post("/api/copilotkit")
async def run(request: Request):
    input_data = RunAgentInput.model_validate_json(await request.body())
    return create_starlette_response(adapter.stream(input_data))
```

Use `error_message` to avoid leaking internal exception details to clients:

```python
adapter = AGUIAdapter(
    graph,
    seed_factory=seed_factory,
    error_message="Something went wrong. Please try again.",
)
```

## Built-in Mapper Chain

Events flow through a priority chain of mappers. The first mapper to claim an event (return non-`None`) wins; unclaimed events fall through to the next mapper.

| Priority | Mapper | Handles | AG-UI Events Produced |
|----------|--------|---------|-----------------------|
| 1 | `SkipInternalMapper` | `Resumed`, `SystemPromptSet` | *(suppressed — claims but emits nothing)* |
| 2 | `InterruptedMapper` | `Interrupted` subclasses | `CustomEvent` (name=`"interrupted"`) |
| 3 | `MessageEventMapper` | `MessageEvent` (AI + tool messages) | `TextMessage*`, `ToolCall*`, `ToolCallResult` |
| 4 | *(user mappers)* | *(your custom logic)* | *(any AG-UI event)* |
| 5 | `FallbackMapper` | Unclaimed `AGUISerializable` events | `CustomEvent` (name=`agui_event_name` when implemented, else class name; value=`agui_dict()`) |

Events without `agui_dict()` are skipped with a one-time warning.

`StreamFrame` reducer data is emitted outside the mapper chain as `StateSnapshot` and `MessagesSnapshot` events (when `include_reducers` is enabled).

`StateSnapshotFrame` is handled outside the mapper chain as AG-UI `StateSnapshot`.

`CustomEventFrame` passthrough is also handled outside the mapper chain as AG-UI `CustomEvent` with `name`/`value` copied through.

The adapter also emits lifecycle events automatically: `RunStarted` at the beginning, `RunFinished` at the end (or `RunError` on exception).

## Connect / Reconnect

For page refreshes and reconnects, use `connect()` to emit checkpoint-backed state without running handlers again:

```python
events = [event async for event in adapter.connect(input_data)]
```

`connect()` emits:

- `StateSnapshot` from checkpoint reducers (empty `{}` for new threads)
- `MessagesSnapshot` from the `messages` reducer (empty `[]` for new threads or when no messages reducer is present)
- pending `Interrupted` events (mapped via the normal mapper chain)

`reconnect()` is an alias for `connect()`.

This is the general HITL rehydration path; it restores UI state and pending interrupts without advancing the graph.

## Endpoint Pattern (Run + Connect)

AG-UI does not require fixed endpoint names. A practical pattern is to expose separate execution and rehydration endpoints:

```python
from fastapi import FastAPI, Request
from ag_ui.core import RunAgentInput
from langgraph_events.agui import create_starlette_response

app = FastAPI()


@app.post("/api/copilotkit")
async def run(request: Request):
    input_data = RunAgentInput.model_validate_json(await request.body())
    return create_starlette_response(adapter.stream(input_data))


@app.post("/api/copilotkit/connect")
async def connect(request: Request):
    input_data = RunAgentInput.model_validate_json(await request.body())
    return create_starlette_response(adapter.connect(input_data))
```

Recommended client behavior:

- call `connect` on page load/refresh to rehydrate state and pending interrupts
- call `stream` only when starting or resuming execution

## Custom Mappers

Implement the `EventMapper` protocol — a single `map()` method. Return `None` to pass, `[]` to suppress, or a list of AG-UI events to emit.

```python
from ag_ui.core import BaseEvent, EventType, CustomEvent
from langgraph_events.agui import EventMapper, MapperContext
from langgraph_events import Event


class PlanningStarted(Event):
    goal: str


class PlanMapper:
    def map(self, event: Event, ctx: MapperContext) -> list[BaseEvent] | None:
        if not isinstance(event, PlanningStarted):
            return None  # pass to next mapper
        return [
            CustomEvent(type=EventType.CUSTOM, name="step_started", value={"goal": event.goal}),
        ]


adapter = AGUIAdapter(graph, seed_factory=seed_factory, mappers=[PlanMapper()])
```

User mappers run after the built-in mappers (priority 4) but before `FallbackMapper`, so they can intercept domain events that would otherwise become generic `CustomEvent`s.

## Resume Support

To handle resumed conversations (e.g., after a human-in-the-loop interrupt), implement `resume_factory`:

```python
from ag_ui.core import RunAgentInput
from langgraph_events.agui import ResumeFactory


class ApprovalSubmitted(Event):
    approved: bool


def resume_factory(
    input_data: RunAgentInput,
    checkpoint_state: dict[str, Any] | None,
) -> Event | None:
    # checkpoint_state includes:
    # - reducers: full reducer snapshot dict
    # - events: reducer-projected event list (if present)
    # - messages: reducer-projected messages (if present)
    # - pending_interrupts: pending Interrupted payloads
    # - is_interrupted: bool
    state = input_data.state or {}
    if "approved" in state:
        return ApprovalSubmitted(approved=state["approved"])
    return None  # fresh run


adapter = AGUIAdapter(
    graph,
    seed_factory=seed_factory,
    resume_factory=resume_factory,
)
```

When `resume_factory` returns an `Event`, the adapter uses `graph.astream_resume()` internally instead of `graph.astream_events()`. When it returns `None`, a fresh run is started with the `seed_factory`.

The second `checkpoint_state` argument is optional; one-argument factories continue to work.

## LangGraph Config Passthrough

`AGUIAdapter.stream()` and `AGUIAdapter.connect()` accept LangGraph config via `RunAgentInput.forwarded_props`.

Supported shapes:

- `forwarded_props["langgraph_config"] = {...}`
- `forwarded_props["config"] = {...}`
- `forwarded_props = {...}` when it already looks like a LangGraph config

The adapter always injects/overrides `configurable.thread_id` from `RunAgentInput.thread_id`, while preserving any other keys (for example `recursion_limit` or tenant/user routing keys in `configurable`).

## AG-UI Spec Coverage

The adapter covers 13 of the 33 AG-UI event types automatically. The remaining types can be emitted via custom `EventMapper` implementations or are not applicable.

| Category | Count | Event Types |
|----------|-------|-------------|
| **Built-in** | 13 | `RunStarted`, `RunFinished`, `RunError`, `TextMessageStart/Content/End`, `ToolCallStart/Args/End`, `ToolCallResult`, `StateSnapshot`, `MessagesSnapshot`, `Custom` |
| **User mapper** | 16 | `TextMessageChunk`, `ToolCallChunk`, `StepStarted/Finished`, `StateDelta`, `ActivitySnapshot/Delta`, `ThinkingStart/End`, `ThinkingTextMessageStart/Content/End`, `Raw`, `ReasoningStart/End` |
| **N/A** | 4 | `ReasoningMessageStart/Content/End/Chunk` — extended reasoning events that require provider-specific integration |
