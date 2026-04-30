# AG-UI Protocol Adapter

[AG-UI](https://docs.ag-ui.com) is an open protocol (by CopilotKit) for streaming agent events to frontends. The `langgraph_events.agui` subpackage maps EventGraph streams to AG-UI SSE events, so any AG-UI-compatible frontend (CopilotKit, custom UIs) can consume your event-driven agents without custom wiring.

```
RunAgentInput -> SeedFactory -> EventGraph.astream_events -> [Mapper Chain] -> SSE
                                                  ^
                              ResumeFactory -> astream_resume (if resuming)

RunAgentInput -> AGUIAdapter.connect/reconnect -> checkpoint snapshots + interrupts
```

`AGUIAdapter` enables real-time assistant token streaming by default (it internally uses `include_llm_tokens=True`) and custom-event passthrough (it internally uses `include_custom_events=True`), emitting `TextMessageStart`/`TextMessageContent`/`TextMessageEnd` while the model is generating. The adapter requires `message_reducer()` on the `EventGraph` for authoritative message delivery.

## Install

```bash
pip install "langgraph-events[agui]"
```

You'll also need `starlette` or `fastapi` for the HTTP layer.

## Quick Start

```python
from fastapi import FastAPI, Request
from ag_ui.core import RunAgentInput
from langgraph.checkpoint.memory import MemorySaver

from langgraph_events import Event, EventGraph, on, MessageEvent, message_reducer
from langgraph_events.agui import AGUIAdapter, create_starlette_response
from langchain_core.messages import HumanMessage, AIMessage


class UserMessageReceived(MessageEvent):
    message: HumanMessage


class AssistantReplied(MessageEvent):
    message: AIMessage


@on(UserMessageReceived)
async def reply(event: UserMessageReceived) -> AssistantReplied:
    return AssistantReplied(message=AIMessage(content="Hello from the agent!"))


graph = EventGraph([reply], checkpointer=MemorySaver(), reducers=[message_reducer()])


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
| 3 | *(user mappers)* | *(your custom logic)* | *(any AG-UI event)* |
| 4 | `FallbackMapper` | Unclaimed `AGUISerializable` events | `CustomEvent` (name=`agui_event_name` when implemented, else class name; value=`agui_dict()`) |

Events without `agui_dict()` are skipped with a one-time warning.

`StreamFrame` reducer data is emitted outside the mapper chain as `StateSnapshot` and `MessagesSnapshot` events (`include_reducers` defaults to `True`). The `message_reducer()` must be registered on the `EventGraph` for `MessagesSnapshot` delivery.

### Shaping client-facing state

`AGUIAdapter(include_reducers=...)` controls what reducer state crosses the wire. It accepts:

- `True` (default) — ship every user reducer.
- `list[str]` — allow-list (e.g. `["focus", "scene"]`).
- `False` — ship no user reducers.
- `Callable[[dict], dict]` — a `StateProjector` for hide / redact / rename / reshape.

The same value is applied symmetrically to outbound `StateSnapshotEvent` and inbound `RunAgentInput.state` echo (so a stale or untrusted client can't inject keys you've decided are internal). Framework-internal channels (`events`, `_cursor`, `_pending`, `_round`) and dedicated AG-UI keys (`messages`) are always stripped first — your projector never sees them.

Hide a few internal reducers with the `drop_reducers()` builder:

```python
from langgraph_events.agui import AGUIAdapter, drop_reducers

adapter = AGUIAdapter(
    graph=graph,
    seed_factory=lambda inp: UserAsked(question=...),
    include_reducers=drop_reducers("debug_count", "scratch"),
)
```

Custom projection (redact / transform):

```python
def project(reducers: dict) -> dict:
    out = dict(reducers)
    if "user" in out:
        out["user"] = mask_pii(out["user"])
    return out

adapter = AGUIAdapter(graph=g, seed_factory=..., include_reducers=project)
```

When `include_reducers` is a callable, the adapter activates *all* reducers at the graph level (it can't introspect names from a callable) — projection happens after computation. For perf-sensitive setups with expensive reducers, prefer the `list[str]` form.

When reducer delta metadata is available (`StreamFrame.changed_reducers`, emitted by `EventGraph` v2 streaming), the adapter suppresses redundant snapshots:

- `MessagesSnapshot` is emitted only when the `messages` reducer changed.
- `StateSnapshot` is emitted once initially, then only when non-dedicated reducers changed.

Messages are delivered through two complementary mechanisms: `MessagesSnapshot` provides authoritative message state from the `message_reducer()`, while LLM tokens stream in real-time as `TextMessageStart`/`TextMessageContent`/`TextMessageEnd` for character-by-character UX. AG-UI clients reconcile the two.

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

User mappers run after the built-in mappers (priority 3) but before `FallbackMapper`, so they can intercept domain events that would otherwise become generic `CustomEvent`s.

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

### Frontend state on resume

When `RunAgentInput.state` is populated alongside a resume, the adapter routes it through `FrontendStateMutated` exactly like the non-resume path:

1. The state dict is projected via `include_reducers` (framework internals + dedicated keys stripped, then your projector if any).
2. A `FrontendStateMutated(state=projected)` event is built.
3. For each reducer subscribing to `FrontendStateMutated`, the adapter computes its contribution and writes it to the channel via `apre_seed` *before* the resume's domain dispatch — so the reducer's `fn` runs (transformations, `SKIP`) and any handler reading **reducer state via parameter injection** (e.g. `def my_handler(event: ApprovalGiven, focus: str | None = None)`) sees the updated value during the resume step.
4. FSM is also injected as a seed to `astream_resume` so it appears in the output stream and the persisted audit log.

**Reducers driven by backend domain events are not affected by FSM dispatch** — their channels stay intact regardless of what the client echoes. The resume's domain event flows through normal dispatch and wins for shared keys.

**`@on(FrontendStateMutated)` *handler* callbacks do not fire on resume.** This is distinct from the parameter-injection pattern above: a function decorated with `@on(FrontendStateMutated)` will not be invoked, because LangGraph's `Command(resume=...)` carries one value and seeds dispatch out-of-graph. The reducer pipeline still runs (#3 above); it's only the dispatched-handler entry point that's unavailable on resume. Use `@on(Resumed)` or `@on(Resumed, interrupted=...)` for resume-time side effects.

## Frontend Tools

The AG-UI spec positions tool calls as "inherently frontend-executed" and as the mechanism for HITL. The adapter wires all three halves of that contract to an `EventGraph`:

1. **Tool definitions in** — a page's `useFrontendTool` registrations arrive on each request as `RunAgentInput.tools`. The `build_langchain_tools(...)` helper converts them to OpenAI-format dicts suitable for `llm.bind_tools(...)`.
2. **Tool calls out** — two paths, both mapping to `ToolCallStart`/`ToolCallArgs`/`ToolCallEnd`:
    - **LLM-initiated** — when the bound LLM streams `tool_call_chunks`, they auto-translate to the streaming triple (transport-level, emitted progressively).
    - **Handler-initiated** — a handler returns the built-in `FrontendToolCallRequested(Interrupted)` event. The adapter emits the full triple and the graph pauses, exactly like `ApprovalRequested(Interrupted)` in [`examples/human_in_the_loop.py`](https://github.com/cadance-io/langgraph-events/blob/main/examples/human_in_the_loop.py). Tool calls become "HITL with typed fields."
3. **Tool results back** — the frontend's handler return value is sent back as a `role: "tool"` message. `detect_new_tool_results(input_data, checkpoint_state)` returns the new `ToolMessage`s; wrap them in a `MessageEvent` (typically `ToolsExecuted(messages=...)`) and return from `resume_factory` to continue the graph.

```python
from langgraph_events import FrontendToolCallRequested, on
from langgraph_events.agui import (
    AGUIAdapter,
    build_langchain_tools,
    detect_new_tool_results,
)


def seed_factory(input_data, checkpoint_state=None):
    return [
        ToolsRegistered(tools=tuple(input_data.tools or [])),
        UserMessageReceived(message=HumanMessage(content=input_data.messages[-1].content)),
    ]


def resume_factory(input_data, checkpoint_state=None):
    results = detect_new_tool_results(input_data, checkpoint_state)
    return ToolsExecuted(messages=tuple(results)) if results else None


@on(UserMessageReceived, ToolsExecuted)
async def call_llm(event, messages, log):
    registered = log.latest(ToolsRegistered)
    tools = build_langchain_tools(list(registered.tools)) if registered else []
    llm = ChatOpenAI(model="gpt-4o-mini").bind_tools(tools) if tools else ChatOpenAI(model="gpt-4o-mini")
    return LLMResponded(message=await llm.ainvoke(messages))
```

Runnable examples:

- [`examples/agui_frontend_tools.py`](https://github.com/cadance-io/langgraph-events/blob/main/examples/agui_frontend_tools.py) — LLM-initiated path with a ReAct-style loop.
- [`examples/agui_confirm_dialog.py`](https://github.com/cadance-io/langgraph-events/blob/main/examples/agui_confirm_dialog.py) — handler-initiated path (deterministic HITL).

**`useCopilotAction` (v1) vs `useFrontendTool` (v2).** The v1 hook consumes tool calls from `MessagesSnapshot` — the existing `MessagesSnapshot` path already covers it unchanged. The v2 hook needs the streaming `ToolCallStart/Args/End` events that this section describes. Both paths coexist: CopilotKit reconciles by `tool_call_id`.

**`parent_message_id` caveat.** For LLM-initiated tool calls, `ToolCallStartEvent.parent_message_id` references the currently-open text message id (or `None` when the assistant emits tool calls without prose). LangChain does not expose the final `AIMessage.id` until the stream ends, so this id is not guaranteed to match the id later carried in `MessagesSnapshot`.

**Reconnect replay.** If a page refresh hits `connect()` while the graph is paused on `FrontendToolCallRequested`, the adapter replays the `ToolCallStart`/`ToolCallArgs`/`ToolCallEnd` triple using the stored `tool_call_id`. CopilotKit's `useFrontendTool` is idempotent by `tool_call_id`, so replay is safe.

**Strict contract — no silent fallbacks.** The adapter rejects malformed tool-call traffic on the spot rather than coercing missing fields:

- `FrontendToolCallRequested(name="")` (or whitespace-only) raises `ValueError` at construction.
- `FrontendToolCallRequested.args` is serialized with `json.dumps()` at emit time; non-JSON-serializable values (e.g. `datetime`) raise `TypeError`, which the adapter surfaces as a `RUN_ERROR`. Keep `args` JSON-compatible — the same constraint the frontend `useFrontendTool` schema imposes.
- An LLM `tool_call_chunk` lacking `index` raises `ValueError` from `astream_events`.
- The first chunk of a streaming call must carry both `id` and `name`; a missing value raises `ValueError`. Continuation chunks may omit them.
- An inbound `role: "tool"` message reaching `detect_new_tool_results` must carry a non-empty `tool_call_id`; missing/empty raises `ValueError`.

Streaming-path errors propagate through the adapter's top-level handler and surface to the frontend as a `RUN_ERROR` event with the diagnostic message. Conformant CopilotKit clients and LangChain chat models satisfy these invariants by default.

## LangGraph Config Passthrough

`AGUIAdapter.stream()` and `AGUIAdapter.connect()` accept LangGraph config via `RunAgentInput.forwarded_props`.

Supported shapes:

- `forwarded_props["langgraph_config"] = {...}`
- `forwarded_props["config"] = {...}`
- `forwarded_props = {...}` when it already looks like a LangGraph config

The adapter always injects/overrides `configurable.thread_id` from `RunAgentInput.thread_id`, while preserving any other keys (for example `recursion_limit` or tenant/user routing keys in `configurable`).

## AG-UI Spec Coverage

The adapter covers 9 of the 33 AG-UI event types automatically. The remaining types can be emitted via custom `EventMapper` implementations or are not applicable.

| Category | Count | Event Types |
|----------|-------|-------------|
| **Built-in** | 12 | `RunStarted`, `RunFinished`, `RunError`, `TextMessageStart/Content/End`, `ToolCallStart/Args/End`, `StateSnapshot`, `MessagesSnapshot`, `Custom` |
| **User mapper** | 17 | `TextMessageChunk`, `ToolCallResult`, `ToolCallChunk`, `StepStarted/Finished`, `StateDelta`, `ActivitySnapshot/Delta`, `ThinkingStart/End`, `ThinkingTextMessageStart/Content/End`, `Raw`, `ReasoningStart/End` |
| **N/A** | 4 | `ReasoningMessageStart/Content/End/Chunk` — extended reasoning events that require provider-specific integration |
