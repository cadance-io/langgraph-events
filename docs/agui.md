# AG-UI Protocol Adapter

[AG-UI](https://docs.ag-ui.com) is an open protocol (by CopilotKit) for streaming agent events to frontends. `langgraph_events.agui` maps `EventGraph` streams to AG-UI SSE events.

```text
RunAgentInput -> SeedFactory -> EventGraph.astream_events -> [Mapper Chain] -> SSE
                                                  ^
                              ResumeFactory -> astream_resume (if resuming)

RunAgentInput -> AGUIAdapter.connect/reconnect -> checkpoint snapshots + interrupts
```

`AGUIAdapter` streams LLM tokens and custom events; requires [`message_reducer()`](reducers.md#message_reducer) for authoritative message delivery.

## Install

```bash
pip install "langgraph-events[agui]"
```

Also need `starlette` or `fastapi` for the HTTP layer.

## Quick Start

```python
from fastapi import FastAPI, Request
from ag_ui.core import RunAgentInput
from langgraph.checkpoint.memory import MemorySaver

from langgraph_events import Event, EventGraph, IntegrationEvent, MessageEvent, message_reducer, on
from langgraph_events.agui import AGUIAdapter, create_starlette_response
from langchain_core.messages import HumanMessage, AIMessage


class UserMessageReceived(IntegrationEvent, MessageEvent):
    message: HumanMessage


class AssistantReplied(IntegrationEvent, MessageEvent):
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

!!! tip "Error handling"
    Use `error_message=...` to avoid leaking exception details: `AGUIAdapter(graph, seed_factory=..., error_message="Something went wrong.")`.

## Built-in Mapper Chain

Mappers claim events in priority order; first non-`None` return wins. Unclaimed events fall through.

| Priority | Mapper | Handles | AG-UI Events Produced |
|---|---|---|---|
| 1 | `SkipInternalMapper` | `Resumed`, `SystemPromptSet` | *(suppressed)* |
| 2 | `InterruptedMapper` | `Interrupted` subclasses | `CustomEvent` (name=`"interrupted"`) |
| 3 | *(user mappers)* | *(your custom logic)* | *(any AG-UI event)* |
| 4 | `FallbackMapper` | Unclaimed `AGUISerializable` events | `CustomEvent` (name=`agui_event_name` if implemented else class name; value=`agui_dict()`) |

Events without `agui_dict()` are skipped with a one-time warning.

### Unmapped-event policy

Most events in a real `EventGraph` are internal orchestration (`Command`s, routing/control-flow `DomainEvent`s) that have no AG-UI representation, so the default per-class warning often reads as noise. Control it with `on_unmapped`:

```python
AGUIAdapter(graph, seed_factory=..., on_unmapped="ignore")
```

| value | behavior |
|---|---|
| `"warn"` *(default)* | Once-per-class `UserWarning`, then drop. Non-breaking. |
| `"ignore"` | Silently drop. The off-switch for apps that are mostly internal events. |
| `"raise"` | Raise `UnmappedEventError` (a `TypeError` subclass) naming the offending class. Strict mode — turns the dev-lint into a hard CI gate. |

The policy applies to both `FallbackMapper` and the non-serializable branch of `InterruptedMapper`. Serializable events (and `InterruptedWithPayload`) are unaffected — they still emit.

Outside the chain, the adapter also emits:

- `StateSnapshot` / `MessagesSnapshot` from `StreamFrame` reducer data (`MessagesSnapshot` requires `message_reducer()`).
- `StateSnapshotFrame` → `StateSnapshot`; `CustomEventFrame` → `CustomEvent` (name/value passthrough).
- Lifecycle: `RunStarted`, `RunFinished`, `RunError` on exception.
- Skips redundant snapshots when `changed_reducers` is available.

### Shaping client-facing state

`AGUIAdapter(include_reducers=...)` controls what reducer state crosses the wire:

| Value | Behaviour |
|---|---|
| `True` (default) | All user reducers |
| `list[str]` | Allow-list only (e.g. `["focus", "scene"]`) |
| `False` | No user reducers (`MessagesSnapshot` still ships) |

Applied symmetrically to outbound state and inbound `RunAgentInput.state` echo (framework internals — `events`, `_cursor`, `_pending`, `_round` — always stripped first). The allow-list also gates **which reducers `EventGraph` computes during streaming**.

!!! note "Symmetry is a security boundary"
    The same value applies to both directions — a stale or untrusted client cannot inject keys you've decided are internal by echoing them back in `RunAgentInput.state`. The framework-internal channels are always stripped first regardless of `include_reducers`.

```python
adapter = AGUIAdapter(
    graph=graph,
    seed_factory=lambda inp: UserAsked(question=...),
    include_reducers=["focus", "scene", "user", "context"],  # debug_count, scratch hidden
)
```

For redaction or value transformation, write a custom `EventMapper`.

!!! note "Message delivery uses two channels"
    Messages reach the client via two paths simultaneously: authoritative `MessagesSnapshot` from the `message_reducer()` channel, plus real-time `TextMessageStart` / `Content` / `End` tokens. AG-UI clients reconcile them by message id.

### Message field mapping

`MessagesSnapshot` converts each LangChain `BaseMessage` to its AG-UI counterpart. The declared fields map like this:

| LangChain | AG-UI | Direction | Notes |
|---|---|---|---|
| `id`, `content` | `id`, `content` | both | `AssistantMessage.content` is `None` for block content; `SystemMessage`/`ToolMessage` degrade it to `""`. |
| `name` | `name` | both | `UserMessage`, `AssistantMessage`, `SystemMessage` only — AG-UI's `ToolMessage` declares no `name`. |
| `AIMessage.tool_calls` | `AssistantMessage.tool_calls` | both | `args` are JSON-encoded into `function.arguments`. |
| `ToolMessage.status` | `ToolMessage.error` | both | `status="error"` sends the content as `error`, or the literal `"error"` when the content is empty, because a falsy `error` reads as a success; an inbound `error` sets `status="error"`. |
| `additional_kwargs["agui"]` | *(extra fields)* | both | Passthrough — see below. |

### Passing extra data on a message

AG-UI messages allow extra fields, so a client can receive structured data the protocol does not declare — a retry hint on a failure notice, for instance. Put a mapping under the reserved `additional_kwargs` key `AGUI_EXTRAS_KEY` (the literal string `"agui"`). Each entry becomes a top-level field on the AG-UI message.

```python
from langchain_core.messages import SystemMessage

from langgraph_events.agui import AGUI_EXTRAS_KEY

SystemMessage(
    content="The step failed.",
    additional_kwargs={AGUI_EXTRAS_KEY: {"failure": {"retryable": True}}},
)
# -> AG-UI SystemMessage(id=..., role="system", content="The step failed.",
#                        failure={"retryable": True})
```

The mapping round-trips. `agui_messages_to_langchain` collects the extra fields an inbound AG-UI message carries back into `additional_kwargs[AGUI_EXTRAS_KEY]`.

!!! warning "Only the reserved key crosses the wire"
    Nothing else in `additional_kwargs` is forwarded. Provider metadata and internal state stay on the backend. An entry that addresses a declared AG-UI field (`content`, `toolCallId`, …) raises `ValueError` naming the key, because it would rewrite protocol data. Rename the entry, or set the field through the LangChain message.

## Connect / Reconnect

`connect()` (alias `reconnect()`) emits checkpoint-backed state without running handlers — `StateSnapshot` + `MessagesSnapshot` from the checkpoint plus any pending `Interrupted` events.

```python
events = [event async for event in adapter.connect(input_data)]
```

Use on page load / refresh.

!!! tip "Typical endpoint split"
    - `/api/copilotkit` (run): `adapter.stream(input_data)`
    - `/api/copilotkit/connect` (reconnect): `adapter.connect(input_data)`
    - Client: call `connect` on load/refresh; call `stream` only to start/resume execution.

    ```python
    @app.post("/api/copilotkit")
    async def run(request: Request):
        input_data = RunAgentInput.model_validate_json(await request.body())
        return create_starlette_response(adapter.stream(input_data))


    @app.post("/api/copilotkit/connect")
    async def connect(request: Request):
        input_data = RunAgentInput.model_validate_json(await request.body())
        return create_starlette_response(adapter.connect(input_data))
    ```

## Custom Mappers

Implement `EventMapper.map()`: return `None` (pass), `[]` (suppress), or AG-UI events.

```python
from ag_ui.core import BaseEvent, EventType, CustomEvent
from langgraph_events import Event, IntegrationEvent
from langgraph_events.agui import EventMapper, MapperContext


class PlanningStarted(IntegrationEvent):
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

User mappers (priority 3) intercept before `FallbackMapper`.

## Resume Support

For resuming (e.g. after HITL), implement `resume_factory(input_data, checkpoint_state)`. Return an `Event` to trigger `astream_resume()`; return `None` for a fresh run via `seed_factory`. `checkpoint_state` is optional — one-argument factories still work.

```python
from ag_ui.core import RunAgentInput
from langgraph_events.agui import ResumeFactory


class ApprovalSubmitted(IntegrationEvent):
    approved: bool


def resume_factory(
    input_data: RunAgentInput,
    checkpoint_state: dict[str, Any] | None,
) -> Event | None:
    # checkpoint_state: CheckpointState TypedDict with reducers, events,
    # messages, pending_interrupts, is_interrupted, snapshot.
    state = input_data.state or {}
    if "approved" in state:
        return ApprovalSubmitted(approved=state["approved"])
    return None  # fresh run


adapter = AGUIAdapter(graph, seed_factory=seed_factory, resume_factory=resume_factory)
```

!!! warning "Frontend state on resume"
    When `RunAgentInput.state` is populated alongside a resume:

    - State dict is projected via `include_reducers` (framework internals + dedicated keys stripped).
    - `FrontendStateMutated(state=projected)` event is built.
    - For each reducer subscribing to `FrontendStateMutated`, the adapter writes via `apre_seed` **before** the resume's domain dispatch — so handlers reading the channel via parameter injection see the updated value.
    - FSM is also injected as a seed to `astream_resume` (appears in stream + audit log).
    - **Domain-driven reducers win for shared keys** — backend channels stay intact regardless of what the client echoes.
    - **`@on(FrontendStateMutated)` *handler* callbacks do NOT fire on resume** (reducer pipeline still runs). Use `@on(Resumed)` or `@on(Resumed, interrupted=...)` for resume-time side effects.

Reducers can subscribe to `FrontendStateMutated` to mirror selected client-state keys server-side:

```python
from langgraph_events import ScalarReducer, SKIP
from langgraph_events.agui import FrontendStateMutated

focus = ScalarReducer(
    event_type=FrontendStateMutated,
    fn=lambda e: e.state.get("focus", SKIP),
)
```

## Frontend Tools

The AG-UI spec positions tool calls as "inherently frontend-executed" and as the mechanism for HITL. The adapter wires all three halves to an `EventGraph`:

1. **Tool definitions in** — page's `useFrontendTool` registrations arrive as `RunAgentInput.tools`. `build_langchain_tools(...)` converts them to OpenAI-format dicts for `llm.bind_tools(...)`.
2. **Tool calls out** — two paths, both mapping to `ToolCallStart`/`ToolCallArgs`/`ToolCallEnd`:
    - **LLM-initiated** — bound LLM `tool_call_chunks` auto-translate to the streaming triple.
    - **Handler-initiated** — return `FrontendToolCallRequested(Interrupted)` from a handler. Graph pauses; AG-UI emits the triple. (Same machinery as `ApprovalRequested(Interrupted)` in [`examples/expense_approval.py`](https://github.com/cadance-io/langgraph-events/blob/main/examples/expense_approval.py).)
3. **Tool results back** — frontend handler return value comes back as `role: "tool"`. `detect_new_tool_results(input_data, checkpoint_state)` returns the new `ToolMessage`s; wrap as `MessageEvent` and return from `resume_factory`.

```python
from langgraph_events import on
from langgraph_events.agui import (
    AGUIAdapter,
    FrontendToolCallRequested,
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

See [`examples/conversation.py`](https://github.com/cadance-io/langgraph-events/blob/main/examples/conversation.py).

### Handler-initiated frontend tools

For backend-initiated tool calls (confirm dialogs, file pickers, deterministic prompts), return `FrontendToolCallRequested(Interrupted)`. Graph pauses; AG-UI streams the call triple; frontend result becomes a typed event on resume.

```python
from langgraph_events import on
from langgraph_events.agui import (
    FrontendToolCallRequested,
    detect_new_tool_results,
)


@on(ShipCommandReceived)
def request_confirmation(event: ShipCommandReceived) -> FrontendToolCallRequested:
    return FrontendToolCallRequested(
        name="confirm",  # must match a useFrontendTool({ name: "confirm", ... }) registration
        args={"prompt": f"Ship release {event.release}?"},
    )


def resume_factory(input_data, checkpoint_state=None):
    results = detect_new_tool_results(input_data, checkpoint_state)
    if not results:
        return None
    return UserConfirmed(messages=tuple(results))


@on(UserConfirmed)
def ship(event: UserConfirmed) -> ShippedRelease:
    approved = bool(json.loads(event.messages[0].content).get("approved"))
    return ShippedRelease(release="v1", approved=approved)
```

!!! note "CopilotKit versioning"
    - v1 (`useCopilotAction`): uses `MessagesSnapshot` (no changes).
    - v2 (`useFrontendTool`): needs streaming `ToolCallStart/Args/End`.
    - Both coexist; CopilotKit reconciles by `tool_call_id`.

!!! warning "`parent_message_id` caveat"
    For LLM-initiated tool calls, `ToolCallStartEvent.parent_message_id` may not match the final `MessagesSnapshot` id — LangChain doesn't expose the final `AIMessage.id` until the stream ends.

!!! note "Reconnect replay"
    If a page refresh hits `connect()` while the graph is paused on `FrontendToolCallRequested`, the adapter replays the triple using the stored `tool_call_id`. CopilotKit's `useFrontendTool` is idempotent by `tool_call_id` — replay is safe.

!!! warning "Strict contract — no silent fallbacks"
    The adapter rejects malformed tool-call traffic at the source:

    - `FrontendToolCallRequested(name="")` (or whitespace-only) raises `ValueError` at construction.
    - `FrontendToolCallRequested.args` is JSON-serialized at emit time; non-serializable values raise `TypeError` → adapter surfaces `RUN_ERROR`. Keep `args` JSON-compatible.
    - An LLM `tool_call_chunk` lacking `index` raises `ValueError` from `astream_events`.
    - First chunk of a streaming call must carry both `id` and `name`; missing raises `ValueError`. Continuation chunks may omit.
    - Inbound `role: "tool"` message must carry a non-empty `tool_call_id`; missing raises `ValueError` from `detect_new_tool_results`.

Streaming-path errors propagate to the frontend as a `RUN_ERROR` event with the diagnostic message.

## Resume Helpers

`detect_new_tool_results` covers the frontend-tool-result arm of resume. A result the client marks with `error` becomes a LangChain `ToolMessage` with `status="error"`, so the model reads the failure as a failure. The other arm — frontend sends `Command(resume=…)` plus new chat messages — has three helpers that collapse `resume_factory` boilerplate:

- **`extract_resume_input(input_data)`** — pull & decode `RunAgentInput.forwarded_props["command"]["resume"]`.
- **`agui_messages_to_langchain(messages, *, drop_invalid_tool_calls=False)`** — convert AG-UI messages (`UserMessage`, `AssistantMessage`, `SystemMessage`, `ToolMessage` — multimodal `UserMessage` content included) to LangChain `BaseMessage`. `ReasoningMessage` / `DeveloperMessage` skipped (DEBUG); `ActivityMessage` / unknown roles raise `ValueError`. With `drop_invalid_tool_calls=True`, tool calls with unparseable JSON args are dropped (WARNING). A `ToolMessage.error` becomes `status="error"`, and extra fields land under `additional_kwargs[AGUI_EXTRAS_KEY]` (see [Message field mapping](#message-field-mapping)).
- **`merge_frontend_messages(input_data, checkpoint_state, *, reducer_name="messages", drop_invalid_tool_calls=True)`** — read existing messages from checkpoint, convert, merge via `add_messages` (id-based dedup; missing ids get UUIDs assigned). Returns a tuple.

```python
from langgraph_events import IntegrationEvent, MessageEvent
from langgraph_events.agui import (
    detect_new_tool_results,
    extract_resume_input,
    merge_frontend_messages,
)


# Domain-specific resume event — define alongside your other events.
class UserResumed(IntegrationEvent, MessageEvent):
    response: object  # resume payload (dict, str, …); your shape
    messages: tuple = ()


def resume_factory(input_data, checkpoint_state=None):
    # 1. Frontend tool result path (covered above)
    tool_results = detect_new_tool_results(input_data, checkpoint_state)
    if tool_results:
        return ToolsExecuted(messages=tuple(tool_results))

    # 2. Command(resume=…) + new chat messages path
    resume_input = extract_resume_input(input_data)
    if resume_input is None:
        return None
    merged = merge_frontend_messages(input_data, checkpoint_state)
    return UserResumed(response=resume_input, messages=merged)
```

!!! tip "Falsy resume handling"
    `extract_resume_input` treats all falsy values (`0`, `""`, `[]`, `{}`, `False`) as "no resume". For meaningful "deny" signals, send `{"approved": false}` instead of bare `false`.

Use `agui_messages_to_langchain` directly for custom merging (e.g. in `seed_factory`).

## Typed Interrupt Payloads

For HITL flows whose frontend needs an action-discriminated dict, subclass `InterruptedWithPayload[PayloadT]` (builds on [`Interrupted`](control-flow.md#interrupted-resumed) from Control Flow) and implement `interrupt_payload()`. `InterruptedMapper` emits the payload as `CustomEvent(name="interrupted", value=...)` — no `agui_dict()` override needed.

```python
from typing import Literal, TypedDict

from langgraph_events import on
from langgraph_events.agui import InterruptedWithPayload


class ReviewPayload(TypedDict):
    kind: Literal["review"]
    draft: str
    revision: int


class ReviewInterrupted(InterruptedWithPayload[ReviewPayload]):
    draft: str
    revision: int

    def interrupt_payload(self) -> ReviewPayload:
        return {"kind": "review", "draft": self.draft, "revision": self.revision}


@on(DraftReady)
def request_review(event: DraftReady) -> ReviewInterrupted:
    return ReviewInterrupted(draft=event.draft, revision=event.revision)
```

Pure `Interrupted` (no payload) is still the right pick for non-frontend HITL.

## LangGraph Config Passthrough

`stream()` / `connect()` accept LangGraph config via `RunAgentInput.forwarded_props` — keys `langgraph_config`, `config`, or the dict itself. The adapter always overrides `configurable.thread_id` from `RunAgentInput.thread_id`; other keys (`recursion_limit`, tenant routing) pass through.

## Soft-timeout

`stream()` accepts an optional `deadline: float` keyword — an absolute `time.monotonic()` reference. When the graph's router observes a current time past the deadline between dispatch rounds, it emits `RunPaused` (see [Control Flow → Soft-timeout](control-flow.md#soft-timeout-runpaused)) and the run finalises cleanly through the same drain + `RunFinishedEvent` path as a normal completion. No early break, no special control flow in the adapter.

```python
from time import monotonic


@app.post("/api/copilotkit")
async def run(input_data: RunAgentInput) -> StreamingResponse:
    # Worker has a hard job_timeout of 180s; soft-pause 30s before.
    hard_budget_s = 180
    soft_margin_s = 30
    return create_starlette_response(
        adapter.stream(
            input_data,
            deadline=monotonic() + (hard_budget_s - soft_margin_s),
        )
    )
```

`RunPaused` is **not surfaced on the AG-UI wire by default** (#88). The class deliberately does not implement `AGUISerializable`, so `FallbackMapper` skips it (one-time warning). The previous default — `CustomEvent(name="interrupted", value={"kind": "soft_timeout", …})` — collided with HITL `Interrupted` events on the same wire name and forced every client to branch on `value.kind`. Apps that want a pause signal on the wire register their own mapper and pick a wire shape that suits their frontend:

```python
from ag_ui.core import BaseEvent, CustomEvent, EventType
from langgraph_events import Event, RunPaused
from langgraph_events.agui import EventMapper, MapperContext


class PauseMapper:
    def map(self, event: Event, ctx: MapperContext) -> list[BaseEvent] | None:
        if not isinstance(event, RunPaused):
            return None
        return [
            CustomEvent(
                type=EventType.CUSTOM,
                name="run.paused",
                value={"elapsed_seconds": event.elapsed_seconds},
            )
        ]


adapter = AGUIAdapter(graph, seed_factory=..., mappers=[PauseMapper()])
```

For an inline pause notice in the message channel (no custom wire event needed), see the reducer recipe in [Control Flow → Surfacing the pause inline](control-flow.md#surfacing-the-pause-inline).

Resume is implicit: the consumer's "Continue" button issues a new `/run` on the same `thread_id` (with a fresh deadline). LangGraph's checkpointer replays from the last completed node — no `Command(resume=...)` required. Position `deadline` strictly tighter than whichever outer hard cancellation the caller has (`asyncio.wait_for`, SAQ `job_timeout`, LangGraph's own `timeout=`) so the soft boundary fires first.

## AG-UI Spec Coverage

Built-in for 12 of 33 event types; the rest via custom mappers or N/A.

| Category | Count | Event Types |
|---|---|---|
| **Built-in** | 12 | `RunStarted`, `RunFinished`, `RunError`, `TextMessageStart/Content/End`, `ToolCallStart/Args/End`, `StateSnapshot`, `MessagesSnapshot`, `Custom` |
| **User mapper** | 17 | `TextMessageChunk`, `ToolCallResult`, `ToolCallChunk`, `StepStarted/Finished`, `StateDelta`, `ActivitySnapshot/Delta`, `ThinkingStart/End`, `ThinkingTextMessageStart/Content/End`, `Raw`, `ReasoningStart/End` |
| **N/A** | 4 | `ReasoningMessageStart/Content/End/Chunk` — extended reasoning, provider-specific |
