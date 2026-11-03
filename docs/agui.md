# AG-UI Protocol Adapter

[AG-UI](https://docs.ag-ui.com) is an open protocol, by CopilotKit, for streaming agent events to a frontend. `langgraph_events.agui` maps an `EventGraph` stream to AG-UI SSE events.

```text
RunAgentInput -> SeedFactory -> EventGraph.astream_events -> [Mapper Chain] -> SSE
                                                  ^
                              ResumeFactory -> astream_resume (if resuming)

RunAgentInput -> AGUIAdapter.connect/reconnect -> RunStarted + checkpoint snapshots + interrupts + RunFinished
```

`AGUIAdapter` streams LLM tokens and custom events. `AGUIAdapter(...)` raises `ValueError` unless the graph registers [`message_reducer()`](reducers.md#message_reducer). That reducer is the authoritative message channel.

## Install

```bash
pip install "langgraph-events[agui]"
```

The HTTP layer also needs `starlette` or `fastapi`.

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

Pass `error_message="Something went wrong."` to `AGUIAdapter(...)` so that exception details do not reach the client. To send the client a specific error code from a factory, see [Async Factories and Refusals](#async-factories-and-refusals).

## Built-in Mapper Chain

Mappers claim events in priority order. The first non-`None` return wins. An unclaimed event falls through.

| Priority | Mapper | Handles | AG-UI Events Produced |
|---|---|---|---|
| 1 | `SkipInternalMapper` | `Resumed`, `SystemPromptSet`, `FrontendStateMutated` | *(suppressed)* |
| 2 | `FrontendToolCallRequestedMapper` | `FrontendToolCallRequested` | `ToolCallStart` / `ToolCallArgs` / `ToolCallEnd` |
| 3 | `InterruptedMapper` | `Interrupted` subclasses | `CustomEvent` (name=`"interrupted"`) |
| 4 | *(user mappers)* | *(your custom logic)* | *(any AG-UI event)* |
| 5 | `FallbackMapper` | Unclaimed `AGUISerializable` events | `CustomEvent` (name=`agui_event_name` if implemented else class name, value=`agui_dict()`) |

An event without `agui_dict()` is skipped with a one-time warning.

### Unmapped-event policy

An internal event, for example a `Command` or a routing `DomainEvent`, has no AG-UI representation. `AGUIAdapter(..., on_unmapped=...)` controls what happens to it:

| value | behavior |
|---|---|
| `"warn"` *(default)* | Once-per-class `UserWarning`, then drop. |
| `"ignore"` | Drop silently. Recommended for an app whose events are mostly internal. |
| `"raise"` | Raise `UnmappedEventError`, a `TypeError` subclass, that names the class. Use it as a CI gate. |

The policy applies to `FallbackMapper` and to the non-serializable branch of `InterruptedMapper`. A serializable event, and an `InterruptedWithPayload`, still emits.

Outside the chain, the adapter also emits:

- `StateSnapshot` / `MessagesSnapshot` from `StreamFrame` reducer data. `MessagesSnapshot` reads the `message_reducer()` channel. Redundant snapshots are skipped when `changed_reducers` is available.
- `StateSnapshotFrame` → `StateSnapshot`.
- `CustomEventFrame` → `CustomEvent` (name/value passthrough).
- `InterruptFrame` → its `value` through the mapper chain. The run yields one frame for each interrupt that it leaves pending.
- Lifecycle: `RunStarted`, `RunFinished`, `RunError` on exception. A handler that raises `RunRefusedError` gets the generic `RunError`, see [Async Factories and Refusals](#async-factories-and-refusals).

### Shaping client-facing state

`AGUIAdapter(include_reducers=...)` controls what reducer state reaches the client:

| Value | Behaviour |
|---|---|
| `True` (default) | All user reducers |
| `list[str]` | Allow-list only (e.g. `["focus", "scene"]`) |
| `False` | No user reducers (`MessagesSnapshot` is still sent) |

The value applies to outbound state and to the inbound `RunAgentInput.state` echo, so a client cannot inject a key you keep internal. Every framework channel, and the `messages` key, is stripped first, whatever `include_reducers` holds. The allow-list also gates **which reducers `EventGraph` computes during streaming**. For redaction or value transformation, write a custom `EventMapper`.

Messages reach the client on two paths at the same time. The authoritative path is `MessagesSnapshot` from the `message_reducer()` channel. The real-time path is the `TextMessageStart` / `Content` / `End` tokens. An AG-UI client reconciles them by message id.

### Message field mapping

`MessagesSnapshot` converts each LangChain `BaseMessage` to its AG-UI counterpart:

| LangChain | AG-UI | Direction | Notes |
|---|---|---|---|
| `id`, `content` | `id`, `content` | both | AG-UI declares both as text. `AssistantMessage.content` is `None` for block content, and `SystemMessage` degrades it to `""`. A `UserMessage` or `ToolMessage` carries block content as content parts. See the `content` blocks row below. |
| `name` | `name` | both | `UserMessage`, `AssistantMessage` and `SystemMessage` only. The AG-UI `ToolMessage` declares no `name`. |
| `AIMessage.tool_calls` | `AssistantMessage.tool_calls` | both | `args` are JSON-encoded into `function.arguments`. |
| `ToolMessage.status` | `ToolMessage.error` | both | `status="error"` sends the content as `error`, or the literal `"error"` when the content is empty, because a falsy `error` reads as a success. A truthy inbound `error` sets `status="error"`. The error *text* does not come back: LangChain's `ToolMessage` has no field for it. |
| *(none)* | `encrypted_value` | neither | Declared on AG-UI `BaseMessage`, `ToolMessage` and `ToolCall`. The adapter never reads it inbound and never sets it outbound, so a value the client sent is dropped. |
| `content` blocks | `content` parts | both | On a `UserMessage` or `ToolMessage`, each part maps to the LangChain standard block of the same modality. A `document` part maps to a `file` block. A `data` source maps to `base64` with `mime_type`, and a `url` source maps to `url`. A part `id` is kept. Outbound, LangChain first translates each block to a standard block, so an `image_url` block also maps, and a `file_id` maps to a `file` source. Outbound, a block with no AG-UI part, such as a `base64` block without `mime_type`, is dropped with a `WARNING` that names the message. Inbound, a part with a `file` source is held, not sent to the model. See [Parts the model does not receive](#parts-the-model-does-not-receive). Inbound, the `metadata` of a kept part is dropped with a `WARNING`, because a LangChain block has no field for it. |
| `additional_kwargs[AGUI_EXTRAS_KEY]["metadata"]` | `metadata` | both | AG-UI 1.0 declares `metadata` as an open mapping of extra data. Before 1.0 it was an extra field, so it stays in the extras slot. `AGUI_EXTRAS_OPEN_FIELDS` lists each such field. A non-mapping value is dropped outbound, with a `WARNING`. |
| *(none)* | `subagent_run_id` | neither | Declared on AG-UI 1.0 messages. The adapter never reads it inbound and never sets it outbound. |
| `additional_kwargs[AGUI_EXTRAS_KEY]` | *(extra fields)* | both | Passthrough. See below. |
| `additional_kwargs[AGUI_HELD_PARTS_KEY]` | `content` parts with a `file` source | both | Held. See [Parts the model does not receive](#parts-the-model-does-not-receive). |

### Passing extra data on a message

An AG-UI message allows extra fields that the protocol does not declare, for example a retry hint on a failure notice. Put a mapping under the reserved `additional_kwargs` key `AGUI_EXTRAS_KEY`, the literal string `"langgraph_events.agui"`. Each entry becomes a top-level field on the AG-UI message. Use the constant, not the literal.

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

Nothing else in `additional_kwargs` is forwarded. Inbound, `agui_messages_to_langchain` and `detect_new_tool_results` both collect the extra fields of an AG-UI message back into `additional_kwargs[AGUI_EXTRAS_KEY]`.

WARNING: Inbound extras arrive verbatim. They flow through `add_messages` into the checkpoint. They return on the next `MessagesSnapshot` to every client on that thread. The adapter neither filters the keys nor inspects the values. Do not treat a value read from the slot as trusted input. If more than one person can reach a thread, strip or scope the slot yourself. `AGUI_EXTRAS_MAX_BYTES` (8 KiB of JSON per message) bounds how much a client can add. Extras over the cap, or extras that do not encode as JSON, are dropped with a `WARNING` naming the message. The message itself still converts.

`MessagesSnapshot` is rebuilt from the **checkpointed** message list on every `connect()`. If that build raises, the client gets a `RunError` on every `connect()`, and the thread stays broken until someone edits the checkpoint. So nothing in the extras path raises. Four cases drop something and warn once per message class:

| Case | Dropped |
|---|---|
| The reserved key holds a non-mapping | The whole value |
| An entry key is not a string | That entry |
| An entry in `AGUI_EXTRAS_OPEN_FIELDS` holds a non-mapping | That entry |
| An entry addresses another declared AG-UI field (`content`, `toolCallId`, …) | That entry. Otherwise it rewrites protocol data. |

Rename the entry, or set the field through the LangChain message.

### Parts the model does not receive

A user or tool message can carry a media part with a `file` source, for example a `DocumentPart`. AG-UI defines that source as a handle that a provider issued. The server would send the handle with its own credentials, so a client could read a file that it does not own. The model must not receive such a part.

Inbound, `agui_messages_to_langchain` and `detect_new_tool_results` hold the part, with its `metadata` and its position, under the reserved `additional_kwargs` key `AGUI_HELD_PARTS_KEY`. Outbound, `MessagesSnapshot` puts each held part back at its position, on `stream()` and on `connect()`. A host that issues its own file handles reads them with `held_parts()`.

```python
from ag_ui.core import DocumentPart, FileSource, TextPart, UserMessage

from langgraph_events.agui import agui_messages_to_langchain, held_parts

doc = DocumentPart(source=FileSource(value="doc-42", provider="host"))
[message] = agui_messages_to_langchain(
    [UserMessage(id="u1", content=[TextPart(text="Read this."), doc])]
)
message.content  # -> [{"type": "text", "text": "Read this."}]  (the model sees only this)
held_parts(message)  # -> [doc]
# The next MessagesSnapshot sends content=[TextPart(text="Read this."), doc].
```

`content` stays a list of standard blocks. If the only part of a message is held, `content` is `[]`. Some providers refuse a message with empty content. If a client sends only held parts, give the model a text of your own.

Each entry of the slot is `{"position": i, "part": p}`. `i` is the position of the part in the AG-UI content list.

`held_parts(message)` returns the held parts in their original order and never raises, because it reads checkpointed data. It skips an entry that does not validate, with a `WARNING` that names the message and the key. The snapshot drops a bad entry the same way. If a handler made the content shorter, a position past the end puts the part at the end, with a `WARNING` that names the message and the key.

A host can merge messages, for example with LangChain's `merge_message_runs`. The merged message keeps every held entry, but each position still counts in its own original message. So the snapshot can restore a part at a shifted position.

WARNING: A held part is client input. The trust note of the extras slot applies to it.

Each reserved slot has its own size cap, measured as JSON per message:

| Slot | Cap | Over the cap |
|---|---|---|
| `AGUI_EXTRAS_KEY` | `AGUI_EXTRAS_MAX_BYTES`, 8 KiB | The extras are dropped with a `WARNING`. The message still converts. |
| `AGUI_HELD_PARTS_KEY` | `AGUI_EXTRAS_MAX_BYTES`, 8 KiB, counted apart from the extras | The held parts are dropped with a `WARNING`. The model still does not receive them, and the message still converts. |

So one message can store up to 16 KiB in the two slots.

## Connect / Reconnect

`connect()`, alias `reconnect()`, runs no handler. It emits `RunStarted`, then `StateSnapshot` and `MessagesSnapshot` from the checkpoint, then each pending `Interrupted` event, then `RunFinished`. If the replay raises, `RunError` ends the stream instead of `RunFinished`. One exception applies: with `on_unmapped="raise"`, an `UnmappedEventError` propagates to the caller and does not become `RunError`. The stream then ends without a terminal event, the same as in `stream()`. An AG-UI client refuses a stream that does not start with `RunStarted`, so do not strip these events. Call `connect` on page load or refresh. Call `stream` only to start or resume execution. Serve it like the Quick Start route: `create_starlette_response(adapter.connect(input_data))`.

## Custom Mappers

Implement `EventMapper.map()`: return `None` (pass), `[]` (suppress), or AG-UI events.

```python
...
from ag_ui.core import BaseEvent, CustomEvent, EventType
from langgraph_events.agui import AGUIAdapter, EventMapper, MapperContext


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

## Resume Support

Implement `resume_factory(input_data, checkpoint_state)`. Return an `Event` to trigger `astream_resume()`. Return `None` for a fresh run via `seed_factory`. On a thread paused on an `Interrupted`, `None` instead replays the checkpoint. The adapter emits `StateSnapshot`, `MessagesSnapshot` and each pending `Interrupted`. It does not call `seed_factory`. `checkpoint_state` is optional. A one-argument factory still works. A `resume_factory` can be `async` and can refuse the run, see [Async Factories and Refusals](#async-factories-and-refusals).

```python
...
from typing import Any


class ApprovalSubmitted(IntegrationEvent):
    approved: bool


def resume_factory(
    input_data: RunAgentInput,
    checkpoint_state: dict[str, Any] | None,
) -> Event | None:
    # checkpoint_state: CheckpointState TypedDict with reducers, events,
    # messages, pending_interrupts, is_interrupted, snapshot. Each reducer
    # value is a shallow copy, so a change to it does not reach the run.
    state = input_data.state or {}
    if "approved" in state:
        return ApprovalSubmitted(approved=state["approved"])
    return None  # fresh run


adapter = AGUIAdapter(graph, seed_factory=seed_factory, resume_factory=resume_factory)
```

When `RunAgentInput.state` is populated alongside a resume:

- The state dict is projected via `include_reducers`. Every framework channel and the `messages` key are stripped.
- A `FrontendStateMutated(state=projected)` event is built. For each reducer that subscribes to it, the adapter writes via `apre_seed` **before** the resume dispatch, so a handler that injects the channel sees the updated value. The event is also a seed event of `astream_resume`, so it appears in the stream and the audit log.
- **A domain-driven reducer wins for a shared key.** A backend channel stays intact, whatever the client echoes.
- **An `@on(FrontendStateMutated)` handler does NOT fire on resume.** Use `@on(Resumed)` or `@on(Resumed, interrupted=...)` for a resume-time side effect.

A reducer can subscribe to `FrontendStateMutated` to project a selected client-state key on the server. A module-level reducer must set `name=`: `ScalarReducer(name="focus", event_type=FrontendStateMutated, fn=lambda e: e.state.get("focus", SKIP))`.

## Async Factories and Refusals

A `seed_factory` or a `resume_factory` can be `async`. The adapter awaits the result. A sync factory still works. The optional `checkpoint_state` argument works the same way for both kinds.

A factory can refuse the run. Raise `RunRefusedError(code, message)`. The adapter then sends `RUN_STARTED`, then one `RUN_ERROR` with that `code` and that `message`, and nothing after. The graph does not start, so the run stores nothing in the checkpoint. The `error_message` of the adapter does not replace the `message` of a refusal. The adapter logs the `code` of a refusal at `WARNING`, without the `message` and without a traceback.

The example builds on the Quick Start. `KNOWN_FILE_IDS` stands in for your own upload store.

```python
...
from ag_ui.core import RunAgentInput
from langchain_core.messages import HumanMessage

from langgraph_events import Event
from langgraph_events.agui import AGUIAdapter, RunRefusedError

KNOWN_FILE_IDS: set[str] = set()


async def file_exists(file_id: str) -> bool:
    return file_id in KNOWN_FILE_IDS


async def seed_factory(input_data: RunAgentInput) -> list[Event]:
    last_msg = input_data.messages[-1]
    if not await file_exists(last_msg.id):
        raise RunRefusedError("unknown_file", "The attached file does not exist.")
    return [UserMessageReceived(message=HumanMessage(content=last_msg.content))]


adapter = AGUIAdapter(graph, seed_factory=seed_factory)
```

A client reads two fields of `RUN_ERROR`: `code` and `message`. A refusal always sets `code`. A generic error has no `code`. A client branches on the presence of `code`.

`connect()` calls no factory, so a refusal cannot occur there.

The adapter catches `RunRefusedError` only from the two factories. A handler that raises `RunRefusedError` gets the same treatment as any other exception: a generic `RUN_ERROR` without a `code`. This path logs at `ERROR` with a traceback, and the traceback includes the `message` of the refusal. The graph has run by then, so the run can have stored events.

## Frontend Tools

The AG-UI spec positions tool calls as "inherently frontend-executed" and as the mechanism for human-in-the-loop (HITL) interaction. The adapter wires all three parts to an `EventGraph`:

1. **Tool definitions in.** The `useFrontendTool` registrations of the page arrive as `RunAgentInput.tools`. `build_langchain_tools(...)` converts them to OpenAI-format dicts for `llm.bind_tools(...)`.
2. **Tool calls out.** Two paths, both map to `ToolCallStart`/`ToolCallArgs`/`ToolCallEnd`:
    - **LLM-initiated.** Bound LLM `tool_call_chunks` translate to the streaming triple when the call name is in `RunAgentInput.tools`. A call to a backend tool emits no `ToolCall*` event, is logged once at `DEBUG`, and the text tokens of the same LLM run still stream. The `MessagesSnapshot` history still holds every tool call, because the [spec](https://docs.ag-ui.com/concepts/tools) scopes `RunAgentInput.tools` to execution. If you list a backend tool, the client executes it. Do not list a backend tool to show it in the UI. Emit `ToolCallStart`, `ToolCallArgs`, `ToolCallEnd` and `ToolCallResult` from a custom [`EventMapper`](#custom-mappers) instead.
    - **Handler-initiated.** Return `FrontendToolCallRequested(Interrupted)` from a handler. See [Handler-initiated frontend tools](#handler-initiated-frontend-tools).
3. **Tool results back.** The return value of the frontend handler comes back as a `role: "tool"` message. `detect_new_tool_results(input_data, checkpoint_state)` returns the new `ToolMessage`s. Wrap them in a `MessageEvent` and return it from `resume_factory`.

```python
...  # ToolsRegistered, ToolsExecuted, LLMResponded as in examples/conversation.py
from langgraph_events import on
from langgraph_events.agui import build_langchain_tools, detect_new_tool_results


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

For a backend-initiated tool call, for example a confirm dialog, a file picker or a deterministic prompt, return `FrontendToolCallRequested(Interrupted)` from a handler. The frontend result becomes a typed event on resume. `ApprovalRequired(Interrupted)` in [`examples/expense_approval.py`](https://github.com/cadance-io/langgraph-events/blob/main/examples/expense_approval.py) uses the same machinery.

```python
import json

from langgraph_events import IntegrationEvent, MessageEvent, on
from langgraph_events.agui import FrontendToolCallRequested, detect_new_tool_results


class OrderRequested(IntegrationEvent):
    total: float


class OrderConfirmed(IntegrationEvent, MessageEvent):
    messages: tuple = ()


class OrderPlaced(IntegrationEvent):
    approved: bool


@on(OrderRequested)
def request_confirmation(event: OrderRequested) -> FrontendToolCallRequested:
    return FrontendToolCallRequested(
        name="confirm",  # must match a useFrontendTool({ name: "confirm", ... }) registration
        args={"prompt": f"Place an order for {event.total}?"},
    )


def resume_factory(input_data, checkpoint_state=None):
    results = detect_new_tool_results(input_data, checkpoint_state)
    if not results:
        return None
    return OrderConfirmed(messages=tuple(results))


@on(OrderConfirmed)
def place(event: OrderConfirmed) -> OrderPlaced:
    approved = bool(json.loads(event.messages[0].content).get("approved"))
    return OrderPlaced(approved=approved)
```

- CopilotKit v1 (`useCopilotAction`) uses `MessagesSnapshot`. v2 (`useFrontendTool`) needs the streaming `ToolCallStart/Args/End`. Both coexist. CopilotKit reconciles by `tool_call_id`.
- For an LLM-initiated tool call, `ToolCallStartEvent.parent_message_id` can differ from the final `MessagesSnapshot` id. LangChain does not expose the final `AIMessage.id` until the stream ends.
- If a page refresh calls `connect()` while the graph is paused on `FrontendToolCallRequested`, the adapter replays the triple with the stored `tool_call_id`. CopilotKit `useFrontendTool` is idempotent by `tool_call_id`.
- Malformed tool-call traffic raises `ValueError` at the source. The cases are an empty or whitespace-only `FrontendToolCallRequested.name`, an LLM `tool_call_chunk` without `index`, a first chunk of a streaming call without both `id` and `name`, and an inbound `role: "tool"` message without a non-empty `tool_call_id`. A continuation chunk can omit `id` and `name`. `FrontendToolCallRequested.args` is JSON-serialized at emit time, so a non-serializable value raises `TypeError`. A streaming-path error reaches the frontend as a `RUN_ERROR` event with the diagnostic message.

## Resume Helpers

`detect_new_tool_results` covers the frontend-tool-result path of resume. A result the client marks with `error` becomes a LangChain `ToolMessage` with `status="error"`. In the other path, the frontend sends `Command(resume=…)` plus new chat messages. Three helpers cover that path:

- **`extract_resume_input(input_data)`**: decode `RunAgentInput.forwarded_props["command"]["resume"]`. Every falsy value (`0`, `""`, `[]`, `{}`, `False`) means "no resume". For a "deny" signal, send `{"approved": false}`, not bare `false`.
- **`agui_messages_to_langchain(messages, *, drop_invalid_tool_calls=False)`**: convert AG-UI messages (`UserMessage`, `AssistantMessage`, `SystemMessage`, `ToolMessage`, multimodal content included, see [Message field mapping](#message-field-mapping)) to LangChain `BaseMessage`. `ReasoningMessage` and `DeveloperMessage` are skipped at `DEBUG`. `ActivityMessage` and an unknown role raise `ValueError`. With `drop_invalid_tool_calls=True`, a tool call with unparseable JSON args is dropped with a `WARNING`. A `ToolMessage.error` becomes `status="error"`, and extra fields land under `additional_kwargs[AGUI_EXTRAS_KEY]`. A content part with a `file` source lands under `additional_kwargs[AGUI_HELD_PARTS_KEY]`. Use it directly for custom merging, for example in `seed_factory`.
- **`detect_new_frontend_messages(input_data, checkpoint_state, *, reducer_name="messages", drop_invalid_tool_calls=True)`**: read the stored messages from `checkpoint_state["reducers"][reducer_name]`, then convert only the inbound messages whose id is not stored. Returns a tuple of the new messages only. A stored message wins over an inbound message with the same id, because AG-UI defines `MessagesSnapshot` as the messages that the server owns. If two inbound messages have the same id, the result holds the last copy, at the position of the first copy.

```python
...  # ToolsExecuted as in examples/conversation.py
from langgraph_events import IntegrationEvent, MessageEvent
from langgraph_events.agui import (
    detect_new_frontend_messages,
    detect_new_tool_results,
    extract_resume_input,
)


# Define the resume event alongside your other events.
class UserResumed(IntegrationEvent, MessageEvent):
    response: object  # resume payload, any shape
    messages: tuple = ()


def resume_factory(input_data, checkpoint_state=None):
    tool_results = detect_new_tool_results(input_data, checkpoint_state)
    if tool_results:
        return ToolsExecuted(messages=tuple(tool_results))  # tool-result arm, as above
    resume_input = extract_resume_input(input_data)
    if resume_input is None:
        return None
    new_messages = detect_new_frontend_messages(input_data, checkpoint_state)
    return UserResumed(response=resume_input, messages=new_messages)
```

WARNING: A stored resume event must carry only the new messages. The checkpoint stores each resume event twice, once as the event and once in `Resumed.value`, and each checkpoint load revives every stored event. A resume event that carries the full history grows the checkpoint by the number of resumes times the number of messages, and each turn loads more slowly. The message reducer already holds the full history. To read it in a `resume_factory`, read `checkpoint_state["reducers"]["messages"]`. A handler that reads `event.messages` as the whole conversation sends only the newest messages, for example to an LLM, with no error. Such a handler must inject the messages reducer instead, with a `messages: list[BaseMessage]` parameter. See [`message_reducer`](reducers.md#message_reducer).

## Typed Interrupt Payloads

`InterruptedWithPayload[PayloadT]` builds on [`Interrupted`](control-flow.md#interrupted-resumed). `InterruptedMapper` emits the result of `interrupt_payload()` as `CustomEvent(name="interrupted", value=...)`. No `agui_dict()` override is needed.

```python
from typing import Literal, TypedDict

from langgraph_events import IntegrationEvent, on
from langgraph_events.agui import InterruptedWithPayload


class DraftWritten(IntegrationEvent):
    draft: str
    revision: int


class ReviewPayload(TypedDict):
    kind: Literal["review"]
    draft: str
    revision: int


class ReviewInterrupted(InterruptedWithPayload[ReviewPayload]):
    draft: str
    revision: int

    def interrupt_payload(self) -> ReviewPayload:
        return {"kind": "review", "draft": self.draft, "revision": self.revision}


@on(DraftWritten)
def request_review(event: DraftWritten) -> ReviewInterrupted:
    return ReviewInterrupted(draft=event.draft, revision=event.revision)
```

Plain `Interrupted`, with no payload, remains the right choice for a paused thread without a frontend.

## LangGraph Config Passthrough

`stream()` and `connect()` accept a LangGraph config via `RunAgentInput.forwarded_props`, under the key `langgraph_config`, the key `config`, or as the dict itself. The adapter always overrides `configurable.thread_id` from `RunAgentInput.thread_id`. Other keys, for example `recursion_limit` or tenant routing, pass through.

## Deadline and signal pauses

`stream()` accepts `deadline=`, an absolute `time.monotonic()` reference, and `pause_on=`, an object with `is_set()` such as an `asyncio.Event`. The router emits a `RunPausedOnDeadline` or a `RunPausedOnSignal`, both `RunPaused` subclasses, and the run finalises through the same `RunFinishedEvent` path as a normal completion. See [Control Flow → Deadline pause](control-flow.md#soft-timeout-runpaused) and [Control Flow → Pause on a signal](control-flow.md#pause-on-a-signal-runpausedonsignal) for the semantics and the resume rule.

`RunPaused` does not implement `AGUISerializable`, so `FallbackMapper` skips it with a one-time warning. An app that wants to send a pause signal to the client registers its own mapper:

```python
...
from langgraph_events import RunPaused


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

## AG-UI Spec Coverage

The adapter emits 12 of the 31 AG-UI event types. The other 19 need a custom mapper.

| Category | Count | Event Types |
|---|---|---|
| **Built-in** | 12 | `RunStarted`, `RunFinished`, `RunError`, `TextMessageStart/Content/End`, `ToolCallStart/Args/End`, `StateSnapshot`, `MessagesSnapshot`, `Custom` |
| **Custom mapper** | 19 | `TextMessageChunk`, `ToolCallChunk`, `ToolCallResult`, `StepStarted/Finished`, `StateDelta`, `ActivitySnapshot/Delta`, `Raw`, `ReasoningStart/End`, `ReasoningEncryptedValue`, `ReasoningMessageStart/Content/End/Chunk`, `SubagentStarted/Finished/Error` |
