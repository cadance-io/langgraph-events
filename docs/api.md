# API Reference

## Event Taxonomy

| Export | Type | Description |
|---|---|---|
| `Namespace` | Base class | Namespace for nested commands and outcomes; exposes `__namespace_name__`, `__reducers__` |
| `Command` | Base class | Imperative intent; must be nested inside a `Namespace`. Auto-exposes `.Outcomes` — union of nested `DomainEvent`s. A `handle` method on the class registers as the command's inline handler when passed to `EventGraph` |
| `DomainEvent` | Base class | Fact inside the domain; must be nested inside a `Namespace` or `Command` |
| `IntegrationEvent` | Base class | Cross-boundary fact; top-level |
| `SystemEvent` | Base class | Framework-emitted fact; top-level |
| `InvariantViolated` | Event | Emitted when an `invariants=` predicate returns false |

## Mixins

| Export | Type | Description |
|---|---|---|
| `Auditable` | Mixin | Marker for auto-logged events; `trail()` returns a compact summary. Compose with any event branch |
| `MessageEvent` | Mixin | Wraps LangChain `BaseMessage`; declares `message` or `messages` field. Compose with any event branch |
| `SystemPromptSet` | Event | Built-in `IntegrationEvent` + `MessageEvent` for system prompts |
| `Event` | Internal | Base class (not user-facing). Valid as type annotation and reducer `event_type=` catch-all |

## Handler Subscription

`@on` subscribes a handler. Three shapes:

- Bare `@on` — event type inferred from the first parameter's annotation.
- `@on(kwargs=...)` — inferred, with modifiers.
- `@on(Type, ...)` — explicit; required for multi-event subscription.

Modifiers:

- `*field_matchers` — [field dispatch](control-flow.md#field-matchers); `type` values do `isinstance`, `str` values do equality.
- `raises=` — [declared exceptions](control-flow.md#handler-exceptions).
- `invariants={InvariantClass: predicate}` — [preconditions](control-flow.md#invariants).

Returns enforced against the declared annotation, or the subscribed `Command.Outcomes` when unannotated.

## Graph & Execution

| Export | Type | Description |
|---|---|---|
| `EventGraph` | Class | Build and run the event-driven graph; accepts `@on`-decorated functions and/or `Command` subclasses with inline `handle`. `services=[...]` (type-keyed) or `services={...}` (name-keyed) injects project dependencies into handler params (see [Concepts › Signature injection](concepts.md#signature-injection)) |
| `EventGraph.from_namespaces()` | Classmethod | Build a graph from domains' inline command handlers; `handlers=` appends external handlers |
| `EventGraph.invoke()` / `.ainvoke()` | Method | Run (sync/async); returns `EventLog` |
| `EventGraph.resume()` / `.aresume()` | Method | Resume an interrupted graph (requires checkpointer) |
| `EventGraph.get_state()` | Method | `GraphState` for a checkpointed thread |
| `EventGraph.namespaces()` | Method | Code-derived snapshot — domains, commands, outcomes, handlers, policies, edges, seeds. Returns a `NamespaceModel` |
| `NamespaceModel.text(view=...)` | Method | Human-readable tree; `view="structure"` or `"choreography"` (default) |
| `NamespaceModel.mermaid(namespace_order=..., reactor_hub_min=...)` | Method | Mermaid `graph LR` choreography diagram (handlers, policies, invariants, edges). `namespace_order` is `"affinity"` (default — cluster heavily-connected namespaces adjacent) or `"alphabetical"` (legacy, byte-stable for snapshot-pinned consumers). `reactor_hub_min=N` opts in to hub-style fanout: any `(source, handler)` producing ≥N solid-or-scatter targets renders as `Source → Hub → {targets}` with the handler name on the hub instead of repeated on every edge. For a structure-only view use `text(view="structure")`. |
| `NamespaceModel.json()` / `.to_dict()` | Method | JSON-serializable snapshot (event classes encoded as qualnames). Top-level `schema_version` is bumped on field removal, rename, or semantic change; additions don't bump. |
| `NamespaceModel.{Namespace, Command, CommandHandler, Policy, Edge, Invariant}` | Nested dataclasses | Frozen dataclasses for programmatic access |
| `NamespaceModel.invariants` | Field | Tuple of `NamespaceModel.Invariant` — every declared invariant with `cls`, `commands`, `declared_by`, `reactors` |
| `EventGraph.compiled` | Property | Underlying `CompiledStateGraph` escape hatch |
| `EventGraph.reducer_names` | Property | `frozenset` of registered reducer names |
| `EventLog` | Class | Immutable query container (see [Concepts](concepts.md#eventlog)) |
| `GraphState` | NamedTuple | `(events, is_interrupted, interrupted)` |

## System Events

| Export | Type | Description |
|---|---|---|
| `Halted` | Event | Signal immediate termination; subclass for domain-specific halts |
| `MaxRoundsExceeded` | Event | `Halted` subtype when `max_rounds` is exceeded |
| `Cancelled` | Event | `Halted` subtype when an async handler is cancelled |
| `Interrupted` | Base class | Subclass with typed fields to pause for human input. For frontend-discriminated payloads see `InterruptedWithPayload` in `langgraph_events.agui` |
| `Resumed` | Event | Emitted on `resume()` with the dispatched event + `interrupted` backref |
| `HandlerRaised` | Event | Emitted when a handler raises a `raises=`-declared exception; carries `handler`, `source_event`, `exception` |
| `Invariant` | Marker class | Subclass to declare a typed invariant; used as a dict key in `invariants=` and as the matcher value in `@on(InvariantViolated, invariant=...)` |
| `InvariantViolated` | Event | Emitted when an `invariants=` predicate returns false; carries `invariant` (instance of the declared `Invariant` subclass), `handler`, `source_event`. Use `@on(InvariantViolated, invariant=SomeInvariant)` to pin to a specific invariant |
| `Scatter` | Class | Fan-out into multiple events; `Scatter[T]` annotates the produced type |

## Reducers

| Export | Type | Description |
|---|---|---|
| `Reducer` | Class | List accumulator; declare as `Namespace` class attribute or pass via `reducers=` |
| `ScalarReducer` | Class | Last-write-wins for a single value; `None` is valid |
| `SKIP` | Sentinel | Return from `ScalarReducer.fn` to leave the value unchanged |
| `message_reducer` | Function | Built-in reducer for `MessageEvent` projection |

## Lifecycle Hooks

| Export | Type | Description |
|---|---|---|
| `on_namespace_finalize(cls, callback)` | Function | Schedule `callback(cls, namespace_cls)` to fire once `cls`'s enclosing `Namespace.__init_subclass__` finishes. Lets class decorators defer `typing.get_type_hints()` calls until forward references to siblings can resolve. Fires immediately if the enclosing Namespace has already finalized (e.g. when applied post-hoc to a bound class). |

## Streaming & Frames

| Export | Type | Description |
|---|---|---|
| `EventGraph.stream_events()` / `.astream_events()` | Method | Yield events as produced; flags below control frame types |
| `EventGraph.stream_resume()` / `.astream_resume()` | Method | Streaming equivalent of `resume()` |
| `emit_custom` / `aemit_custom` | Function | Emit a custom stream event from a handler |
| `emit_state_snapshot` / `aemit_state_snapshot` | Function | Emit a typed state snapshot frame |
| `STATE_SNAPSHOT_EVENT_NAME` | Constant | Protocol name for snapshot events (`"intermediate_state"`) |

Stream frame types live in the `langgraph_events.stream` submodule:

```python
from langgraph_events.stream import (
    StreamFrame,           # (event, reducers, changed_reducers) — include_reducers=True
    LLMToken, LLMStreamEnd,  # LLM token delta / completion — include_llm_tokens=True
    CustomEventFrame, StateSnapshotFrame,  # include_custom_events=True
)
```

## Warnings

| Export | Type | Description |
|---|---|---|
| `OrphanedEventWarning` | Warning | Issued at graph construction when a return type has no subscriber |

## AG-UI Subpackage

Requires `[agui]`. See [AG-UI Adapter](agui.md).

| Export | Type | Description |
|---|---|---|
| `AGUIAdapter` | Class | Map `EventGraph` streams to AG-UI protocol events |
| `AGUIAdapter.stream()` / `.connect()` / `.reconnect()` | Method | Execute / rehydrate checkpoint / refresh |
| `FrontendStateMutated` | Event | Adapter-emitted `IntegrationEvent` carrying `RunAgentInput.state` for client-state-mirror reducers |
| `FrontendToolCallRequested` | Event | `Interrupted` subclass for handler-initiated frontend tool calls (CopilotKit `useFrontendTool`) |
| `InterruptedWithPayload[PayloadT]` | Generic base | Typed-payload variant of `Interrupted`; subclasses implement `interrupt_payload(self) -> PayloadT`. The built-in `InterruptedMapper` recognizes it directly — no `agui_dict()` override needed |
| `AGUICustomEvent` / `AGUISerializable` | Protocol | Override fallback event names / payload serialization |
| `SeedFactory` / `ResumeFactory` / `EventMapper` | Protocol | Input → seed events; resume detection; event mapping |
| `MapperContext` | Dataclass | Shared state (run ID, thread ID, message counter) per stream |
| `create_starlette_response` / `encode_sse_stream` | Function | Framework-agnostic SSE wrapping |

## Serde Subpackage

Opt-in subpackage for namespace-aware checkpoint serialization. Use when nested events with sibling-named outcomes (`Persona.Approve.Approved`, `Story.Approve.Approved`) need to round-trip distinctly through a checkpointer.

| Export | Type | Description |
|---|---|---|
| `NamespaceAwareSerde` | Class | `JsonPlusSerializer` subclass that keys `Event` identity by `(__module__, __qualname__)` rather than `(__module__, __name__)`. Drop-in for any LangGraph checkpointer that accepts `serde=` (e.g. `MemorySaver(serde=NamespaceAwareSerde())`). Non-event payloads encode exactly as the default serde |
