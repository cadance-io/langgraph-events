# API Reference

## Event Taxonomy

| Export | Type | Description |
|---|---|---|
| `Namespace` | Base class | Namespace for nested commands and outcomes; exposes `__namespace_name__`, `__reducers__`. Names are unique per *graph*, not per process — see [namespace scope](concepts.md#namespace-scope) |
| `Command` | Base class | Imperative intent; must be nested inside a `Namespace`, and may not itself be subclassed once declared (`class Rush(Order.Place)` is a `TypeError`). Auto-exposes `.Outcomes` — union of nested `DomainEvent`s. A `handle` method on the class registers as the command's inline handler when passed to `EventGraph` |
| `DomainEvent` | Base class | Fact inside the domain; must be nested inside a `Namespace` or `Command` |
| `IntegrationEvent` | Base class | Cross-boundary fact; top-level |
| `SystemEvent` | Base class | Framework-emitted fact; top-level |
| `InvariantViolated` | Event | Emitted when an `invariants=` predicate returns false |

## Mixins

| Export | Type | Description |
|---|---|---|
| `MessageEvent` | Mixin | Wraps LangChain `BaseMessage`; declares `message` or `messages` field. Compose with any event branch |
| `SystemPromptSet` | Event | Built-in `IntegrationEvent` + `MessageEvent` for system prompts |
| `Event` | Base class | Do not subclass directly. Valid as type annotation, `@on(Event)` catch-all and reducer `event_type=` catch-all. |

## Handler Subscription

`@on` subscribes a handler. Three shapes:

- Bare `@on` — event type inferred from the first parameter's annotation.
- `@on(kwargs=...)` — inferred, with modifiers.
- `@on(Type, ...)` — explicit; required for multi-event subscription.

Modifiers:

- `*field_matchers` — [field dispatch](control-flow.md#field-matchers); `type` values do `isinstance`, `str` values do equality.
- `raises=` — [declared exceptions](control-flow.md#handler-exceptions).
- `retry=RetryPolicy(...)` — [declarative backoff](control-flow.md#retries) around the handler call. Spelled as a `retry` class attribute on a `Command`, alongside `raises`/`invariants`.
- `invariants={InvariantClass: predicate}` — [preconditions](control-flow.md#invariants).
- `node_name=` — pin a **stable node identity** (defaults to the function name) so renaming the function never breaks an interrupted checkpoint. `previously=` (str or tuple) — historic node names to keep resumable after a rename. These, and `retry=`, are reserved (a field named `node_name`/`previously`/`retry` can't be matched via `**field_matchers`; a `Command` declaring an annotated `retry` field also fails at class creation — use `retry: ClassVar = ...` for the policy, or rename the field). Inline `Command.handle()` handlers default their node identity to the **command's `__qualname__`** (e.g. `Order.Place`), not the method name, so it is stable regardless of `handlers=[...]` order; a renamed command declares its historic node names as a class attribute, `previously: ClassVar = (...)`, mirroring `raises`/`invariants` in spelling. All four are read off the command class that declares them — a concrete `Command` may not be subclassed, so a historic node name still identifies exactly one class. See [Handler renames](event-migrations.md#handler-renames).

Returns enforced against the declared annotation, or the subscribed `Command.Outcomes` when unannotated. Omitting the return annotation is legal. An annotation that does not resolve at run time is not, and raises `TypeError` at graph construction. So does a member that is not an `Event` subclass, `Scatter[...]` or `None` — see [Concepts › Return contract](concepts.md#return-contract).

## Graph & Execution

| Export | Type | Description |
|---|---|---|
| `EventGraph` | Class | Build and run the event-driven graph; accepts `@on`-decorated functions and/or `Command` subclasses with inline `handle`. `services=[...]` (type-keyed) or `services={...}` (name-keyed) injects project dependencies into handler params (see [Concepts › Signature injection](concepts.md#signature-injection)). Type-keyed matching needs an annotation that resolves at run time; name-keyed matching needs no annotation, but an annotation that is present is checked against the registered value at graph build. `on_unresumable="raise"\|"halt"\|"warn"` (default `"raise"`) governs `resume()` on a thread not awaiting input |
| `RunScoped` | Class | Marker for a service derived per run. Place `RunScoped(factory)` in the name-keyed `services={...}` mapping. The factory is called with the node's `RunnableConfig` once per node call. Its result is injected under the handler's parameter name. The factory must have a return annotation. The framework checks it against the handler parameter at graph build (see [Concepts › Run-scoped services](concepts.md#run-scoped-services)) |
| `EventGraph.from_namespaces()` | Classmethod | Build a graph from domains' inline command handlers; `handlers=` appends external handlers. With `checkpointer=MemorySaver()` auto-wires a `NamespaceAwareSerde` scoped to the passed namespaces and auto-collects every `@migrate_from` / `@backfill` decorator on those classes. Opt out by passing `MemorySaver(serde=<custom>)` — a user-supplied serde always wins |
| `EventGraph.invoke()` / `.ainvoke()` | Method | Run (sync/async); returns `EventLog` |
| `EventGraph.resume()` / `.aresume()` | Method | Resume an interrupted graph (requires checkpointer). The resume value must be an `Event`. Another value raises `TypeError` before the graph runs, and the thread stays paused |
| `EventGraph.abandon()` / `.aabandon()` | Method | Settles a paused thread without answering its interrupt. `require_interrupt=True` (default) raises `ValueError` if the thread has no pending interrupt. Pass `False` to settle it anyway, recording `discarded=""`. Requires a checkpointer. Raises `ValueError` if the thread has no events to settle (never run, or only `pre_seed()`ed). Returns `None`. See [Ending a pause without answering it](control-flow.md#ending-a-pause-without-answering-it-abandon) |
| `EventGraph.threads_paused_on()` / `.athreads_paused_on()` | Method | Configs for each listed thread whose latest checkpoint has a pending interrupt, optionally filtered to an `Interrupted` class or subclass. `thread_ids=` is required: the caller names the threads, and the library does not walk the store. Get the ids from a server-side query. With `event_type=None`, returns every listed thread that is paused. `thread_ids=None` or a bare `str` raises `TypeError`. Requires a checkpointer. See [Finding candidates server-side](event-migrations.md#finding-candidates-server-side) and [Ending a pause without answering it](control-flow.md#ending-a-pause-without-answering-it-abandon) |
| `EventGraph.unrevivable_threads()` / `.aunrevivable_threads()` | Method | Mapping of thread id to the qualnames its latest checkpoint can no longer revive, wherever the serde met them: settled history, pending interrupt, completed sibling write, or a field nested in a live event. Empty when every listed thread revives. Reads the store, not the baseline, so it sees a settled thread the coverage gates cannot. `thread_ids=` is required, the same as for `threads_paused_on()`. To sweep every thread, get the ids from one `SELECT DISTINCT thread_id FROM checkpoints WHERE checkpoint_ns = ''` query. If one checkpointer stores more than one graph, keep only the thread ids of this graph. Thread ids keep the caller's order, without duplicates. Requires a checkpointer with a `NamespaceAwareSerde`, raises `ValueError` otherwise. See [Retiring an Interrupted subclass](event-migrations.md#retiring-an-interrupted-subclass) |
| `EventGraph.plan_rewrite()` / `.aplan_rewrite()` | Method | `RewriteReport` of the rewrite each listed thread needs, after the same verification as `rewrite_store()`. Writes nothing. `drop=` names event classes whose stored instances leave the log. `thread_ids=` is required, the same as for `threads_paused_on()`. To sweep every thread, get the ids from one `SELECT DISTINCT thread_id FROM checkpoints WHERE checkpoint_ns = ''` query. If one checkpointer stores more than one graph, keep only the thread ids of this graph. A listed id with no checkpoint is reported as refused. `checkpoint_ns=` (default `""`, the root namespace) reads a subgraph namespace. Requires a checkpointer with a `NamespaceAwareSerde` and `legacy_write=False`, raises `ValueError` otherwise. See [Rewriting the live set](event-migrations.md#rewriting-the-live-set) |
| `EventGraph.rewrite_store()` / `.arewrite_store()` | Method | Rewrites the latest checkpoint of each listed thread under the live migration table, through the checkpointer's `put()` under the same checkpoint id, and drops the stored instances of each `drop=` class from `events` and `_pending`. It filters `causes` at the same positions and remaps each source. A cause whose source was dropped becomes `SourceDropped(via, source_type)`. Same arguments and report as `plan_rewrite()`. A refused thread is not written; its `reason` says what to do. After each write, the thread is planned again. A thread that still needs a rewrite is refused: the saver did not overwrite the checkpoint. See [Checkpointers](event-migrations.md#checkpointers). Run while the graph is idle, never inside a rolling deploy window, after a store-level backup. See [Rewriting the live set](event-migrations.md#rewriting-the-live-set) |
| `EventGraph.get_state()` | Method | `GraphState` for a checkpointed thread |
| `EventGraph.namespaces()` | Method | Code-derived snapshot — domains, commands, outcomes, handlers, policies, edges, seeds. Returns a `NamespaceModel` |
| `NamespaceModel.text(view=...)` | Method | Human-readable tree; `view="structure"` or `"choreography"` (default) |
| `NamespaceModel.mermaid(namespace_order=..., reactor_hub_min=..., focus=..., show_raises=..., notes=..., muted=...)` | Method | Mermaid `graph LR` choreography diagram (handlers, policies, invariants, edges). `namespace_order` is `"affinity"` (default — cluster heavily-connected namespaces adjacent) or `"alphabetical"` (legacy, byte-stable for snapshot-pinned consumers). `reactor_hub_min=N` opts in to hub-style fanout: any `(source, handler)` producing ≥N solid-or-scatter targets renders as `Source → Hub → {targets}` with the handler name on the hub instead of repeated on every edge. `focus=NamespaceModel.Focus(...)` draws the selected items, every edge that touches one, the nodes at both ends dimmed as context, and every other edge between two drawn nodes; a selected reaction's solid edges are thick arrows, and a focused diagram draws no entry arrows. `show_raises=False` hides the `(raises)` edges. `notes={key: text}` adds lines under a label, and `muted=[key, ...]` fades what it names: a key is an event qualname, a reducer name, or a reaction name (its edges), so every `Cause.via` is a valid key. A wrong key raises `ValueError` that lists the valid keys. For a structure-only view use `text(view="structure")`. |
| `NamespaceModel.json()` / `.to_dict()` | Method | JSON-serializable snapshot (event classes encoded as qualnames). Top-level `schema_version` is bumped on field removal, rename, or semantic change; additions don't bump. |
| `NamespaceModel.{Namespace, Command, CommandHandler, Policy, Edge, Invariant}` | Nested dataclasses | Frozen dataclasses for programmatic access |
| `NamespaceModel.Edge` | Nested dataclass | `source`, `via`, `target`, `kind` (*how* produced: `solid`/`scatter`/`raises`/`retry`/`framework`), `causation` (*causal role*: `intent`/`react`/`orchestrate`/`chain`/`None`) |
| `NamespaceModel.reducers` | Field | Tuple of `NamespaceModel.Reducer` — every reducer with `name`, `subscribes` and `namespace`. `subscribes` holds the concrete events of the model that the reducer folds, resolved from `event_type` (unlike `Policy.subscribes`, which holds the declared types), or `None` when a `runtime_checkable` Protocol with data members makes the folds unknown. `namespace` is a name, or `None` for a reducer passed to `EventGraph(reducers=...)` |
| `NamespaceModel.Focus` | Nested dataclass | The part of the graph that `mermaid(focus=...)` draws. Fields `namespaces`, `reactions`, `reducers` each take one name or an iterable of names, stored sorted. At least one name is required: an empty `Focus()` raises `ValueError`. Valid names are `list(model.namespaces)`, `[r.name for r in model.reactions]` and `[r.name for r in model.reducers]` |
| `NamespaceModel.invariants` | Field | Tuple of `NamespaceModel.Invariant` — every declared invariant with `cls`, `commands`, `declared_by`, `reactors` |
| `EventGraph.compiled` | Property | Underlying `CompiledStateGraph` escape hatch |
| `EventGraph.reducer_names` | Property | `frozenset` of registered reducer names |
| `EventGraph.resumable_node_names` | Property | `frozenset` of every node name a paused checkpoint can resume at: each canonical handler name and each `@on(previously=...)` alias. The gate's `handlers` rule reads it |
| `EventGraph.reflect(log)` | Method | Deterministic query surface over a run — returns a `Reflection` (see [Reflection](reflection.md)) |
| `Reflection` | Class | Facts-only read-model: `context()`, `tool()`, `overview()`, `event(i)`, `evidence(i)`, `schema()`, `state()`, `.log`. Injectable into handlers by parameter annotation, like `EventLog` |
| `QueryTool` | Frozen dataclass | The `query_log` LLM tool: `name` / `description` / `parameters` (JSON Schema) / `run(...) -> str`. Maps 1:1 to Anthropic / LangChain tool shapes |
| `EventLog` | Class | Immutable query container (see [Concepts](concepts.md#eventlog)). `cause(e)`, `effects(e)` and `flow(e)` read the causes that the log records. `causes` is the tuple of causes aligned with `events`, or `None` when the log records no causes. `EventLog(events, causes=...)` builds a log with causes, and `EventLog(log.events, causes=log.causes)` rebuilds a root log. A log from `after()`, `before()` or `select()` answers like its root. `latest(T, not_followed_by=U)` returns `None` when a `U` comes after the most recent `T`. See [A request that is still open](concepts.md#a-request-that-is-still-open) |
| `Cause` | Frozen dataclass | `(source, via)`: the event that a handler received, and the handler's graph node name. Event stores call `source` the causation ID. `Cause.source` is an event, while `Edge.source` is a type. `via` equals `Edge.via` unless the handler has a stable identity: an inline command handler (`via` is the command qualname), or an `@on(node_name=...)` pin. `via` must be a `str`. Returned by `EventLog.cause()` |
| `UnknownCause` | Class | Base of every case where the cause of an event is unknown. `reason` states why in one sentence. A cause is unknown only for history that the framework did not record |
| `NotRecorded` | Frozen dataclass | `UnknownCause`: the event was written before causes existed, in a checkpoint saved by an older release |
| `SourceDropped` | Frozen dataclass | `UnknownCause` `(via, source_type)`: a handler produced the event, but `rewrite_store(drop=...)` deleted its source. The handler and the qualname of the deleted event stay known |
| `FrameworkEvent` | Frozen dataclass | The framework wrote the event: `RunPaused`, `MaxRoundsExceeded`, `Cancelled` or `Abandoned`. `EventLog.cause()` returns `None` only for a seed |
| `GraphState` | NamedTuple | `(events, is_interrupted, interrupted)`. `events` carries the recorded causes of the thread |
| `RewriteReport` | Dataclass | `(applied, threads)` from `plan_rewrite()` / `rewrite_store()`. `applied` is `False` from the plan. `.refused` is the tuple of refused threads. `str(report)` prints one summary line, then one line per refused thread. Each `ThreadRewrite` has `.checkpoint_ns`, and `.location` names the thread and any subgraph namespace |
| `ThreadRewrite` | Dataclass | One thread's line in a `RewriteReport`: `thread_id`, `status` (`"rewrite"`, `"unchanged"` or `"refused"`), `migrated` as `(stored, live)` identity pairs, `dropped` counts per identity, and `reason` on a refusal |

## System Events

| Export | Type | Description |
|---|---|---|
| `Halted` | Event | Signal immediate termination; subclass for domain-specific halts |
| `MaxRoundsExceeded` | Event | `Halted` subtype when `max_rounds` is exceeded |
| `Cancelled` | Event | `Halted` subtype when an async handler is cancelled |
| `Abandoned` | Event | `Halted` subtype emitted by `EventGraph.abandon()` or `.aabandon()`. `.reason` is the caller-supplied reason. `.discarded` holds the discarded interrupts' **qualnames** (e.g. `Order.ApprovalRequested`, not the bare `ApprovalRequested`). The qualname stays unambiguous under nesting and stable whether or not the class still imports. Values are deduped and joined with `", "` (`""` if none). Match with `in` or `.split(", ")`, never `==`. Recorded, not dispatched: `@on(Abandoned)` never fires |
| `Unresumable` | Event | `Halted` subtype emitted by `EventGraph(on_unresumable="halt")` when `resume()` hits a thread not awaiting input. `.resume_value` is the resume event's type name. See [Handler renames](event-migrations.md#handler-renames). For a deliberate settle instead of this runtime net, see [`abandon()`](control-flow.md#ending-a-pause-without-answering-it-abandon) |
| `RunPaused` | Base class | `SystemEvent` (not `Halted`) base of every soft pause the router emits between dispatch rounds. It names no cause, so building it directly raises `TypeError`. Match it to see every pause. Emitted at most once per `/run` regardless of how many parallel handlers fan in past the deadline. Cursor is advanced past it so a fresh `/run` on the same `thread_id` continues normally. No default AG-UI wire mapping — register a custom `EventMapper` to surface it on the wire |
| `RunPausedOnDeadline` | Event | `RunPaused` subtype emitted when a per-call `deadline=monotonic()+budget` expires between dispatch rounds |
| `RunPausedOnSignal` | Event | `RunPaused` subtype emitted when the per-call `pause_on=` signal reports `is_set()` between dispatch rounds. Same resume semantics and once-per-`/run` gate as every `RunPaused`. When the deadline has also expired, the router emits this event, not `RunPausedOnDeadline`. See [Pause on a signal](control-flow.md#pause-on-a-signal-runpausedonsignal) |
| `PauseSignal` | Protocol | The `pause_on=` contract: an object with `is_set() -> bool`. `asyncio.Event` and `threading.Event` satisfy it. A class, or a value with no callable `is_set`, raises `TypeError` before any round runs. Examples are `event.is_set`, `True` and `asyncio.Event`. Every `EventGraph` entry point declares `pause_on: PauseSignal \| None`, so a type checker rejects them too |
| `Interrupted` | Base class | Subclass with typed fields to pause for human input. For frontend-discriminated payloads see `InterruptedWithPayload` in `langgraph_events.agui` |
| `Resumed` | Event | Emitted on `resume()` with the dispatched event + `interrupted` backref |
| `HandlerRaised` | Event | Emitted when a handler raises a `raises=`-declared exception; carries `handler`, `source_event`, `exception`. The exception keeps its traceback — this is the terminal failure, the one you debug from. With a `retry=` policy a plain `HandlerRaised` fires after the budget is spent. A policy that gives up with attempts left emits a `HandlerAbandoned` subclass instead |
| `HandlerAbandoned` | Base class | `HandlerRaised` subtype. A `RetryPolicy` had attempts left, and the run's deadline or pause signal refused the next wait. Catch it to see every early give-up. It names no cause, so building it directly raises `TypeError` |
| `HandlerAbandonedOnDeadline` | Event | `HandlerAbandoned` subtype emitted when the next backoff would land on or past the run's `deadline=`. The run continues, so `@on(HandlerRaised, exception=X)` catchers run for it |
| `HandlerAbandonedOnSignal` | Event | `HandlerAbandoned` subtype emitted when the run's `pause_on=` signal is set before the next backoff wait. It stays in the log, and no catcher runs on it. `@on(HandlerRaised, exception=X)` matches it. See [Pause on a signal](control-flow.md#pause-on-a-signal-runpausedonsignal) |
| `HandlerRetried` | Event | Emitted before each `retry=` backoff wait; carries `handler`, `source_event`, `exception`, `attempt` (1-based, the call that failed), `delay_seconds`. The exception instance is live (matchers and field injection work) but its `__traceback__` is detached — as are those of its `__cause__`/`__context__` chain and any `ExceptionGroup` members — so a breadcrumb never pins the failed attempt's frame; see [Retries](control-flow.md#retries). A deadline or signal give-up emits none, since no wait happened. Suppress with `RetryPolicy(observe="log"\|"silent")` |
| `RetryPolicy` | Class | Declarative retry for a handler call: `max_attempts` (counts the initial call), `base_delay`, `max_delay`, `strategy` (`"exponential"`/`"constant"`), `jitter` (full jitter, default `True`), `on=` (must overlap `raises=`, in either direction), `respect_retry_after`, `observe`. Distinct from `langgraph.types.RetryPolicy`, which re-runs a whole node |
| `Invariant` | Marker class | Subclass to declare a typed invariant; used as a dict key in `invariants=` and as the matcher value in `@on(InvariantViolated, invariant=...)` |
| `InvariantViolated` | Event | Emitted when an `invariants=` predicate returns false; carries `invariant` (instance of the declared `Invariant` subclass), `handler`, `source_event`. Use `@on(InvariantViolated, invariant=SomeInvariant)` to pin to a specific invariant |
| `Scatter` | Class | Fan-out into multiple events. Return type must enumerate concrete events (`Scatter[WorkItem]` or `Scatter[A \| B]`); bare `Scatter`, `Scatter[Any]`, `Scatter[Event]`, and `Scatter[T]` (TypeVar) are rejected at graph construction with `TypeError` |

## Reducers

| Export | Type | Description |
|---|---|---|
| `Reducer` | Class | List accumulator; declare as `Namespace` class attribute or pass via `reducers=` |
| `ScalarReducer` | Class | Last-write-wins for a single value; `None` is valid |
| `FoldReducer` | Class | Folds each event into accumulating state via `fold(self, state)`; next value depends on the prior. Generic over state type `S` |
| `Foldable` | Protocol | `@runtime_checkable` structural type for events with a `fold` method; types `FoldReducer`'s event arg (don't inherit it) |
| `BaseReducer` | Class | Abstract base for custom reducers — subclass for bespoke channels |
| `SKIP` | Sentinel | Return from `ScalarReducer.fn` (or a `FoldReducer.fold`) to leave the value unchanged |
| `RESET` | Sentinel | Return from a `FoldReducer.fold` to clear the channel back to `default_factory()` |
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
    InterruptFrame,  # pending interrupt — include_llm_tokens or include_custom_events
)
```

## Warnings

| Export | Type | Description |
|---|---|---|
| `OrphanedEventWarning` | Warning | Issued at graph construction when a return type has no subscriber |
| `UnresumableError` | RuntimeError subclass | Raised by `resume()` (default `EventGraph(on_unresumable="raise")`) when the thread is not awaiting input. A thread awaits input while its checkpoint has a scheduled node or a pending interrupt of a registered handler. Causes: paused handler renamed/removed, thread already finished, double-resume, or a thread already settled by [`abandon()`](control-flow.md#ending-a-pause-without-answering-it-abandon). Declare `@on(previously=...)` to recover, or set `on_unresumable="halt"`/`"warn"`. See [Handler renames](event-migrations.md#handler-renames) |
| `DomainPatternWarning` | Warning | A namespace has 2+ events fanning out to an identical target set — a missing shared abstraction |
| `CommandChainWarning` | Warning | An inline `Command.handle()` emits another `Command` (a `chain`-causation edge) — prefer emitting a fact and reacting to it |

## Errors

| Export | Type | Description |
|---|---|---|
| `CommandPrivacyError` | TypeError subclass | Raised at `EventGraph` construction; also at runtime when a handler with a broad base-class annotation (e.g. `-> DomainEvent`) constructs a `Cmd.Private(...)` that bypasses the static contract check. Outcomes nested inside a `Command` are private to that Command's inline `handle()` — neither sibling Commands nor non-inline reactors may produce them |
| `ReducerNotSetError` | ValueError subclass | Raised at injection time (before the handler body runs) when a handler parameter rejects `None` (e.g. `strategy: str`) and the channel value is `None`. Sits outside the `raises=` catch boundary — a broad `raises=ValueError` cannot swallow it. Opt out by widening the annotation to `str | None`, `Any`, `object`, or omitting it |

## AG-UI Subpackage

Requires `[agui]`. See [AG-UI Adapter](agui.md).

| Export | Type | Description |
|---|---|---|
| `AGUIAdapter` | Class | Map `EventGraph` streams to AG-UI protocol events |
| `AGUI_EXTRAS_KEY` | Constant | Reserved `BaseMessage.additional_kwargs` key. Its mapping is forwarded as extra fields on the AG-UI message, and receives the client's extra fields on the way back |
| `AGUIAdapter.stream()` / `.connect()` / `.reconnect()` | Method | Execute / rehydrate checkpoint / refresh |
| `FrontendStateMutated` | Event | Adapter-emitted `IntegrationEvent` carrying `RunAgentInput.state` for client-state-mirror reducers |
| `FrontendToolCallRequested` | Event | `Interrupted` subclass for handler-initiated frontend tool calls (CopilotKit `useFrontendTool`) |
| `InterruptedWithPayload[PayloadT]` | Generic base | Typed-payload variant of `Interrupted`; subclasses implement `interrupt_payload(self) -> PayloadT`. The built-in `InterruptedMapper` recognizes it directly — no `agui_dict()` override needed |
| `AGUICustomEvent` / `AGUISerializable` | Protocol | Override fallback event names / payload serialization |
| `SeedFactory` / `ResumeFactory` / `EventMapper` | Protocol | Input → seed events; resume detection; event mapping |
| `MapperContext` | Dataclass | Shared state (run ID, thread ID, message counter) per stream |
| `create_starlette_response` / `encode_sse_stream` | Function | Framework-agnostic SSE wrapping |
| `extract_resume_input` | Function | Pulls and JSON-decodes `RunAgentInput.forwarded_props["command"]["resume"]`. Returns `None` if absent or falsy |
| `agui_messages_to_langchain` | Function | Converts AG-UI protocol messages to LangChain `BaseMessage` instances. Maps `ToolMessage.error` to `status="error"` and extra fields to `additional_kwargs[AGUI_EXTRAS_KEY]`. Optional `drop_invalid_tool_calls=True` to skip tool calls whose JSON arguments fail to parse |
| `detect_new_frontend_messages` | Function | `resume_factory` helper: reads the stored messages from `checkpoint_state["reducers"][reducer_name]`, then converts only the inbound messages whose id is not stored. Returns a tuple of the new messages only. A stored message wins over an inbound message with the same id |
| `build_langchain_tools` / `detect_new_tool_results` | Function | Convert `RunAgentInput.tools` to OpenAI-format / extract new `ToolMessage`s from a resume request |

## Serde Subpackage

Namespace-aware serialization, auto-wired by default. `EventGraph.from_namespaces(..., checkpointer=MemorySaver())` builds a `NamespaceAwareSerde` scoped to the passed namespaces and auto-collects every `@migrate_from` / `@backfill` on those classes. Opt out by passing `MemorySaver(serde=<custom>)`. See [Event migrations](event-migrations.md).

Exported from `langgraph_events.serde` unless a different module is noted.

| Export | Type | Description |
|---|---|---|
| `NamespaceAwareSerde` | Class | `JsonPlusSerializer` subclass keying `Event` identity by `(__module__, __qualname__)`. Drop-in for any checkpointer accepting `serde=`. `EventGraph.from_namespaces` auto-wires one scoped to its namespaces when the checkpointer carries the default serde; pass your own via `MemorySaver(serde=...)` to opt out. Kwargs: `migrations=`, `namespaces=`, `legacy_write=`. `legacy_write=True` is refused at construction with an origin-scoped fill, any transform or any split |
| `migrate_from` | Decorator | `@migrate_from(*old_qualnames, in_module=None, backfill=None, transform=None)` — records historic identities a class held. `backfill={field: default}` scopes fills to payloads written under *this* decorator's qualname (one qualname per decorator) — the fan-in case where collapsed origins each pin their own discriminator. `transform=fn` rewrites the stored kwargs of payloads written under this decorator's qualname only, before the rename (one qualname per decorator). A module-level `transform` that returns `{}` retires an identity onto an empty tombstone. Auto-collected from the namespaces the serde was built with; stacks; not MRO-inherited |
| `backfill` | Decorator | `@backfill(field, *, default=… \| default_factory=…)` — class-global back-fill for a field that is required in code but absent from pre-existing payloads (one value for every origin and era; acts as the fallback under origin-scoped fills). Same auto-collection as `@migrate_from`; reuses the `AddField` mutable-default guard |
| `transform_fields` | Decorator | `@transform_fields(fn)` — the class stage: rewrites the stored kwargs for a field the class dropped, merged or retyped. `fn` receives a copy of the kwargs and returns the kwargs the constructor receives. Runs after any rename, on payloads from every era, so it must be idempotent. One per class. `migrate_from(transform=fn)` is the origin stage. Same auto-collection as `@backfill`. See [Dropping, merging or retyping a field](event-migrations.md#dropping-merging-or-retyping-a-field) |
| `split_event` | Decorator | `@split_event(select, *, targets=(...))` — applied to the source class, whose identity stays stored. `select` receives a copy of the stored kwargs and returns `None` to keep the source, or `(target_class, kwargs)` to build one of `targets` instead. Runs in the class stage, after any rename and `@transform_fields`, before the fills of the resulting identity. `targets` is validated at serde construction. One per class. Same auto-collection as `@transform_fields`. See [Splitting one stored event into two on a payload value](event-migrations.md#splitting-one-stored-event-into-two-on-a-payload-value) |
| `Migration` | Class | Hand-authored migration group for cross-module / composite cases. `Migration.rename(...)` / `Migration.add_field(...)` / `Migration.transform_fields(...)` / `Migration.split_event(...)` are single-op sugar; pass the live target as `to=<class>` / `target=<class>` / `source=<class>` (refactor-safe) instead of `new_module`/`new_qualname` strings. `add_field` or `transform_fields` keyed on a historic `module`/`qualname` is origin-scoped (applies before the rename) — the escape hatch for a per-origin `default_factory` or transform. `split_event(source=Class, select=fn, targets=(...))` keys on the live source only |
| `MigrationTarget` | Class | `MigrationTarget(graph, baseline, store=None, threads=None)` — `graph` takes a checkpointer and returns the `EventGraph`; `baseline` is the committed file. The gate builds the graph on an empty `MemorySaver`. `store` returns a context manager of the real sync checkpointer; only `rewrite` uses it. `.open_store()` calls it, and raises `TargetError` when there is no `store` or when it returns an async context manager. `threads` takes the saver and returns a `ThreadKey` for each thread and namespace; it is needed for a saver other than `PostgresSaver`, `SqliteSaver` or `MemorySaver`, or for a store that more than one graph shares. `.thread_keys(saver)` lists them. `.snapshot()` builds the graph on an empty `MemorySaver`, and `.build(saver)` on a given saver. Both return the graph, its serde and its `MigrationSet`. Raises `TargetError` (`…serde.migrations`) when `graph` takes no checkpointer, or when the graph has no `NamespaceAwareSerde`. See [The migration gate](event-migrations.md#the-migration-gate) |
| `ThreadKey` | Class (`…serde.migrations`) | `ThreadKey(thread_id, checkpoint_ns)` — one checkpoint namespace of one thread. `""` is the root namespace |
| `list_thread_keys` | Function (`…serde.migrations`) | `list_thread_keys(saver) -> list[ThreadKey]` — each thread and checkpoint namespace of any saver, sorted, through the public `saver.list(None)`. Pass it as `threads=` of a `MigrationTarget`. It reads every checkpoint, so it suits a small store only. See [Checkpointers](event-migrations.md#checkpointers) |
| `rewrite_live_set` | Function (`…serde.migrations`) | `rewrite_live_set(target, *, expect, apply, passes=3, log=None) -> LiveSetOutcome \| WrongRelease` — the deploy step. Returns `WrongRelease` before it opens the store when the migrations of the running code do not have the fingerprint `expect`. It raises `TargetError` before it reads a thread when the saver that `store` opens holds a serde with other migrations. With `apply`, each pass rewrites each checkpoint, root or subgraph, that the last plan did not read unchanged, then plans again. See [Lifecycle of a read-time migration](event-migrations.md#lifecycle-of-a-read-time-migration) |
| `LiveSetOutcome` | Class (`…serde.migrations`) | `.passes` (one `RewriteReport` per apply pass), `.final` (the last plan), `.clean`, `.render(Invocation, expect)`. Each report holds one `ThreadRewrite` per checkpoint namespace of each thread |
| `WrongRelease` | Class (`…serde.migrations`) | `.expected`, `.running` (fingerprints), `.running_labels`, `.render(Invocation, *, apply)` |
| `assert_migration_gate` | Function | `assert_migration_gate(target, *, previous=None)` — runs `check` and raises `GateError` on a finding. Without `previous` the release rule does not run |
| `GateError` | Exception | `AssertionError` raised by `assert_migration_gate`. `.report` is the `GateReport` |
| `check` | Function (`…serde.migrations`) | `check(target, *, previous=None) -> GateReport` — the one gate. Rules in order: `baseline`, `names`, `coverage`, `revive`, `handlers`, `release`. `previous` is the baseline file of the previous release. See [The rules](event-migrations.md#the-rules) |
| `GateReport` | Class (`…serde.migrations`) | `.findings`, `.rules_run`, `.release_not_run` (why the release rule did not run, or `None`), `.held`, `.passed`, `.render(Invocation)` |
| `BaselineStale`, `AnonymousCallable`, `Uncovered`, `Unrevivable`, `HandlerMissing`, `SurvivedRelease` | Classes (`…serde.migrations`) | The typed findings. `Finding` is their union. Each has `.code`, `.subject` and `.lines(invocation)`, which ends with one next command |
| `Invocation` | Class (`…serde.migrations`) | `Invocation(target, previous=None)` — the caller's own CLI arguments, so a finding prints a command to run. `.baseline_update(prune=False)`, `.baseline_hold(label)` and `.rewrite(expect, apply=False)` give each command |
| `update_baseline` | Function (`…serde.migrations`) | `update_baseline(target, *, previous=None)` — writes the version 4 baseline the code needs. With `previous`, it resets the history of each shipped migration the code no longer holds to the `live` fields of `previous`, except for an identity that keeps another migration in the code. Without `previous`, it raises `PreviousRequiredError` when a recorded migration left the code |
| `PreviousRequiredError` | Exception (`…serde.migrations`) | `ValueError` raised by `update_baseline` when migrations left the code and `previous` is not given. `.deleted` holds the `MigrationKey` of each one. `.render(Invocation)` gives the message with the `baseline update --previous` command to run |
| `hold_migration` | Function (`…serde.migrations`) | `hold_migration(target, label, *, reason, previous)` — keeps a shipped migration one more release. The hold records the digest of the `previous` file, so it expires in the next release. A label that the committed baseline does not record yet gets a record |
| `MigrationSet.fingerprint`, `.shipped_in(previous_keys)`, `.deleted_since(previous_keys)` | Methods (`…serde.migrations`) | The fingerprint of the keys, and the comparison with the keys of a previous baseline. A previous key follows a current rename unless its identity was a rename source |
| CLI | Command | `python -m langgraph_events.serde.migrations check <target> [--previous FILE]`, `baseline update <target> [--previous FILE]`, `baseline hold <target> <label> --reason TEXT --previous FILE`, `fingerprint <baseline FILE>`, `rewrite <target> --expect FP [--apply] [--passes N]`. `<target>` is `module:attr` of a `MigrationTarget`. Exit 0 pass, 1 finding or a store that is not clean, 2 usage or target error, 3 `rewrite` under the wrong release |
| `synthesize_legacy_payload` | Function | `synthesize_legacy_payload(module, qualname, kwargs)` — builds the `(format, bytes)` a prior release would have written. For pinning a specific drifted field shape; the migration gate covers identity reachability |
| `replay_reducer` | Function | `replay_reducer(reducer, events)` — rebuild a reducer's channel value from its (already-migrated) event log when the projection/output shape changed. Thin wrapper over `BaseReducer.seed` |
| `NamespaceAwareSerde.revivable_identities` | Method | Read-only `frozenset` of every revivable `(module, qualname)`. The set the gate's `coverage` rule accepts |
| `NamespaceAwareSerde.live_fields` | Method | `{(module, qualname): frozenset of init fields}` for every live event class the serde speaks for. The baseline records it as `live` |
| `assert_resume_recovers` | Function | `assert_resume_recovers(before, after, *, seed, resume_with, thread_id=None)` — behavioral recovery proof: invokes `before` to interrupt, resumes `after` on the same checkpoint, asserts a `Resumed` (real recovery). The behavioral twin of the gate's `handlers` rule; both graphs must share one checkpointer. See [Testing handler recovery](event-migrations.md#testing-handler-recovery) |

Raw `RenameEvent` / `AddField` / `TransformFields` / `SplitEvent` operation constructors are the composite escape hatch in `langgraph_events.serde.migrations` — intentionally **not** re-exported at the top level (the decorator + `Migration` sugar is the public tier). `TransformFields(module, qualname, transform)` rewrites the stored kwargs of one identity; the identity picks the stage, the same rule as `AddField`. `SplitEvent(module, qualname, select, targets)` keys on a live identity and builds one of `targets` when `select` returns it; the class stage only. Wrap either like every raw op: `Migration(operations=(TransformFields(module, qualname, transform),))`.
