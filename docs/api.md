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
- `retry=RetryPolicy(...)` — [declarative backoff](control-flow.md#retries) around the handler call. Spelled as a `retry` class attribute on a `Command`, alongside `raises`/`invariants`.
- `invariants={InvariantClass: predicate}` — [preconditions](control-flow.md#invariants).
- `node_name=` — pin a **stable node identity** (defaults to the function name) so renaming the function never breaks an interrupted checkpoint. `previously=` (str or tuple) — historic node names to keep resumable after a rename. These, and `retry=`, are reserved (a field named `node_name`/`previously`/`retry` can't be matched via `**field_matchers`; a `Command` declaring an annotated `retry` field also fails at class creation — use `retry: ClassVar = ...` for the policy, or rename the field). Inline `Command.handle()` handlers default their node identity to the **command's `__qualname__`** (e.g. `Order.Place`), not the method name, so it is stable regardless of `handlers=[...]` order; a renamed command declares its historic node names as a class attribute, `previously: ClassVar = (...)`, mirroring `raises`/`invariants` in spelling. All four are read off the command class that declares them — a concrete `Command` may not be subclassed, so a historic node name still identifies exactly one class. See [Handler renames](event-migrations.md#handler-renames).

Returns enforced against the declared annotation, or the subscribed `Command.Outcomes` when unannotated.

## Graph & Execution

| Export | Type | Description |
|---|---|---|
| `EventGraph` | Class | Build and run the event-driven graph; accepts `@on`-decorated functions and/or `Command` subclasses with inline `handle`. `services=[...]` (type-keyed) or `services={...}` (name-keyed) injects project dependencies into handler params (see [Concepts › Signature injection](concepts.md#signature-injection)). `on_unresumable="raise"\|"halt"\|"warn"` (default `"raise"`) governs `resume()` on a thread not awaiting input |
| `EventGraph.from_namespaces()` | Classmethod | Build a graph from domains' inline command handlers; `handlers=` appends external handlers. With `checkpointer=MemorySaver()` auto-wires a `NamespaceAwareSerde` scoped to the passed namespaces and auto-collects every `@migrate_from` / `@backfill` decorator on those classes. Opt out by passing `MemorySaver(serde=<custom>)` — a user-supplied serde always wins |
| `EventGraph.invoke()` / `.ainvoke()` | Method | Run (sync/async); returns `EventLog` |
| `EventGraph.resume()` / `.aresume()` | Method | Resume an interrupted graph (requires checkpointer) |
| `EventGraph.abandon()` / `.aabandon()` | Method | Settles a paused thread without answering its interrupt. `require_interrupt=True` (default) raises `ValueError` if the thread has no pending interrupt; pass `False` to settle it anyway, recording `discarded=""`. Requires a checkpointer. Raises `ValueError` if the thread has no events to settle (never run, or only `pre_seed()`ed). Returns `None`. See [Ending a pause without answering it](control-flow.md#ending-a-pause-without-answering-it-abandon) |
| `EventGraph.threads_paused_on()` / `.athreads_paused_on()` | Method | Configs for every thread whose latest checkpoint has a pending interrupt, optionally filtered to an `Interrupted` class or subclass. Requires a checkpointer. Reads every checkpoint the checkpointer holds — O(all checkpoints). See [Ending a pause without answering it](control-flow.md#ending-a-pause-without-answering-it-abandon) |
| `EventGraph.get_state()` | Method | `GraphState` for a checkpointed thread |
| `EventGraph.namespaces()` | Method | Code-derived snapshot — domains, commands, outcomes, handlers, policies, edges, seeds. Returns a `NamespaceModel` |
| `NamespaceModel.text(view=...)` | Method | Human-readable tree; `view="structure"` or `"choreography"` (default) |
| `NamespaceModel.mermaid(namespace_order=..., reactor_hub_min=...)` | Method | Mermaid `graph LR` choreography diagram (handlers, policies, invariants, edges). `namespace_order` is `"affinity"` (default — cluster heavily-connected namespaces adjacent) or `"alphabetical"` (legacy, byte-stable for snapshot-pinned consumers). `reactor_hub_min=N` opts in to hub-style fanout: any `(source, handler)` producing ≥N solid-or-scatter targets renders as `Source → Hub → {targets}` with the handler name on the hub instead of repeated on every edge. For a structure-only view use `text(view="structure")`. |
| `NamespaceModel.json()` / `.to_dict()` | Method | JSON-serializable snapshot (event classes encoded as qualnames). Top-level `schema_version` is bumped on field removal, rename, or semantic change; additions don't bump. |
| `NamespaceModel.{Namespace, Command, CommandHandler, Policy, Edge, Invariant}` | Nested dataclasses | Frozen dataclasses for programmatic access |
| `NamespaceModel.Edge` | Nested dataclass | `source`, `via`, `target`, `kind` (*how* produced: `solid`/`scatter`/`raises`/`retry`/`framework`), `causation` (*causal role*: `intent`/`react`/`orchestrate`/`chain`/`None`) |
| `NamespaceModel.invariants` | Field | Tuple of `NamespaceModel.Invariant` — every declared invariant with `cls`, `commands`, `declared_by`, `reactors` |
| `EventGraph.compiled` | Property | Underlying `CompiledStateGraph` escape hatch |
| `EventGraph.reducer_names` | Property | `frozenset` of registered reducer names |
| `EventGraph.reflect(log)` | Method | Deterministic query surface over a run — returns a `Reflection` (see [Reflection](reflection.md)) |
| `Reflection` | Class | Facts-only read-model: `context()`, `tool()`, `overview()`, `event(i)`, `evidence(i)`, `schema()`, `state()`, `.log`. Injectable into handlers by parameter annotation, like `EventLog` |
| `QueryTool` | Frozen dataclass | The `query_log` LLM tool: `name` / `description` / `parameters` (JSON Schema) / `run(...) -> str`. Maps 1:1 to Anthropic / LangChain tool shapes |
| `EventLog` | Class | Immutable query container (see [Concepts](concepts.md#eventlog)) |
| `GraphState` | NamedTuple | `(events, is_interrupted, interrupted)` |

## System Events

| Export | Type | Description |
|---|---|---|
| `Halted` | Event | Signal immediate termination; subclass for domain-specific halts |
| `MaxRoundsExceeded` | Event | `Halted` subtype when `max_rounds` is exceeded |
| `Cancelled` | Event | `Halted` subtype when an async handler is cancelled |
| `Abandoned` | Event | `Halted` subtype emitted by `EventGraph.abandon()` or `.aabandon()`. `.reason` is the caller-supplied reason. `.discarded` holds the discarded interrupts' type names, deduped and joined with `", "` (`""` if none — match with `in` or `.split(", ")`, never `==`). Recorded, not dispatched: `@on(Abandoned)` never fires |
| `Unresumable` | Event | `Halted` subtype emitted by `EventGraph(on_unresumable="halt")` when `resume()` hits a thread not awaiting input. `.resume_value` is the resume event's type name. See [Handler renames](event-migrations.md#handler-renames). For a deliberate settle instead of this runtime net, see [`abandon()`](control-flow.md#ending-a-pause-without-answering-it-abandon) |
| `RunPaused` | Event | `SystemEvent` (not `Halted`) emitted by the router when a per-call `deadline=monotonic()+budget` expires between dispatch rounds. Emitted at most once per `/run` regardless of how many parallel handlers fan in past the deadline. Cursor is advanced past it so a fresh `/run` on the same `thread_id` continues normally. No default AG-UI wire mapping — register a custom `EventMapper` to surface it on the wire |
| `Interrupted` | Base class | Subclass with typed fields to pause for human input. For frontend-discriminated payloads see `InterruptedWithPayload` in `langgraph_events.agui` |
| `Resumed` | Event | Emitted on `resume()` with the dispatched event + `interrupted` backref |
| `HandlerRaised` | Event | Emitted when a handler raises a `raises=`-declared exception; carries `handler`, `source_event`, `exception`, `abandoned_for_deadline`. The exception keeps its traceback — this is the terminal failure, the one you debug from. With a `retry=` policy it fires only after the budget is spent, or — uniquely tagged `abandoned_for_deadline=True` — once the next backoff would land on or past the run's `deadline=`. The flag is `False` on every other raise |
| `HandlerRetried` | Event | Emitted before each `retry=` backoff wait; carries `handler`, `source_event`, `exception`, `attempt` (1-based, the call that failed), `delay_seconds`. The exception instance is live (matchers and field injection work) but its `__traceback__` is detached — as are those of its `__cause__`/`__context__` chain and any `ExceptionGroup` members — so a breadcrumb never pins the failed attempt's frame; see [Retries](control-flow.md#retries). A deadline give-up emits none, since no wait happened. Suppress with `RetryPolicy(observe="log"\|"silent")` |
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
)
```

## Warnings

| Export | Type | Description |
|---|---|---|
| `OrphanedEventWarning` | Warning | Issued at graph construction when a return type has no subscriber |
| `UnresumableError` | RuntimeError subclass | Raised by `resume()` (default `EventGraph(on_unresumable="raise")`) when the thread is not awaiting input — paused handler renamed/removed, thread already finished, double-resume, or a thread already settled by [`abandon()`](control-flow.md#ending-a-pause-without-answering-it-abandon). Declare `@on(previously=...)` to recover, or set `on_unresumable="halt"`/`"warn"`. See [Handler renames](event-migrations.md#handler-renames) |
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
| `merge_frontend_messages` | Function | `resume_factory` helper: reads messages from `checkpoint_state["reducers"][reducer_name]`, converts `input_data.messages`, merges via `add_messages` (id-based dedup). Returns a tuple |
| `build_langchain_tools` / `detect_new_tool_results` | Function | Convert `RunAgentInput.tools` to OpenAI-format / extract new `ToolMessage`s from a resume request |

## Serde Subpackage

Namespace-aware serialization, auto-wired by default. `EventGraph.from_namespaces(..., checkpointer=MemorySaver())` builds a `NamespaceAwareSerde` scoped to the passed namespaces and auto-collects every `@migrate_from` / `@backfill` on those classes. Opt out by passing `MemorySaver(serde=<custom>)`. See [Event migrations](event-migrations.md).

Exported from `langgraph_events.serde` unless a different module is noted.

| Export | Type | Description |
|---|---|---|
| `NamespaceAwareSerde` | Class | `JsonPlusSerializer` subclass keying `Event` identity by `(__module__, __qualname__)`. Drop-in for any checkpointer accepting `serde=`. `EventGraph.from_namespaces` auto-wires one scoped to its namespaces when the checkpointer carries the default serde; pass your own via `MemorySaver(serde=...)` to opt out. Kwargs: `migrations=`, `namespaces=`, `legacy_write=` |
| `migrate_from` | Decorator | `@migrate_from(*old_qualnames, in_module=None, backfill=None)` — records historic identities a class held. `backfill={field: default}` scopes fills to payloads written under *this* decorator's qualname (one qualname per decorator) — the fan-in case where collapsed origins each pin their own discriminator. Auto-collected from the namespaces the serde was built with; stacks; not MRO-inherited |
| `backfill` | Decorator | `@backfill(field, *, default=… \| default_factory=…)` — class-global back-fill for a field that is required in code but absent from pre-existing payloads (one value for every origin and era; acts as the fallback under origin-scoped fills). Same auto-collection as `@migrate_from`; reuses the `AddField` mutable-default guard |
| `Migration` | Class | Hand-authored migration group for cross-module / composite cases. `Migration.rename(...)` / `Migration.add_field(...)` are single-op sugar; pass the live target as `to=<class>` / `target=<class>` (refactor-safe) instead of `new_module`/`new_qualname` strings. `add_field` keyed on a historic `module`/`qualname` is origin-scoped (applies before the rename) — the escape hatch for a per-origin `default_factory` |
| `assert_all_baselined_cover` | Function | `assert_all_baselined_cover(serde, baseline_path)` — weakest coverage gate: set-membership over `revivable_identities()`; raises `MigrationCoverageError` if an identity is neither live nor rename-covered. Scoped to the serde's `namespaces=` walk plus `events=`. See [coverage gates](event-migrations.md#coverage-gates) |
| `assert_all_baselined_resolve` | Function | `assert_all_baselined_resolve(serde, baseline_path)` — rename-aware gate: asserts every baselined identity resolves to a live `Event` class **without constructing** it. For baselines with `__post_init__`-validated, framework, or module-level events |
| `assert_all_baselined_revive` | Function | `assert_all_baselined_revive(serde, baseline_path)` — strongest gate: pushes a synthesized payload for every baselined identity through the real read path and asserts it revives. No per-event list to maintain |
| `synthesize_legacy_payload` | Function | `synthesize_legacy_payload(module, qualname, kwargs)` — builds the `(format, bytes)` a prior release would have written. For pinning a specific drifted field shape; the loop gate above covers identity reachability |
| `replay_reducer` | Function | `replay_reducer(reducer, events)` — rebuild a reducer's channel value from its (already-migrated) event log when the projection/output shape changed. Thin wrapper over `BaseReducer.seed` |
| `NamespaceAwareSerde.revivable_identities` | Method | Read-only `frozenset` of every revivable `(module, qualname)`. For custom coverage rules; `assert_all_baselined_cover` is the gate |
| `assert_all_baselined_handlers_cover` | Function | `assert_all_baselined_handlers_cover(graph, baseline_path)` — handler analog: asserts every baselined handler node name is still live or covered by an `@on(previously=...)` alias. Takes the `EventGraph`. See [Handler renames](event-migrations.md#handler-renames) |
| `assert_resume_recovers` | Function | `assert_resume_recovers(before, after, *, seed, resume_with, thread_id=None)` — behavioral recovery proof: invokes `before` to interrupt, resumes `after` on the same checkpoint, asserts a `Resumed` (real recovery). The handler analog of `assert_all_baselined_revive`; both graphs must share one checkpointer. See [Testing handler recovery](event-migrations.md#testing-handler-recovery) |
| `CoverageError` | Exception | `AssertionError` base for both coverage failures; `except CoverageError` catches event and handler gates alike |
| `MigrationCoverageError` | Exception | Raised by the event gates (`assert_all_baselined_cover`/`_resolve`/`_revive`); `CoverageError` subclass. `.uncovered` is the offending `(module, qualname)` tuple |
| `HandlerCoverageError` | Exception | Raised by `assert_all_baselined_handlers_cover`; `CoverageError` subclass (sibling of `MigrationCoverageError`). `.uncovered` is the offending handler node-name tuple |
| `write_baseline` | Function (`…serde.migrations.detect`) | `write_baseline(graph, path, *, allow_removed=False)` — snapshots topology. Raises `BaselineRegressionError` rather than silently overwriting away identities the old baseline recorded; `allow_removed=True` for intentional deletes |
| `detect_changes` | Function (`…serde.migrations.detect`) | `detect_changes(graph, baseline_path)` — diffs topology vs baseline into rename/ambiguous/removed buckets. Also runnable as the CI gate `python -m langgraph_events.serde.migrations <module:factory> <baseline>` |
| `BaselineRegressionError` | Exception (`…serde.migrations.detect`) | Raised by `write_baseline`; `ValueError` subclass. `.removed` is the tuple of dropped identities |

Raw `RenameEvent` / `AddField` operation constructors are the composite escape hatch in `langgraph_events.serde.migrations` — intentionally **not** re-exported at the top level (the decorator + `Migration` sugar is the public tier).
