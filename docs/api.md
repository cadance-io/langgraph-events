# API Reference

## Event Taxonomy

| Export | Type | Description |
|---|---|---|
| `Namespace` | Base class | Groups nested commands and outcomes. Exposes `__namespace_name__` and `__reducers__`. Names are unique per graph, not per process. See [namespace scope](concepts.md#namespace-scope) |
| `Command` | Base class | Imperative intent. Must be nested inside a `Namespace`. A concrete `Command` cannot be subclassed (`class Rush(Order.Place)` raises `TypeError`). Exposes `.Outcomes`, the union of its nested `DomainEvent`s. The sole public method of the class is its inline command handler. The method can have any name, `handle` by convention. See [Inline](concepts.md#inline-command-handlers) |
| `DomainEvent` | Base class | Fact inside the domain. Must be nested inside a `Namespace` or `Command` |
| `IntegrationEvent` | Base class | Cross-boundary fact. Top-level |
| `SystemEvent` | Base class | Fact that the library emits. Top-level. See [System Events](#system-events) |

## Mixins

| Export | Type | Description |
|---|---|---|
| `MessageEvent` | Mixin | Wraps LangChain `BaseMessage`. Declares a `message` or `messages` field. Compose with any event branch |
| `SystemPromptSet` | Event | Built-in `IntegrationEvent` + `MessageEvent` for system prompts |
| `Event` | Base class | Do not subclass directly. Valid as a type annotation, as the `@on(Event)` catch-all and as the reducer `event_type=` catch-all |

## Handler Subscription

`@on` subscribes a handler. Three shapes:

- Bare `@on`: the event type is inferred from the annotation of the first parameter.
- `@on(kwargs=...)`: inferred, with modifiers.
- `@on(Type, ...)`: explicit. Required for a multi-event subscription.

Modifiers:

- `*field_matchers`: [field dispatch](control-flow.md#field-matchers). A `type` value matches by `isinstance`. An `Event`, `Exception` or `Invariant` subclass is a valid type. A `str` value matches by equality.
- `raises=`: [declared exceptions](control-flow.md#handler-exceptions).
- `retry=RetryPolicy(...)`: [declarative backoff](control-flow.md#retries) around the handler call. On a `Command`, spell it as a `retry` class attribute, next to `raises` and `invariants`.
- `invariants={InvariantClass: predicate}`: [preconditions](control-flow.md#invariants).
- `node_name=` pins a stable node identity, by default the function name. `previously=` (str or tuple) lists historic node names that stay resumable after a rename. `previously`, `raises`, `invariants` and `retry` are reserved modifier names. A `Command` that declares an annotated field with one of these names fails at class creation. Spell the modifier as `ClassVar`. A field matcher cannot use one of these names either. See [Handler renames](event-migrations.md#handler-renames).

The return value is checked against the declared annotation, or against the subscribed `Command.Outcomes` when the handler has no annotation. An omitted return annotation is legal. An annotation that does not resolve at run time raises `TypeError` at graph construction. A member that is not an `Event` subclass, `Scatter[...]` or `None` raises the same error. See [Return contract](concepts.md#return-contract).

## Graph & Execution

| Export | Type | Description |
|---|---|---|
| `EventGraph` | Class | Builds and runs the graph from `@on` handlers and `Command` classes with an inline command handler. `services=[...]` (type-keyed) or `services={...}` (name-keyed) injects dependencies into handler parameters. `on_unresumable="raise"\|"halt"\|"warn"` (default `"raise"`) governs `resume()` on a thread that is not paused. `context_schema=` declares the type of the run context, passed with `context=` on each run call. `retired=` goes to the topology. See [Signature injection](concepts.md#signature-injection) and [Run context](concepts.md#run-context) |
| `EventTopology(handlers, *, reducers=None, context_schema=None, retired=())` | Class | The handlers, reducers and namespaces of a graph, with no services and no checkpointer. `context_schema` is the type of the run context. A handler parameter of exactly that type receives the context, so it is not a requirement. `retired` lists each event class that no handler returns, but that a stored thread can still hold. An entry that is not an `Event` subclass, or that is abstract, raises `TypeError`. An entry that a handler returns raises `ValueError`. Runs every construction check that needs no service. Has `namespaces()`, `serde()`, `select_reducers()`, `resolve_reducer_names()`, `handler_names`, `resumable_node_names`, `reducers`, `reducer_names`, `derived`, `injectable_names`, `requirements` and `retired_events`. `EventGraph(handlers, ...)` is the shortcut for `EventGraph.of(EventTopology(handlers, ...), ...)`. See [Topology, graph and store](concepts.md#topology-graph-and-store) |
| `EventTopology.context_schema` / `.require_context(context)` | Property / Method | The type of the run context, or `None`. `require_context()` checks the context of one run and returns it, a `dict` coerced to a dataclass or pydantic schema. It raises `TypeError` for a missing context or one of another type. See [Run context](concepts.md#run-context) |
| `EventTopology.from_namespaces(*namespaces, handlers=None, reducers=None, retired=())` | Class method | The topology of the inline command handlers of each namespace, then `handlers`. Collects the same handlers as `EventGraph.from_namespaces()`. Wires no serde |
| `EventTopology.select_reducers(include_reducers)` | Method | The values that an `include_reducers=` value selects for a stream. `True` selects each stored reducer and each `Derived`, in registration order. `False` or `[]` selects none. A list selects its known names in its order. An unknown name in a list gets a warning and is skipped. The result has `names`, and `values(state)` reads each selected value from a state, with an empty default for an unset channel |
| `EventTopology.resolve_reducer_names(include_reducers)` | Method | `list(select_reducers(include_reducers).names)`. A name can be a stored reducer or a `Derived` |
| `EventTopology.reducers` / `.reducer_names` | Property | The stored reducers by name, and their names. A `Derived` is not a stored reducer, so neither holds one |
| `EventTopology.derived` | Property | Each registered `Derived` by its name. Read-only |
| `EventTopology.injectable_names` | Property | Every name a handler parameter can read: each stored reducer and each `Derived` |
| `EventTopology.requirements` | Property | Tuple of `ServiceRequirement(node, qualname, param, annotation)`, one for each handler parameter that only a service can fill. `node` is the graph-node name of the handler. `qualname` is the module and qualified name of the handler function. A variadic parameter is not listed. The annotation is `Unannotated()` when the parameter has none, and `UnresolvedAnnotation(reason)` when it did not resolve |
| `EventTopology.reducers` | Property | Read-only mapping of each registered reducer by its name. It holds the reducers of `reducers=` and the reducers that the namespaces declare |
| `EventTopology.retired_events` | Property | Tuple of each event class of `retired=`, once, in the order given. See [Retired events](concepts.md#retired-events) |
| `EventTopology.serde(migrations=(), *, legacy_write=False)` | Method | Returns a `NamespaceAwareSerde` that reads every event class of the topology, the retired events included, and collects each `@migrate_from` and `@backfill` on them. It is the serde that `from_namespaces(..., checkpointer=...)` wires |
| `EventGraph.of(topology, *, max_rounds=100, services=None, checkpointer=None, store=None, recursion_limit=None, on_unresumable="raise")` | Classmethod | Builds a graph that runs *topology* with *services*. A requirement that no service matches raises `TypeError` |
| `EventStore(topology, checkpointer)` | Class | Store maintenance with no services: `threads_paused_on()`, `abandon()`, `unrevivable_threads()`, `plan_rewrite()` and `rewrite_store()`, each with its async twin. `checkpointer` must be a `BaseCheckpointSaver`. The store does not change its serde. A rewrite needs `checkpointer.serde = topology.serde()` |
| `EventGraph.event_store` | Property | The `EventStore` of the graph. It is `EventStore(graph.topology, checkpointer)`. Raises `ValueError` when the graph has no checkpointer |
| `RunScoped(factory)` | Class | Marker for a service built once per node call, placed in the name-keyed `services={...}` mapping. The factory receives the node `RunnableConfig` and must have a return annotation. A factory whose second parameter is annotated with the `context_schema` type receives `(config, context)`. See [Run-scoped services](concepts.md#run-scoped-services) |
| `EventGraph.from_namespaces(*namespaces, handlers=None, **kwargs)` | Classmethod | Builds a graph from the inline command handlers of the namespaces. `handlers=` appends external handlers. With `checkpointer=`, it replaces LangGraph's untouched default `JsonPlusSerializer` with a `NamespaceAwareSerde` scoped to the graph, and collects every `@migrate_from` and `@backfill` on the graph classes. It keeps a `NamespaceAwareSerde` that the checkpointer holds. Any other serde, for example an `EncryptedSerializer` or a configured `JsonPlusSerializer`, raises `TypeError` and stays on the checkpointer. See [Event migrations](event-migrations.md) |
| `EventGraph.invoke(seed, *, deadline=None, pause_on=None)` / `.ainvoke()` | Method | Runs the graph and returns an `EventLog` |
| `EventGraph.resume(value, *, deadline=None, pause_on=None)` / `.aresume()` | Method | Resumes a paused thread. Requires a checkpointer. The resume value must be an `Event`. Another value raises `TypeError` before the graph runs, and the thread stays paused |
| `EventStore.abandon(config, *, reason="", require_interrupt=True)` / `.aabandon()` | Method | Settles a paused thread with a terminal `Abandoned` event, without answering its interrupt. See [Ending a pause without answering it](control-flow.md#ending-a-pause-without-answering-it-abandon) |
| `EventStore.threads_paused_on(event_type=None, *, thread_ids)` / `.athreads_paused_on()` | Method | Returns the config of each listed thread that is paused, on `event_type` or a subclass when given. `thread_ids=None` or a bare `str` raises `TypeError`. See [Finding candidates server-side](retiring-interrupted.md#finding-candidates-server-side) |
| `EventStore.unrevivable_threads(*, thread_ids)` / `.aunrevivable_threads()` | Method | Maps each listed thread id to the qualnames its latest checkpoint can no longer revive. Raises `ValueError` without a `NamespaceAwareSerde` on the checkpointer. See [Retiring an Interrupted subclass](retiring-interrupted.md) |
| `EventStore.plan_rewrite(*, drop=(), thread_ids, checkpoint_ns="")` / `.aplan_rewrite()` | Method | Returns the `RewriteReport` of the rewrite each listed thread needs, and writes nothing. A listed thread id with no checkpoint is reported as `refused` with reason `no checkpoint for this thread id`. Raises `ValueError` without a `NamespaceAwareSerde` on the checkpointer. See [Rewriting the live set](migration-gate.md#rewriting-the-live-set) |
| `EventStore.rewrite_store(*, drop=(), thread_ids, checkpoint_ns="")` / `.arewrite_store()` | Method | Raises `ValueError` without a `NamespaceAwareSerde` on the checkpointer. Rewrites the latest checkpoint of each listed thread under the live migration table, with the same arguments and report as `plan_rewrite()`. See [Rewriting the live set](migration-gate.md#rewriting-the-live-set) |
| `EventGraph.get_state(config)` | Method | `GraphState` for a checkpointed thread |
| `EventGraph.namespaces()` | Method | Code-derived snapshot: namespaces, commands, outcomes, handlers, edges and seed events. Returns a `NamespaceModel`. See [Namespace introspection](concepts.md#namespace-introspection-visualization) |
| `NamespaceModel.text(view=...)` | Method | Human-readable tree. `view="structure"` or `"choreography"` (default) |
| `NamespaceModel.mermaid(namespace_order=..., reactor_hub_min=..., focus=..., show_raises=..., notes=..., muted=...)` | Method | Mermaid `graph LR` choreography diagram. A `notes=` or `muted=` key is an event qualname, a reducer name or a reaction name. Any other key raises `ValueError`. `namespace_order` is `"affinity"` (default, clusters connected namespaces) or `"alphabetical"` (byte-stable). `reactor_hub_min=N` draws a `(source, handler)` pair with N or more targets through one hub node. A wrong `notes` or `muted` key raises `ValueError` that lists the valid keys. See [Focused diagrams](concepts.md#focused-diagrams) |
| `NamespaceModel.json()` / `.to_dict()` | Method | JSON-serializable snapshot. Event classes are encoded as qualnames. `schema_version` changes on a field removal, rename or semantic change, not on an addition |
| `NamespaceModel.{Namespace, Command, CommandHandler, Policy, Edge, Invariant}` | Nested dataclasses | Frozen dataclasses for programmatic access |
| `NamespaceModel.Edge` | Nested dataclass | `source`, `via`, `target`, `kind` (`solid`/`scatter`/`raises`/`retry`/`framework`), `causation` (`intent`/`react`/`orchestrate`/`chain`/`None`) |
| `NamespaceModel.reducers` | Field | Tuple of `NamespaceModel.Reducer`, each with `name`, `subscribes` and `namespace`. `subscribes` holds the concrete events the reducer projects, or `None` when a `runtime_checkable` Protocol with data members makes them unknown. `namespace` is `None` for a reducer passed to `EventGraph(reducers=...)` |
| `NamespaceModel.derived` | Field | Tuple of `NamespaceModel.Derived`, each with `name`, `inputs` and `namespace`. `inputs` holds the names of the input reducers, in the order of the function. `namespace` is `None` for a `Derived` passed to `EventGraph(reducers=...)`. `json()` holds them under the `derived` key |
| `NamespaceModel.Focus(namespaces=..., reactions=..., reducers=...)` | Nested dataclass | The part of the graph that `mermaid(focus=...)` draws. Each field takes one name or an iterable of names. An empty `Focus()` raises `ValueError`. See [Focused diagrams](concepts.md#focused-diagrams) |
| `NamespaceModel.invariants` | Field | Tuple of `NamespaceModel.Invariant`, each with `cls`, `commands`, `declared_by`, `reactors` |
| `EventGraph.compiled` | Property | The underlying `CompiledStateGraph`, for direct LangGraph calls |
| `EventGraph.reducer_names` | Property | `frozenset` of the stored reducer names. A `Derived` is not one. Forwards to `graph.topology.reducer_names`, the documented path |
| `EventGraph.resumable_node_names` | Property | `frozenset` of every node name a paused checkpoint can resume at: each canonical handler name and each `@on(previously=...)` alias. The gate `handlers` rule reads it. Forwards to `graph.topology.resumable_node_names`, the documented path |
| `EventGraph.reflect(log)` | Method | Returns a `Reflection`, a deterministic query surface over a run. See [Reflection](reflection.md) |
| `Reflection` | Class | Facts-only read model: `context()`, `tool()`, `overview()`, `event(i)`, `evidence(i)`, `schema()`, `state()`, `.log`. Injectable into a handler by parameter annotation, like `EventLog` |
| `QueryTool` | Frozen dataclass | The `query_log` LLM tool: `name`, `description`, `parameters` (JSON Schema) and `run(...) -> str`. Maps 1:1 to the Anthropic and LangChain tool shapes |
| `EventLog` | Class | Immutable query container. `cause(e)`, `effects(e)` and `flow(e)` read the recorded causes. `causes` is the tuple of causes aligned with `events`, or `None` when the log records none. `latest(T, not_followed_by=U)` returns `None` when a `U` comes after the most recent `T`. See [EventLog](concepts.md#eventlog) |
| `Cause` | Frozen dataclass | `(source, via)`: the event a handler received, and the node name of the handler. See [Causes](concepts.md#causes) |
| `UnknownCause` | Class | Base of `NotRecorded` and `SourceDropped`. `reason` states in one sentence why the cause is unknown. See [Causes](concepts.md#causes) |
| `NotRecorded` | Frozen dataclass | `UnknownCause`: the event was written before causes existed. See [Causes](concepts.md#causes) |
| `SourceDropped` | Frozen dataclass | `UnknownCause` `(via, source_type)`: `rewrite_store(drop=...)` deleted the source of the event. See [Causes](concepts.md#causes) |
| `FrameworkEvent` | Frozen dataclass | The library wrote the event: `RunPaused`, `MaxRoundsExceeded`, `Cancelled`, `Abandoned` or `Unresumable`. `EventLog.cause()` returns `None` only for a seed event. See [Causes](concepts.md#causes) |
| `GraphState` | NamedTuple | `(events, is_interrupted, interrupted)`. `events` carries the recorded causes of the thread |
| `RewriteReport` | Dataclass | `(applied, threads)` from `plan_rewrite()` or `rewrite_store()`. `applied` is `False` from a plan. `.refused` is the tuple of refused threads. `str(report)` prints one summary line, then one line per refused thread |
| `ThreadRewrite` | Dataclass | One thread in a `RewriteReport`: `thread_id`, `checkpoint_ns`, `location`, `status` (`"rewrite"`, `"unchanged"` or `"refused"`), `migrated` as `(stored, live)` identity pairs, `dropped` counts per identity, and `reason` on a refusal |

## System Events

| Export | Type | Description |
|---|---|---|
| `Halted` | Event | Signals immediate termination. Subclass it for a domain-specific halt |
| `MaxRoundsExceeded` | Event | `Halted` subtype when `max_rounds` is exceeded |
| `Cancelled` | Event | `Halted` subtype when an async handler is cancelled |
| `Abandoned` | Event | `Halted` subtype that `abandon()` records. It is recorded, not dispatched, so `@on(Abandoned)` never fires. `.reason` is the caller's reason. `.discarded` is a `str`: the qualnames of the discarded interrupts joined with `", "`, or `""` when none. A fanned-out superstep can pause two interrupts, so match it with `in` or split on `", "`, not with `==`. See [Ending a pause without answering it](control-flow.md#ending-a-pause-without-answering-it-abandon) |
| `Unresumable` | Event | `Halted` subtype that `EventGraph(on_unresumable="halt")` records when `resume()` reaches a thread that is not paused. `.resume_value` is the type name of the resume event. See [Handler renames](event-migrations.md#handler-renames). For a deliberate settle, use [`abandon()`](control-flow.md#ending-a-pause-without-answering-it-abandon) |
| `RunPaused` | Base class | `SystemEvent` (not `Halted`) base of every deadline pause and signal pause that the router emits between dispatch rounds. It names no cause, so building it directly raises `TypeError`. Match it to see every pause. Emitted at most once per call, whatever the number of parallel handlers that pass the deadline. The cursor is advanced past it, so a fresh call on the same `thread_id` continues normally. No default AG-UI wire mapping. Register a custom `EventMapper` to put it on the wire |
| `RunPausedOnDeadline` | Event | `RunPaused` subtype emitted when the per-call `deadline=monotonic()+budget` expires between dispatch rounds |
| `RunPausedOnSignal` | Event | `RunPaused` subtype emitted when the per-call `pause_on=` signal reports `is_set()` between dispatch rounds. Same resume semantics and once-per-call rule as every `RunPaused`. When the deadline has also expired, the router emits this event, not `RunPausedOnDeadline`. See [Pause on a signal](control-flow.md#pause-on-a-signal-runpausedonsignal) |
| `PauseSignal` | Protocol | The `pause_on=` contract: an object with `is_set() -> bool`. `asyncio.Event` and `threading.Event` satisfy it. A class, or a value with no callable `is_set`, raises `TypeError` before any round runs. `pause_on=event.is_set`, `pause_on=True` and `pause_on=asyncio.Event` are rejected. Pass the event object itself. Every `EventGraph` entry point declares `pause_on: PauseSignal \| None`, so a type checker rejects these values too |
| `Interrupted` | Base class | Subclass it with typed fields to pause for human input. For a frontend-discriminated payload, see `InterruptedWithPayload` in `langgraph_events.agui` |
| `Resumed` | Event | Emitted on `resume()` with the dispatched event and an `interrupted` backref |
| `HandlerRaised` | Event | Emitted when a handler raises a `raises=`-declared exception. Carries `handler`, `source_event`, `exception`. The exception keeps its traceback. With a `retry=` policy, a plain `HandlerRaised` fires after the budget is spent. A policy that gives up with attempts left emits a `HandlerAbandoned` subclass instead |
| `HandlerAbandoned` | Base class | `HandlerRaised` subtype. A `RetryPolicy` had attempts left, and the run deadline or pause signal refused the next wait. Catch it to see every early give-up. It names no cause, so building it directly raises `TypeError` |
| `HandlerAbandonedOnDeadline` | Event | `HandlerAbandoned` subtype emitted when the next backoff would land on or past the run `deadline=`. The run continues, so `@on(HandlerRaised, exception=X)` handlers run for it |
| `HandlerAbandonedOnSignal` | Event | `HandlerAbandoned` subtype emitted when the run `pause_on=` signal is set before the next backoff wait. It stays in the log, and no handler runs on it. `@on(HandlerRaised, exception=X)` matches it. See [Pause on a signal](control-flow.md#pause-on-a-signal-runpausedonsignal) |
| `HandlerRetried` | Event | Emitted before each `retry=` backoff wait. Carries `handler`, `source_event`, `exception`, `attempt` (1-based, the call that failed), `delay_seconds`. The exception instance is live, but its `__traceback__` is detached, as are those of its `__cause__` and `__context__` chain and of any `ExceptionGroup` member. A deadline or signal give-up emits none, because no wait happened. Suppress with `RetryPolicy(observe="log"\|"silent")`. See [Retries](control-flow.md#retries) |
| `RetryPolicy` | Class | Declarative retry for a handler call: `max_attempts` (counts the initial call), `base_delay`, `max_delay`, `strategy` (`"exponential"`/`"constant"`), `jitter` (full jitter, default `True`), `on=` (must overlap `raises=`, in either direction), `respect_retry_after`, `observe`. Distinct from `langgraph.types.RetryPolicy`, which re-runs a whole node. See [Retries](control-flow.md#retries) |
| `Invariant` | Marker class | Subclass it to declare a typed invariant. Used as a dict key in `invariants=` and as the matcher value in `@on(InvariantViolated, invariant=...)` |
| `InvariantViolated` | Event | Emitted when an `invariants=` predicate returns false. Carries `invariant` (an instance of the declared `Invariant` subclass), `handler`, `source_event`. Use `@on(InvariantViolated, invariant=SomeInvariant)` to pin one invariant. See [Invariants](control-flow.md#invariants) |
| `Scatter` | Class | Fan-out into multiple events. The return type must enumerate concrete events, for example `Scatter[A \| B]`. See [Scatter](control-flow.md#scatter) |

## Reducers

| Export | Type | Description |
|---|---|---|
| `Reducer` | Class | List accumulator. Declare it as a `Namespace` class attribute or pass it through `reducers=` |
| `ScalarReducer` | Class | Last-write-wins for a single value. `None` is valid |
| `FoldReducer` | Class | Folds each event into accumulating state through `fold(self, state)`. The next value depends on the prior value. Generic over state type `S` |
| `Foldable` | Protocol | `@runtime_checkable` structural type for an event with a `fold` method. Types the event argument of `FoldReducer`. Do not inherit it |
| `BaseReducer` | Class | Abstract base for a custom reducer. Subclass it for a bespoke channel |
| `Derived(name, *, fn, namespace=None)` | Class | A value that a pure function computes from stored reducers at node start. Generic over the return type of `fn`. Injected by parameter name. Never stored, so it is not a stored reducer. Frozen and hashable. See [`Derived`](reducers.md#derived) |
| `SKIP` | Sentinel | Return it from `ScalarReducer.fn` or from a `FoldReducer.fold` to leave the value unchanged |
| `RESET` | Sentinel | Return it from a `FoldReducer.fold` to clear the channel back to `default_factory()` |
| `message_reducer` | Function | Built-in reducer for `MessageEvent` projection |

## Lifecycle Hooks

| Export | Type | Description |
|---|---|---|
| `on_namespace_finalize(cls, callback)` | Function | Schedules `callback(cls, namespace_cls)` to fire once the enclosing `Namespace.__init_subclass__` of `cls` finishes. A class decorator can defer a `typing.get_type_hints()` call until a forward reference to a sibling resolves. Fires at once if the enclosing `Namespace` has already finalized |

## Streaming & Frames

`EventGraph.stream_events()` / `.astream_events()` yield events as they are produced. `EventGraph.stream_resume()` / `.astream_resume()` are the streaming equivalents of `resume()`. The frame types are importable `from langgraph_events.stream import StreamFrame, LLMToken, LLMStreamEnd, LLMToolCallChunk, CustomEventFrame, StateSnapshotFrame, InterruptFrame, StreamItem`. See [Streaming](streaming.md) for the flags, the frame tables and the emission helpers.

## Warnings

| Export | Type | Description |
|---|---|---|
| `OrphanedEventWarning` | Warning | Issued at graph construction when a return type has no subscriber |
| `UnresumableError` | RuntimeError subclass | Raised by `resume()` under the default `EventGraph(on_unresumable="raise")` when the thread is not paused. A thread is paused while its checkpoint has a scheduled node or a pending interrupt of a registered handler. Causes: the paused handler was renamed or removed, the thread already finished, a double resume, or a thread settled by [`abandon()`](control-flow.md#ending-a-pause-without-answering-it-abandon). Declare `@on(previously=...)` to recover, or set `on_unresumable="halt"` or `"warn"`. See [Handler renames](event-migrations.md#handler-renames) |
| `DomainPatternWarning` | Warning | Two or more events of one namespace fan out to an identical target set. See [Control Flow](control-flow.md) |
| `CommandChainWarning` | Warning | An inline command handler emits another `Command`, a `chain`-causation edge. See [Control Flow](control-flow.md) |

## Errors

| Export | Type | Description |
|---|---|---|
| `CommandPrivacyError` | TypeError subclass | Raised at `EventGraph` construction, or at run time when a handler with a broad base-class annotation (for example `-> DomainEvent`) builds a `Cmd.Private(...)` that the static contract check cannot see. An outcome nested inside a `Command` is private to the inline command handler of that `Command`. No sibling `Command` and no external handler can produce it |
| `ReducerNotSetError` | ValueError subclass | Raised before the handler body runs when a handler parameter rejects `None` and the channel value is `None`. See [Required values](reducers.md#scalar-reducer-required) |

## AG-UI Subpackage

Requires `[agui]`. See [AG-UI Adapter](agui.md).

| Export | Type | Description |
|---|---|---|
| `AGUIAdapter` | Class | Maps `EventGraph` streams to AG-UI protocol events |
| `AGUI_EXTRAS_KEY` | Constant | Reserved `BaseMessage.additional_kwargs` key. Its mapping is forwarded as extra fields on the AG-UI message, and receives the extra fields of the client on the way back |
| `AGUI_HELD_PARTS_KEY` | Constant | Reserved `BaseMessage.additional_kwargs` key. It holds each inbound content part with a `file` source, which the model does not receive. The snapshot restores each part at its position. See [Parts the model does not receive](agui.md#parts-the-model-does-not-receive) |
| `held_parts(message)` | Function | Returns the parts held on a LangChain message as AG-UI `InputContentPart` objects, in their original order. Skips a corrupt entry with a `WARNING`. Never raises |
| `AGUIAdapter.stream()` / `.connect()` / `.reconnect()` | Method | Execute, rehydrate a checkpoint, refresh. See [Run config](agui.md#run-config) |
| `AGUIAdapter.snapshot()` | Method | Replay the state of a thread without its pending interrupts, for a thread that a live run holds. See [Snapshot while a run holds the thread](agui.md#snapshot-while-a-run-holds-the-thread) |
| `FrontendStateMutated` | Event | Adapter-emitted `IntegrationEvent` that carries `RunAgentInput.state` for a reducer that projects client state |
| `FrontendToolCallRequested` | Event | `Interrupted` subclass for a handler-initiated frontend tool call (CopilotKit `useFrontendTool`) |
| `InterruptedWithPayload[PayloadT]` | Generic base | `Interrupted` variant with a typed payload from `interrupt_payload(self) -> PayloadT`. See [Typed Interrupt Payloads](agui.md#typed-interrupt-payloads) |
| `AGUICustomEvent` / `AGUISerializable` | Protocol | Override the fallback event name or the payload serialization |
| `SeedFactory` / `ResumeFactory` / `EventMapper` | Protocol | Input to seed events, resume detection, event mapping |
| `RunRefusedError(code, message)` | Exception | Raise it from a seed factory or a resume factory to refuse the run. The client gets `RUN_ERROR` with that `code` and `message`, and the run stores nothing. See [Async Factories and Refusals](agui.md#async-factories-and-refusals) |
| `MapperContext` | Dataclass | Shared state (run ID, thread ID, message counter) per stream |
| `create_starlette_response` / `encode_sse_stream` | Function | Framework-agnostic SSE wrapping |
| `extract_resume_input` | Function | Pulls and JSON-decodes `RunAgentInput.forwarded_props["command"]["resume"]`. Returns `None` if absent or falsy |
| `agui_messages_to_langchain` | Function | Converts AG-UI protocol messages to LangChain `BaseMessage` instances. Maps `ToolMessage.error` to `status="error"`, extra fields to `additional_kwargs[AGUI_EXTRAS_KEY]`, and each content part with a `file` source to `additional_kwargs[AGUI_HELD_PARTS_KEY]`. Optional `drop_invalid_tool_calls=True` skips a tool call whose JSON arguments fail to parse |
| `detect_new_frontend_messages` | Function | `resume_factory` helper. Reads the stored messages from `checkpoint_state["reducers"][reducer_name]`, then converts only the inbound messages whose id is not stored. Returns a tuple of the new messages. A stored message wins over an inbound message with the same id |
| `build_langchain_tools` / `detect_new_tool_results` | Function | Convert `RunAgentInput.tools` to OpenAI format, or extract the new `ToolMessage`s from a resume request |

## Serde Subpackage

Namespace-aware serialization, auto-wired by `from_namespaces()`, see [above](#graph-execution). See [Event migrations](event-migrations.md).

Exported from `langgraph_events.serde` unless a different module is noted.

| Export | Type | Description |
|---|---|---|
| `NamespaceAwareSerde(migrations=(), *, namespaces=(), events=(), legacy_write=False)` | Class | `JsonPlusSerializer` subclass that keys `Event` identity by `(__module__, __qualname__)`. Usable as `serde=` for any checkpointer that accepts `serde=`. Auto-wired by `from_namespaces()`, see above. `legacy_write=True` is refused at construction with an origin-scoped fill, any transform or any split |
| `migrate_from` | Decorator | `@migrate_from(*old_qualnames, in_module=None, backfill=None, transform=None)` records historic identities a class held. `backfill={field: default}` scopes a fill to payloads written under the qualname of this decorator. `transform=fn` rewrites the stored kwargs of those payloads before the rename. A module-level `transform` that returns `{}` retires an identity onto an empty tombstone. Auto-collected from the namespaces the serde was built with. Stacks. Not MRO-inherited |
| `backfill` | Decorator | `@backfill(field, *, default=… \| default_factory=…)`: class-global fill for a field that is required in code but absent from old payloads. One value for every origin and era. The fallback under an origin-scoped fill. Same auto-collection as `@migrate_from`. Reuses the `AddField` mutable-default guard |
| `transform_fields` | Decorator | `@transform_fields(fn)`: the class stage. `fn` receives a copy of the stored kwargs and returns the kwargs the constructor receives. Runs after any rename, on payloads from every era, so it must be idempotent. One per class. See [Dropping, merging or retyping a field](event-migrations.md#dropping-merging-or-retyping-a-field) |
| `split_event` | Decorator | `@split_event(select, *, targets=(...))` on the source class. `select` receives a copy of the stored kwargs and returns `None` to keep the source, or `(target_class, kwargs)` to build one of `targets` instead. One per class. See [Splitting one stored event into two on a payload value](event-migrations.md#splitting-one-stored-event-into-two-on-a-payload-value) |
| `Migration` | Class | Hand-authored migration group for a cross-module or composite case. `Migration.rename(...)`, `.add_field(...)`, `.transform_fields(...)` and `.split_event(...)` are single-op constructors. Pass the live target as `to=<class>`, `target=<class>` or `source=<class>`. `add_field` or `transform_fields` keyed on a historic `module`/`qualname` is origin-scoped and applies before the rename. See [Event migrations](event-migrations.md) |
| `MigrationTarget(graph, baseline, store=None, threads=None)` | Class | `graph` takes a checkpointer and returns the `EventGraph` or an `EventStore`. When an `EventStore` checkpointer holds the default LangGraph serde, the target sets `topology.serde()` on it. `baseline` is the committed file. `store` returns a context manager of the real sync checkpointer. `threads` takes the saver and returns a `ThreadKey` per thread and namespace, needed for a saver other than `PostgresSaver`, `SqliteSaver` or `MemorySaver`, or for a store that more than one graph shares. `.snapshot()` builds the graph on an empty `MemorySaver`, `.build(saver)` on a given saver. `.open_store()` and `.thread_keys(saver)` call `store` and `threads`. Raises `TargetError` (`…serde.migrations`) when `graph` takes no checkpointer, or when the graph has no `NamespaceAwareSerde`. See [The migration gate](migration-gate.md#the-migration-gate) |
| `ThreadKey(thread_id, checkpoint_ns)` | Class (`…serde.migrations`) | One checkpoint namespace of one thread. `""` is the root checkpoint namespace |
| `list_thread_keys(saver) -> list[ThreadKey]` | Function (`…serde.migrations`) | Each thread and checkpoint namespace of any saver, sorted, through the public `saver.list(None)`. It reads every checkpoint, so it suits a small store only. See [Checkpointers](migration-gate.md#checkpointers) |
| `rewrite_live_set(target, *, expect, apply, passes=3, drop_retired=False, log=None) -> LiveSetOutcome \| WrongRelease` | Function (`…serde.migrations`) | The deploy step. Returns `WrongRelease` before it opens the store when the migrations of the running code do not have the fingerprint `expect`. A plan writes nothing. With `apply`, each pass rewrites each checkpoint that the last plan did not read unchanged, then plans again. With `drop_retired`, each plan and each pass drops the stored instances of each class in `retired=` of the topology that the target builds. See [Lifecycle of a read-time migration](migration-gate.md#lifecycle-of-a-read-time-migration) |
| `LiveSetOutcome` | Class (`…serde.migrations`) | `.passes` (one `RewriteReport` per apply pass), `.final` (the last plan), `.verdict` (the worst `Verdict` in the last plan), `.accepted(accept)` (`verdict <= accept`), `.clean`, `.render(Invocation, expect, accept=Verdict.CLEAN)` |
| `Verdict` | Enum (`…serde.migrations`) | `CLEAN`, `REWRITE`, `REFUSED`, in order from best to worst. The members compare in that order. `str()` gives the value, the word that `--accept` takes |
| `backup_live_set(target) -> BackupOutcome` | Function (`…serde.migrations`) | Copies the live set of a `PostgresSaver` store, in one transaction, into `checkpoints_bak_<fp>` and `checkpoint_writes_bak_<fp>`. A rerun keeps each row and adds the rows that changed since. Raises `TargetError` when the target sets `threads`, for another saver, or for a failed backup. See [Backup and restore](migration-gate.md#backup-and-restore) |
| `restore_live_set(target) -> RestoreOutcome` | Function (`…serde.migrations`) | Puts back the backup of the running code in one transaction. Raises `TargetError` when the target sets `threads`, for another saver, when the store holds no backup of the running code, when a blob version is missing, or for a failed restore. See [Backup and restore](migration-gate.md#backup-and-restore) |
| `drop_backups(target) -> tuple[str, ...]` | Function (`…serde.migrations`) | Drops each backup except the backup of the running code, in one transaction, and returns the fingerprint of each dropped backup. Raises `TargetError` when the target sets `threads`, for another saver, or for a failed drop. See [Backup and restore](migration-gate.md#backup-and-restore) |
| `BackupOutcome` | Class (`…serde.migrations`) | `.fingerprint`, `.kept` (the `BackupRows` the backup held before the run), `.added` (the `BackupRows` the run copied) |
| `RestoreOutcome` | Class (`…serde.migrations`) | `.fingerprint`, `.restored` (the `BackupRows` the restore put back), `.removed_writes` (the pending writes it deleted) |
| `BackupRows(checkpoints, writes)` | Class (`…serde.migrations`) | A count of checkpoint rows and of pending-write rows. `str()` gives `1 checkpoint and 2 writes` |
| `WrongRelease` | Class (`…serde.migrations`) | `.expected`, `.running` (fingerprints), `.running_labels`, `.render(Invocation, *, apply)` |
| `assert_migration_gate(target, *, previous=None)` | Function | Runs `check` and raises `GateError` on a finding. Without `previous` the release rule does not run |
| `GateError` | Exception | `AssertionError` raised by `assert_migration_gate`. `.report` is the `GateReport` |
| `check(target, *, previous=None) -> GateReport` | Function (`…serde.migrations`) | The one gate. Rules in order: `baseline`, `names`, `coverage`, `revive`, `handlers`, `release`. `previous` is the baseline file of the previous release. See [The rules](migration-gate.md#the-rules) |
| `GateReport` | Class (`…serde.migrations`) | `.findings`, `.rules_run`, `.release_not_run` (why the release rule did not run, or `None`), `.held`, `.passed`, `.render(Invocation)` |
| `BaselineStale`, `AnonymousCallable`, `Uncovered`, `Unrevivable`, `HandlerMissing`, `SurvivedRelease`, `MigrationOutOfScope` | Classes (`…serde.migrations`) | The typed findings. `Finding` is their union. Each has `.code`, `.subject` and `.lines(invocation)`, which ends with one next command |
| `Invocation(target, previous=None, drop_retired=False)` | Class (`…serde.migrations`) | The CLI arguments of the caller, so a finding prints a command to run. `.baseline_update(prune=False)`, `.baseline_hold(label)` and `.rewrite(expect, apply=False)` give each command. `.rewrite()` keeps `--drop-retired` when `drop_retired` is true |
| `update_baseline(target, *, previous=None)` | Function (`…serde.migrations`) | Writes the version 4 baseline the code needs. With `previous`, it resets the history of each shipped migration the code no longer holds to the `live` fields of `previous`. Without `previous`, it raises `PreviousRequiredError` when a recorded migration left the code |
| `PreviousRequiredError` | Exception (`…serde.migrations`) | `ValueError` raised by `update_baseline` when migrations left the code and `previous` is not given. `.deleted` holds the `MigrationKey` of each one. `.render(Invocation)` gives the message with the `baseline update --previous` command to run |
| `hold_migration(target, label, *, reason, previous)` | Function (`…serde.migrations`) | Keeps a shipped migration one more release. The hold records the digest of the `previous` file, so it expires in the next release. See [Labels](migration-gate.md#labels) |
| `MigrationSet.fingerprint`, `.shipped_in(previous_keys)`, `.deleted_since(previous_keys)` | Methods (`…serde.migrations`) | The fingerprint of the keys, and the comparison with the keys of a previous baseline. A previous key follows a current rename unless its identity was a rename source |
| CLI | Command | `python -m langgraph_events.serde.migrations check <target> [--previous FILE]`, `baseline update <target> [--previous FILE]`, `baseline hold <target> <label> --reason TEXT --previous FILE`, `fingerprint <baseline FILE>`, `backup <target>`, `rewrite <target> --expect FP [--apply] [--passes N] [--accept clean\|rewrite\|refused] [--drop-retired]`, `restore <target>`, `drop-backups <target>`. `<target>` is `module:attr` of a `MigrationTarget`. See [Exit codes](migration-gate.md#exit-codes) |
| `synthesize_legacy_payload(module, qualname, kwargs)` | Function | Builds the `(format, bytes)` a prior release would have written, to pin one drifted field shape. See [Testing your migrations](event-migrations.md#testing-your-migrations) |
| `replay_reducer(reducer, events)` | Function | Rebuilds the channel value of a reducer from its migrated event log when the projection shape changed. Thin wrapper over `BaseReducer.seed`. See [Reducer state migration](event-migrations.md#reducer-state-migration) |
| `NamespaceAwareSerde.revivable_identities` | Method | Read-only `frozenset` of every revivable `(module, qualname)`. The set the gate `coverage` rule accepts |
| `NamespaceAwareSerde.live_fields` | Method | `{(module, qualname): frozenset of init fields}` for every live event class the serde reads. The baseline records it as `live` |
| `assert_resume_recovers(before, after, *, seed, resume_with, thread_id=None, context=None)` | Function | Behavioral recovery proof: invokes `before` to pause, resumes `after` on the same checkpoint, and asserts a `Resumed`. Both graphs must share one checkpointer. See [Testing handler recovery](event-migrations.md#testing-handler-recovery) |

The raw operation constructors `RenameEvent`, `AddField`, `TransformFields` and `SplitEvent` live in `langgraph_events.serde.migrations` and are not re-exported at the top level. Wrap one like every raw op: `Migration(operations=(TransformFields(module, qualname, transform),))`. See [Event migrations](event-migrations.md).
