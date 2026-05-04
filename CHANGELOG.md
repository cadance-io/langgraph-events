# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.6.2] - 2026-05-04

### Fixed
- **mermaid**: `render_mermaid_choreography` no longer collapses cross-namespace events that share a leaf class name. Previously, a project with sibling namespaces (e.g. `Persona.Approve.Approved`, `Story.Approve.Approved`, `Scenario.Approve.Approved`) emitted a single mermaid node for all three, merging their incoming/outgoing edges. The renderer now detects leaf-name collisions across the model and escalates only the colliding classes to qualname-based node IDs (`Persona_Approve_Approved`, …); display labels stay terse since the surrounding subgraph cluster already conveys namespace context. Non-colliding diagrams render byte-identically. (#62)

## [0.6.1] - 2026-05-04

### Fixed
- **serde**: `NamespaceAwareSerde` now preserves namespace identity for `Event` instances nested inside `langgraph.types.Interrupt` (the dataclass LangGraph wraps every interrupted value in before checkpointing). Previously, every namespaced `Interrupted`/`InterruptedWithPayload` subclass round-tripped through a checkpointer would silently decode back as `Interrupt(value=None, id=...)` because LangGraph's generic dataclass branch (`EXT_CONSTRUCTOR_KW_ARGS`) recurses into a hardcoded `_msgpack_default` and bypassed our namespace-aware `default=`. `Interrupt` is now intercepted directly under a dedicated ext code so the wrapped value re-enters our encoder. Note: this fix applies to checkpoints written *after* the upgrade — checkpoints already persisted under v0.6.0 still decode their nested events as `None` (same risk profile as before). (#60)

## [0.6.0] - 2026-05-04

### Added
- **EventGraph**: `services=` kwarg for dependency injection in two forms. (1) `services=[chat_model, session_factory]` — type-keyed; handler params resolve by their type annotation via an MRO walk (a base-class annotation matches a registered subclass instance), with exact-type match preferred over subclass match. (2) `services={"primary_chat": a, "backup_chat": b}` — name-keyed; handler params resolve by name. The mapping form allows multiple instances of the same type. Inline `Command.handle(self, chat_model: BaseChatModel)` and external `@on(...)` handlers share the same mechanism. Resolution order: reducer name → framework type (`EventLog` / `RunnableConfig` / `BaseStore`) → service. Eliminates the closure-factory pattern downstream projects use today to shuttle services into handlers.
- **serde**: new opt-in `langgraph_events.serde.NamespaceAwareSerde` — a `JsonPlusSerializer` subclass that keys `Event` identity by `(__module__, __qualname__)` instead of `(__module__, __name__)`. Drop-in for any LangGraph checkpointer that accepts `serde=` (e.g. `MemorySaver(serde=NamespaceAwareSerde())`). Two namespaces with sibling-named events (`Persona.Approve.Approved`, `Story.Approve.Approved`) now round-trip distinctly; non-event payloads encode exactly as the default serde.
- **InterruptedWithPayload[PayloadT]** (in `langgraph_events.agui`): new generic base for HITL with a discriminated frontend payload. Subclasses implement `interrupt_payload(self) -> PayloadT` and inherit from `Interrupted`. The AG-UI `InterruptedMapper` recognises it directly — no `agui_dict()` override needed. Eliminates the project-local "shim base" pattern downstream HITL projects use today to break import cycles between sibling namespace modules.
- **on_namespace_finalize(cls, callback)**: public hook that schedules a callback to fire once the enclosing Namespace's `__init_subclass__` finishes (after `_stamp_nested_namespace` and `_attach_command_outcomes`). Useful for class decorators that need to call `typing.get_type_hints()` against forward references to siblings inside the same in-progress Namespace body — those references can't resolve while the class body is evaluating, but resolve cleanly at finalize time. The callback receives `(cls, namespace_cls)` so decorators can resolve siblings via `vars(namespace_cls)` without touching private state. Re-exported at `langgraph_events.on_namespace_finalize`.

### Changed
- **EventGraph**: handler params with no injection source now raise `TypeError` at graph construction (previously crashed at first dispatch with a missing-keyword error). Two services of the same exact type are rejected at construction. A handler param annotated as a class that matches multiple registered services is rejected at construction with both candidate type names in the message. Resolution prefers an exact-type service match over subclass matches, so `services=[BaseChatModel(), Anthropic()]` cleanly resolves a `param: BaseChatModel` to the base instance. Annotations equal to `object` are skipped — they would otherwise silently match every registered service.
- **agui**: `FrontendToolCallRequested` (previously top-level `langgraph_events.FrontendToolCallRequested`) now lives in `langgraph_events.agui` alongside `FrontendStateMutated`. Update imports to `from langgraph_events.agui import FrontendToolCallRequested`. **The top-level alias still resolves**, but emits a `DeprecationWarning` per access pointing at the new path; it will be removed in a future release. The class itself is unchanged.

### Fixed
- **on_namespace_finalize**: callbacks registered after the enclosing Namespace has already finalized now fire immediately rather than dangling in the registry forever. Previously a decorator applied post-hoc to an already-bound class was a silent no-op.
- **EventGraph**: variadic handler params (`*args` / `**kwargs`) are no longer flagged as unclaimed at graph construction.
- **serde**: `NamespaceAwareSerde.dumps_typed` now warns when an unencodable payload forces fallback to the upstream serializer (which uses leaf-name identity, collision-prone for nested events). `loads_typed` now raises a clear `ValueError` naming the missing class when a checkpoint references an Event class that has been renamed or removed, rather than the opaque `ValueError("ext_hook failed")` from upstream. Imports of LangGraph private helpers are now guarded with a clear `ImportError` if upstream renames them.

## [0.5.2] - 2026-04-30

### Added
- **agui**: `AGUIAdapter(include_reducers=...)` validation — malformed values (anything other than `bool | list[str]`) now raise `TypeError` at construction instead of silently producing empty snapshots at runtime.

### Fixed
- **agui**: `AGUIAdapter.connect()` and the streaming `StateSnapshotEvent` path no longer leak the EventGraph-internal `events` audit log to clients. The audit log is graph-internal and was causing O(history) wire bloat on every client `Send` via `RunAgentInput.state` round-trip. The strip set is now derived from `_internal._BASE_FIELDS` (single source of truth across all four projection sites) rather than hardcoded; future internal channels propagate automatically. `_extract_frontend_state` also strips internal keys as defense-in-depth against stale-client echo.
- **agui**: Resume-time frontend state now flows through `FrontendStateMutated` instead of bypassing dispatch via `apre_seed(raw_state)`. The adapter computes per-reducer contributions from the FSM event (preserving `fn` semantics — transformations, `SKIP`) and writes them to channels via `apre_seed` *before* the resume's domain dispatch, then injects FSM as a seed to `astream_resume` so it appears in the output stream and the persisted audit log. Reducers that subscribe to `FrontendStateMutated` see the same contract on resume as on the non-resume path; reducers that subscribe to backend domain events are no longer clobbered by stale frontend snapshot keys. `@on(FrontendStateMutated)` *handlers* still do not fire on resume — `Command(resume=...)` carries one value and seeds dispatch out-of-graph; use `@on(Resumed)` for resume-time side effects.

### Changed
- **agui**: `AGUIAdapter.__init__` validates the `messages` reducer eagerly (raises `ValueError` immediately, before any other setup).
- **EventGraph**: `stream_resume` and `astream_resume` now accept a `seeds: list[Event] | None = None` kwarg. Seeds are dispatched alongside the resume in the same step — used by `AGUIAdapter` to route `FrontendStateMutated` through reducers on resume. Power users can plumb their own resume-time companion events through this hook.

## [0.5.1] - 2026-04-24

### Added
- `AGUIAdapter` now emits a new built-in `FrontendStateMutated` event (an `IntegrationEvent`, exported from `langgraph_events.agui`) as the first event of each run when `RunAgentInput.state` carries non-empty client-owned state. The dedicated `messages` key is filtered out (driven by `MessagesSnapshotEvent` / `TextMessageEvent`). Reducers mirroring client-driven channels subscribe to it like any other event — e.g. `ScalarReducer(event_type=FrontendStateMutated, fn=lambda e: e.state.get("focus", SKIP))` — and handlers can optionally `@on(FrontendStateMutated)` to react. On resume, the adapter writes the filtered state directly to reducer channels via `graph.apre_seed` before calling `astream_resume` (LangGraph's `ainvoke` on a pending-interrupt thread would consume the interrupt); in the idiomatic state-key-equals-channel-name pattern the values flow through identically to the non-resume path. `FrontendStateMutated` is not echoed back to the client — its downstream reducer changes surface via the usual `StateSnapshotEvent` path. Works with or without a checkpointer on the non-resume path.

## [0.5.0] - 2026-04-22

### Added
- Event taxonomy: `Namespace`, `Command`, `DomainEvent`, `IntegrationEvent`, `SystemEvent`. `Namespace` subclasses act as namespaces for nested commands and outcomes, encoding the `Domain.Command.Outcomes` pattern (`Order.Place.Placed`) directly in Python's class structure. Class-creation enforcement: `Command` subclasses must be nested in a `Namespace`, `DomainEvent` subclasses must be nested in a `Namespace` or `Command`. Existing framework events (`Halted`, `Interrupted`, `Resumed`, `HandlerRaised`, `Cancelled`, `MaxRoundsExceeded`) gain `SystemEvent` as a parent — backwards-compatible since they still inherit `Event` transitively. See `examples/order.py`.
- `Command.Outcomes` — auto-generated union of a command's nested `DomainEvent` classes. Used for `isinstance` checks, introspection (`typing.get_args(Command.Outcomes)`), and as the fallback runtime contract for handlers subscribed to a command. Users may declare `Outcomes` explicitly for `mypy` visibility; the framework validates drift against the nested events at class creation.
- Inline command handlers — a `handle` method defined directly on a `Command` class auto-registers as that command's handler when the class is passed to `EventGraph` or via `EventGraph.from_namespaces(*domains, handlers=...)`. `self` is the command event. Existing `@on(...)` handlers still work and compose in the same graph.
- Strict return-type enforcement — at dispatch, the framework validates handler returns against (a) the declared return annotation, or (b) the subscribed command's `Outcomes` when no annotation is present. Violations raise `TypeError` at the handler's dispatch. Unannotated non-`Command`-subscribing handlers keep the legacy shape-only check.
- Declarative domain reducers — `Reducer` / `ScalarReducer` instances declared as class attributes inside a `Namespace` are auto-named (from the attribute), auto-scoped (only that domain's events contribute), and auto-discovered by `EventGraph` via any handler subscribed to the domain's events. Explicit `reducers=[...]` kwarg still works for graph-wide reducers. Child domains inherit parent reducers via MRO.
- `Invariant` marker base class — subclass to declare a typed invariant (e.g. `class CustomerNotBanned(Invariant)`). The subclass identity drives matching; zero-arg instantiable.
- `invariants=` parameter on `@on()` — dict mapping typed `Invariant` subclasses to sync predicates (`invariants={CustomerNotBanned: lambda log: not log.has(CustomerBanned)}`). Evaluated in two phases per matching event: **pre-check** (before the handler runs, against the current log) and **post-check** (after the handler returns, against `log + emitted events`). Pre-check failure skips the handler; post-check failure drops the handler's emitted events and commits `InvariantViolated` in their place with `would_emit: tuple[Event, ...]` carrying the rolled-back events. Pin a reaction with `@on(InvariantViolated, invariant=CustomerNotBanned)` — pinned reactors fire for both phases without distinguishing. Multiple invariants short-circuit on first failure; async predicates are rejected at decoration; predicate exceptions propagate. Compile-time drift check raises `TypeError` when a pinned `invariant=` matcher references a class no handler declares ("would never fire"). Predicates must be pure functions of `log` — the same predicate runs in both phases. See `examples/order.py`.
- `EventGraph.namespaces()` — returns a `NamespaceModel`: a code-derived snapshot of the graph's structure with two lenses (`view="structure"` — domains → commands → outcomes taxonomy; `view="choreography"` — full event flow with handlers, policies, edges, seeds). Renderers: `text()`, `mermaid()`, `json()` / `to_dict()`. Nested frozen dataclasses (`NamespaceModel.Namespace`, `Command`, `CommandHandler`, `Policy`, `Edge`, `Invariant`) replace the prior `Catalog` / `AggregateEntry` / `CommandEntry` TypedDicts.
- `NamespaceModel.invariants` — first-class node for every declared invariant, with `cls`, `commands` (owning commands), `declared_by` (handler names), and `reactors` (pinned `@on(InvariantViolated, invariant=…)` handler names). Surfaced in `graph.namespaces().text()` as an `Invariants:` section and in `graph.namespaces().mermaid()` as a diamond gate node styled `:::inv` inside the owning domain's subgraph. When an invariant has a pinned reactor, the reactor's output edge leaves the Invariant diamond directly — one clean chain `Command -.->|invariant| Invariant -.->|reactor| Outcome` instead of a disconnected gate and a separate `InvariantViolated` stadium node. The `InvariantViolated` node drops from the diagram entirely when every reactor is pinned (no catch-all `@on(InvariantViolated)`). Ownership-gap arrows are suppressed for outcomes already reached via an invariant chain.
- `EventGraph.from_namespaces(*domains, handlers=None, **kwargs)` — classmethod factory that walks domains' namespaces, auto-registers every command with an inline `handle`, and appends any extra external handlers.
- `@on()` field matchers now accept `str` values for equality match alongside `type` values for `isinstance` match.
- `@on` is now polymorphic: `@on` (bare) and `@on(kwargs=...)` infer the event type from the handler's first parameter annotation, removing the common duplication between decorator argument and annotation. `@on(Type, ...)` remains the explicit form for multi-event subscription or any case where you prefer not to rely on inference. Errors at decoration if the annotation is missing or not a single `Event` subclass, and the error points to the explicit form.
- New examples: `expense_approval.py` (human-in-the-loop approval with Interrupted/resume), `conversation.py` (tool-calling agent with content moderation and AG-UI frontend tools end-to-end). `examples/order.py` grows a `ScalarReducer` attribute + pinned `@on(InvariantViolated, invariant=…)` reaction to illustrate those idioms in the canonical example.

### Changed
- `Reducer` and `ScalarReducer` fields beyond `name` are now keyword-only. Positional calls `Reducer(name, event_type, fn)` break; switch to `Reducer(name="x", event_type=..., fn=...)`. Calls already using kwargs are unaffected. See `docs/migrating.md`.
- Bare `Event` subclassing now raises `TypeError` — use `DomainEvent`, `IntegrationEvent`, `Command`, or `SystemEvent`. See `docs/migrating.md`.
- Handler return types are enforced at dispatch — mismatches raise `TypeError`. Unannotated non-`Command` handlers keep the legacy shape-only check. See `docs/migrating.md`.
- `Auditable` and `MessageEvent` are plain mixins — compose with an event branch (e.g. `IntegrationEvent, Auditable`). See `docs/migrating.md`.
- `SystemPromptSet` is now an `IntegrationEvent` (was `SystemEvent`). `SystemEvent` is reserved for framework-emitted facts; system prompts are user-seeded input. `@on(SystemEvent)` catch-alls and `isinstance(evt, SystemEvent)` branches that treated system prompts as framework signals need updating. See `docs/migrating.md`.

### Removed
- `EventGraph.catalog()` / `EventGraph.describe()` and the `Catalog` / `AggregateEntry` / `CommandEntry` TypedDicts — superseded by `EventGraph.namespaces()` + `NamespaceModel`. See `docs/migrating.md`.
- `examples/react_agent.py` — redundant with `examples/conversation.py` (same send/classify/LLM+tools shape, now reshaped as a `Conversation` namespace).
- `examples/reflection_loop.py` — multi-subscription loop pattern covered by `examples/conversation.py`.
- `examples/human_in_the_loop.py` — HITL pattern subsumed by `examples/expense_approval.py`.
- `examples/agui_frontend_tools.py` — LLM-initiated AG-UI frontend-tool flow folded into `examples/conversation.py`.
- `examples/agui_confirm_dialog.py` — handler-initiated `FrontendToolCallRequested` snippet moved to `docs/agui.md`.

## [0.4.0] - 2026-04-20

### Added
- `raises=` parameter on `@on()` — declare exceptions the framework should catch from a handler. Caught exceptions are surfaced as the new built-in `HandlerRaised` event carrying the raising handler's name (`handler`), the event being processed (`source_event`), and the raw exception (`exception`). Subscribe with `@on(HandlerRaised, exception=MyError)` to react (retry, back off, halt) without try/except boilerplate. Compile-time validation fails if a declared exception has no matching catcher; catchers that add a non-`exception` field matcher (e.g. `source_event=SomeType`) are conservatively not counted toward coverage and must be paired with a broader catcher. Framework-level errors (e.g. calling `invoke()` on an async handler from within a running event loop) are raised outside the `raises=` catch boundary and cannot be swallowed by a broad `raises=Exception`. `exception=` field matchers reject non-`Exception` `BaseException` subclasses (symmetric with `raises=`). See `examples/error_recovery.py`.
- AG-UI frontend tools — `useFrontendTool` (CopilotKit v2) is now idiomatic against an `EventGraph`. The adapter streams `AIMessageChunk.tool_call_chunks` as `ToolCallStart`/`ToolCallArgs`/`ToolCallEnd` (new `LLMToolCallChunk` frame + adapter wiring), and a new built-in `FrontendToolCallRequested(Interrupted)` event maps to the same triple for handler-initiated flows — tool calls become "HITL with typed fields," mirroring `ApprovalRequested(Interrupted)`. Two new helpers in `langgraph_events.agui`: `build_langchain_tools(input_data.tools)` converts AG-UI tool defs to OpenAI-format bindings for `llm.bind_tools(...)`; `detect_new_tool_results(input_data, checkpoint_state)` returns inbound tool messages not yet in the checkpoint so `resume_factory` can return a `MessageEvent` and continue the graph. See `examples/agui_frontend_tools.py` (LLM-initiated) and `examples/agui_confirm_dialog.py` (handler-initiated).

### Changed
- AG-UI frontend-tool plumbing now raises `ValueError` on contract violations instead of silently coercing missing fields. Triggers: `FrontendToolCallRequested(name="")` (or whitespace), an LLM `tool_call_chunk` lacking `index`, the first chunk of a streaming call lacking `id` or `name`, and an inbound `role: "tool"` message lacking `tool_call_id`. The streaming-path errors propagate through the existing `AGUIAdapter.stream()` top-level handler and surface as a `RUN_ERROR` event with the diagnostic message; conformant CopilotKit clients and LangChain chat models are unaffected.

## [0.3.0] - 2026-04-13

### Added
- `EventGraph.apre_seed()` — async counterpart to `pre_seed()`

## [0.2.1] - 2026-04-12

### Fixed
- Init reducer state from checkpoint in `_astream_v2` and `make_seed_node`

## [0.2.0] - 2026-04-06

### Added
- Field-level dispatch for `@on` decorator
- AG-UI protocol adapter with `langgraph-events[agui]` optional dependency
- Custom event emit helpers (`emit_custom`, `aemit_custom`, `emit_state_snapshot`, `aemit_state_snapshot`)
- First-class `StateSnapshotFrame` for state snapshot streaming
- `SKIP` sentinel for scalar reducer no-op returns
- Graceful `Halted` subtypes (`MaxRoundsExceeded`, `Cancelled`) and `OrphanedEventWarning`
- LLM token streaming (`LLMToken`, `LLMStreamEnd` frames)
- MkDocs documentation site on GitHub Pages

### Changed
- Restructured docs for better DX (split concepts, grouped API reference)
- `Interrupted` is now a bare marker class — subclass with typed fields

### Fixed
- AG-UI adapter message deduplication and ID reconciliation
- `connect()` yielding no events for new threads
- Resume interrupt detection for interrupts created during resume

## [0.1.0] - 2026-02-20

### Added
- Core `Event`, `EventGraph`, and `@on` decorator
- `Reducer` and `ScalarReducer` for custom state channels
- `EventLog` with query methods (`first`, `count`, `after`, `before`, `select`)
- Multi-subscription `@on(A, B)` and `Scatter` for fan-out
- `Auditable` and `MessageEvent` base events
- `SystemPromptSet` event for system prompts
- Config and store injection for handlers
- `Interrupted` / `Resumed` events for human-in-the-loop
- Mermaid diagram generation
- BDD-style test suite with pytest-describe
- CI workflow (lint, typecheck, test)

[Unreleased]: https://github.com/cadance-io/langgraph-events/compare/v0.6.2...HEAD
[0.6.2]: https://github.com/cadance-io/langgraph-events/compare/v0.6.1...v0.6.2
[0.6.1]: https://github.com/cadance-io/langgraph-events/compare/v0.6.0...v0.6.1
[0.6.0]: https://github.com/cadance-io/langgraph-events/compare/v0.5.2...v0.6.0
[0.5.2]: https://github.com/cadance-io/langgraph-events/compare/v0.5.1...v0.5.2
[0.5.1]: https://github.com/cadance-io/langgraph-events/compare/v0.5.0...v0.5.1
[0.5.0]: https://github.com/cadance-io/langgraph-events/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/cadance-io/langgraph-events/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/cadance-io/langgraph-events/compare/v0.2.1...v0.3.0
[0.2.1]: https://github.com/cadance-io/langgraph-events/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/cadance-io/langgraph-events/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/cadance-io/langgraph-events/releases/tag/v0.1.0
