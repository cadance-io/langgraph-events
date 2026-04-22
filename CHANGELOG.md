# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

[Unreleased]: https://github.com/cadance-io/langgraph-events/compare/v0.5.0...HEAD
[0.5.0]: https://github.com/cadance-io/langgraph-events/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/cadance-io/langgraph-events/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/cadance-io/langgraph-events/compare/v0.2.1...v0.3.0
[0.2.1]: https://github.com/cadance-io/langgraph-events/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/cadance-io/langgraph-events/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/cadance-io/langgraph-events/releases/tag/v0.1.0
