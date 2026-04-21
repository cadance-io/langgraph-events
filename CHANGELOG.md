# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- `DomainModel.invariants` — first-class domain node for every declared invariant, with `cls`, `commands` (owning commands), `declared_by` (handler names), and `reactors` (pinned `@on(InvariantViolated, invariant=…)` handler names). Surfaced in `graph.domain().text()` as an `Invariants:` section and in `graph.domain().mermaid()` as a diamond gate node styled `:::inv` inside the owning aggregate's subgraph with a dashed `invariant` edge from the gated command.

### Changed
- `invariants=` on `@on()` now takes typed `Invariant` subclasses as dict keys instead of strings — `invariants={CustomerNotBanned: lambda log: ...}`. `InvariantViolated.name` is replaced with `invariant` (an instance of the declared `Invariant` subclass). Reactor matchers move from `@on(InvariantViolated, name="...")` to `@on(InvariantViolated, invariant=InvariantClass)`. A new compile-time drift check raises `TypeError` when a pinned `invariant=` matcher references a class no handler declares ("would never fire"). Subclass `Invariant` once per rule; predicate stays inline in the decorator. See `examples/ddd_order.py`.

## [0.4.0] - 2026-04-20

### Added
- DDD-aligned event taxonomy: `Aggregate`, `Command`, `DomainEvent`, `IntegrationEvent`, `SystemEvent`. Aggregates act as namespaces for nested commands and outcomes, encoding the `Aggregate.Command.Outcomes` pattern (`Order.Place.Placed`) directly in Python's class structure. Class-creation enforcement: `Command` subclasses must be nested in an `Aggregate`, `DomainEvent` subclasses must be nested in an `Aggregate` or `Command`. Existing framework events (`Halted`, `Interrupted`, `Resumed`, `HandlerRaised`, `Cancelled`, `MaxRoundsExceeded`) gain `SystemEvent` as a parent — backwards-compatible since they still inherit `Event` transitively. See `examples/ddd_order.py`.
- `Command.Outcomes` — auto-generated union of a command's nested `DomainEvent` classes. Used for `isinstance` checks, introspection (`typing.get_args(Command.Outcomes)`), and as the fallback runtime contract for handlers subscribed to a command. Users may declare `Outcomes` explicitly for `mypy` visibility; the framework validates drift against the nested events at class creation.
- Inline command handlers — a `handle` method defined directly on a `Command` class auto-registers as that command's handler when the class is passed to `EventGraph` or via `EventGraph.from_aggregates(*aggregates, handlers=...)`. `self` is the command event. Existing `@on(...)` handlers still work and compose in the same graph.
- Strict return-type enforcement — at dispatch, the framework validates handler returns against (a) the declared return annotation, or (b) the subscribed command's `Outcomes` when no annotation is present. Violations raise `TypeError` at the handler's dispatch. Unannotated non-`Command`-subscribing handlers keep the legacy shape-only check.
- Declarative aggregate reducers — `Reducer` / `ScalarReducer` instances declared as class attributes inside an `Aggregate` are auto-named (from the attribute), auto-scoped (only that aggregate's events contribute), and auto-discovered by `EventGraph` via any handler subscribed to the aggregate's events. Explicit `reducers=[...]` kwarg still works for graph-wide reducers. Child aggregates inherit parent reducers via MRO.
- `invariants=` parameter on `@on()` — dict mapping an invariant's name to a sync predicate (predicate receives the current `EventLog`). On a false predicate, the handler is skipped and the framework emits the new built-in `InvariantViolated` event (`name`, `handler`, `source_event` fields). Pin a reaction to a specific invariant with `@on(InvariantViolated, name="...")`; catch all with `@on(InvariantViolated)`. Multiple invariants short-circuit on first failure; async predicates are rejected at decoration; predicate exceptions propagate.
- `EventGraph.domain()` — returns a `DomainModel`: a code-derived snapshot of the graph's domain with two lenses (`view="structure"` — aggregates → commands → outcomes taxonomy; `view="choreography"` — full event flow with handlers, policies, edges, seeds). Renderers: `text()`, `mermaid()`, `json()` / `to_dict()`. Nested frozen dataclasses (`DomainModel.Aggregate`, `Command`, `CommandHandler`, `Policy`, `Edge`) replace the prior `Catalog` / `AggregateEntry` / `CommandEntry` TypedDicts.
- `EventGraph.from_aggregates(*aggregates, handlers=None, **kwargs)` — classmethod factory that walks aggregates' namespaces, auto-registers every command with an inline `handle`, and appends any extra external handlers.
- `@on()` field matchers now accept `str` values for equality match alongside `type` values for `isinstance` match. Enables `@on(InvariantViolated, name="customer not banned")`.
- `@on` is now polymorphic: `@on` (bare) and `@on(kwargs=...)` infer the event type from the handler's first parameter annotation, removing the common duplication between decorator argument and annotation. `@on(Type, ...)` remains the explicit form for multi-event subscription or any case where you prefer not to rely on inference. Errors at decoration if the annotation is missing or not a single `Event` subclass, and the error points to the explicit form.
- `raises=` parameter on `@on()` — declare exceptions the framework should catch from a handler. Caught exceptions are surfaced as the new built-in `HandlerRaised` event carrying the raising handler's name (`handler`), the event being processed (`source_event`), and the raw exception (`exception`). Subscribe with `@on(HandlerRaised, exception=MyError)` to react (retry, back off, halt) without try/except boilerplate. Compile-time validation fails if a declared exception has no matching catcher; catchers that add a non-`exception` field matcher (e.g. `source_event=SomeType`) are conservatively not counted toward coverage and must be paired with a broader catcher. Framework-level errors (e.g. calling `invoke()` on an async handler from within a running event loop) are raised outside the `raises=` catch boundary and cannot be swallowed by a broad `raises=Exception`. `exception=` field matchers reject non-`Exception` `BaseException` subclasses (symmetric with `raises=`). See `examples/error_recovery.py`.
- AG-UI frontend tools — `useFrontendTool` (CopilotKit v2) is now idiomatic against an `EventGraph`. The adapter streams `AIMessageChunk.tool_call_chunks` as `ToolCallStart`/`ToolCallArgs`/`ToolCallEnd` (new `LLMToolCallChunk` frame + adapter wiring), and a new built-in `FrontendToolCallRequested(Interrupted)` event maps to the same triple for handler-initiated flows — tool calls become "HITL with typed fields," mirroring `ApprovalRequested(Interrupted)`. Two new helpers in `langgraph_events.agui`: `build_langchain_tools(input_data.tools)` converts AG-UI tool defs to OpenAI-format bindings for `llm.bind_tools(...)`; `detect_new_tool_results(input_data, checkpoint_state)` returns inbound tool messages not yet in the checkpoint so `resume_factory` can return a `MessageEvent` and continue the graph. See `examples/ddd_conversation.py` for the LLM-initiated streaming path inside a DDD aggregate, and `docs/agui.md` for the handler-initiated snippet.
- New examples: `ddd_expense_approval.py` (DDD + human-in-the-loop approval with Interrupted/resume), `ddd_conversation.py` (DDD aggregate wrapping a ReAct tool-calling agent, content moderation, and AG-UI frontend tools end-to-end). `examples/ddd_order.py` grows a `ScalarReducer` attribute + pinned `@on(InvariantViolated, name="…")` reaction to illustrate those idioms in the canonical example.

### Changed
- AG-UI frontend-tool plumbing now raises `ValueError` on contract violations instead of silently coercing missing fields. Triggers: `FrontendToolCallRequested(name="")` (or whitespace), an LLM `tool_call_chunk` lacking `index`, the first chunk of a streaming call lacking `id` or `name`, and an inbound `role: "tool"` message lacking `tool_call_id`. The streaming-path errors propagate through the existing `AGUIAdapter.stream()` top-level handler and surface as a `RUN_ERROR` event with the diagnostic message; conformant CopilotKit clients and LangChain chat models are unaffected.
- `Reducer` and `ScalarReducer` fields beyond `name` are now keyword-only. Positional calls `Reducer(name, event_type, fn)` break; switch to `Reducer(name="x", event_type=..., fn=...)`. Calls already using kwargs are unaffected. See `docs/migrating.md`.
- Bare `Event` subclassing now raises `TypeError` — use `DomainEvent`, `IntegrationEvent`, `Command`, or `SystemEvent`. See `docs/migrating.md`.
- Handler return types are enforced at dispatch — mismatches raise `TypeError`. Unannotated non-`Command` handlers keep the legacy shape-only check. See `docs/migrating.md`.
- `Auditable` and `MessageEvent` are plain mixins — compose with an event branch (e.g. `IntegrationEvent, Auditable`). See `docs/migrating.md`.
- `SystemPromptSet` is now an `IntegrationEvent` (was `SystemEvent`). `SystemEvent` is reserved for framework-emitted facts; system prompts are user-seeded input. `@on(SystemEvent)` catch-alls and `isinstance(evt, SystemEvent)` branches that treated system prompts as framework signals need updating. See `docs/migrating.md`.

### Removed
- `EventGraph.catalog()` / `EventGraph.describe()` and the `Catalog` / `AggregateEntry` / `CommandEntry` TypedDicts — superseded by `EventGraph.domain()` + `DomainModel`. See `docs/migrating.md`.
- `examples/react_agent.py` — redundant with `examples/ddd_conversation.py` (same send/classify/LLM+tools shape, now reshaped as a `Conversation` aggregate).
- `examples/reflection_loop.py` — multi-subscription loop pattern covered by `examples/ddd_conversation.py`.
- `examples/human_in_the_loop.py` — HITL pattern subsumed by `ddd_expense_approval.py` which adds DDD on top.

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

[Unreleased]: https://github.com/cadance-io/langgraph-events/compare/v0.4.0...HEAD
[0.4.0]: https://github.com/cadance-io/langgraph-events/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/cadance-io/langgraph-events/compare/v0.2.1...v0.3.0
[0.2.1]: https://github.com/cadance-io/langgraph-events/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/cadance-io/langgraph-events/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/cadance-io/langgraph-events/releases/tag/v0.1.0
