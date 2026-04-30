# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **agui**: `AGUIAdapter(include_reducers=...)` validation — malformed values (anything other than `bool | list[str]`) now raise `TypeError` at construction instead of silently producing empty snapshots at runtime.

### Fixed
- **agui**: `AGUIAdapter.connect()` and the streaming `StateSnapshotEvent` path no longer leak the EventGraph-internal `events` audit log to clients. The audit log is graph-internal and was causing O(history) wire bloat on every client `Send` via `RunAgentInput.state` round-trip. The strip set is now derived from `_internal._BASE_FIELDS` (single source of truth across all four projection sites) rather than hardcoded; future internal channels propagate automatically. `_extract_frontend_state` also strips internal keys as defense-in-depth against stale-client echo.
- **agui**: Resume-time frontend state now flows through `FrontendStateMutated` instead of bypassing dispatch via `apre_seed(raw_state)`. The adapter computes per-reducer contributions from the FSM event (preserving `fn` semantics — transformations, `SKIP`) and writes them to channels via `apre_seed` *before* the resume's domain dispatch, then injects FSM as a seed to `astream_resume` so it appears in the output stream and the persisted audit log. Reducers that subscribe to `FrontendStateMutated` see the same contract on resume as on the non-resume path; reducers that subscribe to backend domain events are no longer clobbered by stale frontend snapshot keys (closes the cadance walkthrough-resume hang). `@on(FrontendStateMutated)` *handlers* still do not fire on resume — `Command(resume=...)` carries one value and seeds dispatch out-of-graph; use `@on(Resumed)` for resume-time side effects.

### Changed
- **agui**: `AGUIAdapter.__init__` validates the `messages` reducer eagerly (raises `ValueError` immediately, before any other setup).
- **EventGraph**: `stream_resume` and `astream_resume` now accept a `seeds: list[Event] | None = None` kwarg. Seeds are dispatched alongside the resume in the same step — used by `AGUIAdapter` to route `FrontendStateMutated` through reducers on resume. Power users can plumb their own resume-time companion events through this hook.

## [0.4.1] - 2026-04-24

### Added
- `AGUIAdapter` now emits a new built-in `FrontendStateMutated` event (an `Event`, exported from `langgraph_events.agui`) as the first event of each run when `RunAgentInput.state` carries non-empty client-owned state. The dedicated `messages` key is filtered out (driven by `MessagesSnapshotEvent` / `TextMessageEvent`). Reducers mirroring client-driven channels subscribe to it like any other event — e.g. `ScalarReducer(event_type=FrontendStateMutated, fn=lambda e: e.state.get("focus", SKIP))` — and handlers can optionally `@on(FrontendStateMutated)` to react. On resume, the adapter writes the filtered state directly to reducer channels via `graph.apre_seed` before calling `astream_resume` (LangGraph's `ainvoke` on a pending-interrupt thread would consume the interrupt); in the idiomatic state-key-equals-channel-name pattern the values flow through identically to the non-resume path. `FrontendStateMutated` is not echoed back to the client — its downstream reducer changes surface via the usual `StateSnapshotEvent` path. Works with or without a checkpointer on the non-resume path.

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

[Unreleased]: https://github.com/cadance-io/langgraph-events/compare/v0.4.1...HEAD
[0.4.1]: https://github.com/cadance-io/langgraph-events/compare/v0.4.0...v0.4.1
[0.4.0]: https://github.com/cadance-io/langgraph-events/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/cadance-io/langgraph-events/compare/v0.2.1...v0.3.0
[0.2.1]: https://github.com/cadance-io/langgraph-events/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/cadance-io/langgraph-events/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/cadance-io/langgraph-events/releases/tag/v0.1.0
