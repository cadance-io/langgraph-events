# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

[Unreleased]: https://github.com/cadance-io/langgraph-events/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/cadance-io/langgraph-events/compare/v0.2.1...v0.3.0
[0.2.1]: https://github.com/cadance-io/langgraph-events/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/cadance-io/langgraph-events/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/cadance-io/langgraph-events/releases/tag/v0.1.0
