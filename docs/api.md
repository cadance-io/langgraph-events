# API Reference

This is a concise API index. For detailed behavior, see [Core Concepts](concepts.md) and examples in `examples/`.

## Core

- `Event` - base class for events (auto-frozen dataclasses)
- `on` - decorator for handler subscription
- `EventGraph` - compile and execute event-driven graphs
- `EventLog` - immutable log with query helpers

## EventGraph Methods

- `invoke()` / `ainvoke()`
- `stream_events()` / `astream_events()`
- `resume()` / `aresume()`
- `stream_resume()` / `astream_resume()`
- `get_state()`
- `mermaid()`
- `compiled` property for underlying `CompiledStateGraph`

## Built-in Event Types / Helpers

- `Halted`
- `Interrupted`
- `Resumed`
- `Scatter`
- `Auditable`
- `MessageEvent`
- `SystemPromptSet`

## Reducers

- `Reducer`
- `ScalarReducer`
- `message_reducer()`

## Streaming Frames

- `StreamFrame`
- `LLMToken`
- `LLMStreamEnd`
- `CustomEventFrame`
- `StateSnapshotFrame`
- `STATE_SNAPSHOT_EVENT_NAME`
- `emit_custom()` / `aemit_custom()`
- `emit_state_snapshot()` / `aemit_state_snapshot()`

## AG-UI Subpackage

- `langgraph_events.agui.AGUIAdapter`
- `create_starlette_response()`
- `encode_sse_stream()`
- `SeedFactory`, `ResumeFactory`, `EventMapper`, `MapperContext`

For AG-UI details, see [AG-UI Adapter](agui.md).
