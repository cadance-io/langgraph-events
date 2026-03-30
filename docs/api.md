# API Reference

| Export            | Type       | Description                                     |
|-------------------|------------|-------------------------------------------------|
| `Auditable`       | Base class | Marker class for auto-logged events             |
| `Event`           | Base class | Subclass to define events (auto-frozen)         |
| `EventGraph`      | Class      | Build and run the event-driven graph            |
| `EventGraph.invoke()` | Method | Run graph synchronously, returns `EventLog` |
| `EventGraph.ainvoke()` | Method | Async version of `invoke()` |
| `EventGraph.stream_events()` | Method | Yield events as produced; optional reducer snapshots via `StreamFrame` |
| `EventGraph.astream_events()` | Method | Async version of `stream_events()`; supports `include_llm_tokens=True` (`LLMToken`/`LLMStreamEnd`) and `include_custom_events=True` (`CustomEventFrame`/`StateSnapshotFrame`) |
| `EventGraph.stream_resume()` | Method | Yield events during resume; streaming equivalent of `resume()` |
| `EventGraph.astream_resume()` | Method | Async version of `stream_resume()`; supports `include_llm_tokens=True` (`LLMToken`/`LLMStreamEnd`) and `include_custom_events=True` (`CustomEventFrame`/`StateSnapshotFrame`) |
| `EventGraph.resume()` | Method | Resume interrupted graph with a domain event (requires checkpointer) |
| `EventGraph.aresume()` | Method | Async version of `resume()` |
| `EventGraph.get_state()` | Method | Get `GraphState` for a checkpointed thread |
| `EventGraph.compiled` | Property | Access underlying `CompiledStateGraph` for advanced LangGraph patterns |
| `EventGraph.reducer_names` | Property | `frozenset` of registered reducer names |
| `EventGraph.mermaid()` | Method | Return a Mermaid flowchart of event correlations |
| `emit_custom`    | Function   | Emit a LangGraph custom stream event from a handler |
| `aemit_custom`   | Function   | Async variant of `emit_custom` for async handlers |
| `emit_state_snapshot` | Function | Emit a typed state snapshot stream frame from a handler |
| `aemit_state_snapshot` | Function | Async variant of `emit_state_snapshot` for async handlers |
| `STATE_SNAPSHOT_EVENT_NAME` | Constant | Protocol name used for snapshot custom events (`"intermediate_state"`) |
| `EventLog`        | Class      | Immutable query container over events           |
| `GraphState`      | NamedTuple | `(events, is_interrupted, interrupted)` from `get_state()` |
| `Halted`          | Event      | Signal immediate graph termination              |
| `Interrupted`     | Base class | Bare marker — subclass with typed fields to pause graph |
| `MessageEvent`    | Base class | Mixin for events wrapping LangChain messages    |
| `message_reducer` | Function   | Built-in reducer for `MessageEvent` projection  |
| `on`              | Decorator  | Subscribe a handler to one or more event types  |
| `Reducer`         | Class      | Map events to a named LangGraph state channel   |
| `ScalarReducer`   | Class      | Last-write-wins reducer for single values (None is a valid value) |
| `SKIP`            | Sentinel   | Return from `ScalarReducer` fn to leave value unchanged |
| `Resumed`         | Event      | Created on resume with the dispatched event and `interrupted` backref |
| `Scatter`         | Class      | Fan-out into multiple events; generic `Scatter[T]` annotates the produced type |
| `StreamFrame`     | NamedTuple | `(event, reducers)` yielded by `stream_events()` with `include_reducers` |
| `LLMToken`        | NamedTuple | `(run_id, content)` token delta yielded by async streams with `include_llm_tokens=True` |
| `LLMStreamEnd`    | NamedTuple | `(run_id, message_id)` marker yielded when an LLM stream completes |
| `CustomEventFrame` | NamedTuple | `(name, data)` custom payload yielded from LangGraph `on_custom_event` with `include_custom_events=True` |
| `StateSnapshotFrame` | NamedTuple | `(data)` typed snapshot frame yielded when `on_custom_event` uses `STATE_SNAPSHOT_EVENT_NAME` |
| `SystemPromptSet` | Event      | Built-in `MessageEvent` for system prompts      |

## AG-UI Subpackage

Requires `[agui]` extra. See [AG-UI Adapter](agui.md) for details.

| Export            | Type       | Description                                     |
|-------------------|------------|-------------------------------------------------|
| `AGUIAdapter`     | Class      | Map EventGraph streams to AG-UI protocol events |
| `AGUIAdapter.stream()` | Method | Execute graph and emit AG-UI events for one request |
| `AGUIAdapter.connect()` | Method | Rehydrate current checkpoint state without graph execution |
| `AGUIAdapter.reconnect()` | Method | Alias of `connect()` for refresh/reconnect flows |
| `AGUICustomEvent` | Protocol   | Optional protocol for overriding fallback custom event names via `agui_event_name` |
| `AGUISerializable` | Protocol  | Optional protocol for defining AG-UI payload serialization via `agui_dict()` |
| `SeedFactory`     | Protocol   | Convert `RunAgentInput` to domain seed event(s)  |
| `ResumeFactory`   | Protocol   | Detect resume requests and produce domain events |
| `EventMapper`     | Protocol   | Map domain events to AG-UI events in the mapper chain |
| `MapperContext`   | Dataclass  | Shared state (run ID, thread ID, message counter) for one stream |
| `create_starlette_response` | Function | Wrap AG-UI event stream as a Starlette `StreamingResponse` |
| `encode_sse_stream` | Function | Encode AG-UI events as SSE strings (framework-agnostic) |
