# Streaming

Real-time event streaming, LLM token deltas, and custom telemetry from handlers. Use streaming when you need live UI updates instead of waiting for the full run to complete.

All `invoke`/`stream` methods have async counterparts: `ainvoke()`, `astream_events()`, `aresume()`, `astream_resume()`.

```python
# Async stream with real-time LLM token deltas and passthrough custom frames
from langgraph_events import emit_custom, emit_state_snapshot
from langgraph_events.stream import (
    CustomEventFrame,
    LLMStreamEnd,
    LLMToken,
    StateSnapshotFrame,
)


@on(QueryReceived)
def step(event: QueryReceived) -> ReplyProduced:
    emit_state_snapshot({"messages": [], "step": "draft"})
    emit_custom("tool.progress", {"pct": 50})
    return ReplyProduced(...)


async for item in graph.astream_events(
    QueryReceived(...),
    include_llm_tokens=True,
    include_custom_events=True,
):
    if isinstance(item, LLMToken):
        print(item.content, end="")
    elif isinstance(item, LLMStreamEnd):
        print("\n[done]", item.message_id)
    elif isinstance(item, StateSnapshotFrame):
        print("snapshot:", item.data)
    elif isinstance(item, CustomEventFrame):
        print("custom:", item.name, item.data)
    else:
        print(item)
```

## Stream Options

| Flag | Enables | Frame types yielded |
|------|---------|-------------------|
| `include_reducers=True` | Reducer snapshots alongside events | `StreamFrame(event, reducers, changed_reducers)` |
| `include_llm_tokens=True` | Real-time LLM token deltas | `LLMToken(run_id, content)`, `LLMStreamEnd(run_id, message_id)` |
| `include_custom_events=True` | Custom event passthrough | `CustomEventFrame(name, data)`, `StateSnapshotFrame(data)` |

## Emission Helpers

Emit telemetry from inside handlers without importing LangGraph callback APIs directly:

| Function | Use case |
|----------|----------|
| `emit_custom(name, data)` | Arbitrary stream-only telemetry (sync) |
| `aemit_custom(name, data)` | Async variant |
| `emit_state_snapshot(data)` | Typed state snapshot for UI (sync) |
| `aemit_state_snapshot(data)` | Async variant |

These surface in `astream_events(..., include_custom_events=True)` as `CustomEventFrame` or `StateSnapshotFrame`.

## Reducer Deltas

When `include_reducers=True`, each `StreamFrame` includes `changed_reducers: frozenset[str] | None` — the set of reducer names that the event actually updated. Consumers can use this to skip re-emitting unchanged state. `None` means delta metadata is not available.

For streaming to AG-UI frontends, see the [AG-UI Adapter](agui.md).
