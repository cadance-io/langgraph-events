# Streaming

Streaming yields events in real-time for live UI updates. All methods have async variants: `ainvoke()`, `astream_events()`, `aresume()`, `astream_resume()`.

```python
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

| Flag | Enables | Frame types |
|---|---|---|
| `include_reducers=True` | Reducer snapshots alongside events | `StreamFrame(event, reducers, changed_reducers)` |
| `include_llm_tokens=True` | Real-time LLM token deltas | `LLMToken(run_id, content)`, `LLMStreamEnd(run_id, message_id)` |
| `include_custom_events=True` | Custom event passthrough | `CustomEventFrame(name, data)`, `StateSnapshotFrame(data)` |

## Emission Helpers

Emit telemetry from handlers without importing LangGraph callback APIs.

| Function | Use case |
|---|---|
| `emit_custom(name, data)` / `aemit_custom(...)` | Arbitrary stream-only telemetry |
| `emit_state_snapshot(data)` / `aemit_state_snapshot(...)` | Typed state snapshot for UI |

Surface in `astream_events(..., include_custom_events=True)` as `CustomEventFrame` / `StateSnapshotFrame`.

## Reducer Deltas

With `include_reducers=True`, `StreamFrame.changed_reducers: frozenset[str] | None` carries the names of reducers this event updated. Use to skip re-emitting unchanged state. `None` means delta metadata is unavailable.

See [AG-UI Adapter](agui.md) for streaming to AG-UI frontends.
