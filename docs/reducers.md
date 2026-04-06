# Reducers

Incremental state accumulation for values that are expensive to recompute from the event log each round.

**When to reach for a reducer:** Pure event-driven handlers (`log.filter()`, `log.latest()`) are the default and work for most patterns. Add a `Reducer` when you need incremental accumulation that would be expensive to recompute from the full log each round — the canonical case is `message_reducer()` for LLM conversation history. Add a `ScalarReducer` for last-write-wins configuration values injected directly into handlers. If you find yourself calling `log.filter(X)` and transforming the result the same way in multiple handlers, that's a signal a reducer would help.

## `Reducer`

A `Reducer` maps events to contributions for a named LangGraph state channel. The framework maintains the channel incrementally — handlers receive the accumulated value by declaring a parameter whose name matches the reducer.

```python
from langgraph_events import Reducer, ScalarReducer, message_reducer, EventGraph, on

# --- Reducer: accumulates contributions from matching events ---
history = Reducer("history", event_type=UserMsg, fn=lambda e: [e.text], default=[])


@on(UserMsg)
def respond(event: UserMsg, history: list) -> Reply:
    # history contains all projected values so far
    ...


graph = EventGraph([respond], reducers=[history])
```

## `message_reducer`

Built-in reducer for LangChain message accumulation. Projects `MessageEvent.as_messages()` into a `messages` state channel with smart deduplication via LangGraph's `add_messages`.

```python
messages = message_reducer()
graph = EventGraph([call_llm, handle_tools], reducers=[messages])
log = graph.invoke([
    SystemPromptSet.from_str("You are a helpful assistant."),
    UserMessageReceived(message=HumanMessage(content="Hi")),
])

# Alternative: explicit default list
messages = message_reducer([SystemMessage(content="You are a helpful assistant.")])
```

The parameter name `messages` matches the reducer name, so the framework injects the accumulated message list automatically:

```python
@on(UserMessageReceived, ToolsExecuted)
async def call_llm(event: Event, messages: list[BaseMessage]) -> LLMResponded:
    response = await llm.ainvoke(messages)
    ...
```

See the [ReAct Agent pattern](patterns.md#react-agent-with-message-reducer) for `message_reducer()` in action, and the [Supervisor pattern](patterns.md#multi-agent-supervisor) for a custom `Reducer`.

## `ScalarReducer`

Last-write-wins reducer for single values. Unlike `Reducer` (which accumulates lists), `ScalarReducer` injects a bare value — the most recent non-`SKIP` contribution wins.

```python
temperature = ScalarReducer(
    "temperature", event_type=TempSet, fn=lambda e: e.value, default=0.7
)
```

### `SKIP`

When a `ScalarReducer` function returns `SKIP`, the reducer value is left unchanged. This distinguishes "set to `None`" from "don't update."

```python
from langgraph_events import SKIP, ScalarReducer

temperature = ScalarReducer(
    "temperature", event_type=ConfigUpdated, fn=lambda e: e.temp if e.temp is not None else SKIP, default=0.7
)
```
