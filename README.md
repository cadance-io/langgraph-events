# langgraph-events

Opinionated event-driven abstraction for LangGraph. **State IS events.**

> [!CAUTION]
> **Experimental (v0.1.1)** — This is an early-stage personal project, not a supported product. The API will change without notice or migration path. Do not depend on this for anything you can't easily rewrite. Not published to PyPI.

## What is this?

LangGraph gives you full control over agent topology, but wiring `StateGraph` nodes and conditional edges by hand is tedious. **langgraph-events** replaces that boilerplate with a reactive, event-driven model: define domain events as frozen dataclasses, subscribe handler functions with `@on(EventType)`, and let `EventGraph` derive the full graph topology automatically.

The core principle: **state IS events.** The entire state of a run is an append-only log of typed, immutable events. Handlers read events in; handlers emit events out. The framework does the rest.

## Installation

Not published to PyPI yet. Install directly from GitHub:

```bash
pip install git+https://github.com/cadance-io/langgraph-events.git
```

Requires Python 3.10+ and `langgraph >= 0.2.0` (installed automatically).

## Quick Start

```python
from langgraph_events import Event, EventGraph, on

# 1. Define events (auto-frozen dataclasses — no decorator needed)
class MessageReceived(Event):
    text: str

class MessageClassified(Event):
    label: str

class ReplyProduced(Event):
    text: str

# 2. Subscribe handlers with @on
@on(MessageReceived)
def classify(event: MessageReceived) -> MessageClassified:
    if "help" in event.text.lower():
        return MessageClassified(label="support")
    return MessageClassified(label="general")

@on(MessageClassified)
def respond(event: MessageClassified) -> ReplyProduced:
    if event.label == "support":
        return ReplyProduced(text="Routing you to support...")
    return ReplyProduced(text="Thanks for your message!")

# 3. Build the graph and run
graph = EventGraph([classify, respond])
log = graph.invoke(MessageReceived(text="I need help with my order"))

print(log.latest(ReplyProduced))
# ReplyProduced(text='Routing you to support...')
```

## How It Works

`EventGraph` compiles your handlers into a LangGraph `StateGraph` with a hub-and-spoke reactive loop:

```
seed event
    │
    v
[seed] ──> [dispatch] ──> handler_a ──┐
                ^          handler_b ──┤
                │                      │
             [router] <────────────────┘
                │
                v
             [dispatch] ──> handler_c ──┐
                ^                       │
                │                       │
             [router] <─────────────────┘
                │
                v
             [dispatch] ──> END (no pending events)
```

1. A **seed event** enters the graph.
2. The **router** collects new events, then **dispatch** matches each to subscribed handlers via `isinstance`. Matched handlers run and emit new events.
3. The loop repeats until no handler matches or a `Halted` event appears.

## Key Concepts

### Events

Events are frozen dataclasses that extend `Event`. Immutability guarantees a safe append-only log. Subclasses are automatically made into frozen dataclasses — no decorator needed:

```python
class OrderPlaced(Event):
    order_id: str
    total: float
```

Events support **inheritance**. A handler subscribed to a parent type fires for all subtypes (`isinstance` matching). The built-in `Auditable` marker class is a common example — subscribe once with `@on(Auditable)` and every marked event is captured automatically:

```python
from langgraph_events import Auditable, on

class OrderPlaced(Auditable):
    order_id: str

class OrderShipped(Auditable):
    order_id: str

@on(Auditable)
def audit(event: Auditable) -> None:
    # Fires for OrderPlaced, OrderShipped, and any Auditable subtype
    print(event.trail())
```

### `@on(*EventTypes)`

Decorate a function with `@on(EventType)` to subscribe it. Handlers receive the matching event and optionally an `EventLog`. They return a single `Event`, `None` (side-effect only), or `Scatter`.

```python
@on(UserMessage)
def greet(event: UserMessage) -> Greeting:
    return Greeting(text=f"Hello!")
```

Handlers may also request `config: RunnableConfig` or `store: BaseStore` by parameter name.

**Multi-subscription** — a single handler fires on multiple event types:

```python
@on(UserMessage, ToolResult)
def call_llm(event: Event, log: EventLog) -> AssistantMessage:
    history = log.filter(Event)
    ...
```

### `EventGraph`

The main entry point. Pass a list of handler functions and `EventGraph` derives the topology.

```python
graph = EventGraph(
    [classify, respond, audit],
    max_rounds=50,           # default: 100; prevents infinite loops
    reducers=[my_reducer],   # optional — see Reducer section
)

# Synchronous
log = graph.invoke(SeedEvent(...))

# Multiple seed events
log = graph.invoke([
    SystemPromptSet.from_str("You are helpful"),
    UserMessageReceived(message=HumanMessage(content="Hi")),
])

# Asynchronous
log = await graph.ainvoke(SeedEvent(...))

# Stream events as they're produced
for event in graph.stream_events(SeedEvent(...)):
    print(event)

# Stream with reducer snapshots
for frame in graph.stream_events(SeedEvent(...), include_reducers=True):
    print(frame.event, frame.reducers["messages"])

# --- LangGraph escape hatch ---
# Access the underlying CompiledStateGraph for advanced patterns:
# subgraph composition, custom streaming modes, or direct state access.
compiled = graph.compiled
for chunk in compiled.stream({"events": [SeedEvent(...)]}, stream_mode="updates"):
    print(chunk)
```

`max_rounds` (default: 100) prevents infinite loops — the library auto-sets LangGraph's `recursion_limit` so this is the only knob you need. Override via `invoke(seed, recursion_limit=N)` if needed. All methods have async counterparts: `ainvoke()`, `astream_events()`, `aresume()`.

#### Visualizing the Event Flow

`graph.mermaid()` returns a Mermaid flowchart showing how events correlate through handlers. Events are nodes, handler names are edge labels, and side-effect handlers (returning `None`) are listed in a footer comment.

```python
# Visualize the event correlation graph
print(graph.mermaid())
```

```mermaid
graph LR
    classDef entry fill:none,stroke:none,color:none
    _e0_[ ]:::entry ==> MessageReceived
    MessageReceived -->|classify| MessageClassified
    MessageClassified -->|respond| ReplyProduced
```

### `EventLog`

Immutable, ordered container returned by `invoke`/`ainvoke`. Handlers can also receive it as a second parameter.

```python
@on(DraftProduced)
def evaluate(event: DraftProduced, log: EventLog) -> CritiqueReceived | FinalDraftProduced:
    request = log.latest(WriteRequested)        # most recent event of this type
    all_drafts = log.filter(DraftProduced)      # all events matching this type
    if log.has(CritiqueReceived):               # boolean check
        ...
```

| Method               | Returns             | Description                                    |
|----------------------|---------------------|------------------------------------------------|
| `log.filter(T)`      | `list[T]`           | All events of type T                           |
| `log.latest(T)`      | `T \| None`         | Most recent event of type T                    |
| `log.first(T)`       | `T \| None`         | Earliest event of type T                       |
| `log.has(T)`         | `bool`              | Whether any event of type T exists             |
| `log.count(T)`       | `int`               | Number of events matching type T               |
| `log.select(T)`      | `EventLog`          | Filtered log (chainable)                       |
| `log.after(T)`       | `EventLog`          | Events after first occurrence of T             |
| `log.before(T)`      | `EventLog`          | Events before first occurrence of T            |
| `len(log)`           | `int`               | Total events                                   |
| `log[i]`             | `Event`             | Index access                                   |

### `Halted`

Return a `Halted` event from any handler to immediately stop the graph. No further handlers are dispatched.

```python
@on(Classified)
def guard(event: Classified) -> Reply | Halted:
    if event.label == "blocked":
        return Halted(reason="Content policy violation")
    return Reply(text="OK")
```

### `Scatter`

Return `Scatter([event1, event2, ...])` to fan-out into multiple events. Each becomes a separate pending event, dispatched in the next round. Use `Scatter[WorkItem]` to annotate the produced type — this renders as a dashed edge in `mermaid()` diagrams.

```python
@on(Batch)
def split(event: Batch) -> Scatter[WorkItem]:
    return Scatter([WorkItem(item=i) for i in event.items])

@on(WorkItem)
def process(event: WorkItem) -> WorkDone:
    return WorkDone(result=f"done:{event.item}")

@on(WorkDone)
def gather(event: WorkDone, log: EventLog) -> BatchResult | None:
    all_done = log.filter(WorkDone)
    batch = log.latest(Batch)
    if len(all_done) >= len(batch.items):
        return BatchResult(results=tuple(e.result for e in all_done))
    return None  # not all items done yet
```

### `Auditable`

Marker base class for events that should be auto-logged. Subclass it and subscribe a single `@on(Auditable)` handler to capture every marked event automatically. The built-in `trail()` method returns a compact summary of the event's fields.

```python
class TaskStarted(Auditable):
    name: str

@on(Auditable)
def log_event(event: Auditable) -> None:
    print(event.trail())
    # "[TaskStarted] name='deploy'"
```

### `MessageEvent`

Base class for events that wrap LangChain `BaseMessage` objects. Declare a `message` field (single message) or `messages` field (tuple of messages), and `as_messages()` auto-converts them. Pairs with `message_reducer()` for automatic message history accumulation.

```python
from langchain_core.messages import HumanMessage, AIMessage

class UserMessageReceived(MessageEvent, Auditable):
    message: HumanMessage

class LLMResponded(MessageEvent, Auditable):
    message: AIMessage
```

### `SystemPromptSet`

Built-in `MessageEvent` that wraps a `SystemMessage`. Makes the system prompt a first-class citizen in the event log — visible, queryable, and auditable.

```python
from langgraph_events import SystemPromptSet, message_reducer, EventGraph
from langchain_core.messages import SystemMessage

messages = message_reducer()
graph = EventGraph([call_llm, execute_tools], reducers=[messages])

# Convenience factory
log = graph.invoke([
    SystemPromptSet.from_str("You are a helpful assistant with tools."),
    UserMessageReceived(message=HumanMessage(content="What's the weather?")),
])

# Or construct explicitly
seed = SystemPromptSet(message=SystemMessage(content="You are helpful"))
```

### `Reducer` / `message_reducer`

**When to use reducers:** Pure event-driven handlers (`log.filter()`, `log.latest()`) are the default and work for most patterns. Add a `Reducer` when you need incremental accumulation that would be expensive to recompute from the full log each round — the canonical case is `message_reducer()` for LLM conversation history. Add a `ScalarReducer` for last-write-wins configuration values injected directly into handlers. If you find yourself calling `log.filter(X)` and transforming the result the same way in multiple handlers, that's a signal a reducer would help.

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

# --- message_reducer: built-in for LangChain message accumulation ---
messages = message_reducer()
graph = EventGraph([call_llm, handle_tools], reducers=[messages])
log = graph.invoke([
    SystemPromptSet.from_str("You are a helpful assistant."),
    UserMessageReceived(message=HumanMessage(content="Hi")),
])

# Alternative: explicit default list
messages = message_reducer([SystemMessage(content="You are a helpful assistant.")])

# --- ScalarReducer: last-write-wins, injected as a bare value ---
temperature = ScalarReducer("temperature", event_type=TempSet, fn=lambda e: e.value, default=0.7)
```

The parameter name `messages` matches the reducer name, so the framework injects the accumulated message list automatically:

```python
@on(UserMessageReceived, ToolsExecuted)
async def call_llm(event: Event, messages: list[BaseMessage]) -> LLMResponded:
    response = await llm.ainvoke(messages)
    ...
```

### `Interrupted` / `Resumed`

Return an `Interrupted` event to pause the graph and wait for human input. When the graph is resumed (via `graph.resume()`), the framework creates a `Resumed` event containing the human's response.

Requires a **checkpointer** (e.g., `MemorySaver`).

```python
from langgraph.checkpoint.memory import MemorySaver

@on(OrderPlaced)
def confirm(event: OrderPlaced) -> Interrupted:
    return Interrupted(
        prompt=f"Approve order {event.order_id} for ${event.total}?",
        payload={"order_id": event.order_id},
    )

@on(Resumed)
def finalize(event: Resumed) -> OrderConfirmed | OrderCancelled:
    if event.value == "yes":
        # interrupted is always set by the framework when resuming
        return OrderConfirmed(order_id=event.interrupted.payload["order_id"])
    return OrderCancelled(reason="User declined")

graph = EventGraph([confirm, finalize], checkpointer=MemorySaver())
config = {"configurable": {"thread_id": "order-1"}}

# First call — pauses at the interrupt
graph.invoke(OrderPlaced(order_id="A1", total=99.99), config=config)

# Check state and resume with human input
state = graph.get_state(config)
if state.is_interrupted:
    print(state.interrupted.prompt)
log = graph.resume("yes", config=config)
```

**Typed Event resume** — if you pass an `Event` to `resume()`, it is auto-dispatched alongside `Resumed`. This lets handlers subscribed to the event type fire without manual state injection:

```python
class Approval(Event):
    approved: bool

@on(Approval)
def handle_approval(event: Approval) -> OrderConfirmed | OrderCancelled:
    if event.approved:
        return OrderConfirmed(order_id=event.interrupted.payload["order_id"])
    return OrderCancelled(reason="User declined")

# Resume with a typed event — Approval handler fires automatically
log = graph.resume(Approval(approved=True), config=config)
```

## Patterns & Examples

The patterns below show how these building blocks compose into complete architectures. Each links to a runnable example in `examples/`.

### Reflection Loop (Generate / Critique / Revise)

Multi-subscription creates an autonomous generate/critique/revise cycle; `EventLog.latest()` enforces a revision cap.

[`examples/reflection_loop.py`](examples/reflection_loop.py) · [event flow](examples/reflection_loop.graph.md) — Multi-subscription `@on`, `EventLog.latest()`, revision cap

### ReAct Agent with Message Reducer

Multi-subscription creates the ReAct loop implicitly; `message_reducer` maintains conversation history incrementally as a handler parameter.

[`examples/react_agent.py`](examples/react_agent.py) · [event flow](examples/react_agent.graph.md) — `MessageEvent`, `message_reducer`, `Auditable`

### Multi-Agent Supervisor

A supervisor handler fires on task and specialist completions, using tool-calling for structured routing; a custom `Reducer` projects events into a context channel.

[`examples/supervisor.py`](examples/supervisor.py) · [event flow](examples/supervisor.graph.md) — Custom `Reducer`, tool-calling routing, `Auditable`

### Fan-Out / Fan-In (Map-Reduce)

`Scatter[WorkItem]` fans a batch into individual items; a gathering handler uses `EventLog.filter()` to detect completion.

[`examples/map_reduce.py`](examples/map_reduce.py) · [event flow](examples/map_reduce.graph.md) — `Scatter`, `EventLog.filter()`, gather pattern

### Human-in-the-Loop Approval

`Interrupted` pauses for human input; `Resumed` carries the response back, creating an approval-with-feedback cycle.

[`examples/human_in_the_loop.py`](examples/human_in_the_loop.py) · [event flow](examples/human_in_the_loop.graph.md) — `Interrupted`/`Resumed`, checkpointer, revision cycle

### Content Pipeline (Halted + Streaming)

`Halted` terminates immediately for unsafe content; `stream_events()` yields events live with optional `StreamFrame` reducer snapshots.

[`examples/content_pipeline.py`](examples/content_pipeline.py) · [event flow](examples/content_pipeline.graph.md) — `Halted`, `Reducer`, `stream_events`, `StreamFrame`

## API Reference

| Export            | Type       | Description                                     |
|-------------------|------------|-------------------------------------------------|
| `Auditable`       | Base class | Marker class for auto-logged events             |
| `Event`           | Base class | Subclass to define events (auto-frozen)         |
| `EventGraph`      | Class      | Build and run the event-driven graph            |
| `EventGraph.invoke()` | Method | Run graph synchronously, returns `EventLog` |
| `EventGraph.ainvoke()` | Method | Async version of `invoke()` |
| `EventGraph.stream_events()` | Method | Yield events as produced; optional reducer snapshots via `StreamFrame` |
| `EventGraph.astream_events()` | Method | Async version of `stream_events()` |
| `EventGraph.resume()` | Method | Resume interrupted graph with human input (requires checkpointer) |
| `EventGraph.aresume()` | Method | Async version of `resume()` |
| `EventGraph.get_state()` | Method | Get `GraphState` for a checkpointed thread |
| `EventGraph.compiled` | Property | Access underlying `CompiledStateGraph` for advanced LangGraph patterns |
| `EventGraph.reducer_names` | Property | `frozenset` of registered reducer names |
| `EventGraph.mermaid()` | Method | Return a Mermaid flowchart of event correlations |
| `EventLog`        | Class      | Immutable query container over events           |
| `GraphState`      | NamedTuple | `(events, is_interrupted, interrupted)` from `get_state()` |
| `Halted`          | Event      | Signal immediate graph termination              |
| `Interrupted`     | Event      | Pause graph for human input                     |
| `MessageEvent`    | Base class | Mixin for events wrapping LangChain messages    |
| `message_reducer` | Function   | Built-in reducer for `MessageEvent` projection  |
| `on`              | Decorator  | Subscribe a handler to one or more event types  |
| `Reducer`         | Class      | Map events to a named LangGraph state channel   |
| `ScalarReducer`   | Class      | Last-write-wins reducer for single values (None is a valid value) |
| `Resumed`         | Event      | Created on resume; if value is an `Event`, it's auto-dispatched |
| `Scatter`         | Class      | Fan-out into multiple events; generic `Scatter[T]` annotates the produced type |
| `StreamFrame`     | NamedTuple | `(event, reducers)` yielded by `stream_events()` with `include_reducers` |
| `SystemPromptSet` | Event      | Built-in `MessageEvent` for system prompts      |

## Checkpointer & Graph Evolution

> *This documents current behavior. Details may change between versions — there are no stability guarantees yet.*

When using a checkpointer (`MemorySaver`, `SqliteSaver`, etc.), existing threads retain their checkpoint state across invocations. If you modify the graph between invocations on the same thread, LangGraph handles mismatches through **graceful degradation** — no crashes, but some changes have silent side effects.

### What's safe

| Change | Behavior |
|---|---|
| **Add a handler** | Safe. `dispatch()` is rebuilt from current handlers, so the new handler participates immediately. |
| **Add an event type** | Safe. New events can be emitted and matched normally. |
| **Remove an event type** | Safe. Existing events stay in the log but no handler will match them — they're inert. |

### What to watch out for

| Change | Risk |
|---|---|
| **Remove a handler** (normal checkpoint) | Events that *only* the removed handler subscribed to become undeliverable. The graph halts early — no crash, but incomplete execution. |
| **Remove a handler** (interrupted checkpoint) | If the graph was paused inside the removed handler via `Interrupted`, `graph.resume(value)` silently does nothing. The pending Send to the missing node is dropped. The human-in-the-loop flow breaks without error. |
| **Rename a handler** | Same as remove + add. If an `Interrupted` checkpoint targeted the old name, the resume is lost. |
| **Add a reducer** | The new reducer starts cold — it **misses its default values and all historical event projections**. Only events added after the resume point contribute. |
| **Remove a reducer** | The reducer's channel data is silently dropped from the checkpoint. |

### Best practices

1. **Don't rename handlers with active interrupted threads.** If a thread is paused at an `Interrupted` checkpoint, the handler's function name is baked into the checkpoint. Renaming it silently breaks resume.
2. **Treat reducer addition as a fresh start.** A newly added reducer on an existing thread won't replay historical events. If you need the full history, start a new thread.
3. **Prefer additive changes.** Adding handlers and event types is always safe. Removing them is safe only if no in-flight threads depend on them.
4. **Use separate `thread_id`s after structural changes.** The simplest way to avoid all edge cases is to use a new thread for a modified graph.

## Status

This is a solo experiment, not a team-backed product. Expect:

- No changelog or migration guides between versions
- API surface may shrink or change significantly
- Bug reports welcome, but no SLA on fixes

## Development

```bash
git clone https://github.com/cadance-io/langgraph-events.git
cd langgraph-events
uv sync --group dev
uv run pytest
```

## License

MIT — see [LICENSE](LICENSE).
