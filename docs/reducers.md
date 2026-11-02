# Reducers

- **Use a reducer** in one of three cases. The projection runs in many handlers. You want a value injected by parameter name. A recomputation from the full log each round would be expensive, for example a LangChain message history.
- **Use `EventLog.filter(T)`** for one-shot lookups inside a single handler.

## On a Namespace { #on-a-namespace }

Declare reducers as `Namespace` class attributes. The channel name auto-fills from the attribute name. The namespace scope auto-fills to the enclosing class. `EventGraph` auto-registers on first subscriber.

```python
from __future__ import annotations

from langgraph_events import Command, DomainEvent, Event, EventGraph, Namespace
from langgraph_events import ScalarReducer


class Order(Namespace):
    current_status = ScalarReducer(
        event_type=Event,
        fn=lambda e: (
            "shipped" if isinstance(e, Order.Shipped)
            else "placed" if isinstance(e, Order.Place.Placed)
            else None
        ),
    )

    class Place(Command):
        customer_id: str

        class Placed(DomainEvent):
            order_id: str

        def handle(self, current_status: str | None) -> Order.Place.Placed:
            # `current_status` is injected by parameter name.
            return Order.Place.Placed(order_id=f"o-{self.customer_id}")

    class Shipped(DomainEvent):
        tracking: str


graph = EventGraph([Order.Place])   # reducer auto-discovered from Order
```

- Only sees events whose `__namespace__` matches.
- Namespaces do not inherit: `class Child(Base)` where `Base` is a `Namespace` raises `TypeError`. To share a reducer, declare it free-standing (no owning namespace) and pass it via `reducers=[...]`.
- Cross-namespace name collisions raise `TypeError` at graph construction.
- Explicit `reducers=[...]` wins on conflict with an auto-discovered reducer.

## Graph-wide reducers

For a reducer that spans namespaces or is not namespace-scoped, pass it via `reducers=[...]`. `message_reducer()` uses this form:

```python
...
messages = message_reducer()
graph = EventGraph([call_llm, handle_tools], reducers=[messages])

log = graph.invoke([
    SystemPromptSet.from_str("You are helpful."),
    UserMessageReceived(message=HumanMessage(content="Hi")),
])
```

## `Reducer`

List accumulator. The default merge is `operator.add`. Any LangGraph-compatible reducer function works, for example `add_messages` for id-based dedup.

```python
...
history = Reducer(name="history", event_type=UserMsg, fn=lambda e: [e.text], default=[])
```

## `ScalarReducer`

Last-write-wins scalar. `None` is valid. Return `SKIP` from `fn` to leave the current value unchanged. This distinguishes "set to `None`" from "do not update".

```python
...
temperature = ScalarReducer(name="temperature", event_type=TempSet, fn=lambda e: e.value, default=0.7)

# Conditional updates with SKIP:
temperature = ScalarReducer(
    name="temperature",
    event_type=ConfigUpdated,
    fn=lambda e: e.temp if e.temp is not None else SKIP,
    default=0.7,
)
```

### Required values { #scalar-reducer-required }

The handler parameter annotation declares whether the value can be `None`. A non-`None` annotation opts the parameter into a runtime assertion. `str | None`, `Optional[str]`, `Any`, `object` and a missing annotation opt out. When the channel value is `None` at injection time, `ReducerNotSetError` raises **before** the handler body runs.

```python
...
from langgraph_events import ReducerNotSetError  # catch it if you want


@on(TaskReceived)
def strict(event: TaskReceived, strategy: str) -> Completed:
    # `strategy` is guaranteed non-None, else ReducerNotSetError at injection.
    ...

@on(TaskReceived)
def permissive(event: TaskReceived, strategy: str | None) -> Completed:
    if strategy is None: ...
```

`ReducerNotSetError` is raised outside the `raises=` boundary, so a broad `raises=ValueError` cannot swallow it. See [Errors](api.md#errors). When a parameter annotation does not resolve, the library emits a `UserWarning` for that parameter and treats it as permissive. Annotate against an importable type.

## `FoldReducer`

Accumulator whose next value depends on the **prior** state, a left-fold. `Reducer` appends and `ScalarReducer` takes the last write. Use `FoldReducer` for a counter, a merged dict or a re-derived cursor. Each event owns its transition via `fold(self, state)`, the same polymorphic style as `MessageEvent.as_messages()`. Supply only `name`, `event_type` and a `default_factory`.

```python
from langgraph_events import RESET, FoldReducer, IntegrationEvent


class Incremented(IntegrationEvent):
    by: int = 1
    def fold(self, state): return {"n": state["n"] + self.by}

class Reset(IntegrationEvent):
    def fold(self, state): return RESET   # clears the channel


counter = FoldReducer(
    name="counter",
    event_type=(Incremented, Reset),
    default_factory=lambda: {"n": 0},
)
```

`fold` returns the new state (any value, including `None`), `RESET` to clear the channel back to `default_factory()`, or `SKIP` to leave it unchanged. Pass an explicit `fold=lambda state, event: ...` for an event that carries no `fold` method. The fold state can be any type, a `list` too. Like `ScalarReducer`, a non-`None` handler annotation opts into the [`ReducerNotSetError`](#scalar-reducer-required) assertion.

### Typing { #fold-typing }

`FoldReducer` is generic over its state type `S`, inferred from `default_factory`, so `reducer.empty` and `reducer.seed(...)` are typed (not `Any`):

```python
...
counter = FoldReducer(default_factory=lambda: {"n": 0}, event_type=(Incremented, Reset), name="counter")
reveal_type(counter.empty)   # dict[str, int]
```

The event argument is typed against `Foldable`, a `@runtime_checkable` Protocol satisfied **structurally**. Your event needs only a `fold` method. Do **not** inherit `Foldable`, because it clashes with the event metaclass. An explicit `fold=` function annotates its event parameter as `Foldable`, or leaves it inferred:

```python
...
def step(state: dict[str, int], event: Foldable) -> dict[str, int]:
    return event.fold(state)
```

To have mypy flag a `default_factory`/`fold` state-shape disagreement, pin the state type: `FoldReducer[dict[str, int]](...)`. Without a pin, mypy widens `S` to the join of the two and does not catch the mismatch. The generic does not propagate to handler params. Those are injected by name at runtime.

For a bespoke channel, subclass the public `BaseReducer` directly.

## `message_reducer`

Built-in reducer for LangChain messages. It projects `MessageEvent.as_messages()` into the `messages` channel, using `add_messages` for id-based dedup.

A message without an id gets a UUID id when the reducer projects it, in place on the message object, as `add_messages` does. The id is then saved with the step's write. So a step that pauses keeps one id for the message, and every `MessagesSnapshot` shows that id.

```python
...
messages = message_reducer()
# or with a default prompt:
messages = message_reducer([SystemMessage(content="You are helpful.")])

@on(UserMessageReceived, ToolsExecuted)
async def call_llm(event: Event, messages: list[BaseMessage]) -> LLMResponded: ...
```

See [Conversation Agent](patterns.md#conversation-agui) and [Supervisor](patterns.md#supervisor).

## Pre-seeding

Seed state via events, so the log stays the single source of truth. For external state injection outside the event path (a migration, a test fixture), use `pre_seed`:

```python
...
graph.pre_seed(config, {"my_reducer": value})  # or: await graph.apre_seed(...)
graph.invoke(SeedEvent(), config=config)
```

A pre-seeded value bypasses the event log. `log.filter()` does not reflect it.

## Recovering from projection changes { #replay_reducer }

When a reducer's `fn` or output shape changes, the cached channel value in existing checkpoints is stale. `replay_reducer(my_reducer, events)` rebuilds it from the event log that the checkpoint stores in its `events` channel. See [Recovering with `replay_reducer`](event-migrations.md#recovering-with-replay_reducer).

## Introspection

`graph.reflect(log).state()` projects every registered reducer over a run's
event log from scratch, the runtime answer to "what is the derived state
now", for code and for agents (via the `state` op of the `query_log` tool).
See [Reflection](reflection.md).
