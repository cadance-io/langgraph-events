# Reducers

Reducers cache expensive projections or last-write-wins values. Use `EventLog.filter()` for one-shot queries.

- **Reach for a reducer** when the projection runs in many handlers, when you want a value injected by parameter name, or when recomputing from the full log each round would be expensive (e.g. LangChain message history).
- **Reach for `EventLog.filter(T)`** for one-shot lookups inside a single handler.

## On a Namespace { #on-a-namespace }

Declare reducers as `Namespace` class attributes. Channel name auto-fills from the attribute name; namespace scope auto-fills to the enclosing class; `EventGraph` auto-registers on first subscriber.

```python
from langgraph_events import Namespace, Command, DomainEvent, Event, ScalarReducer


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

        def handle(self, current_status: str | None) -> Placed:
            # `current_status` is injected by parameter name.
            return Order.Place.Placed(order_id=f"o-{self.customer_id}")

    class Shipped(DomainEvent):
        tracking: str


graph = EventGraph([Order.Place])   # reducer auto-discovered from Order
```

- Only sees events whose `__namespace__` matches.
- Child namespaces inherit parent reducers (dedup by name).
- Cross-namespace name collisions raise `TypeError` at graph construction.
- Explicit `reducers=[...]` wins on conflict with an auto-discovered reducer.

## Graph-wide reducers

For reducers spanning namespaces or not namespace-scoped, pass via `reducers=[...]` (the form used by `message_reducer()`):

```python
messages = message_reducer()
graph = EventGraph([call_llm, handle_tools], reducers=[messages])

log = graph.invoke([
    SystemPromptSet.from_str("You are helpful."),
    UserMessageReceived(message=HumanMessage(content="Hi")),
])
```

## `Reducer`

List accumulator. Default merge `operator.add`; any LangGraph-compatible reducer function works (e.g. `add_messages` for smart dedup).

```python
history = Reducer(name="history", event_type=UserMsg, fn=lambda e: [e.text], default=[])
```

## `ScalarReducer`

Last-write-wins scalar. `None` is valid; return `SKIP` from `fn` to leave the current value unchanged (distinguishes "set to `None`" from "don't update").

```python
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

The handler parameter annotation declares whether the value may be `None`. A non-`None` annotation opts the parameter into a runtime assertion: if the channel value is `None` at injection time, `ReducerNotSetError` raises **before** the handler body runs.

```python
from langgraph_events import ReducerNotSetError  # catch it if you want


@on(TaskReceived)
def strict(event: TaskReceived, strategy: str) -> Completed:
    # `strategy` guaranteed non-None — otherwise ReducerNotSetError at injection.
    ...

@on(TaskReceived)
def permissive(event: TaskReceived, strategy: str | None) -> Completed:
    if strategy is None: ...
```

!!! note "`ReducerNotSetError` contract"
    - `ValueError` subclass; exported from `langgraph_events`.
    - Raised at injection time, **outside** the `raises=` catch boundary — a broad `raises=ValueError` cannot silently swallow it.
    - Opt out by widening the annotation: `str | None`, `Optional[str]`, `Any`, `object`, or no annotation.
    - Forward-ref failures: framework emits `UserWarning` and falls back to permissive mode. Annotate against importable types so the assertion sticks.

## `FoldReducer`

Accumulator whose next value depends on the **prior** state — a left-fold, where `Reducer` appends and `ScalarReducer` takes the last write. Use it for counters, merging dicts, or re-derived cursors. Each event owns its transition via `fold(self, state)` (the same polymorphic style as `MessageEvent.as_messages()`); supply only `name`, `event_type`, and a `default_factory`.

```python
from langgraph_events import FoldReducer, RESET


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

`fold` returns: the new state (any value, including `None`); `RESET` to clear the channel back to `default_factory()`; or `SKIP` to leave it unchanged. Pass an explicit `fold=lambda state, event: ...` for events that don't carry a `fold` method. The fold state may be any type — including a `list`. Like `ScalarReducer`, a non-`None` handler annotation opts into the [`ReducerNotSetError`](#scalar-reducer-required) assertion.

### Typing { #fold-typing }

`FoldReducer` is generic over its state type `S`, inferred from `default_factory`, so `reducer.empty` and `reducer.seed(...)` are typed (not `Any`):

```python
counter = FoldReducer(default_factory=lambda: {"n": 0}, event_type=(Incremented, Reset), name="counter")
reveal_type(counter.empty)   # dict[str, int]
```

The event arg is typed against `Foldable` — a `@runtime_checkable` Protocol satisfied **structurally** (your event just needs a `fold` method; do **not** inherit `Foldable` — it would clash with the event metaclass). An explicit `fold=` function should annotate its event param as `Foldable` (or leave it inferred):

```python
def step(state: dict[str, int], event: Foldable) -> dict[str, int]:
    return event.fold(state)
```

To have mypy flag a `default_factory`/`fold` state-shape disagreement, pin the state type: `FoldReducer[dict[str, int]](...)`. (Without a pin, mypy widens `S` to the join of the two and won't catch the mismatch.) The generic does not propagate to handler params — those are injected by name at runtime.

For genuinely bespoke channels, subclass the public `BaseReducer` directly.

## `message_reducer`

Built-in reducer for LangChain messages — projects `MessageEvent.as_messages()` into the `messages` channel using `add_messages` for id-based dedup.

```python
messages = message_reducer()
# or with a default prompt:
messages = message_reducer([SystemMessage(content="You are helpful.")])

@on(UserMessageReceived, ToolsExecuted)
async def call_llm(event: Event, messages: list[BaseMessage]) -> LLMResponded: ...
```

See [Conversation Agent](patterns.md#conversation-agui) and [Supervisor](patterns.md#supervisor).

## Pre-seeding

Seed state via events — the log stays the single source of truth. For external state injection outside the event path (migration, test fixture), use `pre_seed`:

```python
graph.pre_seed(config, {"my_reducer": value})  # or: await graph.apre_seed(...)
graph.invoke(SeedEvent(), config=config)
```

Pre-seeded values bypass the event log — `log.filter()` won't reflect them.

## Recovering from projection changes { #replay_reducer }

When a reducer's `fn` or output shape changes, the cached channel value in existing checkpoints is stale. Replay the (already-migrated) event log to rebuild it:

```python
from langgraph_events.serde import replay_reducer

event_log = checkpointer.get_tuple(config).checkpoint["channel_values"]["event_log"]
rebuilt = replay_reducer(my_reducer, event_log)
# Write `rebuilt` back through the checkpointer's put API.
```

The library doesn't iterate the checkpointer for you — wire the loop in your own startup or migration script. See [Event migrations](event-migrations.md) for the reducer-state matrix.

## Introspection

`graph.reflect(log).state()` projects every registered reducer over a run's
event log from scratch — the runtime answer to "what's the derived state
now", for code and for agents (via the `state` op of the `query_log` tool).
See [Reflection](reflection.md).
