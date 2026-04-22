# Reducers

Incremental state accumulation. `EventLog.filter()` covers most cases — reach for a reducer when recomputing from the full log every round would be expensive (e.g. LangChain message history) or when you want a last-write-wins value injected by name.

## On a Namespace { #on-a-namespace }

Declare a reducer as a class attribute inside a `Namespace`. The channel name auto-fills from the attribute name (you don't pass `name=`), the namespace scope auto-fills to the enclosing class, and `EventGraph` auto-registers it when any handler subscribes to one of the namespace's events:

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

- Only sees events whose `__namespace__` matches. Child namespaces inherit parent reducers (dedup by name).
- Cross-namespace name collisions raise `TypeError` at graph construction.
- Explicit `reducers=[...]` wins on name conflict with an auto-discovered reducer.

## Graph-wide reducers

For reducers that span namespaces or aren't namespace-scoped, pass them explicitly via `reducers=[...]`. This is the form used by `message_reducer()`:

```python
messages = message_reducer()
graph = EventGraph([call_llm, handle_tools], reducers=[messages])

log = graph.invoke([
    SystemPromptSet.from_str("You are helpful."),
    UserMessageReceived(message=HumanMessage(content="Hi")),
])
```

## `Reducer`

Maps matching events to list contributions, merged by a binary operator (default `operator.add` for list concatenation). Any LangGraph-compatible reducer function works — e.g. `add_messages` for smart message deduplication.

```python
history = Reducer(name="history", event_type=UserMsg, fn=lambda e: [e.text], default=[])
```

## `ScalarReducer`

Last-write-wins for single values. Unlike `Reducer` (list), `ScalarReducer` injects the bare value — the most recent non-`SKIP` contribution. `None` is a valid value.

```python
temperature = ScalarReducer(name="temperature", event_type=TempSet, fn=lambda e: e.value, default=0.7)
```

Return `SKIP` from `fn` to leave the current value unchanged — distinguishes "set to `None`" from "don't update":

```python
from langgraph_events import SKIP

temperature = ScalarReducer(
    name="temperature",
    event_type=ConfigUpdated,
    fn=lambda e: e.temp if e.temp is not None else SKIP,
    default=0.7,
)
```

## `message_reducer`

Built-in reducer for LangChain message accumulation. Projects `MessageEvent.as_messages()` into the `messages` channel using `add_messages` for deduplication:

```python
messages = message_reducer()
# or with a default prompt
messages = message_reducer([SystemMessage(content="You are helpful.")])
```

Handler parameter `messages` matches the channel name:

```python
@on(UserMessageReceived, ToolsExecuted)
async def call_llm(event: Event, messages: list[BaseMessage]) -> LLMResponded: ...
```

See the [Conversation Agent pattern](patterns.md#conversation-agui) and [Supervisor](patterns.md#supervisor).

## Pre-seeding

Prefer seeding state as events — the log stays the single source of truth. When external state must be injected outside the event path (migration, test fixture), use `pre_seed`:

```python
graph.pre_seed(config, {"my_reducer": value})
graph.invoke(SeedEvent(), config=config)
# or: await graph.apre_seed(...)
```

Pre-seeded values bypass the event log — `log.filter()` won't reflect them.
