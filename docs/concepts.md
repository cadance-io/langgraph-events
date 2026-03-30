# Core Concepts

## Events

Events are frozen dataclasses that subclass `Event`.

```python
from langgraph_events import Event


class OrderPlaced(Event):
    order_id: str
    total: float
```

Handlers match by `isinstance`, so subscriptions to parent classes also match subclasses.

## `@on(*EventTypes)`

Use `@on(EventType)` to subscribe handlers.

```python
from langgraph_events import on


@on(OrderPlaced)
def reserve_inventory(event: OrderPlaced):
    ...
```

Handlers may return:

- one `Event`
- `None` for side effects
- `Scatter([...])` for fan-out

## `EventGraph`

`EventGraph` compiles handlers into a reactive loop.

```python
graph = EventGraph([reserve_inventory], max_rounds=50)
log = graph.invoke(OrderPlaced(order_id="A1", total=99.0))
```

## `EventLog`

The return value of `invoke()` / `ainvoke()`.

```python
request = log.latest(OrderPlaced)
all_shipments = log.filter(ShipmentCreated)
has_error = log.has(Halted)
```

## `Halted`

Returning `Halted` stops dispatch immediately.

## `Scatter`

Use `Scatter([event1, event2])` to fan out work into multiple events.

## `Interrupted` / `Resumed`

Use `Interrupted` subclasses to pause and wait for human input, then `resume()` with a typed event.

## Reducers

Reducers incrementally project event log data to named channels. Use them when repeatedly recomputing the same derived context is expensive.

- `Reducer` for accumulating projected values
- `ScalarReducer` for last-write-wins values
- `message_reducer()` for conversation histories
