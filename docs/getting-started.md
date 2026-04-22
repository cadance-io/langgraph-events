# Getting Started

## Model your domain

Events are **facts** about what happened. Commands are **intents** for what should happen. Group both under a `Domain` — and, for simple cases, put the handler right there too:

```python
from langgraph_events import Domain, Command, DomainEvent, EventGraph


class Order(Domain):
    class Place(Command):
        customer_id: str
        items: tuple[str, ...]

        class Placed(DomainEvent):
            order_id: str

        class Rejected(DomainEvent):
            reason: str

        def handle(self) -> Placed | Rejected:
            if not self.items:
                return Order.Place.Rejected(reason="empty order")
            return Order.Place.Placed(order_id=f"o-{self.customer_id}")

    class Shipped(DomainEvent):
        tracking: str


graph = EventGraph([Order.Place])
log = graph.invoke(Order.Place(customer_id="alice", items=("book",)))
print(log.latest(Order.Place.Placed))
```

- `Order` is the domain (namespace). `Place` is a command. `Placed` / `Rejected` are its outcomes. `Shipped` is a free event.
- Commands use **imperative** names; events use **past-participle**.
- `Order.Place.Outcomes` is auto-generated as `Placed | Rejected` — used in `isinstance` and enforced as the handler's return contract.
- `handle(self)` is the command's inline handler; `self` is the event.

Need `invariants=`, `raises=`, or a handler across multiple event types? Use the external `@on(...)` form — see [Concepts](concepts.md#on-decorator).

## Run the graph

```python
log = graph.invoke(seed)                    # sync; returns EventLog
log = await graph.ainvoke(seed)             # async
for event in graph.stream_events(seed): ... # stream as produced
```

## Inspect

```python
print(graph.domain().text())             # human-readable tree (choreography)
print(graph.domain().mermaid())          # Mermaid diagram
graph.domain().domains                # structured DomainModel access
log.filter(Order.Place.Placed)
log.latest(Order.Place.Rejected)
log.has(Order.Shipped)
```

## Cross-cutting events

Events that don't belong to any domain — external facts, shared signals — use `IntegrationEvent`:

```python
from langgraph_events import Auditable, IntegrationEvent

class MessageReceived(IntegrationEvent):
    text: str

class TaskStarted(IntegrationEvent, Auditable):  # @on(Auditable) for auto-logging
    name: str
```

## Common Tasks

| I want to... | Reach for... | Docs |
|---|---|---|
| Query past events in a handler | `EventLog` (`log.filter()`, `log.latest()`) | [Concepts](concepts.md#eventlog) |
| Enforce a precondition before a handler runs | `invariants=` on `@on()` | [Control Flow](control-flow.md#invariants) |
| Register every inline handler on a domain | `EventGraph.from_domains(Order)` | [Concepts](concepts.md#inline-command-handlers) |
| Accumulate state across events | `ScalarReducer` on the domain class | [Reducers](reducers.md) |
| Accumulate LangChain messages | `message_reducer()` | [Reducers](reducers.md#message_reducer) |
| Fan out parallel work | `Scatter` | [Control Flow](control-flow.md#scatter) |
| Pause for human approval | `Interrupted` + `graph.resume()` | [Control Flow](control-flow.md#interrupted-resumed) |
| Stop the graph early | Return a `Halted` subclass | [Concepts](concepts.md#system-events) |
| Catch handler exceptions | `raises=` + `@on(HandlerRaised, ...)` | [Control Flow](control-flow.md#handler-exceptions) |
| Stream LLM tokens | `astream_events(include_llm_tokens=True)` | [Streaming](streaming.md) |
| Connect to an AG-UI frontend | `AGUIAdapter` | [AG-UI](agui.md) |
