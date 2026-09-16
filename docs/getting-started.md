# Getting Started

## Model your domain

- **Events** — facts about what happened.
- **Commands** — intents for what should happen.

Group both under a `Namespace`; for simple cases, colocate the handler inline. A `Command`'s nested `DomainEvent`s auto-form its return contract (`Command.Outcomes`):

```python
from langgraph_events import Command, DomainEvent, EventGraph, Namespace


class Order(Namespace):
    class Place(Command):
        customer_id: str
        items: tuple[str, ...]

        class Placed(DomainEvent):
            order_id: str

        class Rejected(DomainEvent):
            reason: str

        def handle(self) -> Order.Place.Placed | Order.Place.Rejected:
            if not self.items:
                return Order.Place.Rejected(reason="empty order")
            return Order.Place.Placed(order_id=f"o-{self.customer_id}")

    class Shipped(DomainEvent):
        tracking: str


graph = EventGraph([Order.Place])
log = graph.invoke(Order.Place(customer_id="alice", items=("book",)))
print(log.latest(Order.Place.Placed))
```

See [Concepts](concepts.md#the-taxonomy) for the full taxonomy and the `Outcomes` contract. For class-level `invariants` / `raises` or cross-domain reactors via `@on(...)`, see [Control Flow](control-flow.md#invariants).

## Run the graph

```python
log = graph.invoke(seed)                    # sync; returns EventLog
log = await graph.ainvoke(seed)             # async
for event in graph.stream_events(seed): ... # stream as produced
```

## Inspect

```python
print(graph.namespaces().text())             # human-readable tree
print(graph.namespaces().mermaid())          # Mermaid diagram
graph.namespaces().namespaces                # structured NamespaceModel access
log.filter(Order.Place.Placed)
log.latest(Order.Place.Rejected)
log.has(Order.Shipped)
```

## Cross-cutting events

`IntegrationEvent` for facts that don't belong to any domain (external signals, shared events). Must live at module scope.

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
| Enforce a precondition before a handler runs | `invariants` on the `Command` (or `@on()` for non-Command handlers) | [Control Flow](control-flow.md#invariants) |
| Register every inline handler on a domain | `EventGraph.from_namespaces(Order)` | [Concepts](concepts.md#inline-command-handlers) |
| Accumulate state across events | `ScalarReducer` on the domain class | [Reducers](reducers.md) |
| Accumulate LangChain messages | `message_reducer()` | [Reducers](reducers.md#message_reducer) |
| Fan out parallel work | `Scatter[Item]` | [Control Flow](control-flow.md#scatter) |
| Pause for human approval | `Interrupted` + `graph.resume()` | [Control Flow](control-flow.md#interrupted-resumed) |
| Retire an `Interrupted` subclass | `graph.abandon(config)` / `.aabandon()` | [Control Flow](control-flow.md#ending-a-pause-without-answering-it-abandon) |
| Stop the graph early | Return a `Halted` subclass | [Concepts](concepts.md#system-events) |
| Catch handler exceptions | `raises` on the `Command` + `@on(HandlerRaised, ...)` | [Control Flow](control-flow.md#handler-exceptions) |
| Retry a transient failure | `retry = RetryPolicy(...)` alongside `raises=` | [Control Flow](control-flow.md#retries) |
| Stream LLM tokens | `astream_events(include_llm_tokens=True)` | [Streaming](streaming.md) |
| Connect to an AG-UI frontend | `AGUIAdapter` | [AG-UI](agui.md) |
| Keep old checkpoints working after a rename | `@migrate_from` / `@on(previously=)` + coverage gates | [Event Migrations](event-migrations.md) |
| Know what's safe to change on a live graph | — | [Checkpointer Evolution](checkpointer-evolution.md) |

## Where to go next

- **Modelling your domain** (taxonomy, handlers, signature injection) → [Concepts](concepts.md)
- **Enforcing rules / handling failures** (invariants, HITL, exceptions, scatter) → [Control Flow](control-flow.md)
- **Building a real example** → [Patterns](patterns.md)
- **Need the full export list?** → [API Reference](api.md)
