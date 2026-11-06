# Getting Started

## Install

Requires Python 3.11+.

```bash
pip install langgraph-events
pip install "langgraph-events[agui]"  # AG-UI adapter
```

## Model your domain

Group commands and events under a `Namespace`. For a simple case, colocate the handler inline. A `Command`'s nested `DomainEvent`s form its return contract (`Command.Outcomes`). See [Concepts](concepts.md#the-taxonomy) for the taxonomy.

```python
from __future__ import annotations

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

For class-level `invariants` / `raises` or cross-namespace handlers via `@on(...)`, see [Control Flow](control-flow.md#invariants).

## Run the graph

```python
seed = Order.Place(customer_id="alice", items=("book",))
log = graph.invoke(seed)                    # sync; returns EventLog
log = await graph.ainvoke(seed)             # async
for event in graph.stream_events(seed): ... # stream as produced
```

## Inspect

```python
...  # graph as above
print(graph.namespaces().text())             # human-readable tree
print(graph.namespaces().mermaid())          # Mermaid diagram
graph.namespaces().namespaces                # structured NamespaceModel access
log.filter(Order.Place.Placed)
log.latest(Order.Place.Rejected)
log.has(Order.Shipped)
```

## Cross-cutting events

`IntegrationEvent` is for a fact that does not belong to any namespace (an external signal, a shared event). It must live at module scope.

```python
from langgraph_events import IntegrationEvent


class MessageReceived(IntegrationEvent):
    text: str
```

## Common Tasks

| I want to... | Use | Docs |
|---|---|---|
| Query past events in a handler | `EventLog` (`log.filter()`, `log.latest()`) | [Concepts](concepts.md#eventlog) |
| Enforce a precondition before a handler runs | `invariants` on the `Command` (or `@on()` for non-Command handlers) | [Control Flow](control-flow.md#invariants) |
| Register every inline handler on a namespace | `EventGraph.from_namespaces(Order)` | [Concepts](concepts.md#inline-command-handlers) |
| Project state from events | `ScalarReducer` on the namespace class | [Reducers](reducers.md) |
| Project LangChain messages | `message_reducer()` | [Reducers](reducers.md#message_reducer) |
| Fan out parallel work | `Scatter[Item]` | [Control Flow](control-flow.md#scatter) |
| Pause for human approval | `Interrupted` + `graph.resume()` | [Control Flow](control-flow.md#interrupted-resumed) |
| Retire an `Interrupted` subclass | `graph.store.abandon(config)` / `.aabandon()` | [Control Flow](control-flow.md#ending-a-pause-without-answering-it-abandon) |
| Stop the graph early | Return a `Halted` subclass | [Concepts](concepts.md#system-events) |
| Catch handler exceptions | `raises` on the `Command` + `@on(HandlerRaised, ...)` | [Control Flow](control-flow.md#handler-exceptions) |
| Retry a transient failure | `retry = RetryPolicy(...)` alongside `raises=` | [Control Flow](control-flow.md#retries) |
| Stream LLM tokens | `astream_events(include_llm_tokens=True)` | [Streaming](streaming.md) |
| Connect to an AG-UI frontend | `AGUIAdapter` | [AG-UI](agui.md) |
| Diagnose a run with an agent | `graph.reflect(log)` + `run.tool()` | [Reflection](reflection.md) |
| Keep old checkpoints working after a rename | `@migrate_from` / `@on(previously=)` + coverage gates | [Event Migrations](event-migrations.md) |
