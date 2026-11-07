# langgraph-events

Opinionated event-driven abstraction for LangGraph. **State IS events.**

> [!CAUTION]
> **Experimental (v0.47.1)** - This is an early-stage personal project, not a supported product. The API will change without notice or migration path.

## Quick Start

Group related commands and events into a `Namespace`. Colocate the handler on the command.

```python
from __future__ import annotations

from langgraph_events import Command, DomainEvent, EventGraph, Namespace


class Order(Namespace):
    class Place(Command):
        customer_id: str

        class Placed(DomainEvent):
            order_id: str

        def handle(self) -> Order.Place.Placed:
            return Order.Place.Placed(order_id=f"o-{self.customer_id}")


graph = EventGraph([Order.Place])
log = graph.invoke(Order.Place(customer_id="alice"))
print(log.latest(Order.Place.Placed))
```

`@on(...)` handlers add invariants, declared exceptions, and cross-namespace reactions. See [Concepts](docs/concepts.md) and [Control Flow](docs/control-flow.md).

## Installation

Requires Python 3.11+.

```bash
pip install langgraph-events

# With AG-UI adapter support
pip install "langgraph-events[agui]"

# From source (development)
pip install git+https://github.com/cadance-io/langgraph-events.git
```

## Documentation

- Docs site: <https://cadance-io.github.io/langgraph-events/>
- [Getting Started](docs/getting-started.md)
- [Core Concepts](docs/concepts.md)
- [API Reference](docs/api.md)

## Development

```bash
uv sync --group dev
uv run pytest tests/
uv run ruff check src/ tests/
uv run mypy src/
```

## License

MIT - see [`LICENSE`](LICENSE).
