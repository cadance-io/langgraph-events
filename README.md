# langgraph-events

Opinionated event-driven abstraction for LangGraph. **State IS events.**

> [!CAUTION]
> **Experimental (v0.4.0)** - This is an early-stage personal project, not a supported product. The API will change without notice or migration path.

## Quick Start

Group related commands and events into a `Domain`; colocate the handler on the command.

```python
from langgraph_events import Command, Domain, DomainEvent, EventGraph


class Order(Domain):
    class Place(Command):
        customer_id: str

        class Placed(DomainEvent):
            order_id: str

        def handle(self) -> Placed:
            return Order.Place.Placed(order_id=f"o-{self.customer_id}")


graph = EventGraph([Order.Place])
log = graph.invoke(Order.Place(customer_id="alice"))
print(log.latest(Order.Place.Placed))
```

External `@on(...)` handlers compose in the same graph — use them for
invariants, declared exceptions, or reactions across domains. See
[Concepts](docs/concepts.md) and [Control Flow](docs/control-flow.md).

## Installation

```bash
pip install langgraph-events

# With AG-UI adapter support
pip install "langgraph-events[agui]"

# From source (development)
pip install git+https://github.com/cadance-io/langgraph-events.git
```

## Documentation

- Docs site (GitHub Pages): <https://cadance-io.github.io/langgraph-events/>
- Local docs index: [`docs/index.md`](docs/index.md)
- Getting started: [`docs/getting-started.md`](docs/getting-started.md)
- Core concepts: [`docs/concepts.md`](docs/concepts.md)
- Patterns: [`docs/patterns.md`](docs/patterns.md)
- API reference: [`docs/api.md`](docs/api.md)
- AG-UI adapter: [`docs/agui.md`](docs/agui.md)
- Checkpointer and graph evolution: [`docs/checkpointer-evolution.md`](docs/checkpointer-evolution.md)

## Development

```bash
uv sync --group dev
uv run pytest tests/
uv run ruff check src/ tests/
uv run mypy src/
```

## License

MIT - see [`LICENSE`](LICENSE).
