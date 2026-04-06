# langgraph-events

Opinionated event-driven abstraction for LangGraph. **State IS events.**

> [!CAUTION]
> **Experimental (v0.2.0)** - This is an early-stage personal project, not a supported product. The API will change without notice or migration path.

## Quick Start

```python
from langgraph_events import Event, EventGraph, on


class MessageReceived(Event):
    text: str


class ReplyProduced(Event):
    text: str


@on(MessageReceived)
def reply(event: MessageReceived) -> ReplyProduced:
    return ReplyProduced(text=f"Echo: {event.text}")


graph = EventGraph([reply])
log = graph.invoke(MessageReceived(text="hello"))
print(log.latest(ReplyProduced))
```

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
