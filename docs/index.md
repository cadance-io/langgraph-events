# langgraph-events

Opinionated event-driven abstraction for LangGraph. **State IS events.**

## What is this?

`langgraph-events` replaces manual `StateGraph` wiring with an event-driven model:

- define domain events as frozen dataclasses
- subscribe handlers with `@on(EventType)`
- build `EventGraph` and let it derive topology automatically

Core principle: the full state is an append-only, typed event log.

## Installation

```bash
pip install git+https://github.com/cadance-io/langgraph-events.git
```

With AG-UI adapter support:

```bash
pip install "langgraph-events[agui] @ git+https://github.com/cadance-io/langgraph-events.git"
```

Requires Python 3.10+.

## Next Steps

- Start with [Getting Started](getting-started.md)
- Learn [Core Concepts](concepts.md)
- Browse [Patterns](patterns.md)
- Check [API Reference](api.md)
