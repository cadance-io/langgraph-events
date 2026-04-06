# langgraph-events

Opinionated event-driven abstraction for LangGraph. **State IS events.**

!!! warning "Experimental (v0.2.0)"
    This is an early-stage personal project, not a supported product. The API will change without notice or migration path. Do not depend on this for anything you can't easily rewrite.

## What is this?

LangGraph gives you full control over agent topology, but wiring `StateGraph` nodes and conditional edges by hand is tedious. **langgraph-events** replaces that boilerplate with a reactive, event-driven model: define domain events as frozen dataclasses, subscribe handler functions with `@on(EventType)`, and let `EventGraph` derive the full graph topology automatically.

The core principle: **state IS events.** The entire state of a run is an append-only log of typed, immutable events. Handlers read events in; handlers emit events out. The framework does the rest.

## Installation

Not published to PyPI yet. Install directly from GitHub:

```bash
pip install git+https://github.com/cadance-io/langgraph-events.git

# With AG-UI adapter support (installs ag-ui-protocol)
pip install "langgraph-events[agui] @ git+https://github.com/cadance-io/langgraph-events.git"
```

Requires Python 3.10+ and `langgraph >= 0.2.0` (installed automatically). The `[agui]` extra adds `ag-ui-protocol` for the [AG-UI protocol adapter](agui.md).

## Next Steps

- Start with [Getting Started](getting-started.md)
- Learn [Core Concepts](concepts.md)
- Then explore as needed:
    - [Control Flow](control-flow.md) — fan-out, human-in-the-loop
    - [Reducers](reducers.md) — incremental state accumulation
    - [Streaming](streaming.md) — real-time events, LLM tokens, telemetry
- Browse [Patterns](patterns.md) for complete runnable examples
- Check [API Reference](api.md) for the full export table
- [AG-UI Adapter](agui.md) for frontend streaming
- [Checkpointer Evolution](checkpointer-evolution.md) for persistence edge cases

## Status

This is a solo experiment, not a team-backed product. Expect:

- No changelog or migration guides between versions
- API surface may shrink or change significantly
- Bug reports welcome, but no SLA on fixes
