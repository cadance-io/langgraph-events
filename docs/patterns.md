# Patterns

Complete runnable examples in `examples/`. Each demonstrates a different combination of core concepts.

## Reflection Loop

Generate, critique, and revise in a loop with automatic revision caps. Demonstrates **multi-subscription** (`@on(WriteRequested, CritiqueReceived)`) for revision cycles, union return types for branching, and the `Auditable` trait for automatic logging.

- Code: [examples/reflection_loop.py](https://github.com/cadance-io/langgraph-events/blob/main/examples/reflection_loop.py)
- Flow: [examples/reflection_loop.graph.md](https://github.com/cadance-io/langgraph-events/blob/main/examples/reflection_loop.graph.md)

## ReAct Agent with Message Reducer

Tool-calling agent with incremental message history. Demonstrates **[`message_reducer()`](reducers.md#message_reducer)** for automatic LangChain message accumulation, `MessageEvent` for typed message wrapping, and a multi-subscription loop (`@on(UserMessageReceived, ToolsExecuted)`) that drives the ReAct cycle.

- Code: [examples/react_agent.py](https://github.com/cadance-io/langgraph-events/blob/main/examples/react_agent.py)
- Flow: [examples/react_agent.graph.md](https://github.com/cadance-io/langgraph-events/blob/main/examples/react_agent.graph.md)

## Multi-Agent Supervisor

Supervisor routes tasks to specialist agents and aggregates context. Demonstrates a **custom [`Reducer`](reducers.md#reducer)** for cross-agent context accumulation, tool-calling for structured routing decisions, and typed events for supervisor-to-specialist communication — no manual subgraph wiring.

- Code: [examples/supervisor.py](https://github.com/cadance-io/langgraph-events/blob/main/examples/supervisor.py)
- Flow: [examples/supervisor.graph.md](https://github.com/cadance-io/langgraph-events/blob/main/examples/supervisor.graph.md)

## Fan-Out / Fan-In (Map-Reduce)

Parallel document summarization with batch splitting and completion detection. Demonstrates **[`Scatter`](control-flow.md#scatter)** for fan-out to multiple work items and `EventLog.filter()` for gather/completion detection with a duplicate-harmless pattern.

- Code: [examples/map_reduce.py](https://github.com/cadance-io/langgraph-events/blob/main/examples/map_reduce.py)
- Flow: [examples/map_reduce.graph.md](https://github.com/cadance-io/langgraph-events/blob/main/examples/map_reduce.graph.md)

## Human-in-the-Loop Approval

Pause execution for human approval and resume with typed events. Demonstrates **[`Interrupted`](control-flow.md#interrupted-resumed)** for pausing the graph, `graph.resume()` with domain events, and revision cycles driven by human feedback. Requires a checkpointer (`MemorySaver`).

- Code: [examples/human_in_the_loop.py](https://github.com/cadance-io/langgraph-events/blob/main/examples/human_in_the_loop.py)
- Flow: [examples/human_in_the_loop.graph.md](https://github.com/cadance-io/langgraph-events/blob/main/examples/human_in_the_loop.graph.md)

## Content Pipeline

Safety gates with early termination and live event streaming. Demonstrates **[`Halted`](concepts.md#halted)** for safety-based graph termination, async **[streaming](streaming.md)** with `astream_events()`, `EventLog.has()` for existence checks, and keyword-based classification (no LLM required).

- Code: [examples/content_pipeline.py](https://github.com/cadance-io/langgraph-events/blob/main/examples/content_pipeline.py)
- Flow: [examples/content_pipeline.graph.md](https://github.com/cadance-io/langgraph-events/blob/main/examples/content_pipeline.graph.md)
