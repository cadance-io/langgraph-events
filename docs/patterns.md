# Patterns

These examples are runnable from `examples/`.

## Reflection Loop

Generate -> critique -> revise with multi-subscription and revision caps.

- Code: [examples/reflection_loop.py](https://github.com/cadance-io/langgraph-events/blob/main/examples/reflection_loop.py)
- Flow: [examples/reflection_loop.graph.md](https://github.com/cadance-io/langgraph-events/blob/main/examples/reflection_loop.graph.md)

## ReAct Agent with Message Reducer

ReAct-style turns with incremental message history.

- Code: [examples/react_agent.py](https://github.com/cadance-io/langgraph-events/blob/main/examples/react_agent.py)
- Flow: [examples/react_agent.graph.md](https://github.com/cadance-io/langgraph-events/blob/main/examples/react_agent.graph.md)

## Multi-Agent Supervisor

Supervisor routes between specialists and aggregates context.

- Code: [examples/supervisor.py](https://github.com/cadance-io/langgraph-events/blob/main/examples/supervisor.py)
- Flow: [examples/supervisor.graph.md](https://github.com/cadance-io/langgraph-events/blob/main/examples/supervisor.graph.md)

## Fan-Out / Fan-In (Map-Reduce)

`Scatter` for fan-out and `EventLog.filter()` for gather completion.

- Code: [examples/map_reduce.py](https://github.com/cadance-io/langgraph-events/blob/main/examples/map_reduce.py)
- Flow: [examples/map_reduce.graph.md](https://github.com/cadance-io/langgraph-events/blob/main/examples/map_reduce.graph.md)

## Human-in-the-Loop Approval

Pause with `Interrupted`, resume with typed events.

- Code: [examples/human_in_the_loop.py](https://github.com/cadance-io/langgraph-events/blob/main/examples/human_in_the_loop.py)
- Flow: [examples/human_in_the_loop.graph.md](https://github.com/cadance-io/langgraph-events/blob/main/examples/human_in_the_loop.graph.md)

## Content Pipeline

Safety halting with `Halted`, plus live event streaming.

- Code: [examples/content_pipeline.py](https://github.com/cadance-io/langgraph-events/blob/main/examples/content_pipeline.py)
- Flow: [examples/content_pipeline.graph.md](https://github.com/cadance-io/langgraph-events/blob/main/examples/content_pipeline.graph.md)
