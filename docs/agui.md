# AG-UI Protocol Adapter

`langgraph_events.agui` maps EventGraph streams to AG-UI protocol events so AG-UI-compatible frontends can consume your agents over SSE.

## Install

```bash
pip install "langgraph-events[agui] @ git+https://github.com/cadance-io/langgraph-events.git"
```

## Minimal FastAPI Integration

```python
from fastapi import FastAPI, Request
from ag_ui.core import RunAgentInput
from langgraph_events.agui import AGUIAdapter, create_starlette_response

app = FastAPI()
adapter = AGUIAdapter(graph, seed_factory=seed_factory)


@app.post("/api/copilotkit")
async def run(request: Request):
    input_data = RunAgentInput.model_validate_json(await request.body())
    return create_starlette_response(adapter.stream(input_data))
```

## Connect / Reconnect

Use `connect()` to rehydrate checkpointed state and pending interrupts without executing handlers again.

```python
events = [event async for event in adapter.connect(input_data)]
```

`reconnect()` is an alias for `connect()`.

## Custom Mapping

Add user mappers to convert domain events into specific AG-UI events before fallback mapping.

```python
adapter = AGUIAdapter(graph, seed_factory=seed_factory, mappers=[MyMapper()])
```

## Resume Support

Provide `resume_factory` to translate frontend state into domain resume events.

```python
adapter = AGUIAdapter(
    graph,
    seed_factory=seed_factory,
    resume_factory=resume_factory,
)
```

For advanced mapper behavior and event type coverage, consult source docs and tests in `src/langgraph_events/agui/` and `tests/test_agui.py`.
