# Reducer services design (issue #193)

## Need

A reducer `fn` receives the event only. A handler can declare a name-keyed service as a
parameter. A reducer cannot. A downstream application needs a `RunScoped` service (the language
of the run) inside its `messages` reducer.

`RunScoped` is a service that a factory derives from the run config.

## API

```python
def project(event: Event, language: Language) -> list[BaseMessage]: ...

Reducer(name="messages", event_type=MessageEvent, fn=project)
EventGraph(handlers, reducers=[...], services={"language": RunScoped(language_for)})
```

- The first parameter of `fn` is the event.
- Each other required parameter resolves by name from the name-keyed `services` mapping. A
  plain value and a `RunScoped` value are both legal.
- A `fn` with one parameter keeps its current behaviour.
- A reducer `fn` receives no `config`, no `store` and no reducer value.

## Scope

- `Reducer` and `ScalarReducer` accept services.
- `FoldReducer` does not. LangGraph captures `FoldReducer._merge` of the original instance at
  compile, so a bound copy cannot reach the channel. A `fold` callable with a third required
  parameter raises `TypeError` at graph build.

## Mechanism: bind, do not thread

"Bind" means: make a copy of the reducer whose `fn` already holds its service values.

1. `collect(events)` and `seed(events)` keep their signature.
2. `BaseReducer._service_params` is a tuple of service parameter names. The default is `()`.
3. `BaseReducer._bind(values)` returns `self`. `Reducer` and `ScalarReducer` override it. The
   override returns `dataclasses.replace(self, fn=_BoundFn(self.fn, values))`.
4. `_BoundFn` is a callable wrapper. Its `repr` shows the service names and never the values.
5. An unbound reducer with service parameters raises `TypeError` in `collect`. The message names
   the reducer and the service parameters.
6. `bind_reducers(reducers, services_by_name, config)` returns the same dict when no reducer has
   a service parameter. Otherwise it returns a new dict with bound copies.
7. `EventGraph._reducers_for(config)` calls `bind_reducers`. `_bind`, `_BoundFn`,
   `bind_reducers` and `_reducers_for` are private.

## The constraint: checkpoint text equals streamed text

The node path receives a config that LangGraph enriched. The stream path holds the raw caller
config. A factory that reads an enriched key gives a different value on each path.

For the reducer seam, the factory receives a normalised config on every path:

```python
{"configurable": {k: v for k, v in configurable.items()
                  if not k.startswith(("__", "checkpoint_"))}}
```

- The normalised config has no `metadata`, no `callbacks` and no `tags`.
- A missing config becomes `{"configurable": {}}`.
- The handler seam does not change. A handler still receives a value from the full config.

Consequence: a handler and a reducer that share a `RunScoped` service call the factory two
times in one node call. This cost is accepted.

## Call sites that must use a bound dict

| Site | Config source |
|---|---|
| Handler node `_finalize` (`_internal.py`) | node `config` |
| Seed node, sync and async (`_internal.py`, `_graph.py` `aseed`) | node `config` |
| `Reflection` injected into a handler (`_internal.py` `_build_inject`) | node `config` |
| Stream shadow: `_update_reducer_state` and the seed frames (`_graph.py`) | `kwargs.get("config")` |
| `AGUIAdapter._reducer_updates_for` (`agui/_adapter.py`) | adapter `config` |
| `EventGraph.reflect(log, config=None)` | new optional parameter |
| `replay_reducer(reducer, events, *, services=None)` | plain values from the caller |

`_update_reducer_state` must take the reducer dict as a parameter. Concurrent streams share
`self._reducers`, so the attribute must not be swapped.

## Build check

The check runs in `EventGraph.__init__`, after the service registries exist.

- It reads `inspect.signature(fn)`. A `ValueError` or `TypeError` from `inspect.signature` means
  "no service parameter".
- A service parameter is a parameter after the first that has no default and whose kind is
  `POSITIONAL_OR_KEYWORD` or `KEYWORD_ONLY`.
- A required positional-only parameter after the first raises `TypeError`.
- A service parameter that is not a key of the name-keyed `services` raises `TypeError`. The
  message names the reducer, the parameter and the known services. It states that a reducer
  `fn` receives the event and name-keyed services only. With the type-keyed sequence form, the
  message states that a reducer service needs the mapping form.
- The 0.33.0 annotation check applies: a service is checked against the parameter annotation.

## Documented rule

A service used by a reducer must be stable for the life of the thread. A from-scratch projection
(`seed`, `reflect`, `replay_reducer`) recomputes from stored events with the current value.

## Tests

- One parametrised suite runs through both branches of `bind_reducers`.
- One test proves that the factory receives equal input on the node path and the stream path.
  The factory records its input. The event is a domain event, not `RunPaused`.
- The seed node has a sync test and an async test.
