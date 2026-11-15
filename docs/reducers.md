# Reducers

- **Use a reducer** in one of three cases. The projection runs in many handlers. You want a value injected by parameter name. A recomputation from the full log each round would be expensive, for example a LangChain message history.
- **Use `EventLog.filter(T)`** for one-shot lookups inside a single handler.

## On a Namespace { #on-a-namespace }

Declare reducers as `Namespace` class attributes. The channel name auto-fills from the attribute name. The namespace scope auto-fills to the enclosing class. `EventGraph` auto-registers on first subscriber.

```python
from __future__ import annotations

from langgraph_events import Command, DomainEvent, Event, EventGraph, Namespace
from langgraph_events import ScalarReducer


class Order(Namespace):
    current_status = ScalarReducer(
        event_type=Event,
        fn=lambda e: (
            "shipped" if isinstance(e, Order.Shipped)
            else "placed" if isinstance(e, Order.Place.Placed)
            else None
        ),
    )

    class Place(Command):
        customer_id: str

        class Placed(DomainEvent):
            order_id: str

        def handle(self, current_status: str | None) -> Order.Place.Placed:
            # `current_status` is injected by parameter name.
            return Order.Place.Placed(order_id=f"o-{self.customer_id}")

    class Shipped(DomainEvent):
        tracking: str


graph = EventGraph([Order.Place])   # reducer auto-discovered from Order
```

- Only sees events whose `__namespace__` matches.
- Namespaces do not inherit: `class Child(Base)` where `Base` is a `Namespace` raises `TypeError`. To share a reducer, declare it free-standing (no owning namespace) and pass it via `reducers=[...]`.
- Cross-namespace name collisions raise `TypeError` at graph construction.
- Explicit `reducers=[...]` wins on conflict with an auto-discovered reducer. A reducer and a [`Derived`](#derived) of one name raise `TypeError`, from any source.
- A reducer in `reducers=[...]` must have a `name`. An empty name raises `ValueError`.

## Graph-wide reducers

For a reducer that spans namespaces or is not namespace-scoped, pass it via `reducers=[...]`. `message_reducer()` uses this form:

```python
...
messages = message_reducer()
graph = EventGraph([call_llm, handle_tools], reducers=[messages])

log = graph.invoke([
    SystemPromptSet.from_str("You are helpful."),
    UserMessageReceived(message=HumanMessage(content="Hi")),
])
```

## `Reducer`

List accumulator. The default merge is `operator.add`. Any LangGraph-compatible reducer function works, for example `add_messages` for id-based dedup.

```python
...
history = Reducer(name="history", event_type=UserMsg, fn=lambda e: [e.text], default=[])
```

## `ScalarReducer`

Last-write-wins scalar. `None` is valid. Return `SKIP` from `fn` to leave the current value unchanged. This distinguishes "set to `None`" from "do not update".

```python
...
temperature = ScalarReducer(name="temperature", event_type=TempSet, fn=lambda e: e.value, default=0.7)

# Conditional updates with SKIP:
temperature = ScalarReducer(
    name="temperature",
    event_type=ConfigUpdated,
    fn=lambda e: e.temp if e.temp is not None else SKIP,
    default=0.7,
)
```

### Required values { #scalar-reducer-required }

The handler parameter annotation declares whether the value can be `None`. A non-`None` annotation opts the parameter into a runtime assertion. `str | None`, `Optional[str]`, `Any`, `object` and a missing annotation opt out. When the channel value is `None` at injection time, `ReducerNotSetError` raises **before** the handler body runs.

```python
...
from langgraph_events import ReducerNotSetError  # catch it if you want


@on(TaskReceived)
def strict(event: TaskReceived, strategy: str) -> Completed:
    # `strategy` is guaranteed non-None, else ReducerNotSetError at injection.
    ...

@on(TaskReceived)
def permissive(event: TaskReceived, strategy: str | None) -> Completed:
    if strategy is None: ...
```

`ReducerNotSetError` is raised outside the `raises=` boundary, so a broad `raises=ValueError` cannot swallow it. See [Errors](api.md#errors). When a parameter annotation does not resolve, the library emits a `UserWarning` for that parameter and treats it as permissive. Annotate against an importable type.

## `FoldReducer`

Accumulator whose next value depends on the **prior** state, a left-fold. `Reducer` appends and `ScalarReducer` takes the last write. Use `FoldReducer` for a counter, a merged dict or a re-derived cursor. Each event owns its transition via `fold(self, state)`, the same polymorphic style as `MessageEvent.as_messages()`. Supply only `name`, `event_type` and a `default_factory`.

```python
from langgraph_events import RESET, FoldReducer, IntegrationEvent


class Incremented(IntegrationEvent):
    by: int = 1
    def fold(self, state): return {"n": state["n"] + self.by}

class Reset(IntegrationEvent):
    def fold(self, state): return RESET   # clears the channel


counter = FoldReducer(
    name="counter",
    event_type=(Incremented, Reset),
    default_factory=lambda: {"n": 0},
)
```

`fold` returns the new state (any value, including `None`), `RESET` to clear the channel back to `default_factory()`, or `SKIP` to leave it unchanged. Pass an explicit `fold=lambda state, event: ...` for an event that carries no `fold` method. The fold state can be any type, a `list` too. Like `ScalarReducer`, a non-`None` handler annotation opts into the [`ReducerNotSetError`](#scalar-reducer-required) assertion.

### Typing { #fold-typing }

`FoldReducer` is generic over its state type `S`, inferred from `default_factory`, so `reducer.empty` and `reducer.seed(...)` are typed (not `Any`):

```python
...
counter = FoldReducer(default_factory=lambda: {"n": 0}, event_type=(Incremented, Reset), name="counter")
reveal_type(counter.empty)   # dict[str, int]
```

The event argument is typed against `Foldable`, a `@runtime_checkable` Protocol satisfied **structurally**. Your event needs only a `fold` method. Do **not** inherit `Foldable`, because it clashes with the event metaclass. An explicit `fold=` function annotates its event parameter as `Foldable`, or leaves it inferred:

```python
...
def step(state: dict[str, int], event: Foldable) -> dict[str, int]:
    return event.fold(state)
```

To have mypy flag a `default_factory`/`fold` state-shape disagreement, pin the state type: `FoldReducer[dict[str, int]](...)`. Without a pin, mypy widens `S` to the join of the two and does not catch the mismatch. The generic does not propagate to handler params. Those are injected by name at runtime.

For a bespoke channel, subclass the public `BaseReducer` directly.

## `Derived` { #derived }

A `Derived` is a value that a pure function computes from its inputs. An input is a stored reducer, another `Derived`, or the `EventLog`. It is not a stored reducer: it has no channel, and the checkpoint never holds it. Use it when a handler acts on a fact that follows from the state. Without it, each handler declares every input and computes the fact again.

One stored reducer cannot give this value. A `FoldReducer` counts the attempts, and a `ScalarReducer` holds the limit that a configuration event sets:

```python
from langgraph_events import (
    Derived, EventGraph, FoldReducer, IntegrationEvent, ScalarReducer, on,
)


class Attempted(IntegrationEvent):
    def fold(self, state: int) -> int:
        return state + 1


class LimitSet(IntegrationEvent):
    limit: int


attempts = FoldReducer(name="attempts", event_type=Attempted, default_factory=int)
limit = ScalarReducer(name="limit", event_type=LimitSet, fn=lambda e: e.limit, default=3)


def remaining_of(attempts: int, limit: int) -> int:
    return max(limit - attempts, 0)


remaining = Derived(name="remaining", fn=remaining_of)


@on(Attempted)
def retry(event: Attempted, remaining: int) -> Retried | GaveUp:
    # `remaining` is injected by parameter name, like a reducer value.
    return Retried() if remaining else GaveUp()


graph = EventGraph([retry], reducers=[attempts, limit, remaining])
```

- **Inputs by parameter.** Each parameter of `fn` is one input. Before a handler runs, the framework calls `fn` with the values of the inputs at node start. An unset input gets the empty default of its reducer.
- **Kinds of input.** A parameter annotated with `EventLog` receives the `EventLog` of the state, with its causes. The parameter name does not matter. For a `functools.partial` or a callable object, the annotations are read from the wrapped function or from `__call__`. Each other parameter name names a stored reducer or another `Derived`. Another `Derived` gives its computed value. `Derived` values must not read each other in a cycle.
- **The log needs an annotation.** A lambda cannot annotate a parameter, so use a `def`, a `functools.partial` of one, or a callable object. A lambda parameter named `log` names a reducer called `log`, not the `EventLog`. The annotation must resolve at run time. An `EventLog` import under `TYPE_CHECKING` does not resolve. A log parameter must have no default, and a partial must not bind it.
- **Once per state read.** One state read computes each `Derived` once, also when several values read it. A handler injection, a stream frame, an AG-UI snapshot and `reflect(log).state()` are each one state read. Nothing is kept across reads. So `fn` runs again in the next read, and an input that reads the log scans the log again. Keep `fn` cheap, or memoize it yourself.
- **Not stored.** The value is never written to the checkpoint. It cannot drift from its inputs, and a new `Derived` needs no backfill on an existing thread. `pre_seed()` and `apre_seed()` refuse a `Derived` name with `ValueError`.
- **Registration.** Pass it in `reducers=[...]`, or declare it as a `Namespace` class attribute. On a namespace, `name` and `namespace` auto-fill, as on a reducer. An input can be any registered stored reducer or `Derived`. The namespace places the value in the structure views only.
- **Names.** `topology.reducer_names` and `topology.reducers` hold the stored reducers only. `topology.injectable_names` holds every name a handler can read: each stored reducer and each `Derived`. `topology.derived` maps each `Derived` by its name.
- **Required values.** As on a `ScalarReducer` parameter, a non-`None` annotation opts into [`ReducerNotSetError`](#scalar-reducer-required). The message asks you to make the `Derived` return a non-`None` value, or to permit `None`.
- **Errors.** `fn` runs before the handler body, outside the `raises=` boundary, like `ReducerNotSetError`. The exception keeps its type and gets a note that names the `Derived`.
- **Services.** A service must not have the name of a `Derived` or a reducer. Graph construction raises `TypeError`, because a parameter cannot name both. A `RunScoped` factory can read a `Derived` by name. See [Run-scoped services](concepts.md#run-scoped-services).
- **Streams and clients.** `stream_events(include_reducers=...)`, `stream_resume(include_reducers=...)` and `AGUIAdapter(include_reducers=...)` accept a `Derived` name. A client then gets the value in `StreamFrame.reducers` and in the `STATE_SNAPSHOT`. `include_reducers=True` selects each stored reducer and each `Derived`, in registration order. `AGUIAdapter` drops a `Derived` key from the inbound client state.
- **A thread with no checkpoint.** `AGUIAdapter.connect()` and `snapshot()` compute each selected value from the empty defaults, as a handler and a stream frame do.
- **Empty defaults.** `fn` must accept the empty default of each input, for example `None` for an unset `ScalarReducer`. An AG-UI replay of a thread with no checkpoint calls `fn` with these defaults, before any handler runs. If `fn` raises there, `connect()` and `snapshot()` end with `RUN_ERROR`.
- **The same value in each place.** A handler parameter, a `StreamFrame`, the AG-UI `STATE_SNAPSHOT` and `reflect(log).state()` compute the value with the same function. A `Derived` that reads the log gets the log of that place. The table below names that log. A thread with no checkpoint gives an empty log.
- **Structure views.** `namespaces().text()`, `.mermaid()` and `.json()` show each `Derived` with its inputs. The mermaid diagram draws it as a hexagon, with a dotted `reads` arrow from each input reducer or `Derived`. The log is no node, so it gets no arrow. The hexagon of a `Derived` that reads the log has the label `reads: log`. The text lists `log` as an input. The JSON gives `"reads_log": true`.
- **Reflection.** `graph.reflect(log).state()` projects each stored reducer over the log, then computes each `Derived` from those projections. A `Derived` that reads the log gets the reflected log.

The log that a `Derived` reads in each place:

| Place | Log |
|---|---|
| Handler parameter | The log at node start |
| `AGUIAdapter` stream frame, and an `astream_events()` or `astream_resume()` frame with `include_llm_tokens=True` or `include_custom_events=True` | The log up to the event of the frame |
| Each other stream frame, from `stream_events()`, `stream_resume()`, `astream_events()` or `astream_resume()` | The log at the end of the step of the event. The frames of one step show the same value |
| AG-UI `STATE_SNAPSHOT` | The checkpointed log |
| `reflect(log).state()` | The reflected log |

After a resume of handlers that paused in one step, the last `astream_events()` frame can order the events of that step differently from the checkpoint. A `Derived` that depends on the order of the log can then show another value in that frame than in the next `STATE_SNAPSHOT`.

A `Derived` that reads the log, and a second `Derived` that reads the first:

```python
from langgraph_events import (
    Derived, EventGraph, EventLog, IntegrationEvent, ScalarReducer, on,
)


class Shipped(IntegrationEvent):
    carrier: str


class Audited(IntegrationEvent):
    summary: str


def carriers_of(log: EventLog) -> tuple[str, ...]:
    return tuple(e.carrier for e in log.filter(Shipped))


carriers = Derived(name="carriers", fn=carriers_of)
summary = Derived(name="summary", fn=lambda carriers: ", ".join(carriers) or "none")


@on(Shipped)
def audit(event: Shipped, summary: str) -> Audited:
    return Audited(summary=summary)


graph = EventGraph([audit], reducers=[carriers, summary])
log = graph.invoke(Shipped(carrier="post"))
assert [e.summary for e in log.filter(Audited)] == ["post"]
```

Two limits apply to `StreamFrame.changed_reducers`:

- It names a `Derived` when an event changes one of its inputs, also when the computed value stays the same. The framework does not call `fn` to compare the values. An input of another `Derived` counts as an input. A `Derived` that reads the log is named for each event, because each event changes the log.
- A `Derived` with no parameter reads no input. Its value is the same for each frame, and `changed_reducers` never names it.

Graph construction raises an error in each of these cases:

| Case | Error | Reason |
|---|---|---|
| A parameter of `fn` is not annotated with `EventLog`, and its name names no registered reducer and no `Derived` | `TypeError` | The input does not exist. The message lists the available reducers and `Derived` values |
| That parameter has a default | `TypeError` | A default does not make a parameter optional. Bind a constant in a closure |
| That parameter has an annotation that does not resolve | `TypeError` | The message names the resolution error. Make the annotation importable at run time |
| `Derived` values read each other in a cycle, or a `Derived` reads itself | `TypeError` | The value cannot be computed. The message names the cycle, for example `first -> second -> first` |
| `fn` has a `*args`, `**kwargs` or positional-only parameter | `TypeError` | Each parameter must name one input |
| `inspect.signature` cannot read `fn`, for example a builtin or a value that is not callable | `TypeError` | The parameter names of `fn` name its inputs. The message names the `Derived` |
| A reducer and a `Derived` have the same name, from `reducers=[...]` or from a namespace | `TypeError` | A handler parameter cannot name both |
| A `Derived` has no name | `ValueError` | A handler parameter cannot name it. Pass `name=`, or declare it on a namespace |
| The name of a `Derived` is a reserved state field, for example `events` | `ValueError` | The framework owns that field. The message says that the name is a `Derived` name |

For two entries of the same kind and name, an entry in `reducers=[...]` wins over a namespace declaration. In `reducers=[...]`, a later entry wins over an earlier one.

## `message_reducer`

Built-in reducer for LangChain messages. It projects `MessageEvent.as_messages()` into the `messages` channel, using `add_messages` for id-based dedup.

A message without an id gets a UUID id when the reducer projects it, in place on the message object, as `add_messages` does. The id is then saved with the step's write. So a step that pauses keeps one id for the message, and every `MessagesSnapshot` shows that id.

```python
...
messages = message_reducer()
# or with a default prompt:
messages = message_reducer([SystemMessage(content="You are helpful.")])

@on(UserMessageReceived, ToolsExecuted)
async def call_llm(event: Event, messages: list[BaseMessage]) -> LLMResponded: ...
```

See [Conversation Agent](patterns.md#conversation-agui) and [Supervisor](patterns.md#supervisor).

## Pre-seeding

Seed state via events, so the log stays the single source of truth. For external state injection outside the event path (a migration, a test fixture), use `pre_seed`:

```python
...
graph.pre_seed(config, {"my_reducer": value})  # or: await graph.apre_seed(...)
graph.invoke(SeedEvent(), config=config)
```

A pre-seeded value bypasses the event log. `log.filter()` does not reflect it.

## Recovering from projection changes { #replay_reducer }

When a reducer's `fn` or output shape changes, the cached channel value in existing checkpoints is stale. `replay_reducer(my_reducer, events)` rebuilds it from the event log that the checkpoint stores in its `events` channel. See [Recovering with `replay_reducer`](event-migrations.md#recovering-with-replay_reducer).

## Introspection

`graph.reflect(log).state()` projects every registered reducer over a run's
event log from scratch, the runtime answer to "what is the derived state
now", for code and for agents (via the `state` op of the `query_log` tool).
See [Reflection](reflection.md).
