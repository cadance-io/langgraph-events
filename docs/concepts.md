# Core Concepts

## State IS events

Append-only log of frozen, typed events. Handlers consume and emit events. Projections (`EventLog.filter()`, reducers) derive views. No mutable shared state.

## The taxonomy

Four event base classes, a `Namespace`, and the `Invariant` marker:

| Class | Role | Naming | Where it lives |
|---|---|---|---|
| `Namespace` | Group of related commands/events | Noun (`Order`) | Top-level |
| `Command` | Intent / request | **Imperative** (`Place`, `Ship`) | Nested in `Namespace` |
| `DomainEvent` | Fact inside the domain | Past-participle (`Placed`) | Nested in `Namespace` or `Command` |
| `IntegrationEvent` | Fact crossing a system boundary | Past-participle | Top-level only |
| `SystemEvent` | Library-emitted fact | Past-participle | Top-level. A `Halted` subclass can nest |
| `Invariant` | Named rule gating a handler | Noun phrase (`CustomerNotBanned`) | Anywhere. Nesting under `Command` is recommended |

`Invariant` is a marker class, not an `Event` subclass. See [control-flow](control-flow.md#invariants).

`Command` and `DomainEvent` must nest inside a `Namespace`. `IntegrationEvent` must be top-level. Direct `Event` subclassing is forbidden. All three raise `TypeError` at class creation.

```python
from langgraph_events import Command, DomainEvent, Namespace


class Order(Namespace):
    class Place(Command):
        customer_id: str

        class Placed(DomainEvent):
            order_id: str

        class Rejected(DomainEvent):
            reason: str

    class Shipped(DomainEvent):
        tracking: str
```

Nesting is syntactic only. `Order.Place.Placed` is a `DomainEvent` with a `__command__` back-reference, not a subclass of `Place`.

A declared command is a leaf of the class hierarchy. `class Rush(Order.Place)` raises `TypeError`. One `Command` is one intent, with its own handler, outcomes and node identity. Share common code through a helper function, not a base class. Subclassing a `Namespace` raises `TypeError` too. See [Reducers](reducers.md#on-a-namespace). Refining an event, `class FastPlaced(Order.Place.Placed)`, and subclassing `Interrupted`, `IntegrationEvent` or `Halted` keep working.

### `Command.Outcomes`

Auto-generated union of the command's nested `DomainEvent` classes. Use it in `isinstance` and as the inline-handler return contract. When mypy must see it, declare an `Outcomes: TypeAlias = …` yourself. The library checks it against the nested events at class creation.

```python
...
isinstance(evt, Order.Place.Outcomes)   # Placed OR Rejected
typing.get_args(Order.Place.Outcomes)   # (Placed, Rejected)
```

## Handlers { #on-decorator }

Two styles: inline on a `Command`, or external via `@on`.

### Inline { #inline-command-handlers }

Name the method after the verb (`place`, `ship`, …) or use `handle`. `self` is the command instance. Pass the class to `EventGraph`. No decorator is needed.

```python
from __future__ import annotations

from langgraph_events import Command, DomainEvent, EventGraph, Namespace


class Order(Namespace):
    class Ship(Command):
        order_id: str

        class Shipped(DomainEvent):
            tracking: str

        def ship(self) -> Order.Ship.Shipped:
            return Order.Ship.Shipped(tracking=f"track-{self.order_id}")


graph = EventGraph([Order.Ship])
# or register every inline handler on a namespace in one call:
graph = EventGraph.from_namespaces(Order)
```

- Exactly one public method per `Command`. A helper must be underscore-prefixed, or class creation raises `TypeError`.
- Annotated return types must cover every nested `DomainEvent`.
- Write the qualified name in the return annotation (`Order.Ship.Shipped`), not the bare nested name (`Shipped`). Under `from __future__ import annotations` the annotation is a string resolved against module globals only, so a bare nested name does not resolve. See [Return contract](#return-contract).
- Each member of the return annotation must be an `Event` subclass, `Scatter[...]` or `None`. Any other class, an event mixin too, raises `TypeError` at graph construction. For a mixin, name the `Event` classes that compose it.
- `DomainEvent`s nested inside a `Command` are **Command-private**. Only that Command's handler can emit them. A recovery handler emits a namespace-level sibling, for example `Order.Rejected`. A violation raises `CommandPrivacyError` at graph construction.

### External: `@on`

Three shorthand forms:

```python
...
# Bare: event type inferred from the annotation
@on
def notify(event: Order.Placed) -> None:
    log_to_audit(event)

# Modifiers only: event type inferred, modifiers applied
@on(raises=NotifyError)
def push_notification(event: Order.Placed) -> None: ...

# Explicit types: required for multi-event subscription
@on(UserMessage, ToolResults)
async def call_llm(event: Event) -> AssistantMessage: ...
```

Bare `@on` requires a single annotated `Event` parameter. Otherwise it raises at decoration.

`@on(Event)` subscribes a handler to every dispatched event. That includes library events such as `Resumed`, `HandlerRaised` and `HandlerRetried`. An `Interrupted` event reaches it when the thread resumes. A `Halted` event never reaches it, because a `Halted` event ends the run before any handler matches. `@on(Event)` does not hide an `OrphanedEventWarning`.

### Signature injection

Handler params resolve from:

- `log: EventLog`: full history
- `config: RunnableConfig` / `store: BaseStore`: LangGraph injections
- Reducer channel by **parameter name** (see [Reducers](reducers.md))
- Field matchers (external only): typed subset dispatch and injection
- **Services**: project dependencies registered on `EventGraph(services=...)`

Resolution order: reducer name, then library type, then service. The first match is used. An unresolved param raises `TypeError` at graph construction.

`services=` accepts two shapes. They are mutually exclusive per graph:

```python
...
# Type-keyed: handler params resolve by annotation. Same-type
# collisions rejected at build; subclass annotations match via MRO walk.
EventGraph(handlers=[...], services=[chat_model, session_factory])

class Order(Namespace):
    class Place(Command):
        class Placed(DomainEvent):
            order_id: str

        async def handle(self, chat_model: BaseChatModel) -> Order.Place.Placed: ...

# Name-keyed: handler params resolve by name. Multiple instances of same type allowed.
EventGraph(
    handlers=[...],
    services={"primary_chat": chat_a, "backup_chat": chat_b},
)

@on(SomeEvent)
def react(event, primary_chat, backup_chat) -> ...: ...
```

Type-keyed injection matches on the resolved annotation. An annotation imported only under `TYPE_CHECKING` does not resolve, so the parameter stays unclaimed and raises `TypeError` at graph construction.

Name-keyed injection needs no annotation. When the annotation resolves, the library checks the registered value against it at graph build. A plain value must be an instance of the annotation. A `RunScoped` factory must return a subtype of the annotation. A mismatch raises `TypeError`. An annotation that does not resolve, or that Python cannot test at run time, such as a `Protocol` without `@runtime_checkable`, still binds the parameter and emits a `UserWarning`.

#### Run-scoped services

A plain service is injected as the same object on every dispatch. For a value derived from the run's `RunnableConfig`, wrap a factory in `RunScoped` and place it in the name-keyed mapping. The library calls the factory with the node's `RunnableConfig` once per node call and injects the result under the handler's parameter name.

```python
...
from langgraph_events import RunScoped

def model_for(config: RunnableConfig) -> ConversationModel:
    return models.main(config)

EventGraph(
    handlers=[...],
    services={"model": RunScoped(model_for), "session_factory": session_factory},
)

@on(SomeEvent)
async def handle(event: SomeEvent, model: ConversationModel) -> ...: ...
```

Rules:

- The factory must be a plain function with a return annotation. A coroutine function, or a factory without a return annotation, raises `TypeError` at `RunScoped(...)`. A `functools.partial` is read through to the wrapped function. A callable object is read through its `__call__`.
- The return annotation must resolve at graph build, so declare its class at module level. It must be a nominal subtype of the handler parameter's annotation, or the build raises `TypeError`. A factory annotated `-> Any` or with a bare type variable is not checked. `Any` nested in the return type, or an unparameterised generic, is not a subtype of a parameterised annotation.
- The factory runs synchronously on both the `invoke` and the `ainvoke` path. Do not do I/O in it.
- `RunScoped` is rejected in the type-keyed sequence form.
- A resumed run receives a freshly built config, not a checkpointed one.
- The factory runs before the handler's `raises=` boundary. An error in the factory is not caught by `raises=`, is not retried, and does not produce `HandlerRaised`. It surfaces as an unhandled node error, with a note that names the handler and the parameter. Keep the factory to a lookup. Validate the config in the factory and raise a clear error.

### Return contract

- Annotated handlers must return a type in the declared union, or `None`.
- Unannotated `Command`-subscribing handlers must return one of `Command.Outcomes`, or `None`. Other unannotated handlers keep a shape-only check.
- Omitting the return annotation is legal. An annotation that fails to resolve raises `TypeError` at graph construction.
- Violations raise `TypeError` at dispatch.

## `EventGraph`

```python
...
graph = EventGraph([place, respond], max_rounds=100)
```

Topology is derived from handler subscriptions. No manual node or edge wiring is needed. `max_rounds` (default 100) sets the recursion limit and emits `MaxRoundsExceeded` (a `Halted` subtype) on overflow. `graph.compiled` exposes the underlying `CompiledStateGraph` for subgraph composition or direct state access.

### Topology, graph and store { #topology-graph-and-store }

`EventGraph` has three parts. Each part is a class that you can build alone:

| Class | Holds | Needs | Use it to |
|---|---|---|---|
| `EventTopology` | the handlers, reducers and namespaces | no service, no checkpointer | render the graph, list handler names, feed the migration gate |
| `EventStore` | a topology and a checkpointer | no service | find, abandon, rewrite and verify stored threads |
| `EventGraph` | a topology, its services and a checkpointer | every service a handler asks for | run, resume and stream threads. `graph.event_store` is its store |

A client that never runs a handler does not need the services. Build the topology once, and
bind the services only where a handler runs:

```python
from langgraph_events import EventGraph, EventStore, EventTopology

topology = EventTopology([place, respond], reducers=[messages])
print(topology.namespaces().mermaid())
print(topology.requirements)  # one ServiceRequirement per service parameter

graph = EventGraph.of(topology, services={"llm": llm}, checkpointer=saver)

store = EventStore(topology, saver)
for config in store.threads_paused_on(ApprovalRequested, thread_ids=ids):
    store.abandon(config, reason="retired")
```

- `topology.requirements` lists each handler parameter that only a service can fill, as
  `ServiceRequirement(node, qualname, param, annotation)`. `node` is the graph-node name of
  the handler. `qualname` is the module and qualified name of the handler function. The
  annotation is `Unannotated()` when the parameter has none. It is `UnresolvedAnnotation(reason)` when it did not resolve.
- `EventGraph.of(topology, ...)` takes the keyword arguments of `EventGraph(...)` except
  `reducers`. A requirement that no service matches raises the same `TypeError` as
  `EventGraph(...)`.
- `EventGraph(handlers, ...)` is `EventGraph.of(EventTopology(handlers, reducers=...), ...)`.
- `graph.event_store` is the store of a graph. It settles a thread through the graph, with its
  services. A graph with no checkpointer has no store, and `graph.event_store` raises `ValueError`.
- `EventStore` does not wire a serde. Set `checkpointer.serde = topology.serde()` before you
  call a method that needs one. `topology.serde()` is the serde that
  `from_namespaces(..., checkpointer=...)` wires. See [Event migrations](event-migrations.md).

### Namespace introspection & visualization

- `graph.namespaces()` returns a `NamespaceModel`: code-derived structure and choreography.
- **Render**: `.text()` (tree), `.text(view="structure")` (taxonomy only), `.mermaid()` (flowchart), `.json()` / `.to_dict()`.
- **Inspect** (all frozen dataclass tuples/dicts):
    - `.namespaces`: `dict[str, NamespaceModel.Namespace]`
    - `.command_handlers`, `.policies`, `.edges`, `.seeds`, `.integration_events`, `.system_events`
- `Edge` carries `kind` (how: `solid`/`scatter`/`raises`/`retry`/`framework`) and `causation` (causal role: `intent`/`react`/`orchestrate`/`chain`). Both surface in `text()`, `json()` and mermaid styling.

Rendered diagrams live on the [Patterns](patterns.md) page. The collapsible legend shows the shape and edge vocabulary.

#### Focused diagrams

`mermaid(focus=...)` draws the selected items, the edges that touch them, and the nodes at both
ends, dimmed as context. `notes` writes runtime facts on nodes and on a handler's edges, and
`muted` fades what the client does not use. A count of how often a handler wrote events comes
from the recorded causes:

```python
...
from collections import Counter

from langgraph_events import Cause, NamespaceModel

model = graph.namespaces()
print(list(model.namespaces), [r.name for r in model.reactions])  # valid names

log = graph.invoke(Order.Place(customer_id="c1"))
emitted = Counter(c.via for e in log if isinstance(c := log.cause(e), Cause))

print(
    model.mermaid(
        focus=NamespaceModel.Focus(namespaces="Order", reactions="notify_customer"),
        notes={via: f"emitted {n}x" for via, n in emitted.items()},
        show_raises=False,
    )
)
```

Every `Cause.via` is a valid note key: an `@on` handler by its name in `model.reactions`, an
inline command handler by its command qualname. A note on an `@on` handler goes under the label
of each of its edges. A handler that returns `None` writes no event, so the count means
"emitted", not "dispatched".

## `EventLog`

Immutable, ordered container returned by `invoke` / `ainvoke`. Inject by type hint.

| Method | Returns |
|---|---|
| `log.filter(T)` | `list[T]` |
| `log.latest(T)` / `log.first(T)` | `T \| None` |
| `log.latest(T, not_followed_by=U)` | `T \| None`: the most recent `T`, or `None` when a `U` comes after it. See [A request that is still open](#a-request-that-is-still-open) |
| `log.has(T)` | `bool` |
| `log.count(T)` | `int` |
| `log.select(T)` / `log.after(T)` / `log.before(T)` | chainable `EventLog` |
| `log.cause(e)` | the origin of `e`: `Cause`, `NotRecorded`, `SourceDropped`, `FrameworkEvent`, or `None` for a seed. See [Causes](#causes) |
| `log.effects(e)` | `tuple[Event, ...]`: the events that a handler produced from `e`, in log order |
| `log.flow(e)` | `tuple[Event, ...]`: the chain of `Cause` that ends at `e` |
| `log.causes` | one origin per event, aligned with `events`, or `None` when the log records no causes |
| `len(log)`, `log[i]` | container protocol |

### A request that is still open

`not_followed_by=` finds the latest request that no answer follows yet. Complex event
processing calls it a negation, or "A not followed by B". Events of `U` before the most recent
`T` do not count.

```python
...
@on(MessageReceived)
def reply(event: MessageReceived, log: EventLog) -> MessageSent:
    pending = log.latest(QuestionAsked, not_followed_by=Answered | Withdrawn)
    if pending is None:
        return MessageSent(text="No question waits for an answer.")
    return MessageSent(text=f"Still waiting for an answer to: {pending.question}")
```

| Rule | Detail |
|---|---|
| Value of `U` | One `Event` class, or a union of `Event` classes written with `\|`. Any other value raises `TypeError`, also on an empty log. |
| An event of both `T` and `U` | It is a match, not a follower. So `T` can also be a member of `U`. |
| A later subclass of `T` | It is a match, also when it is a member of `U`. |
| An older `T` with no follower | It is not returned. Only the most recent `T` is checked. To list every open `T`, loop over `log.filter(T)`. |
| Fields | They are not compared. Any event of `U` closes the `T`. To pair a request with its own answer, compare an ID in your handler. |
| `after(T)` and `before(T)` | They cut at the first `T`, not the latest one. So `log.after(T)` does not give this query. |

### Causes

A graph run records the origin of each event. `log.cause(e)` states it, one case per type:

| `log.cause(e)` returns | Meaning |
|---|---|
| `Cause(source, via)` | The dispatch of `source` to the handler `via` wrote `e`. Usually the handler returned `e`. `InvariantViolated`, `HandlerRaised` and `HandlerRetried` carry the `Cause` of the dispatch they report. |
| `NotRecorded()` | `e` comes from history written before causes existed. |
| `SourceDropped(via, source_type)` | A handler produced `e`, but `rewrite_store(drop=...)` deleted its source. The handler and the type of the deleted event stay known. |
| `FrameworkEvent()` | The library wrote `e`: `RunPaused`, `MaxRoundsExceeded`, `Cancelled`, `Abandoned` or `Unresumable`. |
| `None` | A seed: `e` came from outside, through the `invoke()` input or `pre_seed()`. |

`NotRecorded` and `SourceDropped` are subclasses of `UnknownCause`, and each has a `reason`
sentence. A thread that starts on this release, and that `rewrite_store(drop=...)` never
rewrites, has no unknown cause.

`via` equals `Edge.via` unless the handler has a stable identity. For an inline command handler,
`via` is the command qualname, such as `"Order.Place"`. For an `@on(node_name=...)` pin, it is
the pinned name. `Cause.source` is an event, while `Edge.source` is a type. `HandlerRaised`
states the same two facts as `handler` and `source_event`.

The lookup finds `e` by identity first, then as its one equal event. A copy that matches several
equal events raises `ValueError`: pass the logged object. A log from `after()`, `before()` or
`select()` answers like the log it came from. `cause()` also raises `ValueError` when the log
records no causes, or when `e` is not in the root log.

One dispatch can write several events, for example through `Scatter`. To count how often a
handler was dispatched, count the distinct `id(c.source)` over its `Cause` entries. A
`SourceDropped` keeps `via`, so those dispatches can be counted as a separate number.

The `log.flow(e)` chain starts at the first event whose own cause is not a `Cause`. The value
that answers an `Interrupted`, and the `Resumed` that the library creates, have that
`Interrupted` as their source.

A client that saves events in its own format saves each cause with them, and rebuilds the log
with `EventLog(events, causes=...)`. Each `Cause.source` must be the same object as an earlier
event: pass `events[j]`, not a copy. `EventLog(log.events, causes=log.causes)` rebuilds a root
log, unknown and library-written cases included. The round-trip, with the source saved as its
index in the event list:

```python
...
from langgraph_events import Cause, EventLog

events = list(log.events)
index = {id(e): i for i, e in enumerate(events)}
saved = [(index[id(c.source)], c.via) if isinstance(c, Cause) else c for c in log.causes]

causes = [Cause(events[c[0]], c[1]) if isinstance(c, tuple) else c for c in saved]
rebuilt = EventLog(events, causes=causes)
```

A save written before causes existed has no entries. Load each of its events as `NotRecorded()`,
never as `None`, because `None` claims that the event is a seed.

A direct `graph.compiled.update_state()` or `graph.compiled.invoke()` that writes `events` must
write one `causes` entry per event too. Otherwise every older cause shifts by one position, and
no check can detect it. `pre_seed()` writes the causes for you.

## `Namespace` as the declaration point

A `Namespace` is where related features attach:

- Declarative reducers as class attributes, auto-scoped to namespace events. See [Reducers](reducers.md#on-a-namespace).
- Class-level `invariants` / `raises` on a `Command`, forwarded to the inline command handler. See [Control Flow](control-flow.md#invariants).
- Grouping in `graph.namespaces()`.

### Namespace names are scoped to a graph { #namespace-scope }

A namespace name identifies one namespace **within a graph**, not within the process. Reducer discovery and `graph.namespaces()` both group by name, so two namespaces of the same name in one graph is an error:

```python
...
EventGraph([one.Place, two.Place])
# TypeError: Two different namespaces named 'Trading' reached this graph:
# app.a.Trading and app.b.Trading. Namespace names must be unique within a graph.
```

A namespace reached only through a handler's *return* type counts too.

Across graphs the name is free, so one process can run two versions of one namespace module in sequence against the same checkpointed log. Give each version its own `NamespaceAwareSerde(namespaces=[...])`, because identity resolution consults the serde's namespaces before it imports a module. See [the migration gate rules](migration-gate.md#the-rules).

## System events

`SystemEvent` subclasses control runtime flow. Subscribe like any event. See [Control Flow](control-flow.md) for `Interrupted` / `Resumed`, `HandlerRaised`, `HandlerRetried`, `InvariantViolated`. Full table in [API](api.md#system-events).

To retire an `Interrupted` subclass, settle its paused threads first with `graph.event_store.abandon(config)` / `.aabandon()`. See [Ending a pause without answering it](control-flow.md#ending-a-pause-without-answering-it-abandon).

Custom halts subclass `Halted` and nest under their namespace for locality. `graph.namespaces()` groups them with the namespace's events, not with the library's system events:

```python
...
class Content(Namespace):
    class Classified(DomainEvent):
        label: str

    class Blocked(Halted):
        label: str
```

## Mixins

`MessageEvent` is a plain mixin, not an `Event` subclass. It wraps a LangChain `BaseMessage` in a `message` or `messages` field. Compose it with any event branch and pair it with [`message_reducer()`](reducers.md#message_reducer):

```python
from langchain_core.messages import HumanMessage

from langgraph_events import IntegrationEvent, MessageEvent


class UserMessageReceived(IntegrationEvent, MessageEvent):
    message: HumanMessage
```

`SystemPromptSet` is a built-in `IntegrationEvent` + `MessageEvent`. See [Graph-wide reducers](reducers.md#graph-wide-reducers) for the invoke example.
