# Core Concepts

## State IS events

Append-only log of frozen, typed events. Handlers consume and emit events; projections (`EventLog.filter()`, reducers) derive views. No mutable shared state.

## The taxonomy

Four event base classes plus a `Namespace`:

| Class | Role | Naming | Where it lives |
|---|---|---|---|
| `Namespace` | Group of related commands/events | Noun (`Order`) | Top-level |
| `Command` | Intent / request | **Imperative** (`Place`, `Ship`) | Nested in `Namespace` |
| `DomainEvent` | Fact inside the domain | Past-participle (`Placed`) | Nested in `Namespace` or `Command` |
| `IntegrationEvent` | Fact crossing a system boundary | Past-participle | **Top-level only** (enforced) |
| `SystemEvent` | Framework-emitted fact | Past-participle | Top-level (`Halted` subclasses may nest for locality) |
| `Invariant` | Named rule gating a handler | Noun phrase (`CustomerNotBanned`) | Anywhere; nesting under `Command` encouraged |

Event class names use past-participle — they're facts. `Auditable` / `MessageEvent` are mixins (compose with any branch). `Invariant` is a marker class, not an `Event` subclass — see [control-flow](control-flow.md#invariants).

`Command` / `DomainEvent` must nest inside a `Namespace`; `IntegrationEvent` must be top-level; direct `Event` subclassing is forbidden. All three raise `TypeError` at class creation.

```python
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

Nesting is syntactic only — `Order.Place.Placed` is a `DomainEvent` with a `__command__` back-reference, not a subclass of `Place`.

A *declared* command is also a leaf of the class hierarchy: `class Place(Command)` declares one, but `class Rush(Order.Place)` raises `TypeError`. One `Command` is one intent, with its own handler, outcomes and node identity — two intents are two commands, sharing whatever they have in common through a helper function rather than a base class. Every other subclassing axis is unaffected: refining an event — `class FastPlaced(Order.Place.Placed)` — keeps working, as does subclassing a `Namespace` to inherit its reducers, or `Interrupted` / `IntegrationEvent` / `Halted` to declare your own.

### `Command.Outcomes`

Auto-generated union of the command's nested `DomainEvent` classes; used in `isinstance` and as the inline-handler return contract. Declare an `Outcomes: TypeAlias = …` yourself if you want mypy to see it — drift-checked against the nested events at class creation.

```python
isinstance(evt, Order.Place.Outcomes)   # Placed OR Rejected
typing.get_args(Order.Place.Outcomes)   # (Placed, Rejected)
```

## Handlers { #on-decorator }

Two styles: inline on a `Command`, or external via `@on`.

### Inline { #inline-command-handlers }

The **sole public method** in the class body. Name it after the verb (`place`, `ship`, …) or use `handle`. `self` is the command instance. Pass the class to `EventGraph` — no decorator.

```python
class Order(Namespace):
    class Ship(Command):
        order_id: str

        class Shipped(DomainEvent):
            tracking: str

        def ship(self) -> Order.Ship.Shipped:
            return Order.Ship.Shipped(tracking=f"track-{self.order_id}")


graph = EventGraph([Order.Ship])
# or register every inline handler on a namespace in one call:
graph = EventGraph.from_namespaces(Order, handlers=[react])
```

- Exactly one public method per `Command`; helpers must be underscore-prefixed (else `TypeError` at class creation).
- Annotated return types must cover every nested `DomainEvent`.
- Every name in the return annotation must resolve at run time from the handler module's globals. Write the qualified name (`Order.Ship.Shipped`), not the bare nested name (`Shipped`). A bare nested name works today but fails under `from __future__ import annotations`, because the annotation is then a string resolved against module globals only. An unresolvable return annotation raises `TypeError` at graph construction.
- `DomainEvent`s nested inside a `Command` are **Command-private** — only that Command's handler may emit them. Recovery reactors emit namespace-level siblings (e.g. `Order.Rejected`). Violations raise `CommandPrivacyError` at graph construction.

Declare `invariants` and `raises` as class-level attributes:

```python
class Order(Namespace):
    class Place(Command):
        customer_id: str = ""
        invariants = {CustomerNotBanned: lambda log: not log.has(CustomerBanned)}
        raises = (RateLimitError,)

        class Placed(DomainEvent):
            order_id: str = ""

        def handle(self) -> Order.Place.Placed:
            return Order.Place.Placed(order_id=f"o-{self.customer_id}")
```

### External: `@on`

Three shorthand forms:

```python
# Bare — event type inferred from the annotation
@on
def notify(event: Order.Placed) -> None:
    log_to_audit(event)

# Modifiers only — event type inferred, modifiers applied
@on(raises=NotifyError)
def push_notification(event: Order.Placed) -> None: ...

# Explicit types — required for multi-event subscription
@on(UserMessage, ToolResults)
async def call_llm(event: Event) -> AssistantMessage: ...
```

Bare `@on` requires a single annotated `Event` parameter (errors at decoration otherwise).

### Signature injection

Handler params resolve from:

- `log: EventLog` — full history
- `config: RunnableConfig` / `store: BaseStore` — LangGraph injections
- Reducer channel by **parameter name** (see [Reducers](reducers.md))
- Field matchers (external only) — typed subset dispatch + injection
- **Services** — project dependencies registered on `EventGraph(services=...)`

Resolution order: reducer name → framework type → service. First match wins. Unresolved params raise `TypeError` at graph construction.

`services=` accepts two shapes (mutually exclusive per graph):

```python
# Type-keyed: handler params resolve by annotation. Same-type
# collisions rejected at build; subclass annotations match via MRO walk.
EventGraph(handlers=[...], services=[chat_model, session_factory])

class Story(Namespace):
    class Refine(Command):
        class Refined(DomainEvent):
            text: str

        async def handle(self, chat_model: BaseChatModel) -> Story.Refine.Refined: ...

# Name-keyed: handler params resolve by name. Multiple instances of same type allowed.
EventGraph(
    handlers=[...],
    services={"primary_chat": chat_a, "backup_chat": chat_b},
)

@on(SomeEvent)
def react(event, primary_chat, backup_chat) -> ...: ...
```

The two shapes differ in what they need from the annotation.

Type-keyed injection matches on the resolved annotation. The annotation must be importable at run time. An annotation imported only under `TYPE_CHECKING` does not resolve, so the parameter stays unclaimed and raises `TypeError` at graph construction.

Name-keyed injection matches on the parameter name. It needs no annotation. When the parameter has an annotation that resolves, the framework checks the registered value against it at graph build. A plain value must be an instance of the annotation. A `RunScoped` factory must return a subtype of the annotation. A mismatch raises `TypeError`.

Two cases are not checked. An unresolvable annotation still binds the parameter, but the framework emits a `UserWarning`, because the annotation does not describe what is injected. An annotation that Python cannot test at run time, such as a `Protocol` without `@runtime_checkable`, is not checked. The framework emits a `UserWarning` for it.

#### Run-scoped services

A service in `services=` is injected as the same object on every dispatch. A value derived from the run's `RunnableConfig` cannot be registered as a plain service. `RunScoped` registers it.

Wrap a factory in `RunScoped` and place it in the name-keyed mapping. The framework calls the factory with the node's `RunnableConfig` once per node call. The result is injected under the handler's parameter name.

```python
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

- The factory must be a plain function. A coroutine function raises `TypeError` at `RunScoped(...)`, because the result would be injected without an `await`.
- The factory must have a return annotation. A factory without one raises `TypeError` at `RunScoped(...)`. A `functools.partial` is read through to the wrapped function. A callable object is read through its `__call__`.
- The return annotation must resolve at graph build. A class declared inside a function cannot be named from module globals. Declare it at module level. The framework compares the return annotation with the handler parameter's annotation. A return type that is not a subtype of the parameter annotation raises `TypeError`. A factory annotated `-> Any` or with a bare type variable is not checked. The comparison is nominal. `Any` nested in the return type, or an unparameterised generic, is not a subtype of a parameterised annotation. Align the two annotations.
- The factory runs synchronously on both the `invoke` and the `ainvoke` path. Do not do I/O in it.
- `RunScoped` is rejected in the type-keyed sequence form, because that form resolves by annotation.
- A resumed run receives a freshly built config. The factory runs against the current config, not a checkpointed one.
- The factory runs before the handler's `raises=` boundary. An error in the factory is not caught by `raises=`, is not retried, and does not produce `HandlerRaised`. It surfaces as an unhandled node error, with a note that names the handler and the parameter. Keep the factory to a lookup. It must not raise an error that the handler declares in `raises=`. Validate the config in the factory and raise a clear error.

### Return contract

- Annotated handlers must return a type in the declared union (or `None`).
- Unannotated `Command`-subscribing handlers must return one of `Command.Outcomes` (or `None`); other unannotated handlers keep a shape-only check.
- An omitted annotation is not the same as an unresolvable one. Omitting the return annotation is legal. An annotation that fails to resolve raises `TypeError` at graph construction.
- Violations raise `TypeError` at dispatch.

## `EventGraph`

```python
graph = EventGraph([place, respond], max_rounds=100)
```

Topology derived from handler subscriptions — no manual node/edge wiring. `max_rounds` (default 100) sets the recursion limit and emits `MaxRoundsExceeded` (a `Halted` subtype) on overflow.

### Namespace introspection & visualization

- `graph.namespaces()` returns a `NamespaceModel` — code-derived structure + choreography.
- **Render**: `.text()` (tree), `.text(view="structure")` (taxonomy only), `.mermaid()` (flowchart), `.json()` / `.to_dict()`.
- **Inspect** (all frozen dataclass tuples/dicts):
    - `.namespaces` — `dict[str, NamespaceModel.Namespace]`
    - `.command_handlers`, `.policies`, `.edges`, `.seeds`, `.integration_events`, `.system_events`
- `Edge` carries `kind` (how — `solid`/`scatter`/`raises`/`retry`/`framework`) and `causation` (causal role — `intent`/`react`/`orchestrate`/`chain`). Surfaces in `text()`, `json()`, and mermaid styling.

Rendered diagrams live on the [Patterns](patterns.md) page — the collapsible legend shows the shape/edge vocabulary.

#### Focused diagrams

A graph with a large fixed part and a small part that a client adds on top draws a tall diagram.
`mermaid(focus=...)` draws one part: the selected items, the edges that touch them, and the
nodes at both ends, dimmed as context. `notes` writes runtime facts on nodes and on a
reaction's edges, and `muted` fades what the client does not use. A count of how often a
handler wrote events comes from the recorded causes:

```python
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

Every `Cause.via` is a valid note key: a policy by its reaction name, an inline command
handler by its command qualname. A note on a reaction goes under the label of each of its
edges. A handler that returns `None` writes no event, so the count means "emitted", not
"dispatched".

### Escape hatch

`graph.compiled` exposes the underlying `CompiledStateGraph` for subgraph composition or direct state access.

## `EventLog`

Immutable, ordered container returned by `invoke` / `ainvoke`. Inject by type hint.

```python
@on(DraftProduced)
def evaluate(event: DraftProduced, log: EventLog) -> CritiqueReceived | FinalDraftProduced:
    if log.has(CritiqueReceived):
        ...
    last = log.latest(Order.Place.Placed)
    all_drafts = log.filter(DraftProduced)
```

| Method | Returns |
|---|---|
| `log.filter(T)` | `list[T]` |
| `log.latest(T)` / `log.first(T)` | `T \| None` |
| `log.has(T)` | `bool` |
| `log.count(T)` | `int` |
| `log.select(T)` / `log.after(T)` / `log.before(T)` | chainable `EventLog` |
| `log.cause(e)` | the origin of `e`: `Cause`, `NotRecorded`, `SourceDropped`, `FrameworkEvent`, or `None` for a seed. See [Causes](#causes) |
| `log.effects(e)` | `tuple[Event, ...]`: the events that a handler produced from `e`, in log order |
| `log.flow(e)` | `tuple[Event, ...]`: the chain of `Cause` that ends at `e` |
| `log.causes` | one origin per event, aligned with `events`, or `None` when the log records no causes |
| `len(log)`, `log[i]` | container protocol |

### Causes

A graph run records the origin of each event. `log.cause(e)` states it, one case per type:

| `log.cause(e)` returns | Meaning |
|---|---|
| `Cause(source, via)` | The dispatch of `source` to the handler `via` wrote `e`. Usually the handler returned `e`. `InvariantViolated`, `HandlerRaised` and `HandlerRetried` carry the `Cause` of the dispatch they report. |
| `NotRecorded()` | `e` comes from history written before causes existed. |
| `SourceDropped(via, source_type)` | A handler produced `e`, but `rewrite_store(drop=...)` deleted its source. The handler and the type of the deleted event stay known. |
| `FrameworkEvent()` | The framework wrote `e`: `RunPaused`, `MaxRoundsExceeded`, `Cancelled` or `Abandoned`. |
| `None` | A seed: `e` came from outside, through the `invoke()` input or `pre_seed()`. |

`NotRecorded` and `SourceDropped` are subclasses of `UnknownCause`, and each has a `reason`
sentence. A cause is unknown only for history that the framework did not record: a checkpoint
saved before causes existed, or a source that a store rewrite deleted. A thread that starts on
this release, and that `rewrite_store(drop=...)` never rewrites, has no unknown cause.

`via` equals `Edge.via` unless the handler has a stable identity. For an inline command handler,
`via` is the command qualname, such as `"Order.Place"`. For an `@on(node_name=...)` pin, it is
the pinned name. `Cause.source` is an event, while `Edge.source` is a type. `HandlerRaised`
states the same two facts as `handler` and `source_event`.

The lookup finds `e` by identity first, then as its one equal event. A copy that matches several
equal events raises `ValueError`, because picking one would be a guess: pass the logged object.
A log from `after()`, `before()` or `select()` answers like the log it came from. `cause()`
also raises `ValueError` when the log records no causes, or when `e` is not in the root log.

To count how often a handler was dispatched, count the distinct sources of its causes. One
dispatch can write several events, for example through `Scatter`:

```python
from langgraph_events import Cause

dispatched = len(
    {
        id(c.source)
        for e in log
        if isinstance(c := log.cause(e), Cause) and c.via == "notify_customer"
    }
)
```

A `SourceDropped` keeps `via`, so a count can include those dispatches as a separate number.

`log.effects(e)` lists the events that a handler produced from `e`. `log.flow(e)` gives the
chain of `Cause` that ends at `e`. The chain starts at the first event whose own cause is not a
`Cause`. The value that answers an `Interrupted`, and the `Resumed` that the framework creates,
have that `Interrupted` as their source.

A client that saves events in its own format saves each cause with them, and rebuilds the log
with `EventLog(events, causes=...)`. Each `Cause.source` must be the same object as an earlier
event: pass `events[j]`, not a copy. `log.causes` gives the entries back, so
`EventLog(log.events, causes=log.causes)` rebuilds a root log, unknown and framework cases
included. A recipe for JSON, where the client keeps its own list of events:

```python
import json

from langgraph_events import (
    Cause,
    EventLog,
    FrameworkEvent,
    NotRecorded,
    SourceDropped,
)


def dump_causes(log: EventLog) -> str:
    index = {id(e): i for i, e in enumerate(log.events)}

    def entry(cause):
        match cause:
            case Cause(source=source, via=via):
                return {"source": index[id(source)], "via": via}
            case SourceDropped(via=via, source_type=source_type):
                return {"dropped": source_type, "via": via}
            case NotRecorded():
                return {"not_recorded": True}
            case FrameworkEvent():
                return {"framework": True}
        return None

    return json.dumps([entry(c) for c in log.causes])


def load_log(events: list, saved: str) -> EventLog:
    def cause(entry):
        if entry is None:
            return None
        if "source" in entry:
            return Cause(events[entry["source"]], entry["via"])
        if "dropped" in entry:
            return SourceDropped(via=entry["via"], source_type=entry["dropped"])
        if "not_recorded" in entry:
            return NotRecorded()
        return FrameworkEvent()

    return EventLog(events, causes=[cause(e) for e in json.loads(saved)])
```

A save written before causes existed has no entries. Load each of its events as `NotRecorded()`,
never as `None`: `None` would claim that the event is a seed.

A direct `graph.compiled.update_state()` or `graph.compiled.invoke()` that writes `events` must
write one `causes` entry per event too. Otherwise every older cause shifts by one position, and no check can detect it.
`pre_seed()` writes the causes for you.

## `Namespace` as a feature hub

A `Namespace` is where related features attach:

- Declarative reducers as class attributes (auto-scoped to namespace events) — see [Reducers](reducers.md#on-a-namespace)
- Class-level `invariants` / `raises` on a `Command` (forwarded to inline `handle()`) — see [Control Flow](control-flow.md#invariants)
- Grouping in `graph.namespaces()`

!!! note "On `Namespace`"
    `Namespace` is a namespace — for grouping. A richer construct (with identity and size discipline) may layer on top in a future release.

### Namespace names are scoped to a graph { #namespace-scope }

A namespace name identifies one namespace **within a graph**, not within the process. Two namespaces of the same name reaching a single graph is an error — reducer discovery and `graph.namespaces()` both group by name, so the name has to resolve to one class:

```python
EventGraph([one.Place, two.Place])
# TypeError: Two different namespaces named 'Trading' reached this graph:
# app.a.Trading and app.b.Trading. Namespace names must be unique within a graph.
```

A namespace reached only through a handler's *return* type counts too, so a handler that subscribes to one lifetime and emits another's class is rejected the same way.

Across graphs the name is free. That is what lets one process run several independent engine lifetimes in sequence — a test that runs a scenario, ends it, and starts a fresh one against the same checkpointed log:

```python
import importlib
import app.trading

first = app.trading.Trading
saver.serde = NamespaceAwareSerde(namespaces=[first])
EventGraph([first.Place], checkpointer=saver).invoke(first.Place(sym="AAPL"), config=config)

importlib.reload(app.trading)          # lifetime 2
second = app.trading.Trading
saver.serde = NamespaceAwareSerde(namespaces=[second])
graph = EventGraph([second.Place], checkpointer=saver)

graph.get_state(config).events.latest(second.Place.Placed)   # revived as lifetime 2's class
```

!!! note "Give every lifetime its own serde"
    Checkpointed events are keyed by `(__module__, __qualname__)`, which two lifetimes of one module share. What keeps them apart is the serde: identity resolution is **scope-first**, consulting the namespaces the serde was constructed with before falling back to a module import. So each lifetime needs its own `NamespaceAwareSerde(namespaces=[...])`, as above — then lifetime 1's serde keeps reviving lifetime 1's classes even after lifetime 2 exists, and a namespace defined inside a function revives too, `<locals>` qualname and all ([#150](https://github.com/cadance-io/langgraph-events/issues/150)).

    Events *outside* any namespace — module-level `IntegrationEvent`s, framework `SystemEvent`s — reach the scope through `events=`. `EventGraph.from_namespaces` fills that in from the graph it builds, so the auto-wired path needs nothing; a hand-built serde should pass them explicitly.

## System events

`SystemEvent` subclasses control runtime flow; subscribe like any event. See [Control Flow](control-flow.md) for `Interrupted` / `Resumed`, `HandlerRaised`, `HandlerRetried`, `InvariantViolated`. Full table in [API](api.md#system-events).

To retire an `Interrupted` subclass, settle its paused threads first with `graph.abandon(config)` / `.aabandon()` — see [Ending a pause without answering it](control-flow.md#ending-a-pause-without-answering-it-abandon).

Custom halts subclass `Halted` and nest under their domain for locality; `graph.namespaces()` groups them with the domain's events rather than with framework system events:

```python
class Content(Namespace):
    class Classified(DomainEvent):
        label: str

    class Blocked(Halted):
        label: str

@on(Content.Classified)
def guard(event: Content.Classified) -> Reply | Content.Blocked:
    if event.label == "blocked":
        return Content.Blocked(label=event.label)
    return Reply(text="OK")
```

## Mixins

`Auditable` and `MessageEvent` are plain mixins (not `Event` subclasses). Compose with any event branch. Both subclass `EventMixin`. Subclass `EventMixin` for your own mixin: `@on(MyMixin)` then subscribes to every event that carries it.

**`Auditable`** — auto-logging marker. `@on(Auditable)` subscribes to all marked events:

```python
class OrderPlaced(DomainEvent, Auditable):
    order_id: str

@on(Auditable)
def audit(event: Auditable) -> None:
    print(event.trail())
```

**`MessageEvent`** — wraps LangChain `BaseMessage`; declare a `message` or `messages` field; pair with `message_reducer()`:

```python
class UserMessageReceived(IntegrationEvent, MessageEvent, Auditable):
    message: HumanMessage
```

**`SystemPromptSet`** — built-in `IntegrationEvent` + `MessageEvent`:

```python
log = graph.invoke([
    SystemPromptSet.from_str("You are helpful."),
    UserMessageReceived(message=HumanMessage(content="Hi")),
])
```
