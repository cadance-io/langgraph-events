# Core Concepts

## State IS events

A run is an append-only log of frozen, typed events. Handlers read events in, emit events out. Projections (`EventLog.filter()`, reducers) derive whatever view you need — there is no mutable shared state.

## The taxonomy

Four event base classes plus an aggregate namespace. Pick the right one and the naming discipline follows:

| Class | Role | Naming | Where it lives |
|---|---|---|---|
| `Aggregate` | Namespace / consistency boundary | Noun (`Order`) | Top-level class |
| `Command` | Intent / request | **Imperative** (`Place`, `Ship`) | Nested inside an `Aggregate` |
| `DomainEvent` | Fact inside the domain | Past-participle (`Placed`, `Shipped`) | Nested under an `Aggregate` or `Command` |
| `IntegrationEvent` | Fact crossing a system boundary | Past-participle | Top-level |
| `SystemEvent` | Framework-emitted fact | Past-participle | Top-level (`Halted`, `HandlerRaised`, ...) |
| `Invariant` | Named rule gating a handler | Noun phrase (`CustomerNotBanned`) | Anywhere — nesting under a `Command` is encouraged |

`Auditable` and `MessageEvent` are behavioural **mixins** — compose them with any event branch (e.g. `class Foo(DomainEvent, Auditable)`). `Invariant` is a **marker class**, not an `Event` subclass — see [control-flow](control-flow.md#invariants). Declared invariants surface as first-class nodes in `graph.domain()`: `.invariants` lists every subclass with its owning commands, declaring handlers, and pinned reactors; mermaid diagrams render each as a diamond gate inside its aggregate.

Nesting is enforced at class-creation time — `Command` / `DomainEvent` defined outside an `Aggregate` raise `TypeError`. Direct `Event` subclassing also raises `TypeError` — use one of the four bases above.

```python
class Order(Aggregate):
    class Place(Command):
        customer_id: str

        class Placed(DomainEvent):
            order_id: str

        class Rejected(DomainEvent):
            reason: str

    class Shipped(DomainEvent):
        tracking: str
```

Nested classes inherit nothing implicit. `Order.Place.Placed` is *not* a subclass of `Order.Place` — it's a `DomainEvent` scoped to the `Order` aggregate with a `__command__` back-reference to `Place`.

### `Command.Outcomes`

Auto-generated union of the command's nested `DomainEvent` classes:

```python
isinstance(evt, Order.Place.Outcomes)   # matches Placed OR Rejected
typing.get_args(Order.Place.Outcomes)   # (Placed, Rejected)
```

Declare `Outcomes` yourself if you want `mypy` to see it — the framework drift-checks against the nested events at class-creation time:

```python
from typing import TypeAlias

class Order(Aggregate):
    class Place(Command):
        class Placed(DomainEvent): ...
        class Rejected(DomainEvent): ...
        Outcomes: TypeAlias = Placed | Rejected   # optional; drift-checked
```

## Handlers { #on-decorator }

Two styles: inline `handle` on a command, or the `@on` decorator.

### Inline: `handle` on the command { #inline-command-handlers }

The command owns its handler. `self` is the command instance. Pass the command class to `EventGraph` — no decorator.

```python
class Order(Aggregate):
    class Ship(Command):
        order_id: str

        class Shipped(DomainEvent):
            tracking: str

        def handle(self) -> Shipped:
            return Order.Ship.Shipped(tracking=f"track-{self.order_id}")


graph = EventGraph([Order.Ship])
# or register every inline handler on an aggregate in one call:
graph = EventGraph.from_aggregates(Order, handlers=[react])
```

When an inline `handle` has an explicit return annotation, it must cover every nested `DomainEvent` — dropping an outcome there is almost always a mistake. External `@on(Cmd)` handlers are exempt (distributed outcome production is a valid pattern).

### External: `@on`

Three shapes. Pick the shortest that conveys intent:

```python
# Bare — event type inferred from the annotation
@on
def explain(event: InvariantViolated) -> Order.Place.Rejected:
    return Order.Place.Rejected(reason=type(event.invariant).__name__)

# Modifiers only — event type inferred, modifiers applied
@on(invariants={CustomerNotBanned: lambda log: not log.has(CustomerBanned)})
def place(event: Order.Place) -> Order.Place.Placed:
    return Order.Place.Placed(order_id=f"o-{event.customer_id}")

# Explicit types — required for multi-event subscription
@on(UserMessage, ToolResults)
async def call_llm(event: Event) -> AssistantMessage: ...
```

The bare form errors at decoration if the first parameter is missing, unannotated, or not a single `Event` subclass.

### Signature injection

Handlers receive injections by type or name:

- `log: EventLog` — full history
- `config: RunnableConfig` / `store: BaseStore` — LangGraph injections
- Reducer channel by **parameter name** (see [Reducers](reducers.md))
- Field matchers (external only) — typed subset dispatch plus injection

Async is supported on both forms.

### Return contract

- Annotated handlers must return a type in the declared union (or `None`).
- Unannotated `Command`-subscribing handlers must return one of `Command.Outcomes` (or `None`); other unannotated handlers keep a shape-only check.

Violations raise `TypeError` at dispatch.

## `EventGraph`

```python
graph = EventGraph([place, respond], max_rounds=100)
```

Topology is derived from handler subscriptions — no manual node/edge wiring. `max_rounds` (default 100) auto-sets LangGraph's recursion limit and emits `MaxRoundsExceeded` (a `Halted` subtype) when exceeded.

### Domain introspection & visualization

One entry point — `graph.domain()` — returns a `DomainModel`: a code-derived snapshot of the DDD structure *and* the event-driven flow (choreography). Render it to text, Mermaid, or JSON:

```python
d = graph.domain()

print(d.text())                     # human-readable tree (choreography)
print(d.text(view="structure"))     # taxonomy only — no handlers

print(d.mermaid())                  # graph LR flowchart (choreography)
print(d.mermaid(view="structure"))  # classDiagram of the taxonomy

d.json()                            # JSON snapshot (event classes as qualnames)

# Data access — everything is a frozen dataclass tuple/dict:
d.aggregates              # dict[str, DomainModel.Aggregate]
d.command_handlers        # tuple[DomainModel.CommandHandler, ...]
d.policies                # tuple[DomainModel.Policy, ...]
d.edges                   # tuple[DomainModel.Edge, ...]  — source, via, target, kind
d.seeds                   # tuple[type[Event], ...]       — events with no incoming edges
d.integration_events      # tuple[type[IntegrationEvent], ...]
d.system_events           # tuple[type[SystemEvent], ...]
```

Rendered diagrams live on the [Patterns](patterns.md) page — the collapsible legend at the top shows the shape/edge vocabulary used across every example.

### Escape hatch

`graph.compiled` exposes the underlying `CompiledStateGraph` for subgraph composition, custom streaming modes, or direct state access.

## `EventLog`

Immutable, ordered container returned by `invoke` / `ainvoke`. Handlers receive it by type hint.

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
| `len(log)`, `log[i]` | container protocol |

## Aggregates as a feature hub

An `Aggregate` is a namespace, but also where DDD-aligned features attach: declarative reducers as class attributes (auto-scoped to the aggregate's events), `invariants=` on `@on(Aggregate.Cmd, ...)`, and domain-model grouping in `graph.domain()`. See [Reducers](reducers.md#on-an-aggregate) and [Control Flow](control-flow.md#invariants).

## System events

Framework-emitted events for runtime control — all `SystemEvent` subclasses. Subscribe via `@on(SomeSystemEvent)` like any other event. See [Control Flow](control-flow.md) for `Interrupted` / `Resumed` (HITL), `HandlerRaised` (`raises=`), and `InvariantViolated`. Full table in [API](api.md#system-events).

Custom halts are normal DDD — subclass `Halted` with domain-specific fields. Nest them inside their aggregate for locality; `graph.domain()` groups them with the rest of the aggregate's events rather than with framework system events:

```python
class Content(Aggregate):
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

`Auditable` and `MessageEvent` are plain mixins — not `Event` subclasses. Compose them with any event branch.

**`Auditable`** — marker for auto-logging. `@on(Auditable)` captures every marked event:

```python
class OrderPlaced(DomainEvent, Auditable):
    order_id: str

@on(Auditable)
def audit(event: Auditable) -> None:
    print(event.trail())
```

**`MessageEvent`** — wraps LangChain `BaseMessage` objects. Declare a `message` or `messages` field; pair with `message_reducer()`:

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
