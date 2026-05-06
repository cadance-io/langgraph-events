# Core Concepts

## State IS events

A run is an append-only log of frozen, typed events. Handlers read events in, emit events out. Projections (`EventLog.filter()`, reducers) derive whatever view you need — there is no mutable shared state.

## The taxonomy

Four event base classes plus a `Namespace` namespace. Pick the right one and the naming discipline follows:

| Class | Role | Naming | Where it lives |
|---|---|---|---|
| `Namespace` | Namespace for related commands and events | Noun (`Order`) | Top-level class |
| `Command` | Intent / request | **Imperative** (`Place`, `Ship`) | Nested inside a `Namespace` |
| `DomainEvent` | Fact inside the domain | Past-participle (`Placed`, `Shipped`) | Nested under a `Namespace` or `Command` |
| `IntegrationEvent` | Fact crossing a system boundary | Past-participle | Top-level |
| `SystemEvent` | Framework-emitted fact | Past-participle | Top-level (`Halted`, `HandlerRaised`, ...) |
| `Invariant` | Named rule gating a handler | Noun phrase (`CustomerNotBanned`) | Anywhere — nesting under a `Command` is encouraged |

`Auditable` and `MessageEvent` are behavioural **mixins** — compose them with any event branch (e.g. `class Foo(DomainEvent, Auditable)`). `Invariant` is a **marker class**, not an `Event` subclass — see [control-flow](control-flow.md#invariants). Declared invariants surface as first-class nodes in `graph.namespaces()`: `.invariants` lists every subclass with its owning commands, declaring handlers, and pinned reactors; mermaid diagrams render each as a diamond gate inside its owning domain.

!!! note "On `Namespace`"

    `Namespace` is a namespace — for grouping related commands and events, plus the target for declarative reducers and the `invariants=` kwarg. A richer construct — with identity and size discipline — may layer on top in a future release.

Nesting is enforced at class-creation time — `Command` / `DomainEvent` defined outside a `Namespace` raise `TypeError`. Direct `Event` subclassing also raises `TypeError` — use one of the four bases above.

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

Nested classes inherit nothing implicit. `Order.Place.Placed` is *not* a subclass of `Order.Place` — it's a `DomainEvent` scoped to the `Order` domain with a `__command__` back-reference to `Place`.

### `Command.Outcomes`

Auto-generated union of the command's nested `DomainEvent` classes:

```python
isinstance(evt, Order.Place.Outcomes)   # matches Placed OR Rejected
typing.get_args(Order.Place.Outcomes)   # (Placed, Rejected)
```

Declare `Outcomes` yourself if you want `mypy` to see it — the framework drift-checks against the nested events at class-creation time:

```python
from typing import TypeAlias

class Order(Namespace):
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
class Order(Namespace):
    class Ship(Command):
        order_id: str

        class Shipped(DomainEvent):
            tracking: str

        def handle(self) -> Shipped:
            return Order.Ship.Shipped(tracking=f"track-{self.order_id}")


graph = EventGraph([Order.Ship])
# or register every inline handler on a domain in one call:
graph = EventGraph.from_namespaces(Order, handlers=[react])
```

When an inline `handle` has an explicit return annotation, it must cover every nested `DomainEvent`. A `DomainEvent` nested inside a `Command` is *Command-private*: only that Command's `handle()` may emit it. Reactors that need to surface a domain failure as part of recovery emit a namespace-level sibling event (e.g. `Order.Rejected`), not a Command-private outcome — graph construction raises `CommandPrivacyError` if a non-`handle()` handler returns one.

`invariants` and `raises` for an inline handle are declared as class-level attributes on the `Command`:

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

Three shapes. Pick the shortest that conveys intent:

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

The bare form errors at decoration if the first parameter is missing, unannotated, or not a single `Event` subclass.

### Signature injection

Handlers receive injections by type or name:

- `log: EventLog` — full history
- `config: RunnableConfig` / `store: BaseStore` — LangGraph injections
- Reducer channel by **parameter name** (see [Reducers](reducers.md))
- Field matchers (external only) — typed subset dispatch plus injection
- **Services** — project dependencies registered on `EventGraph(services=...)`

Resolution order: reducer name → framework type (`EventLog` / `RunnableConfig` / `BaseStore`) → service. The first match wins.

`services=` accepts two shapes (mutually exclusive within a graph):

```python
# Type-keyed: handler params resolve by their annotation. Same-type
# collisions are rejected at build; subclass annotations match registered
# subclass instances via an MRO walk.
EventGraph(handlers=[...], services=[chat_model, session_factory])

class Story(Namespace):
    class Refine(Command):
        async def handle(self, chat_model: BaseChatModel) -> Refined:
            ...

# Name-keyed: handler params resolve by name. Allows multiple instances
# of the same type (primary + backup chat models, etc.).
EventGraph(
    handlers=[...],
    services={"primary_chat": chat_a, "backup_chat": chat_b},
)

@on(SomeEvent)
def react(event, primary_chat, backup_chat) -> ...: ...
```

Handler params with no injection source raise `TypeError` at graph construction (not at first dispatch).

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

### Namespace introspection & visualization

One entry point — `graph.namespaces()` — returns a `NamespaceModel`: a code-derived snapshot of the structure *and* the event-driven flow (choreography). Render it to text, Mermaid, or JSON:

```python
d = graph.namespaces()

print(d.text())                     # human-readable tree (choreography)
print(d.text(view="structure"))     # taxonomy only — no handlers

print(d.mermaid())                  # graph LR flowchart (choreography)

d.json()                            # JSON snapshot (event classes as qualnames)

# Data access — everything is a frozen dataclass tuple/dict:
d.namespaces                 # dict[str, NamespaceModel.Namespace]
d.command_handlers        # tuple[NamespaceModel.CommandHandler, ...]
d.policies                # tuple[NamespaceModel.Policy, ...]
d.edges                   # tuple[NamespaceModel.Edge, ...]  — source, via, target, kind
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

## Namespace as a feature hub

A `Namespace` groups related commands and events, and is where related features attach: declarative reducers as class attributes (auto-scoped to the namespace's events), `invariants` / `raises` declared as class-level attributes on a `Command` (forwarded to its inline `handle()`), and namespace grouping in `graph.namespaces()`. See [Reducers](reducers.md#on-a-namespace) and [Control Flow](control-flow.md#invariants).

## System events

Framework-emitted events for runtime control — all `SystemEvent` subclasses. Subscribe via `@on(SomeSystemEvent)` like any other event. See [Control Flow](control-flow.md) for `Interrupted` / `Resumed` (HITL), `HandlerRaised` (`raises=`), and `InvariantViolated`. Full table in [API](api.md#system-events).

Custom halts subclass `Halted` with domain-specific fields. Nest them inside their domain for locality; `graph.namespaces()` groups them with the rest of the domain's events rather than with framework system events:

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
