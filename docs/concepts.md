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

        def ship(self) -> Shipped:
            return Order.Ship.Shipped(tracking=f"track-{self.order_id}")


graph = EventGraph([Order.Ship])
# or register every inline handler on a namespace in one call:
graph = EventGraph.from_namespaces(Order, handlers=[react])
```

- Exactly one public method per `Command`; helpers must be underscore-prefixed (else `TypeError` at class creation).
- Annotated return types must cover every nested `DomainEvent`.
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

        async def handle(self, chat_model: BaseChatModel) -> Refined: ...

# Name-keyed: handler params resolve by name. Multiple instances of same type allowed.
EventGraph(
    handlers=[...],
    services={"primary_chat": chat_a, "backup_chat": chat_b},
)

@on(SomeEvent)
def react(event, primary_chat, backup_chat) -> ...: ...
```

### Return contract

- Annotated handlers must return a type in the declared union (or `None`).
- Unannotated `Command`-subscribing handlers must return one of `Command.Outcomes` (or `None`); other unannotated handlers keep a shape-only check.
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
- `Edge` carries `kind` (how — `solid`/`scatter`/`raises`/`framework`) and `causation` (causal role — `intent`/`react`/`orchestrate`/`chain`). Surfaces in `text()`, `json()`, and mermaid styling.

Rendered diagrams live on the [Patterns](patterns.md) page — the collapsible legend shows the shape/edge vocabulary.

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
| `len(log)`, `log[i]` | container protocol |

## `Namespace` as a feature hub

A `Namespace` is where related features attach:

- Declarative reducers as class attributes (auto-scoped to namespace events) — see [Reducers](reducers.md#on-a-namespace)
- Class-level `invariants` / `raises` on a `Command` (forwarded to inline `handle()`) — see [Control Flow](control-flow.md#invariants)
- Grouping in `graph.namespaces()`

!!! note "On `Namespace`"
    `Namespace` is a namespace — for grouping. A richer construct (with identity and size discipline) may layer on top in a future release.

## Lifecycle archetypes { #archetypes }

When several domains share the *same* lifecycle (e.g. a persistence `Persist` → `Approve` loop across `Persona`, `Story`, `Scenario`) but each needs its **own** event identity and field types, define the behavior once in an **archetype** `Namespace` and **subclass** its commands per domain.

```python
class Persistable(Namespace):                       # archetype — behavior once
    class Policy(Protocol):                          # the bridge: per-entity ops
        def persist(self, candidate: object) -> str: ...

    class Persist(Command):
        candidate: object = None
        def handle(self):
            policy = namespace_of(self).Policy()     # resolve the consuming domain
            return type(self).Persisted(entity_id=policy.persist(self.candidate))

class Persona(Namespace, uses=[Persistable]):
    class Persist(Persistable.Persist):              # inherit behavior…
        candidate: PersonaCandidate = ...            # …override the field type
        class Persisted(DomainEvent):                # …and keep your own event
            entity_id: str = ""
    class Policy:                                    # implement the bridge
        def persist(self, candidate): return persona_store.create(candidate)
```

How it works:

- **Inherit, don't copy.** Subclassing an archetype command reuses its handler — the framework rebinds a *distinct* handler per subclass, so two domains' commands coexist as graph nodes (no `_inline_command` collision). The handler emits its outcome **reflectively** (`type(self).Persisted`), so each subclass yields its own per-entity event. *Archetype handlers must construct outcomes reflectively* — never hard-name the archetype's event class.
- **The Policy bridge.** Per-entity operations live behind a `Policy` the consuming namespace supplies; the archetype handler reaches it with [`namespace_of(self)`](api.md#archetypes) — `namespace_of(self).Policy`. The same bridge works in reactions: `namespace_of(event).Approve(...)`.
- **`uses=[…]`** records the composition on the namespace (`Persona.__uses__`) for intent and introspection. It does not copy or alias members — you reference `Persistable.Persist` / your own `Persona.Persist` directly.
- A subclass that declares its **own** handler overrides the inherited one.

Worked example: [`examples/persistable.py`](https://github.com/cadance-io/langgraph-events/blob/main/examples/persistable.py).

## System events

`SystemEvent` subclasses control runtime flow; subscribe like any event. See [Control Flow](control-flow.md) for `Interrupted` / `Resumed`, `HandlerRaised`, `InvariantViolated`. Full table in [API](api.md#system-events).

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

`Auditable` and `MessageEvent` are plain mixins (not `Event` subclasses). Compose with any event branch.

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
