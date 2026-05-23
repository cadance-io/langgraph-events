# Control Flow

## `Scatter`

Return `Scatter([event1, event2, ...])` to fan out into multiple events; each dispatches separately in the next round. Annotate the return as `Scatter[WorkItem]` (or `Scatter[A | B]` for heterogeneous targets) — renders as a dashed edge in `graph.namespaces().mermaid()`. Bare `Scatter`, `Scatter[Any]`, `Scatter[Event]`, and `Scatter[T]` (TypeVar) are rejected at graph construction with `TypeError` (v0.10).

```python
from langgraph_events import EventLog, IntegrationEvent, Scatter, on


class Batch(IntegrationEvent):
    items: tuple[str, ...]


class WorkItem(IntegrationEvent):
    item: str


class WorkDone(IntegrationEvent):
    result: str


class BatchResult(IntegrationEvent):
    results: tuple[str, ...]


@on(Batch)
def split(event: Batch) -> Scatter[WorkItem]:
    return Scatter([WorkItem(item=i) for i in event.items])


@on(WorkItem)
def process(event: WorkItem) -> WorkDone:
    return WorkDone(result=f"done:{event.item}")


@on(WorkDone)
def gather(event: WorkDone, log: EventLog) -> BatchResult | None:
    all_done = log.filter(WorkDone)
    batch = log.latest(Batch)
    if len(all_done) >= len(batch.items):
        return BatchResult(results=tuple(e.result for e in all_done))
    return None
```

See the [Map-Reduce pattern](patterns.md#scatter-fan-out).

## Invariants

Declare `invariants={InvariantClass: predicate, ...}` as a class-level attribute on a `Command` (or via `invariants=` on `@on()` for external handlers). Each predicate runs twice per matching event:

| Phase | Timing | Log scope | On failure |
|---|---|---|---|
| **Pre-check** | Before handler body | Committed events | Skip handler; emit `InvariantViolated` (`would_emit` empty) |
| **Post-check** | After handler returns | Committed + emitted | Drop emissions; emit `InvariantViolated` (`would_emit` = handler's events) |

!!! note "Atomicity"
    The two phases give the DDD semantic: the domain's consistency rules hold **before** the handler runs and **after** its effects commit.

- Typed markers — subclass `Invariant` once per rule; reference the class on declaration and reactor sides. Typos fail at graph construction.
- Declared on the `Command`'s `invariants` dict; forwarded to the inline `handle()` wrapper.
- Recovery reactors emit **namespace-level** events (Command-private outcomes are reserved for `handle()`).

```python
class CustomerNotBanned(Invariant):
    """Customer must not be on the banned list."""  # pre-check catches


class OrderTotalWithinLimit(Invariant):
    """Cumulative placed amount must stay under a daily limit."""  # post-check catches


class Order(Namespace):
    class Place(Command):
        customer_id: str = ""
        amount: int = 0

        invariants = {
            CustomerNotBanned: lambda log: not log.has(CustomerBanned),
            OrderTotalWithinLimit: lambda log: (
                sum(e.amount for e in log.filter(Order.Place.Placed)) < 100
            ),
        }

        class Placed(DomainEvent):
            order_id: str = ""
            amount: int = 0

        def handle(self) -> Order.Place.Placed:
            return Order.Place.Placed(
                order_id=f"o-{self.customer_id}", amount=self.amount
            )

    class Rejected(DomainEvent):
        """Sibling to ``Place`` — emitted by recovery reactors."""

        reason: str = ""


@on(InvariantViolated, invariant=OrderTotalWithinLimit)
def rolled_back(event: InvariantViolated) -> Order.Rejected:
    rolled = event.would_emit[0]  # the Placed the handler would have emitted
    return Order.Rejected(reason=f"over limit (would emit {rolled.amount})")
```

`CustomerNotBanned`: pre-check only — truth ≠ handler outputs. `OrderTotalWithinLimit`: post-check catches the case where this handler's `Placed` is what pushes the total over.

!!! warning "Invariant reactors"
    - Catch all: `@on(InvariantViolated)`. Pin to a class: `@on(InvariantViolated, invariant=SomeInvariant)`.
    - Framework verifies pinned classes are actually declared somewhere — otherwise `TypeError` at graph construction ("would never fire").
    - Pinned reactors fire for both pre- and post-check failures without distinguishing them — inspect `event.would_emit` to tell them apart.

!!! note "Semantics"
    - Predicates receive `EventLog`; must be **sync** (async rejected at decoration) and **pure functions of `log`**.
    - Pre-check log = committed events. Post-check log = committed + everything the current node call has buffered.
    - Multiple invariants short-circuit; one `InvariantViolated` per phase.
    - Predicate exceptions propagate (not converted to violations).
    - Invariants run around `raises=`: pre-check gates the body entirely; post-check skips on caught exceptions (those emit `HandlerRaised` instead).
    - Post-check is a no-op when the handler returned `None` (empty buffer) or declares no invariants.
    - `Invariant` subclasses must be zero-arg instantiable (framework calls `Cls()` at emission time for `isinstance` matching).

### Modeling errors — when to use what

| Situation | Vehicle |
|---|---|
| Expected domain outcome (including failure) | `DomainEvent` (`Order.Place.Rejected`) |
| Consistency rule gating a command (pre and/or post) | `invariants=` → `InvariantViolated` |
| Infrastructure failure (rate limit, timeout, parse error) | `Exception` + `raises=` → `HandlerRaised` |

## Warnings at construction

`graph.namespaces()` emits two warnings that flag structural smells. Filter with `warnings.filterwarnings("ignore", category=...)` if intentional.

- **`CommandChainWarning`** — `chain`-causation edge: an inline `Command.handle()` returning another `Command`. Prefer emitting a fact and reacting to it.
- **`DomainPatternWarning`** — 2+ events in the same namespace fan out to identical target sets (a missing shared base / common reactor).

!!! note "`IntegrationEvent` placement"
    `IntegrationEvent` must be defined at module scope. Nesting inside a `Namespace` or `Command` raises `TypeError` at class creation (v0.10) — integration events cross a context boundary by definition.

## `Interrupted` / `Resumed`

Subclass `Interrupted` with typed fields to pause for human input. Resume with `graph.resume(event)` (requires a checkpointer); a `Resumed` event emits alongside the dispatched event.

```python
from langgraph.checkpoint.memory import MemorySaver


class OrderConfirmationRequested(Interrupted):
    order_id: str
    total: float


class ApprovalSubmitted(IntegrationEvent):
    approved: bool


@on(OrderPlaced)
def confirm(event: OrderPlaced) -> OrderConfirmationRequested:
    return OrderConfirmationRequested(order_id=event.order_id, total=event.total)


@on(ApprovalSubmitted)
def handle_approval(event: ApprovalSubmitted, log: EventLog) -> OrderConfirmed | OrderCancelled:
    request = log.latest(OrderConfirmationRequested)
    if event.approved:
        return OrderConfirmed(order_id=request.order_id)
    return OrderCancelled(reason="User declined")


graph = EventGraph([confirm, handle_approval], checkpointer=MemorySaver())
config = {"configurable": {"thread_id": "order-1"}}

graph.invoke(OrderPlaced(order_id="A1", total=99.99), config=config)

state = graph.get_state(config)
if state.is_interrupted:
    print(f"Approve {state.interrupted.order_id} for ${state.interrupted.total}?")
log = graph.resume(ApprovalSubmitted(approved=True), config=config)
```

See [HITL pattern](patterns.md#expense-hitl) and [Checkpointer Evolution](checkpointer-evolution.md).

### Typed payloads — `InterruptedWithPayload`

For interrupts whose frontend needs an action-discriminated dict, subclass `langgraph_events.agui.InterruptedWithPayload[PayloadT]` and implement `interrupt_payload(self) -> PayloadT`. The AG-UI adapter recognises the contract directly — see [AG-UI](agui.md).

## Field Matchers

`@on(Event, field=Type)` dispatches only when `event.field` is a `Type` instance; if the handler signature includes a parameter named `field`, the value is injected:

```python
@on(Resumed, interrupted=OrderConfirmationRequested)
def handle(event: Resumed, interrupted: OrderConfirmationRequested) -> OrderConfirmed:
    # `interrupted` is guaranteed to be OrderConfirmationRequested.
    ...
```

- Works on any field typed as `Event` or `Exception`.
- Field names validated at graph construction (typos raise `TypeError`).
- Omit the parameter to filter dispatch without injection.

## Handler Exceptions

Declare `raises=(ExceptionClass, ...)` on a `Command` (or via `raises=` on `@on(...)`). Caught exceptions emit `HandlerRaised` carrying the exception, handler name, and `source_event`; subscribe with `@on(HandlerRaised, exception=…)` to react.

```python
class RateLimitError(Exception):
    def __init__(self, retry_after: float) -> None:
        super().__init__(f"retry after {retry_after}s")
        self.retry_after = retry_after


class Question(Namespace):
    class Ask(Command):
        question: str = ""
        raises = (RateLimitError,)

        class Answered(DomainEvent):
            answer: str = ""

        def handle(self) -> Question.Ask.Answered:
            if upstream_rate_limited():
                raise RateLimitError(retry_after=0.2)
            return Question.Ask.Answered(answer=...)


@on(HandlerRaised, exception=RateLimitError)
def backoff(event: HandlerRaised, exception: RateLimitError) -> Question.Ask:
    return Question.Ask(question=event.source_event.question)  # retry
```

- Every type in `raises=` must be covered by at least one catcher, else graph construction fails with `TypeError`. A catcher covers `X` if it has no field matchers, or only `exception=X`-or-superclass. Non-`exception` matchers don't count.
- Only `Exception` subclasses allowed — `BaseException`/`KeyboardInterrupt`/`SystemExit`/`GeneratorExit`/`asyncio.CancelledError` rejected. `CancelledError` surfaces as `Cancelled` (a `Halted` subtype).
- Unhandled raises propagate and crash the run. Catchers can themselves declare `raises=` to escalate.
- Use `HandlerRaised.source_event` (not `event`) for the triggering event — avoids kwarg collision.

See the [Error Recovery pattern](patterns.md#error-recovery).
