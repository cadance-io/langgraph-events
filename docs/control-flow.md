# Control Flow

Fan-out, invariants, human-in-the-loop pauses, field matchers, and handler exceptions.

## `Scatter`

Return `Scatter([event1, event2, ...])` to fan out into multiple events. Each dispatches separately in the next round. Use `Scatter[WorkItem]` to annotate the produced type — renders as a dashed edge in `graph.domain().mermaid()`.

```python
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

`invariants={InvariantClass: predicate, ...}` on `@on()` blocks the handler before it runs if any predicate returns `False`. A blocked invariant emits `InvariantViolated` (a `SystemEvent`) carrying an `invariant` instance (of the declared class), the `handler` name, and the triggering `source_event`. (A blocked handler never runs, so no return-type contract applies to it.)

Invariants are **typed markers** — subclass `Invariant` once per rule, then reference the class on both the declaration side and the reactor side. Typos fail at graph-construction time, not silently at runtime.

Invariants are declared via the external `@on(...)` form — inline `handle` methods ([Concepts](concepts.md#on-decorator)) don't take modifiers.

```python
class CustomerNotBanned(Invariant):
    """Customer must not be on the banned list."""


@on(
    Order.Place,
    invariants={
        CustomerNotBanned: lambda log: not log.has(CustomerBanned),
    },
)
def place(event: Order.Place) -> Order.Place.Placed:
    return Order.Place.Placed(order_id=f"o-{event.customer_id}")


@on(InvariantViolated, invariant=CustomerNotBanned)
def reject(event: InvariantViolated) -> Order.Place.Rejected:
    return Order.Place.Rejected(reason=type(event.invariant).__name__)
```

Catch every violation with `@on(InvariantViolated)`, or pin a handler to a specific invariant class with the `invariant=` field matcher as shown. At graph construction time the framework verifies that every `invariant=` matcher references a class some handler actually declares — otherwise it raises `TypeError` ("would never fire").

Semantics:

- Predicates receive the current `EventLog` and must be **sync** (async is rejected at decoration).
- Multiple invariants short-circuit on the first failure; one `InvariantViolated` is emitted.
- Predicate exceptions propagate — they are not converted to violations.
- Invariants run before `raises=`; a blocked handler's declared exceptions are not triggered.
- `Invariant` subclasses should be zero-arg instantiable — the framework calls `Cls()` at emission time for `isinstance` matching. Nesting under an `Aggregate` / `Command` is encouraged (DDD-idiomatic) but not enforced.

### Modeling errors — when to use what

| Situation | Vehicle |
|---|---|
| Expected domain outcome (including failure) | `DomainEvent` (`Order.Place.Rejected`) |
| Rule that must hold before a handler runs | `invariants=` → `InvariantViolated` |
| Infrastructure failure (rate limit, timeout, parse error) | `Exception` + `raises=` → `HandlerRaised` |

## `Interrupted` / `Resumed`

`Interrupted` is a bare marker — subclass with typed fields to pause for human input. Resume with `graph.resume(event)`; the event dispatches and a `Resumed` event is emitted alongside. Requires a **checkpointer** (`MemorySaver`, etc.).

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

See the [HITL pattern](patterns.md#expense-hitl) and [Checkpointer Evolution](checkpointer-evolution.md).

## Field Matchers

Narrow dispatch by requiring a field to be a specific type. The handler only fires when the named field is an `isinstance` match; if the handler signature includes a matching parameter, the value is injected:

```python
@on(Resumed, interrupted=OrderConfirmationRequested)
def handle(event: Resumed, interrupted: OrderConfirmationRequested) -> OrderConfirmed:
    # `interrupted` is guaranteed to be OrderConfirmationRequested.
    ...
```

Works on any field typed as `Event` or `Exception`. Field names are validated at graph construction — typos raise `TypeError`. Omitting the parameter still filters dispatch without injection.

## Handler Exceptions

`raises=` declares exceptions the framework catches from a handler; caught exceptions surface as `HandlerRaised` events with the exception, the handler name, and the originating event. Subscribe with a field matcher on `exception` to react:

```python
class RateLimitError(Exception):
    def __init__(self, retry_after: float) -> None:
        super().__init__(f"retry after {retry_after}s")
        self.retry_after = retry_after


@on(Question.Ask, raises=RateLimitError)
def call_llm(event: Question.Ask) -> Question.Ask.Answered:
    if upstream_rate_limited():
        raise RateLimitError(retry_after=0.2)
    return Question.Ask.Answered(answer=...)


@on(HandlerRaised, exception=RateLimitError)
def backoff(event: HandlerRaised, exception: RateLimitError) -> Question.RetryScheduled:
    return Question.RetryScheduled(question=event.source_event.question)
```

Key rules:

- Every type in `raises=` must be covered by at least one catcher, or graph construction fails with `TypeError`. A catcher covers `X` if it has no field matchers, or only `exception=X`-or-superclass. Non-`exception` matchers don't count.
- Only `Exception` subclasses are allowed — `BaseException` / `KeyboardInterrupt` / `SystemExit` / `GeneratorExit` / `asyncio.CancelledError` are rejected. `CancelledError` surfaces as `Cancelled` (a `Halted` subtype).
- Unhandled raises propagate and crash the run. Catchers can themselves declare `raises=` to escalate.
- `HandlerRaised.source_event` (not `event`) holds the triggering event — avoids kwarg collision.

See the [Error Recovery pattern](patterns.md#error-recovery) for retry-and-escalate.
