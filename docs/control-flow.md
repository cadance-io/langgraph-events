# Control Flow

Fan-out, human-in-the-loop pauses, and other advanced dispatch patterns. Reach for these when basic event → handler → event chains aren't enough.

For core graph execution and immediate halting, see [Core Concepts](concepts.md#halted).

## `Scatter`

Return `Scatter([event1, event2, ...])` to fan-out into multiple events. Each becomes a separate pending event, dispatched in the next round. Use `Scatter[WorkItem]` to annotate the produced type — this renders as a dashed edge in `mermaid()` diagrams.

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
    return None  # not all items done yet
```

See the [Map-Reduce pattern](patterns.md#fan-out-fan-in-map-reduce) for a complete runnable example.

## `Interrupted` / `Resumed`

`Interrupted` is a bare marker class — subclass it with domain-specific fields to pause the graph and wait for human input. Resume with `graph.resume(event)` — the event is auto-dispatched (handlers subscribed to its type fire), then the framework creates a `Resumed` event alongside it. `resume()` requires an `Event` instance; passing a plain string or dict raises `TypeError`.

Requires a **checkpointer** (e.g., `MemorySaver`).

```python
from langgraph.checkpoint.memory import MemorySaver


class OrderConfirmationRequested(Interrupted):
    order_id: str
    total: float


class ApprovalSubmitted(Event):
    approved: bool


@on(OrderPlaced)
def confirm(event: OrderPlaced) -> OrderConfirmationRequested:
    return OrderConfirmationRequested(order_id=event.order_id, total=event.total)


@on(ApprovalSubmitted)
def handle_approval(
    event: ApprovalSubmitted, log: EventLog,
) -> OrderConfirmed | OrderCancelled:
    confirm_event = log.latest(OrderConfirmationRequested)
    if event.approved:
        return OrderConfirmed(order_id=confirm_event.order_id)
    return OrderCancelled(reason="User declined")


graph = EventGraph([confirm, handle_approval], checkpointer=MemorySaver())
config = {"configurable": {"thread_id": "order-1"}}

# First call — pauses at the interrupt
graph.invoke(OrderPlaced(order_id="A1", total=99.99), config=config)

# Check state and resume with a typed event
state = graph.get_state(config)
if state.is_interrupted:
    confirm_event = state.interrupted
    print(f"Approve order {confirm_event.order_id} for ${confirm_event.total}?")
log = graph.resume(ApprovalSubmitted(approved=True), config=config)
```

### Field Matchers — Narrow Dispatch by Field Type

**Field matchers** narrow dispatch by requiring a field on the event to be a specific type. Pass `field_name=EventType` as a keyword argument to `@on()` — the handler only fires when that field is an instance of the given type. If the handler signature includes a parameter with the same name, the matched value is injected automatically:

```python
@on(Resumed, interrupted=OrderConfirmationRequested)
def handle_order_confirmation(
    event: Resumed, interrupted: OrderConfirmationRequested,
) -> OrderConfirmed | OrderCancelled:
    # `interrupted` is guaranteed to be OrderConfirmationRequested —
    # the handler only fires when the field matches.
    print(f"Order {interrupted.order_id}: ${interrupted.total}")
    ...
```

Field matchers work on any event field typed as `Event`, not just `interrupted`. If the named field is `None` or doesn't match the given type, the handler is silently skipped. The field name is validated at graph construction — typos raise `TypeError` immediately.

If the handler signature omits the field parameter, the matcher still filters dispatch but no injection occurs:

```python
@on(Resumed, interrupted=OrderConfirmationRequested)
def handle_order_confirmation(event: Resumed) -> OrderConfirmed:
    # Still only fires for OrderConfirmationRequested interrupts,
    # but you'd access event.interrupted directly.
    ...
```

See the [Human-in-the-Loop pattern](patterns.md#human-in-the-loop-approval) for a complete example, and [Checkpointer Evolution](checkpointer-evolution.md) for how graph changes affect interrupted checkpoints.

## Handler Exceptions

Declare exceptions the framework should catch from a handler with `raises=`. Caught exceptions surface as a built-in `HandlerRaised` event; subscribe with a field matcher on `exception` to react — retry, back off, escalate, or halt — without try/except boilerplate at the raise site.

```python
class RateLimitError(Exception):
    def __init__(self, retry_after: float) -> None:
        super().__init__(f"retry after {retry_after}s")
        self.retry_after = retry_after


@on(QuestionAsked, raises=RateLimitError)
def call_llm(event: QuestionAsked) -> AnswerReceived:
    if upstream_rate_limited():
        raise RateLimitError(retry_after=0.2)
    return AnswerReceived(answer=...)


@on(HandlerRaised, exception=RateLimitError)
def backoff_and_retry(
    event: HandlerRaised, exception: RateLimitError,
) -> RetryScheduled:
    # `exception` is injected and typed via field matcher
    return RetryScheduled(question=event.event.question)
```

Rules:

- `raises=` accepts a single class or tuple. Entries must be `Exception` subclasses — `BaseException`, `KeyboardInterrupt`, `SystemExit`, `GeneratorExit`, and `asyncio.CancelledError` are rejected at decoration time.
- Unhandled raises (exception types *not* in `raises=`) still propagate and crash the run. The mechanism is opt-in per-handler — no ambient catch-all.
- **Compile-time check:** every type in `raises=` must be covered by at least one catcher. A catcher covers `X` if it's subscribed to `HandlerRaised` with no `exception=` matcher (catches any) *or* with `exception=X` or a superclass. Missing coverage raises `TypeError` at `EventGraph(...)` construction with a message pointing at the uncovered class.
- Catchers can themselves declare `raises=` to **escalate** — e.g., `backoff_and_retry` above can raise `QuotaExhaustedError` when the retry budget is spent, surfaced as another `HandlerRaised` for a dedicated handler.
- `asyncio.CancelledError` is still surfaced as a `Cancelled` (a `Halted` subtype), not `HandlerRaised` — cancellation is a framework concern, not a domain error.
- The original event being processed is preserved as `HandlerRaised.event`, so catchers can inspect what triggered the failure.

See the [Error Recovery pattern](patterns.md#error-recovery) for a complete runnable example with retry and escalation.
