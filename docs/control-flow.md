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
def handle_approval(event: ApprovalSubmitted, log: EventLog) -> OrderConfirmed | OrderCancelled:
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

See the [Human-in-the-Loop pattern](patterns.md#human-in-the-loop-approval) for a complete example, and [Checkpointer Evolution](checkpointer-evolution.md) for how graph changes affect interrupted checkpoints.
