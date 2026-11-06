# Control Flow

## `Scatter`

Return `Scatter([event1, event2, ...])` to fan out into multiple events. Each dispatches separately in the next round. Annotate the return as `Scatter[WorkItem]`, or `Scatter[A | B]` for heterogeneous targets. It renders as a dashed edge in `graph.namespaces().mermaid()`. Bare `Scatter`, `Scatter[Any]`, `Scatter[Event]` and `Scatter[T]` (TypeVar) are rejected at graph construction with `TypeError`.

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

Declare `invariants={InvariantClass: predicate, ...}` as a class-level attribute on a `Command`, or via `invariants=` on `@on()` for an external handler. Each predicate runs twice per matching event:

| Phase | Timing | Log scope | On failure |
|---|---|---|---|
| **Pre-check** | Before handler body | Committed events | Skip handler, emit `InvariantViolated` (`would_emit` empty) |
| **Post-check** | After handler returns | Committed + buffered by the current node call | Drop emissions, emit `InvariantViolated` (`would_emit` = handler's events) |

```python
from __future__ import annotations

from langgraph_events import Command, DomainEvent, IntegrationEvent, Invariant
from langgraph_events import InvariantViolated, Namespace, on


class CustomerBanned(IntegrationEvent):
    customer_id: str = ""


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
        """Sibling to ``Place``, emitted by a recovery handler."""

        reason: str = ""


@on(InvariantViolated, invariant=OrderTotalWithinLimit)
def rolled_back(event: InvariantViolated) -> Order.Rejected:
    rolled = event.would_emit[0]  # the Placed the handler would have emitted
    return Order.Rejected(reason=f"over limit (would emit {rolled.amount})")
```

- Subclass `Invariant` once per rule. Reference the class on the declaration side and on the handler side. `Invariant` subclasses must be zero-arg instantiable. The library calls `Cls()` at emission time for `isinstance` matching.
- Catch all: `@on(InvariantViolated)`. Pin to a class: `@on(InvariantViolated, invariant=SomeInvariant)`. A pinned class that no declaration names raises `TypeError` at graph construction ("would never fire"). A pinned handler fires for both pre-check and post-check failures. Inspect `event.would_emit` to tell them apart.
- A recovery handler emits a **namespace-level** event. Command-private outcomes are reserved for the inline command handler.
- A predicate receives `EventLog`. It must be **sync** and a **pure function of `log`**. An async predicate raises `TypeError` at `@on` decoration, or at graph construction for a `Command` attribute. Predicate exceptions propagate. They are not converted to violations.
- Multiple invariants short-circuit. One `InvariantViolated` per phase.
- Invariants run around `raises=`. The pre-check gates the body entirely. The post-check is skipped on a caught exception, which emits `HandlerRaised` instead. Post-check is a no-op when the handler returned `None` or declares no invariants.

### Modeling errors

| Situation | Vehicle |
|---|---|
| Expected domain outcome (including failure) | `DomainEvent` (`Order.Place.Rejected`) |
| Consistency rule gating a command (pre and/or post) | `invariants=` → `InvariantViolated` |
| Infrastructure failure (rate limit, timeout, parse error) | `Exception` + `raises=` → `HandlerRaised` |

## Warnings from `graph.namespaces()`

The first `graph.namespaces()` call emits two warnings that flag structural smells. `graph.reflect()` and a `Reflection` injected into a handler build the same model, so they emit them too. When the structure is intentional, filter with `warnings.filterwarnings("ignore", category=...)`.

- **`CommandChainWarning`**: a `chain`-causation edge, an inline command handler that returns another `Command`. Prefer to emit a fact and react to it.
- **`DomainPatternWarning`**: two or more events in the same namespace fan out to identical target sets. A shared base or a common handler is missing.

## `Interrupted` / `Resumed`

Subclass `Interrupted` with typed fields to pause for human input. Resume with `graph.resume(event)`, which requires a checkpointer. A `Resumed` event emits alongside the dispatched event.

```python
from langgraph.checkpoint.memory import MemorySaver

from langgraph_events import EventGraph, EventLog, IntegrationEvent, Interrupted, on


class OrderPlaced(IntegrationEvent):
    order_id: str
    total: float


class OrderConfirmed(IntegrationEvent):
    order_id: str


class OrderCancelled(IntegrationEvent):
    reason: str


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

The resume value must be an `Event`. Another value raises `TypeError` before the graph runs, so the thread stays paused and a later `resume()` can answer it. A handler that pauses two times in one run is resumed two times. After the first resume, the thread is paused again and `get_state(config).is_interrupted` is `True`. Under the default `on_unresumable="raise"`, `resume()` raises [`UnresumableError`](api.md#warnings) only when no registered handler waits for an answer.

For an interrupt whose frontend needs an action-discriminated dict, subclass `langgraph_events.agui.InterruptedWithPayload[PayloadT]`. See [Typed Interrupt Payloads](agui.md#typed-interrupt-payloads).

See the [approval pattern](patterns.md#expense-hitl) and [What changes safely](event-migrations.md#what-changes-safely).

### Ending a pause without answering it — `abandon()`

`graph.event_store.abandon(config, reason=...)` or `aabandon()` ends the pause without answering it. `graph.event_store` requires a checkpointer. `abandon()` settles one thread per call. Find the paused threads with `graph.event_store.threads_paused_on(EventClass, thread_ids=...)` or `athreads_paused_on()`:

```python
...
# candidates: thread ids from a server-side query.
paused = graph.event_store.threads_paused_on(OrderConfirmationRequested, thread_ids=candidates)
for config in paused:
    graph.event_store.abandon(config, reason="retiring OrderConfirmationRequested")
```

`thread_ids=` is required. The library does not read the whole store. See [Finding candidates server-side](retiring-interrupted.md#finding-candidates-server-side). Omit the class to get every listed thread that is paused. To check one thread, pass `thread_ids=[tid]`.

`abandon()` never dispatches the interrupt, and the interrupt never joins the event log. That is why it retires an `Interrupted` subclass: resuming every paused thread first would append the identity you are deleting. It appends a terminal [`Abandoned`](api.md#system-events) event, a [`Halted`](concepts.md#system-events) subtype. It leaves nothing scheduled. It preserves the writes of any completed sibling handler from the same fanned-out superstep. `Abandoned` is recorded, not dispatched. `.discarded` holds the qualnames of the discarded interrupts, joined with `", "`. It settles the latest checkpoint only. A historic checkpoint of the thread keeps its own `__interrupt__` write, so a replay against it still sees the pause. For a class that is already deleted, and for the threads that already answered the interrupt, see [Retiring an Interrupted subclass](retiring-interrupted.md).

WARNING: A release older than 0.29.0 cannot read an `Abandoned` event. A deployment rolled back below 0.29.0 raises `Cannot revive` on every read of an abandoned thread. Before you call `abandon()` in production, confirm that the previous production image also runs 0.29.0 or later.

WARNING: A run that is running on the thread overwrites what `abandon()` did, or the reverse, depending on ordering. Call `abandon()` only on a thread that is that no run uses.

`abandon()` raises `ValueError` if the thread has no events to settle (never run, or only `pre_seed()`ed), or if the thread has no pending interrupt, naming the thread. Pass `require_interrupt=False` to settle such a thread anyway. It records `Abandoned(discarded="")`.

`abandon()` does not consult [`on_unresumable`](api.md#graph-execution). That setting governs an accidental no-op `resume()`, not a deliberate abandonment. A later `resume()` on an abandoned thread still raises [`UnresumableError`](api.md#warnings) under the default setting, naming the abandonment.

## Soft-timeout — `RunPaused`

Pass `deadline=time.monotonic() + budget` to any graph entry point (`invoke`, `ainvoke`, `astream_events`, `stream_events`, `resume`, `aresume`, `astream_resume`, `stream_resume`, or `AGUIAdapter.stream`). When the router observes a wall-clock time past the deadline between dispatch rounds, it emits a `RunPausedOnDeadline(elapsed_seconds=…)` and the run terminates cleanly.

`RunPaused` is the base class of every pause. It names no cause, so `RunPaused(...)` raises `TypeError`. `RunPausedOnDeadline` is the deadline pause. `RunPausedOnSignal` is the [signal pause](#pause-on-a-signal-runpausedonsignal). Match `RunPaused` to see every pause, or match a subclass to see one cause.

```python
...
from time import monotonic

log = graph.invoke(Started(data="job-42"), deadline=monotonic() + 25.0)

if log.latest(RunPaused):
    # Paused. The next `invoke()` or stream call on the same `thread_id`
    # continues from the LangGraph checkpoint, the same mechanism as a
    # normal follow-up turn. No Command(resume=...) is needed.
    ...
```

Unlike `MaxRoundsExceeded`, `RunPaused` is **not terminal across runs**. The router advances the cursor past it, so the next call excludes the old `RunPaused` from `new_events`. Unlike `Interrupted`, which writes a pending interrupt task that `resume()` answers with `Command(resume=value)`, `RunPaused` writes **no** checkpoint task. The caller makes the next call with the same `thread_id` and a fresh deadline, and LangGraph replays from the last completed node.

Rules, shared with the signal pause:

- Set `deadline` strictly tighter than any hard cancellation the caller already has, such as `asyncio.wait_for` or LangGraph's `timeout=`, so that the pause fires first.
- Events emitted in the paused round are persisted in the event log but not dispatched, the same drop-on-pause semantic as `MaxRoundsExceeded`. Write handlers so that a stop at a round boundary leaves recoverable state.
- The first round of a run always dispatches, also with `deadline=0.0`. The resume round always dispatches: the router dispatches the resume value before it checks the deadline or the signal, so a pause does not lose the answer.
- `RunPaused` is emitted **at most once per run**, even when many parallel handlers are that is running when the deadline fires. A downstream projection can rely on one entry per pause.
- The check happens **only between rounds**, never inside a handler, so a handler that blocks cannot be preempted. The one exception is a [`RetryPolicy`](#retries) backoff, which reads the deadline and the signal before every wait and gives up instead of sleeping past them.

`RunPaused` has no default AG-UI wire mapping. See [Deadline and signal pauses](agui.md#deadline-and-signal-pauses).

### Pause on a signal — `RunPausedOnSignal`

A deadline is fixed when the run starts. To stop a live run early, pass `pause_on=` to the same entry points. The value is an object with an `is_set()` method, for example an `asyncio.Event` or a `threading.Event`. The `PauseSignal` protocol states this contract. The router calls `is_set()` between dispatch rounds. When it returns `True`, the router emits `RunPausedOnSignal(elapsed_seconds=…)` and the run ends cleanly.

```python
...
import asyncio

stopping = asyncio.Event()
log = await graph.ainvoke(seed, config=config, pause_on=stopping)

# At shutdown, the process sets the event. Each live run stops at its next round boundary.
stopping.set()
```

- `RunPausedOnSignal` is a subclass of `RunPaused`. Code that matches `RunPaused` also matches a signal pause. Match `RunPausedOnSignal` to tell the two causes apart.
- When the deadline has also expired, the router emits `RunPausedOnSignal`, not `RunPausedOnDeadline`.
- `elapsed_seconds` counts from the start of the run, also when the run has no deadline.
- The handler calls of the current round finish first. Give the run an outer grace period that is longer than one round.
- The value must be an instance with a callable `is_set`. Otherwise the entry point raises `TypeError` before any round runs. The common mistakes are `pause_on=stopping.is_set` and `pause_on=asyncio.Event`, the class. Pass the event itself. A type checker also rejects these values, because each entry point declares `pause_on: PauseSignal | None`.
- A `RetryPolicy` backoff reads the signal before every wait and gives up with a `HandlerAbandonedOnSignal`, which reaches no handler. See [Retries](#retries). Three rules apply to that give-up:
    - The router pauses over it instead of dispatching it. Two cases end the run without a `RunPausedOnSignal`, and the router still does not dispatch the give-up. A sibling handler interrupts in the same round: the run ends at the interrupt, the resume round dispatches the resume value and skips the give-up, and the resumed run reads its own `pause_on=`. The give-up lands in the last round that `max_rounds` allows: the run halts with `MaxRoundsExceeded`.
    - The first set read commits the run to a pause, also when the caller clears the signal between the two reads.
    - A backoff wait that is already in progress when the signal is set runs to its end, and `max_delay` bounds it. The policy gives up before the next wait.

### Surfacing the pause inline

To turn a `RunPaused` into a user-visible system message in the same channel as the chat history, register a custom reducer that handles both `MessageEvent` and `RunPaused`:

```python
from langchain_core.messages import BaseMessage, SystemMessage
from langgraph.graph.message import add_messages
from langgraph_events import Event, MessageEvent, Reducer, RunPaused

def project(event: Event) -> list[BaseMessage]:
    if isinstance(event, MessageEvent):
        return event.as_messages()
    if isinstance(event, RunPaused):
        return [SystemMessage(
            id=f"sys-paused-{event.elapsed_seconds:.6f}",
            content=f"Paused after {round(event.elapsed_seconds)}s. "
                    f"Send a follow-up to continue.",
        )]
    return []

messages = Reducer(
    name="messages",
    event_type=(MessageEvent, RunPaused),
    fn=project,
    reducer=add_messages,
    default=[],
)
```

## Field Matchers

`@on(Event, field=Type)` dispatches only when `event.field` is a `Type` instance. When the handler signature includes a parameter named `field`, the value is injected:

```python
...
@on(Resumed, interrupted=OrderConfirmationRequested)
def handle(event: Resumed, interrupted: OrderConfirmationRequested) -> OrderConfirmed:
    # `interrupted` is guaranteed to be OrderConfirmationRequested.
    ...
```

- A matcher value is an `Event`, `Exception` or `Invariant` subclass, matched by `isinstance`, or a `str`, matched by equality.
- Field names are validated at graph construction. A typo raises `TypeError`.
- Omit the parameter to filter dispatch without injection.

## Handler Exceptions

Declare `raises=(ExceptionClass, ...)` on a `Command`, or via `raises=` on `@on(...)`. A caught exception emits `HandlerRaised` with the exception, the handler name and `source_event`. Subscribe with `@on(HandlerRaised, exception=…)` to react.

```python
...
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
                # Honored only under respect_retry_after=True; decorative here.
                raise RateLimitError(retry_after=0.2)
            return Question.Ask.Answered(answer=...)

    class RetriesExhausted(Halted):
        reason: str = ""


@on(HandlerRaised, exception=RateLimitError)
def backoff(event: HandlerRaised, exception: RateLimitError) -> Question.Ask:
    return Question.Ask(question=event.source_event.question)  # retry
```

- Every type in `raises=` must be covered by at least one handler, or graph construction fails with `TypeError`. A handler covers `X` when it has no field matchers, or only `exception=X` or a superclass. A non-`exception` matcher does not count. A handler counts only when it subscribes to `HandlerRaised` itself, not to a subtype such as `HandlerAbandonedOnSignal`.
- Only `Exception` subclasses are allowed. `BaseException`, `KeyboardInterrupt`, `SystemExit`, `GeneratorExit` and `asyncio.CancelledError` are rejected. `CancelledError` surfaces as `Cancelled`, a `Halted` subtype.
- An unhandled raise propagates and crashes the run. A handler of `HandlerRaised` can itself declare `raises=` to escalate.
- Use `HandlerRaised.source_event`, not `event`, for the triggering event. This avoids a kwarg collision.

## Retries

Declare `retry=RetryPolicy(...)` alongside `raises=`, as a class attribute on a `Command` or via `retry=` on `@on(...)`. The library calls the handler again in place with exponential backoff. `HandlerRaised` fires only when the budget is spent, or when the run's `deadline=` or `pause_on=` cuts the backoff short. So a handler of `HandlerRaised` is an escalation handler.

```python
...
from langgraph_events import RetryPolicy


class Question(Namespace):
    class Ask(Command):
        ...  # the fields, `raises` and the handler, as above
        retry = RetryPolicy(max_attempts=3, base_delay=0.1, max_delay=10.0)


@on(HandlerRaised, exception=RateLimitError)
def give_up(event: HandlerRaised) -> Question.RetriesExhausted:
    return Question.RetriesExhausted(reason=str(event.exception))  # budget spent, or out of time
```

- `max_attempts` counts the **initial call**: `3` means one call plus two retries.
- Delay before retry *n* is `base_delay * 2 ** (n - 1)`, capped at `max_delay`. `strategy="constant"` uses `base_delay` as that ceiling on every retry.
- `jitter=True` (the default) samples the wait uniformly from `[0, ceiling]`, whatever the strategy. So `strategy="constant", jitter=True` still varies per retry, averaging `base_delay / 2`. Pass `jitter=False` for a flat wait.
- `on=(...)` narrows which exceptions retry. Each entry must **overlap** `raises=`: a subclass of a declared raise (`raises=(OSError,)`, `on=(ConnectionResetError,)`) or a superclass of one (`raises=(ConnectionResetError,)`, `on=(OSError,)`). Scope is decided at runtime by `isinstance(exc, on)`. Anything declared in `raises=` but outside `on=` surfaces on its first raise.
- `respect_retry_after=True` prefers a server-supplied `exception.retry_after` over the computed curve, clamped to `[0, max_delay]` and never jittered. A non-numeric or `bool` hint is ignored.
- Each wait emits a `HandlerRetried` (handler, `source_event`, exception, `attempt`, `delay_seconds`). `observe="log"` records a `WARNING` with the exception's type and message instead, and `observe="silent"` records neither. Only `observe="emit"` (the default) writes the event to the log, so only it contributes the `retry` edge to `HandlerRetried` in [`graph.namespaces()`](concepts.md#namespace-introspection-visualization). The `retry xN` annotation on the command shows under every setting.
- Declaring `retry=` without `raises=`, or an `on=` entry **disjoint from** every type in `raises=`, fails at graph construction with `TypeError`.
- Retries happen inside the handler node. They consume no `max_rounds` budget and write no checkpoint between attempts.
- A sleep is never *started* when it would land on or past the run's `deadline=`. The policy gives up at that point, even with attempts left. Nothing is clamped. An attempt that cannot finish before the deadline only delays the pause. A set `pause_on=` signal gives up the same way. See [Pause on a signal](#pause-on-a-signal-runpausedonsignal).
- A deadline give-up is a `HandlerAbandonedOnDeadline`. It is a `HandlerAbandoned`, which is a `HandlerRaised`, so an `@on(HandlerRaised, exception=X)` handler runs for it. An exhausted attempt budget, an out-of-`on=` exception and a handler with no policy each emit a plain `HandlerRaised`. Use `isinstance(event, HandlerAbandonedOnDeadline)` to tell "ran out of time" from "ran out of tries". `HandlerAbandoned` names no cause, so `HandlerAbandoned(...)` raises `TypeError`. The abandoned attempt emits no `HandlerRetried`, because no wait happened.
- **Handlers must be idempotent.** A retried handler re-runs from the top, including any `emit_custom` it fired before raising.
- **`HandlerRetried` carries the exception without its traceback.** Matchers and field injection work on the live instance, but `__traceback__` is detached. The terminal `HandlerRaised` keeps its own. For a *per-attempt* stack, capture it inside the handler while the frames are still live.
- Not to be confused with `langgraph.types.RetryPolicy`, which re-runs an entire LangGraph node. Import this one from `langgraph_events`.

See the [Error Recovery pattern](patterns.md#error-recovery).
