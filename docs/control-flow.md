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

### Ending a pause without answering it — `abandon()`

`graph.resume(event)` answers a pending `Interrupted`. `graph.abandon(config, reason=...)` or `aabandon()` ends the pause without ever answering it. Both require a checkpointer.

```python
graph.abandon(config, reason="retiring OrderConfirmationRequested")
```

`abandon()` settles one thread per call. Find the paused threads with `graph.threads_paused_on(EventClass)` (or `athreads_paused_on()`):

```python
for config in graph.threads_paused_on(OrderConfirmationRequested):
    graph.abandon(config, reason="retiring OrderConfirmationRequested")
```

Omit the class to get every paused thread. `threads_paused_on()` reads every checkpoint the checkpointer holds. Cost is O(all checkpoints), not O(paused threads). A large deployment should filter thread IDs server-side instead of calling it directly.

`abandon()` discards the interrupt instead of answering it: it never dispatches the interrupt, and the interrupt never joins the event log. This is why `abandon()` exists to retire an `Interrupted` subclass. Resuming every paused thread first would append the very identity you are deleting.

The thread is terminal afterwards. `abandon()` appends a terminal [`Abandoned`](api.md#system-events) event (a [`Halted`](concepts.md#system-events) subtype). It leaves nothing scheduled. It preserves any completed sibling handler's writes from the same fanned-out superstep.

`abandon()` raises `ValueError` if the thread has no events to settle (never run, or only `pre_seed()`ed). `abandon()` also raises `ValueError` if the thread has no pending interrupt, naming the thread. This catches a stray ID in a candidate list before it silently closes out settled business history. Pass `require_interrupt=False` to settle such a thread anyway. It records `Abandoned(discarded="")`.

Like every event on this settle path, `Abandoned` is recorded, not dispatched. `@on(Abandoned)` never fires. Read it back like any other event: `graph.get_state(config).events.latest(Abandoned)` gives you `.reason` and `.discarded`. `.reason` is the caller-supplied string, `""` if none. `.discarded` holds the discarded interrupts' **qualnames**, for example `"Order.ApprovalRequested"` for a class nested in a `Namespace`, never the bare `"ApprovalRequested"`. Values are deduped and joined with `", "`, `""` if none. Match it with `in` or split on `", "`. Never use `==`. A fanned-out superstep can pause two interrupts, and `==` stops matching then.

Always the qualname, never the bare class name: it stays unambiguous under nesting. For a plain retirement (handler deleted, or handler and class deleted together with no recovery), it also stays the *same string*: a check written before that retirement keeps matching after it. It does **not** stay the same string once the [tombstone recovery](event-migrations.md#recovering-a-delete-first-deployment) runs. `discarded` then holds the *tombstone's* qualname (e.g. `Order.RetiredApprovalGate`), not the retired class's (`Order.ApprovalRequired`). A check pinned to the original name stops matching from that point on.

`threads_paused_on()` and `abandon()` still work if the `Interrupted` class itself has already been deleted, not just its handler. This is the delete-first mistake this library's docs used to train. Neither can construct the class anymore, so `discarded` then carries the interrupt's last-known qualname instead of a live instance. See [Recovering a delete-first deployment](event-migrations.md#recovering-a-delete-first-deployment) to map that identity back onto a tombstone class. The fix also revives any thread that had already answered the interrupt. `graph.unrevivable_threads()` finds those threads: it reports every thread whose latest checkpoint names an identity that no longer revives, in its settled history or its pending interrupt.

`abandon()` cleans only the live checkpoint. A historic checkpoint for the same thread keeps its own `__interrupt__` write, so time-travel or replay against it still sees the original pause. Retirement is therefore safe only for the identity a current thread is resting on, the same framing as [event class rename/relocate](event-migrations.md#the-minimum-case-rename-inside-a-namespace).

!!! warning "Concurrent runs"
    A run in flight on the thread will silently overwrite whatever `abandon()` did,
    or vice versa, depending on ordering. Call `abandon()` only on a thread that is
    genuinely at rest.

`abandon()` does not consult [`on_unresumable`](api.md#graph-execution). That policy governs an accidental no-op `resume()` (a renamed/removed handler, a double resume), not a deliberate abandonment. A later `resume()` on an abandoned thread still raises [`UnresumableError`](api.md#warnings) under the default policy, naming the abandonment instead of pointing at a handler rename.

### Typed payloads — `InterruptedWithPayload`

For interrupts whose frontend needs an action-discriminated dict, subclass `langgraph_events.agui.InterruptedWithPayload[PayloadT]` and implement `interrupt_payload(self) -> PayloadT`. The AG-UI adapter recognises the contract directly — see [AG-UI](agui.md).

## Soft-timeout — `RunPaused`

Pass `deadline=time.monotonic() + budget` to any graph entry point (`invoke`, `ainvoke`, `astream_events`, `stream_events`, `resume`, `aresume`, `astream_resume`, `stream_resume`, or `AGUIAdapter.stream`). When the router observes a wall-clock time past the deadline between dispatch rounds, it emits a `RunPaused(elapsed_seconds=…)` and the run terminates cleanly.

```python
from time import monotonic

log = graph.invoke(Started(data="job-42"), deadline=monotonic() + 25.0)

if log.latest(RunPaused):
    # Soft-stopped. The next /run on the same thread_id continues
    # from the LangGraph checkpoint — same mechanism as a normal
    # follow-up turn. No Command(resume=...) required.
    ...
```

Unlike `MaxRoundsExceeded`, `RunPaused` is **not terminal across runs**: the router advances the cursor past it so a fresh `/run` on the same `thread_id` excludes the old `RunPaused` from `new_events` and continues. Resume semantics intentionally differ from `Interrupted`:

- `Interrupted` writes a pending interrupt task to the checkpoint; resume uses `Command(resume=value)` to deliver a typed resume value to the paused node.
- `RunPaused` writes **no** checkpoint task; the worker/UI just makes a new `/run` call (with the same `thread_id` and a fresh deadline). LangGraph's checkpointer replays from the last completed node.

Position `deadline` strictly tighter than whichever hard cancellation the caller already has (`asyncio.wait_for`, SAQ `job_timeout`, LangGraph's `timeout=`) so the soft boundary fires first and the wire-format finalize path runs cleanly. In-flight events from the round when the deadline fires are persisted in the event log but not dispatched — same drop-on-pause semantic as `MaxRoundsExceeded`. Handlers should produce events such that a clean round-boundary stop leaves recoverable state.

`RunPaused` is emitted **at most once per `/run`**, even when many parallel handlers are still in flight when the deadline fires. The router gates re-emission so that downstream projections (custom reducers, message-channel notices) can rely on one inline entry per pause.

The deadline is checked **only between rounds**, never inside a handler — so a handler that blocks cannot be preempted by it. The one exception is [`RetryPolicy`](#retries) backoff, which reads the same deadline before every wait: a sleep that would land on or past the boundary is abandoned rather than started, and the give-up surfaces as `HandlerRaised(abandoned_for_deadline=True)`. So you do not have to size the worst-case total backoff, `max_delay * (max_attempts - 1)`, under your deadline budget — the waits between attempts are bounded for you. The attempts themselves are not: a single long-running handler call can still run past the boundary, and the pause lands at the first round boundary after it returns.

### Surfacing the pause inline

A `RunPaused` is just an event in the log. To turn it into a user-visible system message in the same channel as your chat history, register a custom reducer that handles both `MessageEvent` and `RunPaused`:

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

### AG-UI wire shape

`RunPaused` is intentionally **not** surfaced on the AG-UI wire by default. There is no built-in mapping: `FallbackMapper` skips it (one-time warning) because the previous `CustomEvent(name="interrupted", value={"kind": "soft_timeout", …})` overload collided with HITL `Interrupted` events on the same wire name. Apps that want a pause signal on the wire register their own `EventMapper` — see [AG-UI → Custom Mappers](agui.md#custom-mappers).

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
                # Honored only under respect_retry_after=True; decorative here.
                raise RateLimitError(retry_after=0.2)
            return Question.Ask.Answered(answer=...)

    class GaveUp(Halted):
        reason: str = ""


@on(HandlerRaised, exception=RateLimitError)
def backoff(event: HandlerRaised, exception: RateLimitError) -> Question.Ask:
    return Question.Ask(question=event.source_event.question)  # retry
```

- Every type in `raises=` must be covered by at least one catcher, else graph construction fails with `TypeError`. A catcher covers `X` if it has no field matchers, or only `exception=X`-or-superclass. Non-`exception` matchers don't count.
- Only `Exception` subclasses allowed — `BaseException`/`KeyboardInterrupt`/`SystemExit`/`GeneratorExit`/`asyncio.CancelledError` rejected. `CancelledError` surfaces as `Cancelled` (a `Halted` subtype).
- Unhandled raises propagate and crash the run. Catchers can themselves declare `raises=` to escalate.
- Use `HandlerRaised.source_event` (not `event`) for the triggering event — avoids kwarg collision.

## Retries

Declare `retry=RetryPolicy(...)` alongside `raises=` — on a `Command` as a class attribute, or via `retry=` on `@on(...)`. The framework re-invokes the handler in place with exponential backoff; `HandlerRaised` fires only once the budget is spent — or once the run's `deadline=` cuts the backoff short — so catchers become pure escalation handlers.

```python
from langgraph_events import RetryPolicy


class Question(Namespace):
    class Ask(Command):
        question: str = ""

        raises = (RateLimitError,)
        retry = RetryPolicy(max_attempts=3, base_delay=0.1, max_delay=10.0)

        class Answered(DomainEvent):
            answer: str = ""

        def handle(self) -> Question.Ask.Answered:
            if upstream_rate_limited():
                # Honored only under respect_retry_after=True; decorative here.
                raise RateLimitError(retry_after=0.2)
            return Question.Ask.Answered(answer=...)

    class GaveUp(Halted):
        reason: str = ""


@on(HandlerRaised, exception=RateLimitError)
def give_up(event: HandlerRaised) -> Question.GaveUp:
    return Question.GaveUp(reason=str(event.exception))  # budget spent, or out of time
```

- `max_attempts` counts the **initial call**: `3` means one call plus two retries.
- Delay before retry *n* is `base_delay * 2 ** (n - 1)`, capped at `max_delay`. `strategy="constant"` uses `base_delay` as that ceiling on every retry instead of doubling it.
- `jitter` is orthogonal to `strategy` and applies to whichever ceiling the strategy computes. With `jitter=True` (the default) the wait is sampled uniformly from `[0, ceiling]` — full jitter, against thundering herds. Note this means `strategy="constant", jitter=True` still varies per retry, averaging `base_delay / 2`; pass `jitter=False` for a genuinely flat wait.
- `on=(...)` narrows which exceptions retry; each entry must **overlap** `raises=` — either a subclass of a declared raise (`raises=(OSError,)`, `on=(ConnectionResetError,)`) or a superclass of one (`raises=(ConnectionResetError,)`, `on=(OSError,)`). Scope is decided at runtime by `isinstance(exc, on)`, and only what `raises=` already catches ever reaches the policy. Anything declared in `raises=` but outside `on=` surfaces on its first raise — that is how a non-transient error stays non-transient.
- `respect_retry_after=True` prefers a server-supplied `exception.retry_after` over the computed curve, clamped to `[0, max_delay]` and never jittered — so a skewed clock or an already-past `Retry-After` retries immediately rather than crashing the run. A non-numeric or `bool` hint is ignored and the computed curve is used.
- Each wait emits a `HandlerRetried` (handler, `source_event`, exception, `attempt`, `delay_seconds`). Use `observe="log"` for a `WARNING` instead, or `observe="silent"` for neither. Only `observe="emit"` (the default) writes the event to the log, so only it contributes the `retry` edge to `HandlerRetried` in [`graph.namespaces()`](concepts.md#namespace-introspection-visualization) — the diagram tracks what the log will contain, not what was merely declared. The `retry xN` annotation on the command shows under every setting.
- Declaring `retry=` without `raises=`, or an `on=` entry **disjoint from** every type in `raises=`, fails at graph construction with `TypeError` — the policy could never fire.
- Retries happen inside the handler node: they consume no `max_rounds` budget and write no checkpoint between attempts.
- The backoff is **deadline-aware**. A sleep is never *started* if it would land on or past the run's `deadline=` — the policy gives up there and then, even with attempts left, so the run returns to the router instead of sleeping through the soft boundary. Nothing is clamped: burning the rest of the budget on an attempt that probably cannot finish either only delays the pause.
- That give-up is the only raise tagged `abandoned_for_deadline=True`. The field is `False` for an exhausted attempt budget, an out-of-`on=` exception, and a handler with no policy at all, so a catcher can tell "ran out of time" from "ran out of tries". The abandoned attempt emits no `HandlerRetried` — no wait happened.
- **Handlers must be idempotent.** A retried handler re-runs from the top, including any `emit_custom` it fired before raising.
- **`HandlerRetried` carries the exception without its traceback.** The instance is live — matchers and field injection work on it — but `__traceback__` is detached. The terminal `HandlerRaised` keeps its own.
- Not to be confused with `langgraph.types.RetryPolicy`, which re-runs an entire LangGraph node. Import this one from `langgraph_events`.

Idempotence and the detached traceback are two consequences of the same design: the framework holds onto your failure. A retried call re-runs from the top, so any side effect it performed before raising happens again. And because the `events` channel is append-only, whatever a stored event still references lives for the rest of the run — a traceback pins the failing attempt's frame and every local on it, so a breadcrumb that kept one would hold a copy of what the handler was working on (the LLM response, the fetched payload, the dataframe) once per attempt. So the breadcrumb keeps the exception and drops the frames: `HandlerRetried.exception` is the live instance — `@on(HandlerRetried, exception=RateLimitError)` isinstance-matches it, field injection hands it over typed, `str(...)` and `.args` read normally — with `__traceback__` cleared, along with those of its `__cause__`/`__context__` chain and any `ExceptionGroup` members. Only the terminal `HandlerRaised` keeps a traceback, so the framework retains at most one live traceback per failing invocation, and that is the one you debug from.

If you need a *per-attempt* stack, capture it inside the handler while the frames are still live. `observe="log"` records each attempt as a `WARNING`, but logs the exception's type and message, not its frames.

See the [Error Recovery pattern](patterns.md#error-recovery).
