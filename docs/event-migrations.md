# Event migrations

`NamespaceAwareSerde` keys event identity by `(__module__, __qualname__)`, so nested events with the same leaf name revive as different classes. A migration maps a historic identity onto the live class on read. The wire format does not change, so a payload from any earlier library version still revives.

## What changes safely

The table gives the effect of each change on a thread that already has a checkpoint. A paused thread is a thread stopped on an `Interrupted` event.

| Change | Effect on an existing thread | Action |
|---|---|---|
| Add a handler | It takes part at the next run. | None |
| Add an event class | It is emitted and matched normally. | None |
| Remove an event class | Its stored instances stay in the log, and no handler matches them. Under `NamespaceAwareSerde` a read of the thread raises `Cannot revive`. Under LangGraph's default serde the instance revives as `None`, with no error. | Map the identity onto a tombstone. See [Recovering a delete-first deployment](retiring-interrupted.md#recovering-a-delete-first-deployment) |
| Remove an `Interrupted` subclass | A paused thread holds a pending `__interrupt__` write that names the class. The write cannot revive. | `abandon()` each paused thread first. See [Retiring an Interrupted subclass](retiring-interrupted.md) |
| Remove a handler | An event that only the removed handler subscribed to is not delivered, and the run halts early. A thread paused inside the handler cannot resume. `resume()` raises `UnresumableError` by default. | Declare `@on(previously=...)`, or `abandon()` the paused thread |
| Rename or move a handler | A thread paused inside the handler cannot resume. | Declare `@on(previously="old_name")`. See [Handler renames](#handler-renames) |
| Rename or move an event class | Under `NamespaceAwareSerde` a read raises `Cannot revive`. Under LangGraph's default serde the instance revives as `None`, with no error. | Declare `@migrate_from`. The [migration gate](migration-gate.md#the-migration-gate) reports an undeclared rename in CI |
| Add a reducer | Its channel holds no history. Only the events after the next run contribute. | Rebuild the channel with [`replay_reducer`](#recovering-with-replay_reducer) |
| Remove a reducer | Its channel value is dropped from the checkpoint, with no error. | None |

Prefer an additive change. Removal is safe only when no thread is in flight, or after `abandon()` settled each paused thread. A new `thread_id` after a structural change avoids every case above.

## Two tracks, one model

| | Event class renamed or moved | Handler renamed or moved |
|---|---|---|
| **Declare** (recover after the fact) | `@migrate_from("Old.Qualname")` | `@on(previously="old_node")`, or `previously: ClassVar = (...)` on a `Command` |
| **Prevent** (up front) | None. The qualname is the identity | `@on(node_name="stable_id")`. Then rename the function freely |
| **Catch** in CI | [the migration gate](migration-gate.md#the-migration-gate): rules `coverage` and `revive` | [the migration gate](migration-gate.md#the-migration-gate): rule `handlers` |

One gate checks both tracks, from one `MigrationTarget`. The two tracks are independent, so an event and a handler renamed together take both decorators in one PR.

!!! note "The one invariant"
    **Every rename is a decorator and a `baseline update` in the same PR.** `baseline update` records the event identities and the handler node names. The gate reports a decorator without a baseline update as `BaselineStale`.

## Handler renames

A handler is a graph node keyed by its name. A thread paused inside a renamed or moved handler cannot resume until the rename is declared:

```python
from typing import ClassVar

from langgraph_events import Command, IntegrationEvent, Namespace, on


class Confirmed(IntegrationEvent):
    pass


@on(Confirmed, previously="await_confirmation")
def confirm(event: Confirmed) -> None: ...


class Order(Namespace):
    class Place(Command):
        # An inline command handler has no decorator slot.
        previously: ClassVar = ("Order.Submit", "Order.Place.Submit")
```

- `@on(node_name=...)` pins a stable node identity that is not the Python function name. It is for a standalone `@on` function only. `@on(previously=...)` registers an alias node for each historic name, so a paused checkpoint re-enters the renamed handler.
- An alias node is additive, so a handler rename ships in one release. An event rename needs two releases. See [Rolling deploys](#rolling-deploys).
- Prove coverage with [the migration gate](migration-gate.md#the-migration-gate) and recovery with `assert_resume_recovers`. See [Testing handler recovery](#testing-handler-recovery).
- The node identity of an inline command handler is the `__qualname__` of the command, for example `Order.Place`, not the method name. The order of `handlers=[...]` does not matter. A renamed command class is a renamed node. Annotate `previously` as `ClassVar`, or leave it without an annotation. A plain annotation makes it an event field, and the class is rejected at creation.
- `previously` and `@migrate_from` on one class answer different questions. `@migrate_from` revives the old payload bytes. `previously` revives the old checkpoint pointer, the node name in `snapshot.next`. A node-only era, for example a handler the command replaced, has no serde identity. A module-only move has no node rename.
- A historic node name identifies exactly one class. A concrete `Command` cannot be subclassed, so `previously`, like `raises`, `invariants` and `retry`, is read from the command class that declares it. A second command that took over some of the old node names declares its own list.
- When a handler is removed, or renamed without `previously=`, and a thread is still paused inside it, `EventGraph(on_unresumable=...)` decides what `resume()` does. It also fires on a resume of a finished thread and on a double resume:
    - `raise` (default): `UnresumableError`. Use it in development and CI.
    - `halt`: emit a terminal `Unresumable(Halted)` and finalize the thread. Use it in production when the thread must end in the log.
    - `warn`: `UserWarning`, and the thread is untouched. Use it when you handle the thread yourself.

!!! note "A baseline recorded before inline handlers were keyed by qualname"
    An earlier release recorded an inline command handler by a positional name, such as `handle` or `handle_2`. The gate reports those names as `HandlerMissing` until you run `baseline update` once. A checkpoint paused inside an inline command handler before the upgrade cannot resume afterwards. Recover it with `previously: ClassVar = ("handle_N",)` on that command class, or set `on_unresumable="halt"` or `"warn"`.

!!! warning "Renaming an inline Command is both renames at once"
    The command class is the node identity and the event class of the payload in the paused checkpoint. The alias node dispatches by `isinstance` against the new class, so `previously` alone is not enough. Declare `@migrate_from` on the command too, and put a `NamespaceAwareSerde` on the checkpointer. `from_namespaces(..., checkpointer=...)` wires one. With LangGraph's default serializer the alias node re-enters, finds no matching event, and the resume does nothing. `on_unresumable` does not fire, because the thread is still paused.

## The minimum case: rename inside a namespace

```python
from langgraph.checkpoint.memory import MemorySaver

from langgraph_events import Command, DomainEvent, EventGraph, Namespace
from langgraph_events.serde import migrate_from


class Order(Namespace):
    class Place(Command):
        @migrate_from("Order.Placed")
        class Placed(DomainEvent):
            note: str = ""

        def handle(self) -> "Order.Place.Placed":
            return Order.Place.Placed()


graph = EventGraph.from_namespaces(Order, checkpointer=MemorySaver())
```

- `from_namespaces(..., checkpointer=...)` wires a migration-aware serde scoped to the passed namespaces. The serde walks only those namespaces for `@migrate_from` metadata. A decorator on an unrelated imported module contributes nothing.
- `from_namespaces` keeps a `NamespaceAwareSerde` that the checkpointer already carries, for example one with hand-authored `migrations=`. It replaces LangGraph's default `JsonPlusSerializer` only when that serde has its default settings.
- WARNING: `from_namespaces` raises `TypeError` for any other serde, so that it never drops an encryption or a serializer setting. Build the graph with `EventGraph.of(topology, checkpointer=saver)` instead, and give the saver the topology serde:

    ```python
    topology = EventTopology.from_namespaces(Order)
    saver = MemorySaver(serde=EncryptedSerializer(cipher, serde=topology.serde()))
    graph = EventGraph.of(topology, checkpointer=saver)
    ```

    A wrapped serde reads and writes through `topology.serde()`. But the saver then holds no `NamespaceAwareSerde`, so `EventStore.unrevivable_threads()`, `plan_rewrite()` and `rewrite_store()` raise `ValueError` on it.

A multi-step chain lists the oldest qualname first. Stacked decorators give the same chain, with the oldest at the bottom. The serde flattens the chain, so any historic step revives at the current class in one lookup. Prefer the multi-arg form:

```python
...  # inside Order.Place, as above
@migrate_from("Order.Placed", "Order.OldNest.Placed")  # oldest first
class Placed(DomainEvent): ...
```

## Adding a field

An event is a frozen dataclass. A new field with a `default` or a `default_factory` takes it when an old payload omits the key, with no migration. Use `@backfill` when the new field is required. It is collected like `@migrate_from`:

```python
from langgraph_events import Command, DomainEvent, Namespace
from langgraph_events.serde import backfill, migrate_from


class Order(Namespace):
    class Place(Command):
        @migrate_from("Order.Placed")  # renamed
        @backfill("command_id", default="legacy")  # and gained a required field
        class Placed(DomainEvent):
            command_id: str  # required when constructed in code
            note: str = ""
            tags: tuple[str, ...] = ()  # added later. An old payload revives with ()
```

- Composed with `@migrate_from`, the rename applies first, then the fill on the resulting identity.
- Stacked `@backfill` decorators accumulate, one per added field.
- `default` and `default_factory` follow the `AddField` convention. A mutable `default=[]` raises `ValueError` at construction. Use `default_factory=list`.

## Dropping, merging or retyping a field

A rename or a fill cannot remove a stored value or change its shape. `TransformFields` runs a callable on the stored kwargs, and the return value replaces them. The decorator form is `@transform_fields`, collected like `@backfill`. A merge and a retype are the same operation with a different callable:

```python
from langgraph_events import Command, DomainEvent, Namespace
from langgraph_events.serde import transform_fields


def drop_legacy_flag(kw: dict) -> dict:
    kw.pop("legacy_flag", None)  # tolerate an absent key
    return kw


class Order(Namespace):
    class Place(Command):
        @transform_fields(drop_legacy_flag)
        class Placed(DomainEvent):
            order_id: str = ""  # ``legacy_flag`` was dropped
```

The identity picks the stage, the same rule as `AddField`:

| Stage | Form | Keyed on | Runs | Applies to |
|---|---|---|---|---|
| Class stage | `@transform_fields(fn)`, `Migration.transform_fields(target=Class, transform=fn)` | the live class | after the rename, before the class fills | payloads from every era, including payloads the current release writes |
| Origin stage | `@migrate_from("Old", transform=fn)`, `Migration.transform_fields(module=..., qualname="<historic>", transform=fn)` | a historic identity | before the rename, before the origin fills | payloads written under that exact origin only |
| Split | `@split_event(fn, targets=(...))`, `Migration.split_event(source=Class, select=fn, targets=(...))` | the live class | after the class-stage transform, before the fills of the resulting identity | payloads from every era, including payloads the current release writes |

The read path runs the origin stage (transform, then origin fills), the rename, then the class stage (transform, then [split](#splitting-one-stored-event-into-two-on-a-payload-value), then class fills). A fill still applies to a key the transform removed or never produced.

- **Replace, not merge.** The return value is the full kwargs the constructor receives. Return `kw` after an edit in place, or return a new dict. A transform that returns `{}` discards every stored field.
- **A stable name.** Give the transform a function name at module level, or build it with a factory or `functools.partial`. See [Labels](migration-gate.md#labels) for the key of a local function or a partial. The [migration gate](migration-gate.md#the-rules) reports a lambda, a callable object, or a local function that captures a value with no stable form as `AnonymousCallable`.
- **Idempotent.** A class-stage transform sees current payloads for ever, so it must accept a payload from every era. Use `kw.pop("x", None)`, not `del kw["x"]` or `kw["x"]`.
- **One transform per identity and stage.** Compose the steps in one callable. A second decorator is rejected at decoration. A second hand-authored operation is rejected at serde construction.
- **The target must resolve**, live or as a rename source, the same rule as `AddField`.
- **A transform that raises, or returns a non-dict,** fails the read with `Cannot revive <stored identity>: TransformFields raised <Type>: <message>` and a remedy. Under `serde.tolerate_unresolved()` the identity degrades to `UnrevivedIdentity` and is collected, so `unrevivable_threads()` reports it.
- `migrate_from(transform=...)` takes exactly one historic qualname per decorator, the same rule as `backfill=`. The two can sit on one decorator. The transform runs first.
- A transform cannot be written under `legacy_write`. See [Rolling deploys](#rolling-deploys).

## Splitting one stored event into two on a payload value

One stored identity can stand for two outcomes, and both classes are live. A rename cannot host this, because a rename source must not resolve to a live class. `SplitEvent` runs `select` on a copy of the stored kwargs. `select` returns `None` to keep the stored class and kwargs, or `(target_class, kwargs)` to build the target from those kwargs. The decorator form is `@split_event` on the source class, collected like `@transform_fields`:

```python
from dataclasses import field
from typing import Any

from langgraph_events import DomainEvent, Namespace
from langgraph_events.serde import split_event


def select_failed(kw: dict[str, Any]) -> tuple[type, dict[str, Any]] | None:
    result = kw.get("result")
    if result is None or result.get("status") != "error":
        return None  # keep Job.Completed
    return Job.Failed, {"reason": result["message"]}  # resolved at read time


class Job(Namespace):
    class Failed(DomainEvent):
        reason: str = ""

    @split_event(select_failed, targets=(Failed,))
    class Completed(DomainEvent):
        result: dict[str, Any] = field(default_factory=dict)
```

Pin each branch with real values through `synthesize_legacy_payload`:

```python
...  # Job as above
from langgraph_events.serde import NamespaceAwareSerde, synthesize_legacy_payload

serde = NamespaceAwareSerde(namespaces=[Job])
stored = {"result": {"status": "error", "message": "disk full"}}
failed = serde.loads_typed(synthesize_legacy_payload(Job.__module__, "Job.Completed", stored))
assert failed == Job.Failed(reason="disk full")
```

- **Order.** A split runs in the class stage only, keyed on the live source identity. The table under [Dropping, merging or retyping a field](#dropping-merging-or-retyping-a-field) gives its place in the read path. When `select` returns a target, the class fills of the target apply, not the fills of the source. When `select` returns `None`, the fills of the source apply. A payload stored under a historic name that renames onto the source is split too.
- **The class object is the target.** `select` returns the class, not a string, so an IDE rename follows it.
- **`targets` lists every class `select` can return.** Each is validated at serde construction. It must be an `Event` subclass that resolves in the scope of the serde or by import. A target that `select` returns and `targets` does not list is refused at read time.
- **One split per identity.** Compose the cases in one `select`. A second decorator is rejected at decoration. A second hand-authored operation is rejected at serde construction.
- **The source must resolve live.** A split keyed on a rename source is refused at serde construction. Declare the split on the live target instead. It runs after the rename, on every era.
- **A target is not split again.** A split runs once, on the stored identity. A target that is itself a split source keeps the kwargs `select` returned.
- **A split changes what the retirement tools see.** `threads_paused_on(Source)` does not return a thread whose pending interrupt splits to a target. Filter on the target class instead.
- **A `select` failure fails the read** with `Cannot revive <stored identity>: SplitEvent raised <Type>: <message>` and a remedy. Under `serde.tolerate_unresolved()` the stored identity degrades to `UnrevivedIdentity` and is collected, so `unrevivable_threads()` reports it. A failure is: `select` raises, returns a value that is not `None` or a `(target, kwargs)` tuple, returns a target outside `targets`, or returns kwargs that are not a `dict`.
- **There is no dotted discriminator path.** A nested value normally arrives revived, so `result.status` means `getattr`. Under `LANGGRAPH_STRICT_MSGPACK=true`, or when the module of the value is not in `allowed_msgpack_modules`, the same stored bytes arrive as a plain `dict`. Your `select` owns the access, and a `KeyError` there is ordinary Python.
- A split cannot be written under `legacy_write`. See [Rolling deploys](#rolling-deploys).

## Consolidating N classes into one

When several per-entity classes collapse into one shared class with a required discriminator, the value for each old payload depends on the historic identity. A class-global `@backfill` gives one value for everyone. Pin the value per origin with `backfill=` on each `@migrate_from`:

```python
from langgraph_events import Command, DomainEvent, Namespace
from langgraph_events.serde import migrate_from


class EntityLifecycle(Namespace):
    class Approve(Command):
        @migrate_from("Order.Approve.Approved", in_module="app.namespaces", backfill={"entity_type": "order"})
        @migrate_from("Invoice.Approve.Approved", in_module="app.namespaces", backfill={"entity_type": "invoice"})
        class Approved(DomainEvent):
            entity_type: str  # required. An old payload never carried it
            entity_id: str = ""
```

A consolidation ships as three moves in one PR. Delete the old class definitions first. A `@migrate_from` whose historic identity still resolves to a live class is rejected at serde construction with "resolves to a currently-live class". Then decorate the surviving class as above, and run `baseline update`, the [one invariant](#two-tracks-one-model).

- **Precedence:** explicit payload value, then origin-scoped fill, then class-global `@backfill`.
- **Exact-origin contract:** a fill applies only to a payload written under that qualname. On a chain `@migrate_from("A", "B")`, a fill keyed on `B` does not apply to an `A`-era payload. So `backfill=` requires exactly one qualname per decorator, and the multi-arg chain form rejects it. The [`revive` rule](migration-gate.md#the-rules) of the gate cannot catch a mid-chain fill that had to be class-global. An earlier era gets a placeholder fill and passes, and a real payload from that era fails at read.
- **A mutable value** is rejected at decoration. For a per-origin `default_factory`, hand-author `Migration.add_field(module=..., qualname="<historic>", field=..., default_factory=...)`. An `AddField` keyed on a historic identity applies before the rename, to that origin only.
- Two fills for the same `(origin, field)`, for example a decorator fill and a hand-authored one, raise at serde construction. A duplicated origin qualname is caught at decoration.
- A consolidation cannot be written under `legacy_write`. See [Rolling deploys](#rolling-deploys).

### Hand-authored migrations

For a cross-module relocation or a composite operation, use `langgraph_events.serde.migrations` directly:

| Use case | Pattern |
|---|---|
| Single rename | `Migration.rename(to=Class, ...)` |
| Add field | `Migration.add_field(target=Class, field=..., default=...)` |
| Origin-scoped add field | `Migration.add_field(module=..., qualname="<historic>", field=..., default=...)` |
| Transform fields | `Migration.transform_fields(target=Class, transform=fn)` |
| Origin-scoped transform | `Migration.transform_fields(module=..., qualname="<historic>", transform=fn)` |
| Split event | `Migration.split_event(source=Class, select=fn, targets=(Target, ...))` |
| Cross-module rename | `Migration(name=..., operations=(RenameEvent(...),))` |
| Multiple operations | `Migration(name=..., operations=(op1, op2, ...))` |

```python
from app.namespaces import Order
from langgraph_events.serde import Migration

migrations = [
    # Pass the live class, not strings, so that a refactor follows it.
    Migration.rename(old_module="app.namespaces", old_qualname="Order.Placed", to=Order.Place.Placed),
    Migration.add_field(target=Order.Place.Placed, field="command_id", default="legacy"),
]
```

Use strings only for a cross-module case where the class cannot be imported at authoring time. `name` is optional everywhere. Import the raw `RenameEvent`, `AddField`, `TransformFields` and `SplitEvent` from `langgraph_events.serde.migrations`. They are not re-exported at `langgraph_events.serde`.

Every rule above is checked at serde construction, not at the first read. A rename chain that cycles, a duplicate rename source, a dead-end chain or a rename that shadows a live class raises `ValueError`. A fill that names a field the live class does not have raises `ValueError`. A `SplitEvent` with empty `targets` raises `ValueError`, and a non-callable `select` raises `TypeError`. An unknown operation type raises `TypeError`. `@migrate_from(backfill={})` raises `ValueError` at decoration. So a misspelled `@migrate_from("Order.Placedd")` fails at construction.

## Rolling deploys

!!! warning "Rolling deploys require two releases"
    A new pod that writes under a new qualname produces payloads that an old pod cannot revive.

    **Release N+1: `legacy_write=True`**
    ```python
    ...  # NAMESPACES: the namespaces of your app
    from langgraph_events.serde import NamespaceAwareSerde

    serde = NamespaceAwareSerde(namespaces=NAMESPACES, legacy_write=True)
    ```
    A new pod encodes each event under the oldest historic qualname that `@migrate_from` recorded. An old pod of release N reads it through its existing class. Both pod versions can resume the threads of the other.

    **Release N+2: `legacy_write=False` (default)**
    Once release N is fully drained, write the current qualname. Keep `@migrate_from` for the old-format payloads in storage. Delete the decorator in the next release, after the `rewrite` step has read the store clean. See [Lifecycle of a read-time migration](migration-gate.md#lifecycle-of-a-read-time-migration).

    `legacy_write` is scope-symmetric. A decorated class outside `namespaces=` is encoded under its current qualname. Keep `namespaces=` the same on the encode pod and the decode pod.

!!! warning "What cannot be written under `legacy_write`"
    A write relabelled under the oldest historic identity carries the current shape. An old release cannot read a shape that its class does not accept. Drain in-flight threads before the cutover, or accept read-only compatibility: the old payloads revive, and an old pod cannot read a new one. `NamespaceAwareSerde(..., legacy_write=True)` raises `ValueError` at construction for the first three cases:

    1. A transform, in either stage. It runs on read and has no inverse.
    2. A split. An old release has no class for the target.
    3. A consolidation. `legacy_write` relabels every instance under the oldest historic identity, and the old classes do not accept the discriminator.
    4. A required field with a class-global `@backfill`. The library does not refuse this case. The new pod writes the new field, and the class of the old pod rejects the kwarg.

    The framework migrations of the library do not count. See [Framework events](#framework-events).

The serde and migration layer takes no lock and opens no transaction. A migration is pure and writes nothing back, so concurrent reads are safe. Concurrency on one `thread_id` belongs to the checkpointer. `MemorySaver` provides none. The SQLite and Postgres savers bring their own. `replay_reducer` overwrites with the same value from any number of concurrent runners. `baseline update` is not atomic. An update never erases a recorded identity, so a second divergent writer moves the additions of the first writer to `retired`, and the gate reports them. Generate and commit the baseline from one CI job, never in parallel.

## Reducer state migration

A reducer projection lives in a checkpoint channel value. Events already migrate, so most changes are automatic:

| Change | Behaviour | Action |
|---|---|---|
| `list[Event]` reducer, event class renamed | Each event migrates on read | None |
| `ScalarReducer` holding one event, class renamed | Migrates | None |
| `dict[str, list[Event]]` grouping reducer, event renamed | Migrates | None |
| Plain dataclass channel, new field with a default | Revives with the dataclass default | None |
| Plain dataclass channel, new required field | **Revives as `None`, with no error** | [`replay_reducer`](#recovering-with-replay_reducer) or strict mode |
| Pydantic model channel, new required field | **Revives malformed.** It passes `isinstance`, then raises `AttributeError` on access | [`replay_reducer`](#recovering-with-replay_reducer) or strict mode |
| Reducer output shape changed | Revives as the old shape, and a consumer fails downstream | [`replay_reducer`](#recovering-with-replay_reducer) |
| Projection function semantics changed | Stale data, with no exception | [`replay_reducer`](#recovering-with-replay_reducer), the only fix |

### Recovering with `replay_reducer`

```python
...  # checkpointer, config and my_reducer come from your app
from langgraph_events.serde import replay_reducer

# In a startup migration script, run once during the deploy. The library
# does not iterate the checkpointer for you.
tup = checkpointer.get_tuple(config)
events = tup.checkpoint["channel_values"]["events"]  # already migrated on read
rebuilt = replay_reducer(my_reducer, events)
# Write `rebuilt` back through the put API of the checkpointer.
```

`replay_reducer` is a thin wrapper around `BaseReducer.seed(events)`. The reducer default, the namespace filter and the `event_type` predicate all apply.

Strict mode demotes an unrecognised class to a raw kwargs `dict` instead of a malformed object. The first consumer access then raises `AttributeError` or `TypeError`. Process-wide, `export LANGGRAPH_STRICT_MSGPACK=true`, for development and CI. Per serde, `NamespaceAwareSerde(..., allowed_msgpack_modules=[("module", "ClassName"), ...])`, for production. Strict mode does not raise at the serde boundary. To fail at the read instead of at the first access, rebuild the value with `replay_reducer`.

## Framework events

The library owns the read migrations for its own framework events. Every `NamespaceAwareSerde` applies them, whatever `namespaces=` and `events=` you pass.

| Name | Stored payload | Revives as |
|---|---|---|
| `lge-0.41-handler-raised-split` | `HandlerRaised` with `abandoned_for_deadline=True` | `HandlerAbandonedOnDeadline`, without the key |
| `lge-0.41-handler-raised-split` | `HandlerRaised` with `abandoned_for_deadline=False` | `HandlerRaised`, without the key |
| `lge-0.41-handler-raised-split` | `HandlerRaised` without the key | `HandlerRaised`, unchanged |
| `lge-0.41-run-paused-split` | `RunPaused` | `RunPausedOnDeadline` |
| `lge-0.42-halted-transform` | `Halted` with `reason`, from 0.1.x | `Halted`, without the key |
| `lge-0.42-frontend-tool-call-requested-rename` | `FrontendToolCallRequested` in `langgraph_events._event`, from 0.5.x and earlier | `FrontendToolCallRequested` in `langgraph_events.agui._events` |

- The `FrontendToolCallRequested` rename exists only when the `agui` extra is installed.
- `HandlerAbandonedOnSignal` and `RunPausedOnSignal` have no framework migration. No release before 0.41.0 wrote them.
- Before 0.6.0, LangGraph's own `JsonPlusSerializer` wrote each event in its constructor record form. `NamespaceAwareSerde` reads such a record of a framework event through these migrations, and a rewrite stores it in the form of `NamespaceAwareSerde`. A plan reads such a record as a rewrite until then. A record of your own class in that form stays with LangGraph's reader. See [A checkpoint that a plan refuses](migration-gate.md#a-checkpoint-that-a-plan-refuses).
- A framework event is always written under its own live identity. The `legacy_write=True` guards judge your own migrations only.
- Only `NamespaceAwareSerde` applies these migrations. With LangGraph's plain `JsonPlusSerializer`, an old `RunPaused` or `HandlerRaised` revives as `None`, with no error. `EventGraph(checkpointer=...)` keeps the serializer of the checkpointer. Only `EventGraph.from_namespaces` puts a `NamespaceAwareSerde` on it. Otherwise, pass `serde=NamespaceAwareSerde(...)` to the checkpointer yourself.
- A migration of your own on an identity that the library migrates, for example a `split_event` on `HandlerRaised`, raises `ValueError` at construction. Declare it on a subclass of your own, which has its own identity.
- A framework split keys on the framework class only, so a subclass of `HandlerRaised` that you declared is not split. If 0.40.0 stored your subclass, its payload carries the `abandoned_for_deadline` key. Drop the key with a `@transform_fields` on your subclass. If your subclass can carry `True`, map the value to a field of your own before you pop it.

### Rolling back to an older release

An older release has no class for a new framework type. It fails on the first read with `Cannot revive langgraph_events._event.RunPausedOnDeadline`. Draining in-flight threads does not prevent this. The `events` channel is written whole at each checkpoint, so one step on the new release stores the whole history under the new identities. `rewrite_store()` also rewrites the old events. Before you roll back, give the serde of the older release read migrations that map each new type back to its old shape. For a rollback from 0.41.0 to 0.40.0:

```python
...  # Order: a namespace of your app
# On 0.40.0. 0.41.0 refuses this list, because its renames shadow live
# classes. Delete it when you upgrade again.
from langgraph_events import HandlerRaised, RunPaused
from langgraph_events.serde import Migration, NamespaceAwareSerde

LGE = "langgraph_events._event"


def flag_deadline(kw: dict) -> dict:
    return {**kw, "abandoned_for_deadline": True}


ROLLBACK_FROM_041 = (
    Migration.rename(old_module=LGE, old_qualname="HandlerAbandonedOnDeadline", to=HandlerRaised),
    Migration.transform_fields(module=LGE, qualname="HandlerAbandonedOnDeadline", transform=flag_deadline),
    Migration.rename(old_module=LGE, old_qualname="HandlerAbandonedOnSignal", to=HandlerRaised),
    Migration.rename(old_module=LGE, old_qualname="RunPausedOnDeadline", to=RunPaused),
    Migration.rename(old_module=LGE, old_qualname="RunPausedOnSignal", to=RunPaused),
)

serde = NamespaceAwareSerde(ROLLBACK_FROM_041, namespaces=[Order])
```

## What is not migrated

- **A non-Event payload**, such as a Pydantic model, a plain dataclass or a LangGraph `Interrupt` wrapper, goes through LangGraph's default serde. An event nested inside `Interrupt.value` is migrated. A Pydantic model used as an event field must live at module scope. LangGraph's serde stores it by `__name__` and revives it with `getattr` on the module. On a miss it returns a raw `dict` with no error. `NamespaceAwareSerde` raises `ValueError` at construction for a model nested inside a class or a function. A model held behind `Any` or an untyped container is not reachable from an annotation and is not checked.
- **A reducer channel rename** is a LangGraph channel-routing concern. See [What changes safely](#what-changes-safely).
- **A payload that `ormsgpack` refuses to encode** raises at the source, with no fallback. The library has no extension point for the encoder.
- **A non-reducer channel value** has no rebuild path. Recovery is custom: read, transform, then write back through the put API of the saver.

## Testing your migrations

Run [the migration gate](migration-gate.md#the-migration-gate) in a test with `assert_migration_gate`. The gate is static. It proves that an alias exists, not that a specific payload or resume works.

### Testing handler recovery

`assert_resume_recovers` runs the real interrupt and resume path:

```python
...  # await_input, handle_confirm, gather_input, Started and Confirmed come from your app
from langgraph.checkpoint.memory import MemorySaver

from langgraph_events import EventGraph
from langgraph_events.serde import assert_resume_recovers


def test_rename_keeps_checkpoints_resumable():
    saver = MemorySaver()  # `before` and `after` must share one checkpointer
    before = EventGraph([await_input, handle_confirm], checkpointer=saver)
    # `gather_input` is `await_input` renamed, with @on(previously="await_input")
    after = EventGraph([gather_input, handle_confirm], checkpointer=saver)

    assert_resume_recovers(before, after, seed=Started(...), resume_with=Confirmed())
```

The call invokes `before` with `seed`, which must pause through `Interrupted`. It resumes `after` with `resume_with` on the same checkpoint, and asserts that a `Resumed` was emitted. A silent drop or a `halt` fails the assertion. It returns the log for further assertions.

`synthesize_legacy_payload(module, qualname, kwargs)` builds the bytes that an earlier release wrote. Pass them to `serde.loads_typed` and assert the class, as under [Splitting](#splitting-one-stored-event-into-two-on-a-payload-value). Use it only when the shape of a field drifted. A failing test is a dataclass `TypeError` on the missing field.
