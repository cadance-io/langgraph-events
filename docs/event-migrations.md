# Event migrations

See also: [Checkpointer & graph evolution](checkpointer-evolution.md).

`NamespaceAwareSerde` keys event identity by `(__module__, __qualname__)` so nested events with colliding leaf names round-trip distinctly. Migrations rewrite historic identities on read while preserving wire format — payloads from any prior library version remain readable both ways.

## Two tracks, one model

Renaming an **event class** and renaming a **handler** are the same problem at two layers, solved the same way — *declare the prior identity → auto-recover; a CI gate → catch the undeclared case*:

| | Event class renamed/moved | Handler renamed/moved |
|---|---|---|
| **Declare** (recover after the fact) | `@migrate_from("Old.Qualname")` | `@on(previously="old_node")` |
| **Prevent** (up front) | (qualname *is* the identity) | `@on(node_name="stable_id")` — then rename the function freely |
| **Catch** in CI | [the migration gate](#the-migration-gate): rules `coverage` and `revive` | [the migration gate](#the-migration-gate): rule `handlers` |

Event-class migrations come first below; the [handler track](#handler-renames) is at the end. One gate checks both tracks, from one `MigrationTarget`.

!!! note "The one invariant"
    **Every rename = decorator *and* `baseline update` in the same PR.** The gate reports a decorator without a baseline update as `BaselineStale`.

## The minimum case: rename inside a namespace

```python
from langgraph_events import Command, DomainEvent, Namespace
from langgraph_events.serde import migrate_from


class Persona(Namespace):
    class Persist(Command):
        @migrate_from("Persona.Persisted")
        class Persisted(DomainEvent):
            note: str = ""


graph = EventGraph.from_namespaces(
    Persona,
    handlers=[...],
    checkpointer=MemorySaver(),
)
```

- `from_namespaces(..., checkpointer=...)` auto-wires a migration-aware serde scoped to the passed namespaces — no manual `NamespaceAwareSerde` construction.
- The serde walks only those namespaces for `@migrate_from` metadata; decorators on unrelated imported modules contribute nothing.
- Opt out by passing `MemorySaver(serde=<custom>)` — a user-supplied serde always wins.

## Adding a field with a default

**No migration needed.** Events are frozen dataclasses; a new field with a default (or `default_factory`) simply uses it when an old payload omits the key:

```python
class Persisted(DomainEvent):
    note: str = ""
    tags: tuple[str, ...] = ()   # added later — old payloads revive with ()
```

Reach for `@backfill` (below) only when the new field is **required** (no default).

## Multi-step chains

```python
@migrate_from("Persona.Persisted", "Persona.OldNest.Persisted")  # oldest first
class Persisted(DomainEvent): ...
```

- Multi-arg form: oldest qualname first.
- Stacked decorators apply bottom-up; the bottom-most decorator is the oldest. Both forms produce the same chain.
- Serde flattens the chain so any historic step revives directly at the current class in one lookup.

Prefer the multi-arg form for new code.

## Adding a required field

Use `@backfill` when the field is **required** in code but absent from pre-existing payloads. Auto-collected like `@migrate_from`:

```python
from langgraph_events.serde import backfill, migrate_from


class Persona(Namespace):
    class Persist(Command):
        @migrate_from("Persona.Persisted")          # renamed…
        @backfill("command_id", default="legacy")   # …and gained a required field
        class Persisted(DomainEvent):
            command_id: str        # required when constructed in code
            note: str = ""
```

- Compose with `@migrate_from`: rename is applied first, then back-fill on the resulting identity.
- Stacked `@backfill` accumulate (one per added field).
- `default`/`default_factory` follow the `AddField` convention; mutable `default=[]` raises `ValueError` at construction (use `default_factory=list`).

## Dropping, merging or retyping a field

A rename moves an identity. A back-fill adds a field a payload never carried. Neither can remove a stored value or change its shape. `TransformFields` can. It runs a callable on the stored kwargs, and the return value replaces them. The decorator form is `@transform_fields`, auto-collected like `@backfill`:

```python
from langgraph_events.serde import transform_fields


def drop_legacy_flag(kw: dict) -> dict:
    kw.pop("legacy_flag", None)   # tolerate an absent key
    return kw


class Order(Namespace):
    class Place(Command):
        @transform_fields(drop_legacy_flag)
        class Placed(DomainEvent):
            order_id: str = ""     # ``legacy_flag`` was dropped
```

A merge and a retype are the same operation with a different callable:

```python
def merge_name(kw: dict) -> dict:
    first, last = kw.pop("first", None), kw.pop("last", None)
    if first is not None or last is not None:
        kw["name"] = f"{first or ''} {last or ''}".strip()
    return kw


def int_count(kw: dict) -> dict:
    if isinstance(kw.get("count"), str):
        kw["count"] = int(kw["count"])
    return kw
```

**Two stages.** The identity picks the stage, the same rule as `AddField`:

| Stage | Form | Keyed on | Runs | Applies to |
|---|---|---|---|---|
| Class stage | `@transform_fields(fn)`, `Migration.transform_fields(target=Class, transform=fn)` | the live class | after the rename, before the class fills | payloads from every era, including payloads the current release writes |
| Origin stage | `@migrate_from("Old", transform=fn)`, `Migration.transform_fields(module=..., qualname="<historic>", transform=fn)` | a historic identity | before the rename, before the origin fills | payloads written under that exact origin only |
| Split | `@split_event(fn, targets=(...))`, `Migration.split_event(source=Class, select=fn, targets=(...))` | the live class | after the class-stage transform, before the fills of the resulting identity | payloads from every era, including payloads the current release writes |

Read-path order: the origin stage (transform, then origin fills), the rename, then the class stage (transform, then [split](#splitting-one-stored-event-into-two-on-a-payload-value), then class fills). A transform runs before the fills of its stage. A fill still applies to a key the transform removed or never produced.

Semantics:

- **Replace, not merge.** The return value is the full kwargs the constructor receives. Return `kw` after editing it in place, or return a new dict. `transform=lambda kw: {}` discards every stored field.
- **Idempotent.** A class-stage transform sees current payloads for ever. Use `kw.pop("x", None)`. Do not use `del kw["x"]` or `kw["x"]`. A transform must accept a payload from every era.
- **One transform per identity and stage.** Compose the steps in one callable. A second decorator is rejected at decoration. A second hand-authored op is rejected at serde construction.
- **The target must resolve**, live or as a rename source, the same rule as `AddField`.
- **A transform that raises, or returns a non-dict,** fails the read with `Cannot revive <stored identity>: TransformFields raised <Type>: <message>` and a remedy. Under `serde.tolerate_unresolved()` the identity degrades to `UnrevivedIdentity` and is collected, so `unrevivable_threads()` reports it.
- `migrate_from(transform=...)` takes exactly one historic qualname per decorator, the same rule as `backfill=`. The two can sit on one decorator. The transform runs first.

!!! warning "Transforms cannot ride `legacy_write` (enforced)"
    A transform runs on read and has no inverse. A write relabelled under the oldest historic identity would carry the current shape, which the old release's class does not accept. `NamespaceAwareSerde(..., legacy_write=True)` raises at construction when you declare any transform, in either stage. lge's own [framework migrations](#framework-events) do not count. Drain in-flight threads before the cutover, or accept read-only compatibility.

!!! note "The revive rule sends `None` placeholders"
    The baseline records the fields a class dropped. The gate's `revive` rule sends `None` for each of them, so the transform runs through the real read path. A transform that reads a dropped value, for example `kw.pop("legacy_flag").upper()`, raises on `None` and fails the gate. The failure line says so. Guard the value in the transform, or pin a real payload with `synthesize_legacy_payload`.

## Splitting one stored event into two on a payload value

One stored identity can stand for two outcomes. `Job.Completed{result}` was written for every job, and `result` says whether the job failed. The current release has a `Job.Failed` class for that case, and `Job.Completed` stays live. A rename cannot host this split: a rename source must not resolve to a live class. `SplitEvent` can. It runs `select` on a copy of the stored kwargs. `select` returns `None` to keep the stored class and kwargs, or `(target_class, kwargs)` to build the target from those kwargs instead. The decorator form is `@split_event`, applied to the source class and auto-collected like `@transform_fields`:

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
from langgraph_events.serde import NamespaceAwareSerde, synthesize_legacy_payload

serde = NamespaceAwareSerde(namespaces=[Job])

failed = serde.loads_typed(
    synthesize_legacy_payload(
        Job.__module__,
        "Job.Completed",
        {"result": {"status": "error", "message": "disk full"}},
    )
)
assert failed == Job.Failed(reason="disk full")

completed = serde.loads_typed(
    synthesize_legacy_payload(
        Job.__module__, "Job.Completed", {"result": {"status": "ok"}}
    )
)
assert completed == Job.Completed(result={"status": "ok"})
```

**Order.** A split runs in the class stage only, keyed on the live source identity. Read-path order: the origin stage (transform, then origin fills), the rename, the class-stage transform, the split, then the fills of the resulting identity. When `select` returns a target, the target's class fills apply. The source's fills do not. When `select` returns `None`, the source's fills apply. A payload stored under a historic name that renames onto the source is split too. The table under [Dropping, merging or retyping a field](#dropping-merging-or-retyping-a-field) lists every stage.

Semantics:

- **The class object is the target.** `select` returns the class, not a string, so an IDE rename follows it.
- **`targets` lists every class `select` can return.** Each is validated at serde construction: it must be an `Event` subclass that resolves in the serde's scope or by import. A target `select` returns that is not in `targets` is refused at read time.
- **One split per identity.** Compose the cases in one `select`. A second decorator is rejected at decoration. A second hand-authored op is rejected at serde construction.
- **The source must resolve live.** A split keyed on a rename source is refused at serde construction. Declare the split on the live target instead: it runs after the rename, on every era.
- **A target is not split again.** A split runs once, on the stored identity. A target that is itself a split source keeps the kwargs `select` returned.
- **A split changes what the retirement tools see.** `threads_paused_on(Source)` does not return a thread whose pending interrupt splits to a target. Filter on the target class instead.
- **A `select` failure fails the read** with `Cannot revive <stored identity>: SplitEvent raised <Type>: <message>` and a remedy. Under `serde.tolerate_unresolved()` the stored identity degrades to `UnrevivedIdentity` and is collected, so `unrevivable_threads()` reports it. The failure modes:
    - `select` raises.
    - `select` returns a value that is not `None` or a `(target, kwargs)` tuple.
    - `select` returns a target outside `targets`.
    - `select` returns kwargs that are not a `dict`.
- **There is no dotted discriminator path.** A nested value normally arrives revived, so `result.status` means `getattr`. Under `LANGGRAPH_STRICT_MSGPACK=true`, or when the value's module is not in `allowed_msgpack_modules`, the same stored bytes arrive as a plain `dict`. A path resolver would have to try both, and a deployment that changes the allowlist would change which branch fires. The author's callable owns the access instead. A `KeyError` there is ordinary Python, and the traceback names that code.

!!! warning "Splits cannot ride `legacy_write` (enforced)"
    A split runs on read and has no inverse. An old release has no class for the target. `NamespaceAwareSerde(..., legacy_write=True)` raises at construction when you declare any split. lge's own [framework migrations](#framework-events) do not count. Drain in-flight threads before the cutover, or accept read-only compatibility.

!!! note "The revive rule sends `None` placeholders"
    The gate's `revive` rule sends `None` for every required field of the source, and for every recorded field it dropped. Return `None` from `select` when the discriminating value is absent or `None`. The gate then proves the source still constructs. A `select` that reads the placeholder, for example `kw["result"]["status"]`, raises and fails the gate. The failure line says so. Pin each branch with real values through `synthesize_legacy_payload`, as above.

## Consolidating N classes into one

When several per-entity classes collapse into ONE shared class with a required discriminator, the correct value for each old payload is determined by **which historic identity** it was written under — something a class-global `@backfill` (one value for everyone) cannot express. Pin it per origin with `backfill=` on each `@migrate_from`:

```python
class EntityLifecycle(Namespace):
    class Approve(Command):
        @migrate_from("Persona.Approve.Approved", in_module="app.namespaces.persona",
                      backfill={"entity_type": "persona"})
        @migrate_from("Story.Approve.Approved", in_module="app.namespaces.story",
                      backfill={"entity_type": "story"})
        @migrate_from("Scenario.Approve.Approved", in_module="app.namespaces.scenario",
                      backfill={"entity_type": "scenario"})
        class Approved(DomainEvent):
            entity_type: str   # required — old payloads never carried it
            entity_id: str
```

Shipping a consolidation is three moves in one PR:

1. **Delete the old class definitions** — a `@migrate_from` whose historic identity still resolves to a live class is rejected at serde construction ("resolves to a currently-live class"); that error is the library telling you step 1 isn't done, not that the migration is wrong.
2. **Decorate the surviving class** as above.
3. **Update the baseline** (`baseline update`) in the same PR — the [one invariant](#two-tracks-one-model).

Semantics:

- **Precedence:** explicit payload value > origin-scoped fill > class-global `@backfill` (which acts as the fallback for origins without a scoped entry).
- **Exact-origin contract:** a fill applies only to payloads written under *that* qualname. On a temporal chain (`@migrate_from("A", "B")`) a fill keyed on `B` does **not** apply to `A`-era payloads — "every era" is class-global `@backfill`'s job. Accordingly, `backfill=` requires exactly one qualname per decorator; the multi-arg chain form rejects it. Note the gate's [`revive` rule](#the-rules) cannot catch a mid-chain fill that should have been class-global: earlier eras get placeholder-filled and pass, while a real payload from that era fails at read.
- **Mutable values** are rejected at decoration. For a per-origin `default_factory`, hand-author `Migration.add_field(module=..., qualname="<historic>", field=..., default_factory=...)` — an `AddField` keyed on a historic identity is applied *before* the rename, only to that origin.
- Two fills for the same `(origin, field)` raise at serde construction (reachable when a decorator fill collides with a hand-authored one; a duplicated origin *qualname* is caught even earlier, at decoration).

!!! warning "Consolidations cannot ride `legacy_write` (enforced)"
    The two-release rename dance (below) does not work for a fan-in: the old classes don't accept the new discriminator kwarg, and `legacy_write` relabels **every** instance under the *oldest* historic identity — persona, story, and scenario alike. `NamespaceAwareSerde(..., legacy_write=True)` therefore **raises at construction** when origin-scoped fills are present. Drain in-flight threads before cutover, or accept read-only compatibility (old payloads revive; old pods cannot read new ones).

### Hand-authored escape hatch

For cross-module relocations or composite operations, drop to `langgraph_events.serde.migrations`:

| Use case | Pattern |
|---|---|
| Single rename | `Migration.rename(to=Class, ...)` |
| Add field | `Migration.add_field(target=Class, field=..., default=...)` |
| Origin-scoped add field | `Migration.add_field(module=..., qualname="<historic>", field=..., default=...)` |
| Transform fields | `Migration.transform_fields(target=Class, transform=fn)` |
| Origin-scoped transform | `Migration.transform_fields(module=..., qualname="<historic>", transform=fn)` |
| Split event | `Migration.split_event(source=Class, select=fn, targets=(Target, ...))` |
| Cross-module rename | `Migration(name=..., operations=(RenameEvent(...),))` |
| Multiple ops | `Migration(name=..., operations=(op1, op2, ...))` |

```python
from cadance.persona import Persona
from langgraph_events.serde import Migration

migrations = [
    Migration.rename(
        old_module="cadance.persona",
        old_qualname="Persona.Persisted",
        to=Persona.Persist.Persisted,        # ← live class, not strings
    ),
    Migration.add_field(
        target=Persona.Persist.Persisted,    # ← same
        field="command_id",
        default="legacy",
    ),
]
```

Pass the live class for refactor safety; strings only for cross-module cases where the class can't be imported at authoring time. `name` is optional everywhere. Raw `RenameEvent` / `AddField` / `TransformFields` / `SplitEvent` are imported from `langgraph_events.serde.migrations` (not re-exported at `langgraph_events.serde`).

## Rolling deploys

!!! warning "Rolling deploys require two releases"
    New pods writing under a new qualname produce payloads old pods can't revive. Ship the migration in **two releases**:

    **Release N+1: `legacy_write=True`**
    ```python
    serde = NamespaceAwareSerde(namespaces=NAMESPACES, legacy_write=True)
    ```
    New pods encode events under the oldest historic qualname (recorded by `@migrate_from`). Old pods (release N) read those via existing class defs. Both pod versions can resume each other's threads.

    **Release N+2: `legacy_write=False` (default)**
    Once release N is fully drained, flip writes to current qualname. Keep `@migrate_from` — it covers remaining old-format payloads in storage. Drop the decorator after `graph.rewrite_store()` has rewritten every live checkpoint under the new qualname. See [Rewriting the live set](#rewriting-the-live-set).

    `legacy_write` is scope-symmetric: decorated classes outside `namespaces=` are encoded under their current qualname (otherwise the read path of the same serde could not migrate them). Keep `namespaces=` consistent between encode and decode pods.

### Concurrency guarantees

No locks or transactions in the serde/migration layer. Safe by **idempotency** and **by construction**:

- **Concurrent reads are safe** — rewrite is pure, no write-back; rename table is a transitive closure.
- **Old and new code cannot share a process** — a `@migrate_from` whose historic identity still resolves to a live class is rejected at serde construction (the rolling-deploy model is structurally enforced).
- **`legacy_write=True` is coexistence, not a lock** — format compatibility for the two-release window.
- **Required-field addition is a two-release operation, like a rename** — pair with `@backfill` and ship over the N → N+1 cadence.
- **Thread-level concurrency on a single `thread_id` is the checkpointer's job** (`MemorySaver` provides none; SQLite/Postgres savers bring their own).
- **Recovery replay is idempotent** — `replay_reducer` overwrites with the same correct value from any number of concurrent runners.
- **One unprotected spot: `baseline update` is non-atomic.** An update never erases a recorded identity, so a second divergent writer moves the first writer's additions to `retired` and the gate reports them. A true within-call read→write interleave between two CI processes is a TOCTOU the library does not guard. It is a dev/CI tool, not a runtime path — generate and commit the baseline from a **single** CI job, never in parallel.

## Reducer state migration

Reducer projections live in checkpoint channel values. Events already migrate, so most changes are automatic:

| Change | Behaviour | Action |
|---|---|---|
| `list[Event]` reducer + event class renamed | Each event migrates through ext-hook recursively | ✅ Nothing |
| `ScalarReducer` holding single Event + class renamed | Migrates | ✅ Nothing |
| `dict[str, list[Event]]` grouping reducer + event renamed | Migrates | ✅ Nothing |
| Plain dataclass channel + new field WITH default | Revives via dataclass default | ✅ Nothing |
| Plain dataclass channel + new REQUIRED field | **Silently revives as `None`** | ⚠️ [`replay_reducer`](reducers.md#replay_reducer) or strict mode |
| Pydantic model channel + new REQUIRED field | **Revives malformed** — passes `isinstance`, `AttributeError` on access | ⚠️ [`replay_reducer`](reducers.md#replay_reducer) or strict mode |
| Reducer output shape changed | Revives as old shape — consumer crashes downstream | ⚠️ [`replay_reducer`](reducers.md#replay_reducer) |
| Projection function semantics changed | Silent stale data, no exception | ⚠️ [`replay_reducer`](reducers.md#replay_reducer) (only fix) |

### Recovering with `replay_reducer`

```python
from langgraph_events.serde import replay_reducer

# In a startup migration script (run once during deploy):
tup = checkpointer.get_tuple(config)
event_log = tup.checkpoint["channel_values"]["event_log"]   # adjust to your channel name

rebuilt = replay_reducer(my_reducer, event_log)

# Write `rebuilt` back through the checkpointer's put API.
```

- Thin wrapper around `BaseReducer.seed(events)` — reducer default, namespace filter, and `event_type` predicate all apply.
- Composes with event-rename machinery (`event_log` was already migrated on read).
- Library doesn't iterate the checkpointer for you — wire the read/write loop in your own startup script.

### Catching silent revivals loudly

Strict mode demotes unrecognised classes to raw kwargs `dict` instead of malformed objects — the first consumer access trips `AttributeError`/`TypeError`, much louder than `None`.

- Env var (process-wide): `export LANGGRAPH_STRICT_MSGPACK=true`. Set in dev and CI.
- Per-serde allowlist: `NamespaceAwareSerde(..., allowed_msgpack_modules=[("module", "ClassName"), ...])`. Use in production for fine-grained scoping.

Strict mode does NOT raise at the serde boundary. For "fail at deserialization, not first access" semantics, use `replay_reducer` to rebuild from event truth.

## Framework events

lge owns the read migrations for its own framework events, such as `HandlerRaised` and `RunPaused`. Every `NamespaceAwareSerde` applies them, whatever `namespaces=` and `events=` you pass. You do not declare them.

| Name | Stored payload | Revives as |
|---|---|---|
| `lge-0.41-handler-raised-split` | `HandlerRaised` with `abandoned_for_deadline=True` | `HandlerAbandonedOnDeadline`, without the key |
| `lge-0.41-handler-raised-split` | `HandlerRaised` with `abandoned_for_deadline=False` | `HandlerRaised`, without the key |
| `lge-0.41-handler-raised-split` | `HandlerRaised` without the key | `HandlerRaised`, unchanged |
| `lge-0.41-run-paused-split` | `RunPaused` | `RunPausedOnDeadline` |

lge's framework migrations are not relabelled on write. A framework event is always written under its own live identity. The `legacy_write=True` guards judge your own migrations only, so a framework split does not make `legacy_write=True` raise.

`HandlerAbandonedOnSignal` and `RunPausedOnSignal` have no framework migration. No release before 0.41.0 wrote them.

!!! warning "Only `NamespaceAwareSerde` applies these migrations"
    With LangGraph's plain `JsonPlusSerializer`, an old `RunPaused` or `HandlerRaised` revives as `None`, with no error. `EventGraph(checkpointer=...)` keeps the serializer of the checkpointer. Only `EventGraph.from_namespaces` puts a `NamespaceAwareSerde` on it. Otherwise, pass `serde=NamespaceAwareSerde(...)` to the checkpointer yourself.

lge refuses a migration of your own on an identity that lge migrates. For example, a `split_event` or a `transform_fields` on `HandlerRaised` raises `ValueError` at construction. A class-stage transform runs before lge's split, so it could drop the field that the split reads. Declare the migration on a subclass of your own, which has its own identity.

### Your own subclass of a framework event

A framework split keys on the framework class only. A subclass of `HandlerRaised` that you declared has its own identity, so the split does not apply to it. If 0.40.0 stored your subclass, its payload carries the `abandoned_for_deadline` key. Drop the key with a `@transform_fields` on your subclass. If your subclass can carry `True`, map the value to a field of your own before you pop it.

```python
from langgraph_events import HandlerRaised
from langgraph_events.serde import transform_fields


def drop_deadline_flag(kw: dict) -> dict:
    kw.pop("abandoned_for_deadline", None)
    return kw


@transform_fields(drop_deadline_flag)
class Escalated(HandlerRaised):
    note: str = ""
```

### Rolling back to an older release

An older release has no class for a new framework type, such as `HandlerAbandonedOnDeadline`. It fails on the first read with `Cannot revive langgraph_events._event.RunPausedOnDeadline`.

Draining in-flight threads before the rollback does not prevent this. The `events` channel is written whole at each checkpoint. So a thread that takes one step on the new release stores its whole history under the new identities, old events included. `rewrite_store()` also rewrites the old events.

Before you roll back, give the serde of the older release read migrations that map each new type back to its old shape. For a rollback from 0.41.0 to 0.40.0:

```python
# On 0.40.0, the release you roll back to. 0.41.0 refuses this list, because
# its renames shadow live classes: delete it when you upgrade again.
from langgraph_events import HandlerRaised, RunPaused
from langgraph_events.serde import Migration, NamespaceAwareSerde

LGE = "langgraph_events._event"


def flag_deadline(kw: dict) -> dict:
    return {**kw, "abandoned_for_deadline": True}


ROLLBACK_FROM_041 = (
    Migration.rename(
        old_module=LGE, old_qualname="HandlerAbandonedOnDeadline", to=HandlerRaised
    ),
    Migration.transform_fields(
        module=LGE, qualname="HandlerAbandonedOnDeadline", transform=flag_deadline
    ),
    Migration.rename(
        old_module=LGE, old_qualname="HandlerAbandonedOnSignal", to=HandlerRaised
    ),
    Migration.rename(old_module=LGE, old_qualname="RunPausedOnDeadline", to=RunPaused),
    Migration.rename(old_module=LGE, old_qualname="RunPausedOnSignal", to=RunPaused),
)

serde = NamespaceAwareSerde(ROLLBACK_FROM_041, namespaces=[Orders])
```

On 0.40.0, a `HandlerAbandonedOnDeadline` then revives as `HandlerRaised(abandoned_for_deadline=True)`, and a `HandlerAbandonedOnSignal` revives as a plain `HandlerRaised`. A `RunPausedOnDeadline` and a `RunPausedOnSignal` each revive as `RunPaused`.

## What is NOT migrated

- **Non-Event payloads** (Pydantic models, plain dataclasses, LangGraph `Interrupt` wrappers) — flow through LangGraph's default serde. Events nested inside `Interrupt.value` are migrated automatically. A Pydantic model used as an event field must live at module scope. LangGraph's serde stores it by `__name__` and revives it with `getattr` on the module. On a miss it returns a raw `dict` with no error. `NamespaceAwareSerde` raises `ValueError` at construction for a model nested inside a class or a function ([#167](https://github.com/cadance-io/langgraph-events/issues/167)). A model held behind `Any` or an untyped container is not reachable from an annotation and is not checked.
- **Reducer channel-name renames** — LangGraph channel-routing concern; see [Checkpointer evolution](checkpointer-evolution.md).
- **Payloads `ormsgpack` refuses to encode** — error propagates at the source, no fallback. Subclass `NamespaceAwareSerde` and override `_make_default` to extend encoding.
- **Non-reducer channel values** — no analogous rebuild path. Recovery is custom (read → transform → write back through the saver's put API).

## The migration gate

The gate compares the code with a committed baseline file. It fails in CI before a stored payload fails in production. Declare one `MigrationTarget` for your graph. Every command and every test reads it.

```python
# myapp/lge_target.py
from pathlib import Path

from langgraph_events import EventGraph
from langgraph_events.serde.migrations import MigrationTarget

from myapp.namespaces import Order, Persona


def build_graph(checkpointer):
    return EventGraph.from_namespaces(Order, Persona, checkpointer=checkpointer)


target = MigrationTarget(
    graph=build_graph,
    baseline=Path(__file__).parent / "lge-baseline.json",
)
```

`graph` receives a checkpointer and returns the graph. The gate passes an empty `MemorySaver`, so it reads no store. The graph must carry a `NamespaceAwareSerde`. `from_namespaces(..., checkpointer=...)` wires one. A hand-built serde with `migrations=` also works: set it on the checkpointer before you build the graph.

Run the gate in CI, and in a test:

```bash
python -m langgraph_events.serde.migrations check myapp.lge_target:target
```

```python
from langgraph_events.serde.migrations import assert_migration_gate
from myapp.lge_target import target


def test_migration_gate() -> None:
    assert_migration_gate(target)
```

`assert_migration_gate` raises `GateError`, an `AssertionError`. Its `report` holds the typed findings.

Write the baseline the code needs with one command. Commit the file in the same PR as the change:

```bash
python -m langgraph_events.serde.migrations baseline update myapp.lge_target:target
```

### The rules

`check()` runs the rules in this order. Each finding names one next command, with your own arguments.

| Rule | Finding | It means | Next command |
|---|---|---|---|
| `baseline` | `BaselineStale` | The committed file differs from the file the code needs. The finding lists each difference. | `baseline update` |
| `names` | `AnonymousCallable` | A transform or a split uses a lambda or a callable object. A replacement callable would read as the same migration. | Give the callable a module-level name, then `baseline update` |
| `coverage` | `Uncovered` | A recorded identity has no live class and no migration. The finding names live classes with the same last name, a likely rename. | Add the migration, then `baseline update` |
| `revive` | `Unrevivable` | A synthesized payload of the recorded shape fails the real read path. | Add or fix the migration, then `baseline update` |
| `handlers` | `HandlerMissing` | A recorded handler node name has no live handler and no `@on(previously=...)` alias. | Declare the alias, then `baseline update` |
| `release` | `SurvivedRelease` | The previous release shipped this migration, and the code still holds it. | Delete the migration, then `baseline update --previous` |

The baseline rule runs first. The other rules judge the baseline the code needs, not the committed file. So a deleted shipped migration gives one `BaselineStale` finding, and never a request to add the migration back.

The `revive` rule sends a `None` placeholder for every required field of the live class. It also sends one for every recorded field that the live class no longer accepts. A field that a fill supplies gets the real fill value. So a broken fill fails the rule, and a dropped field fails the rule the way a stored payload fails at read. An identity of this library only has to resolve to a live `Event`. The library proves its own shapes in its own tests.

!!! warning "A required field that `__post_init__` validates"
    A `None` placeholder trips a validator on a required field with no fill. The rule then reports `Unrevivable`. Give the field a `@backfill` with a valid value, or a default.

Resolution is scope-first, like the read path. The rule finds a class in the namespaces of the serde first, then by import. So a namespace defined inside a function passes. `NamespaceAwareSerde.revivable_identities()` returns the identities the `coverage` rule accepts.

### The release rule

A read-time migration ships in release N and leaves the code in release N+1. The `release` rule needs the baseline of the previous release. The library does not run git. Give the file with `--previous`:

```bash
git show "$PREV_TAG:myapp/lge-baseline.json" > "$RUNNER_TEMP/prev.json"
python -m langgraph_events.serde.migrations check myapp.lge_target:target \
    --previous "$RUNNER_TEMP/prev.json"
```

- Without `--previous`, the rule does not run. The report says so. A test usually runs without it.
- A previous file of version 1 to 3 records no migrations. The rule does not run against it, and the report says so.
- A previous key follows a rename of the current code onto the live class. A key whose identity was a rename source in the previous file does not move.
- When you delete a shipped migration, run `baseline update --previous <file>`. The update drops the history that the migration covered. Without `--previous`, the update drops no history.

If the store cannot be made clean in time, keep the migration one more release:

```bash
python -m langgraph_events.serde.migrations baseline hold myapp.lge_target:target \
    "transform myapp.events:Order via myapp.events:drop_total" \
    --reason "a subgraph checkpoint still needs it" --previous "$RUNNER_TEMP/prev.json"
```

The hold records the fingerprint of the previous file. A check against another previous file ignores it, so a hold lasts one release. Every check lists every hold.

`python -m langgraph_events.serde.migrations fingerprint <baseline file>` prints the fingerprint of the migrations a file records. It reads the JSON only.

### Labels

One label form names a migration everywhere: in the baseline, in a finding, in a hold and in a read error.

```
<kind> <module>:<qualname>[<field>] [via <module>:<qualname>]
```

`rename myapp.events:Order.Old`, `fill myapp.events:Order[status]` and `transform myapp.events:Order via myapp.events:drop_total` are examples. A read error names a stored identity as `<module>:<qualname>`.

### Exit codes

| Code | `check` | `baseline update`, `baseline hold`, `fingerprint` |
|---|---|---|
| 0 | No finding | The command is done |
| 1 | One or more findings | Not used |
| 2 | A usage or target error, for example a target that does not import, a graph with no `NamespaceAwareSerde`, or a bad baseline file | The same |

The removed form `python -m langgraph_events.serde.migrations <module:factory> <baseline>` exits 2 and prints the new command line.

### When to commit the baseline

Commit the baseline **alongside** the change that needs it, never after. Run `baseline update` in the same PR as a new migration, a renamed handler or a new field.

#### What the file records

```json
{
  "version": 4,
  "events": [
    {"module": "myapp.orders", "qualname": "Order.Place.Placed", "fields": ["order_id", "tracking"]}
  ],
  "retired": [
    {"module": "myapp.orders", "qualname": "Order.ApprovalRequired", "fields": ["order_id"]}
  ],
  "handlers": [{"name": "handle_approval"}],
  "migrations": [
    {"label": "rename myapp.orders:Order.ApprovalRequired"},
    {"label": "fill myapp.orders:Order.Place.Placed[tracking]",
     "hold": {"reason": "a subgraph checkpoint needs it", "against": "lge-m1-3f0c2a9e41b7d655"}}
  ]
}
```

- **`events`** lists every identity the graph reaches. Each entry records the init fields of its class, the same set the serde writes.
- **`fields` is cumulative.** A field that was ever recorded can sit in a checkpoint, so the `revive` rule keeps sending it. An update keeps it. Only an update with `--previous` after a shipped migration left the code resets the fields of the identity that the migration covered.
- **`retired`** lists every identity an update dropped from `events`, with the fields last recorded for it. The entry has no `fields` key when the last record predates version 3. An identity that is live again leaves `retired`. A retired identity must revive through a migration onto a surviving class or a tombstone.
- **`handlers`** lists the handler node names. An update rebuilds it.
- **`migrations`** lists the label of every client migration the code holds, and any hold. lge's own framework migrations are not in it.
- `events` and `retired` never share an identity. The reader raises `ValueError` on an overlap.
- A file of version 1, 2 or 3 still loads. A file with an unknown version raises `ValueError`. The next `baseline update` writes version 4.

## Testing your migrations

Run [the migration gate](#the-migration-gate) in a test with `assert_migration_gate`. The helpers below prove a specific shape or a specific resume.

### Testing handler recovery

The gate above is static — it proves the alias *exists*. To prove a paused checkpoint actually *resumes* through the rename, `assert_resume_recovers` exercises the real interrupt→resume path (the behavioral handler analog of `revive`):

```python
from langgraph.checkpoint.memory import MemorySaver
from langgraph_events.serde import assert_resume_recovers


def test_rename_keeps_checkpoints_resumable():
    saver = MemorySaver()
    before = EventGraph([await_input, handle_confirm], checkpointer=saver)
    # `gather_input` is `await_input` renamed, with @on(previously="await_input")
    after = EventGraph([gather_input, handle_confirm], checkpointer=saver)

    assert_resume_recovers(
        before, after, seed=Started(...), resume_with=Confirmed(),
    )
```

That one call replaces the hand-rolled dance — *build a checkpointer → invoke `before` to interrupt → assert it paused → rebuild `after` → resume → assert it recovered*. It invokes `before` with `seed` (which must pause via `Interrupted`), resumes `after` with `resume_with` on the same checkpoint, and asserts a `Resumed` was emitted (a silent drop or a `halt` would not), returning the log for further assertions. `before` and `after` must share **one** checkpointer instance.

**When to use which:** the gate's `handlers` rule is the zero-maintenance CI sweep (covers every recorded handler, no per-rename test code); `assert_resume_recovers` is the focused behavioral spot-check for a specific rename you want to prove end-to-end.

### `synthesize_legacy_payload` — pin a specific old field shape

The gate does not assert specific *old field values*. Reach for this only when a field's shape drifted — it builds the bytes a prior release would have written:

```python
import pytest
from langgraph_events.serde import NamespaceAwareSerde, synthesize_legacy_payload
from cadance.namespaces import Persona


@pytest.mark.parametrize(
    "module, qualname, kwargs, expected_cls",
    [
        ("cadance.persona", "Persona.Persisted", {"note": "n"}, Persona.Persist.Persisted),
        # only events whose field shape changed — not every rename
    ],
)
def test_revives_release_N_payloads(module, qualname, kwargs, expected_cls):
    serde = NamespaceAwareSerde(namespaces=(Persona,))
    revived = serde.loads_typed(synthesize_legacy_payload(module, qualname, kwargs))
    assert isinstance(revived, expected_cls)
```

A failing test = dataclass `TypeError` on the missing field — exactly where you want a field-shape regression caught.

### Release N → N+1 walkthrough

1. **At release N**, run `baseline update` and commit the JSON.
2. **On the feature branch**, rename the event and add `@migrate_from("Persona.Persisted")` (plus `@backfill` if it gained a required field). Run `baseline update` and commit the JSON in the same PR.
3. **In CI**, `check` runs on every PR. It catches a dropped migration and a class that no longer constructs. With `--previous`, it also catches a shipped migration that release N+1 must delete. Add `synthesize_legacy_payload` entries only if a field's shape changed.

## Handler renames

Handlers evolve under the **same model as events**. A handler becomes a graph *node* keyed by its name; if a thread was interrupted (via `Interrupted`) inside a handler and you later rename or move it, the old node vanishes and the paused checkpoint can no longer resume. Declare the rename and old checkpoints keep working:

| Concern | Events | Handlers |
|---|---|---|
| Declare prior identity → auto-recover | `@migrate_from("Old.Qualname")` | `@on(previously="old_node")` / `previously: ClassVar = (...)` on a `Command` |
| Prevent the break up front | (qualname *is* the identity) | `@on(node_name="stable_id")` — then rename the function freely |
| Catch the undeclared case in CI | the gate's `coverage` and `revive` rules | the gate's `handlers` rule |

```python
# Renamed handler — declare the old node name so paused checkpoints resume:
@on(Confirmed, previously="await_confirmation")
def confirm(event: Confirmed) -> Approved: ...

# Inline Command nodes have no decorator slot — declare it as a class
# attribute, mirroring raises/invariants:
class Persist(Command):
    previously: ClassVar = ("Persona.Persist", "Story.Persist")
    ...
```

- `@on(node_name=...)` pins a **stable node identity** decoupled from the Python function name — rename/move the function with zero checkpoint impact. `@on(previously=...)` registers an **alias node** for each historic name so an in-flight interrupted checkpoint re-enters the renamed handler.
- **Testing:** prove coverage with [the migration gate](#the-migration-gate) and recovery with `assert_resume_recovers` — see [Testing handler recovery](#testing-handler-recovery).
- **Deploy order:** alias nodes are purely **additive**, so a handler rename is safe to ship in a **single release** — unlike an event rename, which needs the two-release `legacy_write` dance (below). The new release's graph carries both the new node and the alias, so old in-flight checkpoints resume against it.
- **Runtime safety net.** If a handler is *removed* (or renamed without `previously=`) and a thread is still paused inside it, `resume()` would otherwise be a silent no-op. `EventGraph(on_unresumable=...)` governs this — it fires on any `resume()` of a thread that isn't awaiting input (also catching resume of an already-finished thread or a double-resume):
    - **`raise`** (default) — `UnresumableError`. Use in dev/CI to fail fast on an undeclared rename.
    - **`halt`** — emit a terminal `Unresumable(Halted)` and finalize the thread. Use in production for graceful, *observable* degradation (the thread ends in the log rather than hanging).
    - **`warn`** — `UserWarning` + no-op (thread untouched). Use when you want a signal but to handle the thread yourself.

    The gate's `handlers` rule catches undeclared renames *before* deploy; `on_unresumable` is the runtime last-resort net for anything that slips through.

    Retiring an `Interrupted` subclass is a related but separate move — see [Retiring an Interrupted subclass](#retiring-an-interrupted-subclass) below.

### Inline command handlers are keyed by the command qualname

An inline `Command.handle()` handler's node identity is the **command's `__qualname__`** (e.g. `Order.Place`), not the method name — so it is stable and order-independent. Reordering the `handlers=[...]` list is safe, and you do **not** need `@on(node_name=...)` to pin it (that pin is for standalone `@on` functions, whose identity is otherwise the function name).

Renaming the command class therefore *is* a node rename. Declare the historic node names as a `previously` class attribute — inline handlers have no decorator slot. Annotate it `ClassVar` (or leave it un-annotated); a plain annotation would make it an event *field* and is rejected at class creation:

```python
class Persist(Command):
    previously: ClassVar = ("Persona.Persist", "Story.Persist", "save")
    ...
```

The lists often rhyme with a `@migrate_from` stack on the same class, but they answer different questions: `@migrate_from` revives the old payload **bytes**; `previously` revives the old checkpoint **pointer** (the node name in `snapshot.next`). Each may carry entries the other can't — a node-only era (a reactor the command replaced, a pre-#97 `handle_N`) has no serde identity, and a module-only move has no node rename. The gate's `handlers` rule catches a missing `previously`. Its `coverage` and `revive` rules catch a missing `@migrate_from`.

A historic node name identifies exactly one class, and it stays that way: a concrete `Command` may not be subclassed (`class Child(Persist)` is a `TypeError`), so `previously` — like `raises`, `invariants` and `retry` — is always read off the command class that declares it. A second command that took over some of the old node names declares its own list.

!!! note "Upgrading a baseline recorded before this fix (#97)"
    Earlier releases recorded inline command handlers by positional names (`handle`, `handle_2`, …). After upgrading, those names no longer resolve, so the gate reports `HandlerMissing` until you **run `baseline update` once**. This firing is expected — it is the gate doing its job. A checkpoint paused inside an inline command handler *before* the upgrade cannot resume afterward (the old positional name was order-dependent and can't be reconstructed); recover a specific one with `previously: ClassVar = ("handle_N",)` on that command class, or set `on_unresumable="halt"/"warn"`.

### Renaming an event *and* a handler together

The two tracks are independent — do both, in one PR:

1. **Decorate both** — `@migrate_from("Old.Qualname")` on the renamed event class; `@on(previously="old_node")` on the renamed handler.
2. **Update the baseline once** — a single `baseline update` records both the event identities *and* the handler node names. Commit it in the same PR.
3. **Run the gate in CI** — its `coverage`, `revive` and `handlers` rules check both tracks. One rule does not cover the other.
4. **Deploy** — the handler alias is single-release safe; the event rename follows the two-release `legacy_write` cadence, so the *event* side gates the rollout.

!!! warning "Renaming an inline Command is always both renames at once"
    The command class is simultaneously the node identity *and* the event class of the payload sitting in the paused checkpoint. The alias node dispatches by `isinstance` against the **new** class, so `previously` alone is not enough: the checkpointed command payload must also revive *as* the new class — `@migrate_from` on the command plus a namespace-aware serde on the checkpointer (`from_namespaces(..., checkpointer=...)` wires it automatically). With the default LangGraph serializer the alias node re-enters but sees no matching event and the resume silently no-ops — and this bypasses `on_unresumable`: the thread *is* still awaiting input, so the safety net never fires.

## Rewriting the live set

Every migration above is read-side. The stored bytes keep the historic identity and the old field shape until something writes the checkpoint again. A thread at rest is never written again. So its `@migrate_from`, `@transform_fields` or `@split_event` must stay for ever. `graph.plan_rewrite()` and `graph.rewrite_store()` rewrite the stored bytes. Closes [#179](https://github.com/cadance-io/langgraph-events/issues/179).

The live set is the latest checkpoint of each thread in the root namespace, plus its pending writes. `rewrite_store()` reads each one through the checkpointer's serde, so every rename, transform, split and fill applies. It writes the result back through the checkpointer's `put()`, under the same checkpoint id, with a new version for each channel the rewrite touched. `drop=` names event classes whose stored instances leave the `events` and `_pending` channels. `plan_rewrite()` does the same reads and the same verification, and writes nothing.

The caller names the threads with `thread_ids=`. The argument is required, and the library does not walk the store. To rewrite every thread, get the ids from one `SELECT DISTINCT thread_id FROM checkpoints WHERE checkpoint_ns = ''` query. If one checkpointer stores more than one graph, keep only the thread ids of this graph. See [Finding candidates server-side](#finding-candidates-server-side). A listed id with no checkpoint is reported as refused, so a typo stays visible.

```python
report = graph.plan_rewrite(drop=(Order.ApprovalRequired,), thread_ids=thread_ids)
print(report)
assert not report.refused

report = graph.rewrite_store(drop=(Order.ApprovalRequired,), thread_ids=thread_ids)
assert not report.refused
```

Both return a `RewriteReport` with one `ThreadRewrite` per thread. `status` is `"rewrite"`, `"unchanged"` or `"refused"`. `report.applied` says whether the rewrite ran: `"rewrite"` in a plan means the thread needs a write, in an applied report it means the thread was written. `migrated` lists each `(stored, live)` identity pair the table rewrote. `dropped` counts the log entries `drop=` removed, per identity. `reason` says what to do with a refused thread. `str(report)` prints one summary line, then one line per refused thread. The async twins are `aplan_rewrite()` and `arewrite_store()`.

Before the write, each rewritten value is encoded the way `put()` encodes it, scanned for a dropped or historic identity, and revived under the strict serde. One failure refuses the thread, and nothing is written for it. A thread is also refused when:

- Its history names an identity the serde cannot revive. Add a [tombstone](#recovering-a-delete-first-deployment) first.
- It is paused on a dropped class. `abandon()` it first.
- A dropped event is pending dispatch on a paused thread. After the drop, the resumed task finds no event to handle and ends the thread with the answer lost. Resume or abandon the thread first.
- It holds a completed task write from a fanned-out sibling. The checkpointer API cannot rewrite that write. Resume or abandon the thread first.
- It holds a pending `__error__` write from a failed task. Run the thread again, or abandon it.
- A dropped instance sits inside a field of another event, or in a reducer value. Leave the class in place, or settle the thread.
- A run advanced the thread between the read and the write. Rerun.

An answered interrupt also sits in `Resumed.interrupted`. When that interrupt is dropped, the rewrite sets the field to `None`. The `Resumed` event stays, and `dropped` does not count the cleared field. `drop=` matches the stored identity exactly. A subclass instance stays in place.

!!! warning "Run it while the graph is idle, never inside a rolling deploy window"
    An old pod cannot read the new bytes. The rewrite also toggles per-instance state on the serde, like `tolerate_unresolved()`, so a concurrent run through the same serde instance corrupts the report. Take a store-level backup first. The previous blob versions stay in the store, but the checkpoint row is overwritten in place.

What the rewrite does not do:

- A historic checkpoint keeps the old bytes. After the class is deleted, `get_state_history()` cannot read it.
- `drop=` removes the stored event only. Reducer state is not recomputed. Use [`replay_reducer`](#recovering-with-replay_reducer).
- An event inside a non-Event container field, a dataclass or a pydantic model, has no migratable identity ([#160](https://github.com/cadance-io/langgraph-events/issues/160)). The rewrite does not reach it.
- A checkpoint under a subgraph namespace is not read, the same as `unrevivable_threads()`.
- `legacy_write=True` is refused with `ValueError`. Under `legacy_write` the rewrite stores the historic identity again.

## Retiring an Interrupted subclass

!!! warning "`threads_paused_on()` and `abandon()` cover paused threads only"
    A thread that already *answered* the interrupt holds the retired class in its **settled** history, not in a pending write. `threads_paused_on()` does not find such a thread, and `abandon()` does not touch it. Reading its history after the class is deleted raises `Cannot revive`. `graph.unrevivable_threads(thread_ids=...)` is the sweep that finds it: it reads the latest checkpoint of each listed thread from the store and reports each identity that no longer revives, settled or pending. Pass every thread id in the store, from one `SELECT DISTINCT thread_id FROM checkpoints WHERE checkpoint_ns = ''` query. If one checkpointer stores more than one graph, keep only the thread ids of this graph. Run it after the class is deleted, against the real store, and treat a non-empty result as a thread that needs the [recovery path](#recovering-a-delete-first-deployment) below. The field-shape half of [#159](https://github.com/cadance-io/langgraph-events/issues/159) is covered by [Dropping, merging or retyping a field](#dropping-merging-or-retyping-a-field).

To retire an `Interrupted` subclass, delete it from the codebase once no live checkpoint still references it. `graph.abandon(config)` / `.aabandon()` settles one paused thread without answering it — see [Ending a pause without answering it](control-flow.md#ending-a-pause-without-answering-it-abandon).

`abandon()` settles one thread per call. `graph.threads_paused_on(EventClass, thread_ids=...)` (or `athreads_paused_on()`) finds the paused threads among the ids you pass. Get the candidate ids from a server-side query. See [Finding candidates server-side](#finding-candidates-server-side) below.

`threads_paused_on()` and `abandon()` read each thread's checkpoint directly, not the graph's compiled topology. Two deletions this survives, with different outcomes:

- The **handler** that produced the interrupt is already removed from the graph. The class still imports, so the interrupt revives normally and a class filter still matches it. This is the common order: retiring an `Interrupted` usually retires the handler that produced it first.
- The **class** itself is already deleted and no longer imports. The normal habit is to ship the class deletion in the same release as the handler's, and this section used to train that habit. The interrupt cannot revive. With no filter, `threads_paused_on()` still returns the thread. With a class filter, it matches nothing: a class filter can never match an identity with no class. `abandon()` still settles the thread, recording the interrupt's last-known qualname in `discarded` instead of a live instance.

`Cannot revive` states the fix directly: settle the thread with `abandon()`/`aabandon()` before deleting the class, or map the dead identity onto a tombstone with `@migrate_from` — see [Recovering a delete-first deployment](#recovering-a-delete-first-deployment) below.

### Finding candidates server-side

Every store sweep takes its thread ids from the caller. This applies to `threads_paused_on()`, `unrevivable_threads()`, `plan_rewrite()`, `rewrite_store()` and their async twins. `thread_ids=` is required. The library does not walk the store. A server-side query is the way to find the ids.

The checkpointer API has no cheap read for the thread ids in a store. It also has no read for "the latest checkpoint of each thread". `list()` with no thread filter returns every checkpoint, historic versions included, and the serde deserializes each one. The Postgres saver also fetches the whole result before the serde runs. On one Postgres store with 60 threads and 15,000 historic checkpoints, that walk did not complete. See [#180](https://github.com/cadance-io/langgraph-events/issues/180). On a store with 149,336 checkpoints in 4,938 threads, the query returned 3.7 GB. One process grew to 9 GB and made a 16 GB machine unresponsive. Up to v0.33.0, the library did this walk when `thread_ids=` was not given. The walk is removed.

Only the store can filter by latest checkpoint and by the `__interrupt__` channel. `thread_ids=` accepts ids from any query, any driver, sync or async.

To sweep every thread, get the ids from one query. The library reads the root checkpoint namespace only, so the query filters on `checkpoint_ns = ''`. The same SQL works on `PostgresSaver` and `SqliteSaver`. If one checkpointer stores more than one graph, keep only the thread ids of this graph.

```sql
SELECT DISTINCT thread_id FROM checkpoints WHERE checkpoint_ns = ''
```

For `InMemorySaver`, read the ids from its storage:

```python
ids = [tid for tid, ns in saver.storage.items() if ns.get("")]
```

The `if ns.get("")` filter is necessary. `InMemorySaver.get_tuple()` on an unknown thread id leaves an empty `storage[thread_id][""]` entry. Without the filter, such an id joins the list, although it has no checkpoint.

To find the threads paused on one class, filter the candidates server-side, then pass the ids.

The Postgres query below reads the `__interrupt__` writes of each thread's latest root checkpoint. It matches the retired qualname as bytes inside the msgpack blob:

```sql
WITH latest AS (
  SELECT DISTINCT ON (thread_id) thread_id, checkpoint_id
  FROM checkpoints
  WHERE checkpoint_ns = ''
  ORDER BY thread_id, checkpoint_id DESC
)
SELECT DISTINCT w.thread_id
FROM checkpoint_writes w
JOIN latest l
  ON w.thread_id = l.thread_id AND w.checkpoint_id = l.checkpoint_id
WHERE w.checkpoint_ns = ''
  AND w.channel = '__interrupt__'
  AND w.type = 'msgpack'
  AND position(convert_to('Order.ApprovalRequired', 'UTF8') in w.blob) > 0
```

The same query for `SqliteSaver`, with the qualname as the bound parameter:

```sql
SELECT DISTINCT w.thread_id
FROM writes w
JOIN (
  SELECT thread_id, MAX(checkpoint_id) AS checkpoint_id
  FROM checkpoints WHERE checkpoint_ns = '' GROUP BY thread_id
) l ON w.thread_id = l.thread_id AND w.checkpoint_id = l.checkpoint_id
WHERE w.checkpoint_ns = ''
  AND w.channel = '__interrupt__'
  AND instr(w.value, CAST(? AS BLOB)) > 0
```

The byte match is a candidate filter only. It also matches a longer qualname, for example `Order.ApprovalRequiredV2`. The library then confirms each candidate through the serde:

```python
paused = await graph.athreads_paused_on(
    Order.ApprovalRequired, thread_ids=candidates
)
for config in paused:
    await graph.aabandon(config, reason="retiring Order.ApprovalRequired")
```

For `unrevivable_threads()`, `plan_rewrite()` and `rewrite_store()`, pass every thread id from the `SELECT DISTINCT thread_id FROM checkpoints WHERE checkpoint_ns = ''` query.

To check one thread, pass its id alone: `graph.threads_paused_on(Order.ApprovalRequired, thread_ids=[tid])`. Do not use `graph.get_state(config)` for this check. Once the handler is deleted, `GraphState.is_interrupted` is `False` and `GraphState.interrupted` is `None` on a thread that is still paused. See the warning under [Sequence](#sequence).

### Sequence

1. Get every thread id in the store from one `SELECT DISTINCT thread_id FROM checkpoints WHERE checkpoint_ns = ''` query. See [Finding candidates server-side](#finding-candidates-server-side). The steps below call this list `thread_ids`. An empty list proves nothing. Check that the query returned ids.
2. Enumerate every thread paused on the class with `graph.threads_paused_on(EventClass, thread_ids=thread_ids)`. Call `graph.abandon(config)` (or `.aabandon()`) on each thread returned.
3. Verify: `graph.threads_paused_on(EventClass, thread_ids=thread_ids) == []`.
4. Plan the rewrite: `report = graph.plan_rewrite(drop=(EventClass,), thread_ids=thread_ids)`. Review it. Verify: `not report.refused`. An *answered* thread holds the class in its settled history. `threads_paused_on()` and `abandon()` never reach such a thread. The plan does.
5. Take a store backup. Outside a rolling deploy window, while the graph is idle: `report = graph.rewrite_store(drop=(EventClass,), thread_ids=thread_ids)`. Verify: `not report.refused`. See [Rewriting the live set](#rewriting-the-live-set).
6. Delete the class from the codebase. Run `baseline update`. The retired identity moves to the baseline's `retired` list. Deploy.
7. After the deploy, query the thread ids again, then verify against the real store: `graph.unrevivable_threads(thread_ids=thread_ids) == {}`. Then delete the `retired` entry from the baseline file.

```python
# conn: a psycopg or sqlite3 connection to the checkpointer database.
ALL_THREAD_IDS = "SELECT DISTINCT thread_id FROM checkpoints WHERE checkpoint_ns = ''"

# Every thread id in the store, from one server-side query.
thread_ids = [row[0] for row in conn.execute(ALL_THREAD_IDS)]
assert thread_ids, "the query returned no thread ids"

for config in graph.threads_paused_on(EventClass, thread_ids=thread_ids):
    graph.abandon(config, reason="retiring EventClass")
assert graph.threads_paused_on(EventClass, thread_ids=thread_ids) == []

report = graph.plan_rewrite(drop=(EventClass,), thread_ids=thread_ids)
print(report)
assert not report.refused
report = graph.rewrite_store(drop=(EventClass,), thread_ids=thread_ids)
assert not report.refused

# After the class is deleted. Reads the store, not the baseline, so a
# stale name in your own code cannot make it report "safe".
thread_ids = [row[0] for row in conn.execute(ALL_THREAD_IDS)]
assert thread_ids, "the query returned no thread ids"
assert graph.unrevivable_threads(thread_ids=thread_ids) == {}
```

`unrevivable_threads()` reports nothing until the class is gone: while the class still imports, every thread revives. Run it once, after step 6. A non-empty result maps each thread id to the qualnames it can no longer revive: a thread the rewrite refused, or one that was written after it. Recover each one with a [tombstone](#recovering-a-delete-first-deployment). It reports an identity wherever the serde met it: in the settled history, in a pending interrupt, in a completed sibling write, or nested in a field of a live event. It needs a `NamespaceAwareSerde` on the checkpointer and raises `ValueError` otherwise.

!!! warning "Do not `abandon()` a thread that `unrevivable_threads()` reports"
    `abandon(config, require_interrupt=False)` would re-serialize that thread's settled history with the placeholder in it. After that, a strict read would return the placeholder in the log with no error. The code refuses. `NamespaceAwareSerde` never stores a placeholder: a write that holds one raises `ValueError` naming the identity. `abandon()`/`aabandon()` refuse first, with a clearer message. They raise `ValueError` naming the thread and the qualnames when the settled history holds a deleted class. They settle only a thread whose sole unrevivable identity is its pending interrupt. The recovery for a settled thread is the [tombstone](#recovering-a-delete-first-deployment) below, not `abandon()`.

!!! warning "Step 3 is not `assert not graph.get_state(config).is_interrupted`"
    Once the handler is deleted (step 4, or already done, which is the common order), `get_state()`'s `is_interrupted` reads the graph's compiled topology and is `False` on a thread that is *still paused*: the check would pass without proving anything. `threads_paused_on()` reads the checkpoint directly and stays accurate regardless of which handlers this graph still registers.

### What the baseline write records

Deleting the class drops an identity from the graph's topology. `baseline update` moves it to the baseline's `retired` list, with the fields last recorded for it. The [migration gate](#the-migration-gate) keeps checking a retired identity. Its `coverage` and `revive` rules fail on it until a migration covers it. That is the gate doing its job. Once `unrevivable_threads()` reports nothing and every paused thread is settled, delete the `retired` entry by hand. From then on, the gate does not check the retired identity.

!!! warning "After the hand edit, the gate cannot see a remaining breakage"
    The gate reads the baseline. Once the `retired` entry is gone, every rule passes whether or not a settled thread out there still cannot revive. `graph.unrevivable_threads()` is the only gate that still sees it, because it reads the store and not the baseline. Keep it in the retirement checklist, step 7 above, and run it against the real store before the hand edit.

!!! warning "Expect the `handlers` rule to fail first"
    Retiring an `Interrupted` usually retires the handler that produced it too. Deleting both together trips the `handlers` rule (`HandlerMissing`) in the same way the `coverage` rule trips: this is the gate doing its job, not a new problem. The same `baseline update` records both the event identity and the handler node in one write, so no separate step is needed. A handler name is not retired. The update drops it from `handlers` at once.

### Recovering a delete-first deployment

If the class was deleted before every paused thread was settled, `threads_paused_on()` and `abandon()` already recover the **paused** case on their own — see the two-deletions note above. A thread that had already **answered** the interrupt (the #159 case) needs the fix below too.

Map the dead identity onto a tombstone class, in a follow-up release. The tombstone does not need the retired fields: `transform=lambda kw: {}` discards every stored field before the rename, so an empty tombstone revives whatever `Order.ApprovalRequired` carried. **Keep the tombstone an `Interrupted` when the retired class was one.** `threads_paused_on()` and `abandon()` rely on it. **Nest the tombstone inside the same `Namespace`.** A domain still has other live members `EventGraph.from_namespaces(...)` wires up, and the namespace walk that collects those also collects the tombstone nested beside them:

```python
from langgraph_events import Command, DomainEvent, Interrupted, Namespace, on
from langgraph_events.serde import migrate_from


class Order(Namespace):
    class Approve(Command):
        class Approved(DomainEvent):
            pass

        def handle(self) -> "Order.Approve.Approved":
            return Order.Approve.Approved()

    @migrate_from("Order.ApprovalRequired", transform=lambda kw: {})
    class RetiredApprovalGate(Interrupted):
        pass


@on(ApprovalSubmitted)
def handle_approval(event: ApprovalSubmitted) -> Order.Approve:
    return Order.Approve()


graph = EventGraph.from_namespaces(
    Order, handlers=[handle_approval], checkpointer=MemorySaver()
)
```

Run this against a store that still holds a thread paused on `Order.ApprovalRequired` and an already-answered one. `graph.threads_paused_on()` lists the paused thread again, matched against the live `RetiredApprovalGate`, not a degraded identity. `graph.abandon(config)` settles it, recording `discarded="Order.RetiredApprovalGate"`. The already-answered thread's history revives too, closing the #159 gap for this one identity without waiting on the library-level fix. Verified end to end, across a real process restart against persisted checkpoint bytes (not just an in-process object), before this recipe was published.

`in_module=` defaults to the decorated class's `__module__`. Pass it explicitly if `Order.ApprovalRequired` lived in a different module than `RetiredApprovalGate` does.

**Deleting the tombstone.** The tombstone is a live class, so `drop=` reaches it. Once every paused thread is settled, run steps 4 and 5 of the [sequence](#sequence) with `drop=(Order.RetiredApprovalGate,)`. Verify `graph.unrevivable_threads() == {}`. Then delete the tombstone, its `@migrate_from` and the baseline's `retired` entry in the next release.

**Onto a sibling instead of a tombstone.** When a live class already stands for the retired one, and every one of its fields has a default, put the same decorator on the sibling. The stored fields are discarded, and the sibling constructs from its defaults. Do not map a stored field onto a sibling field that means something else, for example a message id onto an order id:

```python
class Order(Namespace):
    @migrate_from("Order.ApprovalRequired", transform=lambda kw: {})
    class ApprovalDismissed(DomainEvent):
        id: str = field(default_factory=lambda: uuid4().hex)
```

!!! note "The checkpointer must not already carry a `NamespaceAwareSerde`"
    `from_namespaces(...)` only builds a `NamespaceAwareSerde` when `checkpointer.serde` is not already one: the deliberate opt-out for a hand-supplied serde (see [`api.md`](api.md)). Reuse the same checkpointer *object* across an earlier graph built in this process and this recovery graph, and the auto-wiring silently does nothing: no error, no warning, the tombstone never enters scope. Give the recovery graph a fresh checkpointer object (even against the same underlying store), or build the serde yourself, as in the alternative below.

!!! note "This relies on `Order` still being in play"
    `from_namespaces(...)` only wires a namespace into the auto-collected serde because some handler in `handlers=` still subscribes to or produces something inside it (`Order.Approve` above). Nesting the tombstone alone does not add `Order` to that set. If the *whole* namespace is retired too, or the tombstone has to live at module scope, hand-build the serde instead — see below.

**Alternative: a module-level tombstone, with a hand-built serde.** Use this when the namespace itself has nothing else live, or the tombstone genuinely does not belong inside a `Namespace`. `from_namespaces(...)` has no way to reach a module-level class. Pass it through `events=` on a `NamespaceAwareSerde` you build yourself. Same construction as the nested form otherwise. `Order.Approve` still needs to be in `handlers=` explicitly here, since without `from_namespaces(...)`'s namespace walk nothing else registers its inline `handle()`:

```python
from langgraph_events import EventGraph, Interrupted
from langgraph_events.serde import NamespaceAwareSerde, migrate_from


@migrate_from("Order.ApprovalRequired", transform=lambda kw: {})
class RetiredApprovalGate(Interrupted):
    pass


checkpointer.serde = NamespaceAwareSerde(
    namespaces=(Order,),
    events=(Started, ApprovalSubmitted, RetiredApprovalGate),
)
graph = EventGraph([Order.Approve, handle_approval], checkpointer=checkpointer)
```

Recovers identically to the nested form above. Verified the same way, end to end across a real restart. `events=` must still list every loose event class the graph touches (`Started`, `ApprovalSubmitted`, …), same as any hand-built `NamespaceAwareSerde`. The tombstone is one more entry, not a special case.

## Reserved attributes

Library-private; read directly if introspection needed (neither is MRO-inherited):

- `__lge_migrate_from__` — set by `@migrate_from`; tuple of `(module, qualname)` pairs, oldest first.
- `__lge_backfill__` — set by `@backfill`; accumulated field/default entries.
- `__lge_origin_backfill__` — set by `@migrate_from(backfill=...)`; accumulated `((module, qualname), {field: default})` entries.
- `__lge_transform__` — set by `@transform_fields`; the one transform callable.
- `__lge_origin_transform__` — set by `@migrate_from(transform=...)`; accumulated `((module, qualname), transform)` entries.
- `__lge_split__` — set by `@split_event`; the one `(select, targets)` pair.

## Validation guarantees

Errors raised at serde construction (not at first production read):

- Duplicate `old_*` keys (ambiguous rewrites) — `ValueError`
- Dead-end chains (migration target doesn't resolve to a live class, by serde scope or import) — `ValueError`
- `old_qualname` shadowing a currently-live class — `ValueError`
- Cycles (`A→B` then `B→A`) — `ValueError`
- `AddField` targets that neither resolve to a live class nor match a rename-covered historic identity — `ValueError`
- A fill naming a field the live class doesn't have — `ValueError` (typo caught at construction, not at first read)
- `AddField(default=<mutable>)` — `ValueError` (steers to `default_factory`); `@backfill` funnels into `AddField`, same guard
- Two fills on the same `(identity, field)` pair — `ValueError` naming both migrations
- `legacy_write=True` combined with origin-scoped fills — `ValueError` (consolidations cannot ride legacy writes)
- `TransformFields` targets that neither resolve to a live class nor match a rename-covered historic identity — `ValueError`
- Two transforms on the same identity — `ValueError` naming both migrations
- `legacy_write=True` combined with any transform — `ValueError` (a transform has no inverse)
- `migrate_from(transform=...)` with a multi-qualname chain — `ValueError` at decoration
- A `SplitEvent` source that is a rename source, or does not resolve to a live class — `ValueError`
- A `SplitEvent` target that is not an `Event` subclass, or does not resolve to that class by scope or import — `ValueError`
- `SplitEvent` with empty `targets` — `ValueError`. A `select` that is not callable — `TypeError`
- Two splits on the same identity — `ValueError` naming both migrations. A second `@split_event` on one class — `ValueError` at decoration
- `legacy_write=True` combined with any split — `ValueError` (a split has no inverse)
- `migrate_from(backfill=...)` with a multi-qualname chain, an empty dict, or a mutable value (steers to the `Migration.add_field` escape hatch) — `ValueError` at decoration, even earlier
- A duplicated origin qualname across stacked `@migrate_from` decorators (or within one multi-arg call) — `ValueError` at decoration
- Unknown `Operation` type in `Migration.operations` — `TypeError`

A misspelled `@migrate_from("Persona.Persistedd")` fails at construction, not at first checkpoint load in production.
