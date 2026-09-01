# Event migrations

See also: [Checkpointer & graph evolution](checkpointer-evolution.md).

`NamespaceAwareSerde` keys event identity by `(__module__, __qualname__)` so nested events with colliding leaf names round-trip distinctly. Migrations rewrite historic identities on read while preserving wire format — payloads from any prior library version remain readable both ways.

## Two tracks, one model

Renaming an **event class** and renaming a **handler** are the same problem at two layers, solved the same way — *declare the prior identity → auto-recover; a CI gate → catch the undeclared case*:

| | Event class renamed/moved | Handler renamed/moved |
|---|---|---|
| **Declare** (recover after the fact) | `@migrate_from("Old.Qualname")` | `@on(previously="old_node")` |
| **Prevent** (up front) | (qualname *is* the identity) | `@on(node_name="stable_id")` — then rename the function freely |
| **Catch** in CI | `assert_all_baselined_cover` / `_resolve` / `_revive` | `assert_all_baselined_handlers_cover` |

Event-class migrations come first below; the [handler track](#handler-renames) is at the end. **Mind the gate signatures: the event gates take the `serde`; the handler gate takes the `graph` — don't transpose them.**

!!! note "The one invariant"
    **Every rename = decorator *and* `write_baseline` regen in the same PR.** Adding the decorator without regenerating the baseline (or vice-versa) is a latent bug the gate won't catch until later.

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
3. **Regenerate the baseline** (`write_baseline`) in the same PR — the [one invariant](#two-tracks-one-model).

Semantics:

- **Precedence:** explicit payload value > origin-scoped fill > class-global `@backfill` (which acts as the fallback for origins without a scoped entry).
- **Exact-origin contract:** a fill applies only to payloads written under *that* qualname. On a temporal chain (`@migrate_from("A", "B")`) a fill keyed on `B` does **not** apply to `A`-era payloads — "every era" is class-global `@backfill`'s job. Accordingly, `backfill=` requires exactly one qualname per decorator; the multi-arg chain form rejects it. Note the [revive gate](#coverage-gates) cannot catch a mid-chain fill that should have been class-global: earlier eras get placeholder-filled and pass, while a real payload from that era fails at read.
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

Pass the live class for refactor safety; strings only for cross-module cases where the class can't be imported at authoring time. `name` is optional everywhere. Raw `RenameEvent` / `AddField` are imported from `langgraph_events.serde.migrations` (not re-exported at `langgraph_events.serde`).

## Rolling deploys

!!! warning "Rolling deploys require two releases"
    New pods writing under a new qualname produce payloads old pods can't revive. Ship the migration in **two releases**:

    **Release N+1: `legacy_write=True`**
    ```python
    serde = NamespaceAwareSerde(namespaces=NAMESPACES, legacy_write=True)
    ```
    New pods encode events under the oldest historic qualname (recorded by `@migrate_from`). Old pods (release N) read those via existing class defs. Both pod versions can resume each other's threads.

    **Release N+2: `legacy_write=False` (default)**
    Once release N is fully drained, flip writes to current qualname. Keep `@migrate_from` — it covers remaining old-format payloads in storage. Drop the decorator only after every old payload has been touched by new code.

    `legacy_write` is scope-symmetric: decorated classes outside `namespaces=` are encoded under their current qualname (otherwise the read path of the same serde could not migrate them). Keep `namespaces=` consistent between encode and decode pods.

### Concurrency guarantees

No locks or transactions in the serde/migration layer. Safe by **idempotency** and **by construction**:

- **Concurrent reads are safe** — rewrite is pure, no write-back; rename table is a transitive closure.
- **Old and new code cannot share a process** — a `@migrate_from` whose historic identity still resolves to a live class is rejected at serde construction (the rolling-deploy model is structurally enforced).
- **`legacy_write=True` is coexistence, not a lock** — format compatibility for the two-release window.
- **Required-field addition is a two-release operation, like a rename** — pair with `@backfill` and ship over the N → N+1 cadence.
- **Thread-level concurrency on a single `thread_id` is the checkpointer's job** (`MemorySaver` provides none; SQLite/Postgres savers bring their own).
- **Recovery replay is idempotent** — `replay_reducer` overwrites with the same correct value from any number of concurrent runners.
- **One unprotected spot: `write_baseline` is non-atomic.** Sequential divergent writers are caught by the regression guard (the second raises `BaselineRegressionError`), but a true within-call read→write interleave between two CI processes is a TOCTOU the library does not guard. It is a dev/CI tool, not a runtime path — generate and commit the baseline from a **single** CI job, never in parallel.

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

## What is NOT migrated

- **Non-Event payloads** (Pydantic models, plain dataclasses, LangGraph `Interrupt` wrappers) — flow through LangGraph's default serde. Events nested inside `Interrupt.value` are migrated automatically.
- **Reducer channel-name renames** — LangGraph channel-routing concern; see [Checkpointer evolution](checkpointer-evolution.md).
- **Payloads `ormsgpack` refuses to encode** — error propagates at the source, no fallback. Subclass `NamespaceAwareSerde` and override `_make_default` to extend encoding.
- **Non-reducer channel values** — no analogous rebuild path. Recovery is custom (read → transform → write back through the saver's put API).

## Detection tooling

`detect_changes` diffs current graph topology against a stored baseline:

```python
from langgraph_events.serde.migrations.detect import detect_changes, write_baseline

# After authoring the initial migrations, snapshot:
write_baseline(graph, Path("migrations/baseline.json"))

# In a pre-commit hook:
report = detect_changes(graph, Path("migrations/baseline.json"))
if report.has_changes():
    for rename in report.confident_renames:
        print(f"Likely rename: {rename.old_qualname} → {rename.new_qualname}")
    for ambiguous in report.ambiguous:
        print(f"Ambiguous removal {ambiguous.removed}: {ambiguous.candidates}")
    for removed in report.unmatched_removed:
        print(f"Removed (no candidate match): {removed}")
    raise SystemExit(1)
```

- **Suggestion engine, not applicator** — matches removals to additions by leaf name; multi-match → `ambiguous`; pure deletes → `unmatched_removed`. Never auto-edits.
- **One-line CI gate** for the common case:
  ```bash
  python -m langgraph_events.serde.migrations myapp.graph:build migrations/baseline.json
  ```
  Exits `0` (match), `1` (diverge), `2` (usage error). Drop to the programmatic form only for custom reporters.

### When to commit the baseline

Commit the baseline **alongside** the migration that covers the change — never after. Enforced: `write_baseline` raises `BaselineRegressionError` (`.removed` lists dropped identities) if the new snapshot would drop identities the old baseline recorded.

Workflow:

1. Open the branch that contains the rename.
2. Author the migration (`@migrate_from` / `@backfill` on the surviving class, or hand-authored `Migration`).
3. Run `write_baseline(graph, "migrations/baseline.json")` and commit the regenerated JSON in the same PR.

For intentional deletes (no replacement), pass `allow_removed=True`. The guard compares baseline ↔ topology only; *coverage* (does a migration exist?) is the [coverage gates](#coverage-gates)' job. The baseline file is versioned — a stale snapshot raises `ValueError`.

## Testing your migrations

### Coverage gates

Three free functions assert that every identity in a committed baseline still holds up, at increasing strictness. All share one signature — `gate(serde, baseline_path)` — and raise `AssertionError`:

| Gate | Per identity it… | Constructs? | Scope |
|---|---|---|---|
| `assert_all_baselined_cover` | is in `revivable_identities()` (set membership) | no | namespace-walk ∪ `events=` ∪ rename table |
| `assert_all_baselined_resolve` | resolves to a live `Event` (rename-aware) | no | every identity in the baseline |
| `assert_all_baselined_revive` | revives through the real read path | yes | every identity in the baseline |

```python
from pathlib import Path
from langgraph_events.serde import (
    NamespaceAwareSerde,
    assert_all_baselined_cover,
    assert_all_baselined_resolve,
    assert_all_baselined_revive,
)
from cadance.namespaces import Persona

BASELINE = Path(__file__).parent / "migrations" / "baseline.json"


def test_baseline_coverage():
    serde = NamespaceAwareSerde(namespaces=(Persona,))
    assert_all_baselined_revive(serde, BASELINE)  # or _resolve / _cover
```

**Which one?**

- **`revive`** — the default, strongest gate. Proves reachability *and* constructability; fills required fields with placeholders — except fields the migration table back-fills, which get the *real* injected value so a broken fill fails the gate. A new `@migrate_from`/`@backfill` + regenerated baseline is covered with no new test code.
- **`resolve`** — when the baseline contains events `revive` can't placeholder-construct: construction-time validation (`__post_init__`) on non-back-filled fields, framework `SystemEvents`, or module-level `IntegrationEvents`. Proves the identity still resolves without ever calling `__init__`/`__post_init__`, so a full-graph baseline passes with no filtering and still fails loudly on an uncovered rename/removal.
- **`cover`** — the fast set-membership smoke check. Namespace-walk-scoped, so it misses module-level identities a full-graph baseline emits — use `resolve` for those. Raises `MigrationCoverageError` (an `AssertionError`) whose `.uncovered` lists the offending identities.

`resolve` and `revive` answer the same question the read path answers, and answer it the same way. Identity resolution is **scope-first**: the serde looks the `(module, qualname)` up in the namespaces it was constructed with, then falls back to importing the module and walking the qualname. So `resolve` and `revive` pass for a namespace defined inside a function (whose `<locals>` qualname no import reaches) and stay pinned to the serde's own classes when a second engine lifetime redefines them. `cover` is unaffected — it never resolved anything, being a set-membership check against the namespace walk, which is why it still misses module-level identities that `resolve` and the read path handle fine.

`NamespaceAwareSerde.revivable_identities()` returns the read-only `frozenset` of revivable `(module, qualname)` for custom coverage rules (`AddField` targets add no extra identities — post-rename fills key on live classes, origin-scoped fills on rename sources already in the set).

!!! tip "Which serde / graph do I pass the gate?"
    In tests, construct a **standalone** `NamespaceAwareSerde(namespaces=(...), events=(...))` for the event gates — it mirrors what `from_namespaces(..., checkpointer=...)` auto-wires; the gate does **not** need the graph's internal serde instance. The **handler** gate is different: it takes the `graph` (`assert_all_baselined_handlers_cover(graph, BASELINE)`), because handler identity is a graph-topology concern, not a serde one. Pass `events=` too: the auto-wired serde now carries the graph's `IntegrationEvent`s and `SystemEvent`s, so a hand-built one that omits them will fail `cover` on a baseline that includes those identities.

### Handler coverage gate

[Handler migrations](#handler-renames) have their own gate, here alongside the event gates — same job, one layer up (the graph *nodes* a checkpoint pauses at):

```python
from langgraph_events.serde import assert_all_baselined_handlers_cover


def test_handler_coverage():
    assert_all_baselined_handlers_cover(build_graph(), BASELINE)
```

- Asserts every handler node name in the baseline is still a **live node** or covered by an `@on(previously=...)` alias — the static analog of event `cover`. Raises `HandlerCoverageError` (a `CoverageError`/`AssertionError`; `except CoverageError` catches the event gates too).
- **Signature:** event gates take the `serde`; the handler gate (and `assert_resume_recovers` below) take the **graph**. Don't transpose them.
- **One baseline covers both tracks.** A single `write_baseline(graph, BASELINE)` records event identities *and* handler node names (baseline v2; pre-v2 baselines still load with an empty handler set) — regenerate once, run both gates against it.

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

**When to use which:** `assert_all_baselined_handlers_cover` is the zero-maintenance CI sweep (covers every baselined handler, no per-rename test code); `assert_resume_recovers` is the focused behavioral spot-check for a specific rename you want to prove end-to-end.

### `synthesize_legacy_payload` — pin a specific old field shape

The gates above don't assert specific *old field values*. Reach for this only when a field's shape drifted — it builds the bytes a prior release would have written:

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

1. **At release N**, run `write_baseline(graph, BASELINE)` once and commit the JSON.
2. **On the feature branch**, rename the event and add `@migrate_from("Persona.Persisted")` (plus `@backfill` if it gained a required field). The loop gate already covers the historic identity.
3. **In CI**, the loop gate runs on every PR — catches accidentally-dropped migrations and classes that no longer construct. Add `synthesize_legacy_payload` entries only if a field's shape changed.
4. **At release N+1 cutover**, re-run `write_baseline(graph, BASELINE)` in the same PR as the rename. Next removals are measured against the new baseline.

## Handler renames

Handlers evolve under the **same model as events**. A handler becomes a graph *node* keyed by its name; if a thread was interrupted (via `Interrupted`) inside a handler and you later rename or move it, the old node vanishes and the paused checkpoint can no longer resume. Declare the rename and old checkpoints keep working:

| Concern | Events | Handlers |
|---|---|---|
| Declare prior identity → auto-recover | `@migrate_from("Old.Qualname")` | `@on(previously="old_node")` / `previously: ClassVar = (...)` on a `Command` |
| Prevent the break up front | (qualname *is* the identity) | `@on(node_name="stable_id")` — then rename the function freely |
| Catch the undeclared case in CI | `assert_all_baselined_*` | `assert_all_baselined_handlers_cover(graph, BASELINE)` |

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
- **Testing:** prove coverage and recovery with `assert_all_baselined_handlers_cover` and `assert_resume_recovers` — see [Testing your migrations](#handler-coverage-gate).
- **Deploy order:** alias nodes are purely **additive**, so a handler rename is safe to ship in a **single release** — unlike an event rename, which needs the two-release `legacy_write` dance (below). The new release's graph carries both the new node and the alias, so old in-flight checkpoints resume against it.
- **Runtime safety net.** If a handler is *removed* (or renamed without `previously=`) and a thread is still paused inside it, `resume()` would otherwise be a silent no-op. `EventGraph(on_unresumable=...)` governs this — it fires on any `resume()` of a thread that isn't awaiting input (also catching resume of an already-finished thread or a double-resume):
    - **`raise`** (default) — `UnresumableError`. Use in dev/CI to fail fast on an undeclared rename.
    - **`halt`** — emit a terminal `Unresumable(Halted)` and finalize the thread. Use in production for graceful, *observable* degradation (the thread ends in the log rather than hanging).
    - **`warn`** — `UserWarning` + no-op (thread untouched). Use when you want a signal but to handle the thread yourself.

    The CI handler gate catches undeclared renames *before* deploy; `on_unresumable` is the runtime last-resort net for anything that slips through.

    Retiring an `Interrupted` subclass is a related but separate move — see [Retiring an Interrupted subclass](#retiring-an-interrupted-subclass) below.

### Inline command handlers are keyed by the command qualname

An inline `Command.handle()` handler's node identity is the **command's `__qualname__`** (e.g. `Order.Place`), not the method name — so it is stable and order-independent. Reordering the `handlers=[...]` list is safe, and you do **not** need `@on(node_name=...)` to pin it (that pin is for standalone `@on` functions, whose identity is otherwise the function name).

Renaming the command class therefore *is* a node rename. Declare the historic node names as a `previously` class attribute — inline handlers have no decorator slot. Annotate it `ClassVar` (or leave it un-annotated); a plain annotation would make it an event *field* and is rejected at class creation:

```python
class Persist(Command):
    previously: ClassVar = ("Persona.Persist", "Story.Persist", "save")
    ...
```

The lists often rhyme with a `@migrate_from` stack on the same class, but they answer different questions: `@migrate_from` revives the old payload **bytes**; `previously` revives the old checkpoint **pointer** (the node name in `snapshot.next`). Each may carry entries the other can't — a node-only era (a reactor the command replaced, a pre-#97 `handle_N`) has no serde identity, and a module-only move has no node rename. The handler gate catches a missing `previously`; the event gates catch a missing `@migrate_from`.

A historic node name identifies exactly one class, and it stays that way: a concrete `Command` may not be subclassed (`class Child(Persist)` is a `TypeError`), so `previously` — like `raises`, `invariants` and `retry` — is always read off the command class that declares it. A second command that took over some of the old node names declares its own list.

!!! note "Upgrading a baseline recorded before this fix (#97)"
    Earlier releases recorded inline command handlers by positional names (`handle`, `handle_2`, …). After upgrading, those names no longer resolve, so `assert_all_baselined_handlers_cover` will raise `HandlerCoverageError` until you **regenerate the baseline once** with `write_baseline(graph, BASELINE)`. This firing is expected — it is the gate doing its job. A checkpoint paused inside an inline command handler *before* the upgrade cannot resume afterward (the old positional name was order-dependent and can't be reconstructed); recover a specific one with `previously: ClassVar = ("handle_N",)` on that command class, or set `on_unresumable="halt"/"warn"`.

### Renaming an event *and* a handler together

The two tracks are independent — do both, in one PR:

1. **Decorate both** — `@migrate_from("Old.Qualname")` on the renamed event class; `@on(previously="old_node")` on the renamed handler.
2. **Regenerate the baseline once** — a single `write_baseline(graph, BASELINE)` captures both the event identities *and* the handler node names (baseline v2). Commit it in the same PR.
3. **Run both gates in CI** — they're independent and both required: an event gate (`assert_all_baselined_revive(serde, BASELINE)`) **and** the handler gate (`assert_all_baselined_handlers_cover(graph, BASELINE)`). One does not cover the other.
4. **Deploy** — the handler alias is single-release safe; the event rename follows the two-release `legacy_write` cadence, so the *event* side gates the rollout.

!!! warning "Renaming an inline Command is always both renames at once"
    The command class is simultaneously the node identity *and* the event class of the payload sitting in the paused checkpoint. The alias node dispatches by `isinstance` against the **new** class, so `previously` alone is not enough: the checkpointed command payload must also revive *as* the new class — `@migrate_from` on the command plus a namespace-aware serde on the checkpointer (`from_namespaces(..., checkpointer=...)` wires it automatically). With the default LangGraph serializer the alias node re-enters but sees no matching event and the resume silently no-ops — and this bypasses `on_unresumable`: the thread *is* still awaiting input, so the safety net never fires.

## Retiring an Interrupted subclass

To retire an `Interrupted` subclass, delete it from the codebase once no live checkpoint still references it. `graph.abandon(config)` / `.aabandon()` settles one paused thread without answering it — see [Ending a pause without answering it](control-flow.md#ending-a-pause-without-answering-it-abandon).

`abandon()` settles one thread per call. `graph.threads_paused_on(EventClass)` (or `athreads_paused_on()`) finds the paused threads for you — no need for your own operational records or a direct checkpointer query. It reads each thread's checkpoint directly, so it still finds a thread whose handler has already been removed from the graph — retiring an `Interrupted` usually retires the handler that produced it too, and this is the common order to do it in.

!!! warning "This retires paused threads only — see [#159](https://github.com/cadance-io/langgraph-events/issues/159)"
    A thread that already answered the interrupt holds the retired class in its *settled* history, not a pending one. `threads_paused_on()` does not find it, and `abandon()` does not touch it. Deleting the class then makes that thread raise `Cannot revive` on any read. There is no library tool yet for a settled-history identity — that gap is tracked as [#159](https://github.com/cadance-io/langgraph-events/issues/159). **Retirement is not complete until #159 lands**; until then, deleting the class is safe only if you can prove no thread's settled history references it.

### Sequence

1. Enumerate every thread paused on the class with `graph.threads_paused_on(EventClass)`.
2. Call `graph.abandon(config)` (or `.aabandon()`) on each thread returned.
3. Verify: `graph.get_state(config).is_interrupted` is `False` for every thread.
4. Delete the class from the codebase.
5. Re-baseline: `write_baseline(graph, BASELINE, allow_removed=True)`.

```python
for config in graph.threads_paused_on(EventClass):
    graph.abandon(config, reason="retiring EventClass")
    assert not graph.get_state(config).is_interrupted
```

### Why the baseline write needs `allow_removed=True`

Deleting the class drops an identity from the graph's topology. `write_baseline` compares the new snapshot against the committed baseline and raises `BaselineRegressionError` if it would drop a recorded identity — pass `allow_removed=True` to confirm the drop is intentional (see [When to commit the baseline](#when-to-commit-the-baseline)).

Skip this step and the *next* baseline write for an unrelated change fails with the same error, pointing at the retired class. `abandon()` clearing the checkpoints is what makes the deletion safe; `allow_removed=True` is what makes the baseline write accept it.

Once the re-baseline lands, `assert_all_baselined_cover` and the other [coverage gates](#coverage-gates) stop checking the retired identity — it is no longer in the baseline they read.

!!! warning "Expect `assert_all_baselined_handlers_cover` to fail first"
    Retiring an `Interrupted` usually retires the handler that produced it too. Deleting both together trips the handler gate (`HandlerCoverageError`) in the same way the event gate trips — this is the gate doing its job, not a new problem. The same `write_baseline(graph, BASELINE, allow_removed=True)` re-baselines both the event identity and the handler node in one write; no separate step is needed.

## Reserved attributes

Library-private; read directly if introspection needed (neither is MRO-inherited):

- `__lge_migrate_from__` — set by `@migrate_from`; tuple of `(module, qualname)` pairs, oldest first.
- `__lge_backfill__` — set by `@backfill`; accumulated field/default entries.
- `__lge_origin_backfill__` — set by `@migrate_from(backfill=...)`; accumulated `((module, qualname), {field: default})` entries.

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
- `migrate_from(backfill=...)` with a multi-qualname chain, an empty dict, or a mutable value (steers to the `Migration.add_field` escape hatch) — `ValueError` at decoration, even earlier
- A duplicated origin qualname across stacked `@migrate_from` decorators (or within one multi-arg call) — `ValueError` at decoration
- Unknown `Operation` type in `Migration.operations` — `TypeError`

A misspelled `@migrate_from("Persona.Persistedd")` fails at construction, not at first checkpoint load in production.
