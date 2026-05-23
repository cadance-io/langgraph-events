# Event migrations

See also: [Checkpointer & graph evolution](checkpointer-evolution.md).

`NamespaceAwareSerde` keys event identity by `(__module__, __qualname__)` so nested events with colliding leaf names round-trip distinctly. Migrations rewrite historic identities on read while preserving wire format — payloads from any prior library version remain readable both ways.

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

### Hand-authored escape hatch

For cross-module relocations or composite operations, drop to `langgraph_events.serde.migrations`:

| Use case | Pattern |
|---|---|
| Single rename | `Migration.rename(to=Class, ...)` |
| Add field | `Migration.add_field(target=Class, field=..., default=...)` |
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
- **One unprotected spot: `write_baseline` is non-atomic** (dev/CI tool, not a runtime path). Generate the baseline from a single CI job, not in parallel.

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

For intentional deletes (no replacement), pass `allow_removed=True`. The guard compares baseline ↔ topology only; *coverage* (does a migration exist?) is `assert_covers` / `assert_all_baselined_revive`'s job. The baseline file is versioned — a stale snapshot raises `ValueError`.

## Testing your migrations

Run coverage tests on every PR; three patterns: `assert_all_baselined_revive`, `synthesize_legacy_payload`, `assert_covers`.

### `assert_all_baselined_revive` — zero-maintenance gate

Walks every baselined identity, pushes a synthesized legacy payload through the real read path, asserts it revives.

```python
from pathlib import Path
from langgraph_events.serde import NamespaceAwareSerde, assert_all_baselined_revive
from cadance.namespaces import Persona

BASELINE = Path(__file__).parent / "migrations" / "baseline.json"


def test_every_baselined_identity_revives():
    serde = NamespaceAwareSerde(namespaces=(Persona,))
    assert_all_baselined_revive(serde, BASELINE)
```

- Proves identity reachability + constructability for every baselined `(module, qualname)`.
- Fills required fields with placeholders — does NOT assert specific old field values (use `synthesize_legacy_payload` for that).
- A new `@migrate_from`/`@backfill` + regenerated baseline is covered with no new test code.

### `synthesize_legacy_payload` — pin a specific old field shape

Reach for this only when a field's *shape* drifted. Builds the bytes a prior release would have written:

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

- Pin only when a field's shape genuinely changed (not for plain renames).
- Failing test = dataclass `TypeError` on the missing field — exactly where you want a field-shape regression caught.

### `assert_covers` — every baselined identity reachable?

`NamespaceAwareSerde.assert_covers(baseline_path)` raises `MigrationCoverageError` if any baselined identity is neither live nor covered by a rename migration:

```python
def test_covers_every_baselined_identity():
    serde = NamespaceAwareSerde(namespaces=(Persona,))
    serde.assert_covers(BASELINE)
```

- `MigrationCoverageError` extends `ValueError`; `.uncovered` is the tuple of offending identities.
- Catches accidentally-removed `@migrate_from` before bytes reach production.
- `revivable_identities()` returns the read-only `frozenset` of revivable `(module, qualname)` for custom coverage rules. `AddField` targets are not included (they key on post-rename identity).

### Release N → N+1 walkthrough

1. **At release N**, run `write_baseline(graph, BASELINE)` once and commit the JSON.
2. **On the feature branch**, rename the event and add `@migrate_from("Persona.Persisted")` (plus `@backfill` if it gained a required field). The loop gate already covers the historic identity.
3. **In CI**, the loop gate runs on every PR — catches accidentally-dropped migrations and classes that no longer construct. Add `synthesize_legacy_payload` entries only if a field's shape changed.
4. **At release N+1 cutover**, re-run `write_baseline(graph, BASELINE)` in the same PR as the rename. Next removals are measured against the new baseline.

## Reserved attributes

Library-private; read directly if introspection needed (neither is MRO-inherited):

- `__lge_migrate_from__` — set by `@migrate_from`; tuple of `(module, qualname)` pairs, oldest first.
- `__lge_backfill__` — set by `@backfill`; accumulated field/default entries.

## Validation guarantees

Errors raised at serde construction (not at first production read):

- Duplicate `old_*` keys (ambiguous rewrites) — `ValueError`
- Dead-end chains (migration target doesn't resolve to an importable class) — `ValueError`
- `old_qualname` shadowing a currently-live class — `ValueError`
- Cycles (`A→B` then `B→A`) — `ValueError`
- `AddField` targets that don't resolve — `ValueError`
- `AddField(default=<mutable>)` — `ValueError` (steers to `default_factory`); `@backfill` funnels into `AddField`, same guard
- Unknown `Operation` type in `Migration.operations` — `TypeError`

A misspelled `@migrate_from("Persona.Persistedd")` fails at construction, not at first checkpoint load in production.
