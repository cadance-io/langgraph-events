# Event migrations

`NamespaceAwareSerde` keys event identity by `(__module__, __qualname__)` so nested events with colliding leaf names (`Persona.Approve.Approved` vs. `Story.Approve.Approved`) round-trip distinctly. The flip side: any rename or relocation invalidates every active checkpoint holding the affected class — revival fails on `importlib.import_module` or the dotted `getattr` walk.

Migrations declare the historic identities a class previously held. On read, the serde rewrites old `(module, qualname)` to the current one in memory before reviving. The wire format is unchanged: payloads written by any prior library version remain readable, and payloads written today remain readable by future versions.

## The minimum case: rename inside a namespace

Cadance wants to nest its outcome events inside the producing command. `Persona.Persisted` becomes `Persona.Persist.Persisted`:

```python
from langgraph_events import Command, DomainEvent, Namespace
from langgraph_events.serde import migrate_from

class Persona(Namespace):
    class Persist(Command):
        @migrate_from("Persona.Persisted")
        class Persisted(DomainEvent):
            note: str = ""
```

That's the whole authoring story. `EventGraph.from_namespaces` already knows its namespaces and receives the checkpointer, so it auto-wires a migration-aware serde scoped to exactly those namespaces — you do not construct `NamespaceAwareSerde` or repeat a namespace tuple:

```python
from langgraph.checkpoint.memory import MemorySaver

graph = EventGraph.from_namespaces(
    Persona,
    handlers=[...],
    checkpointer=MemorySaver(),
)
```

The serde walks only those namespaces for `@migrate_from` metadata and assembles a `Migration` per decorated class — no separate collection step. New writes use the current qualname; old payloads with `Persona.Persisted` get rewritten to `Persona.Persist.Persisted` before revival. A `@migrate_from` in some unrelated imported module never contributes migrations or affects `legacy_write` for this serde.

To opt out — hand-authored `migrations=`, `legacy_write=True`, or a custom serde — pass your own `NamespaceAwareSerde` via `MemorySaver(serde=...)`; a user-supplied serde always wins over auto-wiring.

## Adding a field with a default

The common case needs **no migration at all**. Events are frozen dataclasses; a new field that declares a default (or `default_factory`) simply uses it when an old payload omits the key — the serde constructs the class with whatever kwargs the old payload carried:

```python
class Persisted(DomainEvent):
    note: str = ""
    tags: tuple[str, ...] = ()   # added later — old payloads revive with ()
```

Reach for `@backfill` (below) only when the new field is **required** (no default) and you cannot give it one.

## Multi-step chains

A class that was renamed twice declares both historic identities, oldest first:

```python
@migrate_from("Persona.Persisted", "Persona.OldNest.Persisted")
class Persisted(DomainEvent): ...
```

The serde flattens the chain so a payload at any historic step revives directly at the current class in a single dict lookup.

The multi-arg form above is the canonical one. Stacked decorators work too — Python applies decorators bottom-up, so the bottom-most decorator's qualname is the oldest:

```python
@migrate_from("Persona.OldNest.Persisted")   # ← newer
@migrate_from("Persona.Persisted")           # ← oldest (applied first)
class Persisted(DomainEvent): ...
```

Both produce the same chain. Prefer the multi-arg form for new code; the stacked form is useful when migrations are added incrementally across releases.

## Adding a required field

A new field that can carry a dataclass default needs no migration at all (above). The case that *does*: the field is **required** when the event is constructed in code, but pre-existing checkpoints predate it. A plain default can't express that — it would relax the constructor for everyone. `@backfill` declares the legacy value on the class, right next to the field, and is auto-collected exactly like `@migrate_from` — no `migrations=` list, no manual serde:

```python
from langgraph_events.serde import backfill, migrate_from

class Persona(Namespace):
    class Persist(Command):
        @migrate_from("Persona.Persisted")     # renamed…
        @backfill("command_id", default="legacy")  # …and gained a required field
        class Persisted(DomainEvent):
            command_id: str        # required when constructed in code
            note: str = ""
```

`EventGraph.from_namespaces(Persona, checkpointer=MemorySaver())` picks it up — a pre-`command_id` payload revives with `command_id="legacy"`. Stacked `@backfill` accumulate (one per added field). Rename is applied first, then the back-fill on the resulting identity, so the two decorators compose. `default` / `default_factory` follow the `AddField` convention; a mutable `default=[]` raises `ValueError` at serde construction (use `default_factory=list`) — the *same* guard, not a forked rule.

For a **cross-module** relocation, or grouping several operations, drop to the hand-authored escape hatch. Raw operation constructors live in `serde.migrations`:

```python
from langgraph_events.serde import NamespaceAwareSerde
from langgraph_events.serde.migrations import AddField, Migration

serde = NamespaceAwareSerde(
    namespaces=(Persona,),
    migrations=[
        Migration(
            name="add-command-id",
            operations=(
                AddField(
                    module="cadance.persona",
                    qualname="Persona.Persist.Persisted",
                    field="command_id",
                    default="legacy",
                ),
            ),
        ),
    ],
)
```

`AddField` applies after any matching `RenameEvent`, so `module`/`qualname` name the **current** (post-rename) class — exactly what `@backfill` derives for you from the decorated class.

### Single-op sugar

Hand-authored migrations are common enough that a one-op rename or field addition usually doesn't need the full `Migration(name=..., operations=(...))` envelope:

```python
from langgraph_events.serde import Migration

migrations = [
    Migration.rename(
        name="rename-persona-persisted",
        old_module="cadance.persona",
        old_qualname="Persona.Persisted",
        new_module="cadance.persona",
        new_qualname="Persona.Persist.Persisted",
    ),
    Migration.add_field(
        "add-command-id",
        module="cadance.persona",
        qualname="Persona.Persist.Persisted",
        field="command_id",
        default="legacy",
    ),
]
```

The *new* (post-rename) and *target* identities always name a class that still exists, so pass the class itself — refactor-safe, an IDE rename moves with it. The string `new_module`/`new_qualname` and `module`/`qualname` forms remain for the cross-module case where the live class can't be imported at authoring time. The old identity stays a string (that class is gone):

```python
from cadance.persona import Persona

migrations = [
    Migration.rename(
        old_module="cadance.persona",
        old_qualname="Persona.Persisted",
        to=Persona.Persist.Persisted,           # ← live class, not strings
    ),
    Migration.add_field(
        target=Persona.Persist.Persisted,       # ← same
        field="command_id",
        default="legacy",
    ),
]
```

`Migration.rename` and `Migration.add_field` are equivalent to wrapping a single `RenameEvent` / `AddField` operation in a `Migration`. Reach for the full constructor when one migration groups multiple operations.

`name` is optional everywhere — `Migration(operations=(...))` and `Migration.rename(...)` without `name=` both work. Naming helps when the validator surfaces a conflict between two hand-authored migrations.

## Rolling deploys

The library's read path makes new code tolerant of old payloads, but the **write path is asymmetric**: new pods writing under a new qualname produce payloads that old pods (running the previous release) can't revive — they don't know the class yet. To bridge a rolling deploy without resume failures, ship the migration in **two releases**:

### Release N+1 — `legacy_write=True`

```python
serde = NamespaceAwareSerde(
    namespaces=NAMESPACES,
    legacy_write=True,    # ← writes use the OLDEST historic qualname
)
```

While `legacy_write=True`, new pods encode events under the qualname recorded by `@migrate_from` (for classes inside `namespaces=`). Old pods (release N) read those payloads via their existing class definitions. New pods read both old- and new-format payloads via the migration table. Both pod versions can resume each other's threads throughout the rollout window.

`legacy_write` is scope-symmetric: a decorated class outside the serde's `namespaces=` is encoded under its current qualname, not its historic one. Otherwise the read path of the same serde would not know how to migrate the bytes back. Keep `namespaces=` consistent between encode and decode pods.

### Release N+2 — `legacy_write=False` (default)

Once release N is fully drained:

```python
serde = NamespaceAwareSerde(namespaces=NAMESPACES)
```

Writes flip to the current qualname. The decorator entries stay on the classes — they cover any old-format payloads still in storage.

### Eventually — drop the migration

After every old-format payload has been touched by the new code (the next write rewrites under the current qualname), the migration entry becomes purely defensive. Drop it when you're confident no legacy payloads remain.

## Concurrency & rolling deploys

There is **no lock or transaction** in the serde/migration layer. Two instances coming up at once is safe by *idempotency* and *by construction*, not by mutual exclusion — which is sufficient for the rolling-deploy case and keeps the hot path lock-free.

- **Concurrent reads are safe.** The migration ext-hook is a pure rewrite with no write-back; the rename table is the transitive closure, so any historic identity reaches the current class in a single lookup. Re-reading the same checkpoint from any number of pods yields the same revived object.
- **Old and new code cannot share a process — enforced, not assumed.** A `@migrate_from` whose historic identity still resolves to a live class is rejected at serde construction (see [Validation guarantees](#validation-guarantees): "shadowing a currently-live class"). The library therefore *structurally* requires the previous release's class to be gone in the new release's process — exactly the rolling-deploy model. The two releases coexist only as bytes on the shared checkpointer, never as classes in one interpreter.
- **`legacy_write=True` is coexistence, not a lock.** It is the format-compatibility mechanism for the two-release window (see [Rolling deploys](#rolling-deploys)); both releases can revive each other's payloads. Nothing is serialized.
- **Adding a required field is a two-release operation, like a rename.** An instance whose class lacks a field that is present in the bytes fails loudly on construction — never a silent drop. Pair the field addition with `@backfill` and ship it over the same N → N+1 cadence.
- **Thread-level concurrency on a single `thread_id` is the checkpointer's job.** `MemorySaver` provides none; the SQLite/Postgres savers bring their own semantics. `langgraph-events` adds nothing on top — concurrent writers to the same thread are governed entirely by your saver.
- **Recovery replay is idempotent.** `replay_reducer` recomputes the channel value from the (already-migrated) event log and the documented recipe overwrites it. Two instances running the startup-replay script concurrently both write the same correct value; the only residual concern is your checkpointer's own put atomicity.
- **One un-protected spot:** `write_baseline` is a non-atomic read-modify-write. Sequential divergent writers are caught by the regression guard (the second raises `BaselineRegressionError`), but a true within-call read→write interleave between two CI processes is a TOCTOU the library does not guard. It is a dev/CI tool, not a runtime path — generate and commit the baseline from a **single** CI job, not in parallel.

## Reducer state migration

Reducer projections live in checkpoint channel values. The library doesn't add channel-level migration ops because they aren't necessary — the truth is in the events, and events already migrate. Empirically measured behaviour across realistic reducer-state changes (see `tests/test_reducer_migration_scenarios.py`):

| Change | Behaviour | Action needed |
|---|---|---|
| Reducer value is `list[Event]` and an event class was renamed | Each event migrates through the ext-hook recursively | ✅ Nothing — works automatically |
| `ScalarReducer` holding a single Event + class renamed | Migrates | ✅ Nothing |
| `dict[str, list[Event]]` grouping reducer + event renamed | Migrates | ✅ Nothing |
| Plain dataclass channel value + new field WITH default | Revives via the dataclass default | ✅ Nothing |
| Plain dataclass channel value + new REQUIRED field | **Silently revives as `None`** | ⚠️ Use replay or strict mode (below) |
| Pydantic model channel value + new REQUIRED field | **Revives as malformed instance** — passes `isinstance`, `AttributeError` on field access | ⚠️ Use replay or strict mode |
| Reducer output shape changed (e.g., `dict[str, int]` → `dict[str, dict]`) | Revives as old shape — consumer crashes downstream | ⚠️ Use replay |
| Projection function semantics changed (e.g., dollars → cents) | Silent stale data, no exception ever | ⚠️ Use replay (only fix) |

The four ⚠️ cases all share one recovery path: replay events through the current reducer to rebuild the channel value from truth.

### Recovering with `replay_reducer`

```python
from langgraph_events.serde import replay_reducer

# In a startup migration script (run once during deploy):
tup = checkpointer.get_tuple(config)
event_log = tup.checkpoint["channel_values"]["event_log"]   # adjust to your channel name

rebuilt = replay_reducer(my_reducer, event_log)

# Write `rebuilt` back through the checkpointer's put API.
# The exact call depends on your saver (MemorySaver / SqliteSaver / PostgresSaver).
```

`replay_reducer` is a thin wrapper around `BaseReducer.seed(events)` — the reducer's default, namespace filter, and `event_type` predicate all apply uniformly. Composes with the existing event-rename machinery because `event_log` was already migrated on read.

The library doesn't iterate the checkpointer for you. Concrete savers vary in their iteration semantics (`MemorySaver.list(None)` yields all threads; `SqliteSaver` / `PostgresSaver` typically need a `thread_id`). Wire the read/write loop in your own startup script.

### Catching silent revivals loudly

The "silent fail" rows above are upstream LangGraph behaviour for partial dataclass / Pydantic revival. Strict mode flips them from "silently malformed object" to "loudly-broken `dict`" — the first consumer access (`isinstance` check, attribute read) trips a `TypeError` or `AttributeError`, much louder than a `None` channel value.

**Env var (process-wide):**
```bash
export LANGGRAPH_STRICT_MSGPACK=true
```
Set this in development and CI. Unrecognised classes are **demoted to their raw kwargs `dict`** instead of being revived, and LangGraph emits a `logging.warning`:

```
Blocked deserialization of <module>.<ClassName> - not in allowed_msgpack_modules.
Add to allowed_msgpack_modules to allow: [('<module>', '<ClassName>')]
```

The demoted `dict` then fails downstream — `isinstance(value, ExpectedClass)` returns `False`, field access raises `AttributeError`. That's the "loud" part: the failure is impossible to ignore at the first read, instead of festering as `None` in a channel.

Strict mode does **not raise** at the serde boundary. For "fail at deserialization, not at first consumer access" semantics, use `replay_reducer` to rebuild from event truth (see above) — that remains the recovery path for the ⚠️ rows.

**Explicit allowlist (per-serde, fine-grained):**
```python
serde = NamespaceAwareSerde(
    namespaces=NAMESPACES,
    allowed_msgpack_modules=[("cadance.state", "Stats"), ("cadance.state", "Summary")],
)
```
`NamespaceAwareSerde` forwards `**kwargs` to `JsonPlusSerializer`, so any upstream constructor option works. Use this in production when you want strict mode scoped to one serde instance — classes in the allowlist revive normally; everything else is demoted as above.

## What is NOT migrated

**Non-Event payloads in general.** Pydantic models, plain dataclasses, LangGraph `Interrupt` wrappers, and other non-event values flow through LangGraph's default serde paths. The migration system only rewrites `EXT_NAMESPACE_AWARE_EVENT` payloads (your `Event` subclasses). Events nested inside `Interrupt.value` are reached automatically through the serde's recursive ext-hook — no extra wiring needed.

**Reducer channel-name renames.** That's a LangGraph channel-routing concern, not a serde concern. See [Checkpointer & Graph Evolution](checkpointer-evolution.md) for the documented behaviour.

**Payloads ormsgpack refuses to encode.** `NamespaceAwareSerde._default` is a strict superset of upstream's `_msgpack_default`, so an `ormsgpack.MsgpackEncodeError` means a value in state has no encode path at all. The error propagates at the source — there is no fallback. Resolve by either removing the unencodable value from state or extending the encode hook (subclass `NamespaceAwareSerde` and override `_make_default`).

**Channel values that aren't reducer state.** `replay_reducer` rebuilds a reducer's output from its event log, but for non-reducer channel values (e.g. a Pydantic model written directly by a node), no analogous rebuild path exists. If you add a required field to such a class, old payloads either revive missing the attribute (raising `AttributeError` on first access) or are demoted to `dict` under strict mode. The recovery path is custom: read the old value, transform it, write it back through your saver's put API.

## Detection tooling

`detect_changes` diffs the current graph topology against a stored baseline, surfacing event identity changes that need migration entries:

```python
from langgraph_events.serde.migrations.detect import (
    detect_changes,
    write_baseline,
)

# After authoring the initial migrations, snapshot the current topology:
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

`detect_changes` is a **suggestion engine**, not an applicator. It matches removals to additions by leaf name; multi-match cases land in `ambiguous` for human review, and pure deletes land in `unmatched_removed` rather than being silently dropped. The library never auto-edits a migration list.

For the common case you don't need to write the loop above — the package is runnable as a one-line CI gate. Given a factory that builds your graph (`module:attr`, called with no args, or an `EventGraph` instance attribute):

```bash
python -m langgraph_events.serde.migrations myapp.graph:build migrations/baseline.json
```

It exits `0` when the topology matches the baseline, `1` when it diverges (printing the added / removed / rename-candidate buckets), and `2` on a usage error. Wire that command into pre-commit or CI. The library ships no hook *configuration* because the entry point to your `EventGraph` is project-specific — but the runner itself is built in. Drop to the programmatic `detect_changes` form only when you want a custom reporter.

### When to commit the baseline

The baseline records "what did the topology look like before this change." Commit it **alongside the migration that covers the change** — never after. This is enforced, not just advised: `write_baseline` refuses to overwrite an existing baseline when the new snapshot would drop identities the old one recorded, raising `BaselineRegressionError` (`.removed` lists the dropped identities for CI reporters). Writing the baseline after a rename landed — before authoring `@migrate_from` / `@backfill` — is exactly that case, so the silent miss is now a loud failure.

The expected workflow:

1. Open the branch that contains the rename.
2. Author the migration (`@migrate_from` / `@backfill` on the surviving class, or a hand-authored `Migration`).
3. Run `write_baseline(graph, "migrations/baseline.json")` and commit the regenerated JSON in the same PR.

If you genuinely intend to drop an identity (a real delete, no replacement), pass `write_baseline(graph, path, allow_removed=True)` to acknowledge it. The guard compares the on-disk baseline against current topology only — it never inspects the serde or migration table; *coverage* (does a migration actually exist for a renamed identity?) remains the job of `assert_covers` / `assert_all_baselined_revive`. The baseline file is also versioned; `_load_baseline` (and therefore `write_baseline`'s pre-check and `detect_changes`) raises `ValueError` on an incompatible version, so a stale snapshot can't silently mis-classify diffs.

## Testing your migrations

The first production read is the wrong place to discover a missing migration. Make it a `pytest` gate that runs on every PR.

`detect_changes` works against a **graph** topology; the coverage checks below work against a **serde** construction — because coverage depends on the `migrations=` / `namespaces=` the serde was actually built with, the serde owns it.

### `assert_all_baselined_revive` — the zero-maintenance gate

This is the one test most projects need. It walks every identity in the committed baseline, pushes a synthesized legacy payload for each through the real read path, and asserts it revives to an `Event`. A new `@migrate_from` / `@backfill` plus a regenerated baseline is covered with **no new test code** — there is no per-event list to maintain:

```python
from pathlib import Path

from langgraph_events.serde import NamespaceAwareSerde, assert_all_baselined_revive

from cadance.namespaces import Persona

BASELINE = Path(__file__).parent / "migrations" / "baseline.json"


def test_every_baselined_identity_revives():
    serde = NamespaceAwareSerde(namespaces=(Persona,))
    assert_all_baselined_revive(serde, BASELINE)
```

It fills required fields of the resolved live class with placeholders, so it proves **identity reachability + constructability**: every historic identity maps through the migration table to a class that still constructs. It deliberately does *not* assert specific old field values — for the narrow case where a field genuinely changed shape (not merely appeared), pin it explicitly with `synthesize_legacy_payload` below.

### `synthesize_legacy_payload` — pin a specific old field shape

Reach for this only when a field's *shape* drifted and you want to prove a specific old payload still revives — the loop gate above already covers identity reachability for everything else. The serde never encodes under a qualname whose class no longer exists, so you synthesize the bytes a prior release would have written. `synthesize_legacy_payload(module, qualname, kwargs)` does exactly that — no need to import private wire-format symbols:

```python
import pytest
from langgraph_events.serde import (
    NamespaceAwareSerde,
    synthesize_legacy_payload,
)

from cadance.namespaces import Persona

NAMESPACES = (Persona,)


@pytest.mark.parametrize(
    "module, qualname, kwargs, expected_cls",
    [
        (
            "cadance.persona",
            "Persona.Persisted",
            {"note": "n"},
            Persona.Persist.Persisted,
        ),
        # only events whose field shape changed — not every rename
    ],
)
def test_revives_release_N_payloads(module, qualname, kwargs, expected_cls):
    serde = NamespaceAwareSerde(namespaces=NAMESPACES)
    revived = serde.loads_typed(
        synthesize_legacy_payload(module, qualname, kwargs)
    )
    assert isinstance(revived, expected_cls)
```

If the migration is present but the dataclass shape drifted (a new required field), this test fails with the dataclass `TypeError` — exactly where you want a field-shape regression caught.

### `assert_covers` — is every baselined identity still reachable?

`NamespaceAwareSerde.assert_covers(baseline_path)` raises `MigrationCoverageError` if any identity in the baseline is neither still live in the serde's namespaces nor covered by a rename migration:

```python
from pathlib import Path

from langgraph_events.serde import NamespaceAwareSerde

from cadance.namespaces import Persona

BASELINE = Path(__file__).parent / "migrations" / "baseline.json"


def test_covers_every_baselined_identity():
    serde = NamespaceAwareSerde(namespaces=(Persona,))
    serde.assert_covers(BASELINE)
```

`MigrationCoverageError` extends `ValueError`; its `.uncovered` attribute is the tuple of offending `(module, qualname)` identities so a custom CI reporter can format them however it wants. The message points at the three remedies: add `@migrate_from` to the surviving class, append a `Migration` to `migrations=`, or regenerate the baseline if the identity was intentionally dropped.

If a developer removes a `@migrate_from` by accident, this test fails naming the now-uncovered identity — before the bytes ever reach production.

### `revivable_identities` — the introspection escape hatch

For custom coverage rules, `NamespaceAwareSerde.revivable_identities()` returns the read-only `frozenset` of every `(module, qualname)` the serde can revive — the union of live classes in scope and every historic identity a rename migration rewrites. It is a set, not a coverage check; `assert_covers` is the gate. (AddField targets are not included — they key on the post-rename identity and add no new revivable identities.)

### Release N → N+1 walkthrough

1. **At release N**, run `write_baseline(graph, BASELINE)` once and commit the JSON (see [When to commit the baseline](#when-to-commit-the-baseline) — the same "commit alongside the migration entries" rule applies).
2. **On the feature branch**, rename the event and add `@migrate_from("Persona.Persisted")` (plus `@backfill` if it also gained a required field). No new test code: `test_every_baselined_identity_revives` already covers the historic identity through the real read path.
3. **In CI**, the loop gate runs on every PR — it catches an accidentally-dropped migration *and* a class that no longer constructs. Add a `synthesize_legacy_payload` entry only if a field's shape changed and you want to pin the exact old payload.
4. **At release N+1 cutover**, re-run `write_baseline(graph, BASELINE)` in the same PR that introduced the rename. The next round of removals is measured against the new baseline.

## Reserved attributes

`__lge_migrate_from__` is set on every class decorated with `@migrate_from`, carrying the historic-qualname chain. The serde reads it during the namespace walk (driven by `namespaces=`) at construction and uses the recorded oldest entry for `legacy_write=True` rewriting. It is **not** inherited through MRO — a subclass of a decorated class does not pick up its parent's history. Treat the attribute as library-private; if you need to inspect a class's history, read it via `cls.__lge_migrate_from__` directly (it's a tuple of `(module, qualname)` pairs, oldest first).

`__lge_backfill__` is the equivalent marker set by `@backfill`, carrying the accumulated field/default entries. Same contract: collected during the namespace walk, **not** MRO-inherited, library-private.

## Validation guarantees

Errors that would otherwise surface on first production read are raised at serde construction:

- Duplicate `old_*` keys (ambiguous rewrites) — `ValueError`
- Dead-end chains (the migration target doesn't resolve to an importable class) — `ValueError`
- `old_qualname` shadowing a currently-live class (would silently rewrite live payloads) — `ValueError`
- Cycles (`A→B` then `B→A`) — `ValueError`
- `AddField` targets that don't resolve — `ValueError`
- `AddField(default=<mutable>)` — steers user to `default_factory` (`ValueError`). `@backfill` funnels into `AddField`, so the same guard applies to a `@backfill(default=[...])`
- Unknown `Operation` type in `Migration.operations` (anything other than `RenameEvent` / `AddField`) — `TypeError`

A misspelled `@migrate_from("Persona.Persistedd")` fails at `NamespaceAwareSerde(...)` construction, not at the first checkpoint load in production.
