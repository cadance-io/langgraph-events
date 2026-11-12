# Migration gate and deploy lifecycle

This page covers the gate that refuses an unsafe release, the deploy steps, and the store rewrite. Authoring a migration is on [Event migrations](event-migrations.md).

## The migration gate

The gate compares the code with a committed baseline file. It fails in CI before a stored payload fails in production. Declare one `MigrationTarget` for your graph. Every command and every test reads it.

```python
# myapp/lge_target.py
import os
from pathlib import Path

from langgraph.checkpoint.postgres import PostgresSaver
from langgraph_events import EventGraph
from langgraph_events.serde.migrations import MigrationTarget

from myapp.namespaces import Order, Shipment


def build_graph(checkpointer):
    return EventGraph.from_namespaces(Order, Shipment, checkpointer=checkpointer)


target = MigrationTarget(
    graph=build_graph,
    baseline=Path(__file__).parent / "lge-baseline.json",
    store=lambda: PostgresSaver.from_conn_string(os.environ["DATABASE_URL"]),
)
```

`graph` receives a checkpointer and returns the graph. The commands use `graph.event_store`. The factory can return an `EventStore` instead, when a command must not build the services of a run:

```python
from langgraph_events import EventStore, EventTopology


def build_store(checkpointer):
    return EventStore(EventTopology.from_namespaces(Order, Shipment), checkpointer)
```

When the checkpointer holds the default serde of LangGraph, the target sets `topology.serde()` on it. A serde with `migrations=` goes through `topology.serde(migrations)`. Set that serde in the factory, because the target does not replace a serde that is not the default.

`topology.serde()` reads the same event classes as the serde of `from_namespaces(..., checkpointer=...)`. Both collect each `@migrate_from` on an event outside every namespace, for example an `IntegrationEvent`. So both forms give the gate the same migrations. WARNING: A serde built with `namespaces=` only does not collect a migration on such an event. The gate then reports a different migration set. Use `topology.serde()`.

The rest of this section applies to both forms. The gate passes an empty `MemorySaver`, so it reads no store. The graph must carry a `NamespaceAwareSerde`. `from_namespaces(..., checkpointer=...)` wires one. A hand-built serde with `migrations=` also works. Set it on the checkpointer before you build the graph. `store` is for the commands that open the store. See [Lifecycle of a read-time migration](#lifecycle-of-a-read-time-migration).

The gate judges every event class that the graph can store. That set holds each class that the serde reads, from its `namespaces=` and its `events=`. It holds each class of a namespace that the graph uses. It holds each event outside every namespace that a handler subscribes to or returns, for example `HandlerRaised` or an `IntegrationEvent`. A class outside the serde revives by import, so the gate records it as live. A serde built with `namespaces=` only needs no `events=` for the gate. A mixin that a handler subscribes to is not a stored class, and the gate does not record it.

Run the gate in CI, and in a test. Write the baseline the code needs with `baseline update`. Run it in the same PR as a new migration, a renamed handler or a new field, and commit the file in that PR, never after:

```bash
python -m langgraph_events.serde.migrations check myapp.lge_target:target
python -m langgraph_events.serde.migrations baseline update myapp.lge_target:target
```

```python
from langgraph_events.serde.migrations import assert_migration_gate
from myapp.lge_target import target


def test_migration_gate() -> None:
    assert_migration_gate(target)
```

`assert_migration_gate` raises `GateError`, an `AssertionError`. Its `report` holds the typed findings.

### The rules

`check()` runs the rules in this order. Each finding names one next command, with your own arguments.

| Rule | Finding | It means | Next command |
|---|---|---|---|
| `baseline` | `BaselineStale` | The committed file differs from the file the code needs. The finding lists each difference. | `baseline update` |
| `names` | `AnonymousCallable` | A transform or a split uses a lambda, a callable object, or a local function that captures a value with no stable form. A replacement callable would read as the same migration. | Give the callable a module-level name, then `baseline update` |
| `coverage` | `MigrationOutOfScope` | A class of the graph declares a read-time migration, and the serde does not read the class. The migration does nothing. | Pass its namespace in `namespaces=` of the serde, or the class in `events=`, then `baseline update` |
| `coverage` | `Uncovered` | A recorded identity has no live class and no migration. The finding names live classes with the same last name, a likely rename. | Add the migration, then `baseline update` |
| `revive` | `Unrevivable` | A synthesized payload of the recorded shape fails the real read path. | Add or fix the migration, then `baseline update` |
| `handlers` | `HandlerMissing` | A recorded handler node name has no live handler and no `@on(previously=...)` alias. | Declare the alias, then `baseline update` |
| `release` | `SurvivedRelease` | The previous release shipped this migration, and the code still holds it. | Delete the migration, then `baseline update --previous` |

The baseline rule runs first. The other rules judge the baseline the code needs, not the committed file. So a deleted shipped migration gives one `BaselineStale` finding, and never a request to add the migration back. The `coverage` rule accepts each identity in `NamespaceAwareSerde.revivable_identities()` and each event class that the graph can store. Resolution is scope-first, like the read path: the namespaces of the serde first, then import. So a namespace defined inside a function passes.

The `revive` rule sends a `None` placeholder for every required field of the live class, and for every recorded field that the live class no longer accepts. A field that a fill supplies gets the real fill value. So a broken fill, a dropped field that a transform reads, for example `kw.pop("legacy_flag").upper()`, or a `None` placeholder that `__post_init__` validates on a required field with no fill all fail the rule. Guard the value in the callable, give the field a `@backfill` with a valid value or a default, or pin a real payload with `synthesize_legacy_payload`. A `select` must return `None` when the discriminating value is absent or `None`. The failure line says so. An identity of this library only has to resolve to a live `Event`.

### The release rule

A read-time migration ships in release N and leaves the code in release N+1. The `release` rule needs the baseline of the previous release. The library does not run git. Give the file with `--previous`. If the store cannot be made clean in time, keep the migration one more release with `baseline hold`:

```bash
git show "$PREV_TAG:myapp/lge-baseline.json" > "$RUNNER_TEMP/prev.json"
python -m langgraph_events.serde.migrations check myapp.lge_target:target \
    --previous "$RUNNER_TEMP/prev.json"
python -m langgraph_events.serde.migrations baseline hold myapp.lge_target:target \
    "transform myapp.events:Order via myapp.events:drop_total" \
    --reason "a paused thread refuses the rewrite" --previous "$RUNNER_TEMP/prev.json"
```

| Case | Behaviour |
|---|---|
| `check` without `--previous` | The rule does not run, and the report says so. `assert_migration_gate(target, previous=...)` takes the previous file as `previous=`. A test usually has none, so run `check --previous` in CI |
| A previous file of version 1 to 3 | It records no migrations. The rule does not run, and the report says so |
| A rename in the current code | A previous key follows the rename onto the live class. A key whose identity was a rename source in the previous file does not move |
| A migration leaves the code | Run `baseline update --previous <file>`. If the previous release shipped the migration, the update drops the history that it covered. The fields of its class reset to the `live` fields of the previous file. A class that keeps another migration in the code keeps its fields |
| A changed key, for example a new factory argument or a renamed callable | It reads as a deletion and an addition. The `baseline-stale` finding names each deleted key and each new key of the same kind on the same class |
| `baseline update` without `--previous` when a recorded migration left the code | Exit 2. The error names the command with `--previous`. `update_baseline()` raises `PreviousRequiredError`, a `ValueError` whose `.deleted` holds the `MigrationKey` of each migration that left the code |

The hold records the digest of the whole previous file. The baseline of the next release holds the hold record, so its digest is different, and a check against that file ignores the hold. So a hold lasts one release, also when the migrations did not change. To keep the migration one more release, hold it again. Every check lists each hold that applies. Each `survived-release` finding gives the command to hold its migration, and an expired hold shows in it. A hold on a migration that the committed baseline does not record yet adds its record.

### Labels

One label form names a migration everywhere: in the baseline, in a finding, in a hold and in a read error.

```
<kind> <module>:<qualname>[<field>] [via <module>:<qualname>[#<digest>]]
```

`rename myapp.events:Order.Old`, `fill myapp.events:Order[status]` and `transform myapp.events:Order via myapp.events:drop_total` are examples. A transform that a factory builds has a `via` such as `myapp.events:drop_field.<locals>._drop#3f0c2a9e41b7`. The 12 hex digits are the digest of the captured values. The key of a partial holds a digest of its arguments. So a new factory argument gives a new key. A captured value must be a scalar, a tuple, a list, a set, a dict, a class, an enum member or another callable with a stable name. A read error names a stored identity as `<module>:<qualname>`.

`NamespaceAwareSerde.client_migrations()` returns a `MigrationSet`. Its `keys` hold one `MigrationKey` per operation that you declared, and its `fingerprint` names the set. The framework migrations of the library are not in it. `MigrationKey.parse(label)` reads a label back to its key. `python -m langgraph_events.serde.migrations fingerprint <baseline file>` prints the fingerprint of the migrations a file records. It reads the JSON only.

### Exit codes

| Code | `check` | `baseline update`, `baseline hold`, `fingerprint` |
|---|---|---|
| 0 | No finding | The command is done |
| 1 | One or more findings | Not used |
| 2 | A usage or target error, for example a target that does not import, a graph with no `NamespaceAwareSerde`, or a bad baseline file | The same |

The removed form `python -m langgraph_events.serde.migrations <module:factory> <baseline>` exits 2 and prints the new command line. For the exit codes of `rewrite`, see [What `rewrite` does](#what-rewrite-does). For `backup`, `restore` and `drop-backups`, see [Backup and restore](#backup-and-restore).

### What the file records

```json
{
  "version": 4,
  "events": [{"module": "myapp.orders", "qualname": "Order.Place.Placed",
              "fields": ["legacy_total", "order_id", "tracking"], "live": ["order_id", "tracking"]}],
  "retired": [{"module": "myapp.orders", "qualname": "Order.ApprovalRequired", "fields": ["order_id"]}],
  "handlers": [{"name": "handle_approval"}],
  "migrations": [{"label": "rename myapp.orders:Order.ApprovalRequired"},
                 {"label": "fill myapp.orders:Order.Place.Placed[tracking]",
                  "hold": {"reason": "a paused thread refuses the rewrite", "against": "lge-b1-3f0c2a9e41b7d655"}}]
}
```

| Key | It holds |
|---|---|
| `events` | Every live event class that the gate judges, a tombstone in a namespace of the serde included |
| `fields` | Cumulative. A field that was ever recorded can sit in a checkpoint, so the `revive` rule keeps sending it. Only an update with `--previous` after a shipped migration left the code resets the fields of the identity that the migration covered |
| `live` | The init fields of the live class when the file was written, the same set the serde writes. `NamespaceAwareSerde.live_fields()` returns that set for each live class. An update with `--previous` reads `live` from the previous file to reset the fields of a class |
| `retired` | Every identity an update dropped from `events`, with the fields last recorded for it. No `fields` key when the last record predates version 3. An identity that is live again leaves `retired`. A retired identity must revive through a migration onto a surviving class or a tombstone |
| `handlers` | The handler node names. An update rebuilds it. `EventGraph.resumable_node_names` returns each handler node name and each `@on(previously=...)` alias |
| `migrations` | The label of every migration of your own that the code holds, and any hold. The framework migrations of the library are not in it |

`events` and `retired` never share an identity. The reader raises `ValueError` on an overlap. A file of version 1, 2 or 3 still loads. A file with an unknown version raises `ValueError`. The next `baseline update` writes version 4.

## Lifecycle of a read-time migration

A read-time migration is `@migrate_from`, `@backfill`, `@transform_fields`, `@split_event` or a `Migration` in `migrations=`. It ships in release N and leaves the code in release N+1. The [migration gate](#the-migration-gate) records in the baseline file which release shipped each migration. The `rewrite` command makes the store match the code of release N. Release N+1 then reads the store with no migration.

The `backup`, `rewrite`, `restore` and `drop-backups` commands open `store` of the target. The gate does not.

- `store` returns a context manager of a sync checkpointer. An async context manager, for example from `AsyncPostgresSaver.from_conn_string`, stops the command with exit 2. If your app uses an async saver, give `store` the sync saver of the same database, for example `PostgresSaver.from_conn_string(url)` or `SqliteSaver.from_conn_string(path)`.
- `PostgresSaver.from_conn_string` takes the plain libpq form, `postgresql://...`. The SQLAlchemy form with a driver, `postgresql+psycopg://...`, fails in psycopg with `missing "=" after ...`. Remove `+psycopg` before you give the URL to the saver. The commands import the target and call `graph`, so the graph factory runs your full app setup. Run each command in an environment that has the configuration of your app, for example its environment variables. If the app setup opens a SQLAlchemy engine, it needs the driver form `postgresql+psycopg://...`. A plain `postgresql://...` there selects psycopg2, and the setup fails with `No module named 'psycopg2'` when only psycopg 3 is installed.
- The command lists the threads of a `PostgresSaver`, a `SqliteSaver` or a `MemorySaver`. For another saver, or for a store that other graphs share, set `threads=` to a function that takes the saver and returns a `ThreadKey(thread_id, checkpoint_ns)` for each thread and checkpoint namespace of this graph. See [Checkpointers](#checkpointers) and [Backup and restore](#backup-and-restore).

### Checkpointers

A plan writes nothing. An apply writes each checkpoint again under its own id. The built-in thread listing of `PostgresSaver` and `SqliteSaver` calls an internal method of the saver, which a release of the saver package can change. So install the saver through the extra of the library, which pins the range that the tests of the library cover:

```bash
pip install "langgraph-events[postgres]"   # PostgresSaver and psycopg
pip install "langgraph-events[sqlite]"     # SqliteSaver
```

| Checkpointer | Plan and apply | Thread listing of the command | Prune of old blob versions | Backup of the command |
|---|---|---|---|---|
| `PostgresSaver` | Yes | Built in, with one SQL query | Needed. See [Old blob versions](#old-blob-versions) | Built in: `backup`, `restore` and `drop-backups`. See [Backup and restore](#backup-and-restore) |
| `SqliteSaver` | Yes | Built in, with one SQL query | Not needed. `put()` replaces the whole checkpoint row | None. Copy the database file before the rewrite |
| `MemorySaver` | Yes | Built in, from its storage | No prune. The old versions stay in memory | None |
| A custom saver | Yes, when it obeys the three rules below | Set `threads=list_thread_keys` for a small store, or your own function | It depends on the saver | None. Use the tools of its database |
| `AsyncPostgresSaver`, `AsyncSqliteSaver` | Through `aplan_rewrite()` and `arewrite_store()` only | None. Pass `thread_ids=` and `checkpoint_ns=` | The same as the sync saver of the same database | The same as the sync saver of the same database |

A custom saver must obey three rules, or the rewrite cannot store a thread:

1. `put()` of a checkpoint whose id is in the store replaces the stored checkpoint.
2. `put_writes()` of an `__interrupt__` write replaces the stored `__interrupt__` write of the same task.
3. `get_tuple()` returns each pending write value as the object that the serde decoded. A copy does not obey this rule. Under a saver that breaks it, a thread that needs a rewrite and holds a completed task write is refused with a reason that names `get_tuple()`.

The base class `BaseCheckpointSaver` does not promise any of the three rules. So after each write, the rewrite reads the checkpoint and plans it again. Fix the saver, then run `rewrite --apply` again:

| What the read after the write finds | Reason |
|---|---|
| A newer checkpoint id | `thread advanced during the rewrite; rerun` |
| The old channel versions | `the saver did not overwrite checkpoint <id>`. Rule 1 is broken |
| The written versions, and a refusal of its own | `the rewrite is stored, but the read after the write refuses the thread: <reason>` |
| The written versions, and a record that still migrates | `the saver did not store the bytes written to checkpoint <id>`. The reason names the rule that the saver broke |

The plan proves before the write that the rewritten bytes settle: a read of them migrates no record. A class-stage transform that changes its own output again, for example one that adds 1 to a field, does not settle. The plan refuses that thread with `a migration of <module>:<qualname> does not settle`, and writes nothing. A transform that builds a tuple from a stored list settles. A tuple and a list store the same bytes.

`list_thread_keys` lists the threads of any saver through its public `list(None)`. WARNING: `list(None)` reads and decodes every checkpoint of the store, older checkpoints included. On a large store it costs time in proportion to the whole store, and memory when the saver loads all rows at once. For a large store, write a `threads` function that asks the store for each distinct thread id and checkpoint namespace.

### The steps

Each command is `python -m langgraph_events.serde.migrations <command>`.

| Step | Where | Command | It passes when |
|---|---|---|---|
| 1. Add the migration | Development | `baseline update <target>` | The baseline records the new migration |
| 2. Gate each change | CI | `check <target> --previous prev.json` | No shipped migration survives, and the baseline matches the code |
| 3. Before release N serves | In a one-off container of release N | `backup <target>`, then `rewrite <target> --expect <fingerprint of N> --accept rewrite` | Each command exits 0. Release N reads each checkpoint, or can rewrite it. No checkpoint is refused |
| 4. After release N serves | In a container of release N | `backup <target>`, then `rewrite <target> --apply --expect <fingerprint of N> --accept refused`, then `drop-backups <target>` | Each command exits 0. The passes ran until the store settled or `--passes` ran out. The rewrite exits 0 also when a checkpoint stays refused, and the output names each one |
| 5. Delete the migration | Development of release N+1 | Delete it, then `baseline update <target> --previous prev.json` | The baseline drops the history that the migration covered |
| 6. Before release N+1 deploys | In a container of release N | `rewrite <target> --expect <fingerprint of N>` | Exit 0: release N reads the store clean |
| Roll back release N | In a container of release N, before the rollback, when the older release cannot read the shapes of release N | `restore <target>` | Exit 0: the rows of the backup are back |

`prev.json` is the baseline file of the release that runs in production. The library does not run git. A client that tags each release reads the file at the last tag, `git describe --tags --abbrev=0 HEAD^`. A client that deploys continuously reads it at the commit that its deploy tool reports as running. For such a client, one release is one deploy, so the first change merged after a deploy must delete or hold each migration that the deploy shipped. Write this rule into the merge policy of the client. `<fingerprint of N>` is the output of `fingerprint <baseline file of N>`. Run `fingerprint` in the checkout of the deploying commit, then pass the value into the container. Inside the container, the baseline file is the file of the image, so a fingerprint computed there always passes the check.

```bash
FP=$(python -m langgraph_events.serde.migrations fingerprint myapp/lge-baseline.json)
LGE="python -m langgraph_events.serde.migrations"

# Step 3. `&&` runs each command only when the command before it exits 0.
your-deploy-tool run --version N "$LGE backup myapp.lge_target:target && $LGE rewrite myapp.lge_target:target --expect $FP --accept rewrite"

# Step 4.
your-deploy-tool exec "$LGE backup myapp.lge_target:target && $LGE rewrite myapp.lge_target:target --apply --expect $FP --accept refused && $LGE drop-backups myapp.lge_target:target"

# Step 6. The fingerprint of release N comes from prev.json.
FP=$(python -m langgraph_events.serde.migrations fingerprint "$RUNNER_TEMP/prev.json")
your-deploy-tool exec "$LGE rewrite myapp.lge_target:target --expect $FP"

# Rollback of release N.
your-deploy-tool exec "$LGE restore myapp.lge_target:target"
```

Step 3 stops a release that cannot read the store before that release serves. A checkpoint that needs a rewrite is the expected result of a new migration, so `--accept rewrite` lets the deploy go on. A refused checkpoint exits 1, and the deploy stops. Step 4 cannot stop the deploy, so `--accept refused` exits 0 when a checkpoint stays refused. Read the output, and settle each refused checkpoint before step 6. Step 6 is the check of the deletion. CI reads no store, so CI cannot prove that the store is clean. If step 6 exits 1, do not deploy release N+1. Run step 4 again under release N. If a refused thread cannot converge, keep the migration one more release with `baseline hold`. See [The release rule](#the-release-rule). No step names a backup. The backup commands name the backup after the migration fingerprint of the code that runs them, so step 3 and step 4 of release N add to one backup. They need a `PostgresSaver`. See [Backup and restore](#backup-and-restore). For another store, back up the store with the tools of its database before step 3, and leave out these three commands.

`--expect` is a release check. The command compares the migrations of the running code with the fingerprint before it opens the store. A mismatch exits 3. So the step cannot run under the wrong release, for example after a rollback. The command writes each pass line and its report to stdout, and an error to stderr. Some deploy tools show the stderr of a remote command only when it fails, so a successful run still shows what it changed.

### Dropping retired events

`drop=` on the `MigrationTarget` names the event classes whose stored instances `rewrite` drops. Each plan and each pass of the command passes them as `drop=` to `plan_rewrite()` and `rewrite_store()`. See [Rewriting the live set](#rewriting-the-live-set). So the drop runs inside the passes and the verdict of a normal rewrite. The deploy commands do not change. With the default `drop=()`, the command drops nothing.

`--expect` checks the migrations only. `drop` is not part of the migration fingerprint, so a release that changes only `drop` keeps the same `--expect` value. `check` and code review gate `drop`.

WARNING: a plan refuses a thread that is paused on a dropped class, so step 3 exits 1 while such a thread waits. Before you add a class to `drop`, abandon each paused thread of the class with [`abandon()`](control-flow.md#ending-a-pause-without-answering-it-abandon). To find these threads, see [Finding candidates server-side](retiring-interrupted.md#finding-candidates-server-side).

WARNING: a drop removes the stored event only. The drop does not recompute reducer state, so a reducer value keeps what a dropped event added to it. Do not add a class to `drop` when a reducer reads that class. A plan refuses a thread whose reducer value holds a dropped instance. See [A checkpoint that a plan refuses](#a-checkpoint-that-a-plan-refuses).

```python
target = MigrationTarget(
    graph=build_graph,
    baseline=Path(__file__).parent / "lge-baseline.json",
    store=lambda: PostgresSaver.from_conn_string(os.environ["DATABASE_URL"]),
    drop=(Order.RetiredApprovalGate,),
)
```

Each class in `drop` must be in `retired=` of the graph. `drop` deletes stored events, so it takes only a retired tombstone. The serde of the graph must also revive each class to that same class. `plan_rewrite()` applies this second rule. When a class breaks a rule, every command exits 2 with an error that names the class as `module:qualname`. So `check` in CI refuses a live class, or a class that `retired=` does not list, before a deploy runs it.

A plan names each dropped class and its count of log entries, for example `t1: needs a rewrite, drops 2 entries of myapp.order:Order.RetiredApprovalGate`. Under each pass line, an apply names them as `t1: dropped 2 entries of myapp.order:Order.RetiredApprovalGate`. One event that waits for dispatch is two entries: one in `events` and one in `_pending`.

Use `drop` to delete a tombstone through the deploy steps. See [Retired events](concepts.md#retired-events). In the release that holds the tombstone, list it in `retired=` and in `drop`. Step 4 drops its stored instances. Steps 3 and 6 report each thread that still holds one as a rewrite. When step 6 exits 0, the latest checkpoint of each thread holds no tombstone, so the next release can delete it. An older checkpoint keeps the class, so a time-travel read can still meet it. In the change that deletes the tombstone, remove it from `drop` too. See *Deleting the tombstone* in [Recovering a delete-first deployment](retiring-interrupted.md#recovering-a-delete-first-deployment).

### What `rewrite` does

1. It builds the graph on an empty `MemorySaver` and compares its migrations with `--expect`. It opens no store before this check.
2. It opens the store and builds the graph on its saver. When that saver already holds a serde with other migrations, it exits 2 before it reads a thread. The gate and the check judge the serde of step 1, so the serde that writes must be the same.
3. It plans the latest checkpoint of each thread in each checkpoint namespace: the root checkpoint namespace and each subgraph checkpoint namespace. The plan reports a stored instance of a class in `drop` of the target as a rewrite. It takes no backup. Run `backup` before it. See [Backup and restore](#backup-and-restore). For a store with no `checkpoints` table, see [A store with no checkpoint table](#a-store-with-no-checkpoint-table).
4. With `--apply`, it rewrites each checkpoint that the last plan reads as a rewrite. Then it plans every checkpoint again. It repeats this step up to `--passes` times, 3 by default, and stops when the last plan has no checkpoint to rewrite. A refused checkpoint changes only when a person acts, so a pass leaves it out. It logs one line per pass to stdout, then one line per checkpoint that the pass drops events from.
5. The verdict of the last plan and `--accept` decide the exit code.

The verdict of the last plan is the worst state of a checkpoint in it, in this order from best to worst. `--accept` names the worst verdict that still exits 0. The default is `--accept clean`.

| Verdict | The last plan |
|---|---|
| `clean` | Reads every checkpoint unchanged |
| `rewrite` | Reads a checkpoint as a rewrite, and refuses no checkpoint |
| `refused` | Refuses a checkpoint |

| Code | Meaning | The output names |
|---|---|---|
| 0 | The verdict of the last plan is `--accept` or better | The counts of each pass and of the last plan. When the verdict is not `clean`, the word `accepted` and each checkpoint that is not clean. When the passes ran out and the last plan still reads a checkpoint as a rewrite, the output says that the store did not settle, and names the next command: `rewrite --apply` |
| 1 | The verdict of the last plan is worse than `--accept` | Each checkpoint and its reason. When the last plan reads a checkpoint as a rewrite, the output also names the next command: `rewrite --apply` |
| 2 | A usage, target or store error. Examples: the target has no `store`, the graph factory takes no checkpointer, the graph has no `NamespaceAwareSerde`, the serde has `legacy_write=True`, `store` returns an async context manager, the store does not open, the store refuses the thread listing, the connection cannot reach the `checkpoints` table | The error, on one line, with its type when the library did not raise it |
| 3 | Wrong release: the running code does not hold the expected migrations | Both fingerprints, the running migrations and the command |

The Python form is `rewrite_live_set(target, expect=..., apply=...)`. It returns a `LiveSetOutcome`, or a `WrongRelease` when the `--expect` check refuses. `LiveSetOutcome.passes` holds the report of each pass, and `final` holds the last plan. `verdict` gives the `Verdict` of the last plan. The members of `Verdict` compare in the order of the table. `accepted(accept)` is `verdict <= accept`, and the command exits 0 exactly when it is true. `clean` equals `verdict is Verdict.CLEAN`.

Only a plan sees these cases. A migration counts only when the running code holds it. An older release does not hold the new migration, so its plan reads a thread that only the new migration changes as clean. The fence prevents this case. A plain dataclass default never makes a thread need a rewrite. The read fills the default, and the stored payload stays without the field. Only a `@backfill` makes a thread need a rewrite. `unrevivable_threads()` cannot see a retype, a split or a changed meaning. Only a plan under the release that holds the migration sees them.

### A checkpoint that a plan refuses

A refused checkpoint keeps the command from exit 0, and `rewrite_store()` writes nothing for it. Before the write, each rewritten value is encoded the way `put()` encodes it, scanned for a dropped or historic identity, and revived under the strict serde. One failure refuses the thread. The reason says what to do. A thread is refused when:

- Its history names an identity the serde cannot revive. The reason starts with `history names ...`. Add a [tombstone](retiring-interrupted.md#recovering-a-delete-first-deployment) with `@migrate_from`, then run `rewrite --apply`.
- The reason starts with `channel 'events' holds ... not as an event`. A serializer other than `NamespaceAwareSerde` wrote the entry, and its class no longer imports or no longer takes the stored fields. LangGraph's reader returns it as a `dict` or `None`, with no error, so no migration reaches it. A class that imports but that `allowed_msgpack_modules` of the serde does not list also gives a `dict`, for example under `LANGGRAPH_STRICT_MSGPACK=true`. Restore the class under its stored module and name. If the class imports, add it to `allowed_msgpack_modules`. If nobody needs the thread, delete it with the `delete_thread()` method of the checkpointer. Then run `rewrite --apply`. A plan reads such a thread as refused, never as unchanged, so a store with history that no release can read does not pass as clean.
- The reason starts with `the rewritten value ... holds an invalid ...`. A migration builds or replaces a pydantic model that fails the validation of its class. The message names the model, the event field and the error. Fix the migration so that the value validates, then run `rewrite --apply`. LangGraph's reader revives a stored model with its constructor, and on a validation error with `model_construct`, which validates nothing. So once the migration leaves the code, every read accepts the invalid value with no warning. The check reads only a model that a migration built or replaced, and calls `type(model)(**model.model_dump())`, the call of LangGraph's reader. A model that the store already held is stored unchanged. So a valid model that does not rebuild from its own dump, for example through a field alias, does not refuse the thread.
- It is paused on a dropped class, or a dropped event is pending dispatch on a paused thread. After the drop, the resumed task finds no event to handle and ends the thread with the answer lost. Abandon the thread first, with [`abandon()`](control-flow.md#ending-a-pause-without-answering-it-abandon).
- It holds a completed task write from a fanned-out sibling, and the rewrite would change that write. The checkpointer API cannot rewrite the write. Resume or abandon the thread first. When the thread has no pending interrupt, `resume()` and a plain `abandon()` refuse it. Call `abandon(config, require_interrupt=False)` instead. The reason names the task, the channel, and the identity or the index that the write holds. A write changes when it holds a record that a migration rewrites, an instance of a class in `drop=`, or an index into `events` when `drop=` removes an event from `events`. A completed write to `events` or `_pending` that holds an entry that is not an event is also refused, because a resume would append the entry to the log. Restore the class of the entry, or delete the thread. The rewrite keeps every other completed write as it is. A resume applies the write and does not run its task again.
- It holds a pending `__error__` write from a failed task. Run the thread again, or abandon it with `abandon(config, require_interrupt=False)`.
- A dropped instance sits inside a field of another event, or in a reducer value. Leave the class in place, or settle the thread.
- `get_tuple()` of the checkpointer returns a completed task write as another object than the one that the serde decoded. Or the read after the write finds a newer checkpoint, or a saver that broke a rule. See [Checkpointers](#checkpointers).

### Live traffic

The `rewrite` command can run while release N serves runs. It builds its own graph and serde in its own process, so it shares no serde state with the app. Release N writes only live shapes, but a run can still bring old bytes back. A run that read checkpoint C before the rewrite writes its next checkpoint after the rewrite. The new checkpoint keeps the channel versions of C for each channel that the run did not update. Those versions point at the old blobs, and the rewrite does not delete the old blobs. The next plan reads that thread as not clean, and the next pass rewrites it. Only the last plan decides exit 0. A run that holds C for longer than all passes can still write after exit 0. The plan of step 6 then reads the thread and exits 1, so release N+1 does not deploy.

WARNING: Do not let an older release serve runs while the command runs. An older release writes the old shape again, and it cannot read the new bytes. The fence stops the command under an older release, but it cannot stop an older release that still serves traffic. Finish the rolling deploy first.

The Python form `graph.event_store.rewrite_store()` has one more rule. It toggles per-instance state on the serde, like `tolerate_unresolved()`, so a concurrent run through the same serde instance corrupts the report. Call it on a graph whose serde instance serves no run, for example in a one-off process.

### Old blob versions

This step applies to `PostgresSaver` only. It stores each channel value as a blob row under its version, apart from the checkpoint row. The rewrite stores each changed channel under a new version and keeps the checkpoint id, so the store grows by the size of each rewritten channel. An earlier checkpoint of the thread can still name the old blob version. The command does not delete the old versions, because a run that started before the rewrite can still name them. See [Live traffic](#live-traffic). If a named blob were gone, the channel would read as empty, and no plan could see the loss.

WARNING: Do not run the statement while a run that started before the rewrite can still write. That run names an old version, and its checkpoint would lose the channel. The statement also deletes the old versions that the backup of release N names. After that, `restore` under release N exits 2 and changes nothing. Run the statement only when you no longer need to roll back release N.

Delete the old versions after step 6 exits 0. Then no run that started before the rewrite still runs:

```sql
DELETE FROM checkpoint_blobs AS b
WHERE NOT EXISTS (
    SELECT 1 FROM checkpoints AS c
    WHERE c.thread_id = b.thread_id AND c.checkpoint_ns = b.checkpoint_ns
      AND c.checkpoint -> 'channel_versions' ->> b.channel = b.version
) AND EXISTS (
    SELECT 1 FROM checkpoints AS c
    WHERE c.thread_id = b.thread_id AND c.checkpoint_ns = b.checkpoint_ns
      AND c.checkpoint -> 'channel_versions' ->> b.channel > b.version
);
```

The statement deletes a blob version only when no checkpoint names it and a checkpoint names a newer version of the same channel. A running graph writes a blob before the checkpoint that names it, and that blob is newer than each version that a checkpoint names, so the statement keeps it. The versions of `PostgresSaver` are zero-padded, so a text comparison orders them. A checkpoint that a prune of your own deleted also leaves blob versions that no checkpoint names. The statement deletes them too.

### Backup and restore

This section applies to `PostgresSaver` only. The three commands read and write the store in SQL through a private method of the saver, so install the saver through the `postgres` extra. See [Checkpointers](#checkpointers). A copy through `get_tuple()` cannot work, because the serde migrates each record on read. The copy names the columns that the backup table and the live table share, so it still runs after an upgrade of the saver adds a column. Each command runs in one transaction, through the connection of the saver. A report goes to stdout, and an error to stderr.

| Command | What it does | Exit 0 output | Exit 2 |
|---|---|---|---|
| `backup <target>` | Copies the live set into the backup of the running code. It drops no table | The rows that the backup kept and added | A saver other than `PostgresSaver`, a target with `threads=`, a store with a backup and no `checkpoints` table, or a failed statement. Nothing changed |
| `restore <target>` | Puts back the backup of the running code | The rows that it put back and the writes that it removed | The errors of `backup`, a store with no backup of the running code that is not an empty store, or a missing blob version. The message of a missing backup names each backup that the store holds. Nothing changed |
| `drop-backups <target>` | Drops each backup except the backup of the running code | One line per dropped backup, or `nothing to drop` | The errors of `backup`. Nothing changed |

```text
lge backup: lge-m1-dbc7afada42b30d1 kept 1 checkpoint and 1 write, added 1 checkpoint and 0 writes.
lge restore: lge-m1-dbc7afada42b30d1 put back 2 checkpoints and 1 write, and removed 1 write that the backup does not hold.
lge drop-backups: dropped lge-m1-4d1c8f0e2b7a9356.
```

The Python forms are `backup_live_set(target) -> BackupOutcome`, `restore_live_set(target) -> RestoreOutcome` and `drop_backups(target) -> tuple[str, ...]`, the fingerprints of the dropped backups. Each raises `TargetError` where its command exits 2.

| Table | It holds |
|---|---|
| `checkpoints_bak_<fp>` | The latest row of `checkpoints` for each thread and checkpoint namespace |
| `checkpoint_writes_bak_<fp>` | Each row of `checkpoint_writes` of those checkpoints, as of the last run of `backup` |

`<fp>` is the migration fingerprint of the running code, with `_` for each `-`. For example, `lge-m1-dbc7afada42b30d1` gives `checkpoints_bak_lge_m1_dbc7afada42b30d1`. The backup copies the live set, as defined under [Rewriting the live set](#rewriting-the-live-set). The rewrite overwrites only these rows. `PostgresSaver.put()` stores each changed channel under a new version, and an insert into `checkpoint_blobs` never replaces a stored row. So the old blob versions stay in the store, and the backup does not copy them. A rerun of `backup` under the same fingerprint, for example a hotfix with the same migrations, keeps each row that the backup holds. It adds the latest row of each thread that appeared or changed since, with the pending writes of that row. It also adds each pending write that a run added since to a checkpoint that the backup holds. It keeps each write that the backup holds, also when a rewrite changed the live write since. A thread that changed has two rows in the backup, and the restore puts back both, each under its own key.

`restore` runs in a container of the release that serves, before the rollback. A rollback to a release with the same migrations needs no restore. The restore overwrites each row that the backup holds and the store still holds under the same key. For a checkpoint it writes `checkpoint` and `metadata`. For a pending write it writes `channel`, `type` and `blob`. It deletes each pending write of a backed-up checkpoint whose key, `thread_id`, `checkpoint_ns`, `checkpoint_id`, `task_id` and `idx`, the backup does not hold. A run added that write after the last `backup`. It does not change a checkpoint that a run wrote after the backup, because that checkpoint has a new key. Before it writes, the restore checks the blob versions that it puts back. A checkpoint names a version of each channel in `checkpoint -> 'channel_versions'`. `checkpoint_blobs` holds a blob under the key `thread_id`, `checkpoint_ns`, `channel` and `version`. A primitive value stays in `checkpoint -> 'channel_values'`, and a consumed channel has no value, so not every channel has a blob. The restore checks each channel whose version in the backup differs from the version in the live row, and whose value is not inline. When `checkpoint_blobs` no longer holds such a version, a prune deleted it. The restore then exits 2, names the count of missing versions, and changes nothing. Before the check, the restore takes a `SHARE` lock on `checkpoint_blobs` until the end of its transaction. A running prune finishes first, and the check sees what it deleted. A new prune waits until the restore commits. The lock also holds back each run that writes a blob, so stop the runs before the restore. For a small live set, the wait is some milliseconds.

WARNING: A restore cannot make a checkpoint readable that the new release wrote after it started to serve. That checkpoint holds the shapes of the new release. Step 3 copies the rows before the new release serves. Step 4 can copy a row or a pending write that the new release wrote. The restore then puts back that row or write, and its count includes it. After the rollback, the older release cannot read that thread. Before the restore, stop each run of the new release. For each thread that the new release changed, add a tombstone in the older release, or delete the thread.

`drop-backups` drops the backup tables of each fingerprint except the fingerprint of the running code, in the current schema of the store. It drops only the tables whose name has the form `checkpoints_bak_lge_m<n>_<16 hex digits>` or `checkpoint_writes_bak_lge_m<n>_<16 hex digits>`. Step 4 runs it after the rewrite. So the store holds the backup of the release that serves, and a rollback of that release can restore it. A rollback of two releases cannot restore, because step 4 of release N+1 drops the backup of release N. When release N+1 holds the same migrations as release N, both releases write to one backup, and `drop-backups` keeps it. Over a series of releases with the same fingerprint, the backup grows with each deploy. To limit it, drop the two tables of that backup with the tools of the database when no rollback needs them. The next `backup` then starts a new backup.

The three commands take locks on the checkpoint tables. To limit the wait, set `lock_timeout` and `statement_timeout` in the connection options of the store, for example `?options=-c%20lock_timeout%3D5s` in the URL. Then, when another session holds a lock on a checkpoint table, the command fails fast and changes nothing. A target with `threads=` declares a store that other graphs can share. `backup`, `restore` and `drop-backups` exit 2 for it, before they open the store. The commands act on every row of the store, and the backup name does not name the graph. `drop-backups` of one graph would drop the backup of another graph, and two graphs with the same fingerprint would write to one backup. For a shared store, back up the store with the tools of its database.

### A store with no checkpoint table

A store before the first `setup()` of its saver has no `checkpoints` table, for example a new database before release N first serves. Each command reads such a store as an empty store, and creates no table:

| Command | On a store with no `checkpoints` table |
|---|---|
| `rewrite` | Plans 0 threads. The verdict is `clean`, so the plan and `--apply` exit 0 |
| `backup` | Copies nothing and exits 0 with the counts `0` |
| `restore` | Puts back nothing and exits 0 with the counts `0` |

A store with a backup and no `checkpoints` table is not an empty store. `backup` and `restore` then exit 2 and name each backup that the store holds.

When the connection finds no `checkpoints` table, the command looks for one in each schema of the database. When a schema holds one, the command exits 2 and names the schema. The search path of the connection does not include that schema, or the role has no `USAGE` privilege on it.

WARNING: A connection to the wrong database finds no `checkpoints` table. The command cannot tell it from a new store, so it reads an empty store and exits 0. Make sure that the URL of `store` names the database of the app.

A backup of an empty store creates no table. After release N creates the tables, `restore` under release N exits 2 until step 4 runs `backup` again. Release N wrote each checkpoint of that store, so no restore is necessary.

### Subgraph checkpoints

A graph that a handler invokes on the same checkpointer writes its checkpoints under a subgraph checkpoint namespace. The command plans and rewrites the latest checkpoint of each subgraph checkpoint namespace, the same as a root thread. So a completed subgraph run does not keep a migration in the code. A refused subgraph checkpoint prints its namespace and its reason. A completed task write that the rewrite would change refuses a subgraph checkpoint. When the subgraph checkpoint is paused on an interrupt, resume the root thread, then run `rewrite --apply` again. The resume runs the paused subgraph task again under the same namespace, and that run applies the write. A subgraph checkpoint that is paused on a class in `drop` also clears when the root thread resumes. `abandon()` on the root thread clears neither.

A subgraph checkpoint that holds a pending `__error__` write from a failed task is refused on every pass. When the subgraph checkpoint is not paused, a completed task write that the rewrite would change also refuses it on every pass. `abandon()` and `resume()` reach only the root checkpoint namespace. A new run of the root thread writes under a new namespace and leaves the old checkpoint. So no call of this library clears that checkpoint, and `rewrite` cannot exit 0 while it stays. This is a known limit until the refusal rule for an `__error__` write changes. WARNING: `delete_thread()` deletes every checkpoint of the thread, the root checkpoint namespace included. Delete only a thread that nobody needs. The operator has two choices:

- If nobody needs the thread, delete it with the `delete_thread()` method of the checkpointer. Then run `rewrite --apply` again.
- Otherwise hold the migrations that the checkpoint needs with `baseline hold`, one release at a time.

## Rewriting the live set

Every migration above is read-side. The stored bytes keep the historic identity and the old field shape until something writes the checkpoint again. A thread at rest is never written again. So its `@migrate_from`, `@transform_fields` or `@split_event` must stay for ever. `graph.event_store.plan_rewrite()` and `graph.event_store.rewrite_store()` rewrite the stored bytes. For a deploy step, use the `rewrite` command. See [Lifecycle of a read-time migration](#lifecycle-of-a-read-time-migration).

The live set is the latest checkpoint of each thread in each checkpoint namespace, plus its pending writes. `rewrite_store()` reads one checkpoint namespace per call. The default `checkpoint_ns=""` is the root checkpoint namespace. Pass `checkpoint_ns=` to read a subgraph checkpoint namespace. It reads each checkpoint through the serde of the checkpointer, so every rename, transform, split and fill applies. It writes the result back through `put()` of the checkpointer, under the same checkpoint id, with a new version for each channel the rewrite touched. `drop=` names event classes whose stored instances leave the `events` and `_pending` channels. `plan_rewrite()` does the same reads and the same verification, and writes nothing. The caller names the threads with `thread_ids=`. The argument is required, and the library does not walk the store. See [Finding candidates server-side](retiring-interrupted.md#finding-candidates-server-side). A listed id with no checkpoint is reported as refused, so a typo stays visible.

```python
...  # graph, Order and thread_ids come from your app
report = graph.event_store.plan_rewrite(drop=(Order.ApprovalRequired,), thread_ids=thread_ids)
print(report)
assert not report.refused

report = graph.event_store.rewrite_store(drop=(Order.ApprovalRequired,), thread_ids=thread_ids)
assert not report.refused
```

Both return a `RewriteReport` with one `ThreadRewrite` per thread. `status` is `"rewrite"`, `"unchanged"` or `"refused"`. `report.applied` says whether the rewrite ran. `"rewrite"` in a plan means that the thread needs a write. In an applied report it means that the thread was written. `migrated` lists each `(stored, live)` identity pair the table rewrote. `dropped` counts the log entries `drop=` removed, per identity. `reason` says what to do with a refused thread. The reasons are listed under [A checkpoint that a plan refuses](#a-checkpoint-that-a-plan-refuses). `checkpoint_ns` is the checkpoint namespace the call read, and `location` is the thread id with that namespace when it is not the root checkpoint namespace. `str(report)` prints one summary line, then one line per thread that `drop=` removes entries from, then one line per refused thread. The async forms are `aplan_rewrite()` and `arewrite_store()`. They take the same arguments. An answered interrupt also sits in `Resumed.interrupted`. When that interrupt is dropped, the rewrite sets the field to `None`. The `Resumed` event stays, and `dropped` does not count the cleared field. `drop=` matches the stored identity exactly. A subclass instance stays in place.

WARNING: The checkpoint row is overwritten in place, so take a backup first. For a `PostgresSaver`, the `backup` command takes it. See [Backup and restore](#backup-and-restore). An old pod cannot read the new bytes, so run the rewrite only after every pod runs the release that holds the migration. For the rule on a shared serde instance, see [Live traffic](#live-traffic).

What the rewrite does not do:

- A historic checkpoint keeps the old bytes. After the class is deleted, `get_state_history()` cannot read it.
- `drop=` removes the stored event only. Reducer state is not recomputed. Use [`replay_reducer`](event-migrations.md#recovering-with-replay_reducer).
- An event inside a non-Event container field, a dataclass or a pydantic model, has no migratable identity. The rewrite does not reach it.
- `legacy_write=True` is refused with `ValueError`. Under `legacy_write` the rewrite would store the historic identity again.
