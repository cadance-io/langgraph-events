# Retiring an Interrupted subclass

Delete an `Interrupted` subclass from the codebase once no live checkpoint still references it. `graph.event_store.abandon(config)` or `.aabandon()` settles one paused thread without an answer. See [Ending a pause without answering it](control-flow.md#ending-a-pause-without-answering-it-abandon). `graph.event_store.threads_paused_on(EventClass, thread_ids=...)` or `.athreads_paused_on()` finds the paused threads among the ids you pass. See [Finding candidates server-side](#finding-candidates-server-side).

Both methods read the checkpoint of each thread directly, not the compiled topology of the graph. They survive two deletions:

- The **handler** that produced the interrupt is already removed. The class still imports, so the interrupt revives and a class filter matches it. This is the common order.
- The **class** is already deleted. The interrupt cannot revive. With no filter, `threads_paused_on()` still returns the thread. With a class filter, it matches nothing. `abandon()` still settles the thread, and records the last known qualname of the interrupt in `discarded`.

!!! warning "`threads_paused_on()` and `abandon()` cover paused threads only"
    A thread that already answered the interrupt holds the retired class in its settled history. `threads_paused_on()` does not find it. A read of its history after the class is deleted raises `Cannot revive`. `graph.event_store.unrevivable_threads(thread_ids=...)` finds it. It reads the latest checkpoint of each listed thread through the tolerant path of the serde. It maps each thread id to the qualnames it can no longer revive. Such a qualname can sit in the settled history, in a pending interrupt, in a completed sibling write, or nested in a field of a live event. It needs a `NamespaceAwareSerde` on the checkpointer and raises `ValueError` otherwise. It reports nothing while the class still imports, so run it after the class is deleted, against the real store. Recover each reported thread with a [tombstone](#recovering-a-delete-first-deployment). `abandon()` and `aabandon()` raise `ValueError`, naming the thread and the qualnames, when the settled history holds a deleted class. They settle only a thread whose sole unrevivable identity is its pending interrupt. `NamespaceAwareSerde` never stores a placeholder. A write that holds one raises `ValueError` naming the identity.

## Finding candidates server-side

`threads_paused_on()`, `unrevivable_threads()`, `plan_rewrite()`, `rewrite_store()` and their async forms require `thread_ids=`. A missing argument, `None` or a bare `str` raises `TypeError`. The library does not walk the store. Only the store can filter by latest checkpoint and by the `__interrupt__` channel. `thread_ids=` accepts ids from any query, any driver, sync or async.

`threads_paused_on()`, `unrevivable_threads()` and `abandon()` read the root checkpoint namespace only. `plan_rewrite()` and `rewrite_store()` read the namespace in `checkpoint_ns=`, the root by default. So every query filters on `checkpoint_ns = ''`. If one checkpointer stores more than one graph, keep only the thread ids of this graph. Every thread id, on `PostgresSaver` and `SqliteSaver`:

```sql
SELECT DISTINCT thread_id FROM checkpoints WHERE checkpoint_ns = ''
```

On `InMemorySaver`, `get_tuple()` on an unknown thread id leaves an empty `storage[thread_id][""]` entry, so filter it out:

```python
...  # saver: the InMemorySaver of the graph
ids = [tid for tid, ns in saver.storage.items() if ns.get("")]
```

The threads paused on one class come from the `__interrupt__` writes of the latest root checkpoint of each thread. The retired qualname is matched as bytes inside the msgpack blob. On `PostgresSaver`:

```sql
WITH latest AS (
  SELECT DISTINCT ON (thread_id) thread_id, checkpoint_id FROM checkpoints
  WHERE checkpoint_ns = '' ORDER BY thread_id, checkpoint_id DESC
)
SELECT DISTINCT w.thread_id FROM checkpoint_writes w
JOIN latest l ON w.thread_id = l.thread_id AND w.checkpoint_id = l.checkpoint_id
WHERE w.checkpoint_ns = '' AND w.channel = '__interrupt__' AND w.type = 'msgpack'
  AND position(convert_to('Order.ApprovalRequired', 'UTF8') in w.blob) > 0
```

On `SqliteSaver`, the writes table is `writes`, the blob column is `value`, the latest checkpoint is `MAX(checkpoint_id)` grouped by `thread_id`, and the match is `instr(w.value, CAST(? AS BLOB)) > 0` with the qualname as the bound parameter.

The byte match is a candidate filter only. It also matches a longer qualname, for example `Order.ApprovalRequiredV2`. Confirm each candidate through the serde: pass the ids to `threads_paused_on()`, step 2 of the [sequence](#sequence).

To check one thread, pass `thread_ids=[tid]`. Do not use `graph.get_state(config).is_interrupted` for this check. It reads the compiled topology of the graph. Once the handler is deleted, it is `False`, and `GraphState.interrupted` is `None`, on a thread that is still paused.

## Sequence

```python
...  # conn: a psycopg or sqlite3 connection to the checkpointer database.
# EventClass: the Interrupted subclass to retire.
ALL_THREAD_IDS = "SELECT DISTINCT thread_id FROM checkpoints WHERE checkpoint_ns = ''"

# Step 1. Every thread id in the store. An empty list proves nothing.
thread_ids = [row[0] for row in conn.execute(ALL_THREAD_IDS)]
assert thread_ids, "the query returned no thread ids"

# Step 2. Settle each paused thread. Step 3. Verify.
for config in graph.event_store.threads_paused_on(EventClass, thread_ids=thread_ids):
    graph.event_store.abandon(config, reason="retiring EventClass")
assert graph.event_store.threads_paused_on(EventClass, thread_ids=thread_ids) == []

# Step 4. Plan the rewrite and review it. The plan reaches an answered
# thread, which holds the class in its settled history.
report = graph.event_store.plan_rewrite(drop=(EventClass,), thread_ids=thread_ids)
print(report)
assert not report.refused

# Step 5. Take a store backup first. Then rewrite.
report = graph.event_store.rewrite_store(drop=(EventClass,), thread_ids=thread_ids)
assert not report.refused

# Step 6. Delete the class, run `baseline update` and deploy. The update
# moves the identity to the `retired` list, with its last recorded fields.
# Deleting the handler in the same change fails the `handlers` rule with
# `HandlerMissing`. This is expected. A handler name is not retired. The
# update drops it from `handlers` at once.

# Step 7. After the deploy, query the ids again and verify against the real
# store. The gate reads no store. Then delete the `retired` entry from the
# baseline file by hand. No command removes it. `baseline update --previous`
# removes a retired entry only for the source of a deleted shipped rename.
thread_ids = [row[0] for row in conn.execute(ALL_THREAD_IDS)]
assert thread_ids, "the query returned no thread ids"
assert graph.event_store.unrevivable_threads(thread_ids=thread_ids) == {}
```

The [migration gate](migration-gate.md#the-migration-gate) keeps checking a retired identity. Its `coverage` and `revive` rules fail on it until a migration covers it. After the hand edit of step 7, the gate does not check the identity, and every rule passes whether or not a settled thread still cannot revive. WARNING: After the hand edit the gate cannot see a thread that still cannot revive. Run `unrevivable_threads()` against the real store before the hand edit.

## Recovering a delete-first deployment

Condition: the class was deleted before every paused thread was settled. `threads_paused_on()` and `abandon()` recover the paused case on their own. A thread that had already answered the interrupt needs a tombstone. In a follow-up release, map the dead identity onto a tombstone class. `transform=discard_fields` returns `{}`, so the tombstone needs none of the retired fields. Give the transform a module-level name, because the gate reports a lambda. Keep the tombstone an `Interrupted` when the retired class was one, so that `threads_paused_on()` and `abandon()` match it. Nest it inside the same `Namespace`:

```python
from typing import Any

from langgraph.checkpoint.memory import MemorySaver

from langgraph_events import Command, DomainEvent, EventGraph, Interrupted, Namespace
from langgraph_events.serde import migrate_from


def discard_fields(kw: dict[str, Any]) -> dict[str, Any]:
    return {}


class Order(Namespace):
    class Approve(Command):
        class Approved(DomainEvent):
            pass

        def handle(self) -> "Order.Approve.Approved":
            return Order.Approve.Approved()

    @migrate_from("Order.ApprovalRequired", transform=discard_fields)
    class RetiredApprovalGate(Interrupted):
        pass


graph = EventGraph.from_namespaces(Order, checkpointer=MemorySaver())
```

Against a store that holds a paused thread and an answered thread, `threads_paused_on()` lists the paused thread, matched against the live `RetiredApprovalGate`. `abandon(config)` settles it and records `discarded="Order.RetiredApprovalGate"`. The history of the answered thread revives too. `in_module=` defaults to the `__module__` of the decorated class. Pass it when `Order.ApprovalRequired` lived in another module.

!!! note "Two conditions on `from_namespaces(...)`"
    It keeps a `NamespaceAwareSerde` that the checkpointer already carries. A checkpointer object reused from an earlier graph in this process never gets the tombstone, with no error and no warning. Give the recovery graph a fresh checkpointer object on the same store, or build the serde yourself. It wires `Order` into the serde only because a handler still subscribes to or produces something inside it, the inline command handler of `Order.Approve` above. The tombstone alone does not add `Order`. If no handler touches the namespace of the tombstone, or the tombstone lives at module scope, pass the tombstone in `retired=`. See [Retired events](concepts.md#retired-events):

```python
...  # Order and discard_fields as above, without the nested tombstone


@migrate_from("Order.ApprovalRequired", transform=discard_fields)
class RetiredApprovalGate(Interrupted):
    pass


graph = EventGraph.from_namespaces(
    Order, checkpointer=checkpointer, retired=[RetiredApprovalGate]
)
```

The serde that `from_namespaces(...)` wires reads `RetiredApprovalGate`. `EventTopology.from_namespaces(Order, retired=[RetiredApprovalGate]).serde()` is the same serde for an `EventStore`.

**Deleting the tombstone.** The `@migrate_from` ships in the release that adds it. In the next release, the `release` rule reports its `rename` and `transform` keys as `SurvivedRelease`. While a paused thread still needs the tombstone, hold both keys with `baseline hold`, once in each release. Once every paused thread is settled, run steps 4 and 5 of the [sequence](#sequence) with `drop=(Order.RetiredApprovalGate,)`, then verify `graph.event_store.unrevivable_threads(thread_ids=thread_ids) == {}`. In the next release, delete the tombstone and its `@migrate_from`, then run `baseline update --previous <file>`. The update removes the `retired` entry of `Order.ApprovalRequired` and moves the tombstone class to `retired`. Delete that entry by hand, as in step 7.

**Onto a sibling instead of a tombstone.** A sibling is a live class that already stands for the retired one. When every field of the sibling has a default, put the same decorator on the sibling. The stored fields are discarded, and the sibling constructs from its defaults. Do not map a stored field onto a sibling field that means something else, for example a message id onto an order id.
