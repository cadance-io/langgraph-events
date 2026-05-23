# Checkpointer and Graph Evolution

> *This documents current behavior. Details may change between versions — there are no stability guarantees yet.*

See also: [Event migrations](event-migrations.md) — full migration story for renamed / relocated event classes and added required fields.

Existing threads survive graph modifications via graceful degradation — no crashes, but some changes have silent side effects.

## What's Safe

| Change | Behavior |
|---|---|
| **Add a handler** | `dispatch()` is rebuilt from current handlers; new handler participates immediately. |
| **Add an event type** | New events can be emitted and matched normally. |
| **Remove an event type** | Existing events stay in the log but no handler matches them — inert. |

## What to Watch Out For

| Change | Risk |
|---|---|
| **Remove a handler** (normal checkpoint) | Events only the removed handler subscribed to become undeliverable. Graph halts early — no crash, incomplete execution. |
| **Remove a handler** (interrupted checkpoint) | If the graph was paused inside the removed handler via `Interrupted`, `graph.resume(value)` silently does nothing. The pending Send to the missing node is dropped. |
| **Rename a handler** | Same as remove + add. An `Interrupted` checkpoint targeting the old name is lost. |
| **Rename / relocate an Event class** | Old checkpoints fail revival under the default serde. See [Event migrations](event-migrations.md). |
| **Add a reducer** | New reducer starts cold — misses default values and all historical projections. Only post-resume events contribute. |
| **Remove a reducer** | Channel data is silently dropped from the checkpoint. |

## Best Practices

- Don't rename handlers with active interrupted threads — the function name is baked into the checkpoint.
- Treat reducer addition as a fresh start; for full history, use a new thread.
- Prefer additive changes (add handlers/events; removal is safe only with no in-flight threads).
- Use a new `thread_id` after structural changes to avoid all edge cases.
