# Checkpointer and Graph Evolution

> *This documents current behavior. Details may change between versions — there are no stability guarantees yet.*

See also: [Event migrations](event-migrations.md) — the full migration story for renamed / relocated event classes and added required fields, including the coverage gates that fail CI when a baselined event identity is no longer covered.

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
| **Remove a handler** (interrupted checkpoint) | If the graph was paused inside the removed handler via `Interrupted`, `graph.resume(value)` raises `UnresumableError` by default (set `EventGraph(on_unresumable="halt"\|"warn")` to handle it non-fatally). Declare `@on(previously=...)` to keep it resumable. See [Handler renames](event-migrations.md#handler-renames). |
| **Rename a handler** | An `Interrupted` checkpoint targeting the old node is lost — unless you declare `@on(previously="old_name")`, which keeps it resumable. See [Handler renames](event-migrations.md#handler-renames). |
| **Rename / relocate an Event class** | Old checkpoints fail revival under the default serde — author a migration; the [coverage gates](event-migrations.md#coverage-gates) catch an uncovered rename in CI before deploy. |
| **Add a reducer** | New reducer starts cold — misses default values and all historical projections. Only post-resume events contribute. |
| **Remove a reducer** | Channel data is silently dropped from the checkpoint. |

## Best Practices

- Renaming a handler with active interrupted threads is safe **if** you declare `@on(previously=...)` — for an inline `Command`, the `previously: ClassVar = (...)` class attribute — (or pin `@on(node_name=...)` up front); the handler coverage gate flags an undeclared rename in CI.
- Treat reducer addition as a fresh start; for full history, use a new thread.
- Prefer additive changes (add handlers/events; removal is safe only with no in-flight threads).
- Use a new `thread_id` after structural changes to avoid all edge cases.
