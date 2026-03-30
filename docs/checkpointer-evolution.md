# Checkpointer and Graph Evolution

> *This documents current behavior. Details may change between versions — there are no stability guarantees yet.*

When using a checkpointer (`MemorySaver`, `SqliteSaver`, etc.), existing threads retain their checkpoint state across invocations. If you modify the graph between invocations on the same thread, LangGraph handles mismatches through **graceful degradation** — no crashes, but some changes have silent side effects.

## What's Safe

| Change | Behavior |
|---|---|
| **Add a handler** | Safe. `dispatch()` is rebuilt from current handlers, so the new handler participates immediately. |
| **Add an event type** | Safe. New events can be emitted and matched normally. |
| **Remove an event type** | Safe. Existing events stay in the log but no handler will match them — they're inert. |

## What to Watch Out For

| Change | Risk |
|---|---|
| **Remove a handler** (normal checkpoint) | Events that *only* the removed handler subscribed to become undeliverable. The graph halts early — no crash, but incomplete execution. |
| **Remove a handler** (interrupted checkpoint) | If the graph was paused inside the removed handler via `Interrupted`, `graph.resume(value)` silently does nothing. The pending Send to the missing node is dropped. The human-in-the-loop flow breaks without error. |
| **Rename a handler** | Same as remove + add. If an `Interrupted` checkpoint targeted the old name, the resume is lost. |
| **Add a reducer** | The new reducer starts cold — it **misses its default values and all historical event projections**. Only events added after the resume point contribute. |
| **Remove a reducer** | The reducer's channel data is silently dropped from the checkpoint. |

## Best Practices

1. **Don't rename handlers with active interrupted threads.** If a thread is paused at an `Interrupted` checkpoint, the handler's function name is baked into the checkpoint. Renaming it silently breaks resume.
2. **Treat reducer addition as a fresh start.** A newly added reducer on an existing thread won't replay historical events. If you need the full history, start a new thread.
3. **Prefer additive changes.** Adding handlers and event types is always safe. Removing them is safe only if no in-flight threads depend on them.
4. **Use separate `thread_id`s after structural changes.** The simplest way to avoid all edge cases is to use a new thread for a modified graph.
