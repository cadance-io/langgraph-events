# Checkpointer and Graph Evolution

When using checkpointers (`MemorySaver`, `SqliteSaver`, and others), existing threads keep prior checkpoint state. Structural graph changes can alter behavior for in-flight threads.

## Generally Safe

- Adding handlers
- Adding event types
- Removing unused event types

## Risky Changes

- Removing or renaming handlers used by interrupted checkpoints
- Adding reducers to existing threads (new reducer starts without historical projection)
- Removing reducers (channel data disappears from checkpoint state)

## Best Practices

1. Avoid renaming handlers while interrupted threads are active.
2. Treat reducer additions as a fresh-thread migration.
3. Prefer additive changes over destructive structural changes.
4. Use new `thread_id` values after major graph structure updates.
