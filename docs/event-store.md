# Event Store

`langgraph_events.store` keeps the event log as the save file, with no
checkpointer. Use it when the log must survive a crash in the middle of a
run, and when another program must read the log without this library.

## The parts

| Part | Role |
|---|---|
| `Record` | One stored event: `module`, `type` (the class `__qualname__`) and `fields` |
| `EventStore` | The port: `append(records)` and `load()` |
| `MemoryEventStore` | An in-memory store for tests. It keeps JSON text |
| `JsonlEventStore(path)` | One JSON record per line, with a lock and `fsync` |
| `EventCodec` | Events to records and back, through the serde migrations |
| `EventStream` | Runs an `EventGraph` over a store, one superstep at a time |

## Running a graph over a log

```python
from pathlib import Path

from langgraph_events import EventGraph, IntegrationEvent, on
from langgraph_events.store import EventCodec, EventStream, JsonlEventStore

class Request(IntegrationEvent):
    text: str


class Recorded(IntegrationEvent):
    request: Request


@on(Request)
def record(event: Request) -> Recorded:
    return Recorded(request=event)


@on(Recorded)
def observe(event: Recorded) -> None:
    return None


path = Path("books/getting-started.jsonl")
path.parent.mkdir(parents=True, exist_ok=True)
with JsonlEventStore(path) as store:
    graph = EventGraph([record, observe])
    stream = EventStream(graph, store, EventCodec())
    added = stream.invoke(Request(text="hello"))

print([type(event).__name__ for event in added])
```

The parent directory must exist before `JsonlEventStore(path)` runs.
`JsonlEventStore` does not create parent directories.

`EventStream` is synchronous only. It has no async API. It loads the log once. `invoke` takes the same seed as
`EventGraph.invoke`: one event or a list. It gives the graph the whole log,
the cached reducer values and the seed. A handler's injected `EventLog` is
the stored log, then the events of this turn. The events of each superstep
reach the store before the next superstep runs. A crash loses at most the
superstep that was running. `invoke` returns only the events of this turn.
`log` is the whole log.

The graph must have no checkpointer. `EventStream` raises `ValueError`
otherwise. A handler that returns an `Interrupted` event raises
`InterruptWithoutCheckpointerError`: the pause has no place to wait. Record the
pause as an event, and continue on a later `invoke`.

When a handler raises, the events stored before it stay stored. The cached
reducer values fold them.

An append error makes an `EventStream` unusable. The store can write bytes
before it reports an error. Do not retry with that stream. Create a new codec
and stream, then reload the log from the store. `EventStore.append()` does not
promise an atomic append.

`EventCodec` records positions while `encode()` runs. A direct caller that
gets an append error must discard that codec. Create a new codec and decode
the store log before the next encode.

The store persists events only. It does not persist causation metadata. A
reloaded log has no causes. A live handler can inspect causes from its current
turn, but a later `EventStream` cannot recover them.

## The file

Each line is one JSON object. A reader needs only the standard library:

```python
import json

rows = [json.loads(line) for line in open("books/alice.jsonl")]
```

A line counts only once its newline is on disk. A reader must ignore a last
line with no newline: a writer crashed, or is still writing it.

`JsonlEventStore` takes an exclusive `flock` on `<path>.lock` for its whole
lifetime. A second store on the same path raises `StoreLockedError`, in the same
process or in another one. Call `close()`, or use the store as a context
manager, to release the lock. `JsonlEventStore` needs a POSIX system.

Each `append` writes, flushes and calls `fsync`. The first `append` after a
crash cuts the torn last line, so the next record starts on its own line.

## Field values

`fields` holds JSON values. Four single-key objects carry what JSON cannot:

| Stored | Revives as |
|---|---|
| `{"$ref": 3}` | The event at position 3 of the same log, the same object |
| `{"$tuple": [...]}` | A `tuple` |
| `{"$dict": {...}}` | A `dict` that has a key that starts with `$` |
| `{"$repr": "..."}` | A `str`: the `repr()` of a value outside this model |

WARNING: a value outside JSON, such as the exception in `HandlerRaised`,
revives as its `repr()` string, not as the object. Keep event fields in
JSON values and links to earlier events.

A `$ref` must point to an earlier event. A forward, negative or non-integer
pointer raises `ValueError` naming the record.

## Migrations

`EventCodec(migrations, namespaces=..., events=...)` builds a
`NamespaceAwareSerde`. Each record goes through
`NamespaceAwareSerde.revive_event`, so `@migrate_from`, `@backfill`,
`@transform_fields` and `@split_event` apply to a log line as they apply to a
checkpoint. See [Event migrations](event-migrations.md).

The codec resolves a class by `(module, qualname)`: its registry first, then
the serde scope, then the import walk. A library system event resolves under
its own module or under the package name `"langgraph_events"`.

A record that does not revive raises `ValueError` with its position, its
identity and the remedy. A `SystemExit` from a replay function passes through. To read a log whose classes are gone, decode inside
`codec.tolerate_unresolved()`. Each such record then decodes to an
`UnrevivedIdentity`. Do not run an `EventStream` over that log.

## Classes defined at run time

A class that a handler builds at run time does not import. Give the codec a
`replay` map from the event that defines the class to a function that
returns it:

```python
def replay_defined(event):
    return [make_concept_class(event.name, event.fields)]

codec = EventCodec(events=[Defined], replay={Defined: replay_defined})
```

The codec calls the function as soon as that event decodes. It registers each
class before the next record decodes. The function must be the factory that
the live handler calls. It runs no handler. `codec.register(cls)` registers
one class directly.

Use one codec per log. The codec remembers the position of each event, so
`encode` can write a `$ref`. A failed decode restores its event book and replay
registry to the state before that decode.

## Reducer values

`EventStream` folds the log once at load with `BaseReducer.advance`, and
then folds only the new events of each turn. A channel value then equals the
live channel value, because the router and the settle path fold every event
they append. See [Reducers](reducers.md#rebuilding-a-channel-from-the-log).
