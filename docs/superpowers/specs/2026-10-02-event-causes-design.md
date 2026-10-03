# Event causes: record which handler produced each event, and from which event

> Design spec, 2026-10-02. Brainstormed with the user, then checked by parallel
> subagents: a throwaway spike of the state channel, a review of logs that derive from a
> log, a review of checkpoint evolution and naming, a study of a client that persists its own log, and
> a survey of event-sourcing practice. Implementation follows TDD per project conventions.

## Context

When a handler returns an event, the framework knows two facts: the handler that ran,
and the event that the handler was called with. Today the framework drops both facts
when it appends the result to the log. After the run, no query can say which handler
produced an event. The log and the static graph cannot recover the fact either. Two
policies that react to one event type and return one command type give the same log.

A client needs the fact, for example to show how often each handler fired, or what an
event led to.

The Reflection spec (`2026-07-27-reflection-design.md`) cut a *heuristic* causation
engine, because the API must never guess. This design does not guess. It *records* the
cause at dispatch, where the cause is a fact. The Reflection design notes already
expect this: "If events later carry actor or provenance fields, `get` and `evidence`
surface them automatically."

## Decisions

1. **The cause lives next to the log, not in the event.** Every event store that we
   surveyed keeps causation in metadata, outside the event payload: KurrentDB
   (`$causationId`), Marten, Axon, Rails Event Store and NServiceBus. Event classes,
   constructors and equality do not change. A mixin that added fields to the event was
   spiked and rejected: it changed the payload, and a checkpoint copied the whole cause
   chain into each event.
2. **The framework records the cause. The user does not.** This is the shared
   convention of the surveyed tools.
3. **The public API is event in, event out.** A `Cause` holds the source *event*, not a
   log index. No index crosses from one log to another, so a log that derives from a log
   (`after`, `before`, `select`) answers correctly with no extra type.
4. **Storage is index based.** The state channel stores an index, because an index
   survives every serializer and a checkpoint keeps no object identity.
5. **`Reflection` answers by root index.** This keeps its rule "root indices only".
6. **Correlation is derived, not stored.** The root of a flow is the end of the cause
   chain, so `flow()` computes it on demand.

## Public API

`log.cause(event)` states the truth about an event's origin, one case per type:

```python
@dataclass(frozen=True)
class Cause:
    source: Event   # the event that the handler was called with ("causation ID")
    via: str        # the handler node name (HandlerMeta.node_name)


class UnknownCause:                 # base of every unknown case
    reason: str                     # one plain sentence


@dataclass(frozen=True)
class NotRecorded(UnknownCause):    # written before this library recorded causes
    pass


@dataclass(frozen=True)
class SourceDropped(UnknownCause):  # rewrite_store(drop=...) deleted the source
    via: str                        # the handler: still known
    source_type: str                # the qualname of the deleted event


@dataclass(frozen=True)
class FrameworkEvent:               # the framework wrote it: RunPaused,
    pass                            # MaxRoundsExceeded, Cancelled, Abandoned


log.cause(event)     # -> Cause | UnknownCause | FrameworkEvent | None. None: a seed.
log.effects(event)   # -> tuple[Event, ...]: the events that this event caused, in log order
log.flow(event)      # -> tuple[Event, ...]: the chain of Causes that ends at the event
log.causes           # -> tuple of the same values, aligned with events.
                     #    None when the log records no causes.

EventLog(events, causes=None)   # causes: a sequence aligned with events, of the same values
```

| `cause()` returns | Meaning |
|---|---|
| `Cause(source, via)` | A handler produced the event from `source`. |
| `NotRecorded()` | The event came from history written before causes existed. |
| `SourceDropped(via, source_type)` | The handler is known. A store rewrite deleted its source. |
| `FrameworkEvent()` | The framework wrote the event. The event type says which mechanism. |
| `None` | A seed: the event came from outside, through `invoke()` input or `pre_seed()`. |

- Each type holds only the facts that are true. `NotRecorded` has no handler, because
  none was recorded. `SourceDropped` keeps the handler and the type of the deleted event.
- "Unknown" happens only for history that this release did not write: a checkpoint saved
  before this feature, or a source that `rewrite_store(drop=...)` deleted. A thread that
  starts with this release and is never rewritten with `drop` has no unknown cause.
- `source` and `via` mirror `NamespaceModel.Edge(source, via, target)`. A `Cause` is the
  instance-level form of an `Edge`: `Cause.source` is an event, `Edge.source` is a type.
  The name `source` also matches `HandlerRaised.source_event`. The docs map `source` to
  the event-store term "causation ID". The name `causation` is not used, because
  `Edge.causation` already means the causal *role*: intent, react, orchestrate or chain.
- `via` is the stable graph node name (`HandlerMeta.node_name`). For an inline command
  handler, that is the command qualname, not a positional name such as `handle_2`.
  `via` equals `Edge.via` unless the handler has a stable identity: an inline command
  handler, or an `@on(node_name=...)` pin.
- `cause(event)` finds the event by identity first, then as its one equal event. A copy
  that matches several equal events raises `ValueError`, because picking one would be a guess. An event that is not in the root log raises `ValueError`.
- A log that derives from a log shares the root's cause table. `log.after(X).cause(e)`
  gives the same answer as `log.cause(e)`.
- A log without causes (`EventLog(events)` from a plain list) has `causes is None`.
  On such a log, `cause()`, `effects()` and `flow()` raise `ValueError` ("this log records
  no causes").
- The constructor checks each entry. A `Cause` source must be the same object as an
  earlier event in the same log: otherwise `ValueError`, which says to pass `events[j]`,
  not a copy. `via` must be a `str`. Any other entry type raises `TypeError`.
- `log.causes` is the inverse of the constructor: `EventLog(log.events, causes=log.causes)`
  rebuilds a root log, unknown and framework cases included. In a log from `after`,
  `before` or `select`, a source can be an event outside that log. Building a log from
  its `events` and `causes` then raises `ValueError`.

The count of firings is then a plain expression, and it includes dropped sources:

```python
fired = sum(
    1 for e in log
    if isinstance(c := log.cause(e), (Cause, SourceDropped)) and c.via == "notify_customer"
)
```

## Reflection

`Reflection` gets the same facts by root index, through the public `EventLog` API only:

- `get(index)` shows a `cause:` line for every event that is not a seed: `#N via <handler>`,
  `unknown, not recorded`, `unknown, source <Type> dropped by rewrite_store, via
  <handler>`, or `framework`.
- `evidence(index)` lists the recorded cause first, as a fact.
- The `query_log` tool gets one op, `cause`, with an `index` argument. It answers with the
  same text, or `seed`.

## Storage

The graph state gets one channel next to `events`:

```python
"events": Annotated[list[Event], operator.add],
"causes": Annotated[list[tuple[int, str] | None], operator.add],   # (source, via)
# source >= 0: the log index of the source. source < 0: the source is -source positions back.
```

- `causes[i]` describes `events[i]`. Every writer to `events` writes the same number of
  entries to `causes`, in the same order. LangGraph applies the writes of each task to
  every channel in task order, so parallel handlers stay aligned. The spike confirmed
  this for parallel nodes, `Scatter`, a `None` return, one node with several triggers,
  the invariant rollback, retries, interrupt and resume, and `ainvoke`.
- The channel stores plain tuples, not `Cause` objects. A `Cause` class in the state
  becomes a serde identity and triggers the "unregistered type" warning of the
  serializer.
- A handler cannot find the index of its trigger by identity or equality after a
  checkpoint reload. It derives the index instead: the pending events sit at indices
  `[_cursor - len(_pending), _cursor)`. This is exact on every path where a handler runs:
  the seed, the router and `RunPaused`. After `MaxRoundsExceeded`, no handler runs. The
  derivation also holds for a checkpoint saved before this feature, so no extra channel
  is needed.
- Besides `(source, via)` and `None`, the channel stores two tagged tuples:
  `("framework",)` for an event the framework wrote, and
  `("dropped", via, source_type)` for a source that a store rewrite deleted.
- One module, `_causes.py`, owns the storage format. It holds the entry alias
  `CauseEntry`, the constructors of the tagged entries, and one reader,
  `resolve(events, causes) -> (absolute entries, known_from)`. `resolve` applies the relative-source rule and the align-from-the-end
  rule. `EventLog` (built from state) and `rewrite_store(drop=...)` both call it, so
  neither sees a negative source.
- One helper in `_internal.py`, `pad_causes(update, entry)`, adds one entry for each
  event of a state update that writes `events` without causes: `None` for a seed, or the
  framework entry. Every writer outside a handler call uses it.

### Writers

| Writer | Writes to `causes` |
|---|---|
| Handler (`_finalize`) | One `(trigger index, node name)` per event, recorded after each handler call, so the invariant rollback stays aligned. |
| Graph input (`_prepare_input`) | `[None]` per seed. The input writes its own causes, so the channels stay aligned with no guess, also on a checkpoint saved before this feature. The seed node writes no causes. |
| Router: `MaxRoundsExceeded`, `RunPaused` | `[("framework",)]`, through `pad_causes` |
| Async `Cancelled` path | `[("framework",)]`, through `pad_causes` |
| `_settle_supersteps` (abandon) | `[("framework",)]`, through `pad_causes` |
| `pre_seed()` / `apre_seed()` | `[None]` per event in `values["events"]`, through `pad_causes`. These events come from outside, so they are seeds. The AG-UI resume path writes events this way. |
| `rewrite_store(drop=...)` (`_drop_from_log`) | Filters `causes` at the same positions and remaps each source index, after `resolve`. Writes absolute entries back. A cause whose source was dropped becomes `("dropped", via, source_type)`. |

A writer that drifts fails fast. `_finalize` raises `RuntimeError` when a handler call
records a different number of causes than events. `resolve` raises `RuntimeError` when
`len(causes) > len(events)`, with both lengths in the message, and when a stored source
is not an earlier event.

### Interrupt and resume

The value that a human supplies on resume gets the `Interrupted` event as its source,
with the `via` of the handler that interrupted. The human answered that `Interrupted`
event. `Resumed` also gets the `Interrupted` event as its source.

The handler writes `[Interrupted, value, Resumed]` as one contiguous block, but it cannot
know the absolute index of `Interrupted` when it writes, because parallel tasks decide the
final order. The channel therefore allows a **relative source**: a negative number `-d`
means "the event `d` positions before this one". The value stores `-1` and `Resumed`
stores `-2`. A task's block is contiguous in the channel, so the distance stays true in
every later checkpoint. `_causes.resolve` turns a relative source into an absolute one,
for the `EventLog` built from state and for `rewrite_store(drop=...)` before it remaps.

### Checkpoints saved before this feature

LangGraph starts a channel that an old checkpoint lacks at its default, `[]`. The reader
aligns the channels from the end: `offset = len(events) - len(causes)`, and each event
below `offset` is `NotRecorded()`. `Reflection.get` shows `cause: unknown, not
recorded`. If `len(causes) > len(events)`, a writer drifted: `resolve` raises
`RuntimeError` with both lengths.

A thread that paused before this feature resumes normally. Its handler derives the
trigger index from `_cursor` and `_pending`, so the events it returns get a real `Cause`.

## Clients that persist their own log

A client that saves events in its own format saves each cause with them, through its own
event references. For example, a client that stores a reference as `{"$ref": N}` saves
`"cause": {"source": {"$ref": N}, "via": "..."}` on each event that has a cause. It
rebuilds the log with `EventLog(events, causes=...)`. Because a `Cause` holds an event,
the client does not shift any index when it joins the logs of several runs.

## Out of scope

- Streaming and AG-UI do not expose causes yet.
- `NamespaceModel.mermaid()` does not draw counts on edges yet. A client can pass a
  count through `notes`.
- `HandlerRaised.source_event` stays as it is. `evidence` can show when it agrees with
  the recorded cause.
- Stored correlation IDs. `flow()` derives the root on demand.
- A handler that runs and returns nothing leaves no cause. Causes record emissions only.

## TDD order

1. `EventLog(events, causes=...)` construction: a valid cause, a `source` that is not an
   earlier event (raises), the `causes` property, the round trip
   `EventLog(log.events, causes=log.causes)`, and the error for a derived log.
2. `cause`, `effects` and `flow` on a constructed log, including a derived log
   (`after`, `select`) and a log without causes (raises).
3. A graph run: a policy's output has `Cause(source=<trigger>, via="<policy>")`. A seed
   has `None`. `via` equals `Edge.via` unless the handler has a stable identity: an
   inline command handler (`via` is the command qualname), or an `@on(node_name=...)`
   pin (`via` is the pinned name).
4. Alignment cases, one test each: parallel handlers and `Scatter`, one node with several
   triggers, the invariant rollback, `HandlerRaised` and `HandlerRetried`, interrupt and
   resume, `ainvoke`, `MaxRoundsExceeded`, `RunPaused`.
5. Checkpoint round-trip with `NamespaceAwareSerde`, and a thread saved before this
   feature (align from the end).
6. `rewrite_store(drop=...)` remaps the causes.
7. `Reflection.get`, `evidence` and the `cause` op.
