# Event causes: record which handler produced each event, and from which event

> Design spec, 2026-10-02. Brainstormed with the user, then checked by parallel
> subagents: a throwaway spike of the state channel, a review of logs that derive from a
> log, a review of checkpoint evolution and naming, a client study (reflection-lab), and
> a survey of event-sourcing practice. Implementation follows TDD per project conventions.

## Context

When a handler returns an event, the framework knows two facts: the handler that ran,
and the event that the handler was called with. Today the framework drops both facts
when it appends the result to the log. After the run, no query can say which handler
produced an event. The log and the static graph cannot recover the fact either. Two
policies that react to one event type and return one command type give the same log.

A client needs the fact. Example: reflection-lab draws a diagram of the rules a persona
defined, and it must show how often each rule fired.

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

```python
@dataclass(frozen=True)
class Cause:
    source: Event   # the event that the handler was called with ("causation ID")
    via: str        # the handler node name (HandlerMeta.node_name)


log.cause(event)     # -> Cause | None. None for a seed, or for an unknown cause.
log.effects(event)   # -> tuple[Event, ...]: the events that this event caused, in log order
log.flow(event)      # -> tuple[Event, ...]: the cause chain, from the root seed to the event
log.causes           # -> tuple[Cause | None, ...] | None: aligned with events.
                     #    None when the log records no causes.

EventLog(events, causes=None)   # causes: a sequence aligned with events, of Cause | None
```

- `source` and `via` mirror `NamespaceModel.Edge(source, via, target)`. A `Cause` is the
  instance-level form of an `Edge`, and the target is the event itself. The name `source`
  also matches `HandlerRaised.source_event`. The docs map `source` to the event-store
  term "causation ID". The name `causation` is not used, because `Edge.causation`
  already means the causal *role*: intent, react, orchestrate or chain.
- `via` is the stable graph node name (`HandlerMeta.node_name`). For an inline command
  handler, that is the command qualname, not a positional name such as `handle_2`.
  `via` equals `Edge.via` unless the handler has a stable identity: an inline command
  handler, or an `@on(node_name=...)` pin.
- `cause(event)` finds the event by identity first, then by equality, the same way as
  `Reflection._resolve_index`. An event that is not in the root log raises `ValueError`.
- A log that derives from a log shares the root's cause table. `log.after(X).cause(e)`
  gives the same answer as `log.cause(e)`.
- A log without causes (`EventLog(events)` from a plain list) has `causes is None`.
  On such a log, `cause()`, `effects()` and `flow()` raise `ValueError` ("this log records
  no causes"). `cause()` does not return `None`, because `None` means "a seed".
- The constructor checks each `Cause`: its `source` must be an event that appears earlier
  in the same log, by identity. Otherwise the constructor raises `ValueError`. An entry
  that is not a `Cause` or `None` raises `TypeError`.
- `log.causes` is the inverse of the constructor: `EventLog(log.events, causes=log.causes)`
  rebuilds a root log. In a log from `after`, `before` or `select`, a source can be an
  event outside that log. Building a log from its `events` and `causes` then raises
  `ValueError`.

The count of firings is then a plain expression:

```python
fired = sum(1 for e in log if (c := log.cause(e)) and c.via == "hourly_wake_brief")
```

## Reflection

`Reflection` gets the same facts by root index:

- `get(index)` shows `cause: #N via <handler>` when the event has a cause.
- `evidence(index)` lists the recorded cause first, as a fact.
- The `query_log` tool gets one op, `cause`, with an `index` argument. It answers
  `#N via <handler>`, or `seed`.

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
  checkpoint reload. The seed node and the router therefore write `_pending_base`, the
  log index of the first pending event. The handler uses `_pending_base + k` for the
  k-th pending event.
- One module, `_causes.py`, owns the storage format. It holds the entry alias
  `CauseEntry` and one function, `resolve(events, causes) -> (absolute entries,
  known_from)`. `resolve` applies the relative-source rule and the align-from-the-end
  rule. `EventLog` (built from state) and `rewrite_store(drop=...)` both call it, so
  neither sees a negative source.
- One helper in `_internal.py`, `pad_causes(update)`, adds a `None` cause for each event
  of a state update that writes `events` without causes. Every writer outside a handler
  call uses it.

### Writers

| Writer | Writes to `causes` |
|---|---|
| Handler (`_finalize`) | One `(trigger index, node name)` per event, recorded after each handler call, so the invariant rollback stays aligned. On a thread that paused before this feature (no `_pending_base`), `None` per event |
| Seed node | `[None] * min(len(events) - len(causes), len(events) - cursor)`. The graph input writes only `events`, so the seed node pads for it. The second term keeps the older events of a checkpoint saved before this feature unknown. |
| Router: `MaxRoundsExceeded`, `RunPaused` | `[None]`, through `pad_causes` |
| Async `Cancelled` path | `[None]`, through `pad_causes` |
| `_settle_supersteps` (abandon) | `[None]`, through `pad_causes` |
| `pre_seed()` / `apre_seed()` | `[None]` per event in `values["events"]`, through `pad_causes`. The AG-UI resume path writes events this way. |
| `rewrite_store(drop=...)` (`_drop_from_log`) | Filters `causes` at the same positions and remaps each source index, after `resolve`. Writes absolute entries back. A cause whose source was dropped becomes `None`. It also lowers `_pending_base` by the dropped events below it, the same as `_cursor`. |

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
below `offset` has an unknown cause. `cause()` returns `None` for such an event, and
`Reflection.get` shows `cause: unknown`. If `len(causes) > len(events)`, a writer drifted:
`resolve` raises `RuntimeError` with both lengths.

A thread that paused before this feature has no `_pending_base`. When it resumes, the
handler records `None` for each event it returns, so the channels stay aligned.

`Reflection` must tell `unknown` from `seed`, because `cause()` returns `None` for both.
It reads that one fact through the private `EventLog._cause_is_known(event)`, its only
private dependency on `EventLog`.

## Clients that persist their own log

A client that saves events in its own format saves each cause with them, through its own
event references. reflection-lab, for example, saves
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
