# Migrating from 0.3.0

0.4.0 is a large release — DDD-aligned taxonomy, declarative aggregate-scoped reducers, `invariants=`, `raises=`, introspection APIs. **Only four existing behaviours become stricter.** Fix those and the rest is new vocabulary you can adopt at your own pace.

New to the taxonomy? See [Getting Started](getting-started.md) and [`examples/ddd_order.py`](https://github.com/cadance-io/langgraph-events/blob/main/examples/ddd_order.py). Feature list lives in the [CHANGELOG](https://github.com/cadance-io/langgraph-events/blob/main/CHANGELOG.md).

## Breaking changes

### 1. Reducer fields beyond `name` are keyword-only

Positional `Reducer("history", Event, project_fn)` no longer works. Pass the rest as kwargs:

```python
Reducer(name="history", event_type=Event, fn=project_fn)
# or positional name, kwargs rest:
Reducer("history", event_type=Event, fn=project_fn)
```

Enables declaring reducers as class attributes on an `Aggregate` with `name` auto-filled.

### 2. Handler return types are enforced at dispatch

- Handlers with a return annotation must return an instance of the declared union (or `None`).
- Handlers subscribed to a `Command` must return one of that command's `Outcomes` (or `None`) when no annotation is present.
- Unannotated non-`Command` handlers keep the legacy shape-only check.

Violations raise `TypeError` at the handler's dispatch, not in the next consumer.

```python
# Was silent in 0.3.0; raises TypeError in 0.4.0
@on(UserInput)
def handle(event: UserInput) -> Greeting:
    return Complaint(...)   # not in the declared union
```

Fix: match the declared return type, widen the annotation, or drop it (for non-`Command` handlers).

### 3. Bare `Event` subclassing raises `TypeError`

`class Foo(Event)` is no longer allowed. Use `DomainEvent` (inside an Aggregate), `IntegrationEvent` (cross-boundary facts), `Command` (inside an Aggregate), or compose with `Auditable` / `MessageEvent`.

```python
# Before (0.3.0)
class CustomerBanned(Event):
    customer_id: str

# After (0.4.0)
class CustomerBanned(IntegrationEvent):
    customer_id: str
```

### 4. `Auditable` and `MessageEvent` are plain mixins

They no longer inherit `Event`. Compose them with an event branch:

```python
# Before
class TaskStarted(Auditable): ...

# After
class TaskStarted(IntegrationEvent, Auditable): ...
```

## DDD consistency pass (v0.4.x)

Still on 0.4 — a follow-up pass aligning the library's own surface with the DDD taxonomy it evangelizes. All breakage intentional.

### 1. Introspection API consolidated

`EventGraph.catalog()` / `.describe()` / `.mermaid()` replaced by a single `EventGraph.domain()` returning a `DomainModel`:

```python
# Before
cat = graph.catalog()
print(graph.describe())
print(graph.mermaid())

# After
d = graph.domain()
print(d.text())                      # default view = choreography
print(d.text(view="structure"))
print(d.mermaid())                   # default view = choreography
print(d.mermaid(view="structure"))   # classDiagram
d.json()                             # JSON snapshot
d.aggregates, d.command_handlers, d.policies, d.edges, d.seeds
```

The `Catalog` / `AggregateEntry` / `CommandEntry` TypedDicts are replaced by nested frozen dataclasses `DomainModel.Aggregate`, `DomainModel.Command`, `DomainModel.CommandHandler`, `DomainModel.Policy`, `DomainModel.Edge`.

### 2. `SystemPromptSet` is now an `IntegrationEvent`

Before: `class SystemPromptSet(SystemEvent, MessageEvent)`.
After: `class SystemPromptSet(IntegrationEvent, MessageEvent)`.

Rationale: user code constructs it and seeds it as graph input. `SystemEvent` is reserved for framework-emitted facts (`Halted`, `Interrupted`, `HandlerRaised`, ...). "System" in the class name refers to the LLM's system-role message, not to framework origin.

Fix required only if you did one of:
- `@on(SystemEvent)` catch-all that used to pick up system-prompt seeds → now matches only framework events.
- `isinstance(evt, SystemEvent)` branches that treated system prompts as framework signals.

`graph.domain().integration_events` now lists `SystemPromptSet`; `.system_events` does not.

### 3. Non-DDD example portfolio consolidated

All examples now use the DDD taxonomy.

- **`examples/react_agent.py` removed.** Redundant with `examples/ddd_conversation.py` (same `Conversation.Send` → `Sent`/`Blocked` shape, same downstream LLM+tool loop). Direct imports fail — point references at `ddd_conversation.py`.
- **`examples/supervisor.py`** reshaped into a `Task` aggregate. Event moves:

  | Before | After |
  |---|---|
  | `TaskReceived` | `Task.Run` (command) |
  | `ResearchDispatched` | `Task.Research` (sub-command) |
  | `CodeDispatched` | `Task.Code` (sub-command) |
  | `ResearchCompleted` | `Task.Research.Completed` |
  | `CodeProduced` | `Task.Code.Produced` |
  | `ResultFinalized` | `Task.Finalized` |

- **`examples/content_pipeline.py`** reshaped into a `Content` aggregate. Event moves:

  | Before | After |
  |---|---|
  | `ContentReceived` | `Content.Process` (command with inline `handle`) |
  | `ContentClassified` | `Content.Process.Classified` |
  | `ContentApproved` | `Content.Approved` |
  | `AnalysisProduced` | `Content.Analyzed` |
  | `ContentBlocked` | `Content.Blocked` (still a `Halted` subtype, nested for locality) |
  | `PipelineStage` (base) | `StageLabelled` Protocol (reducer uses structural typing) |

- **`examples/map_reduce.py`** reshaped into a `Batch` aggregate. Event moves:

  | Before | After |
  |---|---|
  | `BatchReceived` | `Batch.Summarize` (command) |
  | `DocDispatched` | `Batch.DocDispatched` |
  | `DocSummarized` | `Batch.DocSummarized` |
  | `BatchSummarized` | `Batch.Summarize.Summarized` |

- **`examples/error_recovery.py`** reshaped into a `Question` aggregate. Event moves:

  | Before | After |
  |---|---|
  | `QuestionAsked` | `Question.Ask` (command) |
  | `AnswerReceived` | `Question.Ask.Answered` |
  | `RetryScheduled` | `Question.RetryScheduled` |
  | `GaveUp` | `Question.GaveUp` (still a `Halted` subtype) |

Direct imports of the old class names (`from examples.map_reduce import BatchReceived`, etc.) fail. Expected — examples are reference material; copy shapes into your code, don't import from them.

## Non-breaking changes

- `Halted`, `Interrupted`, `Resumed`, `HandlerRaised`, `Cancelled`, `MaxRoundsExceeded` now inherit `SystemEvent`. Existing `@on(Halted)` / `isinstance` checks still match.
- Aggregate class names must be unique within a process — an accepted v1 constraint.

## FAQ

**Do I have to use the DDD taxonomy?**
Yes. Every event must use one of the four base classes: `DomainEvent`, `IntegrationEvent`, `Command`, or `SystemEvent`. Cross-cutting facts that don't belong to an aggregate use `IntegrationEvent`.

**Will my existing `@on(Halted)` / `@on(Interrupted)` subscriptions still fire?**
Yes. Those classes now inherit `SystemEvent` (which inherits `Event`). `isinstance` traverses the MRO.

**Do I need to update my existing tests?**
Only if you hit one of the four breaking changes above.

**Can I mix external `@on(...)` handlers and inline `handle` methods in the same graph?**
Yes — pass both to `EventGraph([...])`, or call `EventGraph.from_aggregates(Order, handlers=[external_fn])`.
