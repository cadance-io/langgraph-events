# Migration guide

## v0.4.0 → v0.5.0

0.5.0 is a large release — event taxonomy (`Namespace`, `Command`, `DomainEvent`), declarative namespace-scoped reducers, `invariants=`, strict return-type enforcement, introspection APIs. Most of it is additive vocabulary you can adopt at your own pace. **Eight existing behaviours become stricter** and need attention.

New to the taxonomy? See [Getting Started](getting-started.md) and [`examples/order.py`](https://github.com/cadance-io/langgraph-events/blob/main/examples/order.py). Feature list lives in the [CHANGELOG](https://github.com/cadance-io/langgraph-events/blob/main/CHANGELOG.md).

## Breaking changes

### 1. Reducer fields beyond `name` are keyword-only

Positional `Reducer("history", Event, project_fn)` no longer works. Pass the rest as kwargs:

```python
Reducer(name="history", event_type=Event, fn=project_fn)
# or positional name, kwargs rest:
Reducer("history", event_type=Event, fn=project_fn)
```

Enables declaring reducers as class attributes on a `Namespace` with `name` auto-filled.

### 2. Handler return types are enforced at dispatch

- Handlers with a return annotation must return an instance of the declared union (or `None`).
- Handlers subscribed to a `Command` must return one of that command's `Outcomes` (or `None`) when no annotation is present.
- Unannotated non-`Command` handlers keep the legacy shape-only check.

Violations raise `TypeError` at the handler's dispatch, not in the next consumer.

```python
# Was silent in 0.4.0; raises TypeError in 0.5.0
@on(UserInput)
def handle(event: UserInput) -> Greeting:
    return Complaint(...)   # not in the declared union
```

Fix: match the declared return type, widen the annotation, or drop it (for non-`Command` handlers).

### 3. Bare `Event` subclassing raises `TypeError`

`class Foo(Event)` is no longer allowed. Use `DomainEvent` (inside a `Namespace`), `IntegrationEvent` (cross-boundary facts), `Command` (inside a `Namespace`), or compose with `Auditable` / `MessageEvent`.

```python
# Before (0.4.0)
class CustomerBanned(Event):
    customer_id: str

# After (0.5.0)
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

### 5. Introspection API consolidated

`EventGraph.catalog()` / `.describe()` / `.mermaid()` replaced by a single `EventGraph.namespaces()` returning a `NamespaceModel`:

```python
# Before
cat = graph.catalog()
print(graph.describe())
print(graph.mermaid())

# After
d = graph.namespaces()
print(d.text())                      # default view = choreography
print(d.text(view="structure"))      # taxonomy only — no handlers
print(d.mermaid())                   # graph LR choreography
d.json()                             # JSON snapshot
d.namespaces, d.command_handlers, d.policies, d.edges, d.seeds
```

The `Catalog` / `AggregateEntry` / `CommandEntry` TypedDicts are replaced by nested frozen dataclasses `NamespaceModel.Namespace`, `NamespaceModel.Command`, `NamespaceModel.CommandHandler`, `NamespaceModel.Policy`, `NamespaceModel.Edge`.

### 6. `SystemPromptSet` is now an `IntegrationEvent`

Before: `class SystemPromptSet(SystemEvent, MessageEvent)`.
After: `class SystemPromptSet(IntegrationEvent, MessageEvent)`.

Rationale: user code constructs it and seeds it as graph input. `SystemEvent` is reserved for framework-emitted facts (`Halted`, `Interrupted`, `HandlerRaised`, ...). "System" in the class name refers to the LLM's system-role message, not to framework origin.

Fix required only if you did one of:

- `@on(SystemEvent)` catch-all that used to pick up system-prompt seeds → now matches only framework events.
- `isinstance(evt, SystemEvent)` branches that treated system prompts as framework signals.

`graph.namespaces().integration_events` now lists `SystemPromptSet`; `.system_events` does not.

## Example portfolio reshaped

All examples now use the `Namespace` taxonomy.

- **`examples/react_agent.py` removed.** Redundant with `examples/conversation.py` (same `Conversation.Send` → `Sent`/`Blocked` shape, same downstream LLM+tool loop). Direct imports fail — point references at `conversation.py`.
- **`examples/reflection_loop.py` removed.** Multi-subscription loop pattern covered by `examples/conversation.py`.
- **`examples/human_in_the_loop.py` removed.** HITL pattern covered by `examples/expense_approval.py`.
- **`examples/supervisor.py`** reshaped into a `Task` domain. Event moves:

  | Before | After |
  |---|---|
  | `TaskReceived` | `Task.Run` (command) |
  | `ResearchDispatched` | `Task.Research` (sub-command) |
  | `CodeDispatched` | `Task.Code` (sub-command) |
  | `ResearchCompleted` | `Task.Research.Completed` |
  | `CodeProduced` | `Task.Code.Produced` |
  | `ResultFinalized` | `Task.Finalized` |

- **`examples/content_pipeline.py`** reshaped into a `Content` domain. Event moves:

  | Before | After |
  |---|---|
  | `ContentReceived` | `Content.Process` (command with inline `handle`) |
  | `ContentClassified` | `Content.Process.Classified` |
  | `ContentApproved` | `Content.Approved` |
  | `AnalysisProduced` | `Content.Analyzed` |
  | `ContentBlocked` | `Content.Blocked` (still a `Halted` subtype, nested for locality) |
  | `PipelineStage` (base) | `StageLabelled` Protocol (reducer uses structural typing) |

- **`examples/map_reduce.py`** reshaped into a `Batch` domain. Event moves:

  | Before | After |
  |---|---|
  | `BatchReceived` | `Batch.Summarize` (command) |
  | `DocDispatched` | `Batch.DocDispatched` |
  | `DocSummarized` | `Batch.DocSummarized` |
  | `BatchSummarized` | `Batch.Summarize.Summarized` |

- **`examples/error_recovery.py`** reshaped into a `Question` domain. Event moves:

  | Before | After |
  |---|---|
  | `QuestionAsked` | `Question.Ask` (command) |
  | `AnswerReceived` | `Question.Ask.Answered` |
  | `RetryScheduled` | `Question.RetryScheduled` |
  | `GaveUp` | `Question.GaveUp` (still a `Halted` subtype) |

Direct imports of the old class names (`from examples.map_reduce import BatchReceived`, etc.) fail. Expected — examples are reference material; copy shapes into your code, don't import from them.

The AG-UI frontend-tool examples `agui_frontend_tools.py` and `agui_confirm_dialog.py` are folded into `conversation.py` (LLM-initiated streaming path inside a `Conversation` domain) and `docs/agui.md` (handler-initiated snippet).

## Non-breaking changes

- `Halted`, `Interrupted`, `Resumed`, `HandlerRaised`, `Cancelled`, `MaxRoundsExceeded` now inherit `SystemEvent`. Existing `@on(Halted)` / `isinstance` checks still match.
- `Namespace` class names must be unique within a process.
- `invariants=` predicates are now evaluated twice per matching event: **pre-check** (before the handler runs, against the current log) and **post-check** (after the handler returns, against `log + emitted events`). Predicates must be pure functions of `log` — idempotent and deterministic. `InvariantViolated` gains a `would_emit: tuple[Event, ...]` field, defaulted to `()`; pre-check failures leave it empty, post-check failures populate it with the rolled-back events. Pinned reactors (`@on(InvariantViolated, invariant=Foo)`) fire for both phases without distinguishing.

## FAQ

**Do I have to use the taxonomy?**
Yes. Every event must use one of the four base classes: `DomainEvent`, `IntegrationEvent`, `Command`, or `SystemEvent`. Cross-cutting facts that don't belong to a domain use `IntegrationEvent`.

**Will my existing `@on(Halted)` / `@on(Interrupted)` subscriptions still fire?**
Yes. Those classes now inherit `SystemEvent` (which inherits `Event`). `isinstance` traverses the MRO.

**Do I need to update my existing tests?**
Only if you hit one of the six breaking changes above.

**Can I mix external `@on(...)` handlers and inline `handle` methods in the same graph?**
Yes — pass both to `EventGraph([...])`, or call `EventGraph.from_namespaces(Order, handlers=[external_fn])`.
