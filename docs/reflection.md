# Reflection

`Reflection` is a **deterministic, agent-harnessable query surface** over an
event log. It packages a run's `EventLog` together with the graph's static
`NamespaceModel` and reducers, and exposes them as facts an LLM agent can
query lazily — instead of holding the whole log in its context window.

The governing principle: **the API states facts; the agent reasons.** Every
op returns something deterministic — listings, field dumps, static topology,
reducer projections, verdict-free evidence joins. Nothing is inferred,
scored, or guessed, so the API can never be confidently wrong. Correlating
facts into a causal story ("*why* did this happen?") is the querying agent's
job.

## Getting a Reflection

Two ways, one construction path — the graph supplies the model and reducers:

```python
# 1. Post-run
log = graph.invoke(Sales.Place(order_id="A-1", amount=250))
run = graph.reflect(log)

# 2. Mid-run, injected into a handler by annotation (like EventLog)
@on(Sales.Cancel.Cancelled)
def brief_agent(event: Sales.Cancel.Cancelled, run: Reflection) -> None:
    response = llm.invoke(
        system=f"You are a support agent.\n{run.context()}",
        tools=[run.tool()],
    )
```

An injected `Reflection` is a mid-dispatch snapshot — like an injected
`EventLog`, it reflects the events appended so far and should not be stored
beyond the handler call. The same goes for `run.tool()`: the tool closes
over that snapshot, so an agent framework that retains tool objects across
turns silently pins stale data.

`run.log` exposes the raw `EventLog` with its full Python query surface
(`filter`/`select`/`latest`/`after`/…). `Reflection` deliberately does not
re-wrap those methods: every `#index` in every rendering refers to a
position in the root log, with no exceptions.

## The context card

`run.context(tail=3)` returns a small, bounded text block for a prompt: the
run's size and status, any anomalies, and the last few events as
`#index EventName(...)` lines — an invitation to drill in with the tool:

```text
[run context] 6 events, status: completed — use the query_log tool to inspect the event log
anomalies:
  #3 HandlerRaised(handler='charge', source_event=Payments.Charge(...), ...)
recent events (last 3 of 6):
  #3 HandlerRaised(...)
  #4 Cancel(order_id='A-1', reason='payment declined')
  #5 Cancelled(order_id='A-1', reason='payment declined')
```

## The query_log tool

`run.tool()` returns a `QueryTool` — one framework-agnostic dispatch tool.
Its `description` teaches the ops, the `#<index>` convention, and the graph's
full event-type vocabulary grouped by namespace, so a ReAct agent knows the
queryable universe up front. The dataclass shape
(`name` / `description` / `parameters` JSON Schema / `run` callable
returning `str`) maps 1:1 onto an Anthropic tool dict or LangChain's
`StructuredTool.from_function`:

```python
# Anthropic
tool = run.tool()
anthropic_tool = {
    "name": tool.name,
    "description": tool.description,
    "input_schema": tool.parameters,
}
# then the standard tool-use loop; your half is one line per tool_use block:
content = tool.run(**block.input)  # goes back as the tool_result content
```

```python
# LangChain — the tool's own JSON schema drives the StructuredTool
from langchain_core.tools import StructuredTool

lc_tool = StructuredTool.from_function(
    func=tool.run,
    name=tool.name,
    description=tool.description,
    args_schema=tool.parameters,  # skip signature inference; use the real schema
)
```

### Ops

The base vocabulary mirrors `EventLog`'s own query functions; `type` is an
event-type name (subclass-aware, like the Python API):

| op | arguments | returns |
|---|---|---|
| `overview` | — | totals, counts by kind/namespace, seeds, anomalies, status |
| `list` | `index` (offset), `limit` | all events in order, paged |
| `get` | `index` | full field dump of one event + kind/namespace/command |
| `filter` / `select` | `type`, `limit` | matching events as `#index` lines |
| `latest` / `first` | `type` | newest / oldest match |
| `has` / `count` | `type` | `true`/`false` / a number |
| `after` / `before` | `type`, `limit` | events after / before the first match |
| `evidence` | `index` | every fact on how that event came to be |
| `state` | — | reducer projections over the log |
| `schema` | — | the static topology: what *can* cause what |

Base kinds (`Event`, `Command`, `DomainEvent`, `IntegrationEvent`,
`SystemEvent`) match whole categories: `count(type="DomainEvent")`.

Errors come back as guidance strings the agent can self-correct from —
`error: unknown type 'Placd' — did you mean: Placed? …`,
`error: index 42 out of range (log has 6 events, valid: 0..5, negatives count
from the end)`. Only
input errors are caught; genuine bugs propagate.

### evidence — the join that replaces "why"

`evidence(index)` lists, with **no verdicts and no selection**:

1. **Explicit links** — event-valued fields resolved to log positions
   (`HandlerRaised.source_event`, `Resumed.interrupted`), by identity with a
   labeled equality fallback.
2. **Owning command** — the outcome's command class and every preceding
   instance of it.
3. **Static edge candidates** — every model edge targeting this event's
   type, with its causation kind (`intent`/`react`/`orchestrate`/`chain`),
   handler, and preceding source instances.
4. **Forward face** — edges sourced at this type, with subsequent target
   instances.

The agent correlates; the API only joins. A typical ReAct trace:

```text
>>> query_log(op='evidence', index=4)
evidence for #4 Cancel(order_id='A-1', reason='payment declined')
possible causes (static edges, candidate instances):
  HandlerRaised --cancel_on_decline [orchestrate]--> Cancel: #3
possible effects (static edges, subsequent instances):
  Cancel --cancel [intent]--> Cancelled: #5
```

See `examples/reflection_agent.py` for a complete offline diagnosis loop —
overview → locate → evidence-walk → conclusion.

## state

`run.state()` (and the `state` op) projects each registered reducer over the
full log from scratch: `{name: reducer.seed(list(log))}` — the runtime
answer to "what's the derived state now".

!!! note
    Projections use `BaseReducer.seed()`, which does not run a `Reducer`'s
    custom merge fn. For `message_reducer` (add_messages dedup semantics)
    the projection can differ from the live channel value.

## Trust and exposure

Tool outputs are built from **event field values — untrusted runtime data**.
If your events carry end-user content, that content reaches the querying
agent's context verbatim (bounded per value: 40 chars in listings, 2000 in
`get`/`state`). Treat `query_log` results as data, never as instructions,
in your agent's system prompt. Be aware of the exposure scope: `get` and
`state` return full field values (including anything sensitive an event or
exception carries — e.g. `HandlerRaised.exception` often reprs connection
strings), and `schema` reveals handler names and topology. There is no
redaction hook yet; if your events hold secrets or PII, don't hand this
tool to an agent whose transcript you wouldn't show those values to.

## Design notes

- **Deterministic only.** An earlier design inferred causation with
  nearest-preceding heuristics and certainty scores; it was cut. Facts are
  cheap to verify and impossible to get wrong; interpretation belongs to the
  agent driving the tool.
- **Root indices only.** `select`-style narrowing lives on `run.log` (plain
  `EventLog` results); the reflective surface never re-indexes a slice.
- **Richer events, richer facts.** If events later carry actor or
  provenance fields, `get` and `evidence` surface them automatically — no
  API change needed.
