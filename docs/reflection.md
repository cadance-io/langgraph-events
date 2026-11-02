# Reflection

`Reflection` packages a run's `EventLog` with the graph's static `NamespaceModel`
and reducers as facts that an LLM agent queries lazily, instead of holding the
whole log in its context window. Every op returns a deterministic listing, field
dump, topology, projection or evidence join, and the agent reasons over them.

## Getting a Reflection

Two ways, one construction path. The graph supplies the model and reducers:

```python
...  # graph and Sales as above
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

An injected `Reflection` is a mid-dispatch snapshot. Like an injected
`EventLog`, it reflects the events appended so far. Do not store it beyond the
handler call. The same applies to `run.tool()`: the tool closes over that
snapshot, so an agent framework that retains tool objects across turns keeps
stale data.

`run.log` exposes the raw `EventLog` with its full Python query surface
(`filter`/`select`/`latest`/`after`/…). `Reflection` deliberately does not
re-wrap those methods: every `#index` in every rendering refers to a
position in the root log, with no exceptions.

## The context card

`run.context(tail=3)` returns a small, bounded text block for a prompt: the
run's size and status, any anomalies, and the last few events as
`#index EventName(...)` lines. The agent passes an index to the tool to
inspect an event:

```text
[run context] 6 events, status: completed — use the query_log tool to inspect the event log
anomalies:
  #3 HandlerRaised(handler='charge', source_event=Payments.Charge(order_id='A-1', amoun..., exception=CardDeclinedError('card declined for ...)
recent events (last 3 of 6):
  #3 HandlerRaised(handler='charge', source_event=Payments.Charge(order_id='A-1', amoun..., exception=CardDeclinedError('card declined for ...)
  #4 Cancel(order_id='A-1', reason='payment declined')
  #5 Cancelled(order_id='A-1', reason='payment declined')
```

## The query_log tool

`run.tool()` returns a `QueryTool`, one framework-agnostic dispatch tool.
Its `description` teaches the ops, the `#<index>` convention, and the graph's
full event-type vocabulary grouped by namespace, so a ReAct agent knows the
queryable types before the first call. The dataclass shape
(`name` / `description` / `parameters` JSON Schema / `run` callable
returning `str`) maps 1:1 onto an Anthropic tool dict or LangChain's
`StructuredTool.from_function`:

```python
...  # run as above
# Anthropic
tool = run.tool()
anthropic_tool = {
    "name": tool.name,
    "description": tool.description,
    "input_schema": tool.parameters,
}
# Run the standard tool-use loop. For each tool_use block, call:
content = tool.run(**block.input)  # goes back as the tool_result content
```

```python
...  # run as above
# LangChain: the JSON schema of the tool drives the StructuredTool
from langchain_core.tools import StructuredTool

lc_tool = StructuredTool.from_function(
    func=tool.run,
    name=tool.name,
    description=tool.description,
    args_schema=tool.parameters,  # use the real schema, not signature inference
)
```

### Ops

The base vocabulary mirrors `EventLog`'s own query functions. `type` is an
event-type name (subclass-aware, like the Python API):

| op | arguments | returns |
|---|---|---|
| `overview` | — | totals, counts by kind/namespace, seeds, anomalies, status |
| `list` | `index` (offset), `limit` | all events in order, paged |
| `get` | `index` | full field dump of one event + kind/namespace/command + recorded cause |
| `filter` / `select` | `type`, `limit` | matching events as `#index` lines |
| `latest` / `first` | `type` | newest / oldest match. The op takes no `not_followed_by`: only the Python API has it |
| `has` / `count` | `type` | `true`/`false` / a number |
| `after` / `before` | `type`, `limit` | events after / before the first match |
| `evidence` | `index` | every fact on how that event came to be |
| `cause` | `index` | the recorded cause: `#N via <handler>`, `seed`, `framework`, or `unknown, <reason>`. The handler name equals `Edge.via`, except for an inline command handler or an `@on(node_name=...)` handler, which show their stable node name |
| `state` | — | reducer projections over the log |
| `schema` | — | the static topology: what *can* cause what |

Base kinds (`Event`, `Command`, `DomainEvent`, `IntegrationEvent`,
`SystemEvent`) match whole categories: `count(type="DomainEvent")`.

Errors come back as guidance strings the agent can self-correct from:
`error: unknown type 'Placd' — did you mean: Placed? …`,
`error: index 42 out of range (log has 6 events, valid: 0..5, negatives count
from the end)`. Only input errors are caught. Genuine bugs propagate.

### evidence

`evidence(index)` lists, with **no verdicts and no selection**:

1. **Recorded cause** — the handler that produced the event, and the event it received, as
   `#N via <handler>`. The library records it at dispatch, so it is a fact, not a candidate.
   A seed shows `seed`, and an event that the library wrote shows `framework`. An unknown
   cause shows `unknown` and its reason: `not recorded` for history written before causes
   existed, or `source <Type> dropped by rewrite_store, via <handler>`.
2. **Explicit links** — event-valued fields resolved to log positions
   (`HandlerRaised.source_event`, `Resumed.interrupted`), by identity with a
   labeled equality fallback.
3. **Owning command** — the outcome's command class and every preceding
   instance of it.
4. **Static edge candidates** — every model edge targeting this event's
   type, with its causation kind (`intent`/`react`/`orchestrate`/`chain`),
   handler, and preceding source instances.
5. **Forward face** — edges sourced at this type, with subsequent target
   instances.

The API joins. The agent correlates. A typical ReAct trace:

```text
>>> query_log(op='evidence', index=4)
evidence for #4 Cancel(order_id='A-1', reason='payment declined')
recorded cause: #3 via cancel_on_decline
possible causes (static edges, candidate instances):
  HandlerRaised --cancel_on_decline [orchestrate]--> Cancel: #3
possible effects (static edges, subsequent instances):
  Cancel --cancel [intent]--> Cancelled: #5
```

See `examples/reflection_agent.py` for a complete offline diagnosis loop:
overview → locate → evidence-walk → conclusion.

## state

`run.state()` (and the `state` op) projects each registered reducer over the
full log from scratch: `{name: reducer.seed(list(log))}`.

!!! note
    Projections use `BaseReducer.seed()`, which does not run a `Reducer`'s
    custom merge fn. For `message_reducer` (add_messages dedup semantics)
    the projection can differ from the live channel value.

## Trust and exposure

Tool outputs are built from event field values, which are untrusted runtime
data. End-user content in an event reaches the querying agent's context
verbatim (40 characters per value in listings, 2000 in `get` and `state`), so
instruct the agent to treat `query_log` results as data, never as
instructions. `get` and `state` return full field values, including anything
sensitive that an event or an exception carries, and `schema` reveals handler
names and topology. There is no redaction hook, so do not hand this tool to an
agent whose transcript must not show those values.
