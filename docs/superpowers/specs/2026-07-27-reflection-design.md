# Reflection: a deterministic, agent-harnessable query surface over the event log

> Design spec, 2026-07-27. Brainstormed interactively, hardened through DHH-lens and
> GoF-lens subagent reviews, then re-centered on a determinism principle. Implementation
> follows TDD per project conventions.

## Context

An LLM agent running inside (or on top of) an `EventGraph` needs to reason about *how we
got here — what happened* without holding the whole event log in context. Today
`EventLog` offers low-level Python queries, `NamespaceModel` reflects static code
topology, and nothing packages them for an agent. The phantom `log.reduced_state`
referenced in `examples/order.py` docstrings never landed.

**Governing principle:** the API is a *harness* — it offers only **deterministic facts**
and leaves reasoning, correlation, and causal interpretation to the ReAct agent driving
it. It stays close to the existing `EventLog` API. The API never guesses, so it can
never be wrong.

An earlier draft included a heuristic causation engine (`why`/`effects` with
nearest-preceding attribution and a certainty taxonomy). It was cut on review:
interpretation belongs to the agent. Its deterministic residue is the `evidence` op —
a verdict-free join.

## The product

`graph.reflect(log)` returns a **`Reflection`** — a thin read-model bundling the log
with the graph's static model and reducers — whose centerpiece is one ReAct-loop tool,
**`query_log`**, mirroring `EventLog`'s own query functions plus equally deterministic
ops. Injectable into handlers by annotation, like `EventLog` today.

```python
@on(NeedsDecision)
def decide(event: NeedsDecision, run: Reflection) -> Decision:   # injected, enriched
    response = llm.invoke(
        system=f"...\n{run.context()}",    # bounded overview card
        tools=[run.tool()],                # query_log, ReAct-driven
    )
```

Sample ReAct trace:
`count(type="HandlerRaised")` → `1` → `filter(type="HandlerRaised")` →
`#12 HandlerRaised(...)` → `evidence(index=12)` → explicit source link + candidate
causes → the agent concludes the chain itself.

## The op set — deterministic only

**Native ops** (mirror `EventLog` exactly, subclass-aware `isinstance`; `type` is an
event-type name): `filter | select | latest | first | has | count | after | before`.
Listings render as compact `#index EventName(...)` lines, capped by `limit`
(default 20) with an "…and N more — refine your query" footer. `after`/`before` anchor
on the **first** matching instance, exactly like `EventLog.after/before`.

**Additional deterministic ops:**

- `get(index)` — full-field dump of one event, plus kind, namespace, owning command.
- `list` — plain ordered listing of the whole log (paged via `limit` + `index` offset).
  Order is a fact; narrative is the agent's job.
- `overview` — arithmetic aggregation: totals, counts by kind/namespace, seeds,
  anomaly listing (`HandlerRaised`/`InvariantViolated`/`Halted`/`Interrupted`/
  `RunPaused`), run status (`completed | halted | interrupted | paused`).
- `state` — reducer projections `{name: reducer.seed(list(log))}` (lands the phantom
  `reduced_state`). Documented caveat: `seed()` skips merge fns, so `message_reducer`
  projections may differ from live channel values.
- `schema` — the static topology, verbatim from the cached `NamespaceModel.text()`:
  what *can* cause what. A fact about the code.
- `evidence(index)` — the deterministic join replacing `why`. For `log[i]`, list with
  **no verdicts, no selection**:
  1. *Explicit links*: `source_event` (`HandlerRaised`/`InvariantViolated`) and
     `interrupted` (`Resumed`) resolved to `#index` by identity (equality fallback,
     latest match, labeled as such).
  2. *Owning command*: `type(e).__command__` + indices of all preceding instances.
  3. *Static edge candidates*: every model edge with `isinstance(e, edge.target)`,
     each with causation kind, handler (`via`), and indices of preceding instances of
     `edge.source`.
  4. *Forward face*: edges where `isinstance(e, edge.source)`, with indices of
     subsequent instances of their targets.
  The agent correlates; the API only joins.

Errors: unknown op/type/index return guidance **strings** ("error: unknown type
'OrderPlacd' — did you mean OrderPlaced? valid: …"); only
`(ValueError, IndexError, KeyError)` are caught — genuine bugs propagate. All `#index`
values are root-log indices everywhere, no exceptions.

## Public API

```python
class Reflection:
    def __init__(self, log: EventLog, *, model: NamespaceModel,
                 reducers: Mapping[str, BaseReducer]) -> None: ...
    @property
    def log(self) -> EventLog: ...
    def context(self, *, tail: int = 5) -> str: ...
    def tool(self) -> QueryTool: ...
    def overview(self) -> str: ...
    def event(self, index: int) -> str: ...          # tool op: get
    def evidence(self, event: Event | int) -> str: ...
    def schema(self) -> str: ...
    def state(self) -> dict[str, Any]: ...

@dataclass(frozen=True)
class QueryTool:
    name: str                       # "query_log"
    description: str                # ops + #<index> convention + event-type vocabulary
    parameters: dict[str, Any]      # JSON Schema: op (enum), type?, index?, limit?
    run: Callable[..., str]
```

Written invariant: **every tool op is a one-line delegation to a public `Reflection`
method or an `EventLog` method** — the tool never grows logic of its own.

Module layout (house `_namespace/` convention — model + sibling renderers):

```
src/langgraph_events/_reflection/
    __init__.py     # exports Reflection, QueryTool
    _core.py        # Reflection facade
    _evidence.py    # evidence join (pure function over log + model)
    _text.py        # all formatting
    _tool.py        # QueryTool + build_tool(): dispatch, type-name resolution, vocabulary
```

Wiring: `EventGraph.reflect(log)`; `NamespaceModel` cached at the compile boundary
(injection/schema must not re-trigger pattern warnings); handler injection mirrors the
existing `hint is EventLog` seam; `Reflection` + `QueryTool` exported in `__all__`.
No changes to `_event_log.py`.

## Review-driven decisions (recorded)

- **No `EventLog` impersonation** (unanimous DHH/GoF): `Reflection.log` exposes the raw
  log; nothing re-wrapped; root indices only. (Causation-style scans over sliced tuples
  would have been silently wrong.)
- **Two public names**: `Reflection`, `QueryTool`. Python query methods return `str`
  (`state()` returns `dict`); no public result-object zoo.
- **No log-only degraded mode**: one construction path (`graph.reflect` + injection).
- **Issue #57 (`ReducerRegistry`) unbundled**: `state()` uses a private three-line
  projection; the public registry + agui migration ships separately.
- **Determinism over inference**: the API states facts; the agent reasons. A future
  actor/provenance dimension stamped on events (a downstream product's roadmap
  requirement) needs no API change — richer events are just richer facts for
  `get`/`evidence` to surface.

## Deferred

- Heuristic causation (`why`/`effects`) — revisit only if real agent traces show
  `evidence` costs too many reasoning tokens.
- `EventLog.reflect()` log-only mode.
- Field-predicate filtering (`filter(type=..., where={...})`).
- Mermaid run-trace rendering over the same facts.
- Public `ReducerRegistry` (#57).

## Verification

Full suite + mypy strict + ruff; docs fences executed by `test_docs_code_fences.py`;
a POC example (`examples/reflection_agent.py`) drives `query_log` in a scripted
ReAct loop against a real failure scenario and reaches a correct conclusion from
facts alone.
