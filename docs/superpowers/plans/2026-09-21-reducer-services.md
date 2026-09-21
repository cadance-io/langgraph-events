# Reducer Services Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A `Reducer` or `ScalarReducer` `fn` can declare name-keyed services (issue #193).

**Architecture:** Bind, do not thread. A private `_bind` makes a copy of a reducer whose `fn`
already holds its service values. Each call site swaps its reducer dict for a bound dict.
`collect(events)` and `seed(events)` keep their signature. For the reducer seam, a `RunScoped`
factory receives a normalised config, so the node path and the stream path cannot diverge.

**Tech Stack:** Python 3.11+, LangGraph, pytest with pytest-describe, beartype, ruff, mypy strict.

**Spec:** `docs/superpowers/specs/2026-09-21-reducer-services-design.md`

## Global Constraints

- Run all tooling through `uv`: `uv run pytest tests/`, `uv run ruff check src/ tests/`,
  `uv run ruff format src/ tests/`, `uv run mypy src/`. Do not call bare `python` or `pytest`.
- TDD iron law: no production code without a failing test first. Verify red for the expected
  reason, then green, then the full suite stays green.
- Tests are BDD-style with pytest-describe: `describe_` groups by API surface, `when_` mirrors a
  code branch, `it_` names the assertion. Test each behaviour once, at the API boundary.
- An event class used as a handler type annotation must be defined at module level. A shared
  event class lives in `tests/conftest.py`.
- Event names use a past participle, for example `NoticeRaised`. Never `RaiseNotice`.
- Every docstring, comment, error message and commit message uses ASD-STE100 Simplified
  Technical English: one idea per sentence, active voice for an instruction, no contraction,
  no slang, no metaphor, no semicolon that joins clauses.
- Line length 88. mypy strict must pass.
- Never commit with `--no-verify`. If a hook fails, fix the cause.
- Do not hand-edit a version string.
- `_bind`, `_BoundFn`, `_service_params`, `bind_reducers`, `_caller_config` and `_reducers_for`
  are private. Do not export them from `langgraph_events/__init__.py`.
- `collect(events)` and `seed(events)` keep their signature on every reducer kind.
- The handler service seam in `_build_inject` must keep its behaviour and its error messages.
- Commit message format: `feat: <summary> (#193)` or `test:` / `docs:` / `refactor:`. End each
  commit message with the line `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`.

---

### Task 1: Service parameters, `_BoundFn`, `_bind` and the unbound guard

**Files:**
- Modify: `src/langgraph_events/_reducer.py`
- Test: `tests/test_reducer_services.py` (create)

**Interfaces:**
- Produces:
  - `_reducer._service_params(fn: Callable[..., Any]) -> tuple[str, ...]`
  - `_reducer._BoundFn(fn, values: Mapping[str, Any])`, a callable `(event) -> Any`
  - `BaseReducer._service_params` property, returns `tuple[str, ...]`, default `()`
  - `BaseReducer._bind(self, values: Mapping[str, Any]) -> BaseReducer`, default returns `self`
  - `Reducer` and `ScalarReducer` override both.

**Rules for `_service_params(fn)`:**
- A `_BoundFn` gives `()`.
- `inspect.signature(fn)` that raises `ValueError` or `TypeError` gives `()`. Example: `str`.
- Skip the first parameter. For each other parameter: skip it when it has a default or its kind
  is `VAR_POSITIONAL` or `VAR_KEYWORD`. A `POSITIONAL_ONLY` parameter without a default raises
  `TypeError`. The others are service parameters, in declaration order.
- Example: `operator.attrgetter("x")` reports `(*args, **kwargs)`, so it gives `()`.

**`_BoundFn`:**

```python
class _BoundFn:
    """A reducer ``fn`` that already holds its service values.

    The ``repr`` shows the service names only. A service value can be a secret.
    """

    __slots__ = ("_fn", "_values")

    def __init__(self, fn: Callable[..., Any], values: Mapping[str, Any]) -> None:
        self._fn = fn
        self._values = dict(values)

    def __call__(self, event: Any) -> Any:
        return self._fn(event, **self._values)

    def __repr__(self) -> str:
        name = getattr(self._fn, "__qualname__", repr(self._fn))
        return f"<bound {name} services={sorted(self._values)}>"
```

**`_bind` override (same body in `Reducer` and `ScalarReducer`):**

```python
def _bind(self, values: Mapping[str, Any]) -> BaseReducer:
    names = self._service_params
    if not names:
        return self
    return dataclasses.replace(
        self, fn=_BoundFn(self.fn, {n: values[n] for n in names})
    )
```

**Unbound guard:** at the start of `Reducer.collect` and `ScalarReducer.collect`, when
`self._service_params` is not empty, raise:

```python
TypeError(
    f"Reducer {self.name!r} fn declares the service parameter(s) "
    f"{list(names)}, but the reducer is not bound to a run. Register it on an "
    f"EventGraph with a name-keyed services= mapping. For replay_reducer, pass "
    f"services=..."
)
```

Compute `_service_params` from `self.fn` on each access. Do not cache it in a dataclass field:
`dataclasses.replace` resets an `init=False` field.

- [ ] **Step 1: Write the failing tests** in `tests/test_reducer_services.py`. Put shared
  module-level fixtures at the top: a `NoticeRaised(Event)` event with a `text: str` field
  (check `tests/conftest.py` first and reuse an existing event if one fits), and
  `def _project(event, language): return [f"{language}:{event.text}"]`. Cover:
  - `describe_service_params`: one-parameter `fn` gives `()`. A two-parameter `fn` gives
    `("language",)`. A parameter with a default is skipped. `str` gives `()`.
    `operator.attrgetter("text")` gives `()`. A required positional-only parameter raises
    `TypeError`. A keyword-only parameter is a service parameter.
  - `describe_bind`, parametrised over `Reducer` and `ScalarReducer`: `_bind` of a reducer
    without a service parameter returns the same object. `_bind({"language": "fr"})` returns a
    different object. `collect` on the bound copy gives the `fr:` value. `name` and
    `namespace` of the copy equal the original. `repr(bound)` does not contain the service
    value (use a value such as `"sk-secret"`). `_bind` ignores an extra key in `values`.
  - `describe_unbound_guard`: `collect` and `seed` on an unbound reducer with a service
    parameter raise `TypeError` that names the reducer and `language`.
  - A `BaseReducer` subclass that is not a dataclass: `_bind` returns `self`.
- [ ] **Step 2: Verify red.** Run `uv run pytest tests/test_reducer_services.py -q`.
  Expected: fail with `AttributeError` or `ImportError` on the missing names.
- [ ] **Step 3: Implement** in `_reducer.py` as specified above.
- [ ] **Step 4: Verify green.** Run the same file, then `uv run pytest tests/ -q`,
  `uv run ruff check src/ tests/`, `uv run ruff format src/ tests/`, `uv run mypy src/`.
- [ ] **Step 5: Commit.** `feat: a reducer fn can hold bound service values (#193)`

---

### Task 2: Normalised config and `bind_reducers`

**Files:**
- Modify: `src/langgraph_events/_internal.py`
- Test: `tests/test_reducer_services.py`

**Interfaces:**
- Consumes: `BaseReducer._service_params`, `BaseReducer._bind` (Task 1).
- Produces:
  - `_internal._caller_config(config: RunnableConfig | None) -> RunnableConfig`
  - `_internal.bind_reducers(reducers: dict[str, BaseReducer], services_by_name: dict[str, Any] | None, config: RunnableConfig | None) -> dict[str, BaseReducer]`

```python
def _caller_config(config: RunnableConfig | None) -> RunnableConfig:
    """Return the part of *config* that is equal on every reducer path.

    A node receives a config that LangGraph enriched. The stream path holds
    the raw caller config. A reducer service factory receives this view, so
    both paths give it equal input.
    """
    configurable = (config or {}).get("configurable") or {}
    return {
        "configurable": {
            k: v
            for k, v in configurable.items()
            if not k.startswith(("__", "checkpoint_"))
        }
    }
```

**`bind_reducers` rules:**
- When no reducer has a service parameter, return the *same* dict object. This is the fast
  path.
- Otherwise build the normalised config one time. Resolve each needed service name one time
  per call, also when two reducers share it. A `RunScoped` value: call
  `svc.factory(_caller_config(config))`. A plain value: use it. When the factory raises, add
  a note with `exc.add_note(...)` and re-raise. The note text:
  `f"Reducer {name!r} requested the run-scoped service {param!r}, but its factory raised."`
- Return a new dict `{name: r._bind(values)}` in the same key order.
- A service name that is missing from `services_by_name` cannot occur after the build check of
  Task 3. Let the `KeyError` propagate.

- [ ] **Step 1: Write the failing tests.** `describe_caller_config`: `None` gives
  `{"configurable": {}}`. Keys `__pregel_runtime`, `__lge_deadline`, `checkpoint_ns`,
  `checkpoint_id` are removed. `thread_id` and a caller key stay. `metadata`, `callbacks` and
  `tags` are absent from the result. `describe_bind_reducers`: the fast path returns the same
  dict object (`is`). A `RunScoped` service resolves from `configurable`. A plain service
  binds. Two reducers that share one `RunScoped` service call the factory one time (count the
  calls). The factory receives exactly the normalised config (record the argument). A factory
  that raises keeps its exception type and carries the note (`exc.__notes__`).
- [ ] **Step 2: Verify red.** `uv run pytest tests/test_reducer_services.py -q`.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Verify green.** Same file, then the full suite, ruff, format, mypy.
- [ ] **Step 5: Commit.** `feat: bind reducers to a normalised run config (#193)`

---

### Task 3: Build check and `EventGraph._reducers_for`

**Files:**
- Modify: `src/langgraph_events/_graph.py` (`_verify_service_name_types` near line 930,
  `EventGraph.__init__` near line 1131)
- Modify: `src/langgraph_events/_reducer.py` only if `FoldReducer` needs a helper
- Test: `tests/test_reducer_services.py`

**Interfaces:**
- Consumes: `_service_params` (Task 1), `bind_reducers` (Task 2).
- Produces: `EventGraph._reducers_for(self, config: RunnableConfig | None) -> dict[str, BaseReducer]`.

**Requirements:**
1. `_reducers_for(config)` returns
   `bind_reducers(self._reducers, self._services_by_name, config)`.
2. In `EventGraph.__init__`, after `_resolve_run_scoped_types(...)`, check each reducer:
   - A `FoldReducer` whose `fold` callable has a third required parameter raises `TypeError`:
     `f"FoldReducer {name!r} fold declares the parameter {param!r}. A FoldReducer cannot receive a service. Use a Reducer or a ScalarReducer."`
     Apply the same signature tolerance as `_service_params`. Read `FoldReducer` first: its
     default fold calls `event.fold(state)`.
   - Each service parameter of a `Reducer` or `ScalarReducer` must be a key of
     `self._services_by_name`. If not, raise `TypeError`:
     `f"Reducer {name!r} fn parameter {param!r} is not a key of services. A reducer fn receives the event and name-keyed services only. It receives no config, no store and no reducer value. Known services: {sorted(known)}."`
     When the type-keyed sequence form is in use (`self._services_by_type` is not empty), end
     the message with: `" A reducer service needs the name-keyed mapping form: services={'name': value}."`
3. Apply the 0.33.0 annotation check to each reducer service parameter. Refactor
   `_verify_service_name_types` so that it takes an owner label and the hints, not a
   `HandlerMeta`: `_verify_service_name_types(owner: str, hints: Sequence[tuple[str, Any]], services_by_name, run_scoped_types)`.
   The handler call passes `owner=f"Handler {meta.name!r}"`. Every existing handler message
   must stay byte-identical, so the existing tests in `tests/test_run_scoped.py` and the
   service annotation tests stay green with no edit. The reducer call passes
   `owner=f"Reducer {name!r} fn"`. Resolve the reducer hints the same way
   `extract_handler_meta` resolves `service_name_hints` (read `_handler.py`). An annotation
   that does not resolve is skipped for a reducer.

- [ ] **Step 1: Write the failing tests.** `describe_build_check`: an unknown service
  parameter raises with the reducer name, the parameter and the known services. A `config`
  parameter raises the same error (it is not a service key). The sequence form adds the
  mapping-form sentence. A `FoldReducer` with a third required `fold` parameter raises. A plain
  value with a wrong type raises `TypeError` that starts with `Reducer 'messages' fn`. A
  `RunScoped` factory with a wrong return type raises. `fn=str`-style builtins build without an
  error. A class-attribute reducer on a `Namespace` with a service parameter builds (read
  `tests/test_reducer_namespace.py` for the pattern). `describe_reducers_for`: without a
  service parameter it returns `graph._reducers` itself. With one it returns bound copies.
- [ ] **Step 2: Verify red.**
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Verify green.** Same file, `tests/test_run_scoped.py`, the full suite, ruff,
  format, mypy.
- [ ] **Step 5: Commit.** `feat: check reducer service parameters at graph build (#193)`

---

### Task 4: Node paths use bound reducers

**Files:**
- Modify: `src/langgraph_events/_internal.py` (`make_seed_node` near line 134,
  `make_handler_node` `_prepare` and `_finalize` near lines 746 to 790)
- Modify: `src/langgraph_events/_graph.py` (`aseed` and the `make_seed_node(...)` call near
  line 1262)
- Test: `tests/test_reducer_services.py`

**Interfaces:**
- Consumes: `bind_reducers` (Task 2).
- Produces: `make_seed_node(reducers=None, services_by_name=None)` whose node function is
  `seed(state: StateDict, config: RunnableConfig) -> StateDict`.

**Requirements:**
1. The seed node takes `config` and binds one time per call, only when it will call a reducer
   (`reds` is not empty and there is work: first run, or `new_events` is not empty). Update
   `aseed` in `_graph.py` to `async def aseed(state, config)` and pass `config` through.
2. In the handler node, `_finalize` binds only when `new_events` is not empty. Pass `config`
   to `_finalize`. Update both the sync and the async runner.
3. `_build_inject` gives `Reflection` the bound dict. Bind only when `meta.reflection_param` is
   set. `reducer_params` injection reads `r.empty` only, so it can keep the unbound dict.
4. Do not change how a handler service resolves.

- [ ] **Step 1: Write the failing tests** at the `EventGraph` API. Use a module-level seed
  event and handler. Register `services={"language": RunScoped(lambda c: c["configurable"]["language"])}`
  (use a named function, a lambda has no return annotation and that is legal). Cover:
  `graph.invoke(seed, config={"configurable": {"language": "fr"}})` gives an `fr:` value in the
  reducer channel for a seed event (seed node) and for a handler-produced event (handler
  node). The same through `await graph.ainvoke(...)` (async seed wrapper). A second run on one
  thread with a checkpointer (`MemorySaver`) adds a bound contribution. A handler that declares
  a `Reflection` parameter can call `.state()` without an error and sees the `fr:` value. A
  plain (not `RunScoped`) service works. Parametrise the reducer kind over `Reducer` and
  `ScalarReducer` where the assertion allows it.
- [ ] **Step 2: Verify red.** Expected: `TypeError` from the unbound guard.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Verify green.** Same file, the full suite, ruff, format, mypy.
- [ ] **Step 5: Commit.** `feat: node paths project events with bound reducers (#193)`

---

### Task 5: Stream shadow and AG-UI adapter use bound reducers

**Files:**
- Modify: `src/langgraph_events/_graph.py` (`_update_reducer_state` near line 2711,
  `_astream_v2` near lines 2785 to 2850, and every other use of `self._reducers[...]` that
  calls `collect`, `seed`, `has_contributions` or `reducer` in a stream method)
- Modify: `src/langgraph_events/agui/_adapter.py` (`_reducer_updates_for` near line 441 and
  its callers)
- Test: `tests/test_reducer_services.py`, and the adapter test file that covers resume (find
  it with `grep -rn "_reducer_updates_for\|apre_seed" tests/`)

**Interfaces:**
- Consumes: `EventGraph._reducers_for(config)` (Task 3).
- Produces: `_update_reducer_state(self, state, event, reducer_names, reducers)` with the
  reducer dict as an explicit parameter. `_reducer_updates_for(self, events, config)`.

**Requirements:**
1. In each stream method, call `reducers = self._reducers_for(kwargs.get("config"))` one time
   at the start, then use `reducers` for every `collect`, `seed`, `has_contributions` and
   `reducer` access in that method. Do not assign to `self._reducers`. Concurrent streams share
   the attribute.
2. `_update_reducer_state` takes the dict as a parameter.
3. In the adapter, `_reducer_updates_for` takes the run `config` and uses
   `self._graph._reducers_for(config)`.
4. First run `grep -n "_reducers\[" src/langgraph_events/_graph.py src/langgraph_events/agui/_adapter.py`
   and list each hit in your report with its decision (bound, or unbound with the reason).

- [ ] **Step 1: Write the failing tests.**
  - Stream equality. A `RunScoped` factory records each config it receives in a module-level
    list. Build a graph with a `MemorySaver`. Run
    `async for frame in graph.astream_events(seed, include_reducers=True, config=cfg)` (read
    `docs/streaming.md` and an existing stream test for the exact call). Assert: the reducer
    value in the last `StreamFrame` equals the checkpoint value from
    `graph.compiled.get_state(cfg).values`. Assert: every recorded factory input is equal, and
    equals `{"configurable": {"thread_id": ..., "language": "fr"}}`. The projected event is a
    domain event, not `RunPaused`.
  - A factory that reads `config.get("metadata")` receives `None` on both paths.
  - Streaming without `config=` and a factory that reads a missing key raises the `KeyError`
    of the factory. It does not raise a "config is missing" error.
  - Adapter resume: `_reducer_updates_for` returns a bound contribution.
- [ ] **Step 2: Verify red.**
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Verify green.** Same file, the adapter tests, the full suite, ruff, format,
  mypy.
- [ ] **Step 5: Commit.** `feat: the stream path projects events with bound reducers (#193)`

---

### Task 6: `reflect(log, config=None)` and `replay_reducer(..., services=None)`

**Files:**
- Modify: `src/langgraph_events/_graph.py` (`reflect` near line 1198)
- Modify: `src/langgraph_events/serde/migrations/_core.py` (`replay_reducer` near line 1586)
- Test: `tests/test_reducer_services.py`

**Interfaces:**
- Consumes: `_reducers_for` (Task 3), `_bind` (Task 1).
- Produces:
  - `EventGraph.reflect(self, log: EventLog, config: RunnableConfig | None = None) -> Reflection`
  - `replay_reducer(reducer: BaseReducer, events: Iterable[Event], *, services: Mapping[str, Any] | None = None) -> Any`

**Requirements:**
1. `reflect` passes `reducers=self._reducers_for(config)`.
2. `replay_reducer` calls `reducer._bind(services).seed(list(events))` when `services` is not
   `None`. `services` holds plain values. A `RunScoped` value in `services` raises `TypeError`:
   `"replay_reducer has no run config. Pass the resolved service value, not a RunScoped."`
3. Extend both docstrings. State that the value comes from the current service, not from the
   run that produced the events.

- [ ] **Step 1: Write the failing tests.** `reflect(log, config=cfg).state()` gives the `fr:`
  value. `reflect(log).state()` with a `RunScoped` factory that reads a missing key raises the
  factory `KeyError`. `replay_reducer(r, events, services={"language": "fr"})` gives the `fr:`
  value. `replay_reducer(r, events)` on a reducer with a service parameter raises the unbound
  `TypeError`. A `RunScoped` in `services` raises `TypeError`. A reducer without a service
  parameter keeps its current behaviour with and without `services`.
- [ ] **Step 2: Verify red.**
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Verify green.** Same file, the full suite, ruff, format, mypy.
- [ ] **Step 5: Commit.** `feat: reflect and replay_reducer accept reducer services (#193)`

---

### Task 7: Documentation and changelog

**Files:**
- Modify: `docs/reducers.md`, `docs/api.md`, `docs/reflection.md` (the `reflect` signature),
  `docs/event-migrations.md` (the `replay_reducer` signature, near the existing recipe),
  `CHANGELOG.md` (`[Unreleased]`, under `### Added`)
- Test: the docs snippet test, if one exists (find it with `grep -rln "docs/" tests/`)

**Requirements:**
1. `docs/reducers.md` gets a section "Services in a reducer fn". It must contain:
   - A runnable example with a `RunScoped` language service. Event names use a past participle.
   - The rule: the first parameter is the event, each other required parameter resolves by name
     from the name-keyed `services` mapping.
   - The limits: no `config`, no `store`, no reducer value. `FoldReducer` cannot receive a
     service.
   - A warning block, consequence first: a factory used by a reducer receives only the
     caller-supplied `configurable` keys. It receives no `metadata`, no runtime context and no
     store. Reason: the checkpoint value and the streamed value must be equal.
   - A warning block: a service used by a reducer must be stable for the life of the thread.
     If it changes, a from-scratch projection (`reflect`, `replay_reducer`) gives a value that
     differs from the checkpoint.
   - The cost: a handler and a reducer that share a `RunScoped` service call the factory two
     times in one node call.
2. `docs/api.md`: update the `reflect` and `replay_reducer` signatures and the `RunScoped`
   entry.
3. `CHANGELOG.md`: one `### Added` entry under `[Unreleased]` that names #193. Do not edit a
   version string.
4. Run each new docs snippet verbatim with `uv run python` from a scratch file outside the repo
   and paste the output in your report.

- [ ] **Step 1: Write the docs.**
- [ ] **Step 2: Run the snippets verbatim.** Fix the docs, not the library, when a snippet
  fails because of the snippet. Report a library defect, do not fix it.
- [ ] **Step 3: Verify.** `uv run pytest tests/ -q`, `uv run mkdocs build --strict` if
  `mkdocs.yml` exists.
- [ ] **Step 4: Commit.** `docs: services in a reducer fn (#193)`
