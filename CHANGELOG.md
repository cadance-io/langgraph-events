# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **`EventGraph.unrevivable_threads()` / `.aunrevivable_threads()`.** The store-walking gate
  from [#159](https://github.com/cadance-io/langgraph-events/issues/159). Reads every thread's
  latest checkpoint through the serde's tolerant path and returns a mapping of thread id to the
  qualnames it can no longer revive, from the settled `events` history and the pending interrupt
  both. Empty when every thread revives. The baseline coverage gates compare the topology to a
  committed snapshot and never read a checkpoint, so after `write_baseline(..., allow_removed=True)`
  they stay green while a settled thread still raises `Cannot revive`. This sweep is the one that
  sees it. Replaces the hand-rolled step 4 recipe in *Retiring an Interrupted subclass*, whose own
  warning admitted a stale string literal would report "safe". Requires a checkpointer. Cost is
  O(all checkpoints), like `threads_paused_on()`.

- **`EventGraph.abandon()` / `.aabandon()`.** Settles a paused thread without answering its
  pending `Interrupted`. Closes [#162](https://github.com/cadance-io/langgraph-events/issues/162).
  Ends the thread on a terminal `Abandoned(Halted)`, via the same three-superstep settle
  primitive `on_unresumable="halt"` uses. The tool for retiring an `Interrupted` subclass:
  resuming every paused thread first would instead append the retired identity back into the log.
  Requires a checkpointer. Raises `ValueError` on a thread with no events to settle. Ignores
  `on_unresumable` (that policy governs an *accidental* no-op resume, not a deliberate
  abandonment). Returns `None`. Callers who want the log call `graph.get_state(config).events`.

- **`Abandoned`.** New `Halted` subtype recorded by `abandon()`/`aabandon()`. `.reason` is the
  caller-supplied reason. `.discarded` is the qualname(s) of the interrupt(s) thrown away.

- A `resume()` on an already-abandoned thread now names the abandonment in its
  `UnresumableError` message instead of pointing at a handler rename/removal.

- **`EventGraph.threads_paused_on()` / `.athreads_paused_on()`.** Configs for every thread
  whose latest checkpoint has a pending interrupt, optionally filtered to an `Interrupted`
  class or subclass. Closes the discovery gap in the `abandon()` retirement workflow: a client
  no longer needs `graph.compiled` or `snapshot.tasks[*].interrupts[*].value` to find the
  threads to abandon. Reads the checkpoint's raw pending writes directly, not the graph's
  compiled topology, so it still finds a thread paused on a handler already removed from the
  graph, the common retirement shape (an `Interrupted` usually retires the handler that
  produced it too). Requires a checkpointer. Reads every checkpoint the checkpointer holds:
  cost is O(all checkpoints), not O(paused threads), so a large deployment should filter
  thread ids server-side instead. Raises `ValueError` if the checkpointer's `list()`/`alist()`
  is unimplemented, naming the method (closes part of [#164]).

- **`abandon()` / `aabandon()` gained `require_interrupt: bool = True`.** By default, both now
  raise `ValueError` on a thread with no pending interrupt, naming the thread and pointing at
  `require_interrupt=False`. Previously they settled such a thread silently, recording
  `Abandoned(discarded="")`. This appended a terminal event onto settled business history with
  no warning. The pending-interrupt check reads the checkpoint directly, same as
  `threads_paused_on()`, so it does not raise on a genuinely paused thread whose handler is
  already gone from the graph. Pass `require_interrupt=False` to keep the old behaviour
  (closes part of [#164]).

- **`threads_paused_on()`/`abandon()` now survive a deleted `Interrupted` class, not only a
  removed handler.** A stored pending interrupt naming a class that no longer imports used to
  raise `Cannot revive` from both, the exact tool meant to clean up that state. Both now
  degrade: the thread stays in `threads_paused_on()`'s result, and `abandon()`/`aabandon()`
  settle it under the default `require_interrupt=True`, recording the interrupt's last-known
  qualname in `Abandoned.discarded` instead of a live instance. Scoped to these two operations
  only: every other read (`get_state()`, `resume()`, `invoke()`, …) stays strict, so a genuine
  revival bug still raises. The `Cannot revive` message now states the remedy: settle with
  `abandon()`/`aabandon()` before deleting the class, or map the identity onto a tombstone with
  `@migrate_from()` — see [Recovering a delete-first deployment](event-migrations.md#recovering-a-delete-first-deployment)
  (closes part of [#164]).

- **`serde.UnreachableMigrationWarning`.** `NamespaceAwareSerde` now warns at construction when
  a `@migrate_from`-decorated class lives in a module its `namespaces=`/`events=` already
  reaches, but was never itself passed in. Its migration silently did nothing before this. The
  warning names the class and says how to fix it (nest it in a passed `Namespace`, or add it to
  `events=`). Scoped to modules already reachable through this construction, not a process-wide
  scan, so an unrelated engine lifetime's decorated classes are never flagged.

[#164]: https://github.com/cadance-io/langgraph-events/issues/164

### Fixed

- **The retirement docs' step 4 sweep could never fail.** It compared
  `type(e).__qualname__` against `"EventClass"` (a placeholder inside a string literal, which
  never equals a real qualname). The snippet therefore always computed `unsafe == []`. It
  green-lit the delete regardless of what the store held. Now binds `RETIRING = EventClass`
  and compares `type(e).__qualname__ == RETIRING.__qualname__`. Verified against a store with
  a genuinely unsafe answered thread, which the snippet now reports.

- **`write_baseline` raised `AttributeError` on a `str` path.** Every sibling gate
  (`assert_all_baselined_cover`/`_resolve`/`_revive`/`_handlers_cover`) already accepts
  `Path | str`. `write_baseline` was the one outlier, and the retirement docs' own workflow
  printed a bare string. Now accepts `Path | str` and coerces, matching its siblings.

- **`Cannot revive`'s remedy misdirected on a field-shape `TypeError`.** When the identity
  resolved to a live class (including a tombstone already carrying `@migrate_from`) but
  construction failed on a field mismatch, the message still said to map the identity onto a
  tombstone with `@migrate_from(...)`, naming the class that already **is** the tombstone. Now
  says the target class does not declare the field the stored payload carries, naming the
  field.

- **The nested tombstone recipe silently no-oped when the checkpointer already carried a
  `NamespaceAwareSerde`** (e.g. reused from an earlier graph in the same process).
  `from_namespaces(...)`'s auto-wiring deliberately skips rebuilding one that is already there
  (see `api.md`), so the tombstone never entered scope, with no error or warning. Documented
  inline in the recipe, not only in a table three documents away.

- **`Abandoned.discarded` recorded the leaf class name, not the qualname.** `"ApprovalRequested"`
  when the class was still live, but `"Order.ApprovalRequested"` (the qualname) once it was
  deleted. The same field changed shape under the exact axis a retirement changes. A check
  written before the deletion (`match with in or .split(", ")`) could stop matching after it.
  Always the qualname now, live class or not: unambiguous under nesting and stable across the
  deletion (closes part of [#164]).

- **The published `@migrate_from` retirement recipe did not work.** It printed a module-level
  tombstone class and claimed `EventGraph.from_namespaces(...)` would collect it. That method
  has no `events=` kwarg and never reaches a module-level class, so the recipe recovered
  nothing, silently. Replaced with a nested-in-`Namespace` recipe (the one
  `from_namespaces(...)` actually wires up) as the primary path, and the module-level form as a
  separate, complete, hand-built-serde alternative. The tombstone in both was field-free,
  correct only when the retired class also had no fields, and the alternative's
  `EventGraph([...])` call dropped the inline `Order.Approve` command it needs, tripping
  `OrphanedEventWarning`. Both snippets now carry the retired class's fields (with an inline
  comment saying why) and register every handler they need. Both were verified end to end,
  warning-free, across a real process restart against persisted checkpoint bytes (closes part
  of [#164]).

- **Two coverage-gate messages had no remedy line, and two had a grammar/count mismatch.**
  `assert_all_baselined_resolve`/`assert_all_baselined_revive` named the broken identity but not
  the fix. Both now end with the same remedy line `MigrationCoverageError` already had.
  `MigrationCoverageError`/`HandlerCoverageError` said "1 identity ... are neither" and "1
  ... handler no longer resolve" regardless of count. The verb now agrees ("is"/"resolves" for
  one, "are"/"resolve" for more than one).

- **`GraphState.interrupted` was `None` on a first pause.** `get_state().interrupted` read only
  the event log, and an `Interrupted` joins the log only on resume. A thread paused for the
  first time therefore reported `is_interrupted=True` with `interrupted=None`. The published
  `docs/control-flow.md` example crashed on this (`AttributeError` reading `.order_id`). Now
  falls back to the snapshot's pending interrupt payload when the log has none yet (closes part
  of [#164]).

- **`on_unresumable="halt"` re-armed the thread it was supposed to retire.** The policy appended
  its terminal `Unresumable(Halted)` event with a single `update_state` call. That call re-ran
  routing against the checkpoint's stale `_pending` state. Routing then rescheduled the
  already-paused node. A later `resume()` call then passed the pending check. It ran for real
  and wrote the retired `Interrupted` identity back into the event log.

- **A halted thread stopped dispatching events.** The same `update_state` call left `_cursor`
  behind the appended terminal event. The event then re-entered the *next* run's pending window.
  This tripped the `Halted` dispatch gate without an error. A later `invoke()` on the same
  thread appended its event but did not dispatch it. No handler fired, and no error was raised.

Both defects are fixed by a three-superstep clear/append/clear write. The write clears pending
tasks, appends the terminal event with `_cursor` and `_pending` reset, then clears again. This
replaces the single `update_state` call. A halted thread now ends with nothing scheduled and no
stale pending state. Any completed sibling write from a fanned-out superstep survives.

## [0.28.0] - 2026-08-28

### Removed

- **Namespace subclassing** — `class Child(Base)` where `Base` is a `Namespace` now raises
  `TypeError`. Deprecated in 0.27.0 with a one-minor-version grace period, removed here.

  Only reducers ever inherited. Nested commands and events did not, which left a child namespace
  quietly incomplete: `EventGraph.from_namespaces` skipped its inherited inline handlers, and its
  inherited events fell outside a serde's scope and bled across engine lifetimes
  ([#157](https://github.com/cadance-io/langgraph-events/issues/157)).

  Declare each namespace independently. To share a reducer, declare it free-standing — with no
  owning namespace — and pass it via `reducers=[...]`. A reducer declared on the old parent stays
  bound to that parent and folds nothing for anyone else.

  Composing a non-`Namespace` mixin (`class Task(Namespace, Auditable)`) is unaffected.

## [0.27.0] - 2026-08-28

### Added

- **Warnings anchor at your code, not the library's.** Every warning the library raises now
  computes its own `stacklevel` by walking out to the first frame outside the package, instead of
  carrying a hand-counted number. Twelve call sites carried six different values, and each was
  correct only for the exact call depth it was written at — two had already gone stale and pointed
  at library internals or at the importing module's `import` line.

- **`NamespaceAwareSerde(events=[...])`** — event classes that live outside every namespace
  (module-level `IntegrationEvent`s, framework `SystemEvent`s) now reach the serde's scope. Without
  them those identities resolved by import and so were shared between two engine lifetimes of one
  module — the bleed #150 fixed for namespaced events. `@migrate_from` on such a class was also
  silently ignored; it is now collected ([#155](https://github.com/cadance-io/langgraph-events/issues/155)).

  `EventGraph.from_namespaces` fills `events=` in automatically from the graph it builds, so the
  auto-wired path needs no change. The serde is now wired after graph construction rather than
  before — the loose events are only knowable once handlers' subscriptions and return types have
  been parsed.

### Changed

- **Namespace subclassing is deprecated** — `class Child(Base)` where `Base` is a `Namespace` now
  emits a `DeprecationWarning`. Only reducers ever inherited; nested commands and events do not, so
  a child namespace is quietly incomplete: `EventGraph.from_namespaces` skips its inherited inline
  handlers, and its inherited events fall outside a serde's scope and bleed across engine lifetimes
  ([#157](https://github.com/cadance-io/langgraph-events/issues/157)). Declare each namespace
  independently and share reducers via `reducers=[...]`. Behaviour is unchanged for one minor
  version.

- **The inline-outcome-coverage error no longer contradicts itself** — coverage is decided by class
  identity but was reported by `__name__`, so two distinct same-named classes produced "declares
  return type `Placed` but does not cover outcome(s): `Placed`", advice already satisfied. Colliding
  names are now qualified, and a `<locals>` qualname is called out as the usual cause — a class
  defined inside a function whose string annotation resolved elsewhere
  ([#151](https://github.com/cadance-io/langgraph-events/issues/151)).


- **A serde handed two engine lifetimes now raises instead of binding silently** — the scope map is
  keyed by `(module, qualname)`, which two lifetimes of one module share, so
  `NamespaceAwareSerde(namespaces=[first.Trading, second.Trading])` used to bind every shared
  identity to whichever came last and make revival depend on argument order. It now raises
  `ValueError`, matching `EventGraph`, which rejects the same mistake at graph build. Give each
  lifetime its own serde.

  Migration-validation diagnostics also stopped saying "importable": with scope-first resolution a
  class reachable only through `namespaces=` is live too, so a chain terminus, an `AddField`
  target, and a shadowing rename source now all say "by this serde's namespace scope or by import".

### Fixed

- **`NamespaceAwareSerde` now resolves event identity through its own `namespaces=` scope** before
  falling back to a module import. The read path built the `(module, qualname)` map on the encode
  side already (`legacy_write`'s `oldest_historic`) but never consulted the scope on the way back
  in, so a namespace whose qualname carries `<locals>` — the normal shape for in-process
  behavioural tests, notebooks, and REPL work — could be checkpointed and never revived, and two
  engine lifetimes of one module silently shared their classes: once lifetime 2 existed, lifetime
  1's serde revived into lifetime 2's classes
  ([#150](https://github.com/cadance-io/langgraph-events/issues/150)).

  ```python
  def lifetime():
      class Trading(Namespace):
          class Place(Command):
              sym: str
              class Placed(DomainEvent):
                  sym: str
      return Trading

  T = lifetime()
  serde = NamespaceAwareSerde(namespaces=[T])
  serde.loads_typed(serde.dumps_typed(T.Place.Placed(sym="AAPL")))   # was ValueError
  ```

  Migration validation and the test gates ask the same "does this identity reach a live class?"
  question, so they now answer it the same way: `@migrate_from` / `@backfill` on a function-local
  class no longer fails serde construction as a dead-end chain or an unresolvable `AddField`
  target, `assert_all_baselined_resolve` and `assert_all_baselined_revive` resolve scope-first, and
  the shadowing guard sees a rename source that is still live *in scope* — an error that an
  import-only probe could not detect. `revivable_identities()` and
  `assert_all_baselined_cover` are unchanged: both were always keyed on the namespace walk rather
  than on imports, and widening them to count whatever happens to be importable would weaken the
  gate. `assert_all_baselined_handlers_cover` is untouched — handler names are graph topology, not
  event identity.

  Events reached by no namespace walk — module-level `IntegrationEvent`s, framework
  `SystemEvent`s — still resolve by import, unchanged.

## [0.26.0] - 2026-08-28

### Changed

- **Namespace names are scoped to the graph, not the process** — `Namespace` no longer keeps a
  process-global registry, so redefining a name a previous namespace used is valid. This lets one
  process run several independent engine lifetimes in sequence: run a scenario, end it, and start a
  fresh one that resumes the same checkpointed log, with no subprocess and no file hand-off
  ([#148](https://github.com/cadance-io/langgraph-events/issues/148)).

  ```python
  importlib.reload(app.trading)          # lifetime 2 redefines Trading
  second = app.trading.Trading
  saver.serde = NamespaceAwareSerde(namespaces=[second])
  graph = EventGraph([second.Place], checkpointer=saver)
  graph.get_state(config).events.latest(second.Place.Placed)   # lifetime 1's log, lifetime 2's class
  ```

  The uniqueness guard moves to where ambiguity actually causes harm. Two *different* namespaces of
  the same name reaching one graph is still a `TypeError`, now naming both classes with their
  modules — reducer discovery and `graph.namespaces()` group by name, so within a graph the name
  must resolve to one class. Subscribed *and* produced event types are checked, so a handler
  emitting another lifetime's class is caught too. Nested events carry a `__namespace_cls__` stamp
  alongside `__namespace__`, namespace-scoped reducers match on that class rather than the name,
  and `NamespaceModel.Namespace` gains a `cls` field (absent from `to_dict()` / `json()`, so
  `schema_version` is unchanged).

  Lifetimes are **sequential, not concurrent**: checkpointed events are keyed by
  `(__module__, __qualname__)` and resolved by import, so two lifetimes of the same module share
  that identity. Namespaces must be importable at module scope
  ([#150](https://github.com/cadance-io/langgraph-events/issues/150)).

  The only behaviour lost is the import-time `TypeError` on a duplicate namespace name. Code that
  worked before had at most one class per name, so nothing that passed starts failing — unless it
  deliberately asserted on that error.

## [0.25.1] - 2026-08-27

### Changed

- **`EventGraph.compiled` is about five times cheaper** — LangGraph read the source of every
  node function at compile time to look for nested graphs. An EventGraph node never holds one,
  so every node now declares no dependencies and the read is skipped. A consumer that builds a
  fresh `EventGraph` per test, or per schema change, sees compile drop from ~11 ms to ~2 ms for
  ten handlers (~50 ms to ~10 ms for thirty). No API change.

## [0.25.0] - 2026-08-24

### Added

- **Declarative retry with exponential backoff (`RetryPolicy`)** — declare `retry=RetryPolicy(...)`
  next to `raises=`, either as a class attribute on a `Command` or via `retry=` on `@on(...)`, and
  the framework re-invokes the handler in place with full-jitter exponential backoff. With a policy
  declared, `HandlerRaised` fires only once the retry budget is spent — or once the run's
  `deadline=` cuts the backoff short (below) — so catchers stop counting attempts and re-emitting
  commands and become pure escalation handlers. A handler with no `retry=` is unaffected —
  `HandlerRaised` still fires on its first raise.

  ```python
  class Ask(Command):
      raises = (RateLimitError,)
      retry = RetryPolicy(max_attempts=3, base_delay=0.1, max_delay=10.0)
  ```

  `max_attempts` counts the initial call. Delay before retry *n* is `base_delay * 2 ** (n - 1)`
  capped at `max_delay`; `strategy="constant"` uses `base_delay` as that ceiling every time instead
  of doubling it. `jitter` is orthogonal to `strategy` and applies to whichever ceiling the strategy
  computes — with `jitter=True` (the default) the wait is sampled uniformly from `[0, ceiling]`, so
  `strategy="constant"` still varies per retry unless you also pass `jitter=False`. `on=` narrows
  which exceptions retry and must overlap `raises=`; `respect_retry_after=True` prefers a
  server-supplied `exception.retry_after` (clamped to `[0, max_delay]`, so a skewed clock cannot
  produce a negative wait).

  Declaring `retry=` without `raises=`, or an `on=` entry disjoint from `raises=`, is a `TypeError`
  at graph construction — such a policy could never fire. Overlap in either direction is live:
  `on=(OSError,)` against `raises=(ConnectionResetError,)` retries, because scope is decided at
  runtime by `isinstance(exc, on)`.

  Retries run inside the handler node: they consume no `max_rounds` budget and write no checkpoint
  between attempts. **Retried handlers must be idempotent** — the handler re-runs from the top,
  including any `emit_custom` it fired before raising.

  The backoff is **deadline-aware**: the policy reads the run's `deadline=` before every wait and
  refuses to start a backoff that would land on or past it, giving up even with attempts left.
  Nothing is clamped — spending what is left of the budget on an attempt that probably cannot
  finish either only delays the pause — so instead of sleeping through the soft boundary and into
  the caller's hard cancellation, the run returns to the router, which emits `RunPaused` at the
  first round boundary past the deadline. A run with no `deadline=` is unaffected, and the deadline
  is read once per handler node, not per event.

  `graph.namespaces()` surfaces the policy: a `retry` field on `NamespaceModel.CommandHandler` and
  `.Policy`, a `retry` object in `.to_dict()`, and a `retry xN` annotation in `.text()`. A policy
  that actually emits (`observe="emit"`, the default) also contributes a first-class
  `NamespaceModel.Edge` of `kind="retry"` from the handler's subscribed events to `HandlerRetried`,
  so a reactor on `HandlerRetried` no longer renders as a source with nothing producing it.
  Mermaid draws that edge as a finely dotted cyan arrow, distinct from the grey dashed `raises`
  escalation, and `to_dict()`/`json()` carry it beside the `raises` one; `.text()` keeps reporting
  the policy as the `retry xN` annotation above and grows no edge of its own.
  `observe="log"`/`"silent"` never write the event to the log and so get no edge: the diagram
  tracks what the log will contain, not what was merely declared. Pure additions, so
  `SCHEMA_VERSION` is unchanged — no existing field is removed, renamed, or given a new meaning,
  and consumers that switch on `Edge.kind` should already treat an unrecognised kind as opaque.

  Not to be confused with `langgraph.types.RetryPolicy`, which re-runs an entire LangGraph node.

- **`HandlerRaised.abandoned_for_deadline`** — a new `bool` field on the existing event, `True`
  only when a `RetryPolicy` still had attempts left but the next backoff would have landed on or
  past the run's `deadline=`. It is `False` for an exhausted attempt budget, an out-of-`on=`
  exception, and a handler with no policy, so operators (and catchers) can tell "ran out of time"
  from "ran out of tries" without inferring it from timestamps. Defaults to `False`, so existing
  catchers and serialized logs are unaffected.

- **`HandlerRetried`** — a `SystemEvent` emitted before each backoff wait, carrying `handler`,
  `source_event`, `exception`, `attempt` (1-based, the call that failed) and `delay_seconds`. Part
  of the anomaly set the reflection surface reports. A deadline give-up emits none — no wait
  happened. Suppress it with `RetryPolicy(observe="log")` for a `WARNING` carrying the exception's
  type and message (not its frames) instead, or `observe="silent"` for neither.

  Its `exception` is the live instance — `@on(HandlerRetried, exception=SomeError)`
  isinstance-matches it and field injection still hands it over typed — but its **traceback is
  detached**, along with those of its `__cause__`/`__context__` chain and any `ExceptionGroup`
  members. A traceback pins the failing attempt's frame and every local on it, and the `events`
  channel is append-only, so a breadcrumb that kept one would hold whatever the handler was working
  on for the rest of the run, once per attempt: with a 5 MB response body in scope and
  `max_attempts=3` that is 15 MB stranded rather than 5 MB. Only the terminal `HandlerRaised` keeps
  its traceback — that is the one you debug from — so the framework retains at most one live
  traceback per failing invocation.

### Changed

- **BREAKING: a concrete `Command` may not be subclassed.** `class Ask(Command)` is unchanged —
  that is how a Command is declared. `class Child(Ask)`, where `Ask` is a declared Command, now
  raises `TypeError` at class creation, naming both classes. One Command is one intent, with its
  own handler, outcomes and node identity; a subclass is a second intent wearing the first one's.
  Share what two commands have in common through a helper function or a shared policy object
  (`retry: ClassVar = SHARED_POLICY`), and declare the second command independently.

  `raises`, `invariants`, `retry` and `previously` are still read off the command class — but never
  from a parent Command, so the documented asymmetry between them is gone: `previously` was read
  from the class's own `__dict__` precisely because the other three were MRO-inherited, and with no
  parent Command to inherit from all four read the same way. The `Command.handle()` privacy rule
  loses its parent-Command case for the same reason: an outcome is private to exactly the one
  Command it is nested under.

  Untouched: `Namespace` subclassing (child domains still inherit parent reducers via the MRO),
  `DomainEvent` subclassing another `DomainEvent` (refining an event keeps its namespace and
  `__command__`), and `Interrupted` / `IntegrationEvent` / `Halted` subclassing. One narrow shape
  changes with it: a module-level `class Hybrid(Command, SomeStampedDomainEvent)` used to be
  admitted by the same scaffolding — it now gets the ordinary "must be nested inside a Namespace"
  rejection.

- **BREAKING: `retry` is now a reserved name.** Two shapes that worked before now fail:

  - `@on(SomeEvent, retry=...)` no longer registers a **field matcher** — `retry=` is a named
    keyword on `@on()`, so a non-`RetryPolicy` value raises `TypeError`. A handler matching on an
    event field literally named `retry` must be rewritten; there is no positional escape hatch,
    the same way `node_name`/`previously` are reserved.
  - A `Command` declaring an annotated `retry` **field** (e.g. `retry: int = 0`) raises `TypeError`
    at class creation. `retry` joins `_RESERVED_MODIFIERS` alongside `raises`/`invariants`/
    `previously`, so a policy can never become a dataclass field serialized into every checkpoint
    — but the guard cannot tell a policy from a payload field of the same name. Rename the field.

### Fixed

- **Removing a `Command`'s `raises` or `invariants` no longer leaves the old contract enforced.**
  `@on()` stamped `_raises` and `_invariants` onto the handler function only when the value was
  non-empty. Because `EventGraph(...)` re-stamps every inline `Command` handler from its class
  attributes on each build, deleting a declaration between two builds in the same process left the
  earlier stamp in place: the second graph kept turning the removed exception into `HandlerRaised`
  instead of letting it propagate, kept evaluating the removed invariant, and kept demanding a
  catcher for a type nobody declares. Both are now stamped unconditionally, matching `previously`
  and `retry`. `node_name` stays conditional on purpose — it has no class-level declaration surface,
  so an unconditional stamp would erase an explicit `@on(node_name=...)` pin rather than refresh it.

- **Legend diagram taught the wrong colour for four arrow kinds.** The hand-written "Diagram
  vocabulary" legend — rendered into `docs/patterns.md` and the top of every
  `examples/*.graph.md`, and the key readers use to decode all the other diagrams — numbered its
  `linkStyle` directives one short of the edges they described. Mermaid counts edges from 0 in
  declaration order, including the invisible entry `==>` seed and *each hop* of a chained
  `A -.-> B -.-> C` line, so the legend declares ten edges (0–9) — but `raises` grey was pinned
  to 1, `scatter` purple to 2 and ownership grey to 3. The plain `-->` return arrow therefore
  wore `raises` grey, `(raises)` wore `scatter` purple, `scatter` wore ownership grey, and the
  ownership `-.-` arrow was left unstyled. Indices are now correct and pinned by a test that
  recovers every legend edge in mermaid's own order and asserts the style landing on it is the
  constant the renderer uses for that edge kind. Only the legend was affected — the per-example
  diagrams resolve their indices in `MermaidFlowchart.render`, which counts seeds and chained
  hops correctly by construction, and were always right.

- **A field-shape mismatch on revive now names the class and the field.** `NamespaceAwareSerde`
  revives a checkpointed event by calling the live class with the stored payload verbatim. An
  `Event` is a frozen dataclass, so a stored key the class has since dropped, or a required field
  the class has since gained with no `Migration.add_field`, raises `TypeError` out of `__init__`.
  That type was missing from the ext-hook's `except` clause, so nothing reached the error channel
  and ormsgpack's generic `ValueError: ext_hook failed` — no module, no class, no field, and a
  `__cause__` of `None` — was all the caller saw. The clause now catches `TypeError` alongside
  `ImportError` and `AttributeError`, and records `Cannot revive <module>.<qualname>: TypeError:
  <message>. The class may have been renamed or removed, or its fields may have changed, since the
  checkpoint was written.`

  This is a diagnostic change only. The revive still raises, and a payload that revives today is
  unaffected. A named added-field mismatch is fixed with `Migration.add_field`. An extra stored key
  reaches no absorber at all, because `__init__` rejects every keyword the class does not declare.

## [0.24.0] - 2026-08-23

### Added

- **AG-UI message passthrough (`AGUI_EXTRAS_KEY`)** — a mapping under the reserved
  `additional_kwargs["langgraph_events.agui"]` key on a LangChain message becomes extra top-level
  fields on the AG-UI message in `MessagesSnapshot`. AG-UI models set `extra="allow"`, so a client
  can now receive structured data the protocol does not declare — a `{"failure": {"retryable":
  true}}` hint on a failure notice, for instance. Nothing else in `additional_kwargs` crosses the
  wire, so provider metadata and internal state stay on the backend. `agui_messages_to_langchain`
  and `detect_new_tool_results` mirror it: the extra fields an inbound AG-UI message carries are
  collected back under the same key.

  The key is package-qualified on purpose. `additional_kwargs` is a shared namespace that
  LangChain provider integrations also write into, and there is no opt-in and no version gate, so
  a plain `"agui"` would have started forwarding a consumer's existing key silently. **Check
  `additional_kwargs` for prior use before adopting this.** Read the constant, not the literal.

- **`AGUI_EXTRAS_MAX_BYTES`** — a size cap on the extra fields one inbound AG-UI message may
  carry, 8 KiB of JSON. Inbound extras are unvalidated client input: they enter
  `additional_kwargs`, flow through `add_messages` into the checkpoint, and are served back out on
  the next `MessagesSnapshot` — to every client on that thread, not only the one that sent it.
  Without a cap a client could grow a thread's checkpoint without bound. Extras over the cap, or
  extras that do not encode as JSON, are dropped with a `WARNING` naming the message; the message
  itself still converts. The trust boundary is documented in `docs/agui.md` and on
  `collect_inbound_extras`.

### Changed

- **`BaseMessage.name` now reaches the AG-UI message.** `agui_messages_to_langchain` already read
  `name` inbound, but the outbound mapper dropped it, so a named message did not round-trip.
  AG-UI declares `name` on `UserMessage`, `AssistantMessage` and `SystemMessage`. Its
  `ToolMessage` declares no `name`, so tool messages are unchanged. This changes the wire payload
  for every existing consumer, which is why it is filed here rather than under `Fixed`.

### Fixed

- **`ToolMessage.status` and AG-UI `ToolMessage.error` now map in both directions.** A tool result
  the client marked as failed reached the model as a success, because every conversion path
  dropped the field. `detect_new_tool_results` and `agui_messages_to_langchain` now set
  `status="error"` when the AG-UI `error` is truthy, and `MessagesSnapshot` now sends `error` when
  the LangChain status is `"error"`. The outbound `error` is always truthy: an errored tool
  message with empty content sends the literal `"error"` instead, because an empty string is
  falsy and a client writing `if (msg.error)` would read the failure as a success. Truthiness is
  the one rule that governs the field in both directions, so a client that initialises
  `error: ""` on every tool result reports no failure. Only *presence* maps: the error text does
  not come back, because LangChain's `ToolMessage` has no field for it and `content` is where a
  tool's failure reason belongs. This is ag-ui issue #2226, whose upstream fix covered only the
  inbound half of `ag_ui_langgraph`.
- **A bad extras mapping no longer poisons a thread.** `MessagesSnapshot` is rebuilt from the
  **checkpointed** message list on every `connect()`. A collision or a non-mapping value used to
  raise `ValueError` inside that build. The raise escaped the adapter's async generator into the
  consumer's HTTP handler and never became a `RunError`, so one bad value broke every later
  connect on that thread until someone edited the checkpoint. The build now degrades instead: a
  non-mapping value is dropped whole, a non-string key or an entry naming a declared AG-UI field
  drops that entry only, and each cause warns once per message class.
- **A `ToolMessage` with block content no longer breaks the stream.** The `tool` branch of the
  snapshot mapper passed list content straight to AG-UI, which declares `content: str`, so the
  pydantic `ValidationError` surfaced as a `RunError` and killed the connect stream. It now
  degrades to `""`, matching the `system` branch, and logs a `WARNING` naming the tool call
  because the content is lost.
- **`detect_new_tool_results` carries message extras.** The two inbound paths disagreed:
  `agui_messages_to_langchain` gained both the error status and the extras, while
  `detect_new_tool_results` gained only the status, so a consumer routing frontend tool results
  through it lost the extra fields the client sent.

## [0.23.1] - 2026-08-22

### Fixed

- **`Event` and `HandlerReturn` are now re-exported, not merely importable.** Both were imported
  into the package namespace under a `# noqa: F401` but never re-exported, so a consumer running
  mypy with `no_implicit_reexport` (which `strict = true` implies) got
  `Module "langgraph_events" does not explicitly export attribute "Event"` — despite `Event`
  being annotated in four places across the docs and `HandlerReturn` being a documented handler
  return type. `Event` joins `__all__`; `HandlerReturn` re-exports through an explicit
  `as` alias, staying out of `__all__` as intended so `import *` is unchanged.

## [0.23.0] - 2026-08-22

### Added

- **Reflection** — a deterministic, agent-harnessable query surface over the
  event log. `graph.reflect(log)` returns a `Reflection` bundling the run's
  `EventLog` with the graph's `NamespaceModel` and reducers: `context()`
  (bounded prompt card), `overview()`, `event(i)`, `evidence(i)` (verdict-free
  join of explicit links, owning command, static-edge candidates, forward
  face), `schema()`, `state()` (reducer projections — the long-promised
  `reduced_state`), and `.log`. Facts only; correlation is the querying
  agent's job.
- **`query_log` tool** — `Reflection.tool()` returns a framework-agnostic
  `QueryTool` (one dispatch tool for ReAct loops) mirroring `EventLog`'s query
  functions (`filter`/`select`/`latest`/`first`/`has`/`count`/`after`/`before`)
  plus `overview`/`list`/`get`/`evidence`/`state`/`schema`. Event types are
  addressed by name; the description embeds the vocabulary grouped by
  namespace; errors return guidance strings the agent can self-correct from.
- **`Reflection` handler injection** — annotate a handler parameter with
  `Reflection` to receive an enriched mid-dispatch snapshot, mirroring
  `EventLog` injection.
- `examples/reflection_agent.py` — an offline ReAct-style diagnosis loop
  ("why was this order cancelled?") driven entirely through `query_log`.
- **Event constructors are now visible to type checkers.**
  `Event.__init_subclass__` applies `dataclasses.dataclass(frozen=True)`, but
  nothing told a type checker that — and the package ships `py.typed`, so every
  consumer had to silence `call-arg` on every event construction. `Event` now
  carries `@dataclass_transform(frozen_default=True)`, so an event's generated
  `__init__` is visible, its fields keep their declared types, an unknown or
  mistyped keyword is rejected, and assignment is refused. Pinned by
  `tests/test_event_typing.py`, which the mypy gate now checks.
- **CI runs the test suite on every supported Python** (3.11, 3.12, 3.13), and
  the release workflow gates on the same matrix. Both previously ran a single
  interpreter, so version-specific breakage was invisible.

### Changed

- `EventGraph.namespaces()` now builds the `NamespaceModel` once and caches
  it (handler metadata is immutable after construction); repeated calls no
  longer rebuild or re-emit pattern warnings.

### Removed

- **BREAKING: Python 3.10 is no longer supported.** The minimum is now 3.11.
  Nothing supported 3.10 in practice — ruff already targeted `py311`, CI never
  ran it, and 43 tests fail there — so the declaration was a claim nothing
  checked. **Existing 3.10 installations keep working and nothing breaks at
  runtime**: `requires-python` means pip and uv simply will not offer this
  version to a 3.10 environment, so a 3.10 user stays on 0.22.0 rather than
  receiving a broken upgrade.

### Fixed

- **`__set_name__` rejections are asserted portably.** Python 3.12 stopped
  wrapping exceptions raised in `__set_name__`, so the nesting-rule tests —
  which asserted `RuntimeError` and read `__cause__` — failed on 3.12 and 3.13.
  They now assert the exact shape each interpreter produces. The library's
  behaviour is unchanged; only the tests were version-bound.

## [0.22.0] - 2026-06-14

### Added
- **`FoldReducer` — a third built-in channel shape for accumulating state** (#111). Where `Reducer` appends and `ScalarReducer` takes the last write — both projecting an event in isolation — `FoldReducer` left-folds each matching event into a single state object whose next value depends on the prior state (counters, merging dicts, re-derived cursors). Each event owns its transition through `fold(self, state)` (mirroring `MessageEvent.as_messages()`), so callers supply only the channel `name`, the `event_type(s)`, and a `default_factory`; pass an explicit `fold=` for events that don't carry the method. A `fold` returning the new `RESET` sentinel clears the channel back to `default_factory()`, `SKIP` leaves it unchanged, and any other value — including `None` — becomes the new state. The base handles the streaming path, the channel-merge path (including the contributions-vs-pre-folded-state duality on `update_state`/replay, via a private wrapper so the fold state may itself be a `list`), and seed/replay — removing ~60 lines of error-prone per-channel `BaseReducer` plumbing. `FoldReducer` is **generic over its state type `S`** (inferred from `default_factory`), so `reducer.empty`/`reducer.seed(...)` are typed rather than `Any`; pin `FoldReducer[StateType](...)` to have mypy flag a `default_factory`/`fold` shape disagreement.
- **`BaseReducer` is now public** (#111). Bespoke reducers must subclass it; it was previously reachable only via the private `langgraph_events._reducer`. Exported alongside the existing `Reducer`/`ScalarReducer`.
- **`Foldable` protocol** (#111). A `@runtime_checkable` structural type for events carrying a `fold(self, state)` method; it types `FoldReducer`'s event argument without `Any`. Satisfied structurally — events must not inherit it.

### Changed
- **A reducer's `event_type` annotation is widened to `type | types.UnionType | tuple[type, ...]`** (#111). Passing an `A | B` union or a tuple of types already worked at runtime (`isinstance` accepts both) but was mistyped, forcing a `# type: ignore[assignment]` at every multi-type reducer. Type-only change.

## [0.21.0] - 2026-06-13

### Added
- **The reserved-modifier guard now covers all three `Command` class-level modifiers** (#105). Declaring `raises` or `invariants` as an annotated dataclass field (missing `ClassVar`) raises at class creation with the same `ClassVar`-steering message as `previously`: an annotated `raises: tuple = (...)` silently became a frozen field serializing exception classes into every checkpoint payload while routing kept working, and `invariants: dict = {...}` died inside dataclasses with advice ("use `default_factory`") that would silently disable invariant enforcement — a factory field has no class attribute for the framework to read. PEP 563 string annotations are judged by dataclasses' own ClassVar resolution, never a framework heuristic, so working spellings (including module-level `ClassVar` aliases) are unaffected.
- **Reserved-node collisions rejected at build** (#105). A `previously=` alias or a handler node name (an explicit `@on(node_name=...)` pin, or a function literally named after the node) claiming a reserved graph node — the framework's `__seed__`/`__router__` or LangGraph's `__start__`/`__end__` — now fails at `EventGraph(...)` construction with a framework message naming the claimant. Before, all four built cleanly and died at first compile/invoke with an error that never named the declaration that smuggled the name in (``Node `__seed__` already present.``).

### Changed
- **Breaking (pathological):** a Command declaring a genuine event *field* named `raises` or `invariants` now errors at class creation (#105) — such a field already silently collided with the modifier reads, so no working class is affected.
- **Inline-command handler names elided across every namespace rendering** (#107, #108). An inline `Command.handle()` is identified by its command — the method name (`handle`, the positional `handle_2`/`handle_3` dedup suffix, or even a deliberate rename like `place`) carries no information and the suffix churned snapshots on reorder. So inline names are now dropped everywhere they were redundant: mermaid edge labels (causation/raises markers like `-->|"[chain]"|` and `-.->|"(raises)"|` remain), the side-effect footer (lists the command), reactor-hub nodes (anonymous dot keyed by command qualname, preserving the fanout-concentration layout), the text view's `(handlers: …)` suffix and causal-notes `via`, the JSON `commands.*.handlers` field (now `[]` for inline-only commands — recoverable from the top-level `command_handlers[].inline`), and the build-time `raises=` coverage error (names the command, not `handle_2`). Free `@on(...)` reactors and external `@on(SomeCommand)` handlers keep their function names everywhere. Also dedupes identical `(raises)` edges from a multi-exception `raises=(E1, E2)`. Render/report-only: node identity and checkpoint-resume keys are untouched; the JSON `handlers` shape change and a one-time mermaid snapshot regeneration are the only downstream effects. The `NamespaceModel` JSON `schema_version` bumps `2` → `3` for the changed-meaning of `commands.*.handlers`.

## [0.20.0] - 2026-06-11

### Added
- **Inline `Command` nodes can declare historic node names — `previously: ClassVar = (...)`** (#103). Mirrors the existing `raises`/`invariants` class-level modifiers: `_expand_command_handlers` now forwards the attribute into the same alias machinery as `@on(previously=...)`, so a checkpoint paused under a renamed command node (e.g. the `Persona/Story/Scenario.Persist` → `EntityLifecycle.Persist` consolidation) resumes instead of degrading per `on_unresumable`. Deliberately **not** MRO-inherited — a historic node name identifies exactly one class; a subclass must not capture its parent's checkpoints. The handler coverage gate picks the aliases up automatically. Note: renaming a command class is *both* renames at once — pair `previously` with `@migrate_from` and a namespace-aware serde so the checkpointed payload revives as the new class (see the new docs warning).
- **New validation guards** (fail early, never at first production read): `@on(previously=...)` accepts only a str or sequence of non-empty str — non-string/whitespace names (previously died as a baffling LangGraph `add_node`/compile error) and exhaustible iterables like generators (silently empty on the second graph build) are rejected at decoration, with the error voiced as the spelling the user wrote (`@on()` vs `Command 'X'`); alias collision/duplicate errors now name the claimant by its checkpoint identity (command qualname), naming both claimants for a duplicate; declaring `previously` as an annotated dataclass field on a `Command` (missing `ClassVar`) raises at class creation — it would otherwise silently serialize into every checkpoint payload while aliasing appeared to work.

### Fixed
- **Docs prescribed an unwritable recovery** for pre-#97 positional checkpoint names: `@on(previously="handle_N")` "on that command's handler" cannot be expressed for an inline method (the inferring path rejects `self`-only signatures). The class attribute is the supported spelling.

## [0.19.0] - 2026-06-11

### Added
- **Origin-scoped backfill — `@migrate_from(..., backfill={field: default})`** (#101). When N event classes collapse into one surviving class, each origin can now pin its own value for a required field the old payloads never carried (e.g. a discriminator), applied before the rename. Precedence: payload value > origin fill > class-global `@backfill`. Hand-authored `AddField` keyed on a historic identity is now accepted and origin-scoped — the escape hatch for per-origin `default_factory`. See the ["Consolidating N classes into one"](docs/event-migrations.md#consolidating-n-classes-into-one) recipe.
- **New validation guards** (all before first production read): duplicate `(identity, field)` fills and fill field names that don't exist on the live class raise at serde construction; a duplicated origin qualname, an empty/mutable `backfill=` value, and `backfill=` on a multi-qualname chain raise at decoration; `legacy_write=True` with origin-scoped fills raises at construction (consolidations cannot ride legacy writes).

### Changed
- **`assert_all_baselined_revive` now exercises back-fills instead of masking them with placeholders.** Fields covered by an `AddField` get the real injected value instead of a `None` placeholder — a broken fill fails in CI, and `__post_init__` validation on a back-filled field now passes the gate (previously it needed `assert_all_baselined_resolve`).

## [0.18.0] - 2026-06-08

### Fixed
- **Inline `Command.handle()` nodes now have a stable, order-independent checkpoint identity** (#97). Inline command handlers were registered as graph nodes named after the method (`handle`) and then de-duplicated **positionally** (`handle`, `handle_2`, …) by their order in `EventGraph(handlers=[...])`. Because several `Command.handle()` methods are real pause points (they return `Interrupted` / AG-UI `FrontendToolCallRequested`), a checkpoint could pause *inside* one of these positionally-named nodes — so reordering `handlers=[...]` silently remapped which command each `handle_N` dispatched to, with nothing in the baseline/coverage tooling able to detect it (the baseline stores a reorder-invariant sorted set of names). Inline command-handler nodes are now keyed by the command's `__qualname__` (e.g. `Order.Place`) — the same stable identity used elsewhere for command privacy and return contracts — so the node a paused checkpoint resumes into never depends on registration order. `graph.handler_names`, the resume gate, and `write_baseline`'s `handlers` list all record these qualname identities. The human-readable handler label shown in choreography / mermaid / `HandlerRaised`/`InvariantViolated` diagnostics is unchanged (still the method name).

### Changed
- **Inline command handler baselines must be regenerated once after upgrading** (#97). A pre-#97 baseline recorded inline command handlers by their positional `handle`/`handle_N` names, which no longer resolve to a live node — so `assert_all_baselined_handlers_cover` will (correctly) raise `HandlerCoverageError` until you run `write_baseline(graph, BASELINE)` to re-record the qualname identities. A checkpoint that was paused inside an inline command handler under the old positional name before the upgrade will not resume afterward (the old name was order-dependent and cannot be reconstructed safely); recover a specific in-flight checkpoint by adding `@on(previously="handle_N")` to that command's handler, or set `EventGraph(on_unresumable="halt"|"warn")` to degrade gracefully.

## [0.17.0] - 2026-06-08

### Fixed
- **`aresume()` / `astream_resume()` no longer crash against async-only checkpointers** (#95). The `on_unresumable` resume-pending gate introduced in 0.15.0 read checkpoint state through the **synchronous** `get_state`, which `AsyncPostgresSaver` (and any async-only checkpointer) rejects from the running event loop — raising `asyncio.InvalidStateError` before any policy (`raise`/`warn`/`halt`) could run, and taking down every async resume (including the library's own `AGUIAdapter` SSE path). The async resume gate and the async `on_unresumable` policy arms now read and write checkpoint state exclusively through the async API (`aget_state`/`aupdate_state`), matching how `ainvoke()`-based flows already behave. The sync `resume()`/`stream_resume()` paths are unchanged.

### Added
- **`EventGraph.aget_state(config)` — async sibling of `get_state()`.** Reads event-level thread state through the async checkpointer API, so deployments on async-only checkpointers can inspect a thread's state and interrupt status from within the event loop without tripping `InvalidStateError`.

## [0.16.0] - 2026-06-08

### Added
- **`assert_resume_recovers(before, after, *, seed, resume_with)` — handler-recovery test helper.** The behavioral analog of `assert_all_baselined_revive` for handlers: it invokes `before` to pause a thread, resumes `after` on the same checkpoint, and asserts a `Resumed` was emitted (real recovery, not a silent drop or `halt`), collapsing the hand-rolled interrupt→rebuild→resume dance into one call. Both graphs must share one checkpointer. Exported from `langgraph_events.serde` / `langgraph_events.serde.migrations`. The `docs/event-migrations.md` "Testing your migrations" chapter now covers **both** tracks — a "Handler coverage gate" and "Testing handler recovery" section sit alongside the event gates.

## [0.15.0] - 2026-06-08

### Added
- **Handler evolution — alias recovery + CI coverage gate.** Handlers now evolve under the same model as events. `@on(node_name="stable_id")` pins a **stable node identity** decoupled from the Python function name, so renaming/moving the function never breaks an interrupted checkpoint. `@on(previously="old_node")` (str or tuple) declares historic node names; the graph registers an **alias node** per name so a thread paused inside a renamed handler (via `Interrupted`) re-enters it transparently on `resume()`. `assert_all_baselined_handlers_cover(graph, baseline_path)` (exported from `langgraph_events.serde` / `langgraph_events.serde.migrations`) is the handler analog of the event coverage gates: it asserts every baselined handler node name is still live or alias-covered, raising the new `HandlerCoverageError` otherwise. `HandlerCoverageError` and `MigrationCoverageError` are now siblings under a shared `CoverageError(AssertionError)` base (catch both with `except CoverageError`), each with its own structured `.uncovered`. `write_baseline` now records handler node names — baseline files bump to **v2** while older **v1** baselines still load (their handler set is treated as empty). A colliding alias is rejected at graph build. See [Handler renames](docs/event-migrations.md#handler-renames).
- **`EventGraph(on_unresumable=...)` — runtime resume policy.** `resume()` on a thread that is not awaiting input (paused handler renamed/removed, thread already finished, or a double-resume) no longer silently no-ops. `on_unresumable="raise"` (default) raises the new `UnresumableError`; `"warn"` emits a `UserWarning` and leaves the log unchanged; `"halt"` appends a terminal `Unresumable(Halted)` event and finalizes the thread. The trigger keys on the checkpoint having no scheduled work, so a legitimate `pre_seed`-before-resume (e.g. `AGUIAdapter`'s `FrontendStateMutated`) and a Phase-1 `@on(previously=...)` alias both resume normally. Applies to `resume`/`aresume`/`stream_resume`/`astream_resume`. `Unresumable` and `UnresumableError` are exported from `langgraph_events`.

### Changed
- **`resume()` raises on a non-resumable thread (behavior change).** Resuming a thread that is not awaiting input previously did nothing silently; it now raises `UnresumableError` by default. Opt into `EventGraph(on_unresumable="halt")` or `"warn"` to keep it non-fatal.

## [0.14.0] - 2026-06-08

### Added
- **`assert_all_baselined_resolve(serde, baseline_path)` — resolution-only coverage gate** (#92). A rename-aware reachability check that proves every baselined identity still resolves to a live `Event` class **without constructing** it (no `__init__`/`__post_init__`). It is the gate to reach for when a full-graph baseline contains events `assert_all_baselined_revive` can't placeholder-construct — anything with construction-time validation (e.g. agui's `FrontendToolCallRequested`, which rejects an empty `name` in `__post_init__`), framework `SystemEvents`, or module-level `IntegrationEvents`. Such a baseline now passes with no filtering, and still fails loudly when any identity is renamed/removed without a covering migration. `revive` remains the stronger constructability gate for placeholder-tolerant events. Exported from `langgraph_events.serde` and `langgraph_events.serde.migrations`. See [Event migrations](docs/event-migrations.md#coverage-gates).

### Changed
- **Coverage gates unified into one free-function family (breaking)** (#92). The three baseline-coverage checks now share one signature — `gate(serde, baseline_path)` — and one error base. `NamespaceAwareSerde.assert_covers(baseline_path)` is **removed**; use `assert_all_baselined_cover(serde, baseline_path)` (exported from `langgraph_events.serde` / `langgraph_events.serde.migrations`), which has the identical set-membership semantics. `MigrationCoverageError` now subclasses **`AssertionError`** (was `ValueError`) so all three gates raise a single catchable base, while it keeps its structured `.uncovered` tuple. `revivable_identities()` is unchanged. **Migration**: replace `serde.assert_covers(BASELINE)` with `assert_all_baselined_cover(serde, BASELINE)`; if you caught `MigrationCoverageError` as a `ValueError`, catch `AssertionError` (or the class itself) instead.

## [0.13.0] - 2026-05-31

### Added
- **`AGUIAdapter(on_unmapped=...)` policy for events with no AG-UI mapping** (#90). A keyword-only `on_unmapped: Literal["warn", "ignore", "raise"]` knob controls what happens when an event reaches `FallbackMapper` (or `InterruptedMapper`'s non-serializable branch) without implementing `AGUISerializable`. `"warn"` (default) keeps today's once-per-class `UserWarning` then drops — **non-breaking**. `"ignore"` silently drops, the off-switch for apps that are mostly internal orchestration events. `"raise"` raises the new `UnmappedEventError` (a `TypeError` subclass) naming the offending class, turning the dev-lint into a hard CI gate. The policy applies to both fallback sites; serializable events and `InterruptedWithPayload` are unaffected.

## [0.12.0] - 2026-05-28

### Fixed
- **`RunPaused` is now emitted at most once per `/run`** (#88). Pre-fix, the router re-emitted `RunPaused` on every router invocation while `time.monotonic() >= deadline`, so any parallel handlers still in flight when the deadline expired could each fan-in and trigger another emission. Each instance carried a fresh `elapsed_seconds`, defeating id-based dedup in any reducer projecting `RunPaused` into a downstream channel (e.g. inline pause-notice messages accumulated one entry per emission instead of one per pause). The router now tracks a `_run_paused_emitted` flag on the state; late fan-ins past the deadline drain cleanly without re-emitting. The seed resets the flag for each fresh `/run` so subsequent runs on the same `thread_id` work normally.

### Changed
- **`RunPaused` no longer has a default AG-UI wire mapping** (breaking — #88). The class no longer implements `agui_event_name` / `agui_dict()`, so `FallbackMapper` skips it (one-time warning) instead of emitting `CustomEvent(name="interrupted", value={"kind": "soft_timeout", …})`. The previous default collided on the wire with HITL `Interrupted` events (same `name="interrupted"`, discriminated by `value.kind`) and forced every client to branch on the discriminator. Apps that want a pause signal on the wire register a custom mapper and choose the shape themselves:
  ```python
  from ag_ui.core import CustomEvent, EventType
  from langgraph_events import RunPaused

  class PauseMapper:
      def map(self, event, ctx):
          if not isinstance(event, RunPaused):
              return None
          return [CustomEvent(
              type=EventType.CUSTOM,
              name="run.paused",
              value={"elapsed_seconds": event.elapsed_seconds},
          )]

  adapter = AGUIAdapter(graph, seed_factory=..., mappers=[PauseMapper()])
  ```
  For an inline message-channel pause notice, see the reducer recipe in `docs/control-flow.md#surfacing-the-pause-inline`.

## [0.11.0] - 2026-05-25

### Fixed
- **Mermaid v11 pipe-label bracket/brace quoting**: `NamespaceModel.mermaid()` now wraps any pipe-edge label containing `[`/`]` or `{`/`}` in double quotes, so the `[chain]` / `[orchestrate]` causation suffix renders under `mermaid@11.x` instead of tripping the node-shape parser (`SQE`/`SQS` expected error). Braces are covered defensively for the same class of bug (diamond/hex node-shape literals); no current emitter produces them. Labels without these chars are unchanged. Affected example snapshots (`supervisor`, `error_recovery`, `expense_approval`) and `docs/patterns.md` regenerate with quoted causation labels; no other diagrams change. (closes #85)

### Added
- **Per-call soft-timeout via `deadline=` kwarg + `RunPaused(SystemEvent)`** — every graph entry point (`invoke`, `ainvoke`, `resume`, `aresume`, `stream_events`, `astream_events`, `stream_resume`, `astream_resume`) and `AGUIAdapter.stream` now accept `deadline: float | None = None`, an absolute `time.monotonic()` reference. When the router observes a wall-clock time past the deadline between dispatch rounds, it emits `RunPaused(elapsed_seconds=…)` (a new `SystemEvent`, **not** a `Halted` subtype) and the run terminates cleanly through the existing finalize path. The cursor advances past `RunPaused` so a fresh `/run` on the same `thread_id` excludes it from `new_events` and continues from the LangGraph checkpoint — resume is implicit, not via `Command(resume=...)`. On the AG-UI side, `RunPaused` maps to `CustomEvent(name="interrupted", value={"kind": "soft_timeout", "elapsed_seconds": …})` via the existing `FallbackMapper`, sharing the wire vocabulary with HITL pauses (discriminated by `value.kind`). Closes #83.

## [0.10.0] - 2026-05-23

### Added
- **Required `ScalarReducer` values via handler type annotations** — when a handler declares a reducer parameter whose type annotation rejects `None` (e.g. `strategy: str`), the framework raises `ReducerNotSetError` at injection time if the channel value is `None`. Use `str | None` / `Optional[str]` / `Any` / `object` (or omit the annotation) to opt out and keep the legacy permissive behavior. `ReducerNotSetError` is a `ValueError` subclass exported from `langgraph_events`; the precondition raise sits outside the `raises=` catch boundary, so a broad `raises=ValueError` declaration cannot silently swallow it. Note: handlers previously receiving `None` through a non-`None`-typed parameter and handling it internally will now raise at injection — widen the annotation to opt out.
- **`NamespaceModel.Edge.causation`** — a causal-role axis alongside `kind`: `intent` (a command emits its own outcome), `react` (a reactor emits a fact), `orchestrate` (a reactor emits a command — a saga move), `chain` (a command emits another command), or `None` (`raises`/`framework`/boundary-crossing edges). Surfaces in `text()`, `to_dict()`/`json()` (additive — no schema bump), and `mermaid()` (`orchestrate` bold, `chain` dashed-warning; `intent`/`react` unchanged, so no diagram churn). The `supervisor`, `error_recovery`, and `expense_approval` example diagrams now mark their orchestration edges.
- **`CommandChainWarning`** — emitted at `namespaces()` time for every `chain`-causation edge (an inline `Command.handle()` returning another `Command`). Filter via `warnings.filterwarnings("ignore", category=CommandChainWarning)`.
- **Event migration story for `NamespaceAwareSerde`**: the `@migrate_from(*old_qualnames)` decorator lets consumers rename or relocate event classes without invalidating existing checkpoints. `EventGraph.from_namespaces(Persona, checkpointer=MemorySaver())` auto-wires a `NamespaceAwareSerde` scoped to exactly those namespaces — no manual serde construction and no namespace-tuple double-threading. A user-supplied `NamespaceAwareSerde` (custom `migrations=`/`legacy_write=`) passed via `MemorySaver(serde=...)` always wins (the opt-out). Old `(module, qualname)` is rewritten to the current identity in memory on read; the wire format is unchanged so rolling deploys with prior library versions keep working. `AddField(module=..., qualname=..., field=..., default=...)` covers new required fields on existing events; `Migration.rename(...)` / `Migration.add_field(...)` are single-op sugar over the raw `Migration(operations=(...))` form, with `name` optional everywhere. Hand-authored `Migration` lists for cross-module renames or composite operations flow through the serde's `migrations=` kwarg and compose with the decorator-driven collection. A `legacy_write=True` flag makes new code emit payloads under the oldest historic qualname for the rollout window — scope-symmetric with the read path, so out-of-scope decorated classes are NOT relabelled. `detect_changes(graph, baseline_path)` + `write_baseline(graph, path)` (under `langgraph_events.serde.migrations.detect`) provide a suggestion engine for pre-commit hooks; the baseline file is versioned and a mismatch raises at read time. Reducer projections are intentionally not migrated — events are the truth and projections re-derive. See [Event migrations](docs/event-migrations.md). (closes #70)
- **Pre-deploy migration confidence surface** — close the loop *before* a deploy instead of discovering a missing migration on the first production read. `synthesize_legacy_payload(module, qualname, kwargs)` (exported from `langgraph_events.serde` / `langgraph_events.serde.migrations`) builds the exact `(format, bytes)` a prior release would have written, so a CI test can assert it still revives under the current migration table without importing private wire-format symbols. `NamespaceAwareSerde.assert_covers(baseline_path)` raises `MigrationCoverageError` (exported from `langgraph_events.serde.migrations.detect`, `.uncovered` lists the offending identities) if any identity in the baseline is neither still live nor covered by a rename migration. `NamespaceAwareSerde.revivable_identities()` exposes the read-only union of live + migration-source identities for custom coverage rules. See [Event migrations](docs/event-migrations.md). (closes #70)
- **`assert_all_baselined_revive(serde, baseline_path)`** (exported from `langgraph_events.serde` / `langgraph_events.serde.migrations`) — pushes a synthesized legacy payload for every baselined identity through the real ext-hook and asserts it revives to an `Event`, filling required fields of the resolved live class with placeholders. Zero per-event maintenance: a new `@migrate_from` plus a regenerated baseline is covered with no new test code. `python -m langgraph_events.serde.migrations <module:factory> <baseline>` exits non-zero when topology diverges, making a forgotten migration a build failure rather than a production read error.
- **`@backfill(field, *, default=… | default_factory=…)`** (exported from `langgraph_events.serde` / `langgraph_events.serde.migrations`) — the class-scoped, auto-collected sibling of `@migrate_from` for the "added a now-required field" case. The back-fill value lives on the class, not in a remote `migrations=` list, and is picked up by the same namespace walk — so `EventGraph.from_namespaces(NS, checkpointer=MemorySaver())` covers it with zero manual serde wiring. Composes with `@migrate_from` on the same class (rename first, then back-fill on the resulting identity); stacked decorators accumulate; reuses the `AddField` mutable-default guard (not a forked rule). For cross-module relocations or composite operations, the hand-authored `Migration` / raw `AddField` escape hatch remains in `serde.migrations`. See [Event migrations](docs/event-migrations.md#adding-a-required-field).
- **`Migration.rename(to=<class>)` / `Migration.add_field(target=<class>)`** — the live (post-rename / AddField target) identity may now be passed as the class object; module/qualname are derived from it, so an IDE rename moves with the code instead of silently breaking a string. The string forms remain for the cross-module case; fully backward compatible.
- **`replay_reducer(reducer, events)`** (exported from `langgraph_events.serde`) — names the recovery pattern for cases where a reducer's projection function or output shape changed and the cached channel value is stale. Reads the migrated event log, re-runs the reducer's projection via `BaseReducer.seed`, and returns the rebuilt value for the caller to write back through the checkpointer. Pairs with the empirically-mapped reducer-state matrix in `docs/event-migrations.md`.

### Changed
- **`IntegrationEvent` placement is now enforced (breaking)**. An `IntegrationEvent` nested inside a `Namespace` or `Command` raises `TypeError` at class creation — it crosses a context boundary by definition. Symmetric with the existing `Command` / `DomainEvent` nesting rules; `SystemEvent` is unaffected (`Halted` subclasses may still nest for locality). **Migration**: move any nested `IntegrationEvent` to module level.
- **`write_baseline` refuses silent regression (breaking)**: `write_baseline(graph, path)` now raises `BaselineRegressionError` (exported from `langgraph_events.serde.migrations.detect`; `.removed` lists the dropped `(module, qualname)` identities) when *path* already exists and the new snapshot would drop identities the old baseline recorded — overwriting them away would make `detect_changes` / `assert_covers` permanently blind to a forgotten migration. The previously prose-only rule ("commit the baseline alongside the migration, never after") is now enforced. Pass `write_baseline(graph, path, allow_removed=True)` for intentional deletes. The check compares baseline ↔ topology only; it never inspects the serde — coverage stays with `assert_covers` / `assert_all_baselined_revive`. **Migration**: if a regenerate legitimately drops identities, pass `allow_removed=True`; otherwise author the `@migrate_from` / `@backfill` / `Migration` first, then regenerate.
- **Top-level `langgraph_events.serde` surface narrowed (breaking)**: raw `RenameEvent` and `AddField` are no longer re-exported from `langgraph_events.serde`. The common path is decorator-first (`@migrate_from`) plus `Migration.rename` / `Migration.add_field` sugar. The raw operation constructors remain importable from `langgraph_events.serde.migrations` for the rare composite multi-op `Migration`. **Migration**: change `from langgraph_events.serde import RenameEvent, AddField` to `from langgraph_events.serde.migrations import RenameEvent, AddField`.
- **`NamespaceAwareSerde.dumps_typed` no longer falls through to `JsonPlusSerializer.dumps_typed` on `MsgpackEncodeError`**. The fallback was a no-op in the default config (upstream re-raised the same error) and, with the parent's binary-fallback kwarg enabled, would silently emit unsafe-binary bytes that bypass the migration table. The encode error now propagates at the source. `ormsgpack.MsgpackEncodeError` is a `TypeError` subclass, so existing `except TypeError:` handlers continue to catch it; the only user-visible diff is that the `UserWarning` no longer fires before the raise.
- **Scatter must enumerate concrete event types**. Return annotations of the form `-> Scatter`, `-> Scatter[Any]`, `-> Scatter[Event]`, `-> Scatter[DomainEvent]`, `-> Scatter[IntegrationEvent]`, `-> Scatter[SystemEvent]`, and `-> Scatter[T]` (TypeVar) are now rejected at `EventGraph` construction with a `TypeError`. These shapes contributed no usable types to the privacy graph, so the v0.9.0 `enforce_command_privacy()` check passed silently and the leak was only caught at runtime — turning the previous build-time `CommandPrivacyError` into something developers could paper over by widening the annotation. **Migration**: replace `-> Scatter` with `-> Scatter[Event1 | Event2 | ...]` enumerating the concrete events you scatter. The error message points at the right form (and steers away from demoting the offending `Command` to `DomainEvent`/`IntegrationEvent`, which would silently lose privacy guarantees).
- **NamespaceModel JSON schema bumped to v2**. `CommandHandler.has_untyped_scatter` and `Policy.has_untyped_scatter` are removed (the represented condition is now impossible at build time). The "Scatter handlers: …" footer in mermaid output and the bare `Scatter` token in text output are both gone for the same reason. Consumers reading `to_dict()` must update.

### Fixed
- **Scatter return-type parser**: `Scatter[A | B]` now extracts both `A` and `B` as scatter targets, matching the behavior of the equivalent `Scatter[A] | Scatter[B]` form. Previously the Union argument was rejected by an `isinstance(..., type)` check and silently dropped, so `info.scatter_types` came back empty — orphan-warning detection and choreography diagrams missed those targets entirely. (closes #66)

## [0.9.0] - 2026-05-07

### Added
- **CommandPrivacyError**: new exported exception class (subclass of `TypeError`). Raised at `EventGraph` construction when a handler emits a `DomainEvent` it isn't allowed to emit. Two symmetric rules enforce that *outcomes nested inside a `Command` are private to that Command's inline `handle()`*:
  1. An inline `Command.handle()` may only return events nested under that same Command (or a parent Command, via inheritance) — not sibling/namespace-level events, not another Command's private outcomes.
  2. Any non-inline handler (`@on(...)` reactor) is forbidden from emitting a Command-private event. The owning Command's `handle()` is the single canonical producer; recovery and observer reactors emit namespace-level sibling events instead.
- **Class-level `invariants` and `raises` on `Command`**: declare invariants and catchable exceptions for an inline `handle()` directly on the Command class (e.g. `invariants: ClassVar = {Inv: predicate}`, `raises: ClassVar = (RateLimitError,)`). The framework forwards them to the synthesized `@on(Cmd)` wrapper, honoring inheritance via `getattr` so a child Command picks up a parent's declarations. Closes a gap where modifiers were only available on the external `@on(Cmd, ...)` form — prerequisite for the privacy migration above, since recovery patterns that previously rode on external `@on(Cmd, ...)` declarations now live on the Command class itself.
- **Domain-named inline handlers**: a Command's inline handler can now be named after its verb (`place`, `ship`, `submit`, …) instead of the generic `handle`. The framework picks up the sole public method in the class body. To preserve the "one intent, one handler" semantic, declaring more than one public method on a Command raises `TypeError` at class creation; underscore-prefix helpers to opt them out of the count.

### Changed
- `@on(Cmd) -> Cmd.Outcome` patterns are no longer allowed. A Command's nested outcomes must be produced by `Cmd.handle()` only. Tests, examples, and docs that previously relied on the external `@on(Cmd) -> Cmd.Outcome` form now use inline `handle()` (with class-level `invariants` / `raises` where needed). Recovery patterns that produced nested outcomes from outside (`@on(InvariantViolated) -> Cmd.Rejected`, `@on(HandlerRaised) -> Cmd.Rejected`) emit namespace-level events instead — for example, `Order.Place.Rejected` becomes `Order.Rejected` in the canonical `examples/order.py`.

### Fixed
- **Inline-handle outcome-coverage messages**: single-outcome Commands no longer report duplicate outcome names (`Done, Done`) in the missing-outcomes error — the synthesized `Outcomes` alias is excluded from the walk over nested DomainEvents. Bare `-> Scatter` on a Command's inline handler is now rejected up front with a message pointing at `Scatter[…]` or dropping the annotation, instead of falling through to a confusing `(no types)` coverage error. (closes #73)

## [0.8.0] - 2026-05-05

### Fixed
- **serde**: `NamespaceAwareSerde` no longer silently degrades non-event payloads (Pydantic models, plain dataclasses, project types) to `dict` on `langgraph-checkpoint>=4.0.3`. The fallback for ext codes the namespace-aware serde doesn't own previously dispatched through the *module-level* `_msgpack_ext_hook` from `langgraph.checkpoint.serde.jsonplus`, which 4.0.3 rebound to a strict hook (`_create_msgpack_ext_hook(allowed_modules=None)`). That bypassed both `LANGGRAPH_STRICT_MSGPACK` and the `JsonPlusSerializer` constructor's `allowed_msgpack_modules` argument, so every checkpointed non-event value came back as a `dict` (with a logged `Blocked deserialization of <module>.<name>` warning) and downstream code blew up with `AttributeError` when it called methods on the supposed instance. The fallback now routes through the parent's *per-instance* `_unpack_ext_hook`, so the constructor-level allowlist and `LANGGRAPH_STRICT_MSGPACK` work as documented for non-event payloads passing through `NamespaceAwareSerde`. (#68)

### Added
- **agui**: `agui_messages_to_langchain(messages, *, drop_invalid_tool_calls=False) -> list[BaseMessage]` — public helper at `langgraph_events.agui` for converting AG-UI protocol messages to LangChain `BaseMessage` instances. Mirrors the existing internal LangChain→AG-UI direction. Handles `UserMessage` (string and multimodal content with the `url > data > id` priority cascade), `AssistantMessage` (with optional `tool_calls`, parsing `function.arguments` JSON), `SystemMessage`, and `ToolMessage`. `ReasoningMessage` and `DeveloperMessage` are skipped with a DEBUG log; `ActivityMessage` and unknown roles raise `ValueError`. With `drop_invalid_tool_calls=True`, tool calls whose `function.arguments` fail to JSON-parse are dropped (WARNING-logged); if all of an `AssistantMessage`'s tool calls are dropped, the message itself is dropped. Removes the need to depend on `ag-ui-langgraph` for this utility.
- **agui**: `merge_frontend_messages(input_data, checkpoint_state, *, reducer_name="messages", drop_invalid_tool_calls=True) -> tuple[BaseMessage, ...]` — high-level helper for `ResumeFactory` implementations. Reads existing messages from `checkpoint_state["reducers"][reducer_name]`, converts `input_data.messages` via `agui_messages_to_langchain`, and merges via langgraph's `add_messages` (id-based dedup). Defensive default for malformed tool-call JSON; pass `drop_invalid_tool_calls=False` for strict parity with upstream.
- **agui**: `extract_resume_input(input_data) -> Any` — pulls resume input from `RunAgentInput.forwarded_props["command"]["resume"]`. If the value is a string, attempts `json.loads` (returns the decoded JSON value — dict, list, scalar — on success; the raw string on `JSONDecodeError`). Dicts/lists/numbers pass through unchanged. Returns `None` if absent or falsy.

## [0.7.0] - 2026-05-05

### Added
- **DomainPatternWarning**: new exported warning class. Emitted at `EventGraph.namespaces()` time when 2+ events in the same namespace fan out (via 2+ distinct reactor handlers) to identical target sets — typically a sign that a shared abstraction was missed (a common base event or a single reactor on a common subscription would collapse them). Detection is exact-set-equality only; subset/superset overlaps don't qualify. Silence via the standard warnings filter: `warnings.filterwarnings("ignore", category=DomainPatternWarning)`. The warning fires once per pattern at construction time; renderers (`mermaid()`, `text()`, etc.) do not re-emit.
- **NamespaceModel.mermaid(namespace_order=…)**: new keyword controlling how namespace subgraph clusters are sequenced. Pass `"alphabetical"` to preserve the legacy alphabetical order (useful for snapshot-pinned consumers). Defaults to `"affinity"` — see Changed.
- **NamespaceModel.mermaid(reactor_hub_min=…)**: new opt-in keyword (default `None`) for hub-style fanout rendering. When set to an integer `N`, any `(source, handler)` pair producing `≥ N` solid-or-scatter targets is rewritten as `Source --> Hub --> {targets}` — the handler name moves from being repeated on every fanout edge to a single label on the hub node (small circle, `:::hub` styling), placed inside the source's subgraph. Useful for tracing in large graphs where the same handler-name label clutters the area around high-fanout sources. Invariant-gated reactors are not hubbed (the invariant chain already concentrates dispatch); `raises` edges are not hubbed (single error path). Per-target arrow style (solid vs scatter) is preserved on the `Hub --> Target` edges. No default behavior change — leave unset for the existing flat fanout output.

### Changed
- **mermaid**: subgraph clusters now sort by inter-namespace edge affinity by default (greedy nearest-neighbor) instead of alphabetically. Heavily-connected namespaces land adjacent in the rendered diagram, substantially shortening the long crossing arrows that dominate multi-namespace graphs (Definition→Persona/Story/Scenario in real usage). Affinity counts solid + scatter + framework reaction edges plus invariant chain edges; ownership-fill `-.- ` arrows and `raises` edges don't contribute. Ties break alphabetically. **Default rendered output changes for any consumer with multi-namespace graphs** — pass `mermaid(namespace_order="alphabetical")` to opt out for snapshot stability.

## [0.6.2] - 2026-05-04

### Fixed
- **mermaid**: `render_mermaid_choreography` no longer collapses cross-namespace events that share a leaf class name. Previously, a project with sibling namespaces (e.g. `Persona.Approve.Approved`, `Story.Approve.Approved`, `Scenario.Approve.Approved`) emitted a single mermaid node for all three, merging their incoming/outgoing edges. The renderer now detects leaf-name collisions across the model and escalates only the colliding classes to qualname-based node IDs (`Persona_Approve_Approved`, …); display labels stay terse since the surrounding subgraph cluster already conveys namespace context. Non-colliding diagrams render byte-identically. (#62)

## [0.6.1] - 2026-05-04

### Fixed
- **serde**: `NamespaceAwareSerde` now preserves namespace identity for `Event` instances nested inside `langgraph.types.Interrupt` (the dataclass LangGraph wraps every interrupted value in before checkpointing). Previously, every namespaced `Interrupted`/`InterruptedWithPayload` subclass round-tripped through a checkpointer would silently decode back as `Interrupt(value=None, id=...)` because LangGraph's generic dataclass branch (`EXT_CONSTRUCTOR_KW_ARGS`) recurses into a hardcoded `_msgpack_default` and bypassed our namespace-aware `default=`. `Interrupt` is now intercepted directly under a dedicated ext code so the wrapped value re-enters our encoder. Note: this fix applies to checkpoints written *after* the upgrade — checkpoints already persisted under v0.6.0 still decode their nested events as `None` (same risk profile as before). (#60)

## [0.6.0] - 2026-05-04

### Added
- **EventGraph**: `services=` kwarg for dependency injection in two forms. (1) `services=[chat_model, session_factory]` — type-keyed; handler params resolve by their type annotation via an MRO walk (a base-class annotation matches a registered subclass instance), with exact-type match preferred over subclass match. (2) `services={"primary_chat": a, "backup_chat": b}` — name-keyed; handler params resolve by name. The mapping form allows multiple instances of the same type. Inline `Command.handle(self, chat_model: BaseChatModel)` and external `@on(...)` handlers share the same mechanism. Resolution order: reducer name → framework type (`EventLog` / `RunnableConfig` / `BaseStore`) → service. Eliminates the closure-factory pattern downstream projects use today to shuttle services into handlers.
- **serde**: new opt-in `langgraph_events.serde.NamespaceAwareSerde` — a `JsonPlusSerializer` subclass that keys `Event` identity by `(__module__, __qualname__)` instead of `(__module__, __name__)`. Drop-in for any LangGraph checkpointer that accepts `serde=` (e.g. `MemorySaver(serde=NamespaceAwareSerde())`). Two namespaces with sibling-named events (`Persona.Approve.Approved`, `Story.Approve.Approved`) now round-trip distinctly; non-event payloads encode exactly as the default serde.
- **InterruptedWithPayload[PayloadT]** (in `langgraph_events.agui`): new generic base for HITL with a discriminated frontend payload. Subclasses implement `interrupt_payload(self) -> PayloadT` and inherit from `Interrupted`. The AG-UI `InterruptedMapper` recognises it directly — no `agui_dict()` override needed. Eliminates the project-local "shim base" pattern downstream HITL projects use today to break import cycles between sibling namespace modules.
- **on_namespace_finalize(cls, callback)**: public hook that schedules a callback to fire once the enclosing Namespace's `__init_subclass__` finishes (after `_stamp_nested_namespace` and `_attach_command_outcomes`). Useful for class decorators that need to call `typing.get_type_hints()` against forward references to siblings inside the same in-progress Namespace body — those references can't resolve while the class body is evaluating, but resolve cleanly at finalize time. The callback receives `(cls, namespace_cls)` so decorators can resolve siblings via `vars(namespace_cls)` without touching private state. Re-exported at `langgraph_events.on_namespace_finalize`.

### Changed
- **EventGraph**: handler params with no injection source now raise `TypeError` at graph construction (previously crashed at first dispatch with a missing-keyword error). Two services of the same exact type are rejected at construction. A handler param annotated as a class that matches multiple registered services is rejected at construction with both candidate type names in the message. Resolution prefers an exact-type service match over subclass matches, so `services=[BaseChatModel(), Anthropic()]` cleanly resolves a `param: BaseChatModel` to the base instance. Annotations equal to `object` are skipped — they would otherwise silently match every registered service.
- **agui**: `FrontendToolCallRequested` (previously top-level `langgraph_events.FrontendToolCallRequested`) now lives in `langgraph_events.agui` alongside `FrontendStateMutated`. Update imports to `from langgraph_events.agui import FrontendToolCallRequested`. **The top-level alias still resolves**, but emits a `DeprecationWarning` per access pointing at the new path; it will be removed in a future release. The class itself is unchanged.

### Fixed
- **on_namespace_finalize**: callbacks registered after the enclosing Namespace has already finalized now fire immediately rather than dangling in the registry forever. Previously a decorator applied post-hoc to an already-bound class was a silent no-op.
- **EventGraph**: variadic handler params (`*args` / `**kwargs`) are no longer flagged as unclaimed at graph construction.
- **serde**: `NamespaceAwareSerde.dumps_typed` now warns when an unencodable payload forces fallback to the upstream serializer (which uses leaf-name identity, collision-prone for nested events). `loads_typed` now raises a clear `ValueError` naming the missing class when a checkpoint references an Event class that has been renamed or removed, rather than the opaque `ValueError("ext_hook failed")` from upstream. Imports of LangGraph private helpers are now guarded with a clear `ImportError` if upstream renames them.

## [0.5.2] - 2026-04-30

### Added
- **agui**: `AGUIAdapter(include_reducers=...)` validation — malformed values (anything other than `bool | list[str]`) now raise `TypeError` at construction instead of silently producing empty snapshots at runtime.

### Fixed
- **agui**: `AGUIAdapter.connect()` and the streaming `StateSnapshotEvent` path no longer leak the EventGraph-internal `events` audit log to clients. The audit log is graph-internal and was causing O(history) wire bloat on every client `Send` via `RunAgentInput.state` round-trip. The strip set is now derived from `_internal._BASE_FIELDS` (single source of truth across all four projection sites) rather than hardcoded; future internal channels propagate automatically. `_extract_frontend_state` also strips internal keys as defense-in-depth against stale-client echo.
- **agui**: Resume-time frontend state now flows through `FrontendStateMutated` instead of bypassing dispatch via `apre_seed(raw_state)`. The adapter computes per-reducer contributions from the FSM event (preserving `fn` semantics — transformations, `SKIP`) and writes them to channels via `apre_seed` *before* the resume's domain dispatch, then injects FSM as a seed to `astream_resume` so it appears in the output stream and the persisted audit log. Reducers that subscribe to `FrontendStateMutated` see the same contract on resume as on the non-resume path; reducers that subscribe to backend domain events are no longer clobbered by stale frontend snapshot keys. `@on(FrontendStateMutated)` *handlers* still do not fire on resume — `Command(resume=...)` carries one value and seeds dispatch out-of-graph; use `@on(Resumed)` for resume-time side effects.

### Changed
- **agui**: `AGUIAdapter.__init__` validates the `messages` reducer eagerly (raises `ValueError` immediately, before any other setup).
- **EventGraph**: `stream_resume` and `astream_resume` now accept a `seeds: list[Event] | None = None` kwarg. Seeds are dispatched alongside the resume in the same step — used by `AGUIAdapter` to route `FrontendStateMutated` through reducers on resume. Power users can plumb their own resume-time companion events through this hook.

## [0.5.1] - 2026-04-24

### Added
- `AGUIAdapter` now emits a new built-in `FrontendStateMutated` event (an `IntegrationEvent`, exported from `langgraph_events.agui`) as the first event of each run when `RunAgentInput.state` carries non-empty client-owned state. The dedicated `messages` key is filtered out (driven by `MessagesSnapshotEvent` / `TextMessageEvent`). Reducers mirroring client-driven channels subscribe to it like any other event — e.g. `ScalarReducer(event_type=FrontendStateMutated, fn=lambda e: e.state.get("focus", SKIP))` — and handlers can optionally `@on(FrontendStateMutated)` to react. On resume, the adapter writes the filtered state directly to reducer channels via `graph.apre_seed` before calling `astream_resume` (LangGraph's `ainvoke` on a pending-interrupt thread would consume the interrupt); in the idiomatic state-key-equals-channel-name pattern the values flow through identically to the non-resume path. `FrontendStateMutated` is not echoed back to the client — its downstream reducer changes surface via the usual `StateSnapshotEvent` path. Works with or without a checkpointer on the non-resume path.

## [0.5.0] - 2026-04-22

### Added
- Event taxonomy: `Namespace`, `Command`, `DomainEvent`, `IntegrationEvent`, `SystemEvent`. `Namespace` subclasses act as namespaces for nested commands and outcomes, encoding the `Domain.Command.Outcomes` pattern (`Order.Place.Placed`) directly in Python's class structure. Class-creation enforcement: `Command` subclasses must be nested in a `Namespace`, `DomainEvent` subclasses must be nested in a `Namespace` or `Command`. Existing framework events (`Halted`, `Interrupted`, `Resumed`, `HandlerRaised`, `Cancelled`, `MaxRoundsExceeded`) gain `SystemEvent` as a parent — backwards-compatible since they still inherit `Event` transitively. See `examples/order.py`.
- `Command.Outcomes` — auto-generated union of a command's nested `DomainEvent` classes. Used for `isinstance` checks, introspection (`typing.get_args(Command.Outcomes)`), and as the fallback runtime contract for handlers subscribed to a command. Users may declare `Outcomes` explicitly for `mypy` visibility; the framework validates drift against the nested events at class creation.
- Inline command handlers — a `handle` method defined directly on a `Command` class auto-registers as that command's handler when the class is passed to `EventGraph` or via `EventGraph.from_namespaces(*domains, handlers=...)`. `self` is the command event. Existing `@on(...)` handlers still work and compose in the same graph.
- Strict return-type enforcement — at dispatch, the framework validates handler returns against (a) the declared return annotation, or (b) the subscribed command's `Outcomes` when no annotation is present. Violations raise `TypeError` at the handler's dispatch. Unannotated non-`Command`-subscribing handlers keep the legacy shape-only check.
- Declarative domain reducers — `Reducer` / `ScalarReducer` instances declared as class attributes inside a `Namespace` are auto-named (from the attribute), auto-scoped (only that domain's events contribute), and auto-discovered by `EventGraph` via any handler subscribed to the domain's events. Explicit `reducers=[...]` kwarg still works for graph-wide reducers. Child domains inherit parent reducers via MRO.
- `Invariant` marker base class — subclass to declare a typed invariant (e.g. `class CustomerNotBanned(Invariant)`). The subclass identity drives matching; zero-arg instantiable.
- `invariants=` parameter on `@on()` — dict mapping typed `Invariant` subclasses to sync predicates (`invariants={CustomerNotBanned: lambda log: not log.has(CustomerBanned)}`). Evaluated in two phases per matching event: **pre-check** (before the handler runs, against the current log) and **post-check** (after the handler returns, against `log + emitted events`). Pre-check failure skips the handler; post-check failure drops the handler's emitted events and commits `InvariantViolated` in their place with `would_emit: tuple[Event, ...]` carrying the rolled-back events. Pin a reaction with `@on(InvariantViolated, invariant=CustomerNotBanned)` — pinned reactors fire for both phases without distinguishing. Multiple invariants short-circuit on first failure; async predicates are rejected at decoration; predicate exceptions propagate. Compile-time drift check raises `TypeError` when a pinned `invariant=` matcher references a class no handler declares ("would never fire"). Predicates must be pure functions of `log` — the same predicate runs in both phases. See `examples/order.py`.
- `EventGraph.namespaces()` — returns a `NamespaceModel`: a code-derived snapshot of the graph's structure with two lenses (`view="structure"` — domains → commands → outcomes taxonomy; `view="choreography"` — full event flow with handlers, policies, edges, seeds). Renderers: `text()`, `mermaid()`, `json()` / `to_dict()`. Nested frozen dataclasses (`NamespaceModel.Namespace`, `Command`, `CommandHandler`, `Policy`, `Edge`, `Invariant`) replace the prior `Catalog` / `AggregateEntry` / `CommandEntry` TypedDicts.
- `NamespaceModel.invariants` — first-class node for every declared invariant, with `cls`, `commands` (owning commands), `declared_by` (handler names), and `reactors` (pinned `@on(InvariantViolated, invariant=…)` handler names). Surfaced in `graph.namespaces().text()` as an `Invariants:` section and in `graph.namespaces().mermaid()` as a diamond gate node styled `:::inv` inside the owning domain's subgraph. When an invariant has a pinned reactor, the reactor's output edge leaves the Invariant diamond directly — one clean chain `Command -.->|invariant| Invariant -.->|reactor| Outcome` instead of a disconnected gate and a separate `InvariantViolated` stadium node. The `InvariantViolated` node drops from the diagram entirely when every reactor is pinned (no catch-all `@on(InvariantViolated)`). Ownership-gap arrows are suppressed for outcomes already reached via an invariant chain.
- `EventGraph.from_namespaces(*domains, handlers=None, **kwargs)` — classmethod factory that walks domains' namespaces, auto-registers every command with an inline `handle`, and appends any extra external handlers.
- `@on()` field matchers now accept `str` values for equality match alongside `type` values for `isinstance` match.
- `@on` is now polymorphic: `@on` (bare) and `@on(kwargs=...)` infer the event type from the handler's first parameter annotation, removing the common duplication between decorator argument and annotation. `@on(Type, ...)` remains the explicit form for multi-event subscription or any case where you prefer not to rely on inference. Errors at decoration if the annotation is missing or not a single `Event` subclass, and the error points to the explicit form.
- New examples: `expense_approval.py` (human-in-the-loop approval with Interrupted/resume), `conversation.py` (tool-calling agent with content moderation and AG-UI frontend tools end-to-end). `examples/order.py` grows a `ScalarReducer` attribute + pinned `@on(InvariantViolated, invariant=…)` reaction to illustrate those idioms in the canonical example.

### Changed
- `Reducer` and `ScalarReducer` fields beyond `name` are now keyword-only. Positional calls `Reducer(name, event_type, fn)` break; switch to `Reducer(name="x", event_type=..., fn=...)`. Calls already using kwargs are unaffected. See `docs/migrating.md`.
- Bare `Event` subclassing now raises `TypeError` — use `DomainEvent`, `IntegrationEvent`, `Command`, or `SystemEvent`. See `docs/migrating.md`.
- Handler return types are enforced at dispatch — mismatches raise `TypeError`. Unannotated non-`Command` handlers keep the legacy shape-only check. See `docs/migrating.md`.
- `Auditable` and `MessageEvent` are plain mixins — compose with an event branch (e.g. `IntegrationEvent, Auditable`). See `docs/migrating.md`.
- `SystemPromptSet` is now an `IntegrationEvent` (was `SystemEvent`). `SystemEvent` is reserved for framework-emitted facts; system prompts are user-seeded input. `@on(SystemEvent)` catch-alls and `isinstance(evt, SystemEvent)` branches that treated system prompts as framework signals need updating. See `docs/migrating.md`.

### Removed
- `EventGraph.catalog()` / `EventGraph.describe()` and the `Catalog` / `AggregateEntry` / `CommandEntry` TypedDicts — superseded by `EventGraph.namespaces()` + `NamespaceModel`. See `docs/migrating.md`.
- `examples/react_agent.py` — redundant with `examples/conversation.py` (same send/classify/LLM+tools shape, now reshaped as a `Conversation` namespace).
- `examples/reflection_loop.py` — multi-subscription loop pattern covered by `examples/conversation.py`.
- `examples/human_in_the_loop.py` — HITL pattern subsumed by `examples/expense_approval.py`.
- `examples/agui_frontend_tools.py` — LLM-initiated AG-UI frontend-tool flow folded into `examples/conversation.py`.
- `examples/agui_confirm_dialog.py` — handler-initiated `FrontendToolCallRequested` snippet moved to `docs/agui.md`.

## [0.4.0] - 2026-04-20

### Added
- `raises=` parameter on `@on()` — declare exceptions the framework should catch from a handler. Caught exceptions are surfaced as the new built-in `HandlerRaised` event carrying the raising handler's name (`handler`), the event being processed (`source_event`), and the raw exception (`exception`). Subscribe with `@on(HandlerRaised, exception=MyError)` to react (retry, back off, halt) without try/except boilerplate. Compile-time validation fails if a declared exception has no matching catcher; catchers that add a non-`exception` field matcher (e.g. `source_event=SomeType`) are conservatively not counted toward coverage and must be paired with a broader catcher. Framework-level errors (e.g. calling `invoke()` on an async handler from within a running event loop) are raised outside the `raises=` catch boundary and cannot be swallowed by a broad `raises=Exception`. `exception=` field matchers reject non-`Exception` `BaseException` subclasses (symmetric with `raises=`). See `examples/error_recovery.py`.
- AG-UI frontend tools — `useFrontendTool` (CopilotKit v2) is now idiomatic against an `EventGraph`. The adapter streams `AIMessageChunk.tool_call_chunks` as `ToolCallStart`/`ToolCallArgs`/`ToolCallEnd` (new `LLMToolCallChunk` frame + adapter wiring), and a new built-in `FrontendToolCallRequested(Interrupted)` event maps to the same triple for handler-initiated flows — tool calls become "HITL with typed fields," mirroring `ApprovalRequested(Interrupted)`. Two new helpers in `langgraph_events.agui`: `build_langchain_tools(input_data.tools)` converts AG-UI tool defs to OpenAI-format bindings for `llm.bind_tools(...)`; `detect_new_tool_results(input_data, checkpoint_state)` returns inbound tool messages not yet in the checkpoint so `resume_factory` can return a `MessageEvent` and continue the graph. See `examples/agui_frontend_tools.py` (LLM-initiated) and `examples/agui_confirm_dialog.py` (handler-initiated).

### Changed
- AG-UI frontend-tool plumbing now raises `ValueError` on contract violations instead of silently coercing missing fields. Triggers: `FrontendToolCallRequested(name="")` (or whitespace), an LLM `tool_call_chunk` lacking `index`, the first chunk of a streaming call lacking `id` or `name`, and an inbound `role: "tool"` message lacking `tool_call_id`. The streaming-path errors propagate through the existing `AGUIAdapter.stream()` top-level handler and surface as a `RUN_ERROR` event with the diagnostic message; conformant CopilotKit clients and LangChain chat models are unaffected.

## [0.3.0] - 2026-04-13

### Added
- `EventGraph.apre_seed()` — async counterpart to `pre_seed()`

## [0.2.1] - 2026-04-12

### Fixed
- Init reducer state from checkpoint in `_astream_v2` and `make_seed_node`

## [0.2.0] - 2026-04-06

### Added
- Field-level dispatch for `@on` decorator
- AG-UI protocol adapter with `langgraph-events[agui]` optional dependency
- Custom event emit helpers (`emit_custom`, `aemit_custom`, `emit_state_snapshot`, `aemit_state_snapshot`)
- First-class `StateSnapshotFrame` for state snapshot streaming
- `SKIP` sentinel for scalar reducer no-op returns
- Graceful `Halted` subtypes (`MaxRoundsExceeded`, `Cancelled`) and `OrphanedEventWarning`
- LLM token streaming (`LLMToken`, `LLMStreamEnd` frames)
- MkDocs documentation site on GitHub Pages

### Changed
- Restructured docs for better DX (split concepts, grouped API reference)
- `Interrupted` is now a bare marker class — subclass with typed fields

### Fixed
- AG-UI adapter message deduplication and ID reconciliation
- `connect()` yielding no events for new threads
- Resume interrupt detection for interrupts created during resume

## [0.1.0] - 2026-02-20

### Added
- Core `Event`, `EventGraph`, and `@on` decorator
- `Reducer` and `ScalarReducer` for custom state channels
- `EventLog` with query methods (`first`, `count`, `after`, `before`, `select`)
- Multi-subscription `@on(A, B)` and `Scatter` for fan-out
- `Auditable` and `MessageEvent` base events
- `SystemPromptSet` event for system prompts
- Config and store injection for handlers
- `Interrupted` / `Resumed` events for human-in-the-loop
- Mermaid diagram generation
- BDD-style test suite with pytest-describe
- CI workflow (lint, typecheck, test)

[Unreleased]: https://github.com/cadance-io/langgraph-events/compare/v0.28.0...HEAD
[0.28.0]: https://github.com/cadance-io/langgraph-events/compare/v0.27.0...v0.28.0
[0.27.0]: https://github.com/cadance-io/langgraph-events/compare/v0.26.0...v0.27.0
[0.26.0]: https://github.com/cadance-io/langgraph-events/compare/v0.25.1...v0.26.0
[0.25.1]: https://github.com/cadance-io/langgraph-events/compare/v0.25.0...v0.25.1
[0.25.0]: https://github.com/cadance-io/langgraph-events/compare/v0.24.0...v0.25.0
[0.24.0]: https://github.com/cadance-io/langgraph-events/compare/v0.23.1...v0.24.0
[0.23.1]: https://github.com/cadance-io/langgraph-events/compare/v0.23.0...v0.23.1
[0.23.0]: https://github.com/cadance-io/langgraph-events/compare/v0.22.0...v0.23.0
[0.22.0]: https://github.com/cadance-io/langgraph-events/compare/v0.21.0...v0.22.0
[0.21.0]: https://github.com/cadance-io/langgraph-events/compare/v0.20.0...v0.21.0
[0.20.0]: https://github.com/cadance-io/langgraph-events/compare/v0.19.0...v0.20.0
[0.19.0]: https://github.com/cadance-io/langgraph-events/compare/v0.18.0...v0.19.0
[0.18.0]: https://github.com/cadance-io/langgraph-events/compare/v0.17.0...v0.18.0
[0.17.0]: https://github.com/cadance-io/langgraph-events/compare/v0.16.0...v0.17.0
[0.16.0]: https://github.com/cadance-io/langgraph-events/compare/v0.15.0...v0.16.0
[0.15.0]: https://github.com/cadance-io/langgraph-events/compare/v0.14.0...v0.15.0
[0.14.0]: https://github.com/cadance-io/langgraph-events/compare/v0.13.0...v0.14.0
[0.13.0]: https://github.com/cadance-io/langgraph-events/compare/v0.12.0...v0.13.0
[0.12.0]: https://github.com/cadance-io/langgraph-events/compare/v0.11.0...v0.12.0
[0.11.0]: https://github.com/cadance-io/langgraph-events/compare/v0.10.0...v0.11.0
[0.10.0]: https://github.com/cadance-io/langgraph-events/compare/v0.9.0...v0.10.0
[0.9.0]: https://github.com/cadance-io/langgraph-events/compare/v0.8.0...v0.9.0
[0.8.0]: https://github.com/cadance-io/langgraph-events/compare/v0.7.0...v0.8.0
[0.7.0]: https://github.com/cadance-io/langgraph-events/compare/v0.6.2...v0.7.0
[0.6.2]: https://github.com/cadance-io/langgraph-events/compare/v0.6.1...v0.6.2
[0.6.1]: https://github.com/cadance-io/langgraph-events/compare/v0.6.0...v0.6.1
[0.6.0]: https://github.com/cadance-io/langgraph-events/compare/v0.5.2...v0.6.0
[0.5.2]: https://github.com/cadance-io/langgraph-events/compare/v0.5.1...v0.5.2
[0.5.1]: https://github.com/cadance-io/langgraph-events/compare/v0.5.0...v0.5.1
[0.5.0]: https://github.com/cadance-io/langgraph-events/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/cadance-io/langgraph-events/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/cadance-io/langgraph-events/compare/v0.2.1...v0.3.0
[0.2.1]: https://github.com/cadance-io/langgraph-events/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/cadance-io/langgraph-events/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/cadance-io/langgraph-events/releases/tag/v0.1.0
