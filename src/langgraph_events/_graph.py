"""EventGraph — the main entry point for building event-driven graphs."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, NamedTuple, cast

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.types import Command as LGCommand

from langgraph_events._custom_event import STATE_SNAPSHOT_EVENT_NAME
from langgraph_events._event import (
    Abandoned,
    Command,
    Event,
    Interrupted,
    Namespace,
    Unresumable,
)
from langgraph_events._event_log import EventLog
from langgraph_events._internal import (
    pad_causes,
)
from langgraph_events._pause import _with_pause_budget
from langgraph_events._rewrite import (
    _INTERRUPT_CHANNEL,
)
from langgraph_events._services import parse_services
from langgraph_events._store import EventStore
from langgraph_events._topology import EventTopology
from langgraph_events._warn import warn_user

if TYPE_CHECKING:
    from collections.abc import (
        AsyncIterator,
        Callable,
        Iterator,
        Mapping,
        Sequence,
    )

    from langchain_core.runnables import RunnableConfig
    from langgraph.graph.state import CompiledStateGraph
    from langgraph.store.base import BaseStore
    from langgraph.types import Checkpointer, Interrupt, StateSnapshot

    from langgraph_events._namespace import NamespaceModel
    from langgraph_events._reducer import BaseReducer
    from langgraph_events._reflection import Reflection
    from langgraph_events._services import Services
    from langgraph_events._types import PauseSignal

    _GraphInput = dict[str, Any] | LGCommand


class UnresumableError(RuntimeError):
    """Raised by ``resume()`` when the thread is not awaiting input.

    Default behavior of ``EventGraph(on_unresumable="raise")``: the paused
    handler was renamed/removed, the thread already completed, or it was
    resumed twice — in every case the resume would have been a silent no-op.
    Declare ``@on(previously=...)`` to recover a renamed handler, or set
    ``on_unresumable="halt"``/``"warn"`` to handle it non-fatally.
    """


class GraphState(NamedTuple):
    """Event-focused snapshot of a checkpointed thread."""

    events: EventLog
    is_interrupted: bool
    interrupted: Interrupted | None


class StreamFrame(NamedTuple):
    """A yielded frame from ``stream_events()`` when ``include_reducers`` is enabled.

    Contains the event and a snapshot of all requested reducer values at
    the point the event was produced.
    """

    event: Event
    reducers: dict[str, list[Any]]
    changed_reducers: frozenset[str] | None = None


class LLMToken(NamedTuple):
    """A text delta from an LLM invocation during handler execution."""

    run_id: str
    content: str


class LLMStreamEnd(NamedTuple):
    """Signals an LLM stream completed."""

    run_id: str
    message_id: str | None  # LangChain AIMessage.id for dedup


class LLMToolCallChunk(NamedTuple):
    """A tool-call chunk from an LLM stream.

    First chunk for a given ``call_index`` carries ``tool_call_id`` and
    ``name``; subsequent chunks may have those as empty strings and only
    carry ``args_delta`` (a partial JSON string). ``call_index`` is
    LangChain's ``tool_call_chunks[i].index`` — it groups chunks that
    belong to the same call when the LLM emits multiple in parallel.
    """

    run_id: str
    call_index: int
    tool_call_id: str
    name: str
    args_delta: str


class CustomEventFrame(NamedTuple):
    """A custom event frame emitted from LangGraph's v2 stream API."""

    name: str
    data: Any


class InterruptFrame(NamedTuple):
    """An interrupt that the run left pending.

    The async stream yields one frame for each pending interrupt, after the
    run ends, in checkpoint task order. Only the stream with
    ``include_llm_tokens`` or ``include_custom_events`` yields it. The other
    stream paths keep their item types, so a consumer of those paths gets no
    new item type.

    ``value`` is the payload of the interrupt. For a handler that returns an
    :class:`Interrupted` event, the value is that event.
    """

    value: Any


class StateSnapshotFrame(NamedTuple):
    """A typed state snapshot custom frame emitted from v2 stream events."""

    data: dict[str, Any]


StreamItem = (
    Event
    | StreamFrame
    | LLMToken
    | LLMStreamEnd
    | LLMToolCallChunk
    | CustomEventFrame
    | StateSnapshotFrame
    | InterruptFrame
)


def _snapshot_interrupts(snapshot: StateSnapshot) -> list[Interrupt]:
    """Every interrupt that *snapshot* holds pending, in checkpoint task order.

    Reads ``StateSnapshot.tasks[].interrupts``. A task with no interrupt
    holds an empty tuple.
    """
    return [i for task in snapshot.tasks for i in task.interrupts]


def _coerce_snapshot_data(data: object) -> dict[str, Any]:
    if isinstance(data, dict):
        return cast("dict[str, Any]", data)
    return {}


def _validate_on_unresumable(value: str) -> None:
    """Reject an unknown ``on_unresumable`` policy at construction.

    The value set intentionally diverges from ``AGUIAdapter(on_unmapped=...)``:
    the default is ``"raise"`` (not ``"warn"``) because a non-resumable resume
    is a real bug worth failing on, and the third option is ``"halt"`` (emit a
    terminal event) rather than ``"ignore"`` (silently drop) — there is no
    silent option here on purpose.
    """
    if value not in ("raise", "halt", "warn"):
        raise ValueError(
            f"on_unresumable must be 'raise', 'halt', or 'warn', got {value!r}"
        )


class EventGraph:
    """Build and run an event-driven graph from ``@on`` handlers.

    Topology is auto-derived from handler subscriptions.  Internally builds
    a LangGraph ``StateGraph`` with a hub-and-spoke reactive loop.

    Accepts a single seed event or a list of seed events::

        graph = EventGraph([classify, route, review])

        # Single seed event
        log = graph.invoke(DocumentReceived(doc_id="1", content="..."))

        # Multiple seed events (e.g. system prompt + user message)
        log = graph.invoke([
            SystemPromptSet.from_str("You are helpful"),
            UserMessageReceived(message=HumanMessage(content="Hi")),
        ])

    ``EventGraph(handlers, ...)`` is ``EventGraph.of(EventTopology(handlers,
    reducers=...), ...)``. See :class:`EventTopology` and :class:`EventStore`.
    """

    def __init__(
        self,
        handlers: list[Callable[..., Any]],
        *,
        max_rounds: int = 100,
        reducers: list[BaseReducer] | None = None,
        services: Sequence[Any] | Mapping[str, Any] | None = None,
        checkpointer: Checkpointer = None,
        store: BaseStore | None = None,
        recursion_limit: int | None = None,
        on_unresumable: Literal["raise", "halt", "warn"] = "raise",
    ) -> None:
        if not handlers:
            raise ValueError("EventGraph requires at least one handler")
        _validate_on_unresumable(on_unresumable)
        topology = EventTopology.__new__(EventTopology)
        expanded = topology._hold_reducers(handlers, reducers)
        # A bad services= is refused after the reducer checks and before
        # any handler check.
        parsed = parse_services(services)
        topology._hold_handlers(expanded)
        self._bind(
            topology,
            max_rounds=max_rounds,
            services=parsed,
            checkpointer=checkpointer,
            store=store,
            recursion_limit=recursion_limit,
            on_unresumable=on_unresumable,
        )

    @classmethod
    def of(
        cls,
        topology: EventTopology,
        *,
        max_rounds: int = 100,
        services: Sequence[Any] | Mapping[str, Any] | None = None,
        checkpointer: Checkpointer = None,
        store: BaseStore | None = None,
        recursion_limit: int | None = None,
        on_unresumable: Literal["raise", "halt", "warn"] = "raise",
    ) -> EventGraph:
        """Build a graph that runs *topology* with *services*.

        Takes the same keyword arguments as ``EventGraph(...)``, except
        ``reducers``: the topology holds them. Each entry of
        :attr:`EventTopology.requirements` must match a service, else this
        raises ``TypeError``. One topology can bind several graphs.
        """
        _validate_on_unresumable(on_unresumable)
        graph = cls.__new__(cls)
        graph._bind(
            topology,
            max_rounds=max_rounds,
            services=parse_services(services),
            checkpointer=checkpointer,
            store=store,
            recursion_limit=recursion_limit,
            on_unresumable=on_unresumable,
        )
        return graph

    def _bind(
        self,
        topology: EventTopology,
        *,
        max_rounds: int,
        services: Services,
        checkpointer: Checkpointer,
        store: BaseStore | None,
        recursion_limit: int | None,
        on_unresumable: Literal["raise", "halt", "warn"],
    ) -> None:
        """Bind *services* to the handlers of *topology*: run every
        construction check that needs a service."""
        self._topology = topology
        self._max_rounds = max_rounds
        self._recursion_limit = recursion_limit
        self._on_unresumable = on_unresumable
        self._checkpointer = checkpointer
        self._base_store = store
        self._handlers = topology._bind(services)
        self._compiled_graph: CompiledStateGraph | None = None
        self._event_store = (
            EventStore._of_graph(topology, checkpointer, self._compile)
            if isinstance(checkpointer, BaseCheckpointSaver)
            else None
        )

    @property
    def topology(self) -> EventTopology:
        """The topology that this graph runs."""
        return self._topology

    @property
    def event_store(self) -> EventStore:
        """The :class:`EventStore` of the threads of this graph.

        It finds, abandons and rewrites stored threads. It settles a thread
        through this graph, with its services. Raises ``ValueError`` when
        the graph has no checkpointer.
        """
        if self._event_store is None:
            raise ValueError("event_store requires a checkpointer")
        return self._event_store

    def _require_checkpointer(self, method: str) -> BaseCheckpointSaver:
        if not isinstance(self._checkpointer, BaseCheckpointSaver):
            raise ValueError(f"{method}() requires a checkpointer")
        return self._checkpointer

    def namespaces(self) -> NamespaceModel:
        """Return the :class:`NamespaceModel` of the topology. See
        :meth:`EventTopology.namespaces`."""
        return self._topology.namespaces()

    def reflect(self, log: EventLog) -> Reflection:
        """Return a :class:`Reflection` — deterministic query surface over *log*.

        Bundles the log with this graph's namespace model and reducers so an
        agent (or code) can query facts about a run: listings, field dumps,
        static topology, reducer projections, and verdict-free evidence joins.
        """
        from langgraph_events._reflection import Reflection  # noqa: PLC0415

        return Reflection(
            log, model=self.namespaces(), reducers=self._topology.reducers
        )

    @property
    def reducer_names(self) -> frozenset[str]:
        """See :attr:`EventTopology.reducer_names`."""
        return self._topology.reducer_names

    @property
    def handler_names(self) -> frozenset[str]:
        """See :attr:`EventTopology.handler_names`."""
        return self._topology.handler_names

    @property
    def resumable_node_names(self) -> frozenset[str]:
        """See :attr:`EventTopology.resumable_node_names`."""
        return self._topology.resumable_node_names

    @property
    def compiled(self) -> CompiledStateGraph:
        """The underlying LangGraph ``CompiledStateGraph``.

        This is the bridge to full LangGraph when you need features
        beyond the EventGraph API: subgraph composition, custom
        streaming modes, direct state access, or advanced checkpointer
        workflows.

        The instance is compiled lazily on first access and cached.

        Warning: the ``causes`` channel must stay aligned with ``events``. A
        direct ``update_state`` that writes ``events`` without one ``causes``
        entry per event shifts every older cause, and no check can detect it.
        Write events through :meth:`pre_seed`, or write a ``None`` cause for
        each event in the same update.
        """
        return self._compile()

    def _compile(self) -> CompiledStateGraph:
        """Compile into a LangGraph ``CompiledStateGraph`` (internal)."""
        if self._compiled_graph is None:
            self._compiled_graph = self._topology._compile(
                [(b.meta, self._topology._handler_node(b)) for b in self._handlers],
                max_rounds=self._max_rounds,
                recursion_limit=self._recursion_limit,
                checkpointer=self._checkpointer,
                store=self._base_store,
            )
        return self._compiled_graph

    @staticmethod
    def _prepare_input(seed: Event | list[Event]) -> dict[str, Any]:
        """Build the input dict from a seed event or list of events.

        The input writes a ``None`` cause for each seed, so ``causes`` stays
        aligned with ``events`` with no guess, also on a checkpoint saved
        before causes existed.
        """
        seeds = seed if isinstance(seed, list) else [seed]
        return {"events": seeds, "causes": [None] * len(seeds)}

    def _run(
        self,
        inp: _GraphInput,
        *,
        deadline: float | None,
        pause_on: PauseSignal | None,
        **kwargs: Any,
    ) -> EventLog:
        kwargs["config"] = _with_pause_budget(kwargs.get("config"), deadline, pause_on)
        compiled = self._compile()
        result = compiled.invoke(inp, **kwargs)
        return EventLog._from_state(result["events"], result.get("causes"))

    async def _arun(
        self,
        inp: _GraphInput,
        *,
        deadline: float | None,
        pause_on: PauseSignal | None,
        **kwargs: Any,
    ) -> EventLog:
        kwargs["config"] = _with_pause_budget(kwargs.get("config"), deadline, pause_on)
        compiled = self._compile()
        result = await compiled.ainvoke(inp, **kwargs)
        return EventLog._from_state(result["events"], result.get("causes"))

    @classmethod
    def from_namespaces(
        cls,
        *domains: type[Namespace],
        handlers: list[Callable[..., Any]] | None = None,
        **kwargs: Any,
    ) -> EventGraph:
        """Build an ``EventGraph`` from domains' inline command handlers.

        Walks each domain's class namespace and registers every ``Command``
        that defines an inline handler. Commands without one are silently
        skipped — register those via the ``handlers=`` kwarg or
        ``EventGraph([...])`` directly, which errors on missing handlers.

        The ``handlers=`` kwarg is appended as-is — useful for reaction
        handlers subscribed to ``DomainEvent``s, ``HandlerRaised``,
        ``InvariantViolated``, etc.

        Example::

            graph = EventGraph.from_namespaces(Order, Customer,
                                            handlers=[react])
        """
        collected: list[Any] = []
        for dom in domains:
            if not (isinstance(dom, type) and issubclass(dom, Namespace)):
                raise TypeError(
                    f"from_namespaces expects Namespace subclasses, got {dom!r}"
                )
            for attr in dom.__dict__.values():
                if (
                    isinstance(attr, type)
                    and issubclass(attr, Command)
                    and getattr(attr, "__command_handler__", None) is not None
                ):
                    collected.append(attr)
        if handlers:
            collected.extend(handlers)

        # Auto-wire migration-aware deserialization: the user only writes
        # @migrate_from on the class. We have both the domains and the
        # checkpointer here, so scope a NamespaceAwareSerde to these
        # namespaces. A user-supplied NamespaceAwareSerde (possibly
        # carrying hand-authored migrations=) is left untouched — that is
        # the opt-out. The serde still owns coverage; we own construction.
        graph = cls(collected, **kwargs)

        # Wired after construction, not before: the events that live outside
        # every namespace — module-level IntegrationEvents, framework
        # SystemEvents — are only knowable once the graph has parsed its
        # handlers' subscriptions and return types. Without them those
        # identities resolve by import and are shared between engine
        # lifetimes of one module. ``compiled`` is lazy, so the serde is in
        # place well before anything reads it.
        checkpointer = kwargs.get("checkpointer")
        if checkpointer is not None:
            from langgraph_events.serde import NamespaceAwareSerde  # noqa: PLC0415

            serde = getattr(checkpointer, "serde", None)
            if serde is not None and not isinstance(serde, NamespaceAwareSerde):
                # Not ``domains``: a namespace can reach the graph through
                # ``handlers=`` alone, and its events would then keep
                # resolving by import — the bleed #155 exists to close.
                # The topology holds the full set.
                checkpointer.serde = graph.topology.serde()

        return graph

    def invoke(
        self,
        seed: Event | list[Event],
        *,
        deadline: float | None = None,
        pause_on: PauseSignal | None = None,
        **kwargs: Any,
    ) -> EventLog:
        """Run the graph synchronously with one or more seed events.

        Args:
            seed: A single event or list of events to start the graph.
            deadline: A ``time.monotonic()`` instant. At the first round
                boundary past it, the router emits ``RunPausedOnDeadline``
                and the run ends cleanly. ``None`` (default) disables.
            pause_on: An object with ``is_set()``, for example an
                ``asyncio.Event``. At the first round boundary where
                ``is_set()`` returns ``True``, the router emits
                ``RunPausedOnSignal`` and the run ends cleanly. ``None``
                (default) disables.

        Returns an ``EventLog`` containing all events produced during the run.

        Raises:
            TypeError: *pause_on* is a class, or has no callable ``is_set``.
        """
        return self._run(
            self._prepare_input(seed), deadline=deadline, pause_on=pause_on, **kwargs
        )

    async def ainvoke(
        self,
        seed: Event | list[Event],
        *,
        deadline: float | None = None,
        pause_on: PauseSignal | None = None,
        **kwargs: Any,
    ) -> EventLog:
        """Run the graph asynchronously with one or more seed events.

        Takes the same arguments as :meth:`invoke`.
        """
        return await self._arun(
            self._prepare_input(seed), deadline=deadline, pause_on=pause_on, **kwargs
        )

    def pre_seed(self, config: RunnableConfig, values: dict[str, Any]) -> None:
        """Inject external state into reducer channels before the first run.

        Use this to hydrate reducers from an external source (e.g. a database
        migration or test fixture) when modelling the data as seed events isn't
        practical.  Call it once before ``invoke``/``ainvoke``::

            graph.pre_seed(config, {"my_reducer": existing_value})
            graph.invoke(StartEvent(), config=config)

        Each event in ``values["events"]`` gets a ``None`` cause.

        Requires a checkpointer.
        """
        self._require_checkpointer("pre_seed")
        compiled = self._compile()
        compiled.update_state(config, pad_causes(values, None), as_node="__seed__")

    async def apre_seed(self, config: RunnableConfig, values: dict[str, Any]) -> None:
        """Async version of :meth:`pre_seed`."""
        self._require_checkpointer("apre_seed")
        compiled = self._compile()
        await compiled.aupdate_state(
            config, pad_causes(values, None), as_node="__seed__"
        )

    @staticmethod
    def _awaits_resume(snapshot: StateSnapshot) -> bool:
        """Whether *snapshot* has work to resume into.

        A resume is awaited when the checkpoint's ``next`` is set, or when a
        task holds a pending interrupt. With neither, nothing is scheduled.
        Then ``resume()`` does nothing: the paused handler is gone, the thread
        finished, or the thread was resumed twice.

        ``next`` is read because a caller can ``pre_seed`` before it resumes,
        for example ``AGUIAdapter`` commits a ``FrontendStateMutated``. That
        write hides the pending interrupt, but ``next`` stays set. A Phase-1
        ``@on(previously=...)`` alias keeps the node live, so a declared
        rename still reports ``True``.

        The interrupts are read because LangGraph before 1.2.13 drops from
        ``next`` a task that holds a saved resume write. A handler that pauses
        a second time in one run is such a task. See :meth:`_is_interrupted`.
        """
        return bool(snapshot.next) or EventGraph._is_interrupted(snapshot)

    def _resume_is_pending(self, kwargs: dict[str, Any]) -> bool:
        """Whether the thread of ``kwargs["config"]`` has work to resume into.

        See :meth:`_awaits_resume`. If no ``config`` was passed, assume
        pending and let the normal path surface any error.
        """
        config = kwargs.get("config")
        if config is None:
            return True
        return self._awaits_resume(self._compile().get_state(config))

    async def _aresume_snapshot(
        self, kwargs: dict[str, Any], snapshot: StateSnapshot | None
    ) -> StateSnapshot | None:
        """The checkpoint of ``kwargs["config"]`` for an async resume.

        Returns *snapshot* when the caller already loaded it. Otherwise loads
        it with ``aget_state``, so an async resume never drives an async-only
        checkpointer synchronously from the running event loop. Returns
        ``None`` if no ``config`` was passed.
        """
        if snapshot is not None:
            return snapshot
        config = kwargs.get("config")
        if config is None:
            return None
        return await self._compile().aget_state(config)

    async def _aresume_is_pending(
        self, kwargs: dict[str, Any], snapshot: StateSnapshot | None = None
    ) -> bool:
        """Async sibling of :meth:`_resume_is_pending`.

        Pass *snapshot* when the checkpoint is already loaded, so this check
        does not load it again. See :meth:`_aresume_snapshot`.
        """
        snapshot = await self._aresume_snapshot(kwargs, snapshot)
        return snapshot is None or self._awaits_resume(snapshot)

    def _unresumable_message(
        self, log: EventLog | None = None, config: RunnableConfig | None = None
    ) -> str:
        """Diagnostic for a ``resume()`` that would be a no-op.

        Keys the abandoned diagnosis on the *last* event, not
        ``log.latest(Halted)`` (a whole-log search): a thread reused
        after :meth:`abandon` must fall back to the generic message once
        ``Abandoned`` is no longer latest.
        """
        if log and isinstance(log[-1], Abandoned):
            abandoned = log[-1]
            thread_id = (config or {}).get("configurable", {}).get("thread_id")
            detail = ""
            if abandoned.reason:
                detail += f" reason={abandoned.reason!r}."
            if abandoned.discarded:
                detail += f" discarded={abandoned.discarded!r}."
            return (
                f"resume() called on thread {thread_id!r}, which was "
                f"abandoned via abandon()/aabandon().{detail} The thread was "
                "deliberately settled without an answer and is terminal; it "
                "cannot be resumed."
            )
        return (
            "resume() called on a thread that is not awaiting input. The paused "
            "handler may have been renamed/removed, or the thread already "
            "completed. Declare @on(previously=...) to recover a renamed "
            "handler, or set EventGraph(on_unresumable='halt'|'warn')."
        )

    def _unresumable_short_circuits(
        self, log: EventLog | None = None, config: RunnableConfig | None = None
    ) -> bool:
        """Apply ``on_unresumable``'s ``raise``/``warn`` arm. ``False``
        means append a terminal event (``halt``). ``True`` means return
        the log unchanged (``warn``). ``raise`` raises.

        Caller supplies the state read. The async path must ``await
        aget_state``: an async-only checkpointer rejects a sync read
        from the running loop.
        """
        if self._on_unresumable == "raise":
            raise UnresumableError(self._unresumable_message(log, config))
        if self._on_unresumable == "warn":
            warn_user(self._unresumable_message(log, config))
            return True
        return False

    def _apply_unresumable_policy(
        self, value: Event, kwargs: dict[str, Any]
    ) -> EventLog:
        """Sync ``on_unresumable`` for a resume that would be a no-op.

        Runs only when :meth:`_resume_is_pending` is ``False``, so no
        completed sibling write can be lost on this path.
        """
        config = kwargs["config"]
        log = self.get_state(config).events
        if self._unresumable_short_circuits(log, config):
            return log
        return self.event_store._settle(config, self._unresumable_event(value))

    async def _aapply_unresumable_policy(
        self, value: Event, kwargs: dict[str, Any]
    ) -> EventLog:
        """Async sibling of :meth:`_apply_unresumable_policy`."""
        config = kwargs["config"]
        log = (await self.aget_state(config)).events
        if self._unresumable_short_circuits(log, config):
            return log
        return await self.event_store._asettle(config, self._unresumable_event(value))

    @staticmethod
    def _unresumable_event(value: Event) -> Unresumable:
        return Unresumable(resume_value=type(value).__name__)

    @staticmethod
    def _require_resume_event(value: object) -> None:
        """Raise ``TypeError`` if *value* is not an ``Event``.

        The check runs before the graph runs. LangGraph saves a resume value
        in the checkpoint before the paused handler reads it. A rerun then
        reads the saved value again, so a bad value strands the thread.
        """
        if not isinstance(value, Event):
            got = type(value).__name__
            raise TypeError(f"resume() requires an Event instance, got {got}")

    def resume(
        self,
        value: Event,
        *,
        deadline: float | None = None,
        pause_on: PauseSignal | None = None,
        **kwargs: Any,
    ) -> EventLog:
        """Resume an interrupted graph with a domain event.

        The event is auto-dispatched (handlers subscribed to its type fire),
        then a ``Resumed`` event is created alongside it. If the thread is not
        awaiting input, the ``on_unresumable`` policy applies (default raise).
        The resume round always dispatches. A pause can start at the next
        round.

        Args:
            value: The domain event to resume with.
            deadline: A ``time.monotonic()`` instant. At the first round
                boundary past it, the router emits ``RunPausedOnDeadline``
                and the run ends cleanly. ``None`` (default) disables.
            pause_on: An object with ``is_set()``, for example an
                ``asyncio.Event``. At the first round boundary where
                ``is_set()`` returns ``True``, the router emits
                ``RunPausedOnSignal`` and the run ends cleanly. ``None``
                (default) disables.

        Raises:
            TypeError: *pause_on* is a class, or has no callable ``is_set``.
        """
        self._require_checkpointer("resume")
        self._require_resume_event(value)
        if not self._resume_is_pending(kwargs):
            return self._apply_unresumable_policy(value, kwargs)
        return self._run(
            LGCommand(resume=value), deadline=deadline, pause_on=pause_on, **kwargs
        )

    async def aresume(
        self,
        value: Event,
        *,
        deadline: float | None = None,
        pause_on: PauseSignal | None = None,
        **kwargs: Any,
    ) -> EventLog:
        """Async version of resume().

        The event is auto-dispatched (handlers subscribed to its type fire),
        then a ``Resumed`` event is created alongside it. If the thread is not
        awaiting input, the ``on_unresumable`` policy applies (default raise).
        Takes the same arguments as :meth:`resume`.
        """
        self._require_checkpointer("aresume")
        self._require_resume_event(value)
        if not await self._aresume_is_pending(kwargs):
            return await self._aapply_unresumable_policy(value, kwargs)
        return await self._arun(
            LGCommand(resume=value), deadline=deadline, pause_on=pause_on, **kwargs
        )

    @staticmethod
    def _is_interrupted(snapshot: StateSnapshot) -> bool:
        """Whether *snapshot* is currently paused at a real interrupt.

        Reads ``snapshot.tasks[*].interrupts``, not ``snapshot.next``. A
        cancelled or crashed graph also leaves ``next`` set. Before LangGraph
        1.2.13, ``next`` also drops a task that holds a ``__resume__`` write,
        for example a handler that pauses a second time in one run.

        Topology-dependent: ``snapshot.tasks`` only reports a task the
        graph that produced *snapshot* can still schedule, so a thread
        paused on a handler this graph no longer registers looks
        uninterrupted here. Used only by :meth:`_graph_state`, reached
        through a caller's own working graph. Everywhere else
        (discovery, ``abandon()``) reads the checkpoint's raw pending
        writes instead — see :meth:`EventStore._pending_interrupt_writes`.
        """
        return bool(_snapshot_interrupts(snapshot))

    @staticmethod
    def _pending_interrupt_events(snapshot: StateSnapshot) -> list[Event]:
        """Event instance(s) behind every pending interrupt, deduped by
        type name, in discovery order. Empty if none.

        Scans every task, not just ``tasks[0]``: a fanned-out dispatch
        can pause more than one. Skips a non-``Event`` payload, e.g. a
        raw ``langgraph.types.interrupt("...")`` call.

        Topology-dependent, same caveat as :meth:`_is_interrupted`: used
        only by :meth:`_graph_state`'s first-pause fallback.
        """
        seen: set[str] = set()
        values: list[Event] = []
        for i in _snapshot_interrupts(snapshot):
            value = i.value
            if isinstance(value, Event) and type(value).__name__ not in seen:
                seen.add(type(value).__name__)
                values.append(value)
        return values

    def _graph_state(self, snapshot: StateSnapshot) -> GraphState:
        """Build a :class:`GraphState` from a checkpoint snapshot.

        Shared by the sync :meth:`get_state` and async :meth:`aget_state` so
        the snapshot-to-state logic stays in one place across both paths.

        Deliberately a pure function of *snapshot*: no checkpointer read
        happens here, which keeps ``is_interrupted``/``interrupted``
        topology-dependent (see :meth:`_is_interrupted`). A caller
        holding a ``GraphState`` reached it through its own working
        graph, unlike discovery (:meth:`EventStore.threads_paused_on`) or
        ``abandon()``, which read the checkpoint directly.
        """
        log = EventStore._events(snapshot)
        is_interrupted = self._is_interrupted(snapshot)
        interrupted = log.latest(Interrupted) if is_interrupted else None
        if is_interrupted and interrupted is None:
            # An Interrupted joins the log only on resume. A first pause
            # has none there yet. Fall back to the snapshot's pending
            # interrupt payload.
            for value in self._pending_interrupt_events(snapshot):
                if isinstance(value, Interrupted):
                    interrupted = value
                    break
        return GraphState(
            events=log,
            is_interrupted=is_interrupted,
            interrupted=interrupted,
        )

    def get_state(self, config: RunnableConfig) -> GraphState:
        """Get event-level state of a checkpointed thread."""
        self._require_checkpointer("get_state")
        return self._graph_state(self._compile().get_state(config))

    async def aget_state(self, config: RunnableConfig) -> GraphState:
        """Async version of :meth:`get_state`.

        Reads the checkpoint via ``aget_state`` so async-only checkpointers
        (e.g. ``AsyncPostgresSaver``) aren't driven synchronously from the
        running event loop.
        """
        self._require_checkpointer("aget_state")
        return self._graph_state(await self._compile().aget_state(config))

    # --- LLM token helpers ---

    @staticmethod
    def _extract_text_content(chunk: object) -> str:
        """Extract text from a LangChain chat model stream chunk."""
        content = getattr(chunk, "content", None)
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict) and item.get("type") == "text":
                    text = item.get("text")
                    if isinstance(text, str):
                        parts.append(text)
            return "".join(parts)
        return ""

    # --- High-level event streaming ---

    @staticmethod
    def _events_from_chunk(chunk: object, seen: set[int]) -> list[Event]:
        """Extract unseen events from an updates-mode stream chunk.

        A chunk marked ``cached`` replays the saved writes of a task that
        finished before a pause. The run that wrote them streamed them, and
        the checkpoint already holds them, so they are not new events.
        LangGraph also marks a node cache hit ``cached``, but ``_compile``
        sets no node cache, so here the marker means only that replay.
        """
        events: list[Event] = []
        if isinstance(chunk, dict) and not chunk.get("__metadata__", {}).get("cached"):
            for node_output in chunk.values():
                if isinstance(node_output, dict):
                    for event in node_output.get("events", []):
                        eid = id(event)
                        if eid not in seen:
                            seen.add(eid)
                            events.append(event)
        return events

    def _frames_from_values(
        self,
        state: dict[str, Any],
        prev_count: int,
        reducer_names: list[str],
    ) -> tuple[int, list[StreamFrame]]:
        """Extract new events and reducer snapshots from a values-mode state."""
        all_events: list[Event] = state.get("events", [])
        new_events = all_events[prev_count:]
        if not new_events:
            return prev_count, []
        reducers = {
            name: state.get(name, self._topology.reducers[name].empty)
            for name in reducer_names
        }
        return len(all_events), [
            StreamFrame(event=e, reducers=reducers, changed_reducers=None)
            for e in new_events
        ]

    async def _astream_v2(  # noqa: PLR0912, PLR0915
        self,
        inp: _GraphInput,
        seeds: list[Event],
        *,
        snapshot: StateSnapshot | None,
        reducer_names: list[str],
        include_llm_tokens: bool,
        include_custom_events: bool,
        **kwargs: Any,
    ) -> AsyncIterator[StreamItem]:
        """Stream events using LangGraph's v2 event API with LLM token support.

        *snapshot* is the checkpoint of ``config`` before the run, or ``None``
        when the caller did not load it.
        """
        compiled = self._compile()
        is_resume = isinstance(inp, LGCommand)

        # Initialize incremental reducer state.  When a checkpoint exists
        # (subsequent run / resume), start from the checkpointed values so
        # the shadow state matches what the compiled graph already restored.
        # Fall back to r.seed() only on the true first run.
        reducer_state: dict[str, Any] = {}
        checkpoint_values: dict[str, Any] | None = None
        if reducer_names and self._checkpointer is not None:
            config = kwargs.get("config")
            if config is not None:
                if snapshot is None:
                    snapshot = await compiled.aget_state(config)
                if snapshot.values:
                    checkpoint_values = snapshot.values

        for name in reducer_names:
            if checkpoint_values is not None:
                reducer_state[name] = checkpoint_values.get(
                    name, self._topology.reducers[name].empty
                )
            else:
                reducer_state[name] = self._topology.reducers[name].seed(seeds)

        # Merge seed contributions on top of checkpointed state.
        # - On astream_events second-run: seeds carry new events whose
        #   contributions need to layer on top of the restored checkpoint
        #   (LangGraph's seed_node will copy them into state["events"], but
        #   the in-adapter reducer_state shadow needs the contribution
        #   computed here so the StreamFrame yielded just below reflects it).
        # - On astream_resume (is_resume=True): skipped — the caller
        #   (e.g. AGUIAdapter) is responsible for pre-seeding contributions
        #   via apre_seed before calling astream_resume, because
        #   Command(resume=...) doesn't route seeds through the seed_node.
        #   Re-applying contributions here would double-count for
        #   accumulator reducers (operator.add channel reducers concatenate
        #   the contribution twice).  See
        #   `AGUIAdapter._resume_event_stream` for the canonical pattern.
        if checkpoint_values is not None and not is_resume:
            for s in seeds:
                self._topology._update_reducer_state(reducer_state, s, reducer_names)

        # Yield seed events
        for s in seeds:
            if reducer_names:
                changed = frozenset(
                    name
                    for name in reducer_names
                    if self._topology.reducers[name].has_contributions(
                        self._topology.reducers[name].collect([s])
                    )
                )
                yield StreamFrame(
                    event=s,
                    reducers=dict(reducer_state),
                    changed_reducers=changed,
                )
            else:
                yield s

        seen: set[int] = set()
        interrupts: list[Interrupt] = []

        # Runnable.astream_events fills LangGraph's default recursion_limit
        # before Pregel merges compiled.config, so the default would win.
        kwargs["config"] = {**(compiled.config or {}), **(kwargs.get("config") or {})}
        async for raw in compiled.astream_events(
            inp,
            version="v2",
            stream_mode="updates",
            **kwargs,
        ):
            raw_event = raw.get("event")

            if raw_event == "on_chat_model_stream":
                if not include_llm_tokens:
                    continue
                run_id = raw.get("run_id", "")
                chunk = raw.get("data", {}).get("chunk")
                if chunk is None or not run_id:
                    continue
                delta = self._extract_text_content(chunk)
                if delta:
                    yield LLMToken(run_id=run_id, content=delta)
                for tc_chunk in getattr(chunk, "tool_call_chunks", None) or ():
                    index = tc_chunk.get("index")
                    if index is None:
                        raise ValueError(
                            f"LLM tool_call_chunk missing 'index' "
                            f"(run_id={run_id!r}, chunk={tc_chunk!r}). "
                            "LangChain normalizes this — a missing index "
                            "indicates a non-conformant provider or a bug "
                            "upstream."
                        )
                    yield LLMToolCallChunk(
                        run_id=run_id,
                        call_index=index,
                        tool_call_id=tc_chunk.get("id") or "",
                        name=tc_chunk.get("name") or "",
                        args_delta=tc_chunk.get("args") or "",
                    )
                continue

            if raw_event == "on_chat_model_end":
                if not include_llm_tokens:
                    continue
                run_id = raw.get("run_id", "")
                if run_id:
                    output = raw.get("data", {}).get("output")
                    message_id = getattr(output, "id", None)
                    yield LLMStreamEnd(run_id=run_id, message_id=message_id)
                continue

            if raw_event == "on_custom_event":
                if not include_custom_events:
                    continue
                if raw.get("name", "") == STATE_SNAPSHOT_EVENT_NAME:
                    yield StateSnapshotFrame(
                        data=_coerce_snapshot_data(raw.get("data")),
                    )
                else:
                    yield CustomEventFrame(
                        name=raw.get("name", ""),
                        data=raw.get("data"),
                    )
                continue

            if raw_event != "on_chain_stream" or raw.get("name") != "LangGraph":
                continue

            chunk = raw.get("data", {}).get("chunk")
            if isinstance(chunk, dict):
                interrupts.extend(chunk.get(_INTERRUPT_CHANNEL, ()))
            for event in self._events_from_chunk(chunk, seen):
                if reducer_names:
                    changed = self._topology._update_reducer_state(
                        reducer_state,
                        event,
                        reducer_names,
                    )
                    yield StreamFrame(
                        event=event,
                        reducers=dict(reducer_state),
                        changed_reducers=changed,
                    )
                else:
                    yield event

        for frame in await self._apending_interrupt_frames(
            interrupts, kwargs.get("config")
        ):
            yield frame

    async def _apending_interrupt_frames(
        self, streamed: list[Interrupt], config: RunnableConfig | None
    ) -> list[InterruptFrame]:
        """One frame for each interrupt that the run left pending.

        *streamed* holds the interrupts in the order that the run streamed
        them. Handlers that pause in one step finish in any order. So for
        more than one interrupt, the checkpoint is loaded again, because it
        holds the interrupts in task order.
        """
        interrupts = streamed
        if len(streamed) > 1 and config is not None:
            after = await self._compile().aget_state(config)
            interrupts = _snapshot_interrupts(after)
        return [
            InterruptFrame(value=i.value) for i in interrupts if i.value is not None
        ]

    def _stream_sync(
        self,
        inp: _GraphInput,
        seeds: list[Event],
        reducer_names: list[str],
        *,
        deadline: float | None,
        pause_on: PauseSignal | None,
        **kwargs: Any,
    ) -> Iterator[Event | StreamFrame]:
        """Shared sync streaming core for stream_events/stream_resume."""
        kwargs["config"] = _with_pause_budget(kwargs.get("config"), deadline, pause_on)
        compiled = self._compile()
        if not reducer_names:
            yield from seeds
            seen: set[int] = set()
            for chunk in compiled.stream(inp, stream_mode="updates", **kwargs):
                yield from self._events_from_chunk(chunk, seen)
        else:
            prev_count = 0
            first = True
            for state in compiled.stream(inp, stream_mode="values", **kwargs):
                if first:
                    first = False
                    continue
                prev_count, frames = self._frames_from_values(
                    state, prev_count, reducer_names
                )
                yield from frames

    async def _astream_core(
        self,
        inp: _GraphInput,
        seeds: list[Event],
        reducer_names: list[str],
        **kwargs: Any,
    ) -> AsyncIterator[Event | StreamFrame]:
        """Shared async streaming core for astream_events/astream_resume."""
        compiled = self._compile()
        if not reducer_names:
            for s in seeds:
                yield s
            seen: set[int] = set()
            async for chunk in compiled.astream(inp, stream_mode="updates", **kwargs):
                for event in self._events_from_chunk(chunk, seen):
                    yield event
        else:
            prev_count = 0
            first = True
            async for state in compiled.astream(inp, stream_mode="values", **kwargs):
                if first:
                    first = False
                    continue
                prev_count, frames = self._frames_from_values(
                    state, prev_count, reducer_names
                )
                for frame in frames:
                    yield frame

    def stream_events(
        self,
        seed: Event | list[Event],
        *,
        include_reducers: bool | list[str] = False,
        deadline: float | None = None,
        pause_on: PauseSignal | None = None,
        **kwargs: Any,
    ) -> Iterator[Event | StreamFrame]:
        """Yield individual events as they are produced during graph execution.

        Higher-level alternative to ``compiled.stream()`` — yields ``Event``
        objects directly instead of raw LangGraph state dicts.  Seed events
        are yielded first, followed by events produced by handlers.

        Args:
            seed: A single event or list of events to start the graph.
            include_reducers: When truthy, yields ``StreamFrame`` tuples
                instead of bare events.  Pass ``True`` for all reducers or
                a list of reducer names for selective inclusion.
            deadline: A ``time.monotonic()`` instant. At the first round
                boundary past it, the router emits ``RunPausedOnDeadline``
                and the run ends cleanly. ``None`` (default) disables.
            pause_on: An object with ``is_set()``, for example an
                ``asyncio.Event``. At the first round boundary where
                ``is_set()`` returns ``True``, the router emits
                ``RunPausedOnSignal`` and the run ends cleanly. ``None``
                (default) disables.

        Raises:
            TypeError: *pause_on* is a class, or has no callable ``is_set``.
        """
        inp = self._prepare_input(seed)
        kwargs.pop("stream_mode", None)
        reducer_names = self._topology._resolve_reducer_names(include_reducers)
        yield from self._stream_sync(
            inp,
            inp["events"],
            reducer_names,
            deadline=deadline,
            pause_on=pause_on,
            **kwargs,
        )

    def stream_resume(
        self,
        value: Event,
        *,
        seeds: list[Event] | None = None,
        include_reducers: bool | list[str] = False,
        deadline: float | None = None,
        pause_on: PauseSignal | None = None,
        **kwargs: Any,
    ) -> Iterator[Event | StreamFrame]:
        """Yield events produced when resuming an interrupted graph.

        Streaming equivalent of ``resume()`` — accepts a domain ``Event``,
        yields events as they are produced.  The ``Command(resume=value)``
        stays internal, exactly like ``resume()``.

        Args:
            value: The domain event to resume with.
            seeds: Optional events to dispatch alongside the resume in the
                same step.  Their reducer contributions layer onto the
                checkpointed state *before* the resume's downstream events
                run, so resume-handler outputs win for shared keys.  Used
                by ``AGUIAdapter`` to route ``FrontendStateMutated`` through
                the dispatch chain on resume.
            include_reducers: When truthy, yields ``StreamFrame`` tuples
                instead of bare events.  Pass ``True`` for all reducers or
                a list of reducer names for selective inclusion.
            deadline: A ``time.monotonic()`` instant. At the first round
                boundary past it, the router emits ``RunPausedOnDeadline``
                and the run ends cleanly. ``None`` (default) disables.
            pause_on: An object with ``is_set()``, for example an
                ``asyncio.Event``. At the first round boundary where
                ``is_set()`` returns ``True``, the router emits
                ``RunPausedOnSignal`` and the run ends cleanly. ``None``
                (default) disables.

        Raises:
            TypeError: *pause_on* is a class, or has no callable ``is_set``.
        """
        self._require_checkpointer("stream_resume")
        self._require_resume_event(value)
        if not self._resume_is_pending(kwargs):
            log = self._apply_unresumable_policy(value, kwargs)  # raises for raise
            if self._on_unresumable == "halt":
                latest = log.latest(Unresumable)
                if latest is not None:
                    yield latest
            return
        kwargs.pop("stream_mode", None)
        reducer_names = self._topology._resolve_reducer_names(include_reducers)
        yield from self._stream_sync(
            LGCommand(resume=value),
            seeds or [],
            reducer_names,
            deadline=deadline,
            pause_on=pause_on,
            **kwargs,
        )

    async def _astream_entry(
        self,
        inp: _GraphInput,
        seeds: list[Event],
        *,
        include_reducers: bool | list[str],
        include_llm_tokens: bool,
        include_custom_events: bool,
        snapshot: StateSnapshot | None,
        deadline: float | None,
        pause_on: PauseSignal | None,
        **kwargs: Any,
    ) -> AsyncIterator[StreamItem]:
        """Shared async-stream dispatcher — picks v2 vs core based on flags.

        Only the v2 path uses *snapshot*, to seed its reducer state. The core
        path loads no checkpoint, so it has no use for *snapshot*.
        """
        kwargs.pop("stream_mode", None)
        kwargs["config"] = _with_pause_budget(kwargs.get("config"), deadline, pause_on)
        reducer_names = self._topology._resolve_reducer_names(include_reducers)
        delegate = (
            self._astream_v2(
                inp,
                seeds,
                snapshot=snapshot,
                reducer_names=reducer_names,
                include_llm_tokens=include_llm_tokens,
                include_custom_events=include_custom_events,
                **kwargs,
            )
            if include_llm_tokens or include_custom_events
            else self._astream_core(inp, seeds, reducer_names, **kwargs)
        )
        async for item in delegate:
            yield item

    async def astream_resume(
        self,
        value: Event,
        *,
        seeds: list[Event] | None = None,
        include_reducers: bool | list[str] = False,
        include_llm_tokens: bool = False,
        include_custom_events: bool = False,
        _snapshot: StateSnapshot | None = None,
        deadline: float | None = None,
        pause_on: PauseSignal | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[StreamItem]:
        """Async version of ``stream_resume()``.

        Streaming equivalent of ``aresume()`` — accepts a domain ``Event``,
        yields events as they are produced.

        Args:
            value: The domain event to resume with.
            seeds: Optional events to dispatch alongside the resume in the
                same step.  Their reducer contributions layer onto the
                checkpointed state *before* the resume's downstream events
                run, so resume-handler outputs win for shared keys.  Used
                by ``AGUIAdapter`` to route ``FrontendStateMutated`` through
                the dispatch chain on resume.
            include_reducers: When truthy, yields ``StreamFrame`` tuples
                instead of bare events.  Pass ``True`` for all reducers or
                a list of reducer names for selective inclusion.
            include_llm_tokens: When True, yields ``LLMToken`` and
                ``LLMStreamEnd`` frames for LLM token-level streaming.
            include_custom_events: When True, yields ``CustomEventFrame``
                and ``StateSnapshotFrame`` frames for ``on_custom_event`` payloads
                from LangGraph.
            _snapshot: Private to ``AGUIAdapter``. Do not pass it. It is the
                checkpoint of ``config`` that the adapter loaded for the turn.
                The pending check uses it, so it does not load the checkpoint
                again. With ``include_llm_tokens`` or ``include_custom_events``,
                the reducer seed uses it too.
            deadline: A ``time.monotonic()`` instant. At the first round
                boundary past it, the router emits ``RunPausedOnDeadline``
                and the run ends cleanly. ``None`` (default) disables.
            pause_on: An object with ``is_set()``, for example an
                ``asyncio.Event``. At the first round boundary where
                ``is_set()`` returns ``True``, the router emits
                ``RunPausedOnSignal`` and the run ends cleanly. ``None``
                (default) disables.

        Raises:
            TypeError: *pause_on* is a class, or has no callable ``is_set``.
        """
        self._require_checkpointer("astream_resume")
        self._require_resume_event(value)
        snapshot = await self._aresume_snapshot(kwargs, _snapshot)
        if not await self._aresume_is_pending(kwargs, snapshot):
            # raises for the 'raise' policy; warn/halt return a log to adapt
            log = await self._aapply_unresumable_policy(value, kwargs)
            if self._on_unresumable == "halt":
                latest = log.latest(Unresumable)
                if latest is not None:
                    yield latest
            return
        async for item in self._astream_entry(
            LGCommand(resume=value),
            seeds or [],
            include_reducers=include_reducers,
            include_llm_tokens=include_llm_tokens,
            include_custom_events=include_custom_events,
            snapshot=snapshot,
            deadline=deadline,
            pause_on=pause_on,
            **kwargs,
        ):
            yield item

    async def astream_events(
        self,
        seed: Event | list[Event],
        *,
        include_reducers: bool | list[str] = False,
        include_llm_tokens: bool = False,
        include_custom_events: bool = False,
        _snapshot: StateSnapshot | None = None,
        deadline: float | None = None,
        pause_on: PauseSignal | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[StreamItem]:
        """Async version of ``stream_events()``.

        Args:
            seed: A single event or list of events to start the graph.
            include_reducers: When truthy, yields ``StreamFrame`` tuples
                instead of bare events.  Pass ``True`` for all reducers or
                a list of reducer names for selective inclusion.
            include_llm_tokens: When True, yields ``LLMToken`` and
                ``LLMStreamEnd`` frames for LLM token-level streaming.
            include_custom_events: When True, yields ``CustomEventFrame``
                and ``StateSnapshotFrame`` frames for ``on_custom_event`` payloads
                from LangGraph.
            _snapshot: Private to ``AGUIAdapter``. Do not pass it. It is the
                checkpoint of ``config`` that the adapter loaded for the turn.
                The reducer seed uses it, so it does not load the checkpoint
                again. Only the stream with ``include_llm_tokens`` or
                ``include_custom_events`` uses it, because the other stream
                loads no checkpoint.
            deadline: A ``time.monotonic()`` instant. At the first round
                boundary past it, the router emits ``RunPausedOnDeadline``
                and the run ends cleanly. ``None`` (default) disables.
            pause_on: An object with ``is_set()``, for example an
                ``asyncio.Event``. At the first round boundary where
                ``is_set()`` returns ``True``, the router emits
                ``RunPausedOnSignal`` and the run ends cleanly. ``None``
                (default) disables.

        Raises:
            TypeError: *pause_on* is a class, or has no callable ``is_set``.
        """
        inp = self._prepare_input(seed)
        async for item in self._astream_entry(
            inp,
            inp["events"],
            include_reducers=include_reducers,
            include_llm_tokens=include_llm_tokens,
            include_custom_events=include_custom_events,
            snapshot=_snapshot,
            deadline=deadline,
            pause_on=pause_on,
            **kwargs,
        ):
            yield item
