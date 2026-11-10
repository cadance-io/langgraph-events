"""AGUIAdapter — core orchestrator mapping EventGraph streams to AG-UI events."""

from __future__ import annotations

import copy
import inspect
import logging
import uuid
from functools import partial
from typing import TYPE_CHECKING, Any, Literal, TypedDict, TypeVar, cast

from ag_ui.core import (
    BaseEvent,
    CustomEvent,
    EventType,
    RunErrorEvent,
    RunFinishedEvent,
    RunStartedEvent,
    TextMessageContentEvent,
    TextMessageEndEvent,
    TextMessageStartEvent,
    ToolCallArgsEvent,
    ToolCallEndEvent,
    ToolCallStartEvent,
)

from langgraph_events._event import Event, Interrupted
from langgraph_events._graph import _snapshot_interrupts
from langgraph_events._pause import _FRAMEWORK_KEY_PREFIX, _with_pause_budget
from langgraph_events.stream import (
    CustomEventFrame,
    InterruptFrame,
    LLMStreamEnd,
    LLMToken,
    LLMToolCallChunk,
    StateSnapshotFrame,
    StreamFrame,
    StreamItem,
)

from ._context import MapperContext
from ._errors import RunRefusedError
from ._events import FrontendStateMutated
from ._mappers import (
    FallbackMapper,
    UnmappedEventError,
    build_messages_snapshot,
    build_state_snapshot,
    default_mappers,
)
from ._state import _DEDICATED_EVENT_KEYS, _default_state_projection

logger = logging.getLogger(__name__)


class CheckpointState(TypedDict):
    """Checkpoint-derived state passed to seed/resume factories."""

    reducers: dict[str, Any]
    events: Any
    messages: Any
    pending_interrupts: list[Any]
    is_interrupted: bool
    snapshot: Any  # raw LangGraph StateSnapshot for advanced access


if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Awaitable, Callable, Iterable, Iterator
    from collections.abc import Set as AbstractSet

    from ag_ui.core import RunAgentInput
    from langchain_core.runnables import RunnableConfig
    from langgraph.types import StateSnapshot

    from langgraph_events._graph import EventGraph
    from langgraph_events._types import PauseSignal

    from ._protocols import EventMapper, ResumeFactory, SeedFactory


_T = TypeVar("_T")

_SERVER_ONLY_KEYS = frozenset({"thread_id", "checkpoint_id", "checkpoint_ns"})
"""The ``configurable`` keys that a client cannot set. They choose a checkpoint."""


class _FactoryRefusalError(Exception):
    """Carry a :class:`RunRefusedError` from a factory to ``_as_run``.

    A handler can also raise ``RunRefusedError``. Only a refusal from a
    factory gets its own ``RUN_ERROR`` code, so the factory call wraps it.
    """

    def __init__(self, refusal: RunRefusedError) -> None:
        super().__init__(refusal)
        self.refusal = refusal


async def _call_factory(
    factory: Callable[..., _T | Awaitable[_T]],
    input_data: RunAgentInput,
    checkpoint_state: CheckpointState | None,
    *,
    accepts_state: bool,
) -> _T:
    """Call a sync or async seed or resume *factory*.

    A ``RunRefusedError`` from the factory leaves as ``_FactoryRefusalError``.
    """
    args = (input_data, checkpoint_state) if accepts_state else (input_data,)
    try:
        result = factory(*args)
        if inspect.isawaitable(result):
            return await result
        return result
    except RunRefusedError as refusal:
        raise _FactoryRefusalError(refusal) from refusal


class AGUIAdapter:
    """Map an EventGraph's typed event stream to the AG-UI protocol.

    Orchestrates a pipeline: EventGraph → mapper chain → AG-UI events.
    """

    def __init__(
        self,
        graph: EventGraph,
        *,
        seed_factory: SeedFactory,
        resume_factory: ResumeFactory | None = None,
        mappers: list[EventMapper] | None = None,
        include_reducers: bool | list[str] = True,
        error_message: str | None = None,
        on_unmapped: Literal["warn", "ignore", "raise"] = "warn",
    ) -> None:
        if resume_factory is not None and graph._checkpointer is None:
            raise ValueError(
                "AGUIAdapter resume_factory requires a checkpointer on the EventGraph"
            )
        if "messages" not in graph.topology.reducers:
            raise ValueError(
                "AGUIAdapter requires a message_reducer() on the EventGraph. "
                "Add reducers=[message_reducer()] when constructing your "
                "EventGraph."
            )

        # Validate input shape early.  Bools and lists are the only
        # supported forms — fail loudly on garbage instead of silently
        # producing an empty snapshot at runtime.
        if not isinstance(include_reducers, (bool, list)):
            raise TypeError(
                f"include_reducers must be bool | list[str], "
                f"got {type(include_reducers).__name__}"
            )

        if on_unmapped not in ("warn", "ignore", "raise"):
            raise ValueError(
                f"on_unmapped must be 'warn', 'ignore', or 'raise', got {on_unmapped!r}"
            )

        self._graph = graph
        self._seed_factory = seed_factory
        self._resume_factory = resume_factory
        self._error_message = error_message
        self._on_unmapped = on_unmapped
        self._seed_accepts_state = self._accepts_extra_positional(seed_factory)
        self._resume_accepts_state = (
            self._accepts_extra_positional(resume_factory)
            if resume_factory is not None
            else False
        )

        # Force-include "messages" in the activation set so message-driven
        # AG-UI events can read reducer state.  Layer 2 of
        # default_state_projection then strips it from the snapshot.
        # Note: `False` resolves to activation=["messages"] (so
        # MessagesSnapshot still works) while the projection returns `{}` —
        # empty StateSnapshot, populated MessagesSnapshot.  Intentional.
        resolved = graph.topology.resolve_reducer_names(include_reducers)
        if "messages" not in resolved:
            include_reducers = (
                [*include_reducers, "messages"]
                if isinstance(include_reducers, list)
                else ["messages"]
            )
        self._activation: bool | list[str] = include_reducers
        self._selection = graph.topology._select_reducers(resolved)
        # Pre-compute the allow-list set for fast filtering.  None for
        # bool=True (no filter beyond the framework strip).
        self._allowed_keys: frozenset[str] | None = (
            frozenset(include_reducers) if isinstance(include_reducers, list) else None
        )

        # Build mapper chain: built-ins → user mappers → fallback
        chain: list[Any] = default_mappers(on_unmapped)
        if mappers:
            chain.extend(mappers)
        chain.append(FallbackMapper(on_unmapped))
        self._mappers: list[EventMapper] = chain

    def _project_state(self, reducers: dict[str, Any]) -> dict[str, Any]:
        """Project raw reducer state to client-facing state.

        Strips framework-internal channels (``events``, ``_cursor``, …) and
        dedicated AG-UI keys (``messages``), then applies the
        ``include_reducers`` allow-list (if any).  Used symmetrically in
        both directions: outbound ``StateSnapshotEvent`` and inbound
        ``RunAgentInput.state`` echo into ``FrontendStateMutated``.
        """
        base = _default_state_projection(reducers)
        if self._allowed_keys is None:
            return base
        return {k: v for k, v in base.items() if k in self._allowed_keys}

    def _map_event(self, event: object, ctx: MapperContext) -> list[BaseEvent]:
        """Run event through the mapper chain. First non-None wins."""
        for mapper in self._mappers:
            result = mapper.map(event, ctx)
            if result is not None:
                return result
        return []  # pragma: no cover — FallbackMapper always claims

    @staticmethod
    def _interrupts_from_snapshot(snapshot: StateSnapshot) -> list[Any]:
        """Extract Interrupted payload events from a checkpoint snapshot."""
        return [i.value for i in _snapshot_interrupts(snapshot) if i.value is not None]

    async def _aget_checkpoint_snapshot(
        self, config: RunnableConfig
    ) -> StateSnapshot | None:
        """Return async checkpoint snapshot, or None when unsupported."""
        if self._graph._checkpointer is None:
            return None
        return await self._graph.compiled.aget_state(config)

    @staticmethod
    def _build_checkpoint_state(snapshot: StateSnapshot) -> CheckpointState:
        """Build resume-factory state payload from a checkpoint snapshot.

        The graph seeds its reducer state from *snapshot* after the factory
        runs. So the factory gets a copy of each reducer value, and a change
        to a copy does not reach the run.
        """
        values = snapshot.values if isinstance(snapshot.values, dict) else {}
        reducers = {name: copy.copy(value) for name, value in values.items()}
        pending = AGUIAdapter._interrupts_from_snapshot(snapshot)
        return {
            "reducers": reducers,
            "events": reducers.get("events"),
            "messages": reducers.get("messages"),
            "pending_interrupts": pending,
            "is_interrupted": bool(pending),
            "snapshot": snapshot._replace(values=reducers),
        }

    @staticmethod
    def _accepts_extra_positional(fn: Callable[..., object]) -> bool:
        """Whether *fn* accepts a second positional arg (checkpoint state)."""
        parameters = tuple(inspect.signature(fn).parameters.values())
        if any(p.kind is inspect.Parameter.VAR_POSITIONAL for p in parameters):
            return True
        positional = [
            p
            for p in parameters
            if p.kind
            in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            )
        ]
        return len(positional) >= 2

    @classmethod
    def _build_config(
        cls,
        input_data: RunAgentInput,
        thread_id: str,
        *,
        config: RunnableConfig | None = None,
        client_configurable: AbstractSet[str] = frozenset(),
        deadline: float | None = None,
        pause_on: PauseSignal | None = None,
    ) -> RunnableConfig:
        """Build the LangGraph config of one run at the client trust boundary.

        The server *config* is the only source of run config. The adapter
        ignores the client run config in ``forwarded_props``. The one exception
        is a client ``configurable`` key that *client_configurable* names. A
        server key wins over a client key of the same name. ``thread_id`` always
        comes from *thread_id*. Delegates ``deadline`` and ``pause_on`` to
        :func:`_with_pause_budget`, the same writer as the :class:`EventGraph`
        entry points.
        """
        if isinstance(client_configurable, str):
            raise TypeError(
                "client_configurable=... must be a set of key names, not the "
                f"string {client_configurable!r}."
            )
        server_configurable = (config or {}).get("configurable") or {}
        for argument, keys, owned in (
            ("config", server_configurable, {"thread_id"}),
            ("client_configurable", client_configurable, _SERVER_ONLY_KEYS),
        ):
            reserved = sorted(
                key
                for key in keys
                if key in owned or key.startswith(_FRAMEWORK_KEY_PREFIX)
            )
            if reserved:
                raise ValueError(
                    f"{argument}=... must not name the reserved configurable "
                    f"keys {reserved!r}. The adapter owns thread_id and every "
                    "__lge_ key. Only the server can choose a checkpoint."
                )

        client = cls._client_configurable(input_data)
        run_config = {
            **(config or {}),
            "configurable": {
                **{key: client[key] for key in client_configurable if key in client},
                **server_configurable,
                "thread_id": thread_id,
            },
        }
        return _with_pause_budget(
            cast("RunnableConfig", run_config), deadline, pause_on
        )

    @staticmethod
    def _client_configurable(input_data: RunAgentInput) -> dict[str, Any]:
        """Return the client ``configurable`` in ``forwarded_props``, or ``{}``.

        A client sends it under ``langgraph_config``, under ``config``, or at
        the top level of ``forwarded_props``.
        """
        forwarded = input_data.forwarded_props
        if not isinstance(forwarded, dict):
            return {}
        candidate = forwarded.get("langgraph_config")
        if not isinstance(candidate, dict):
            candidate = forwarded.get("config")
        if not isinstance(candidate, dict):
            candidate = forwarded
        configurable = candidate.get("configurable")
        return configurable if isinstance(configurable, dict) else {}

    def connect(
        self,
        input_data: RunAgentInput,
        *,
        config: RunnableConfig | None = None,
    ) -> AsyncIterator[BaseEvent]:
        """Emit checkpoint-backed state without executing the graph.

        The replay is one AG-UI run. It starts with ``RUN_STARTED`` and ends
        with ``RUN_FINISHED``, or with ``RUN_ERROR`` when the replay raises.

        Args:
            input_data: AG-UI ``RunAgentInput``.
            config: The server run config.  The checkpointer reads the
                checkpoint with it, for example with a tenant key in
                ``configurable``.  Pass the same config as to
                :meth:`stream`.  The adapter ignores the client run config.
                ``None`` (default) reads the checkpoint with ``thread_id``
                only.

        Raises:
            ValueError: *config* ``configurable`` names ``thread_id`` or a
                key with the framework prefix ``__lge_``. Or
                ``input_data.thread_id`` is empty. ``connect()`` raises it
                when it is called, before the stream starts.
        """
        ctx = self._run_context(input_data)
        run_config = self._build_config(input_data, ctx.thread_id, config=config)
        return self._as_run(
            ctx, self._replay(run_config, partial(self._checkpoint_events, ctx=ctx))
        )

    reconnect = connect

    def snapshot(
        self,
        input_data: RunAgentInput,
        *,
        config: RunnableConfig | None = None,
    ) -> AsyncIterator[BaseEvent]:
        """Emit the events of :meth:`connect` without the pending interrupts.

        It takes the same arguments and raises the same errors. Use it when a
        live run holds the thread. See ``docs/agui.md``.
        """
        ctx = self._run_context(input_data)
        run_config = self._build_config(input_data, ctx.thread_id, config=config)
        return self._as_run(ctx, self._replay(run_config, self._snapshot_events))

    async def _replay(
        self,
        config: RunnableConfig,
        events: Callable[[StateSnapshot | None], Iterable[BaseEvent]],
    ) -> AsyncIterator[BaseEvent]:
        for agui_event in events(await self._aget_checkpoint_snapshot(config)):
            yield agui_event

    def _checkpoint_events(
        self, checkpoint: StateSnapshot | None, ctx: MapperContext
    ) -> Iterator[BaseEvent]:
        """Yield the snapshots, then the pending interrupts, of *checkpoint*."""
        yield from self._snapshot_events(checkpoint)
        if checkpoint is None:
            return
        for interrupted in self._interrupts_from_snapshot(checkpoint):
            yield from self._map_event(interrupted, ctx)

    def _snapshot_events(self, checkpoint: StateSnapshot | None) -> list[BaseEvent]:
        reducers = (
            checkpoint.values
            if checkpoint is not None and isinstance(checkpoint.values, dict)
            else {}
        )
        if reducers:
            reducers = reducers | self._selection.derived_values(reducers)
        return [
            build_state_snapshot(self._project_state(reducers)),
            build_messages_snapshot(reducers.get("messages") or []),
        ]

    def _should_gate_with_checkpoint_replay(
        self,
        resume_event: Event | None,
        checkpoint_state: CheckpointState | None,
    ) -> bool:
        """Whether interrupt gate should short-circuit to checkpoint replay."""
        return (
            resume_event is None
            and checkpoint_state is not None
            and checkpoint_state["is_interrupted"]
        )

    def _extract_frontend_state(
        self,
        input_data: RunAgentInput,
    ) -> dict[str, Any] | None:
        """Return the projected client state, or ``None`` if there's
        nothing to emit.

        AG-UI ships ``RunAgentInput.state`` as a snapshot.  We apply the
        same projection used outbound (framework internals stripped,
        dedicated keys stripped, then the user's ``include_reducers``) so
        a stale or untrusted client can't echo back keys the dev decided
        are internal.  A ``Derived`` key is dropped too: the server computes
        it, so a client cannot set it.  Empty / non-dict state is treated as
        "no update."
        """
        raw = input_data.state
        if not isinstance(raw, dict) or not raw:
            return None
        derived = self._graph.topology.derived
        filtered = {
            k: v for k, v in self._project_state(raw).items() if k not in derived
        }
        return filtered or None

    def _prepend_frontend_state(
        self,
        input_data: RunAgentInput,
        seed: Event | list[Event],
    ) -> list[Event]:
        """Return a seed list with ``FrontendStateMutated`` prepended
        when the input carries non-empty, non-dedicated state.

        Used on the non-resume path; callers pass the result to
        ``graph.astream_events(...)``.
        """
        seed_list: list[Event] = list(seed) if isinstance(seed, list) else [seed]
        filtered = self._extract_frontend_state(input_data)
        if filtered is None:
            return seed_list
        return [FrontendStateMutated(state=filtered), *seed_list]

    async def _resume_event_stream(
        self,
        input_data: RunAgentInput,
        resume_event: Event,
        config: RunnableConfig,
        snapshot: StateSnapshot | None,
        context: object,
    ) -> AsyncIterator[StreamItem]:
        """Resume path that routes ``FrontendStateMutated`` through both
        reducer dispatch and the LangGraph channel state.

        For each reducer subscribing to ``FrontendStateMutated``, compute the
        contribution and ``apre_seed`` it to the channel so handlers reading
        reducer state via parameter injection on the resumed step see the
        update.  This preserves reducer ``fn`` semantics (transformations,
        ``SKIP``) — unlike the old ``apre_seed(raw_state)`` bypass that wrote
        client values directly and could clobber backend-authoritative
        channels with stale snapshot keys.

        Then pass FSM as a seed to ``astream_resume`` so it appears in the
        output stream and audit log alongside the resume event.

        **Limitation**: ``@on(FrontendStateMutated)`` handlers do not fire on
        resume — LangGraph's ``Command(resume=...)`` carries one value and
        seeds are dispatched out-of-graph.  Use ``@on(Resumed)`` for
        resume-time side effects.
        """
        fsm = self._build_resume_frontend_state_event(input_data)
        if fsm is not None:
            await self._commit_resume_frontend_state(fsm, config)
            # The commit writes a new checkpoint, so *snapshot* is stale.
            snapshot = None
        async for item in self._graph.astream_resume(
            resume_event,
            seeds=[fsm] if fsm is not None else [],
            include_reducers=self._activation,
            include_llm_tokens=True,
            include_custom_events=True,
            _snapshot=snapshot,
            config=config,
            context=context,
        ):
            yield item

    def _build_resume_frontend_state_event(
        self, input_data: RunAgentInput
    ) -> FrontendStateMutated | None:
        """Build a ``FrontendStateMutated`` from the projected client state,
        or ``None`` when there's nothing to apply on resume.
        """
        filtered = self._extract_frontend_state(input_data)
        if filtered is None:
            return None
        return FrontendStateMutated(state=filtered)

    async def _commit_resume_frontend_state(
        self, fsm: FrontendStateMutated, config: RunnableConfig
    ) -> None:
        """Persist FSM into the audit log AND apply reducer contributions
        to user channels — both before the resume's domain dispatch runs.

        Channel reducers merge the writes (``operator.add`` for ``events``,
        scalar / accumulator for user reducers).  We do NOT emit FSM via
        the seed_node here: ``Command(resume=...)`` carries one value and
        doesn't route seeds through ``inp["events"]``, so this ``apre_seed``
        is the sole path that lands FSM in the persisted ``events``
        channel.  ``_astream_v2`` yields seeds into the output stream but
        deliberately skips re-applying seed contributions on resume to
        avoid double-counting accumulator reducers.

        If a future refactor unifies seed dispatch through ``inp["events"]``,
        drop the ``{"events": [fsm]}`` entry to avoid duplication.
        """
        updates: dict[str, Any] = {"events": [fsm]}
        updates.update(self._reducer_updates_for([fsm]))
        await self._graph.apre_seed(config, updates)

    def _reducer_updates_for(self, events: list[Event]) -> dict[str, Any]:
        """Compute per-channel reducer contributions for *events*.

        Runs each registered reducer's ``collect`` against *events* and
        returns the dict of channel updates (only including channels with
        actual contributions).  Used on the resume path to materialise FSM
        reducer effects into LangGraph channel state so subsequent handlers
        see them via parameter injection.
        """
        updates: dict[str, Any] = {}
        for name, reducer in self._graph.topology.reducers.items():
            contribution = reducer.collect(events)
            if reducer.has_contributions(contribution):
                updates[name] = contribution
        return updates

    async def _build_event_stream(
        self,
        input_data: RunAgentInput,
        checkpoint_state: CheckpointState | None,
        resume_event: Event | None,
        config: RunnableConfig,
        snapshot: StateSnapshot | None,
        context: object,
    ) -> AsyncIterator[StreamItem]:
        """Create the underlying EventGraph async event stream.

        *snapshot* is the checkpoint that the turn loaded before the run. The
        graph uses it, so it does not load the checkpoint again.
        """
        if resume_event is not None:
            return self._resume_event_stream(
                input_data, resume_event, config, snapshot, context
            )
        seed: Event | list[Event] = await _call_factory(
            self._seed_factory,
            input_data,
            checkpoint_state,
            accepts_state=self._seed_accepts_state,
        )
        seeds = self._prepend_frontend_state(input_data, seed)
        return self._graph.astream_events(
            seeds,
            include_reducers=self._activation,
            include_llm_tokens=True,
            include_custom_events=True,
            _snapshot=snapshot,
            config=config,
            context=context,
        )

    @staticmethod
    def _is_interrupt(event: object) -> bool:
        return isinstance(event, Interrupted)

    def _events_from_llm_token(
        self, item: LLMToken, ctx: MapperContext
    ) -> list[BaseEvent]:
        message_id, is_new = ctx.ensure_stream_message_id(item.run_id)
        events: list[BaseEvent] = []
        if is_new:
            events.append(
                TextMessageStartEvent(
                    type=EventType.TEXT_MESSAGE_START,
                    message_id=message_id,
                    role="assistant",
                )
            )
        events.append(
            TextMessageContentEvent(
                type=EventType.TEXT_MESSAGE_CONTENT,
                message_id=message_id,
                delta=item.content,
            )
        )
        return events

    def _events_from_llm_stream_end(
        self,
        item: LLMStreamEnd,
        ctx: MapperContext,
    ) -> list[BaseEvent]:
        events: list[BaseEvent] = []
        closed_id = ctx.close_stream_message_id(item.run_id)
        if closed_id is not None:
            events.append(
                TextMessageEndEvent(
                    type=EventType.TEXT_MESSAGE_END,
                    message_id=closed_id,
                )
            )
        for tc_id in ctx.close_tool_calls_for_run(item.run_id):
            events.append(
                ToolCallEndEvent(
                    type=EventType.TOOL_CALL_END,
                    tool_call_id=tc_id,
                )
            )
        return events

    def _events_from_llm_tool_call_chunk(
        self,
        item: LLMToolCallChunk,
        ctx: MapperContext,
    ) -> list[BaseEvent]:
        tracked = ctx.ensure_tool_call_id(
            item.run_id, item.call_index, item.tool_call_id, item.name
        )
        if tracked is None:
            return []
        resolved_id, is_new = tracked
        events: list[BaseEvent] = []
        if is_new:
            events.append(
                ToolCallStartEvent(
                    type=EventType.TOOL_CALL_START,
                    tool_call_id=resolved_id,
                    tool_call_name=item.name,
                    parent_message_id=ctx.current_stream_message_id(item.run_id),
                )
            )
        if item.args_delta:
            events.append(
                ToolCallArgsEvent(
                    type=EventType.TOOL_CALL_ARGS,
                    tool_call_id=resolved_id,
                    delta=item.args_delta,
                )
            )
        return events

    def _events_from_stream_frame(
        self,
        item: StreamFrame,
        ctx: MapperContext,
        *,
        is_resume: bool,
        state_snapshot_emitted: bool,
    ) -> tuple[list[BaseEvent], bool]:
        event = item.event
        if is_resume and self._is_interrupt(event):
            return [], state_snapshot_emitted

        events: list[BaseEvent] = []
        changed_reducers = item.changed_reducers
        should_emit_state_snapshot = not state_snapshot_emitted
        if changed_reducers is not None:
            should_emit_state_snapshot = should_emit_state_snapshot or any(
                name not in _DEDICATED_EVENT_KEYS for name in changed_reducers
            )

        if should_emit_state_snapshot:
            events.append(build_state_snapshot(self._project_state(item.reducers)))
            state_snapshot_emitted = True

        messages = item.reducers.get("messages")
        if (
            messages is not None
            and changed_reducers is not None
            and "messages" in changed_reducers
        ):
            events.append(build_messages_snapshot(messages))

        events.extend(self._map_event(event, ctx))
        return events, state_snapshot_emitted

    def _events_from_bare_item(
        self,
        item: object,
        ctx: MapperContext,
        *,
        is_resume: bool,
    ) -> list[BaseEvent]:
        if is_resume and self._is_interrupt(item):
            return []
        return self._map_event(item, ctx)

    async def _stream_event_source(  # noqa: PLR0912
        self,
        input_data: RunAgentInput,
        ctx: MapperContext,
        config: RunnableConfig,
        context: object,
    ) -> AsyncIterator[BaseEvent]:
        """Stream domain events through the mapper chain."""
        # Determine resume vs fresh run — lazy checkpoint read
        needs_checkpoint = (
            self._seed_accepts_state
            or self._resume_accepts_state
            or self._graph._checkpointer is not None
        )
        checkpoint_snapshot = (
            await self._aget_checkpoint_snapshot(config) if needs_checkpoint else None
        )
        checkpoint_state = (
            self._build_checkpoint_state(checkpoint_snapshot)
            if checkpoint_snapshot is not None
            else None
        )
        resume_event: Event | None = (
            None
            if self._resume_factory is None
            else await _call_factory(
                self._resume_factory,
                input_data,
                checkpoint_state,
                accepts_state=self._resume_accepts_state,
            )
        )
        is_resume = resume_event is not None
        emitted_state_snapshot = False

        # Interrupt gate: re-emit state without executing when interrupted
        if self._should_gate_with_checkpoint_replay(resume_event, checkpoint_state):
            for agui_event in self._checkpoint_events(checkpoint_snapshot, ctx):
                yield agui_event
            return

        event_stream = await self._build_event_stream(
            input_data,
            checkpoint_state,
            resume_event,
            config,
            checkpoint_snapshot,
            context,
        )

        async for item in event_stream:
            if isinstance(item, LLMToken):
                for agui_event in self._events_from_llm_token(item, ctx):
                    yield agui_event
            elif isinstance(item, LLMToolCallChunk):
                for agui_event in self._events_from_llm_tool_call_chunk(item, ctx):
                    yield agui_event
            elif isinstance(item, LLMStreamEnd):
                for agui_event in self._events_from_llm_stream_end(item, ctx):
                    yield agui_event
            elif isinstance(item, StateSnapshotFrame):
                yield build_state_snapshot(item.data)
            elif isinstance(item, CustomEventFrame):
                yield CustomEvent(
                    type=EventType.CUSTOM,
                    name=item.name,
                    value=item.data,
                )
            elif isinstance(item, InterruptFrame):
                for agui_event in self._map_event(item.value, ctx):
                    yield agui_event
            elif isinstance(item, StreamFrame):
                (
                    agui_events,
                    emitted_state_snapshot,
                ) = self._events_from_stream_frame(
                    item,
                    ctx,
                    is_resume=is_resume,
                    state_snapshot_emitted=emitted_state_snapshot,
                )
                for agui_event in agui_events:
                    yield agui_event
            else:
                for agui_event in self._events_from_bare_item(
                    item,
                    ctx,
                    is_resume=is_resume,
                ):
                    yield agui_event

        for message_id in ctx.drain_open_stream_message_ids():
            yield TextMessageEndEvent(
                type=EventType.TEXT_MESSAGE_END,
                message_id=message_id,
            )
        for tc_id in ctx.drain_open_tool_call_ids():
            yield ToolCallEndEvent(
                type=EventType.TOOL_CALL_END,
                tool_call_id=tc_id,
            )

    def stream(
        self,
        input_data: RunAgentInput,
        *,
        config: RunnableConfig | None = None,
        context: object = None,
        client_configurable: AbstractSet[str] = frozenset(),
        deadline: float | None = None,
        pause_on: PauseSignal | None = None,
    ) -> AsyncIterator[BaseEvent]:
        """Yield AG-UI events for one request.

        This is the core method — call it from your endpoint handler
        and pass the result to ``encode_sse_stream()`` or
        ``create_starlette_response()``.

        Args:
            input_data: AG-UI ``RunAgentInput``.
            config: The server run config, for example ``configurable``,
                ``recursion_limit`` or ``metadata``.  It is the only source of
                run config.  A handler reads it from its ``RunnableConfig``.
                ``None`` (default) gives the run only its ``thread_id``.
            context: The run context, an instance of the graph's
                ``EventGraph(context_schema=...)``.  The adapter passes it to
                the run on a start and on a resume.  It is not checkpointed.
                A client cannot set it.  ``None`` (default) is valid only
                for a graph without a context schema.
            client_configurable: The names of the client ``configurable``
                keys that the run accepts from ``forwarded_props``.  The
                adapter ignores every other part of the client run config.
                A key of the server *config* wins over a client key of the
                same name.  Empty (default) accepts no client key.  A
                client cannot choose a checkpoint, so ``checkpoint_id`` and
                ``checkpoint_ns`` are reserved here.  The server *config*
                can set them.
            deadline: Absolute ``time.monotonic()`` value.  When the run's
                router observes a current time >= deadline between
                dispatch rounds, it emits :class:`RunPausedOnDeadline` and
                the run finalises cleanly.  ``RunPaused`` has no default
                wire mapping.  Register a custom mapper to send it.  Pair
                with a strictly larger outer hard cancellation
                (``asyncio.wait_for`` / ``timeout=`` etc.) to avoid
                wire-format violations from mid-stream cancellation.
                ``None`` (default) disables.
            pause_on: An object with ``is_set()``, for example an
                ``asyncio.Event``.  When the router observes
                ``is_set()`` between dispatch rounds, it emits
                :class:`RunPausedOnSignal` and the run finalises cleanly,
                the same way as a deadline pause.  Set the signal to stop
                a live run early, for example when the process shuts down.
                ``None`` (default) disables.

        Raises:
            TypeError: *pause_on* is a class, or has no callable
                ``is_set``.  Or *client_configurable* is a ``str``.  Or the
                graph has a context schema and *context* is ``None``.
                ``stream()`` raises it when it is called, before the stream
                starts.
            ValueError: *config* ``configurable`` names ``thread_id`` or a
                key with the framework prefix ``__lge_``.  Or
                *client_configurable* names one of these keys,
                ``checkpoint_id`` or ``checkpoint_ns``.  Or
                ``input_data.thread_id`` is empty.  ``stream()`` raises it
                when it is called, before the stream starts.
        """
        self._graph.topology.require_context(context)
        ctx = self._run_context(input_data)
        run_config = self._build_config(
            input_data,
            ctx.thread_id,
            config=config,
            client_configurable=client_configurable,
            deadline=deadline,
            pause_on=pause_on,
        )
        return self._as_run(
            ctx, self._stream_event_source(input_data, ctx, run_config, context)
        )

    @staticmethod
    def _run_context(input_data: RunAgentInput) -> MapperContext:
        """Build the mapper context of one run, with a new run id if absent."""
        if not input_data.thread_id:
            raise ValueError(
                "RunAgentInput.thread_id must name a thread, not "
                f"{input_data.thread_id!r}."
            )
        return MapperContext(
            run_id=input_data.run_id or str(uuid.uuid4()),
            thread_id=input_data.thread_id,
            input_data=input_data,
        )

    async def _as_run(
        self, ctx: MapperContext, source: AsyncIterator[BaseEvent]
    ) -> AsyncIterator[BaseEvent]:
        """Wrap *source* in ``RUN_STARTED`` and ``RUN_FINISHED``.

        If *source* raises, close each open message and tool call, then end
        the stream with ``RUN_ERROR`` instead of ``RUN_FINISHED``. A factory
        refusal ends it with ``RUN_ERROR`` that carries the refusal code.
        """
        yield RunStartedEvent(
            type=EventType.RUN_STARTED,
            thread_id=ctx.thread_id,
            run_id=ctx.run_id,
        )

        try:
            async for agui_event in source:
                yield agui_event

        except UnmappedEventError:
            # on_unmapped="raise" is a strict CI gate — propagate instead of
            # masking it as a RunErrorEvent.
            raise

        except _FactoryRefusalError as refused:
            refusal = refused.refusal
            logger.warning("AG-UI run refused: %s", refusal.code)
            yield RunErrorEvent(
                type=EventType.RUN_ERROR,
                code=refusal.code,
                message=refusal.message,
            )
            return

        except Exception as exc:
            logger.exception("AG-UI run failed")
            for message_id in ctx.drain_open_stream_message_ids():
                yield TextMessageEndEvent(
                    type=EventType.TEXT_MESSAGE_END,
                    message_id=message_id,
                )
            for tc_id in ctx.drain_open_tool_call_ids():
                yield ToolCallEndEvent(
                    type=EventType.TOOL_CALL_END,
                    tool_call_id=tc_id,
                )
            yield RunErrorEvent(
                type=EventType.RUN_ERROR,
                message=self._error_message or str(exc),
            )
            return

        yield RunFinishedEvent(
            type=EventType.RUN_FINISHED,
            thread_id=ctx.thread_id,
            run_id=ctx.run_id,
        )
