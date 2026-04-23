"""AGUIAdapter — core orchestrator mapping EventGraph streams to AG-UI events."""

from __future__ import annotations

import inspect
import logging
import uuid
from typing import TYPE_CHECKING, Any, TypedDict, cast

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
from langgraph_events.stream import (
    CustomEventFrame,
    LLMStreamEnd,
    LLMToken,
    LLMToolCallChunk,
    StateSnapshotFrame,
    StreamFrame,
    StreamItem,
)

from ._context import MapperContext
from ._events import FrontendStateMutated
from ._mappers import (
    FallbackMapper,
    build_messages_snapshot,
    build_state_snapshot,
    default_mappers,
)

logger = logging.getLogger(__name__)

_DEDICATED_EVENT_KEYS: frozenset[str] = frozenset({"messages"})


def _strip_dedicated_keys(reducers: dict[str, Any]) -> dict[str, Any]:
    """Return *reducers* without keys that have dedicated AG-UI events."""
    return {k: v for k, v in reducers.items() if k not in _DEDICATED_EVENT_KEYS}


class CheckpointState(TypedDict):
    """Checkpoint-derived state passed to seed/resume factories."""

    reducers: dict[str, Any]
    events: Any
    messages: Any
    pending_interrupts: list[Any]
    is_interrupted: bool
    snapshot: Any  # raw LangGraph StateSnapshot for advanced access


if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from ag_ui.core import RunAgentInput

    from langgraph_events._graph import EventGraph

    from ._protocols import EventMapper, ResumeFactory, SeedFactory


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
    ) -> None:
        if resume_factory is not None and graph._checkpointer is None:
            raise ValueError(
                "AGUIAdapter resume_factory requires a checkpointer on the EventGraph"
            )
        self._graph = graph
        self._seed_factory = seed_factory
        self._resume_factory = resume_factory
        self._include_reducers = include_reducers
        self._error_message = error_message
        self._seed_accepts_state = self._accepts_extra_positional(seed_factory)
        self._resume_accepts_state = (
            self._accepts_extra_positional(resume_factory)
            if resume_factory is not None
            else False
        )

        # Ensure message_reducer is present and included
        resolved = graph._resolve_reducer_names(include_reducers)
        if "messages" not in resolved:
            if "messages" in graph._reducers:
                # User excluded "messages" — force-include it
                if isinstance(include_reducers, list):
                    include_reducers = [*include_reducers, "messages"]
                else:
                    include_reducers = ["messages"]
                self._include_reducers = include_reducers
            else:
                raise ValueError(
                    "AGUIAdapter requires a message_reducer() on the "
                    "EventGraph. "
                    "Add reducers=[message_reducer()] when constructing "
                    "your EventGraph."
                )

        # Build mapper chain: built-ins → user mappers → fallback
        chain: list[Any] = default_mappers()
        if mappers:
            chain.extend(mappers)
        chain.append(FallbackMapper())
        self._mappers: list[EventMapper] = chain

    def _map_event(self, event: Any, ctx: MapperContext) -> list[BaseEvent]:
        """Run event through the mapper chain. First non-None wins."""
        for mapper in self._mappers:
            result = mapper.map(event, ctx)
            if result is not None:
                return result
        return []  # pragma: no cover — FallbackMapper always claims

    @staticmethod
    def _interrupts_from_snapshot(snapshot: Any) -> list[Any]:
        """Extract Interrupted payload events from a checkpoint snapshot."""
        if not snapshot.next:
            return []
        result: list[Any] = []
        for task in snapshot.tasks:
            for interrupt in getattr(task, "interrupts", ()):
                value = getattr(interrupt, "value", None)
                if value is not None:
                    result.append(value)
        return result

    async def _aget_checkpoint_snapshot(self, config: Any) -> Any | None:
        """Return async checkpoint snapshot, or None when unsupported."""
        if self._graph._checkpointer is None:
            return None
        return await self._graph.compiled.aget_state(config)

    @staticmethod
    def _build_checkpoint_state(snapshot: Any) -> CheckpointState:
        """Build resume-factory state payload from a checkpoint snapshot."""
        reducers = snapshot.values if isinstance(snapshot.values, dict) else {}
        pending = AGUIAdapter._interrupts_from_snapshot(snapshot)
        return {
            "reducers": reducers,
            "events": reducers.get("events"),
            "messages": reducers.get("messages"),
            "pending_interrupts": pending,
            "is_interrupted": bool(pending),
            "snapshot": snapshot,
        }

    @staticmethod
    def _accepts_extra_positional(fn: Any) -> bool:
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

    def _call_resume_factory(
        self,
        input_data: RunAgentInput,
        checkpoint_state: CheckpointState | None,
    ) -> Any:
        """Call resume_factory with optional checkpoint state if supported."""
        if self._resume_factory is None:
            return None
        factory = cast("Any", self._resume_factory)
        if self._resume_accepts_state:
            return factory(input_data, checkpoint_state)
        return factory(input_data)

    def _call_seed_factory(
        self,
        input_data: RunAgentInput,
        checkpoint_state: CheckpointState | None,
    ) -> Any:
        """Call seed_factory with optional checkpoint state if supported."""
        factory = cast("Any", self._seed_factory)
        if self._seed_accepts_state:
            return factory(input_data, checkpoint_state)
        return factory(input_data)

    @staticmethod
    def _build_config(input_data: RunAgentInput, thread_id: str) -> dict[str, Any]:
        """Build LangGraph config, including passthrough forwarded props."""
        config: dict[str, Any] = {"configurable": {"thread_id": thread_id}}
        forwarded_raw = input_data.forwarded_props
        forwarded = forwarded_raw if isinstance(forwarded_raw, dict) else {}
        candidate: Any = None

        if isinstance(forwarded.get("langgraph_config"), dict):
            candidate = forwarded["langgraph_config"]
        elif isinstance(forwarded.get("config"), dict):
            candidate = forwarded["config"]
        elif isinstance(forwarded, dict) and (
            "configurable" in forwarded or "recursion_limit" in forwarded
        ):
            candidate = forwarded

        if not isinstance(candidate, dict):
            return config

        merged = {**candidate}
        configurable = candidate.get("configurable")
        merged["configurable"] = {
            **(configurable if isinstance(configurable, dict) else {}),
            "thread_id": thread_id,
        }
        return merged

    async def connect(self, input_data: RunAgentInput) -> AsyncIterator[BaseEvent]:
        """Emit checkpoint-backed state without executing the graph."""
        run_id = input_data.run_id or str(uuid.uuid4())
        thread_id = input_data.thread_id or str(uuid.uuid4())
        ctx = MapperContext(
            run_id=run_id,
            thread_id=thread_id,
            input_data=input_data,
        )
        config = self._build_config(input_data, thread_id)
        snapshot = await self._aget_checkpoint_snapshot(config)
        if snapshot is None:
            yield build_state_snapshot({})
            yield build_messages_snapshot([])
            return

        reducers = snapshot.values if isinstance(snapshot.values, dict) else {}
        yield build_state_snapshot(_strip_dedicated_keys(reducers))
        yield build_messages_snapshot(reducers.get("messages") or [])

        for interrupted in self._interrupts_from_snapshot(snapshot):
            for agui_event in self._map_event(interrupted, ctx):
                yield agui_event

    reconnect = connect

    def _should_gate_with_checkpoint_replay(
        self,
        resume_event: Any,
        checkpoint_state: CheckpointState | None,
    ) -> bool:
        """Whether interrupt gate should short-circuit to checkpoint replay."""
        return (
            resume_event is None
            and checkpoint_state is not None
            and checkpoint_state["is_interrupted"]
        )

    @staticmethod
    def _extract_frontend_state(
        input_data: RunAgentInput,
    ) -> dict[str, Any] | None:
        """Return the pre-filtered client state, or ``None`` if there's
        nothing to emit.

        AG-UI ships ``RunAgentInput.state`` as a snapshot.  Dedicated keys
        (``messages``) are driven by purpose-built AG-UI events and are
        stripped here so they don't collide.  Empty / non-dict state is
        treated as "no update."
        """
        raw = input_data.state
        if not isinstance(raw, dict) or not raw:
            return None
        filtered = _strip_dedicated_keys(raw)
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
        return [FrontendStateMutated(state=filtered), *seed_list]  # type: ignore[call-arg]

    async def _apply_frontend_state_for_resume(
        self,
        input_data: RunAgentInput,
        config: Any,
    ) -> None:
        """Write client state directly to reducer channels via ``apre_seed``.

        NOTE: On resume we bypass the ``FrontendStateMutated`` dispatch
        path.  ``ainvoke`` on a thread with a pending interrupt would
        consume the interrupt before ``astream_resume`` runs, so we
        write channel values directly (matching state key -> channel
        name).  Trade-off: the reducer's ``fn`` is not invoked on
        resume — in the idiomatic ``fn=lambda e: e.state.get(name,
        SKIP)`` pattern the value flows through unchanged, but
        non-identity ``fn`` logic applies only on non-resume runs.
        No-op when the input state is empty or contains only dedicated
        keys.
        """
        filtered = self._extract_frontend_state(input_data)
        if filtered is None:
            return
        await self._graph.apre_seed(config, filtered)

    async def _resume_event_stream(
        self,
        input_data: RunAgentInput,
        resume_event: Any,
        config: Any,
    ) -> AsyncIterator[StreamItem]:
        """Resume path that first commits FrontendStateMutated to the
        checkpoint, then defers to ``astream_resume``.
        """
        await self._apply_frontend_state_for_resume(input_data, config)
        async for item in self._graph.astream_resume(
            resume_event,
            include_reducers=self._include_reducers,
            include_llm_tokens=True,
            include_custom_events=True,
            config=config,
        ):
            yield item

    def _build_event_stream(
        self,
        input_data: RunAgentInput,
        checkpoint_state: CheckpointState | None,
        resume_event: Any,
        config: Any,
    ) -> AsyncIterator[StreamItem]:
        """Create the underlying EventGraph async event stream."""
        if resume_event is not None:
            return self._resume_event_stream(input_data, resume_event, config)
        seed = self._call_seed_factory(input_data, checkpoint_state)
        seeds = self._prepend_frontend_state(input_data, seed)
        return self._graph.astream_events(
            seeds,
            include_reducers=self._include_reducers,
            include_llm_tokens=True,
            include_custom_events=True,
            config=config,
        )

    @staticmethod
    def _is_interrupt(event: Any) -> bool:
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
        resolved_id, is_new = ctx.ensure_tool_call_id(
            item.run_id, item.call_index, item.tool_call_id, item.name
        )
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
    ) -> tuple[list[BaseEvent], bool, bool]:
        event = item.event
        if is_resume and self._is_interrupt(event):
            return [], False, state_snapshot_emitted

        events: list[BaseEvent] = []
        changed_reducers = item.changed_reducers
        should_emit_state_snapshot = not state_snapshot_emitted
        if changed_reducers is not None:
            should_emit_state_snapshot = should_emit_state_snapshot or any(
                name not in _DEDICATED_EVENT_KEYS for name in changed_reducers
            )

        if should_emit_state_snapshot:
            events.append(build_state_snapshot(_strip_dedicated_keys(item.reducers)))
            state_snapshot_emitted = True

        messages = item.reducers.get("messages")
        if (
            messages is not None
            and changed_reducers is not None
            and "messages" in changed_reducers
        ):
            events.append(build_messages_snapshot(messages))

        events.extend(self._map_event(event, ctx))
        return (
            events,
            self._is_interrupt(event),
            state_snapshot_emitted,
        )

    def _events_from_bare_item(
        self,
        item: Any,
        ctx: MapperContext,
        *,
        is_resume: bool,
    ) -> tuple[list[BaseEvent], bool]:
        if is_resume and self._is_interrupt(item):
            return [], False
        return self._map_event(item, ctx), self._is_interrupt(item)

    async def _stream_event_source(  # noqa: PLR0912
        self,
        input_data: RunAgentInput,
        ctx: MapperContext,
        config: Any,
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
        resume_event = self._call_resume_factory(input_data, checkpoint_state)
        is_resume = resume_event is not None
        emitted_interrupt_in_stream = False
        emitted_state_snapshot = False

        # Interrupt gate: re-emit state without executing when interrupted
        if self._should_gate_with_checkpoint_replay(resume_event, checkpoint_state):
            state = cast("CheckpointState", checkpoint_state)
            yield build_state_snapshot(_strip_dedicated_keys(state["reducers"]))
            messages = state["messages"]
            if messages is not None:
                yield build_messages_snapshot(messages)
            for interrupted in state["pending_interrupts"]:
                for agui_event in self._map_event(interrupted, ctx):
                    yield agui_event
            return

        event_stream = self._build_event_stream(
            input_data,
            checkpoint_state,
            resume_event,
            config,
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
            elif isinstance(item, StreamFrame):
                (
                    agui_events,
                    emitted_interrupt,
                    emitted_state_snapshot,
                ) = self._events_from_stream_frame(
                    item,
                    ctx,
                    is_resume=is_resume,
                    state_snapshot_emitted=emitted_state_snapshot,
                )
                emitted_interrupt_in_stream = (
                    emitted_interrupt_in_stream or emitted_interrupt
                )
                for agui_event in agui_events:
                    yield agui_event
            else:
                agui_events, emitted_interrupt = self._events_from_bare_item(
                    item,
                    ctx,
                    is_resume=is_resume,
                )
                emitted_interrupt_in_stream = (
                    emitted_interrupt_in_stream or emitted_interrupt
                )
                for agui_event in agui_events:
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

        # Detect interrupts from checkpoint state
        if not emitted_interrupt_in_stream:
            snapshot = await self._aget_checkpoint_snapshot(config)
            interrupt_events = (
                self._interrupts_from_snapshot(snapshot) if snapshot is not None else []
            )
            for interrupted in interrupt_events:
                for agui_event in self._map_event(interrupted, ctx):
                    yield agui_event

    async def stream(self, input_data: RunAgentInput) -> AsyncIterator[BaseEvent]:
        """Yield AG-UI events for one request.

        This is the core method — call it from your endpoint handler
        and pass the result to ``encode_sse_stream()`` or
        ``create_starlette_response()``.
        """
        run_id = input_data.run_id or str(uuid.uuid4())
        thread_id = input_data.thread_id or str(uuid.uuid4())

        ctx = MapperContext(
            run_id=run_id,
            thread_id=thread_id,
            input_data=input_data,
        )

        yield RunStartedEvent(
            type=EventType.RUN_STARTED,
            thread_id=thread_id,
            run_id=run_id,
        )

        config: Any = self._build_config(input_data, thread_id)

        try:
            async for agui_event in self._stream_event_source(input_data, ctx, config):
                yield agui_event

        except Exception as exc:
            logger.exception("EventGraph stream failed")
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
            thread_id=thread_id,
            run_id=run_id,
        )
