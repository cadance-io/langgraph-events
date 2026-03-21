"""AGUIAdapter — core orchestrator mapping EventGraph streams to AG-UI events."""

from __future__ import annotations

import logging
import uuid
from typing import TYPE_CHECKING, Any

from ag_ui.core import (
    BaseEvent,
    EventType,
    RunErrorEvent,
    RunFinishedEvent,
    RunStartedEvent,
    TextMessageContentEvent,
    TextMessageEndEvent,
    TextMessageStartEvent,
)

from langgraph_events._event import Interrupted
from langgraph_events._graph import LLMStreamEnd, LLMToken, StreamFrame

from ._context import MapperContext
from ._mappers import (
    FallbackMapper,
    build_messages_snapshot,
    build_state_snapshot,
    default_mappers,
)

logger = logging.getLogger(__name__)

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
        include_reducers: bool | list[str] = False,
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

    def _get_interrupt_events(self, config: Any) -> list[Any]:
        """Extract Interrupted events from checkpoint after stream ends."""
        if self._graph._checkpointer is None:
            return []
        snapshot = self._graph.compiled.get_state(config)
        if not snapshot.next:
            return []
        result: list[Any] = []
        for task in snapshot.tasks:
            for interrupt in getattr(task, "interrupts", ()):
                value = getattr(interrupt, "value", None)
                if value is not None:
                    result.append(value)
        return result

    async def _stream_event_source(  # noqa: PLR0912
        self,
        input_data: RunAgentInput,
        ctx: MapperContext,
        config: Any,
    ) -> AsyncIterator[BaseEvent]:
        """Stream domain events through the mapper chain."""
        # Determine resume vs fresh run
        resume_event = (
            self._resume_factory(input_data)
            if self._resume_factory is not None
            else None
        )
        is_resume = resume_event is not None
        prev_message_ids: tuple[int | str, ...] = ()

        if resume_event is not None:
            event_stream = self._graph.astream_resume(
                resume_event,
                include_reducers=self._include_reducers,
                include_llm_tokens=True,
                config=config,
            )
        else:
            seed = self._seed_factory(input_data)
            event_stream = self._graph.astream_events(
                seed,
                include_reducers=self._include_reducers,
                include_llm_tokens=True,
                config=config,
            )

        async for item in event_stream:
            if isinstance(item, LLMToken):
                message_id, is_new = ctx.ensure_stream_message_id(item.run_id)
                if is_new:
                    yield TextMessageStartEvent(
                        type=EventType.TEXT_MESSAGE_START,
                        message_id=message_id,
                        role="assistant",
                    )
                yield TextMessageContentEvent(
                    type=EventType.TEXT_MESSAGE_CONTENT,
                    message_id=message_id,
                    delta=item.content,
                )
            elif isinstance(item, LLMStreamEnd):
                if item.message_id:
                    ctx.mark_streamed_ai_message(item.message_id)
                closed_id = ctx.close_stream_message_id(item.run_id)
                if closed_id is not None:
                    yield TextMessageEndEvent(
                        type=EventType.TEXT_MESSAGE_END,
                        message_id=closed_id,
                    )
            elif isinstance(item, StreamFrame):
                event = item.event
                if is_resume and isinstance(event, Interrupted):
                    continue
                yield build_state_snapshot(item.reducers)
                messages = item.reducers.get("messages")
                if messages is not None:
                    current_ids = tuple(getattr(m, "id", id(m)) for m in messages)
                    if current_ids != prev_message_ids:
                        prev_message_ids = current_ids
                        yield build_messages_snapshot(messages)
                for agui_event in self._map_event(event, ctx):
                    yield agui_event
            else:
                # Bare Event (no reducers)
                if is_resume and isinstance(item, Interrupted):
                    continue
                for agui_event in self._map_event(item, ctx):
                    yield agui_event

        for message_id in ctx.drain_open_stream_message_ids():
            yield TextMessageEndEvent(
                type=EventType.TEXT_MESSAGE_END,
                message_id=message_id,
            )

        # Detect interrupts from checkpoint state
        for interrupted in self._get_interrupt_events(config):
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

        config: Any = {
            "configurable": {"thread_id": thread_id},
        }

        try:
            async for agui_event in self._stream_event_source(input_data, ctx, config):
                yield agui_event

        except Exception as exc:
            logger.exception("EventGraph stream failed")
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
