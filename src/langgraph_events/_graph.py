"""EventGraph — the main entry point for building event-driven graphs."""

from __future__ import annotations

from typing import Any, Callable

from langgraph.graph import END, START, StateGraph

from langgraph_events._event import Event
from langgraph_events._event_log import EventLog
from langgraph_events._handler import HandlerMeta, extract_handler_meta
from langgraph_events._internal import (
    _FullState,
    _InputState,
    _OutputState,
    make_dispatch,
    make_handler_node,
    make_router_node,
    make_seed_node,
)


class EventGraph:
    """Build and run an event-driven graph from ``@on`` handlers.

    Topology is auto-derived from handler subscriptions.  Internally builds
    a LangGraph ``StateGraph`` with a hub-and-spoke reactive loop.

    Example::

        graph = EventGraph([classify, route, review])
        log = graph.invoke(DocumentReceived(doc_id="1", content="..."))
        print(log.latest(ProcessingComplete))
    """

    def __init__(
        self,
        handlers: list[Callable],
        *,
        max_rounds: int = 100,
    ) -> None:
        if not handlers:
            raise ValueError("EventGraph requires at least one handler")

        self._max_rounds = max_rounds
        self._handler_metas: list[HandlerMeta] = []

        seen_names: dict[str, int] = {}
        for fn in handlers:
            meta = extract_handler_meta(fn)
            # Deduplicate node names
            if meta.name in seen_names:
                seen_names[meta.name] += 1
                meta = HandlerMeta(
                    name=f"{meta.name}_{seen_names[meta.name]}",
                    fn=meta.fn,
                    event_types=meta.event_types,
                    wants_log=meta.wants_log,
                    is_async=meta.is_async,
                )
            else:
                seen_names[meta.name] = 1
            self._handler_metas.append(meta)

    def compile(self, **kwargs: Any):
        """Compile into a LangGraph ``CompiledStateGraph``.

        All keyword arguments are forwarded to ``StateGraph.compile()``,
        e.g. ``checkpointer=MemorySaver()``.
        """
        graph = StateGraph(
            _FullState,
            input_schema=_InputState,
            output_schema=_OutputState,
        )

        # --- nodes ---
        seed_node = make_seed_node()
        router_node = make_router_node(self._max_rounds)
        dispatch_fn = make_dispatch(self._handler_metas)

        graph.add_node("__seed__", seed_node)
        graph.add_node("__router__", router_node)

        handler_names: list[str] = []
        for meta in self._handler_metas:
            handler_node = make_handler_node(meta)
            graph.add_node(meta.name, handler_node)
            handler_names.append(meta.name)

        # --- edges ---
        graph.add_edge(START, "__seed__")

        # dispatch from seed and from router
        destinations = handler_names + [END]
        graph.add_conditional_edges("__seed__", dispatch_fn, destinations)
        graph.add_conditional_edges("__router__", dispatch_fn, destinations)

        # all handlers fan-in back to router
        for name in handler_names:
            graph.add_edge(name, "__router__")

        return graph.compile(**kwargs)

    def invoke(self, seed_event: Event, **kwargs: Any) -> EventLog:
        """Run the graph synchronously with a seed event.

        Returns an ``EventLog`` containing all events produced during the run.
        """
        compiled = self.compile(**kwargs.pop("compile_kwargs", {}))
        result = compiled.invoke(
            {"events": [seed_event]},
            **kwargs,
        )
        return EventLog(result["events"])

    async def ainvoke(self, seed_event: Event, **kwargs: Any) -> EventLog:
        """Run the graph asynchronously with a seed event."""
        compiled = self.compile(**kwargs.pop("compile_kwargs", {}))
        result = await compiled.ainvoke(
            {"events": [seed_event]},
            **kwargs,
        )
        return EventLog(result["events"])

    def stream(self, seed_event: Event, **kwargs: Any):
        """Stream graph execution. Pass-through to compiled graph's stream.

        Accepts all LangGraph ``stream_mode`` options.
        """
        compiled = self.compile(**kwargs.pop("compile_kwargs", {}))
        return compiled.stream(
            {"events": [seed_event]},
            **kwargs,
        )

    async def astream(self, seed_event: Event, **kwargs: Any):
        """Async stream graph execution."""
        compiled = self.compile(**kwargs.pop("compile_kwargs", {}))
        async for chunk in compiled.astream(
            {"events": [seed_event]},
            **kwargs,
        ):
            yield chunk
