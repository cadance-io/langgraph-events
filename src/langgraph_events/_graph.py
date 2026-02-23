"""EventGraph — the main entry point for building event-driven graphs."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from langgraph.graph import END, START, StateGraph

from langgraph_events._event_log import EventLog
from langgraph_events._handler import HandlerMeta, extract_handler_meta
from langgraph_events._internal import (
    _InputState,
    _OutputState,
    build_state_schema,
    make_dispatch,
    make_handler_node,
    make_router_node,
    make_seed_node,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable, Iterator

    from langgraph.graph.state import CompiledStateGraph

    from langgraph_events._event import Event
    from langgraph_events._reducer import Reducer


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
    """

    def __init__(
        self,
        handlers: list[Callable[..., Any]],
        *,
        max_rounds: int = 100,
        reducers: list[Reducer] | None = None,
    ) -> None:
        if not handlers:
            raise ValueError("EventGraph requires at least one handler")

        self._max_rounds = max_rounds
        self._reducers: dict[str, Reducer] = {r.name: r for r in (reducers or [])}
        self._handler_metas: list[HandlerMeta] = []
        self._compiled_cache: dict[frozenset[tuple[str, int]], CompiledStateGraph] = {}

        reducer_names = frozenset(self._reducers.keys())
        seen_names: dict[str, int] = {}
        for fn in handlers:
            meta = extract_handler_meta(fn, reducer_names=reducer_names)
            # Deduplicate node names
            if meta.name in seen_names:
                seen_names[meta.name] += 1
                meta = HandlerMeta(
                    name=f"{meta.name}_{seen_names[meta.name]}",
                    fn=meta.fn,
                    event_types=meta.event_types,
                    log_param=meta.log_param,
                    is_async=meta.is_async,
                    reducer_params=meta.reducer_params,
                )
            else:
                seen_names[meta.name] = 1
            self._handler_metas.append(meta)

    def compile(self, **kwargs: Any) -> CompiledStateGraph:
        """Compile into a LangGraph ``CompiledStateGraph``.

        All keyword arguments are forwarded to ``StateGraph.compile()``,
        e.g. ``checkpointer=MemorySaver()``.

        Results are cached — calling with the same kwargs returns the same
        compiled graph.
        """
        cache_key = frozenset((k, id(v)) for k, v in kwargs.items())
        if cache_key in self._compiled_cache:
            return self._compiled_cache[cache_key]

        # Dynamic state schema with per-reducer channels
        state_schema = build_state_schema(self._reducers)

        graph: StateGraph[Any] = StateGraph(
            state_schema,
            input_schema=_InputState,  # type: ignore[arg-type]
            output_schema=_OutputState,  # type: ignore[arg-type]
        )

        # --- nodes ---
        seed_node = make_seed_node(reducers=self._reducers)
        router_node = make_router_node(self._max_rounds)
        dispatch_fn = make_dispatch(self._handler_metas)

        graph.add_node("__seed__", seed_node)  # type: ignore[call-overload]
        graph.add_node("__router__", router_node)  # type: ignore[call-overload]

        handler_names: list[str] = []
        for meta in self._handler_metas:
            handler_node = make_handler_node(meta, reducers=self._reducers)
            graph.add_node(meta.name, handler_node)
            handler_names.append(meta.name)

        # --- edges ---
        graph.add_edge(START, "__seed__")

        # dispatch from seed and from router
        destinations = [*handler_names, END]
        graph.add_conditional_edges("__seed__", dispatch_fn, destinations)
        graph.add_conditional_edges("__router__", dispatch_fn, destinations)

        # all handlers fan-in back to router
        for name in handler_names:
            graph.add_edge(name, "__router__")

        compiled = graph.compile(**kwargs)
        self._compiled_cache[cache_key] = compiled
        return compiled

    @staticmethod
    def _normalize_seed(seed: Event | list[Event]) -> list[Event]:
        """Normalize seed input to a list of events."""
        if isinstance(seed, list):
            return seed
        return [seed]

    def invoke(
        self, seed: Event | list[Event], **kwargs: Any
    ) -> EventLog:
        """Run the graph synchronously with one or more seed events.

        Args:
            seed: A single event or list of events to start the graph.

        Returns an ``EventLog`` containing all events produced during the run.
        """
        compiled = self.compile(**kwargs.pop("compile_kwargs", {}))
        result = compiled.invoke(
            {"events": self._normalize_seed(seed)},
            **kwargs,
        )
        return EventLog(result["events"])

    async def ainvoke(
        self, seed: Event | list[Event], **kwargs: Any
    ) -> EventLog:
        """Run the graph asynchronously with one or more seed events."""
        compiled = self.compile(**kwargs.pop("compile_kwargs", {}))
        result = await compiled.ainvoke(
            {"events": self._normalize_seed(seed)},
            **kwargs,
        )
        return EventLog(result["events"])

    def stream(
        self, seed: Event | list[Event], **kwargs: Any
    ) -> Iterator[dict[str, Any] | Any]:
        """Stream graph execution. Pass-through to compiled graph's stream.

        Accepts all LangGraph ``stream_mode`` options.
        """
        compiled = self.compile(**kwargs.pop("compile_kwargs", {}))
        return compiled.stream(
            {"events": self._normalize_seed(seed)},
            **kwargs,
        )

    async def astream(
        self, seed: Event | list[Event], **kwargs: Any
    ) -> AsyncIterator[dict[str, Any] | Any]:
        """Async stream graph execution."""
        compiled = self.compile(**kwargs.pop("compile_kwargs", {}))
        async for chunk in compiled.astream(
            {"events": self._normalize_seed(seed)},
            **kwargs,
        ):
            yield chunk

    # --- High-level event streaming ---

    def stream_events(
        self, seed: Event | list[Event], **kwargs: Any
    ) -> Iterator[Event]:
        """Yield individual events as they are produced during graph execution.

        Higher-level alternative to ``stream()`` — yields ``Event`` objects
        directly instead of raw LangGraph state dicts.  Seed events are
        yielded first, followed by events produced by handlers.
        """
        seeds = self._normalize_seed(seed)
        yield from seeds

        kwargs.pop("stream_mode", None)
        compiled = self.compile(**kwargs.pop("compile_kwargs", {}))
        seen = set()
        for chunk in compiled.stream(
            {"events": seeds},
            stream_mode="updates",
            **kwargs,
        ):
            if isinstance(chunk, dict):
                for node_output in chunk.values():
                    if isinstance(node_output, dict):
                        for event in node_output.get("events", []):
                            eid = id(event)
                            if eid not in seen:
                                seen.add(eid)
                                yield event

    async def astream_events(
        self, seed: Event | list[Event], **kwargs: Any
    ) -> AsyncIterator[Event]:
        """Async version of ``stream_events()``."""
        seeds = self._normalize_seed(seed)
        for s in seeds:
            yield s

        kwargs.pop("stream_mode", None)
        compiled = self.compile(**kwargs.pop("compile_kwargs", {}))
        seen = set()
        async for chunk in compiled.astream(
            {"events": seeds},
            stream_mode="updates",
            **kwargs,
        ):
            if isinstance(chunk, dict):
                for node_output in chunk.values():
                    if isinstance(node_output, dict):
                        for event in node_output.get("events", []):
                            eid = id(event)
                            if eid not in seen:
                                seen.add(eid)
                                yield event
