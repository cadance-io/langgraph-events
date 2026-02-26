"""EventGraph — the main entry point for building event-driven graphs."""

from __future__ import annotations

import typing
from typing import TYPE_CHECKING, Any, NamedTuple, TypedDict

from langgraph.graph import END, START, StateGraph

from langgraph_events._event import Event, Interrupted, Resumed, Scatter
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

    from langgraph_events._reducer import Reducer


def _parse_return_types(
    fn: Callable[..., Any],
) -> tuple[list[type[Event]], list[type[Event]], bool, bool, bool]:
    """Parse handler return annotation.

    Returns a 5-tuple:
    ``(event_types, scatter_types, has_scatter, has_interrupted, has_annotation)``.

    Returns:
        event_types: Event subclasses the handler can produce (solid edges).
        scatter_types: Event subclasses produced via Scatter (dashed edges).
        has_scatter: Whether the handler returns bare (untyped) Scatter.
        has_interrupted: Whether the handler returns Interrupted.
        has_annotation: Whether the handler has a return type annotation at all.
    """
    try:
        hints = typing.get_type_hints(fn)
    except Exception:
        hints = {}

    return_hint = hints.get("return")
    if return_hint is None:
        return [], [], False, False, False

    # If the top-level hint is Scatter[X], wrap it so the loop sees Scatter[X]
    # as a single candidate (otherwise get_args returns the type params).
    origin = typing.get_origin(return_hint)
    if origin is Scatter:
        candidates = (return_hint,)
    else:
        args = typing.get_args(return_hint)
        candidates = args if args else (return_hint,)

    event_types: list[type[Event]] = []
    scatter_types: list[type[Event]] = []
    has_scatter = False
    has_interrupted = False
    for arg in candidates:
        if arg is type(None):
            continue
        if arg is Scatter:
            has_scatter = True
        elif typing.get_origin(arg) is Scatter:
            # Parameterized Scatter[X] — extract event types
            for scatter_arg in typing.get_args(arg):
                if isinstance(scatter_arg, type) and issubclass(scatter_arg, Event):
                    scatter_types.append(scatter_arg)
        elif isinstance(arg, type) and issubclass(arg, Event):
            if issubclass(arg, Interrupted):
                has_interrupted = True
            event_types.append(arg)

    return event_types, scatter_types, has_scatter, has_interrupted, True


class StreamFrame(NamedTuple):
    """A yielded frame from ``stream_events()`` when ``include_reducers`` is enabled.

    Contains the event and a snapshot of all requested reducer values at
    the point the event was produced.
    """

    event: Event
    reducers: dict[str, list[Any]]


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
        self._compiled_cache: dict[frozenset[tuple[str, Any]], CompiledStateGraph] = {}

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

    @staticmethod
    def _mermaid_footer_entry(
        meta: HandlerMeta, has_scatter: bool, solid: list[str], dashed: list[str]
    ) -> tuple[str, str] | None:
        """Return ``(kind, entry)`` if handler belongs in footer, else None."""
        if has_scatter and not solid and not dashed:
            subscribed = ", ".join(t.__name__ for t in meta.event_types)
            return "scatter", f"{meta.fn.__name__} ({subscribed})"
        if not solid and not dashed:
            subscribed = ", ".join(t.__name__ for t in meta.event_types)
            return "side_effect", f"{meta.fn.__name__} ({subscribed})"
        return None

    def mermaid(self) -> str:  # noqa: PLR0912
        """Return a Mermaid flowchart showing event correlation.

        Events are nodes, handlers are edge labels.
        Seed events (no incoming edges) get a stadium shape and distinct fill.
        Typed ``Scatter[X]`` produces dashed edges; bare ``Scatter`` goes to a
        comment footer.  Side-effect handlers (-> None) are listed in a footer.
        If any handler produces ``Interrupted`` and any subscribes to
        ``Resumed``, a dashed framework edge connects them.
        """
        lines = ["graph LR"]
        edge_lines: list[str] = []
        side_effects: list[str] = []
        scatter_handlers: list[str] = []
        any_produces_interrupted = False
        any_subscribes_resumed = False
        all_sources: set[str] = set()
        all_targets: set[str] = set()

        for meta in self._handler_metas:
            event_types, scatter_types, has_scatter, has_interrupted, has_annotation = (
                _parse_return_types(meta.fn)
            )
            label = meta.fn.__name__

            if has_interrupted:
                any_produces_interrupted = True
            if any(issubclass(t, Resumed) for t in meta.event_types):
                any_subscribes_resumed = True

            solid_targets = [t.__name__ for t in event_types]
            dashed_targets = [t.__name__ for t in scatter_types]
            if not has_annotation:
                solid_targets.append("?")

            footer = self._mermaid_footer_entry(
                meta, has_scatter, solid_targets, dashed_targets
            )
            if footer is not None:
                (scatter_handlers if footer[0] == "scatter" else side_effects).append(
                    footer[1]
                )
                continue

            for src_type in meta.event_types:
                src = src_type.__name__
                all_sources.add(src)
                for target in solid_targets:
                    all_targets.add(target)
                    edge_lines.append(f"    {src} -->|{label}| {target}")
                for target in dashed_targets:
                    all_targets.add(target)
                    edge_lines.append(f"    {src} -.->|{label}| {target}")

        if any_produces_interrupted and any_subscribes_resumed:
            edge_lines.append("    Interrupted -.-> Resumed")
            all_sources.add("Interrupted")
            all_targets.add("Interrupted")
            all_targets.add("Resumed")

        # Seed events: sources that never appear as targets
        seed_events = sorted(all_sources - all_targets)
        if seed_events:
            lines.append("    classDef seed fill:#dae8fc,stroke:#6c8ebf")
            for seed in seed_events:
                lines.append(f"    {seed}([{seed}]):::seed")

        lines.extend(edge_lines)

        if scatter_handlers:
            lines.append(f"%% Scatter handlers: {', '.join(scatter_handlers)}")
        if side_effects:
            lines.append(f"%% Side-effect handlers: {', '.join(side_effects)}")

        return "\n".join(lines)

    def compile(
        self,
        *,
        _output_reducer_names: frozenset[str] | None = None,
        **kwargs: Any,
    ) -> CompiledStateGraph:
        """Compile into a LangGraph ``CompiledStateGraph``.

        All keyword arguments are forwarded to ``StateGraph.compile()``,
        e.g. ``checkpointer=MemorySaver()``.

        Results are cached — calling with the same kwargs returns the same
        compiled graph.
        """
        cache_key = frozenset(
            [
                *((k, id(v)) for k, v in kwargs.items()),
                ("_output_reducer_names", _output_reducer_names),
            ]
        )
        if cache_key in self._compiled_cache:
            return self._compiled_cache[cache_key]

        # Dynamic state schema with per-reducer channels
        state_schema = build_state_schema(self._reducers)

        # Build output schema — include reducer channels when requested
        out_schema: Any = _OutputState
        if _output_reducer_names:
            from langgraph_events._event import Event as _Event  # noqa: PLC0415

            reducer_fields: dict[str, Any] = {"events": list[_Event]}
            for name in _output_reducer_names:
                if name in self._reducers:
                    reducer_fields[f"_r_{name}"] = list
            out_schema = TypedDict("_OutputWithReducers", reducer_fields)  # type: ignore[misc,no-redef]

        graph: StateGraph[Any] = StateGraph(
            state_schema,
            input_schema=_InputState,  # type: ignore[arg-type]
            output_schema=out_schema,
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

    def invoke(self, seed: Event | list[Event], **kwargs: Any) -> EventLog:
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

    async def ainvoke(self, seed: Event | list[Event], **kwargs: Any) -> EventLog:
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

    def _resolve_reducer_names(self, include_reducers: bool | list[str]) -> list[str]:
        """Return reducer names to include, or empty list for disabled."""
        if include_reducers is True:
            return list(self._reducers.keys())
        if include_reducers:  # non-empty list
            return [n for n in include_reducers if n in self._reducers]
        return []

    @staticmethod
    def _frames_from_values(
        state: dict[str, Any],
        prev_count: int,
        reducer_names: list[str],
    ) -> tuple[int, list[StreamFrame]]:
        """Extract new events and reducer snapshots from a values-mode state."""
        all_events: list[Event] = state.get("events", [])
        new_events = all_events[prev_count:]
        if not new_events:
            return prev_count, []
        reducers = {name: state.get(f"_r_{name}", []) for name in reducer_names}
        return len(all_events), [
            StreamFrame(event=e, reducers=reducers) for e in new_events
        ]

    def stream_events(
        self,
        seed: Event | list[Event],
        *,
        include_reducers: bool | list[str] = False,
        **kwargs: Any,
    ) -> Iterator[Event | StreamFrame]:
        """Yield individual events as they are produced during graph execution.

        Higher-level alternative to ``stream()`` — yields ``Event`` objects
        directly instead of raw LangGraph state dicts.  Seed events are
        yielded first, followed by events produced by handlers.

        Args:
            seed: A single event or list of events to start the graph.
            include_reducers: When truthy, yields ``StreamFrame`` tuples
                instead of bare events.  Pass ``True`` for all reducers or
                a list of reducer names for selective inclusion.
        """
        seeds = self._normalize_seed(seed)
        kwargs.pop("stream_mode", None)
        compile_kwargs = kwargs.pop("compile_kwargs", {})

        reducer_names = self._resolve_reducer_names(include_reducers)
        if not reducer_names:
            compiled = self.compile(**compile_kwargs)
            yield from seeds
            seen: set[int] = set()
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
        else:
            compiled = self.compile(
                _output_reducer_names=frozenset(reducer_names),
                **compile_kwargs,
            )
            prev_count = 0
            first = True
            for state in compiled.stream(
                {"events": seeds},
                stream_mode="values",
                **kwargs,
            ):
                if first:
                    # Skip initial input state — reducers not yet populated
                    first = False
                    continue
                prev_count, frames = self._frames_from_values(
                    state, prev_count, reducer_names
                )
                yield from frames

    async def astream_events(
        self,
        seed: Event | list[Event],
        *,
        include_reducers: bool | list[str] = False,
        **kwargs: Any,
    ) -> AsyncIterator[Event | StreamFrame]:
        """Async version of ``stream_events()``.

        Args:
            seed: A single event or list of events to start the graph.
            include_reducers: When truthy, yields ``StreamFrame`` tuples
                instead of bare events.  Pass ``True`` for all reducers or
                a list of reducer names for selective inclusion.
        """
        seeds = self._normalize_seed(seed)
        kwargs.pop("stream_mode", None)
        compile_kwargs = kwargs.pop("compile_kwargs", {})

        reducer_names = self._resolve_reducer_names(include_reducers)
        if not reducer_names:
            compiled = self.compile(**compile_kwargs)
            for s in seeds:
                yield s
            seen: set[int] = set()
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
        else:
            compiled = self.compile(
                _output_reducer_names=frozenset(reducer_names),
                **compile_kwargs,
            )
            prev_count = 0
            first = True
            async for state in compiled.astream(
                {"events": seeds},
                stream_mode="values",
                **kwargs,
            ):
                if first:
                    # Skip initial input state — reducers not yet populated
                    first = False
                    continue
                prev_count, frames = self._frames_from_values(
                    state, prev_count, reducer_names
                )
                for frame in frames:
                    yield frame
