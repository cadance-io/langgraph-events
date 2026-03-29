"""EventGraph — the main entry point for building event-driven graphs."""

from __future__ import annotations

import typing
import warnings
from typing import TYPE_CHECKING, Any, NamedTuple, TypedDict, cast

from langgraph.graph import END, START, StateGraph

from langgraph_events._custom_event import STATE_SNAPSHOT_EVENT_NAME
from langgraph_events._event import Event, Interrupted, Resumed, Scatter
from langgraph_events._event_log import EventLog
from langgraph_events._handler import HandlerMeta, extract_handler_meta
from langgraph_events._internal import (
    _BASE_FIELDS,
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
    from langgraph.store.base import BaseStore

    from langgraph_events._reducer import BaseReducer


class ReturnInfo(NamedTuple):
    """Parsed handler return-type annotation."""

    event_types: list[type[Event]]
    scatter_types: list[type[Event]]
    has_scatter: bool
    has_interrupted: bool
    has_annotation: bool


def _parse_return_types(fn: Callable[..., Any]) -> ReturnInfo:
    """Parse handler return annotation into a ``ReturnInfo``."""
    try:
        hints = typing.get_type_hints(fn)
    except Exception as exc:
        warnings.warn(
            f"Failed to resolve return type hints for handler {fn.__qualname__!r}; "
            f"treating as unannotated for topology parsing. ({exc})",
            stacklevel=3,
        )
        hints = {}

    return_hint = hints.get("return")
    if return_hint is None:
        return ReturnInfo([], [], False, False, False)

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

    return ReturnInfo(event_types, scatter_types, has_scatter, has_interrupted, True)


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


class LLMToken(NamedTuple):
    """A text delta from an LLM invocation during handler execution."""

    run_id: str
    content: str


class LLMStreamEnd(NamedTuple):
    """Signals an LLM stream completed."""

    run_id: str
    message_id: str | None  # LangChain AIMessage.id for dedup


class CustomEventFrame(NamedTuple):
    """A custom event frame emitted from LangGraph's v2 stream API."""

    name: str
    data: Any


class StateSnapshotFrame(NamedTuple):
    """A typed state snapshot custom frame emitted from v2 stream events."""

    data: dict[str, Any]


StreamItem = (
    Event
    | StreamFrame
    | LLMToken
    | LLMStreamEnd
    | CustomEventFrame
    | StateSnapshotFrame
)


def _coerce_snapshot_data(data: Any) -> dict[str, Any]:
    if isinstance(data, dict):
        return cast("dict[str, Any]", data)
    return {}


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
        reducers: list[BaseReducer] | None = None,
        checkpointer: Any = None,
        store: BaseStore | None = None,
    ) -> None:
        if not handlers:
            raise ValueError("EventGraph requires at least one handler")

        self._max_rounds = max_rounds
        self._checkpointer = checkpointer
        self._store = store
        self._reducers: dict[str, BaseReducer] = {r.name: r for r in (reducers or [])}
        conflicts = set(self._reducers.keys()) & set(_BASE_FIELDS.keys())
        if conflicts:
            raise ValueError(
                f"Reducer name(s) {conflicts} conflict with reserved state fields"
            )
        self._handler_metas: list[HandlerMeta] = []
        self._compiled_graph: CompiledStateGraph | None = None

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
                    config_param=meta.config_param,
                    store_param=meta.store_param,
                )
            else:
                seen_names[meta.name] = 1
            self._handler_metas.append(meta)

        self._return_info: dict[str, ReturnInfo] = {}
        for meta in self._handler_metas:
            info = _parse_return_types(meta.fn)
            self._return_info[meta.name] = info
            if info.has_annotation and any(t is Event for t in info.event_types):
                raise ValueError(
                    f"Handler '{meta.name}' return type includes base 'Event'. "
                    f"Use specific types (e.g., TypeA | TypeB)."
                )

    @staticmethod
    def _mermaid_footer_entry(
        meta: HandlerMeta, has_scatter: bool, solid: list[str], dashed: list[str]
    ) -> tuple[str, str] | None:
        """Return ``(kind, entry)`` if handler belongs in footer, else None."""
        if not solid and not dashed:
            subscribed = ", ".join(t.__name__ for t in meta.event_types)
            kind = "scatter" if has_scatter else "side_effect"
            return kind, f"{meta.name} ({subscribed})"
        return None

    def mermaid(self) -> str:  # noqa: PLR0912
        """Return a Mermaid flowchart showing event correlation.

        Events are nodes, handlers are edge labels.
        Seed events (no incoming edges) get a thick entry edge.
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
            info = self._return_info[meta.name]
            label = meta.name

            if info.has_interrupted:
                any_produces_interrupted = True
            if any(issubclass(t, Resumed) for t in meta.event_types):
                any_subscribes_resumed = True

            solid_targets = [t.__name__ for t in info.event_types]
            dashed_targets = [t.__name__ for t in info.scatter_types]
            if not info.has_annotation:
                solid_targets.append("?")

            footer = self._mermaid_footer_entry(
                meta, info.has_scatter, solid_targets, dashed_targets
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
            lines.append("    classDef entry fill:none,stroke:none,color:none")
            for i, seed in enumerate(seed_events):
                lines.append(f"    _e{i}_[ ]:::entry ==> {seed}")

        lines.extend(edge_lines)

        if scatter_handlers:
            lines.append(f"%% Scatter handlers: {', '.join(scatter_handlers)}")
        if side_effects:
            lines.append(f"%% Side-effect handlers: {', '.join(side_effects)}")

        return "\n".join(lines)

    @property
    def reducer_names(self) -> frozenset[str]:
        """The names of all registered reducers."""
        return frozenset(self._reducers.keys())

    @property
    def compiled(self) -> CompiledStateGraph:
        """The underlying LangGraph ``CompiledStateGraph``.

        This is the bridge to full LangGraph when you need features
        beyond the EventGraph API: subgraph composition, custom
        streaming modes, direct state access, or advanced checkpointer
        workflows.

        The instance is compiled lazily on first access and cached.
        """
        return self._compile()

    def _compile(self) -> CompiledStateGraph:
        """Compile into a LangGraph ``CompiledStateGraph`` (internal)."""
        if self._compiled_graph is not None:
            return self._compiled_graph

        # Dynamic state schema with per-reducer channels
        state_schema = build_state_schema(self._reducers)

        # Always include reducer channels — filtering is an output concern
        out_schema: Any = _OutputState
        if self._reducers:
            reducer_fields: dict[str, Any] = {"events": list[Event]}
            for name, r in self._reducers.items():
                reducer_fields[name] = r.output_type()
            _OutputWithReducers = TypedDict("_OutputWithReducers", reducer_fields)  # type: ignore[misc]
            out_schema = _OutputWithReducers

        graph: StateGraph[Any] = StateGraph(
            state_schema,
            input_schema=_InputState,  # type: ignore[arg-type]
            output_schema=out_schema,
        )

        # --- nodes ---
        seed_node = make_seed_node(reducers=self._reducers)
        router_node = make_router_node(self._max_rounds)
        dispatch_fn = make_dispatch(self._handler_metas)

        graph.add_node("__seed__", cast("Any", seed_node))
        graph.add_node("__router__", cast("Any", router_node))

        handler_names: list[str] = []
        for meta in self._handler_metas:
            handler_node = make_handler_node(meta, reducers=self._reducers)
            graph.add_node(meta.name, cast("Any", handler_node))
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

        compile_kwargs: dict[str, Any] = {}
        if self._checkpointer is not None:
            compile_kwargs["checkpointer"] = self._checkpointer
        if self._store is not None:
            compile_kwargs["store"] = self._store
        self._compiled_graph = graph.compile(**compile_kwargs)

        # EventGraph controls termination via max_rounds. Set recursion_limit
        # high enough that it never fires before max_rounds does.
        # Each round is at most 1 router + all handlers.
        n_handlers = len(self._handler_metas)
        needed = self._max_rounds * (n_handlers + 1) + 1
        existing = (self._compiled_graph.config or {}).get("recursion_limit", 25)
        self._compiled_graph.config = {
            **(self._compiled_graph.config or {}),
            "recursion_limit": max(needed, existing),
        }

        return self._compiled_graph

    def _require_checkpointer(self, method: str) -> None:
        if self._checkpointer is None:
            raise ValueError(f"{method}() requires a checkpointer")

    @staticmethod
    def _prepare_input(seed: Event | list[Event]) -> dict[str, Any]:
        """Build the input dict from a seed event or list of events."""
        if isinstance(seed, list):
            return {"events": seed}
        return {"events": [seed]}

    def _run(self, inp: Any, **kwargs: Any) -> EventLog:
        compiled = self._compile()
        result = compiled.invoke(inp, **kwargs)
        return EventLog._from_owned(result["events"])

    async def _arun(self, inp: Any, **kwargs: Any) -> EventLog:
        compiled = self._compile()
        result = await compiled.ainvoke(inp, **kwargs)
        return EventLog._from_owned(result["events"])

    def invoke(self, seed: Event | list[Event], **kwargs: Any) -> EventLog:
        """Run the graph synchronously with one or more seed events.

        Args:
            seed: A single event or list of events to start the graph.

        Returns an ``EventLog`` containing all events produced during the run.
        """
        return self._run(self._prepare_input(seed), **kwargs)

    async def ainvoke(self, seed: Event | list[Event], **kwargs: Any) -> EventLog:
        """Run the graph asynchronously with one or more seed events."""
        return await self._arun(self._prepare_input(seed), **kwargs)

    def resume(self, value: Event, **kwargs: Any) -> EventLog:
        """Resume an interrupted graph with a domain event.

        The event is auto-dispatched (handlers subscribed to its type fire),
        then a ``Resumed`` event is created alongside it.
        """
        self._require_checkpointer("resume")
        from langgraph.types import Command  # noqa: PLC0415

        return self._run(Command(resume=value), **kwargs)

    async def aresume(self, value: Event, **kwargs: Any) -> EventLog:
        """Async version of resume().

        The event is auto-dispatched (handlers subscribed to its type fire),
        then a ``Resumed`` event is created alongside it.
        """
        self._require_checkpointer("aresume")
        from langgraph.types import Command  # noqa: PLC0415

        return await self._arun(Command(resume=value), **kwargs)

    def get_state(self, config: Any) -> GraphState:
        """Get event-level state of a checkpointed thread."""
        self._require_checkpointer("get_state")
        compiled = self._compile()
        snapshot = compiled.get_state(config)
        all_events = snapshot.values.get("events", [])
        log = EventLog(all_events)
        is_interrupted = bool(snapshot.next)
        return GraphState(
            events=log,
            is_interrupted=is_interrupted,
            interrupted=log.latest(Interrupted) if is_interrupted else None,
        )

    # --- LLM token helpers ---

    @staticmethod
    def _extract_text_content(chunk: Any) -> str:
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

    def _update_reducer_state(
        self,
        state: dict[str, Any],
        event: Event,
        reducer_names: list[str],
    ) -> None:
        """Incrementally update reducer state with a single new event."""
        for name in reducer_names:
            r = self._reducers[name]
            contribution = r.collect([event])
            if r.has_contributions(contribution):
                reducer_fn = getattr(r, "reducer", None)
                if callable(reducer_fn):
                    state[name] = reducer_fn(state[name], contribution)
                else:
                    state[name] = contribution

    # --- High-level event streaming ---

    def _resolve_reducer_names(self, include_reducers: bool | list[str]) -> list[str]:
        """Return reducer names to include, or empty list for disabled."""
        if include_reducers is True:
            return list(self._reducers.keys())
        if include_reducers:  # non-empty list
            unknown = set(include_reducers) - set(self._reducers.keys())
            if unknown:
                warnings.warn(
                    f"Unknown reducer name(s) {unknown} in include_reducers; "
                    f"available: {set(self._reducers.keys())}",
                    stacklevel=2,
                )
            return [n for n in include_reducers if n in self._reducers]
        return []

    @staticmethod
    def _events_from_chunk(chunk: Any, seen: set[int]) -> list[Event]:
        """Extract unseen events from an updates-mode stream chunk."""
        events: list[Event] = []
        if isinstance(chunk, dict):
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
            name: state.get(name, self._reducers[name].empty) for name in reducer_names
        }
        return len(all_events), [
            StreamFrame(event=e, reducers=reducers) for e in new_events
        ]

    async def _astream_v2(  # noqa: PLR0912
        self,
        inp: Any,
        seeds: list[Event],
        *,
        reducer_names: list[str],
        include_llm_tokens: bool,
        include_custom_events: bool,
        **kwargs: Any,
    ) -> AsyncIterator[StreamItem]:
        """Stream events using LangGraph's v2 event API with LLM token support."""
        compiled = self._compile()

        # Initialize incremental reducer state from seeds
        reducer_state: dict[str, Any] = {}
        for name in reducer_names:
            reducer_state[name] = self._reducers[name].seed(seeds)

        # Yield seed events
        for s in seeds:
            if reducer_names:
                yield StreamFrame(event=s, reducers=dict(reducer_state))
            else:
                yield s

        seen: set[int] = set()

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
                delta = self._extract_text_content(chunk)
                if run_id and delta:
                    yield LLMToken(run_id=run_id, content=delta)
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
            for event in self._events_from_chunk(chunk, seen):
                if reducer_names:
                    self._update_reducer_state(reducer_state, event, reducer_names)
                    yield StreamFrame(event=event, reducers=dict(reducer_state))
                else:
                    yield event

    def _stream_sync(
        self,
        inp: Any,
        seeds: list[Event],
        reducer_names: list[str],
        **kwargs: Any,
    ) -> Iterator[Event | StreamFrame]:
        """Shared sync streaming core for stream_events/stream_resume."""
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
        inp: Any,
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
        """
        inp = self._prepare_input(seed)
        kwargs.pop("stream_mode", None)
        reducer_names = self._resolve_reducer_names(include_reducers)
        yield from self._stream_sync(inp, inp["events"], reducer_names, **kwargs)

    def stream_resume(
        self,
        value: Event,
        *,
        include_reducers: bool | list[str] = False,
        **kwargs: Any,
    ) -> Iterator[Event | StreamFrame]:
        """Yield events produced when resuming an interrupted graph.

        Streaming equivalent of ``resume()`` — accepts a domain ``Event``,
        yields events as they are produced.  The ``Command(resume=value)``
        stays internal, exactly like ``resume()``.

        Args:
            value: The domain event to resume with.
            include_reducers: When truthy, yields ``StreamFrame`` tuples
                instead of bare events.  Pass ``True`` for all reducers or
                a list of reducer names for selective inclusion.
        """
        self._require_checkpointer("stream_resume")
        from langgraph.types import Command  # noqa: PLC0415

        kwargs.pop("stream_mode", None)
        reducer_names = self._resolve_reducer_names(include_reducers)
        yield from self._stream_sync(Command(resume=value), [], reducer_names, **kwargs)

    async def astream_resume(
        self,
        value: Event,
        *,
        include_reducers: bool | list[str] = False,
        include_llm_tokens: bool = False,
        include_custom_events: bool = False,
        **kwargs: Any,
    ) -> AsyncIterator[StreamItem]:
        """Async version of ``stream_resume()``.

        Streaming equivalent of ``aresume()`` — accepts a domain ``Event``,
        yields events as they are produced.

        Args:
            value: The domain event to resume with.
            include_reducers: When truthy, yields ``StreamFrame`` tuples
                instead of bare events.  Pass ``True`` for all reducers or
                a list of reducer names for selective inclusion.
            include_llm_tokens: When True, yields ``LLMToken`` and
                ``LLMStreamEnd`` frames for LLM token-level streaming.
            include_custom_events: When True, yields ``CustomEventFrame``
                and ``StateSnapshotFrame`` frames for ``on_custom_event`` payloads
                from LangGraph.
        """
        self._require_checkpointer("astream_resume")
        from langgraph.types import Command  # noqa: PLC0415

        inp: Any = Command(resume=value)
        kwargs.pop("stream_mode", None)
        reducer_names = self._resolve_reducer_names(include_reducers)

        delegate = (
            self._astream_v2(
                inp,
                [],
                reducer_names=reducer_names,
                include_llm_tokens=include_llm_tokens,
                include_custom_events=include_custom_events,
                **kwargs,
            )
            if include_llm_tokens or include_custom_events
            else self._astream_core(inp, [], reducer_names, **kwargs)
        )
        async for item in delegate:
            yield item

    async def astream_events(
        self,
        seed: Event | list[Event],
        *,
        include_reducers: bool | list[str] = False,
        include_llm_tokens: bool = False,
        include_custom_events: bool = False,
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
        """
        inp = self._prepare_input(seed)
        seeds = inp["events"]
        kwargs.pop("stream_mode", None)
        reducer_names = self._resolve_reducer_names(include_reducers)

        delegate = (
            self._astream_v2(
                inp,
                seeds,
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
