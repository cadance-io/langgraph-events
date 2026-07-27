"""QueryTool — the event log's query surface packaged as one LLM tool.

Framework-agnostic: a frozen dataclass whose shape maps 1:1 onto an Anthropic
tool dict or LangChain's ``StructuredTool.from_function``. Every op delegates
to a public ``Reflection`` query or an index-preserving mirror of an
``EventLog`` query — the tool never grows semantics of its own.
"""

from __future__ import annotations

import difflib
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from langgraph_events._event import (
    _NAMESPACE_REGISTRY,
    Command,
    DomainEvent,
    Event,
    IntegrationEvent,
    SystemEvent,
    _iter_nested_events,
)
from langgraph_events._reflection._text import event_line

if TYPE_CHECKING:
    from collections.abc import Callable

    from langgraph_events._event_log import EventLog
    from langgraph_events._namespace import NamespaceModel
    from langgraph_events._reflection._core import Reflection

_TYPE_OPS = ("filter", "select", "latest", "first", "has", "count", "after", "before")
_INDEX_OPS = ("get", "evidence")
_OPS = ("overview", "list", *_TYPE_OPS, *_INDEX_OPS, "state", "schema")

_BASE_KINDS: tuple[type[Event], ...] = (
    Event,
    Command,
    DomainEvent,
    IntegrationEvent,
    SystemEvent,
)

_DEFAULT_LIMIT = 20


@dataclass(frozen=True)
class QueryTool:
    """A single dispatch tool over one run's event log. Facts only."""

    name: str
    description: str
    parameters: dict[str, Any]
    run: Callable[..., str]


def _model_classes(model: NamespaceModel, log: EventLog) -> list[type[Event]]:
    """Every event class queryable by name: registered namespaces (walked via
    the framework's blessed nested-event iterator), model-only namespaces,
    integration/system events, and anything present in the log."""
    classes: list[type[Event]] = []
    for name, namespace in model.namespaces.items():
        registered = _NAMESPACE_REGISTRY.get(name)
        if registered is not None:
            classes.extend(_iter_nested_events(registered, recurse_commands=True))
        else:
            for command in namespace.commands.values():
                classes.append(command.cls)
                classes.extend(command.outcomes)
            classes.extend(namespace.events)
    classes.extend(model.integration_events)
    classes.extend(model.system_events)
    classes.extend(type(e) for e in log)
    return list(dict.fromkeys(classes))


def _type_vocabulary(classes: list[type[Event]]) -> dict[str, type[Event]]:
    """Name → class map: qualified names always, simple names when unique."""
    mapping: dict[str, type[Event]] = {}
    for cls in classes:
        namespace = getattr(cls, "__namespace__", None)
        qualified = f"{namespace}.{cls.__name__}" if namespace else cls.__name__
        mapping[qualified] = cls
    by_simple: dict[str, set[type[Event]]] = {}
    for cls in classes:
        by_simple.setdefault(cls.__name__, set()).add(cls)
    for name, candidates in by_simple.items():
        if len(candidates) == 1:
            mapping.setdefault(name, next(iter(candidates)))
    for base in _BASE_KINDS:
        mapping.setdefault(base.__name__, base)
    return mapping


def _vocabulary_section(
    classes: list[type[Event]], vocabulary: dict[str, type[Event]]
) -> str:
    """Advertise exactly the names the vocabulary accepts, grouped."""

    def canonical(cls: type[Event]) -> str:
        simple = cls.__name__
        if vocabulary.get(simple) is cls:
            return simple
        namespace = getattr(cls, "__namespace__", None)
        return f"{namespace}.{simple}" if namespace else simple

    groups: dict[str, list[str]] = {}
    for cls in classes:
        group = getattr(cls, "__namespace__", None)
        if group is None:
            if issubclass(cls, IntegrationEvent):
                group = "integration"
            elif issubclass(cls, SystemEvent):
                group = "system"
            else:
                group = "other"
        groups.setdefault(group, []).append(canonical(cls))
    lines = ["event types (grouped by namespace):"]
    lines.extend(f"  {group}: {', '.join(names)}" for group, names in groups.items())
    lines.append(
        "base kinds (match whole categories): "
        + ", ".join(b.__name__ for b in _BASE_KINDS)
    )
    return "\n".join(lines)


_DESCRIPTION_HEADER = """\
Query this run's event log. Returns facts only — correlate and conclude yourself.
Every event is addressed as #<index> (stable root-log positions); pass that
number as the `index` argument to drill down.
ops:
  overview — run shape: counts, seeds, anomalies, status
  list — all events in order (paged: index = offset, limit)
  get(index) — full detail of one event
  filter(type) / select(type) — events matching type (subclass-aware)
  latest(type) / first(type) — newest / oldest match
  has(type) / count(type) — existence / count
  after(type) / before(type) — events after / before the first match
  evidence(index) — all facts on how that event came to be: explicit links,
    owning command, static-edge candidates, downstream face
  state — reducer projections over the log
  schema — the static topology: what can cause what
"""


def _listing(pairs: list[tuple[int, Event]], limit: int) -> str:
    shown = pairs[:limit]
    lines = [event_line(i, e) for i, e in shown]
    remainder = len(pairs) - len(shown)
    if remainder > 0:
        lines.append(f"…and {remainder} more — refine your query")
    return "\n".join(lines) if lines else "no matching events"


def _run_type_op(  # noqa: PLR0911 — flat op dispatch, one return per op
    op: str, log: EventLog, event_type: type[Event], limit: int
) -> str:
    events = log.events
    if op in ("filter", "select"):
        pairs = [(i, e) for i, e in enumerate(events) if isinstance(e, event_type)]
        return _listing(pairs, limit)
    if op == "count":
        return str(log.count(event_type))
    if op == "has":
        return "true" if log.has(event_type) else "false"
    if op == "latest":
        for i in range(len(events) - 1, -1, -1):
            if isinstance(events[i], event_type):
                return event_line(i, events[i])
        return f"no {event_type.__name__} events in this log"
    anchor = next((i for i, e in enumerate(events) if isinstance(e, event_type)), None)
    if anchor is None:
        return f"no {event_type.__name__} events in this log"
    if op == "first":
        return event_line(anchor, events[anchor])
    # after / before — anchored on the first match, root indices preserved
    if op == "after":
        segment = list(enumerate(events[anchor + 1 :], start=anchor + 1))
    else:
        segment = list(enumerate(events[:anchor]))
    return _listing(segment, limit)


def _render_state(state: dict[str, Any]) -> str:
    if not state:
        return "no reducers registered"
    return "\n".join(f"{name}: {value!r}" for name, value in state.items())


def build_tool(reflection: Reflection) -> QueryTool:
    """Build the query_log tool over *reflection*."""
    log = reflection.log
    classes = _model_classes(reflection._model, log)
    vocabulary = _type_vocabulary(classes)

    def run(  # noqa: PLR0911 — flat op dispatch, one return per op
        *,
        op: str = "",
        type: str | None = None,
        index: int | None = None,
        limit: int = _DEFAULT_LIMIT,
    ) -> str:
        if op not in _OPS:
            return f"error: unknown op {op!r}. valid ops: {', '.join(_OPS)}"
        try:
            if op in _TYPE_OPS:
                if type is None:
                    return f"error: op {op!r} requires the `type` argument"
                event_type = vocabulary.get(type)
                if event_type is None:
                    close = difflib.get_close_matches(type, vocabulary, n=3)
                    hint = f" — did you mean: {', '.join(close)}?" if close else ""
                    return (
                        f"error: unknown type {type!r}{hint} "
                        f"valid types: {', '.join(sorted(vocabulary))}"
                    )
                return _run_type_op(op, log, event_type, limit)
            if op in _INDEX_OPS:
                if index is None:
                    return f"error: op {op!r} requires the `index` argument"
                if op == "get":
                    return reflection.event(index)
                return reflection.evidence(index)
            if op == "list":
                pairs = list(enumerate(log))[index or 0 :]
                return _listing(pairs, limit)
            if op == "overview":
                return reflection.overview()
            if op == "state":
                return _render_state(reflection.state())
            return reflection.schema()
        except (IndexError, ValueError, KeyError) as exc:
            return f"error: {exc}"

    description = _DESCRIPTION_HEADER + _vocabulary_section(classes, vocabulary)
    parameters = {
        "type": "object",
        "properties": {
            "op": {"type": "string", "enum": list(_OPS)},
            "type": {"type": "string", "description": "event type name"},
            "index": {"type": "integer", "description": "#<index> of an event"},
            "limit": {"type": "integer", "description": "max listing lines"},
        },
        "required": ["op"],
    }
    return QueryTool(
        name="query_log", description=description, parameters=parameters, run=run
    )
