"""Text renderers for Reflection output — compact, #index-addressed, deterministic."""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

from langgraph_events._event import (
    Command,
    DomainEvent,
    Event,
    Halted,
    HandlerRaised,
    IntegrationEvent,
    Interrupted,
    InvariantViolated,
    Resumed,
    RunPaused,
    SystemEvent,
)

if TYPE_CHECKING:
    from langgraph_events._event_log import EventLog
    from langgraph_events._namespace import NamespaceModel

_MAX_VALUE_LEN = 40

_ANOMALY_TYPES = (HandlerRaised, InvariantViolated, Halted, Interrupted, RunPaused)


def event_line(index: int, event: Event) -> str:
    """Render one event as ``#3 Placed(order_id='o1')`` with truncated values."""
    parts = []
    for f in dataclasses.fields(event):  # type: ignore[arg-type]
        value = repr(getattr(event, f.name))
        if len(value) > _MAX_VALUE_LEN:
            value = value[: _MAX_VALUE_LEN - 3] + "..."
        parts.append(f"{f.name}={value}")
    return f"#{index} {type(event).__name__}({', '.join(parts)})"


def kind_of(event: Event) -> str:
    """Classify an event into its taxonomy kind."""
    if isinstance(event, Command):
        return "command"
    if isinstance(event, DomainEvent):
        return "domain"
    if isinstance(event, IntegrationEvent):
        return "integration"
    if isinstance(event, SystemEvent):
        return "system"
    return "event"


def run_status(log: EventLog) -> str:
    """Derive the run's terminal status from the log — a fact, not a judgment."""
    if log.has(Halted):
        return "halted"
    interrupted = log.latest(Interrupted)
    if interrupted is not None:
        events = log.events
        after_interrupt = events[events.index(interrupted) :]
        if not any(isinstance(e, Resumed) for e in after_interrupt):
            return "interrupted"
    if log and isinstance(log[-1], RunPaused):
        return "paused"
    return "completed"


def seed_indices(log: EventLog, model: NamespaceModel) -> list[int]:
    """Indices of the leading events matching the model's seed types."""
    if not model.seeds:
        return []
    seed_types = tuple(model.seeds)
    indices = []
    for i, event in enumerate(log):
        if not isinstance(event, seed_types):
            break
        indices.append(i)
    return indices


def anomaly_lines(log: EventLog) -> list[str]:
    """Indexed lines for every anomaly event, in log order."""
    return [
        event_line(i, e) for i, e in enumerate(log) if isinstance(e, _ANOMALY_TYPES)
    ]


def _counts(log: EventLog) -> tuple[dict[str, int], dict[str, int]]:
    by_kind: dict[str, int] = {}
    by_namespace: dict[str, int] = {}
    for event in log:
        kind = kind_of(event)
        by_kind[kind] = by_kind.get(kind, 0) + 1
        namespace = getattr(type(event), "__namespace__", None)
        if namespace is not None:
            by_namespace[namespace] = by_namespace.get(namespace, 0) + 1
    return by_kind, by_namespace


def render_overview(log: EventLog, model: NamespaceModel) -> str:
    """The overview op: totals, counts, seeds, anomalies, status."""
    by_kind, by_namespace = _counts(log)
    lines = [f"run overview: {len(log)} events, status: {run_status(log)}"]
    if by_kind:
        lines.append(
            "kinds: " + ", ".join(f"{k}: {n}" for k, n in sorted(by_kind.items()))
        )
    if by_namespace:
        lines.append(
            "namespaces: "
            + ", ".join(f"{ns}: {n}" for ns, n in sorted(by_namespace.items()))
        )
    seeds = seed_indices(log, model)
    if seeds:
        lines.append("seeds:")
        lines.extend(f"  {event_line(i, log[i])}" for i in seeds)
    anomalies = anomaly_lines(log)
    if anomalies:
        lines.append("anomalies:")
        lines.extend(f"  {line}" for line in anomalies)
    else:
        lines.append("anomalies: none")
    return "\n".join(lines)


def render_event_detail(index: int, log: EventLog) -> str:
    """The get op: one event, every field on its own line, plus taxonomy facts."""
    event = log[index]
    if index < 0:
        index = len(log) + index
    lines = [f"#{index} {type(event).__name__}"]
    for f in dataclasses.fields(event):  # type: ignore[arg-type]
        lines.append(f"  {f.name}: {getattr(event, f.name)!r}")
    lines.append(f"  kind: {kind_of(event)}")
    namespace = getattr(type(event), "__namespace__", None)
    if namespace is not None:
        lines.append(f"  namespace: {namespace}")
    command = getattr(type(event), "__command__", None)
    if command is not None:
        lines.append(f"  command: {command.__name__}")
    return "\n".join(lines)


def render_context(log: EventLog, model: NamespaceModel, *, tail: int) -> str:
    """The prompt context card: bounded shape of the run + tool pointer."""
    total = len(log)
    lines = [
        f"[run context] {total} events, status: {run_status(log)} — "
        "use the query_log tool to inspect the event log"
    ]
    anomalies = anomaly_lines(log)
    if anomalies:
        lines.append("anomalies:")
        lines.extend(f"  {line}" for line in anomalies)
    shown = min(tail, total)
    lines.append(f"recent events (last {shown} of {total}):")
    start = total - shown
    lines.extend(f"  {event_line(start + i, e)}" for i, e in enumerate(log[start:]))
    return "\n".join(lines)
