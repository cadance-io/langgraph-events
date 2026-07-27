"""The evidence op — a verdict-free join of log facts around one event.

Lists explicit instance links, the owning command, matching static edges with
their candidate instances, and the forward face. Never selects a cause: the
querying agent correlates; this module only joins.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

from langgraph_events._event import Event
from langgraph_events._reflection._text import event_line

if TYPE_CHECKING:
    from langgraph_events._event_log import EventLog
    from langgraph_events._namespace import NamespaceModel


def find_index(
    log: EventLog, event: Event, *, equality_before: int | None = None
) -> int | None:
    """Locate *event* in *log* — identity first, then latest equal instance.

    ``equality_before`` bounds the equality fallback (identity search stays
    global): a *backward* link must never resolve to a position at or after
    its effect, even when equal instances repeat later in the log.
    """
    events = log.events
    for i in range(len(events) - 1, -1, -1):
        if events[i] is event:
            return i
    stop = len(events) if equality_before is None else equality_before
    for i in range(stop - 1, -1, -1):
        if type(events[i]) is type(event) and events[i] == event:
            return i
    return None


def _indices(indices: list[int]) -> str:
    return ", ".join(f"#{i}" for i in indices) if indices else "none in this log"


def _explicit_link_lines(index: int, event: Event, log: EventLog) -> list[str]:
    """One line per event-valued field, resolved to its position in the log."""
    lines = []
    for f in dataclasses.fields(event):  # type: ignore[arg-type]
        value = getattr(event, f.name)
        if not isinstance(value, Event):
            continue
        j = find_index(log, value, equality_before=index)
        if j is None or j == index:
            continue
        marker = "" if log[j] is value else " (equality match)"
        lines.append(f"  {f.name}: #{j} {type(value).__name__}{marker}")
    return lines


def _backward_edge_lines(
    index: int, event: Event, log: EventLog, model: NamespaceModel
) -> tuple[list[str], bool]:
    """Edges targeting this event's type, each with its preceding instances."""
    lines = []
    any_instances = False
    for edge in model.edges:
        if not isinstance(event, edge.target):
            continue
        preceding = [j for j in range(index) if isinstance(log[j], edge.source)]
        any_instances = any_instances or bool(preceding)
        lines.append(
            f"  {edge.source.__name__} --{edge.via} [{edge.causation or edge.kind}]"
            f"--> {type(event).__name__}: {_indices(preceding)}"
        )
    return lines, any_instances


def _forward_edge_lines(
    index: int, event: Event, log: EventLog, model: NamespaceModel
) -> list[str]:
    """Edges sourced at this event's type, each with its subsequent instances."""
    lines = []
    for edge in model.edges:
        if not isinstance(event, edge.source):
            continue
        subsequent = [
            j for j in range(index + 1, len(log)) if isinstance(log[j], edge.target)
        ]
        lines.append(
            f"  {type(event).__name__} --{edge.via} [{edge.causation or edge.kind}]"
            f"--> {edge.target.__name__}: {_indices(subsequent)}"
        )
    return lines


def render_evidence(index: int, log: EventLog, model: NamespaceModel) -> str:
    """Render every deterministic fact bearing on how ``log[index]`` came to be.

    *index* must be canonical (0-based, in range) — callers go through
    ``Reflection._resolve_index``.
    """
    event = log[index]
    lines = [f"evidence for {event_line(index, event)}"]

    explicit = _explicit_link_lines(index, event, log)
    if explicit:
        lines.append("explicit links (event fields):")
        lines.extend(explicit)

    command = getattr(type(event), "__command__", None)
    preceding_commands: list[int] = []
    if command is not None:
        preceding_commands = [j for j in range(index) if isinstance(log[j], command)]
        lines.append(
            f"owning command: {command.__name__} — "
            f"preceding instances: {_indices(preceding_commands)}"
        )

    causes, any_edge_instances = _backward_edge_lines(index, event, log, model)
    if causes:
        lines.append("possible causes (static edges, candidate instances):")
        lines.extend(causes)

    if not explicit and not preceding_commands and not any_edge_instances:
        lines.append("no backward evidence in this log (seed or externally injected)")

    effects = _forward_edge_lines(index, event, log, model)
    if effects:
        lines.append("possible effects (static edges, subsequent instances):")
        lines.extend(effects)

    return "\n".join(lines)
