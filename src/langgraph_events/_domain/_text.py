"""Human-readable text rendering of a :class:`DomainModel`."""

from __future__ import annotations

from langgraph_events._domain._model import DomainModel, _event_label, _node_class


def _command_annotations(
    d: DomainModel, cmd_cls: type
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    """Collect dedup'd raises + invariant names + scatter targets from every
    command handler subscribed to *cmd_cls*."""
    raises: list[str] = []
    invariants: list[str] = []
    scatters: list[str] = []
    for ch in d.command_handlers:
        if cmd_cls not in ch.commands:
            continue
        for exc in ch.raises:
            if exc.__name__ not in raises:
                raises.append(exc.__name__)
        for inv in ch.invariants:
            label = inv.__name__
            if label not in invariants:
                invariants.append(label)
        for tgt in ch.scatters:
            label = f"Scatter[{_event_label(tgt)}]"
            if label not in scatters:
                scatters.append(label)
        if ch.has_untyped_scatter and "Scatter" not in scatters:
            scatters.append("Scatter")
    return tuple(raises), tuple(invariants), tuple(scatters)


def _policy_targets(p: DomainModel.Policy) -> str:
    """Comma-joined produces + scatter annotations, or empty string."""
    parts = [_event_label(t) for t in p.produces]
    for tgt in p.scatters:
        parts.append(f"Scatter[{_event_label(tgt)}]")
    if p.has_untyped_scatter:
        parts.append("Scatter")
    return ", ".join(parts)


def _render_taxonomy_lines(  # noqa: PLR0912
    d: DomainModel, *, include_handlers: bool
) -> list[str]:
    """Shared aggregate/integration/system block used by both text views.

    ``include_handlers`` adds ``  (handlers: ...)`` suffixes + ``raises`` +
    ``[invariant: …]`` annotations on commands in the choreography view;
    structure view omits those flow concerns but still marks free-standing
    ``Halted`` events with a ``[Halted]`` tag.
    """
    lines: list[str] = []
    if d.aggregates:
        lines.append("Aggregates:")
        for agg_name, agg in d.aggregates.items():
            lines.append(f"  {agg_name}")
            for cmd_name, cmd in agg.commands.items():
                suffix_parts: list[str] = []
                if include_handlers:
                    if cmd.handlers:
                        suffix_parts.append(f"handlers: {', '.join(cmd.handlers)}")
                    raises, invariants, scatters = _command_annotations(d, cmd.cls)
                    if scatters:
                        suffix_parts.append(f"scatters {', '.join(scatters)}")
                    if raises:
                        suffix_parts.append(f"raises {', '.join(raises)}")
                    if invariants:
                        suffix_parts.append(f"invariant: {', '.join(invariants)}")
                suffix = f"  ({'; '.join(suffix_parts)})" if suffix_parts else ""
                lines.append(f"    Command: {cmd_name}{suffix}")
                for outcome in cmd.outcomes:
                    lines.append(f"      → {_event_label(outcome)}")
            for event in agg.events:
                halt_tag = "  [Halted]" if _node_class(event) == "halt" else ""
                lines.append(f"    Event: {_event_label(event)}{halt_tag}")
    if d.integration_events:
        lines.append("Integration events:")
        for int_ev in d.integration_events:
            lines.append(f"  {_event_label(int_ev)}")
    if d.system_events:
        lines.append("System events:")
        for sys_ev in d.system_events:
            lines.append(f"  {_event_label(sys_ev)}")
    if d.invariants:
        lines.append("Invariants:")
        for inv in d.invariants:
            on_cmds = ", ".join(_event_label(c) for c in inv.commands)
            annotations = [f"on {on_cmds}"] if on_cmds else []
            if include_handlers and inv.reactors:
                annotations.append(f"reacted by: {', '.join(inv.reactors)}")
            suffix = f"  ({'; '.join(annotations)})" if annotations else ""
            lines.append(f"  {inv.cls.__name__}{suffix}")
    return lines


def render_text_structure(d: DomainModel) -> str:
    return "\n".join(_render_taxonomy_lines(d, include_handlers=False))


def render_text_choreography(d: DomainModel) -> str:
    lines = _render_taxonomy_lines(d, include_handlers=True)
    if d.policies:
        lines.append("Policies:")
        for p in d.policies:
            subs = ", ".join(_event_label(t) for t in p.subscribes)
            targets = _policy_targets(p)
            flow = f"{subs} → {targets}" if targets else subs
            annotations: list[str] = []
            if p.raises:
                annotations.append(f"raises {', '.join(e.__name__ for e in p.raises)}")
            if p.invariants:
                annotations.append(
                    f"invariant: {', '.join(i.__name__ for i in p.invariants)}"
                )
            if p.side_effect:
                annotations.append("side-effect")
            tail = f"  [{'; '.join(annotations)}]" if annotations else ""
            lines.append(f"  {p.name}  ({flow}){tail}")
    framework = [e for e in d.edges if e.kind == "framework"]
    if framework:
        lines.append("Framework:")
        for e in framework:
            lines.append(f"  {_event_label(e.source)} → {_event_label(e.target)}")
    if d.seeds:
        lines.append("Seed events:")
        for s in d.seeds:
            lines.append(f"  {_event_label(s)}")
    return "\n".join(lines)
