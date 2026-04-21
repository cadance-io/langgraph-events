"""JSON serialization of a :class:`DomainModel`."""

from __future__ import annotations

from typing import Any

from langgraph_events._domain._model import DomainModel


def _qn(cls: type) -> str:
    return cls.__qualname__


def _encode_reaction(r: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "name": r.name,
        "produces": [_qn(t) for t in r.produces],
        "scatters": [_qn(t) for t in r.scatters],
        "raises": [_qn(t) for t in r.raises],
        "invariants": [_qn(t) for t in r.invariants],
        "field_matchers": [list(fm) for fm in r.field_matchers],
        "side_effect": r.side_effect,
        "has_annotation": r.has_annotation,
        "has_untyped_scatter": r.has_untyped_scatter,
    }
    if isinstance(r, DomainModel.CommandHandler):
        base["kind"] = "command_handler"
        base["commands"] = [_qn(t) for t in r.commands]
        base["inline"] = r.inline
    else:
        base["kind"] = "policy"
        base["subscribes"] = [_qn(t) for t in r.subscribes]
    return base


def encode_model(d: DomainModel) -> dict[str, Any]:
    aggregates: dict[str, Any] = {}
    for agg_name, agg in d.aggregates.items():
        aggregates[agg_name] = {
            "name": agg.name,
            "commands": {
                cmd_name: {
                    "type": _qn(cmd.cls),
                    "outcomes": [_qn(t) for t in cmd.outcomes],
                    "handlers": list(cmd.handlers),
                }
                for cmd_name, cmd in agg.commands.items()
            },
            "events": [_qn(t) for t in agg.events],
        }
    return {
        "aggregates": aggregates,
        "integration_events": [_qn(t) for t in d.integration_events],
        "system_events": [_qn(t) for t in d.system_events],
        "command_handlers": [_encode_reaction(r) for r in d.command_handlers],
        "policies": [_encode_reaction(r) for r in d.policies],
        "edges": [
            {
                "source": _qn(e.source),
                "via": e.via,
                "target": _qn(e.target),
                "kind": e.kind,
            }
            for e in d.edges
        ],
        "seeds": [_qn(t) for t in d.seeds],
        "invariants": [
            {
                "cls": _qn(inv.cls),
                "commands": [_qn(c) for c in inv.commands],
                "declared_by": list(inv.declared_by),
                "reactors": list(inv.reactors),
            }
            for inv in d.invariants
        ],
    }
