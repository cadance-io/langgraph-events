"""JSON serialization of a :class:`NamespaceModel`."""

from __future__ import annotations

from typing import Any

from langgraph_events._namespace._model import NamespaceModel

SCHEMA_VERSION = "3"
"""Top-level ``schema_version`` string stamped on ``to_dict()`` output.

Bumped when fields are removed, renamed, or change meaning. Additions
don't bump. Consumers should treat an unexpected major as opaque data.

- "3" (#108): per-command ``handlers`` lists external handler names only;
  an inline-handled command reports ``[]`` (the inline handler stays in
  the top-level ``command_handlers`` with ``inline: true``).
"""


def _qn(cls: type) -> str:
    return cls.__qualname__


def _encode_retry(policy: Any) -> dict[str, Any] | None:
    """Encode a ``RetryPolicy``, or ``None`` when the handler declares none."""
    if policy is None:
        return None
    return {
        "max_attempts": policy.max_attempts,
        "base_delay": policy.base_delay,
        "max_delay": policy.max_delay,
        "strategy": policy.strategy,
        "jitter": policy.jitter,
        "on": [_qn(t) for t in policy.on],
        "respect_retry_after": policy.respect_retry_after,
        "observe": policy.observe,
    }


def _encode_reaction(r: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "name": r.name,
        "produces": [_qn(t) for t in r.produces],
        "scatters": [_qn(t) for t in r.scatters],
        "raises": [_qn(t) for t in r.raises],
        "invariants": [_qn(t) for t in r.invariants],
        "field_matchers": [list(fm) for fm in r.field_matchers],
        "retry": _encode_retry(r.retry),
        "side_effect": r.side_effect,
        "has_annotation": r.has_annotation,
    }
    if isinstance(r, NamespaceModel.CommandHandler):
        base["kind"] = "command_handler"
        base["commands"] = [_qn(t) for t in r.commands]
        base["inline"] = r.inline
    else:
        base["kind"] = "policy"
        base["subscribes"] = [_qn(t) for t in r.subscribes]
    return base


def encode_model(d: NamespaceModel) -> dict[str, Any]:
    namespaces: dict[str, Any] = {}
    for namespace_name, dom in d.namespaces.items():
        namespaces[namespace_name] = {
            "name": dom.name,
            "commands": {
                cmd_name: {
                    "type": _qn(cmd.cls),
                    "outcomes": [_qn(t) for t in cmd.outcomes],
                    "handlers": list(cmd.handlers),
                }
                for cmd_name, cmd in dom.commands.items()
            },
            "events": [_qn(t) for t in dom.events],
        }
    return {
        "schema_version": SCHEMA_VERSION,
        "namespaces": namespaces,
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
                "causation": e.causation,
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
