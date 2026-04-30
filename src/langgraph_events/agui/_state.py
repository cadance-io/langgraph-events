"""State projection: deciding which reducer state crosses the AG-UI boundary.

Two layers of stripping always run before any user-supplied projection:

1. **Framework internals** — channels managed by EventGraph itself
   (``events``, ``_cursor``, ``_pending``, ``_round``).  Never visible to
   clients.  Derived from :data:`langgraph_events._internal._BASE_FIELDS`
   so adding a new internal channel propagates automatically.
2. **Dedicated AG-UI keys** — reducer keys driven by purpose-built AG-UI
   events (e.g. ``messages`` ships via MESSAGES_SNAPSHOT, not in
   STATE_SNAPSHOT).

The user-facing :class:`StateProjector` callable, accepted by
``AGUIAdapter(include_reducers=...)``, runs against the post-stripped dict.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from langgraph_events._internal import _BASE_FIELDS

StateProjector = Callable[[dict[str, Any]], dict[str, Any]]
"""A projection from raw reducer state to client-facing state.

Receives a dict of reducer-name → value (already stripped of framework-internal
channels and dedicated AG-UI keys) and returns the dict to ship to the client.
Used to hide internal reducers, redact sensitive fields, rename keys, etc.

**Symmetric application.**  ``AGUIAdapter`` invokes the projector in *both*
directions: outbound for ``StateSnapshotEvent``, and inbound for
``RunAgentInput.state`` echo into ``FrontendStateMutated``.  Filter-style
projectors (``drop_reducers(...)``, list form) work cleanly in both.
*Transformation* projectors (e.g. PII redaction that adds new keys) also run
inbound — if you want a key like ``redacted_user`` to appear *only* outbound,
gate it on the input shape or split the logic.
"""

_FRAMEWORK_INTERNAL_KEYS: frozenset[str] = frozenset(_BASE_FIELDS)
_DEDICATED_EVENT_KEYS: frozenset[str] = frozenset({"messages"})
_RESERVED_KEYS: frozenset[str] = _FRAMEWORK_INTERNAL_KEYS | _DEDICATED_EVENT_KEYS


def default_state_projection(reducers: dict[str, Any]) -> dict[str, Any]:
    """Strip framework-internal channels and dedicated AG-UI keys.

    Always applied before any user-supplied :class:`StateProjector`.
    Module-internal: devs interact via ``include_reducers`` and never need to
    call this themselves.
    """
    return {k: v for k, v in reducers.items() if k not in _RESERVED_KEYS}


def drop_reducers(*names: str) -> StateProjector:
    """Build a deny-list projection — keep everything except the named keys.

    Convenience for the common "hide a few internal reducers" case::

        from langgraph_events.agui import drop_reducers

        AGUIAdapter(
            graph=g,
            seed_factory=...,
            include_reducers=drop_reducers("debug_count", "scratch"),
        )

    Framework-internal channels (``events``, ``_cursor``, …) and dedicated
    AG-UI keys (``messages``) are stripped automatically and don't need to
    be listed.  Names that don't match any reducer are silently ignored.
    """
    excluded = frozenset(names)

    def projector(reducers: dict[str, Any]) -> dict[str, Any]:
        return {k: v for k, v in reducers.items() if k not in excluded}

    return projector


def _identity(reducers: dict[str, Any]) -> dict[str, Any]:
    return reducers


def _empty(_reducers: dict[str, Any]) -> dict[str, Any]:
    return {}


def _normalize(spec: bool | list[str] | StateProjector) -> StateProjector:
    """Resolve any ``include_reducers`` shape into a single :class:`StateProjector`.

    The adapter calls this once at construction time so the runtime hot path
    is a single function call.  Raises :class:`TypeError` for malformed input
    (a callable, list, ``True``, or ``False`` — anything else is rejected).
    """
    if spec is True:
        return _identity
    if spec is False:
        return _empty
    if isinstance(spec, list):
        allowed = frozenset(spec)
        return lambda reducers: {k: v for k, v in reducers.items() if k in allowed}
    if callable(spec):
        return spec
    raise TypeError(
        f"include_reducers must be bool | list[str] | StateProjector, "
        f"got {type(spec).__name__}"
    )
