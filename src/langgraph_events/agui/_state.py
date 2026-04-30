"""State projection: deciding which reducer state crosses the AG-UI boundary.

Two layers of stripping always run on every snapshot, in both directions:

1. **Framework internals** — channels managed by EventGraph itself
   (``events``, ``_cursor``, ``_pending``, ``_round``).  Never visible to
   clients.  Derived from :data:`langgraph_events._internal._BASE_FIELDS`
   so adding a new internal channel propagates automatically.
2. **Dedicated AG-UI keys** — reducer keys driven by purpose-built AG-UI
   events (e.g. ``messages`` ships via MESSAGES_SNAPSHOT, not in
   STATE_SNAPSHOT).

User-facing control is via ``AGUIAdapter(include_reducers=...)`` which
accepts ``bool | list[str]``.
"""

from __future__ import annotations

from typing import Any

from langgraph_events._internal import _BASE_FIELDS

_FRAMEWORK_INTERNAL_KEYS: frozenset[str] = frozenset(_BASE_FIELDS)
_DEDICATED_EVENT_KEYS: frozenset[str] = frozenset({"messages"})
_RESERVED_KEYS: frozenset[str] = _FRAMEWORK_INTERNAL_KEYS | _DEDICATED_EVENT_KEYS


def _default_state_projection(reducers: dict[str, Any]) -> dict[str, Any]:
    """Strip framework-internal channels and dedicated AG-UI keys.

    Always applied before any user-supplied ``include_reducers`` filter.
    Module-internal: devs interact via ``include_reducers``.
    """
    return {k: v for k, v in reducers.items() if k not in _RESERVED_KEYS}
