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
accepts ``bool | list[str] | _Drop`` — see :func:`drop_reducers` for the
ergonomic deny-list builder.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from langgraph_events._internal import _BASE_FIELDS

if TYPE_CHECKING:
    from collections.abc import Iterable

_FRAMEWORK_INTERNAL_KEYS: frozenset[str] = frozenset(_BASE_FIELDS)
_DEDICATED_EVENT_KEYS: frozenset[str] = frozenset({"messages"})
_RESERVED_KEYS: frozenset[str] = _FRAMEWORK_INTERNAL_KEYS | _DEDICATED_EVENT_KEYS


def default_state_projection(reducers: dict[str, Any]) -> dict[str, Any]:
    """Strip framework-internal channels and dedicated AG-UI keys.

    Always applied before any user-supplied ``include_reducers`` filter.
    Module-internal: devs interact via ``include_reducers``.
    """
    return {k: v for k, v in reducers.items() if k not in _RESERVED_KEYS}


class _Drop:
    """Internal marker carrying the names ``drop_reducers`` was asked to hide.

    The adapter resolves this against the graph's reducer set at construction
    time, producing a concrete ``list[str]`` allow-list — so the runtime path
    is always ``bool | list[str]``, never an opaque callable.
    """

    __slots__ = ("excluded",)

    def __init__(self, excluded: Iterable[str]) -> None:
        self.excluded: frozenset[str] = frozenset(excluded)


def drop_reducers(*names: str) -> _Drop:
    """Build a deny-list spec — keep every reducer except the named ones.

    Sugar over the ``list[str]`` allow-list form, resolved against the graph's
    reducer set at adapter construction time::

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
    return _Drop(excluded=names)
