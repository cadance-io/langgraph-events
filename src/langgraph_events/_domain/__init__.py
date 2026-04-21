"""Domain model introspection — a code-derived DDD snapshot of an EventGraph.

``EventGraph.domain()`` returns a :class:`DomainModel` instance. Everything
else — human-readable text, Mermaid diagrams, JSON export — hangs off that
single object. See the module docstring on :class:`DomainModel` for the full
surface.
"""

from __future__ import annotations

from langgraph_events._domain._model import DomainModel, View

__all__ = ["DomainModel", "View"]
