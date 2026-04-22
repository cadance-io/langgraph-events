"""Namespace model introspection — a code-derived DDD snapshot of an EventGraph.

``EventGraph.namespaces()`` returns a :class:`NamespaceModel` instance. Everything
else — human-readable text, Mermaid diagrams, JSON export — hangs off that
single object. See the module docstring on :class:`NamespaceModel` for the full
surface.
"""

from __future__ import annotations

from langgraph_events._namespace._model import NamespaceModel, View

__all__ = ["NamespaceModel", "View"]
