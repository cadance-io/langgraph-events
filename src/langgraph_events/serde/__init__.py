"""Namespace-aware checkpoint serde — opt-in alternative to LangGraph's default.

Default ``JsonPlusSerializer`` keys dataclass identity by ``cls.__name__``,
which is leaf-only for nested events (``Persona.Approve.Approved.__name__``
is ``"Approved"``). Two namespaces that share a leaf event name therefore
collide on checkpoint round-trip. ``NamespaceAwareSerde`` keys ``Event``
subclasses by ``__qualname__`` instead, walking the attribute chain on the
defining module to revive the right class.

Usage::

    from langgraph.checkpoint.memory import MemorySaver
    from langgraph_events.serde import NamespaceAwareSerde

    graph = EventGraph(
        handlers=[...],
        checkpointer=MemorySaver(serde=NamespaceAwareSerde()),
    )
"""

from langgraph_events.serde._jsonplus import NamespaceAwareSerde

__all__ = ["NamespaceAwareSerde"]
