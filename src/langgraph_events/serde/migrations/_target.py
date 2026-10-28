"""The graph and the baseline file that every gate command reads."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from langgraph.checkpoint.memory import MemorySaver

# The module, not the class: this module loads while ``_jsonplus`` is still
# importing ``serde.migrations``, so the class is read at call time.
from langgraph_events.serde import _jsonplus

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from langgraph.checkpoint.base import BaseCheckpointSaver

    from langgraph_events import EventGraph
    from langgraph_events.serde._jsonplus import NamespaceAwareSerde
    from langgraph_events.serde.migrations._core import MigrationSet


class TargetError(Exception):
    """The target cannot give a graph with a ``NamespaceAwareSerde``."""


@dataclass(frozen=True)
class MigrationTarget:
    """One graph and its committed baseline.

    ``graph`` builds the graph on the checkpointer it receives. The gate
    passes an empty in-memory saver, so building reads no store.
    ``baseline`` is the path of the committed baseline file.
    """

    graph: Callable[[BaseCheckpointSaver[Any]], EventGraph]
    baseline: Path

    def snapshot(self) -> Snapshot:
        """What the code holds, built once for every rule.

        Raises:
            TargetError: when the graph has no ``NamespaceAwareSerde``, or
                when a client migration has no valid key.
        """
        graph = self.graph(MemorySaver())
        serde = getattr(graph._checkpointer, "serde", None)
        if not isinstance(serde, _jsonplus.NamespaceAwareSerde):
            raise TargetError(
                f"The graph of the target has the serde {type(serde).__name__}, "
                f"not a NamespaceAwareSerde. Build it with "
                f"EventGraph.from_namespaces(..., checkpointer=saver), or set "
                f"saver.serde to a NamespaceAwareSerde before you build it."
            )
        try:
            migrations = serde.client_migrations()
        except ValueError as error:
            raise TargetError(
                f"A client migration has no valid key: {error}"
            ) from error
        return Snapshot(graph, serde, migrations)


@dataclass(frozen=True)
class Snapshot:
    """The graph, its serde and its client migrations."""

    graph: EventGraph
    serde: NamespaceAwareSerde
    migrations: MigrationSet
