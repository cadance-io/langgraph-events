"""Domain-pattern smell detection for namespace models.

Surfaces ``DomainPatternWarning`` when 2+ events in a single namespace fan
out (via 2+ distinct reactor handlers) to identical target sets — usually a
sign that a shared abstraction was missed (a common base event or a single
reactor on a common subscription would collapse them).

Detection runs once at ``_build_domain_model`` time. Silence via the
standard Python warnings machinery::

    import warnings
    from langgraph_events import DomainPatternWarning
    warnings.filterwarnings("ignore", category=DomainPatternWarning)
"""

from __future__ import annotations

import warnings
from collections import defaultdict
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from langgraph_events._event import Event
    from langgraph_events._namespace._model import NamespaceModel


class DomainPatternWarning(UserWarning):
    """Multiple events in one namespace fan out to identical target sets.

    Often indicates a missing shared abstraction. Silence via::

        import warnings
        from langgraph_events import DomainPatternWarning
        warnings.filterwarnings("ignore", category=DomainPatternWarning)
    """


class CommandChainWarning(UserWarning):
    """An inline ``Command.handle()`` emits another ``Command``.

    A ``chain``-causation edge. Usually the intent should either split into
    a reactor on the first command's outcome, or collapse into one larger
    command. Silence via::

        import warnings
        from langgraph_events import CommandChainWarning
        warnings.filterwarnings("ignore", category=CommandChainWarning)
    """


def _qualname(cls: type) -> str:
    """Return ``cls.__qualname__`` with ``<locals>.`` strip-out for cleanliness."""
    return cls.__qualname__.replace("<locals>.", "")


def emit_command_chain_warnings(model: NamespaceModel) -> None:
    """Warn once per ``causation == "chain"`` edge.

    ``Command → Command`` via an inline handler: the producing command
    reaches straight into another command instead of emitting a fact and
    letting a reactor decide. Cheap, total scan over edges already carrying
    the causal role.
    """
    for edge in model.edges:
        if edge.causation != "chain":
            continue
        warnings.warn(
            f"{_qualname(edge.source)}.{edge.via}() emits "
            f"{_qualname(edge.target)}, another Command "
            f"(causation='chain'). Prefer emitting a DomainEvent and "
            f"reacting to it, or collapse the two intents into one command.",
            category=CommandChainWarning,
            stacklevel=5,
        )


def emit_domain_pattern_warnings(model: NamespaceModel) -> None:
    """Scan *model* for identical-target-set fanouts and warn for each.

    A pattern fires when, within one namespace, 2+ distinct source events
    (commands or domain events) each fan out to the *same* set of targets
    via 2+ distinct reactor handlers. Subset / superset overlaps don't
    qualify — only exact set equality.
    """
    # (namespace, source_cls, handler_name) -> set of target classes
    pair_targets: dict[tuple[str, type[Event], str], set[type[Event]]] = defaultdict(
        set
    )
    for edge in model.edges:
        if edge.kind not in ("solid", "scatter"):
            continue
        ns = getattr(edge.source, "__namespace__", None)
        if ns is None:
            continue
        pair_targets[(ns, edge.source, edge.via)].add(edge.target)

    # Group (source, handler) pairs by (namespace, frozen target_set).
    grouped: dict[tuple[str, frozenset[type[Event]]], list[tuple[type[Event], str]]] = (
        defaultdict(list)
    )
    for (ns, source, handler), targets in pair_targets.items():
        if len(targets) < 2:
            continue
        grouped[(ns, frozenset(targets))].append((source, handler))

    for (ns, target_set), pairs in grouped.items():
        sources = {s for s, _ in pairs}
        handlers = {h for _, h in pairs}
        if len(sources) < 2 or len(handlers) < 2:
            continue

        sorted_targets = sorted(target_set, key=_qualname)
        sorted_pairs = sorted(pairs, key=lambda sh: (_qualname(sh[0]), sh[1]))
        targets_repr = "{" + ", ".join(_qualname(t) for t in sorted_targets) + "}"
        pairs_repr = ", ".join(f"{_qualname(s)} ({h})" for s, h in sorted_pairs)
        msg = (
            f"namespace {ns!r} has {len(sources)} events fanning out to identical "
            f"target set {targets_repr}: {pairs_repr}. Consider unifying these "
            "reactors or extracting a shared base event."
        )
        warnings.warn(msg, category=DomainPatternWarning, stacklevel=5)
