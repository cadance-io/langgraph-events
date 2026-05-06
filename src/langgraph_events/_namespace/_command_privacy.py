"""Static check that a Command's outcomes are nested under it.

Two symmetric rules, applied at ``EventGraph`` construction:

- A ``Command.handle()`` may only emit ``DomainEvent``s nested under that
  same Command (or under a parent Command, for inheritance).
- An ``@on(...)`` reactor must not emit a ``DomainEvent`` nested inside any
  Command — those are private to their owning Command's ``handle()``.

Violations raise :class:`CommandPrivacyError`. Framework events
(``Interrupted``, ``IntegrationEvent``, ``HandlerRaised`` etc.) carry no
``__command__`` semantics and are exempt.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from langgraph_events._event import Command, DomainEvent

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    from langgraph_events._graph import ReturnInfo
    from langgraph_events._handler import HandlerMeta


class CommandPrivacyError(TypeError):
    """A handler emitted a ``DomainEvent`` it isn't allowed to emit.

    Either a ``Command.handle()`` returned an event not nested under the
    Command, or a reactor emitted an event private to some Command. Subclasses
    :class:`TypeError` to match other namespace-level invariants the library
    raises (duplicate namespace name, ``Outcomes`` drift).
    """


def enforce_command_privacy(
    handler_metas: Iterable[HandlerMeta],
    return_info: Mapping[str, ReturnInfo],
) -> None:
    """Validate every handler's emitted events against the privacy rule.

    ``return_info`` is the ``EventGraph._return_info`` map — already populated
    from ``_parse_return_types`` during graph construction. Reusing it avoids
    re-resolving type hints (which would re-emit warnings for hints that fail
    to resolve, since the resolution cache only stores successes).
    """
    for meta in handler_metas:
        info = return_info[meta.name]
        owner: type[Command] | None = getattr(meta.fn, "_inline_command", None)
        for event_cls in (*info.event_types, *info.scatter_types):
            if not (isinstance(event_cls, type) and issubclass(event_cls, DomainEvent)):
                continue
            event_owner: type[Command] | None = getattr(event_cls, "__command__", None)
            if owner is not None:
                # Inline Command.handle: must emit only events nested under
                # itself (or under a parent Command, for inheritance).
                if event_owner is None or not issubclass(owner, event_owner):
                    raise CommandPrivacyError(
                        _unnested_outcome_msg(owner, event_cls, event_owner)
                    )
            elif event_owner is not None:
                # Any non-inline handler: forbidden from emitting Command-
                # private events. Only the owning Command's inline handle()
                # may produce them — colocate the emission there.
                raise CommandPrivacyError(
                    _reactor_leak_msg(meta, event_cls, event_owner)
                )


def _qualname(cls: type) -> str:
    return cls.__qualname__.replace("<locals>.", "")


def _unnested_outcome_msg(
    owner: type[Command],
    event_cls: type[DomainEvent],
    event_owner: type[Command] | None,
) -> str:
    if event_owner is None:
        return (
            f"{_qualname(owner)}.handle() returns {_qualname(event_cls)}, but "
            f"{_qualname(event_cls)} is declared at the namespace level. Nest "
            f"it inside {_qualname(owner)} (or stop emitting it from handle())."
        )
    return (
        f"{_qualname(owner)}.handle() returns {_qualname(event_cls)}, which is "
        f"private to {_qualname(event_owner)}, not {_qualname(owner)}. Move it "
        f"under {_qualname(owner)} (or stop emitting it from handle())."
    )


def _reactor_leak_msg(
    meta: HandlerMeta,
    event_cls: type[DomainEvent],
    event_owner: type[Command],
) -> str:
    return (
        f"Reactor {meta.name!r} emits {_qualname(event_cls)}, which is private "
        f"to {_qualname(event_owner)}. Only {_qualname(event_owner)}.handle() "
        f"may emit it."
    )
