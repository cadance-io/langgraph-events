"""The storage format of the ``causes`` state channel.

``causes[i]`` describes ``events[i]``. This module owns the format: the
alias of one entry, and :func:`resolve`, which turns the stored entries into
absolute ones. ``EventLog`` and ``rewrite_store`` read the channel only
through :func:`resolve`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypeAlias

if TYPE_CHECKING:
    from collections.abc import Sequence

CauseEntry: TypeAlias = tuple[int | None, str] | None
"""One entry of the ``causes`` channel: ``(source, via)``, or ``None``.

A source ``>= 0`` is a log index. A source ``< 0`` counts back from its own
event: ``-1`` is the event just before it. An ``Interrupted`` block uses it,
because its handler cannot know the absolute index of ``Interrupted``.
A source ``None`` means the handler is known but its source is not: a
handler that resumed on a thread saved before causes existed, or a source
that ``rewrite_store(drop=...)`` removed. ``None`` as the whole entry means
no handler produced the event: a seed.
"""


def resolve(
    events: Sequence[Any], causes: Sequence[Any] | None
) -> tuple[list[CauseEntry], int]:
    """The absolute entry of each event in *events*, and ``known_from``.

    The channels align from the end. A checkpoint saved before causes
    existed holds fewer causes than events. Each event below ``known_from``
    then has an unknown cause, and its entry is ``None``. Raises
    ``RuntimeError`` when a writer left the channels out of step: more
    causes than events, or a source that is not an earlier event.
    """
    stored = list(causes or ())
    known_from = len(events) - len(stored)
    if known_from < 0:
        raise RuntimeError(
            f"the causes channel holds {len(stored)} entries for {len(events)} "
            f"events. A writer to events left causes out of step. This is a "
            f"framework bug."
        )
    entries: list[CauseEntry] = [None] * known_from
    for position, entry in enumerate(stored, start=known_from):
        entries.append(None if entry is None else _absolute(position, entry))
    return entries, known_from


def _absolute(position: int, entry: Sequence[Any]) -> tuple[int | None, str]:
    source, via = entry
    if source is None:
        return (None, via)
    absolute = position + source if source < 0 else source
    if not 0 <= absolute < position:
        raise RuntimeError(
            f"the stored cause of event #{position} points to #{absolute}, which "
            f"is not an earlier event. A writer to events left causes out of "
            f"step. This is a framework bug."
        )
    return (absolute, via)
