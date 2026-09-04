"""Apply-side migration: plan the rewrite of one thread's live checkpoint.

``EventGraph.plan_rewrite()`` and ``rewrite_store()`` walk each thread's
latest checkpoint. The serde's read path already applies every rename,
transform, split and fill, so a checkpoint read through ``get_tuple()``
and written back through ``put()`` lands under live identities and live
field shapes. This module owns the pure step in between: decide whether
the thread needs a write, drop the stored events the caller named, and
build the checkpoint the graph will store. It never touches a store.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from langgraph_events._event import Event, Resumed

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Sequence

    from langgraph.checkpoint.base import Checkpoint, CheckpointTuple

    from langgraph_events.serde._jsonplus import (
        NamespaceAwareSerde,
        ReadRecord,
        UnrevivedIdentity,
    )

_INTERRUPT_CHANNEL = "__interrupt__"
"""The pending-write channel the rewrite writes back in place."""

_ERROR_CHANNEL = "__error__"
"""A failed task's write. The task runs again on the next invoke."""

_LOG_CHANNELS = ("events", "_pending")
"""The two channels ``drop`` filters. ``_cursor`` indexes ``events``."""

RewriteStatus = Literal["rewrite", "unchanged", "refused"]


@dataclass(frozen=True)
class ThreadRewrite:
    """One thread's line in a :class:`RewriteReport`.

    ``migrated`` lists each ``(stored, live)`` identity pair the migration
    table rewrote, as ``"module:qualname"`` strings, deduped in read
    order. ``dropped`` counts the stored events ``drop`` removed, per
    identity. ``reason`` says what to do when the status is
    ``"refused"``.
    """

    thread_id: str
    status: RewriteStatus
    migrated: tuple[tuple[str, str], ...] = ()
    dropped: dict[str, int] = field(default_factory=dict)
    reason: str = ""


@dataclass(frozen=True)
class RewriteReport:
    """What :meth:`EventGraph.plan_rewrite` found, or what
    :meth:`EventGraph.rewrite_store` did.

    ``applied`` is ``False`` from ``plan_rewrite()``. A ``"rewrite"``
    status then means the thread would be rewritten. From
    ``rewrite_store()`` it means the thread was rewritten.
    """

    applied: bool
    threads: tuple[ThreadRewrite, ...]

    @property
    def refused(self) -> tuple[ThreadRewrite, ...]:
        """Every thread the walk refused, with its reason."""
        return tuple(t for t in self.threads if t.status == "refused")

    def __str__(self) -> str:
        counts = {"rewrite": 0, "unchanged": 0, "refused": 0}
        for thread in self.threads:
            counts[thread.status] += 1
        mode = "applied" if self.applied else "plan"
        total = len(self.threads)
        noun = "thread" if total == 1 else "threads"
        lines = [
            f"{mode}: {total} {noun}, {counts['rewrite']} rewrite, "
            f"{counts['unchanged']} unchanged, {counts['refused']} refused"
        ]
        lines.extend(f"  {t.thread_id}: refused, {t.reason}" for t in self.refused)
        return "\n".join(lines)


@dataclass(frozen=True)
class ThreadPlan:
    """The outcome of :func:`plan_thread`, plus what to store when the
    status is ``"rewrite"``."""

    result: ThreadRewrite
    checkpoint: Checkpoint | None = None
    new_versions: dict[str, Any] = field(default_factory=dict)
    interrupt_writes: tuple[tuple[str, str, Any], ...] = ()


def identity_label(module: str, qualname: str) -> str:
    return f"{module}:{qualname}"


def validate_drop(
    serde: NamespaceAwareSerde, drop: Iterable[type], method: str
) -> tuple[type[Event], ...]:
    """Each ``drop`` entry must be an ``Event`` class this serde revives to
    that same class. Raises ``ValueError`` naming *method* otherwise."""
    from langgraph_events.serde.migrations._core import (  # noqa: PLC0415
        _resolve_identity,
    )

    classes: list[type[Event]] = []
    for cls in drop:
        if not (isinstance(cls, type) and issubclass(cls, Event)):
            raise ValueError(
                f"{method}() drop= takes Event classes, got {cls!r}. Pass the "
                f"class whose stored instances must leave the log."
            )
        try:
            live = _resolve_identity(
                cls.__module__, cls.__qualname__, scope=serde._scope
            )
        except (ImportError, AttributeError):
            live = None
        if live is not cls:
            raise ValueError(
                f"{method}() cannot drop {cls.__qualname__}: the checkpointer's "
                f"serde does not revive that identity to this class. Pass it "
                f"via events= or nest it inside a namespace passed via "
                f"namespaces=, so the walk can recognise its stored instances."
            )
        classes.append(cls)
    return tuple(classes)


def _migrated_pairs(reads: Sequence[ReadRecord]) -> tuple[tuple[str, str], ...]:
    """The touched records as ``(stored, live)`` labels, deduped in read
    order."""
    pairs: list[tuple[str, str]] = []
    for stored, live, touched in reads:
        pair = (identity_label(*stored), identity_label(*live))
        if touched and pair not in pairs:
            pairs.append(pair)
    return tuple(pairs)


def _interrupt_payloads(writes: Sequence[tuple[str, str, Any]]) -> list[Any]:
    """Every payload inside the ``__interrupt__`` pending writes. A write
    holds one ``Interrupt`` or a sequence of them, a framework detail."""
    payloads: list[Any] = []
    for _task_id, channel, value in writes:
        if channel != _INTERRUPT_CHANNEL:
            continue
        entries = value if isinstance(value, (list, tuple)) else [value]
        payloads.extend(getattr(entry, "value", entry) for entry in entries)
    return payloads


def _log_entries(values: dict[str, Any], channel: str) -> list[Any] | None:
    """The entries of a log channel, or ``None`` when the channel is
    absent or not a list. One definition for every drop rule."""
    entries = values.get(channel)
    return entries if isinstance(entries, list) else None


def _count_drops(
    values: dict[str, Any], drop: tuple[type[Event], ...]
) -> dict[str, int]:
    """How many stored entries of each ``drop`` class leave the log
    channels. A ``Resumed`` back-reference is cleared, not counted, so
    the count matches the log entries an operator can see."""
    counts: dict[str, int] = {}
    for channel in _LOG_CHANNELS:
        for entry in _log_entries(values, channel) or ():
            if type(entry) in drop:
                label = identity_label(type(entry).__module__, type(entry).__qualname__)
                counts[label] = counts.get(label, 0) + 1
    return counts


def _clear_resumed(entry: Any, drop: tuple[type[Event], ...]) -> Any:
    """A ``Resumed`` keeps the interrupt it answered and the value that
    answered it. A dropped class in either slot becomes ``None``. The
    field is typed optional for exactly this: once the class is deleted
    the reference could not revive anyway."""
    if not isinstance(entry, Resumed):
        return entry
    changes = {
        name: None
        for name in ("value", "interrupted")
        if type(getattr(entry, name)) in drop
    }
    return entry.model_copy(update=changes) if changes else entry


def plan_thread(
    *,
    thread_id: str,
    tup: CheckpointTuple,
    unresolved: Sequence[UnrevivedIdentity],
    reads: Sequence[ReadRecord],
    serde: NamespaceAwareSerde,
    drop: tuple[type[Event], ...],
    next_version: Callable[[Any, None], Any],
) -> ThreadPlan:
    """Decide what to do with one thread, and build the checkpoint to
    store when the answer is ``"rewrite"``.

    *tup* was read inside ``serde.tolerate_unresolved()`` and
    ``serde._record_reads()``; *unresolved* and *reads* are those two
    collectors. This function runs after both blocks closed, so its
    verification reads are strict. ``drop`` matches the stored identity
    exactly, not the class hierarchy: the rule ``validate_drop`` and the
    byte scan apply.
    """
    pairs = _migrated_pairs(reads)
    values: dict[str, Any] = tup.checkpoint["channel_values"]
    dropped = _count_drops(values, drop)
    drop_identities = {(cls.__module__, cls.__qualname__) for cls in drop}
    present = any(live in drop_identities for _stored, live, _touched in reads)

    def refused(reason: str) -> ThreadPlan:
        result = ThreadRewrite(
            thread_id, "refused", migrated=pairs, dropped=dropped, reason=reason
        )
        return ThreadPlan(result)

    if unresolved:
        names = ", ".join(dict.fromkeys(u.qualname for u in unresolved))
        return refused(
            f"history names {names}, which the serde cannot revive; add a "
            f"tombstone first"
        )
    if not pairs and not dropped and not present:
        return ThreadPlan(ThreadRewrite(thread_id, "unchanged"))

    pending = list(tup.pending_writes or ())
    reason = _pending_refusal(pending, values, drop)
    if reason is not None:
        return refused(reason)

    interrupt_writes = tuple(w for w in pending if w[1] == _INTERRUPT_CHANNEL)
    new_values, changed = _drop_from_log(values, drop)
    checkpoint = _copy_checkpoint(tup.checkpoint, new_values)
    reason, new_versions = _verify_and_bump(
        serde, checkpoint, changed, interrupt_writes, drop_identities, next_version
    )
    if reason is not None:
        return refused(reason)
    return ThreadPlan(
        ThreadRewrite(thread_id, "rewrite", migrated=pairs, dropped=dropped),
        checkpoint=checkpoint,
        new_versions=new_versions,
        interrupt_writes=interrupt_writes,
    )


def _pending_refusal(
    pending: Sequence[tuple[str, str, Any]],
    values: dict[str, Any],
    drop: tuple[type[Event], ...],
) -> str | None:
    """Why the pending writes stop this rewrite, or ``None``."""
    interrupt_writes = [w for w in pending if w[1] == _INTERRUPT_CHANNEL]
    for payload in _interrupt_payloads(interrupt_writes):
        if type(payload) in drop:
            return (
                f"thread is paused on {type(payload).__qualname__}, a dropped "
                f"class; abandon it first"
            )
    if interrupt_writes and any(
        type(entry) in drop for entry in _log_entries(values, "_pending") or ()
    ):
        # The resumed task would find no matching event, return nothing,
        # and end the thread unpaused with the resume payload lost.
        return (
            "a dropped event is pending dispatch on a paused thread; resume "
            "or abandon the thread first"
        )
    for task_id, channel, _value in pending:
        if channel == _ERROR_CHANNEL:
            return (
                "thread has a pending __error__ write from a failed task; run "
                "it again or abandon it, then rerun"
            )
        if channel != _INTERRUPT_CHANNEL:
            # A completed task write is insert-only in both savers: the
            # API cannot rewrite it.
            return (
                f"thread has a completed task write on task {task_id!r}; "
                f"resume or abandon it, then rerun"
            )
    return None


def _verify_and_bump(
    serde: NamespaceAwareSerde,
    checkpoint: Checkpoint,
    changed: set[str],
    interrupt_writes: Sequence[tuple[str, str, Any]],
    forbidden: set[tuple[str, str]],
    next_version: Callable[[Any, None], Any],
) -> tuple[str | None, dict[str, Any]]:
    """Verify every value ``put`` will encode, and bump the version of
    each channel whose value changed or carries an event. Returns the
    refusal reason, or ``None`` and the ``new_versions`` for ``put``."""
    historic = set(serde._rename_table)
    versions: dict[str, Any] = checkpoint["channel_versions"]
    new_versions: dict[str, Any] = {}
    for channel, value in checkpoint["channel_values"].items():
        where = f"channel {channel!r}"
        problem, identities = _verify_value(serde, value, where, forbidden, historic)
        if problem is not None:
            return problem, {}
        if channel in changed or identities:
            old = versions.get(channel)
            new = next_version(old, None)
            versions[channel] = new
            new_versions[channel] = new
            _mirror_seen(checkpoint["versions_seen"], channel, old, new)
    for _task_id, channel, value in interrupt_writes:
        where = f"the pending {channel} write"
        problem, _identities = _verify_value(serde, value, where, forbidden, historic)
        if problem is not None:
            return problem, {}
    return None, new_versions


def _drop_from_log(
    values: dict[str, Any], drop: tuple[type[Event], ...]
) -> tuple[dict[str, Any], set[str]]:
    """Filter ``events`` and ``_pending``. Lower ``_cursor`` by the number
    of dropped ``events`` entries that sat below it, so the next run
    still dispatches exactly the entries it would have dispatched."""
    new_values = dict(values)
    changed: set[str] = set()
    if not drop:
        return new_values, changed
    cursor = values.get("_cursor")
    for channel in _LOG_CHANNELS:
        entries = _log_entries(values, channel)
        if entries is None:
            continue
        kept = [
            _clear_resumed(entry, drop) for entry in entries if type(entry) not in drop
        ]
        if kept == entries:
            continue
        new_values[channel] = kept
        changed.add(channel)
        if channel == "events" and isinstance(cursor, int):
            below = sum(1 for entry in entries[:cursor] if type(entry) in drop)
            if below:
                new_values["_cursor"] = cursor - below
                changed.add("_cursor")
    return new_values, changed


def _copy_checkpoint(checkpoint: Checkpoint, values: dict[str, Any]) -> Checkpoint:
    """A shallow copy with its own version maps, so the plan can bump
    versions without touching the tuple it was read from."""
    copied: dict[str, Any] = dict(checkpoint)
    copied["channel_values"] = values
    copied["channel_versions"] = dict(checkpoint["channel_versions"])
    copied["versions_seen"] = {
        node: dict(seen) for node, seen in checkpoint["versions_seen"].items()
    }
    return copied  # type: ignore[return-value]


def _mirror_seen(
    versions_seen: dict[str, dict[str, Any]], channel: str, old: Any, new: Any
) -> None:
    """Pregel triggers a node when a channel version is greater than the
    version the node has seen. A bumped version must not read as an
    update, so every entry that equals the old version follows it."""
    for seen in versions_seen.values():
        if channel in seen and seen[channel] == old:
            seen[channel] = new


def _verify_value(
    serde: NamespaceAwareSerde,
    value: Any,
    where: str,
    forbidden: set[tuple[str, str]],
    historic: set[tuple[str, str]],
) -> tuple[str | None, list[tuple[str, str]]]:
    """Encode *value* the way ``put`` will, then prove the bytes name no
    dropped or historic identity and revive under the strict serde.

    *where* names the value in a refusal, ``"channel 'events'"`` or
    ``"the pending __interrupt__ write"``. Returns ``(problem,
    identities)``. *problem* is ``None`` when the value passed.
    *identities* lists every event identity the bytes hold, so the
    caller knows whether the channel carries an event.
    """
    from langgraph_events.serde._jsonplus import (  # noqa: PLC0415
        _scan_identities,
    )

    try:
        data = serde.dumps_typed(value)
    except (ValueError, TypeError) as exc:
        # ``MsgpackEncodeError`` is a ``TypeError``. One unencodable
        # value refuses this thread and must not stop the walk.
        return f"cannot encode the value in {where} ({exc})", []
    if data[0] != "msgpack":
        return None, []
    identities = _scan_identities(data[1])
    for identity in identities:
        if identity in forbidden:
            return (
                f"{identity[1]} remains in {where} after the drop; resume or "
                f"abandon the thread, or leave the class in place",
                identities,
            )
        if identity in historic:
            return (
                f"the re-encode still names historic identity {identity[1]} in {where}",
                [],
            )
    try:
        serde.loads_typed(data)
    except ValueError as exc:
        return f"the rewritten value in {where} does not revive ({exc})", identities
    return None, identities
