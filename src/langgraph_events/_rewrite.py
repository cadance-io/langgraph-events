"""Apply-side migration: plan the rewrite of one thread's live checkpoint.

``EventGraph.plan_rewrite()`` and ``rewrite_store()`` read the latest
checkpoint of each thread the caller names. The serde's read path
already applies every rename, transform, split and fill, so a
checkpoint read through ``get_tuple()`` and written back through
``put()`` lands under live identities and live field shapes. This
module owns the pure step in between: decide whether the thread needs a
write, drop the stored events the caller named, and build the
checkpoint the graph will store. It never touches a store.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass, replace
from typing import TYPE_CHECKING, Any, Literal

from langgraph_events._causes import dropped, resolve
from langgraph_events._event import Event, Resumed
from langgraph_events._identity import Identity

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Sequence

    from langgraph.checkpoint.base import ChannelVersions, Checkpoint, CheckpointTuple

    from langgraph_events._causes import CauseEntry
    from langgraph_events.serde._jsonplus import (
        DecodedValue,
        NamespaceAwareSerde,
        ReadRecord,
        UnrevivedIdentity,
    )

_INTERRUPT_CHANNEL = "__interrupt__"
"""The pending-write channel the rewrite writes back in place."""

_ERROR_CHANNEL = "__error__"
"""A failed task's write. The task runs again on the next invoke."""

_LOG_CHANNELS = ("events", "_pending")
"""The two channels ``drop`` filters. ``_POSITION_CHANNELS`` index ``events``."""

_POSITION_CHANNELS = ("_cursor",)
"""The channels that hold a position in ``events``."""

_POSITION_WRITES = ("causes", *_POSITION_CHANNELS)
"""The channels whose task write can hold a position in ``events``."""

RewriteStatus = Literal["rewrite", "unchanged", "refused"]


@dataclass(frozen=True)
class ThreadRewrite:
    """One thread's line in a :class:`RewriteReport`.

    ``migrated`` lists each ``(stored, live)`` identity pair the migration
    table rewrote, as ``"module:qualname"`` strings, deduped in read
    order. ``dropped`` counts the stored events ``drop`` removed, per
    identity. ``reason`` says what to do when the status is
    ``"refused"``. ``checkpoint_ns`` is the checkpoint namespace the walk
    read. ``""`` is the root namespace.
    """

    thread_id: str
    status: RewriteStatus
    migrated: tuple[tuple[str, str], ...] = ()
    dropped: dict[str, int] = field(default_factory=dict)
    reason: str = ""
    checkpoint_ns: str = ""

    @property
    def location(self) -> str:
        """The thread id, and the checkpoint namespace when it is not the
        root namespace."""
        if not self.checkpoint_ns:
            return self.thread_id
        return f"{self.thread_id} in subgraph namespace {self.checkpoint_ns!r}"


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
        lines.extend(f"  {t.location}: refused, {t.reason}" for t in self.refused)
        return "\n".join(lines)


@dataclass(frozen=True)
class RewritePlan:
    """A ``"rewrite"`` outcome of :func:`plan_thread`, plus what to store."""

    result: ThreadRewrite
    checkpoint: Checkpoint
    new_versions: dict[str, Any]
    interrupt_writes: tuple[tuple[str, str, Any], ...]


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
    for stored, live, touched, _built in reads:
        pair = (Identity(*stored).label, Identity(*live).label)
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
                label = Identity(type(entry).__module__, type(entry).__qualname__).label
                counts[label] = counts.get(label, 0) + 1
    return counts


def _clear_resumed(entry: object, drop: tuple[type[Event], ...]) -> object:
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
    return replace(entry, **changes) if changes else entry


def plan_thread(
    *,
    thread_id: str,
    checkpoint_ns: str,
    tup: CheckpointTuple,
    unresolved: Sequence[UnrevivedIdentity],
    decoded: Sequence[DecodedValue],
    serde: NamespaceAwareSerde,
    drop: tuple[type[Event], ...],
    next_version: Callable[[Any, None], Any],
) -> ThreadRewrite | RewritePlan:
    """Decide what to do with one thread, and build the checkpoint to
    store when the answer is ``"rewrite"``.

    *tup* was read inside ``serde.tolerate_unresolved()`` and
    ``serde._record_reads(decoded)``; *unresolved* and *decoded* are those
    two collectors. This function runs after both blocks closed, so its
    verification reads are strict. ``drop`` matches the stored identity
    exactly, not the class hierarchy: the rule ``validate_drop`` and the
    byte scan apply. A non-root *checkpoint_ns* changes only the remedy of
    a refusal, because ``abandon()`` and ``resume()`` reach the root only.
    """
    reads = [record for _value, records in decoded for record in records]
    pairs = _migrated_pairs(reads)
    values: dict[str, Any] = tup.checkpoint["channel_values"]
    dropped = _count_drops(values, drop)
    drop_identities = {(cls.__module__, cls.__qualname__) for cls in drop}
    present = any(live in drop_identities for _stored, live, *_ in reads)

    def refused(reason: str) -> ThreadRewrite:
        return ThreadRewrite(
            thread_id, "refused", migrated=pairs, dropped=dropped, reason=reason
        )

    reason = _history_refusal(unresolved, values)
    if reason is not None:
        return refused(reason)
    if not pairs and not dropped and not present:
        return ThreadRewrite(thread_id, "unchanged")

    pending = list(tup.pending_writes or ())
    reason = _pending_refusal(
        pending, values, drop, decoded, subgraph=bool(checkpoint_ns)
    )
    if reason is not None:
        return refused(reason)

    interrupt_writes = tuple(w for w in pending if w[1] == _INTERRUPT_CHANNEL)
    new_values, changed = _drop_from_log(values, drop)
    checkpoint = _copy_checkpoint(tup.checkpoint, new_values)
    built = frozenset(id(value) for *_, values in reads for value in values)
    reason, new_versions = _verify_and_bump(
        serde,
        checkpoint,
        changed,
        interrupt_writes,
        drop_identities,
        built,
        next_version,
    )
    if reason is not None:
        return refused(reason)
    return RewritePlan(
        ThreadRewrite(thread_id, "rewrite", migrated=pairs, dropped=dropped),
        checkpoint=checkpoint,
        new_versions=new_versions,
        interrupt_writes=interrupt_writes,
    )


def _history_refusal(
    unresolved: Sequence[UnrevivedIdentity], values: dict[str, Any]
) -> str | None:
    """Why the stored history stops this rewrite, or ``None``."""
    if unresolved:
        names = ", ".join(dict.fromkeys(u.qualname for u in unresolved))
        return (
            f"history names {names}, which the serde cannot revive; add a "
            f"tombstone first"
        )
    return _entries_not_events(values)


def _entries_not_events(values: dict[str, Any]) -> str | None:
    """Why a log channel holds an entry that is not an event, or ``None``."""
    for channel in _LOG_CHANNELS:
        problem = _not_events(_log_entries(values, channel) or [])
        if problem is not None:
            return f"channel {channel!r} {problem}"
    return None


def _not_events(log: list[Any]) -> str | None:
    """Why *log* holds an entry that is not an event, or ``None``.

    LangGraph's own serializer revives a record whose class no longer
    imports, or no longer takes the stored fields, as a ``dict`` or
    ``None``, with no error. No migration reaches such a record. A class
    that its allow-list blocks also revives as a ``dict``.
    """
    entries = [e for e in log if not isinstance(e, Event)]
    if not entries:
        return None
    kinds = " or ".join(sorted({type(e).__name__ for e in entries}))
    count = (
        "1 entry that revives"
        if len(entries) == 1
        else f"{len(entries)} entries that revive"
    )
    noun = "an event" if len(entries) == 1 else "events"
    blocked = (
        ". A dict also comes from a class that imports but that the "
        "allowed_msgpack_modules of the serde does not list, for example "
        "under LANGGRAPH_STRICT_MSGPACK=true. Then add each class to "
        "allowed_msgpack_modules, then rerun"
        if any(isinstance(e, dict) for e in entries)
        else ""
    )
    return (
        f"holds {count} as {kinds}, not as {noun}. "
        f"A serializer other than NamespaceAwareSerde wrote them, and their "
        f"class no longer imports or no longer takes the stored fields, so "
        f"no migration reaches them. Restore each class under its stored "
        f"module and name, or delete the thread with the checkpointer's "
        f"delete_thread() if nobody needs it, then rerun{blocked}"
    )


_RESUME_THE_ROOT = (
    "resume the root thread, then rerun. abandon() does not clear a subgraph checkpoint"
)
"""The remedy for a paused subgraph checkpoint. A resume of the root thread
continues the subgraph run."""

_NO_CALL_CLEARS = (
    "abandon() and resume() reach only the root namespace, so no call clears "
    "this write. If nobody needs the thread, delete every checkpoint of it "
    "with the checkpointer's delete_thread(), then rerun. Otherwise keep the "
    "migrations and the classes that this checkpoint needs"
)
"""The remedy for a task write in a subgraph checkpoint. A new run of the
root thread writes under a new namespace and leaves this checkpoint."""


def _pending_refusal(
    pending: Sequence[tuple[str, str, Any]],
    values: dict[str, Any],
    drop: tuple[type[Event], ...],
    decoded: Sequence[DecodedValue],
    *,
    subgraph: bool,
) -> str | None:
    """Why the pending writes stop this rewrite, or ``None``. A *subgraph*
    checkpoint gets the remedy that reaches it."""
    interrupt_writes = [w for w in pending if w[1] == _INTERRUPT_CHANNEL]
    for payload in _interrupt_payloads(interrupt_writes):
        if type(payload) in drop:
            return (
                f"thread is paused on {type(payload).__qualname__}, a dropped "
                f"class; {_RESUME_THE_ROOT if subgraph else 'abandon it first'}"
            )
    if interrupt_writes and any(
        type(entry) in drop for entry in _log_entries(values, "_pending") or ()
    ):
        # The resumed task would find no matching event, return nothing,
        # and end the thread unpaused with the resume payload lost.
        remedy = "resume or abandon the thread first"
        return (
            f"a dropped event is pending dispatch on a paused thread; "
            f"{_RESUME_THE_ROOT if subgraph else remedy}"
        )
    shifts = any(type(entry) in drop for entry in _log_entries(values, "events") or ())
    for task_id, channel, value in pending:
        if channel == _ERROR_CHANNEL:
            remedy = "run it again or abandon it, then rerun"
            return (
                f"thread has a pending __error__ write from a failed task; "
                f"{_NO_CALL_CLEARS if subgraph else remedy}"
            )
        if channel == _INTERRUPT_CHANNEL:
            continue
        where = f"thread has a completed task write on task {task_id!r} in"
        problem = (
            _not_events(value)
            if channel in _LOG_CHANNELS and isinstance(value, list)
            else None
        )
        if problem is not None:
            # A resume would append the entry to the log, so its remedy
            # is the remedy of the entry.
            return f"{where} channel {channel!r} that {problem}"
        problem = _write_problem(channel, value, decoded, drop, shifts=shifts)
        if problem is not None:
            remedy = "resume or abandon it, then rerun"
            return (
                f"{where} channel {channel!r} that {problem}; "
                f"{_NO_CALL_CLEARS if subgraph else remedy}"
            )
    return None


def _write_problem(
    channel: str,
    value: object,
    decoded: Sequence[DecodedValue],
    drop: tuple[type[Event], ...],
    *,
    shifts: bool,
) -> str | None:
    """What a completed task write holds that the rewrite would change, or
    ``None``.

    ``PostgresSaver``, ``SqliteSaver`` and ``MemorySaver`` store a completed
    task write once and never replace it, so the rewrite keeps the write as
    it is. A resume applies the write and does not run its task again. So
    the decode of the write must read no touched record and no class in
    *drop*. *shifts* is ``True`` when *drop* removes an ``events`` entry. A
    position in the write then names another event.
    """
    records = next((r for result, r in decoded if result is value), None)
    if records is None:
        return "this serde did not decode, so the rewrite cannot read it"
    dropped = {(cls.__module__, cls.__qualname__) for cls in drop}
    for stored, live, touched, _built in records:
        if live in dropped:
            return f"holds {Identity(*live).label}, a class in drop"
        if touched:
            return f"holds {Identity(*stored).label}, which a migration rewrites"
    if shifts and channel in _POSITION_WRITES and value not in (None, []):
        return "holds a log position, and drop removes an event from the log"
    return None


def _verify_and_bump(
    serde: NamespaceAwareSerde,
    checkpoint: Checkpoint,
    changed: set[str],
    interrupt_writes: Sequence[tuple[str, str, Any]],
    forbidden: set[tuple[str, str]],
    built: frozenset[int],
    next_version: Callable[[Any, None], Any],
) -> tuple[str | None, dict[str, Any]]:
    """Verify every value ``put`` will encode, and bump the version of
    each channel whose value changed or carries an event. *built* holds
    the ``id()`` of each field value a migration built. Returns the
    refusal reason, or ``None`` and the ``new_versions`` for ``put``."""
    historic = set(serde._tables.rename_table)
    versions: dict[str, Any] = checkpoint["channel_versions"]
    new_versions: dict[str, Any] = {}
    for channel, value in checkpoint["channel_values"].items():
        where = f"channel {channel!r}"
        problem, identities = _verify_value(
            serde, value, where, forbidden, historic, built
        )
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
        problem, _identities = _verify_value(
            serde, value, where, forbidden, historic, built
        )
        if problem is not None:
            return problem, {}
    return None, new_versions


def _drop_from_log(
    values: dict[str, Any], drop: tuple[type[Event], ...]
) -> tuple[dict[str, Any], set[str]]:
    """Filter ``events`` and ``_pending``. Lower each position by the number
    of dropped ``events`` entries below it, so the next run still dispatches
    exactly the entries it would have dispatched. Remap ``causes``."""
    new_values = dict(values)
    changed: set[str] = set()
    if not drop:
        return new_values, changed
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
    events = _log_entries(values, "events")
    if events is not None and "events" in changed:
        _shift_positions(values, events, drop, new_values, changed)
    return new_values, changed


def _shift_positions(
    values: dict[str, Any],
    events: list[Any],
    drop: tuple[type[Event], ...],
    new_values: dict[str, Any],
    changed: set[str],
) -> None:
    """Lower each position channel past the dropped events, and remap ``causes``."""
    for channel in _POSITION_CHANNELS:
        position = values.get(channel)
        if not isinstance(position, int):
            continue
        below = sum(1 for entry in events[:position] if type(entry) in drop)
        if below:
            new_values[channel] = position - below
            changed.add(channel)
    causes = _log_entries(values, "causes")
    if causes is None:
        return
    remapped = _remap_causes(causes, events, drop)
    if remapped != causes:
        new_values["causes"] = remapped
        changed.add("causes")


def _remap_causes(
    causes: list[Any], events: list[Any], drop: tuple[type[Event], ...]
) -> list[CauseEntry]:
    """Filter ``causes`` at the dropped positions and remap each source.

    :func:`~langgraph_events._causes.resolve` aligns the channels and makes
    each source absolute first, so this remaps absolute indices only and
    writes absolute entries back. An event below ``known_from`` keeps no
    entry, so the channels still align from the end. A cause whose source
    was dropped becomes :func:`~langgraph_events._causes.dropped`: the handler
    and the type of the deleted event stay known.
    """
    entries, known_from = resolve(events, causes)
    new_position: dict[int, int] = {}
    for old, event in enumerate(events):
        if type(event) not in drop:
            new_position[old] = len(new_position)
    remapped: list[CauseEntry] = []
    for position in range(known_from, len(events)):
        if position not in new_position:
            continue
        entry = entries[position]
        if entry is None:
            remapped.append(None)
            continue
        source = entry[0]
        if not isinstance(source, int):
            remapped.append(entry)
            continue
        via = entry[1]
        target = new_position.get(source)
        remapped.append(
            dropped(via, type(events[source]).__qualname__)
            if target is None
            else (target, via)
        )
    return remapped


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
    versions_seen: dict[str, ChannelVersions],
    channel: str,
    old: object,
    new: str | int | float,
) -> None:
    """Pregel triggers a node when a channel version is greater than the
    version the node has seen. A bumped version must not read as an
    update, so every entry that equals the old version follows it."""
    for seen in versions_seen.values():
        if channel in seen and seen[channel] == old:
            seen[channel] = new


def _verify_value(
    serde: NamespaceAwareSerde,
    value: object,
    where: str,
    forbidden: set[tuple[str, str]],
    historic: set[tuple[str, str]],
    built: frozenset[int],
) -> tuple[str | None, list[tuple[str, str]]]:
    """Encode *value* the way ``put`` will, then prove the bytes name no
    dropped or historic identity and revive under the strict serde, and
    that each pydantic model a migration built in *value* validates.

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
    return _revival_problem(serde, data, value, where, built), identities


def _revival_problem(
    serde: NamespaceAwareSerde,
    data: tuple[str, bytes],
    value: object,
    where: str,
    built: frozenset[int],
) -> str | None:
    """Why the bytes *data* of *value* in *where* do not revive, or do not
    settle, or why a model a migration built in *value* is invalid, or
    ``None``.

    The bytes settle when a read of them migrates no record. Then a read
    of the store after the write migrates nothing, unless the saver did
    not store these bytes.
    """
    with serde._record_reads() as reads:
        start = len(reads)
        try:
            serde.loads_typed(data)
        except ValueError as exc:
            return f"the rewritten value in {where} does not revive ({exc})"
    unsettled = dict.fromkeys(live for _stored, live in _migrated_pairs(reads[start:]))
    if unsettled:
        return (
            f"a migration of {', '.join(unsettled)} does not settle: a read of "
            f"the rewritten value in {where} changes it again. Make each "
            f"class-stage transform return its own output unchanged, then rerun"
        )
    invalid = _invalid_model(value, "", built, inside=False)
    if invalid is None:
        return None
    return (
        f"the rewritten value in {where} holds an invalid {invalid}. Fix the "
        f"migration that builds the value so that the value validates, then "
        f"rerun"
    )


def _invalid_model(
    value: object, path: str, built: frozenset[int], *, inside: bool
) -> str | None:
    """The first pydantic model that a migration built inside an event of
    *value* and that fails the validation of its class, as
    ``<model> in <event>.<field>: <error>``.

    LangGraph's serializer revives a stored model with its constructor, and
    on a validation error with ``model_construct``, which validates nothing.
    So a rewrite must not store an invalid model: once the migration that
    built it leaves the code, every read accepts it with no warning. A model
    that the store already held is not checked: a valid model can fail the
    rebuild from its dump, for example through an alias, and the rewrite
    stores it unchanged. *path* is the event field that holds *value*.
    *inside* is ``True`` below a value whose ``id()`` is in *built*.
    """
    inside = inside or id(value) in built
    dump = getattr(value, "model_dump", None)
    if path and not isinstance(value, type) and callable(dump):
        if not inside:
            return None
        try:
            type(value)(**dump())
        except (ValueError, TypeError) as exc:
            error = " ".join(str(exc).split())
            return f"{type(value).__name__} in {path}: {error}"
        return None
    if isinstance(value, Event) and is_dataclass(value):
        children: Iterable[tuple[object, str]] = (
            (getattr(value, f.name), f"{type(value).__qualname__}.{f.name}")
            for f in fields(value)
        )
    elif isinstance(value, dict):
        children = ((item, path) for item in value.values())
    elif isinstance(value, (list, tuple, set, frozenset)):
        children = ((item, path) for item in value)
    else:
        return None
    for child, child_path in children:
        problem = _invalid_model(child, child_path, built, inside=inside)
        if problem is not None:
            return problem
    return None
