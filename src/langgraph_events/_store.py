"""EventStore: maintain the checkpoints of one topology, with no services."""

from __future__ import annotations

import contextlib
import functools
from dataclasses import replace
from typing import TYPE_CHECKING, Any, NamedTuple, cast

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END
from langgraph.types import StateUpdate

from langgraph_events._causes import FRAMEWORK
from langgraph_events._event import (
    Abandoned,
    Event,
    Interrupted,
)
from langgraph_events._event_log import EventLog
from langgraph_events._internal import pad_causes
from langgraph_events._rewrite import (
    RewritePlan,
    RewriteReport,
    ThreadRewrite,
    plan_thread,
    validate_drop,
)

if TYPE_CHECKING:
    from collections.abc import (
        Callable,
        Iterable,
    )

    from langchain_core.runnables import RunnableConfig
    from langgraph.checkpoint.base import CheckpointTuple
    from langgraph.graph.state import CompiledStateGraph
    from langgraph.types import StateSnapshot

    from langgraph_events._topology import EventTopology
    from langgraph_events.serde._jsonplus import (
        DecodedValue,
        NamespaceAwareSerde,
        UnrevivedIdentity,
    )

    _PlanThread = Callable[..., ThreadRewrite | RewritePlan]
    """``plan_thread`` with the serde, drop classes and version function bound."""


def _no_checkpoint(thread_id: str) -> ThreadRewrite:
    """A ``thread_ids`` entry the store does not hold. Reported, so a typo
    in an exported id list cannot vanish from the report."""
    return ThreadRewrite(
        thread_id, "refused", reason="no checkpoint for this thread id"
    )


def _plan_read(
    plan: _PlanThread,
    cfg: RunnableConfig,
    tup: CheckpointTuple | None,
    unresolved: list[UnrevivedIdentity],
    decoded: list[DecodedValue],
) -> ThreadRewrite | RewritePlan:
    """The plan of one read checkpoint, or a refusal when the store has none."""
    thread_id = cfg["configurable"]["thread_id"]
    if tup is None:
        return _no_checkpoint(thread_id)
    return plan(thread_id=thread_id, tup=tup, unresolved=unresolved, decoded=decoded)


def _planned_outcome(planned: ThreadRewrite | RewritePlan) -> ThreadRewrite:
    return planned.result if isinstance(planned, RewritePlan) else planned


_SEE_CHECKPOINTERS = (
    "Fix the saver, then rerun. See 'Checkpointers' in docs/migration-gate.md"
)


def _written_outcome(
    tup: CheckpointTuple,
    planned: RewritePlan,
    after: CheckpointTuple | None,
    again: ThreadRewrite | RewritePlan,
) -> ThreadRewrite:
    """What one write of *planned* did, from the read *after* the write and
    the plan *again* of that read.

    A run between the read and the write lands a newer checkpoint, so the
    latest id is not the written id. The base class of a saver does not
    promise that ``put()`` replaces a stored checkpoint of the same id, or
    that ``put_writes()`` replaces an ``__interrupt__`` write. The plan
    proved that the written bytes settle: a read of them migrates no
    record. So a plan *again* that still finds a rewrite reads bytes that
    the saver did not store.
    """
    written = tup.config["configurable"].get("checkpoint_id")
    if after is None or after.config["configurable"].get("checkpoint_id") != written:
        reason = "thread advanced during the rewrite; rerun"
    elif after.checkpoint["channel_versions"] != planned.checkpoint["channel_versions"]:
        reason = (
            f"the saver did not overwrite checkpoint {written}; put() of the "
            f"saver must replace a stored checkpoint of the same id. "
            f"{_SEE_CHECKPOINTERS}"
        )
    elif isinstance(again, ThreadRewrite) and again.status == "refused":
        reason = (
            f"the rewrite is stored, but the read after the write refuses the "
            f"thread: {again.reason}"
        )
    elif isinstance(again, RewritePlan):
        rule = (
            "put_writes() of the saver must replace an __interrupt__ write of "
            "the same task"
            if planned.interrupt_writes
            else "put() of the saver must store each channel value it receives "
            "under its new version"
        )
        reason = (
            f"the saver did not store the bytes written to checkpoint "
            f"{written}: they read unchanged before the write, but a read of "
            f"the store still migrates them. {rule}. {_SEE_CHECKPOINTERS}"
        )
    else:
        return planned.result
    return replace(planned.result, status="refused", reason=reason)


def _rewrite_report(
    applied: bool, results: list[ThreadRewrite], checkpoint_ns: str
) -> RewriteReport:
    return RewriteReport(
        applied=applied,
        threads=tuple(replace(r, checkpoint_ns=checkpoint_ns) for r in results),
    )


_ALL_THREAD_IDS_QUERY = (
    "SELECT DISTINCT thread_id FROM checkpoints WHERE checkpoint_ns = ''"
)
"""The server-side query that returns every root thread id in a
Postgres or SQLite checkpointer store. Named in the ``None`` error."""


def _explicit_thread_ids(method: str, thread_ids: Iterable[str] | None) -> list[str]:
    """*thread_ids* deduped in caller order. Rejects ``None`` and a bare
    ``str``.

    ``None`` once made the method walk every checkpoint in the store.
    That walk is removed, so ``None`` raises instead of finding nothing.
    A bare ``str`` raises because iterating its characters finds nothing.
    """
    if thread_ids is None:
        raise TypeError(
            f"{method}() thread_ids must be an iterable of thread ids, not "
            f"None. Since 0.34.0 the library does not walk the store. To "
            f"sweep every thread, pass the ids from {_ALL_THREAD_IDS_QUERY}. "
            f"See 'Finding candidates server-side' in "
            f"docs/retiring-interrupted.md."
        )
    if isinstance(thread_ids, str):
        raise TypeError(
            f"{method}() thread_ids must be an iterable of thread ids, "
            f"not one str: pass [{thread_ids!r}] for a single thread."
        )
    return list(dict.fromkeys(thread_ids))


class _PendingInterrupts(NamedTuple):
    """Topology-independent read of a checkpoint's pending interrupt(s).
    See :meth:`EventStore._read_pending_interrupts`.

    ``has_interrupt`` is ``True`` for any pending ``__interrupt__``
    write, including a non-``Event`` payload (e.g. a raw
    ``langgraph.types.interrupt("...")`` call) or an unrevivable one:
    what ``require_interrupt`` and ``event_type=None`` discovery gate
    on. ``events`` narrows that set to successfully-revived ``Event``
    payloads, deduped by qualname: what a class filter matches against.
    ``unresolved_names`` holds the recorded qualname of every interrupt
    whose class no longer imports, deduped in discovery order: no class
    to check, so a filter can never match one, but ``abandon()`` still
    folds them into ``discarded``.
    """

    has_interrupt: bool
    events: list[Event]
    unresolved_names: list[str]


class EventStore:
    """Maintain the checkpoints of one topology, with no services.

    Finds paused threads, abandons them, finds unrevivable threads, and
    plans and applies a store rewrite. None of these runs a handler, so the
    store needs the topology and the checkpointer only::

        store = EventStore(topology, saver)
        configs = store.threads_paused_on(Approval, thread_ids=ids)
        for config in configs:
            store.abandon(config, reason="retired")

    :attr:`EventGraph.event_store` is the store of the topology and the
    checkpointer of a graph.
    """

    def __init__(
        self, topology: EventTopology, checkpointer: BaseCheckpointSaver[Any]
    ) -> None:
        if not isinstance(checkpointer, BaseCheckpointSaver):
            raise TypeError(
                f"EventStore requires a BaseCheckpointSaver, got "
                f"{type(checkpointer).__name__}"
            )
        self._topology = topology
        self._checkpointer = checkpointer

    @functools.cached_property
    def _compiled(self) -> CompiledStateGraph:
        """The compiled graph that a settle writes through.

        A settle writes through ``bulk_update_state(as_node="__seed__")``.
        LangGraph runs only the writers of that node and the dispatch
        function, never a handler body. Each handler node refuses to run, so
        a store can never call a handler without its services.
        """
        return self._topology._compile(None, checkpointer=self._checkpointer)

    @property
    def topology(self) -> EventTopology:
        """The topology of the stored threads."""
        return self._topology

    @staticmethod
    def _events(snapshot: StateSnapshot) -> EventLog:
        """The event log of *snapshot*, with its recorded causes."""
        return EventLog._from_state(
            snapshot.values.get("events", []), snapshot.values.get("causes")
        )

    @staticmethod
    def _settle_supersteps(
        events: EventLog, terminal: Event
    ) -> list[list[StateUpdate]]:
        """Build the clear/append/clear supersteps that settle a thread onto
        *terminal* at rest.

        The leading clear (``values=None, as_node=END``, LangGraph's
        clear-all-tasks branch) skips ``ERROR``/``INTERRUPT`` pending
        writes, preserving an already-completed write from a fanned-out
        sibling.

        The suite pins the leading clear and the ``_cursor`` reset, each
        with its own failing test. The trailing clear and
        ``_pending: []`` are a second guard. Neither is pinned alone.
        """
        appended = [terminal]
        return [
            [StateUpdate(None, END)],
            [
                StateUpdate(
                    pad_causes(
                        {
                            "events": appended,
                            "_cursor": len(events) + len(appended),
                            "_pending": [],
                        },
                        FRAMEWORK,
                    ),
                    "__seed__",
                )
            ],
            [StateUpdate(None, END)],
        ]

    def _settle(self, config: RunnableConfig, terminal: Event) -> EventLog:
        """Append *terminal* and settle the thread via :meth:`_settle_supersteps`."""
        # `_cursor` is correct only because this reads via `get_state`,
        # whose snapshot applies the same pending writes the leading
        # clear commits. A direct checkpoint read undercounts `_cursor`
        # without raising an error, and no test in this suite catches it.
        events = self._events(self._compiled.get_state(config))
        self._compiled.bulk_update_state(
            config,
            self._settle_supersteps(events, terminal),
        )
        return self._events(self._compiled.get_state(config))

    async def _asettle(self, config: RunnableConfig, terminal: Event) -> EventLog:
        """Async sibling of :meth:`_settle`."""
        # Same `_cursor` reasoning as `_settle`, via `aget_state`.
        events = self._events(await self._compiled.aget_state(config))
        await self._compiled.abulk_update_state(
            config,
            self._settle_supersteps(events, terminal),
        )
        return self._events(await self._compiled.aget_state(config))

    @staticmethod
    def _pending_interrupt_writes(pending_writes: Iterable[Any]) -> _PendingInterrupts:
        """Read every pending ``__interrupt__`` checkpoint write into a
        :class:`_PendingInterrupts`.

        Reads a checkpoint's raw ``pending_writes``, topology-
        independent unlike :meth:`_pending_interrupt_events`. A thread
        paused on a handler the current graph no longer registers still
        carries this write: ``StateSnapshot.tasks`` would not show it.
        This is the mechanism behind discovery and ``abandon()``.

        A write's value is one interrupt or a sequence of them (a
        framework detail, handled either way). A payload degraded to
        ``UnrevivedIdentity`` (only possible when the read ran inside
        ``NamespaceAwareSerde.tolerate_unresolved``) contributes its
        stored qualname to ``unresolved_names`` instead of ``events``.
        """
        from langgraph_events.serde._jsonplus import (  # noqa: PLC0415
            UnrevivedIdentity,
        )

        has_interrupt = False
        seen: set[str] = set()
        unresolved_seen: set[str] = set()
        events: list[Event] = []
        unresolved_names: list[str] = []
        for _task_id, channel, value in pending_writes:
            if channel != "__interrupt__":
                continue
            entries = value if isinstance(value, (list, tuple)) else [value]
            for entry in entries:
                has_interrupt = True
                payload = getattr(entry, "value", entry)
                if (
                    isinstance(payload, Event)
                    and type(payload).__qualname__ not in seen
                ):
                    seen.add(type(payload).__qualname__)
                    events.append(payload)
                elif (
                    isinstance(payload, UnrevivedIdentity)
                    and payload.qualname not in unresolved_seen
                ):
                    unresolved_seen.add(payload.qualname)
                    unresolved_names.append(payload.qualname)
        return _PendingInterrupts(
            has_interrupt=has_interrupt,
            events=events,
            unresolved_names=unresolved_names,
        )

    def _tolerant_read(
        self,
    ) -> contextlib.AbstractContextManager[list[UnrevivedIdentity]]:
        """Context manager that degrades an unrevivable event identity
        instead of raising, for the checkpointer's current serde. Yields
        the serde's collector of every identity degraded inside the
        block. See ``NamespaceAwareSerde.tolerate_unresolved``.

        A no-op yielding an empty list for any serde other than
        :class:`~langgraph_events.serde.NamespaceAwareSerde`, the only
        one this library ships that raises ``Cannot revive`` in the
        first place.

        Used by :meth:`_read_checkpoint_tuple`/
        :meth:`_aread_checkpoint_tuple` and by
        :meth:`abandon`/:meth:`aabandon`, which must also read the
        thread's event log and settle it despite the same unrevivable
        identity.
        """
        from langgraph_events.serde._jsonplus import (  # noqa: PLC0415
            NamespaceAwareSerde,
        )

        serde = getattr(self._checkpointer, "serde", None)
        if isinstance(serde, NamespaceAwareSerde):
            return serde.tolerate_unresolved()
        return contextlib.nullcontext([])

    def _read_checkpoint_tuple(
        self, config: RunnableConfig, method: str
    ) -> tuple[CheckpointTuple | None, list[UnrevivedIdentity]]:
        """*config*'s latest checkpoint tuple, read inside
        :meth:`_tolerant_read`, with every identity that read degraded.

        A stored blob naming a class that no longer imports degrades to
        an ``UnrevivedIdentity`` instead of raising, precisely on the
        thread a retirement caller is hunting.

        Any *other* checkpointer failure still propagates, re-raised
        naming *method* and the thread so the caller knows which one is
        unreadable. Never swallowed: skipping it here would make that
        thread silently invisible to every caller of this method,
        discovery included.
        """
        thread_id = config.get("configurable", {}).get("thread_id")
        try:
            with self._tolerant_read() as unresolved:
                tup = self._checkpointer.get_tuple(config)
        except Exception as exc:
            raise RuntimeError(
                f"{method}() could not read thread {thread_id!r}'s checkpoint: {exc}"
            ) from exc
        return tup, list(unresolved)

    async def _aread_checkpoint_tuple(
        self, config: RunnableConfig, method: str
    ) -> tuple[CheckpointTuple | None, list[UnrevivedIdentity]]:
        """Async sibling of :meth:`_read_checkpoint_tuple`, via
        ``aget_tuple``."""
        thread_id = config.get("configurable", {}).get("thread_id")
        try:
            with self._tolerant_read() as unresolved:
                tup = await self._checkpointer.aget_tuple(config)
        except Exception as exc:
            raise RuntimeError(
                f"{method}() could not read thread {thread_id!r}'s checkpoint: {exc}"
            ) from exc
        return tup, list(unresolved)

    def _read_pending_interrupts(
        self, config: RunnableConfig, method: str
    ) -> _PendingInterrupts:
        """*config*'s latest checkpoint's pending interrupt(s), read
        straight from the checkpointer through
        :meth:`_read_checkpoint_tuple`. See
        :meth:`_pending_interrupt_writes`.
        """
        tup, _unresolved = self._read_checkpoint_tuple(config, method)
        if tup is None:
            return _PendingInterrupts(
                has_interrupt=False, events=[], unresolved_names=[]
            )
        return self._pending_interrupt_writes(tup.pending_writes or ())

    async def _aread_pending_interrupts(
        self, config: RunnableConfig, method: str
    ) -> _PendingInterrupts:
        """Async sibling of :meth:`_read_pending_interrupts`, via
        :meth:`_aread_checkpoint_tuple`."""
        tup, _unresolved = await self._aread_checkpoint_tuple(config, method)
        if tup is None:
            return _PendingInterrupts(
                has_interrupt=False, events=[], unresolved_names=[]
            )
        return self._pending_interrupt_writes(tup.pending_writes or ())

    @staticmethod
    def _require_settleable(
        method: str, snapshot: StateSnapshot, config: RunnableConfig
    ) -> None:
        """Raise ``ValueError`` if the thread has no events to settle.

        Checks the event log, not ``snapshot.created_at is None``: a
        ``pre_seed``ed-only thread has a checkpoint and would pass that
        check.
        """
        if snapshot.values.get("events"):
            return
        thread_id = config.get("configurable", {}).get("thread_id")
        raise ValueError(
            f"{method}() has no events to settle on thread {thread_id!r}: the "
            "thread was never run (or only pre_seed()ed)."
        )

    @staticmethod
    def _require_pending_interrupt(
        method: str, pending: _PendingInterrupts, config: RunnableConfig
    ) -> None:
        """Raise ``ValueError`` if *pending* (from
        :meth:`_read_pending_interrupts`) has no interrupt at all,
        including a non-``Event`` payload, which still genuinely pauses
        the thread even though ``discarded`` cannot name it.

        Guards the default ``require_interrupt=True``: without it,
        ``abandon()``/``aabandon()`` would silently settle an
        already-completed thread onto a terminal ``Abandoned``.
        """
        if pending.has_interrupt:
            return
        thread_id = config.get("configurable", {}).get("thread_id")
        raise ValueError(
            f"{method}() found no pending interrupt on thread {thread_id!r}. "
            "Pass require_interrupt=False to settle it anyway."
        )

    @classmethod
    def _require_revivable_history(
        cls,
        method: str,
        events: Iterable[Any],
        unresolved: Iterable[UnrevivedIdentity],
        pending: _PendingInterrupts,
        config: RunnableConfig,
    ) -> None:
        """Raise ``ValueError`` if the thread's settled history holds an
        identity the serde could not revive (#170).

        *events* is the log the settle rewrites, read inside
        :meth:`_tolerant_read`. Any ``UnrevivedIdentity`` entry in it
        refuses the settle, even when its qualname matches the pending
        interrupt: a thread paused twice on one deleted class carries
        that class in both places. *unresolved* is the collector the
        same read yields. It is the extra guard for a placeholder nested
        inside a live event's field, where the log walk cannot see it.
        There, the pending interrupt's own qualname is excused: a
        degraded pending interrupt is the documented delete-first
        recovery, recorded in ``discarded`` and never written back.

        The serde refuses to store a placeholder in any case. This check
        runs first to name the thread and to write nothing at all.
        """
        from langgraph_events.serde._jsonplus import (  # noqa: PLC0415
            UnrevivedIdentity,
        )

        stored = [entry for entry in events if isinstance(entry, UnrevivedIdentity)]
        nested = [
            identity
            for identity in unresolved
            if identity.qualname not in pending.unresolved_names
        ]
        names = cls._unrevived_qualnames([*stored, *nested])
        if not names:
            return
        thread_id = config.get("configurable", {}).get("thread_id")
        raise ValueError(
            f"{method}() refuses to settle thread {thread_id!r}: its settled "
            f"history names {', '.join(names)}, which the checkpointer's serde "
            f"cannot revive. Settling would write a placeholder into the log. "
            f"Map each identity onto a tombstone with @migrate_from instead. "
            f"See 'Recovering a delete-first deployment' in "
            f"docs/retiring-interrupted.md."
        )

    def abandon(
        self,
        config: RunnableConfig,
        *,
        reason: str = "",
        require_interrupt: bool = True,
    ) -> None:
        """Settle a thread onto a terminal ``Abandoned`` without
        dispatching whatever ``Interrupted`` it was paused on. Use this
        to retire an ``Interrupted`` subclass: resuming every paused
        thread first would instead append that identity to the log.

        If ``require_interrupt`` is ``True`` (the default) and the thread
        has no pending interrupt, raises ``ValueError`` naming the
        thread. Pass ``require_interrupt=False`` to settle such a thread
        anyway, recording ``Abandoned(discarded="")``.

        Settles a thread even when the pending interrupt names an
        ``Interrupted`` subclass already deleted from the codebase,
        recording the class's last-known qualname in ``discarded``
        instead of a live instance. Raises ``ValueError`` instead when
        the settled history holds a deleted class: settling would write
        a placeholder into the log, and the serde refuses to store one.
        Recover that thread with a tombstone (see "Recovering a
        delete-first deployment" in ``docs/retiring-interrupted.md``).

        Raises ``ValueError`` if the thread has no events to settle
        (never run, or only ``pre_seed``ed). Ignores
        ``on_unresumable``: that policy governs an accidental no-op
        ``resume()``, not a deliberate abandonment.

        Runs no graph, so returns no log: call
        ``graph.get_state(config).events`` for it.
        """
        with self._tolerant_read() as unresolved:
            snapshot = self._compiled.get_state(config)
            self._require_settleable("abandon", snapshot, config)
            pending = self._read_pending_interrupts(config, "abandon")
            if require_interrupt:
                self._require_pending_interrupt("abandon", pending, config)
            self._require_revivable_history(
                "abandon", snapshot.values["events"], unresolved, pending, config
            )
            discarded = ", ".join(
                (
                    *(type(v).__qualname__ for v in pending.events),
                    *pending.unresolved_names,
                )
            )
            self._settle(config, Abandoned(reason=reason, discarded=discarded))

    async def aabandon(
        self,
        config: RunnableConfig,
        *,
        reason: str = "",
        require_interrupt: bool = True,
    ) -> None:
        """Async version of :meth:`abandon`."""
        with self._tolerant_read() as unresolved:
            snapshot = await self._compiled.aget_state(config)
            self._require_settleable("aabandon", snapshot, config)
            pending = await self._aread_pending_interrupts(config, "aabandon")
            if require_interrupt:
                self._require_pending_interrupt("aabandon", pending, config)
            self._require_revivable_history(
                "aabandon", snapshot.values["events"], unresolved, pending, config
            )
            discarded = ", ".join(
                (
                    *(type(v).__qualname__ for v in pending.events),
                    *pending.unresolved_names,
                )
            )
            await self._asettle(config, Abandoned(reason=reason, discarded=discarded))

    @staticmethod
    def _thread_config(thread_id: str, checkpoint_ns: str = "") -> RunnableConfig:
        """The config that addresses the latest checkpoint of *thread_id* in
        *checkpoint_ns*. ``""`` is the root namespace."""
        configurable = {"thread_id": thread_id}
        if checkpoint_ns:
            configurable["checkpoint_ns"] = checkpoint_ns
        return cast("RunnableConfig", {"configurable": configurable})

    @staticmethod
    def _matches_event_type(
        pending: _PendingInterrupts, event_type: type[Interrupted] | None
    ) -> bool:
        """Whether *pending* (from :meth:`_read_pending_interrupts`) has
        an interrupt matching *event_type*: with ``None``, any pending
        interrupt matches, including a non-``Event`` payload."""
        if not pending.has_interrupt:
            return False
        if event_type is None:
            return True
        return any(isinstance(v, event_type) for v in pending.events)

    def threads_paused_on(
        self,
        event_type: type[Interrupted] | None = None,
        *,
        thread_ids: Iterable[str],
    ) -> list[RunnableConfig]:
        """Configs for each thread in *thread_ids* whose latest checkpoint
        has a pending interrupt. With *event_type*, keeps only threads
        paused on that class or a subclass. With ``event_type=None``,
        returns every listed thread that is paused.

        Reads each thread's raw checkpoint directly, not this graph's
        compiled topology. Two deletions this unlocks, with different
        outcomes:

        - The **handler** that produced the interrupt is already removed
          from this graph. The class still imports, so the interrupt
          revives normally and *event_type* filtering still works. This
          is the common order: retiring an ``Interrupted`` usually
          retires the handler that produced it first.
        - The **class** itself has been deleted and no longer imports.
          The interrupt cannot revive. With no filter
          (``event_type=None``), the thread is still returned. With a
          class filter, it matches nothing: a filter can never match an
          identity with no class. ``abandon()`` records the interrupt's
          last-known qualname in ``discarded`` instead of a live
          instance.

        The caller names the threads in ``thread_ids``. The method reads
        those threads only, deduped in caller order. It does not walk the
        store. A listed thread with no checkpoint, or with no matching
        interrupt, stays out of the result. Get the ids from one
        server-side query. See *Finding candidates server-side* in
        ``docs/retiring-interrupted.md``. ``thread_ids=None`` or a bare
        ``str`` raises ``TypeError``.
        """
        ids = _explicit_thread_ids("threads_paused_on", thread_ids)
        configs: list[RunnableConfig] = []
        for tid in ids:
            cfg = self._thread_config(tid)
            pending = self._read_pending_interrupts(cfg, "threads_paused_on")
            if self._matches_event_type(pending, event_type):
                configs.append(cfg)
        return configs

    async def athreads_paused_on(
        self,
        event_type: type[Interrupted] | None = None,
        *,
        thread_ids: Iterable[str],
    ) -> list[RunnableConfig]:
        """Async version of :meth:`threads_paused_on`."""
        ids = _explicit_thread_ids("athreads_paused_on", thread_ids)
        configs: list[RunnableConfig] = []
        for tid in ids:
            cfg = self._thread_config(tid)
            pending = await self._aread_pending_interrupts(cfg, "athreads_paused_on")
            if self._matches_event_type(pending, event_type):
                configs.append(cfg)
        return configs

    @staticmethod
    def _unrevived_qualnames(unresolved: Iterable[UnrevivedIdentity]) -> list[str]:
        """Qualnames of *unresolved*, deduped in discovery order."""
        names: list[str] = []
        for identity in unresolved:
            if identity.qualname not in names:
                names.append(identity.qualname)
        return names

    def _require_namespace_aware_serde(self, method: str) -> NamespaceAwareSerde:
        """Raise ``ValueError`` naming *method* unless the checkpointer's
        serde is a ``NamespaceAwareSerde``.

        No other serde degrades an identity, so the sweep would return
        an empty mapping while seeing nothing. The default LangGraph
        serializer revives a deleted class as ``None`` with no error.
        """
        from langgraph_events.serde._jsonplus import (  # noqa: PLC0415
            NamespaceAwareSerde,
        )

        serde = getattr(self._checkpointer, "serde", None)
        if isinstance(serde, NamespaceAwareSerde):
            return serde
        raise ValueError(
            f"{method}() needs a NamespaceAwareSerde on the checkpointer, "
            f"found {type(serde).__name__}. Only that serde can report an "
            f"identity it cannot revive. Wire one with "
            f"EventGraph.from_namespaces(..., checkpointer=...) or set "
            f"checkpointer.serde = NamespaceAwareSerde(...)."
        )

    def unrevivable_threads(self, *, thread_ids: Iterable[str]) -> dict[str, list[str]]:
        """Each thread in *thread_ids* whose latest checkpoint holds an
        event identity the checkpointer's serde can no longer revive.

        Returns a mapping of thread id to the qualnames that checkpoint
        could not revive. Thread ids keep the caller's order, without
        duplicates. Qualnames are deduped in read order. The mapping is
        empty when every listed thread revives.

        Reads the real store, not the baseline. The coverage gates
        compare the topology to a committed baseline and never read a
        checkpoint, so they cannot see a settled thread that still raises
        ``Cannot revive``. This method reads that thread.

        Reads each thread's latest checkpoint through the serde's
        tolerant path and collects every identity that path degraded.
        So it reports an identity in the settled ``events`` history, in
        a pending interrupt, in a completed sibling write, or nested in
        a field of a live event. A thread that already *answered* an
        interrupt on a deleted class is reported. :meth:`threads_paused_on`
        never reaches that thread.

        Reads the latest checkpoint of each thread only, under the root
        checkpoint namespace. An older checkpoint version is not read.
        A historic pending interrupt that ``abandon()`` already settled
        can still raise on ``get_state_history()``. See *Ending a pause
        without answering it* in ``docs/control-flow.md``. A checkpoint
        written by this graph embedded as a subgraph lives under a child
        namespace and is not read, the same as :meth:`threads_paused_on`.

        The caller names the threads in ``thread_ids``, the same as
        :meth:`threads_paused_on`. A listed thread with no checkpoint, or
        one that revives, stays out of the mapping. To sweep every
        thread, get the ids from one query:
        ``SELECT DISTINCT thread_id FROM checkpoints WHERE checkpoint_ns = ''``.
        If one checkpointer stores more than one graph, keep only the
        thread ids of this graph.

        Requires a checkpointer serde that is a
        :class:`~langgraph_events.serde.NamespaceAwareSerde`. Raises
        ``ValueError`` otherwise. Raises ``RuntimeError`` naming the
        thread if one checkpoint cannot be read.
        """
        ids = _explicit_thread_ids("unrevivable_threads", thread_ids)
        self._require_namespace_aware_serde("unrevivable_threads")
        found: dict[str, list[str]] = {}
        for tid in ids:
            cfg = self._thread_config(tid)
            _tup, unresolved = self._read_checkpoint_tuple(cfg, "unrevivable_threads")
            if unresolved:
                found[tid] = self._unrevived_qualnames(unresolved)
        return found

    async def aunrevivable_threads(
        self, *, thread_ids: Iterable[str]
    ) -> dict[str, list[str]]:
        """Async version of :meth:`unrevivable_threads`."""
        ids = _explicit_thread_ids("aunrevivable_threads", thread_ids)
        self._require_namespace_aware_serde("aunrevivable_threads")
        found: dict[str, list[str]] = {}
        for tid in ids:
            cfg = self._thread_config(tid)
            _tup, unresolved = await self._aread_checkpoint_tuple(
                cfg, "aunrevivable_threads"
            )
            if unresolved:
                found[tid] = self._unrevived_qualnames(unresolved)
        return found

    def plan_rewrite(
        self,
        *,
        drop: Iterable[type[Event]] = (),
        thread_ids: Iterable[str],
        checkpoint_ns: str = "",
    ) -> RewriteReport:
        """Report what :meth:`rewrite_store` would do. Writes nothing.

        Reads the latest checkpoint of each thread in *thread_ids*
        through the checkpointer's serde. The read path applies every
        rename, transform, split and fill, so the plan lists the stored
        identities the migration table rewrote. ``drop`` names event
        classes whose stored instances leave the ``events`` and
        ``_pending`` channels. Each thread that needs a write is verified
        end to end before it is reported as a rewrite.

        The caller names the threads in ``thread_ids``. The method does
        not walk the store. To sweep every thread, get the ids from one
        query:
        ``SELECT DISTINCT thread_id FROM checkpoints WHERE checkpoint_ns = ''``.
        If one checkpointer stores more than one graph, keep only the
        thread ids of this graph. A listed id with no checkpoint is
        reported as refused, so a typo stays visible.

        *checkpoint_ns* names the checkpoint namespace to read. The default
        ``""`` is the root namespace. A subgraph checkpoint lives under a
        child namespace, for example the checkpoints of a graph that a
        handler invokes on the same checkpointer.

        Raises ``ValueError`` without a
        :class:`~langgraph_events.serde.NamespaceAwareSerde`, under
        ``legacy_write=True``, or for a ``drop`` class the serde cannot
        revive. See *Retiring an Interrupted subclass* in
        ``docs/retiring-interrupted.md``.
        """
        return self._rewrite_walk(
            "plan_rewrite",
            apply=False,
            drop=drop,
            thread_ids=thread_ids,
            checkpoint_ns=checkpoint_ns,
        )

    def rewrite_store(
        self,
        *,
        drop: Iterable[type[Event]] = (),
        thread_ids: Iterable[str],
        checkpoint_ns: str = "",
    ) -> RewriteReport:
        """Rewrite the latest checkpoint of each thread in *thread_ids*
        under the live migration table, and drop the stored events
        ``drop`` names.

        Same threads, same *checkpoint_ns* and same report as
        :meth:`plan_rewrite`. Each thread
        the plan marks as a rewrite is written back through the
        checkpointer's ``put()`` under its existing checkpoint id, with a
        new version for each channel the rewrite touched, and its
        ``__interrupt__`` pending write is rewritten in place. A thread
        that advanced between the read and the write is reported as
        refused and is not rewritten. The plan refuses a thread whose
        rewritten bytes do not settle: a read of them migrates a record
        again. A thread whose stored bytes still need a rewrite after the
        write is reported as refused: the saver did not overwrite the
        checkpoint, or did not store the bytes it was given.

        WARNING: run it while the graph is idle, never inside a rolling
        deploy window. An old pod cannot read the new bytes. Take a
        store-level backup first: the previous blob versions stay in the
        store, but the checkpoint row is overwritten in place. For a
        ``PostgresSaver``, ``backup_live_set()`` and the ``backup`` command
        take it. Review ``plan_rewrite()`` before this call.

        Historic checkpoints keep the old bytes. A ``drop`` removes the
        stored event only. Reducer state is not recomputed. Each
        rewritten thread is read and planned twice: once before the
        write, and once after it to prove that the thread did not move
        and that the saver stored the rewrite.
        """
        return self._rewrite_walk(
            "rewrite_store",
            apply=True,
            drop=drop,
            thread_ids=thread_ids,
            checkpoint_ns=checkpoint_ns,
        )

    def _write_plan(
        self, cfg: RunnableConfig, tup: CheckpointTuple, plan: RewritePlan, method: str
    ) -> None:
        """Store one planned rewrite under the checkpoint id of *tup*."""
        saver = self._checkpointer
        saver.put(
            self._rewrite_base_config(cfg, tup),
            plan.checkpoint,
            tup.metadata,
            plan.new_versions,
        )
        for task_id, channel, value in plan.interrupt_writes:
            saver.put_writes(tup.config, [(channel, value)], task_id)

    def _read_and_plan(
        self,
        cfg: RunnableConfig,
        serde: NamespaceAwareSerde,
        plan: _PlanThread,
        method: str,
    ) -> tuple[CheckpointTuple | None, ThreadRewrite | RewritePlan]:
        """The latest checkpoint at *cfg*, and the plan of it."""
        decoded: list[DecodedValue] = []
        with serde._record_reads(decoded):
            tup, unresolved = self._read_checkpoint_tuple(cfg, method)
        return tup, _plan_read(plan, cfg, tup, unresolved, decoded)

    @staticmethod
    def _rewrite_base_config(
        cfg: RunnableConfig, tup: CheckpointTuple
    ) -> RunnableConfig:
        """The config ``put`` takes: the parent, so the parent link stays,
        or a bare config of the thread and namespace on a thread with one
        checkpoint. ``MemorySaver.put`` reads ``checkpoint_ns`` with no
        default."""
        return tup.parent_config or {
            "configurable": {
                "thread_id": cfg["configurable"]["thread_id"],
                "checkpoint_ns": cfg["configurable"].get("checkpoint_ns", ""),
            }
        }

    def _rewrite_walk(
        self,
        method: str,
        *,
        apply: bool,
        drop: Iterable[type[Event]],
        thread_ids: Iterable[str],
        checkpoint_ns: str,
    ) -> RewriteReport:
        ids = _explicit_thread_ids(method, thread_ids)
        serde, plan = self._prepare_rewrite(method, drop, checkpoint_ns)
        results: list[ThreadRewrite] = []
        for tid in ids:
            cfg = self._thread_config(tid, checkpoint_ns)
            tup, planned = self._read_and_plan(cfg, serde, plan, method)
            if isinstance(planned, RewritePlan) and apply and tup is not None:
                self._write_plan(cfg, tup, planned, method)
                after, again = self._read_and_plan(cfg, serde, plan, method)
                results.append(_written_outcome(tup, planned, after, again))
            else:
                results.append(_planned_outcome(planned))
        return _rewrite_report(apply, results, checkpoint_ns)

    def _prepare_rewrite(
        self, method: str, drop: Iterable[type[Event]], checkpoint_ns: str
    ) -> tuple[NamespaceAwareSerde, _PlanThread]:
        """Shared preconditions of the rewrite walk. Returns the serde
        and the planner bound to this graph's serde, drop classes,
        checkpoint namespace and version function."""
        saver = self._checkpointer
        serde = self._require_namespace_aware_serde(method)
        if serde._legacy_write:
            raise ValueError(
                f"{method}() cannot run under legacy_write=True: the rewrite "
                f"would store every event under its historic identity again. "
                f"Drop legacy_write first, after the rolling deploy drained."
            )
        plan = functools.partial(
            plan_thread,
            checkpoint_ns=checkpoint_ns,
            serde=serde,
            drop=validate_drop(serde, drop, method),
            next_version=saver.get_next_version,
        )
        return serde, plan

    async def aplan_rewrite(
        self,
        *,
        drop: Iterable[type[Event]] = (),
        thread_ids: Iterable[str],
        checkpoint_ns: str = "",
    ) -> RewriteReport:
        """Async version of :meth:`plan_rewrite`, with the same
        *checkpoint_ns*. Writes nothing."""
        return await self._arewrite_walk(
            "aplan_rewrite",
            apply=False,
            drop=drop,
            thread_ids=thread_ids,
            checkpoint_ns=checkpoint_ns,
        )

    async def arewrite_store(
        self,
        *,
        drop: Iterable[type[Event]] = (),
        thread_ids: Iterable[str],
        checkpoint_ns: str = "",
    ) -> RewriteReport:
        """Async version of :meth:`rewrite_store`, with the same
        *checkpoint_ns*."""
        return await self._arewrite_walk(
            "arewrite_store",
            apply=True,
            drop=drop,
            thread_ids=thread_ids,
            checkpoint_ns=checkpoint_ns,
        )

    async def _arewrite_walk(
        self,
        method: str,
        *,
        apply: bool,
        drop: Iterable[type[Event]],
        thread_ids: Iterable[str],
        checkpoint_ns: str,
    ) -> RewriteReport:
        """Async sibling of :meth:`_rewrite_walk`, via ``aget_tuple``,
        ``aput`` and ``aput_writes``."""
        ids = _explicit_thread_ids(method, thread_ids)
        serde, plan = self._prepare_rewrite(method, drop, checkpoint_ns)
        results: list[ThreadRewrite] = []
        for tid in ids:
            cfg = self._thread_config(tid, checkpoint_ns)
            tup, planned = await self._aread_and_plan(cfg, serde, plan, method)
            if isinstance(planned, RewritePlan) and apply and tup is not None:
                await self._awrite_plan(cfg, tup, planned, method)
                after, again = await self._aread_and_plan(cfg, serde, plan, method)
                results.append(_written_outcome(tup, planned, after, again))
            else:
                results.append(_planned_outcome(planned))
        return _rewrite_report(apply, results, checkpoint_ns)

    async def _awrite_plan(
        self, cfg: RunnableConfig, tup: CheckpointTuple, plan: RewritePlan, method: str
    ) -> None:
        """Async sibling of :meth:`_write_plan`."""
        saver = self._checkpointer
        await saver.aput(
            self._rewrite_base_config(cfg, tup),
            plan.checkpoint,
            tup.metadata,
            plan.new_versions,
        )
        for task_id, channel, value in plan.interrupt_writes:
            await saver.aput_writes(tup.config, [(channel, value)], task_id)

    async def _aread_and_plan(
        self,
        cfg: RunnableConfig,
        serde: NamespaceAwareSerde,
        plan: _PlanThread,
        method: str,
    ) -> tuple[CheckpointTuple | None, ThreadRewrite | RewritePlan]:
        """Async sibling of :meth:`_read_and_plan`."""
        decoded: list[DecodedValue] = []
        with serde._record_reads(decoded):
            tup, unresolved = await self._aread_checkpoint_tuple(cfg, method)
        return tup, _plan_read(plan, cfg, tup, unresolved, decoded)
