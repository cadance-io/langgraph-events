"""The reducers and ``Derived`` values of a graph, and the values a stream shows."""

import dataclasses
import functools
import types
from typing import TYPE_CHECKING, Any

from langgraph_events._inputs import bind_derived
from langgraph_events._internal import _BASE_FIELDS
from langgraph_events._reducer import BaseReducer, Derived, StateRead
from langgraph_events._warn import warn_user

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

    from langgraph_events._causes import CauseEntry
    from langgraph_events._event import Event, Namespace
    from langgraph_events._reducer import Readable


def _kind(entry: BaseReducer | Derived[Any]) -> str:
    return "Derived" if isinstance(entry, Derived) else "reducer"


def _register(
    entries: dict[str, BaseReducer | Derived[Any]],
    entry: BaseReducer | Derived[Any],
    *,
    replace: bool,
) -> None:
    """Add *entry* to *entries* under its name.

    A reducer and a ``Derived`` of the same name raise ``TypeError`` from
    any source, because a handler parameter cannot name both. For two
    entries of the same kind, *replace* decides which one stays.
    """
    if not entry.name:
        raise ValueError(
            f"{_kind(entry)} {entry!r} has no name. Pass name=..., or declare "
            f"it as a Namespace class attribute."
        )
    existing = entries.get(entry.name)
    if existing is not None and _kind(existing) != _kind(entry):
        raise TypeError(
            f"The name {entry.name!r} is given to a reducer and a Derived. "
            f"Give each one its own name."
        )
    if existing is None or replace:
        entries[entry.name] = entry


def _discover(
    namespaces: Iterable[type[Namespace]],
) -> dict[str, BaseReducer | Derived[Any]]:
    """Each reducer and ``Derived`` that *namespaces* declare, by name.

    Two different declarations of one name raise ``TypeError``.
    """
    discovered: dict[str, BaseReducer | Derived[Any]] = {}
    for namespace_cls in namespaces:
        for entry in namespace_cls.__reducers__:
            existing = discovered.get(entry.name)
            if existing is None:
                discovered[entry.name] = entry
            elif existing is not entry:
                raise TypeError(
                    f"Reducer name {entry.name!r} collides between domains: "
                    f"{existing!r} and {entry!r}"
                )
    return discovered


@dataclasses.dataclass(frozen=True)
class ReducerRegistry:
    """The stored reducers and ``Derived`` values of one graph.

    *readables* maps each name that a handler can read to the stored
    reducer or the bound ``Derived`` that gives its value, in registration
    order.
    """

    reducers: Mapping[str, BaseReducer]
    derived: Mapping[str, Derived[Any]]
    readables: Mapping[str, Readable]

    @classmethod
    def build(
        cls,
        explicit: Sequence[BaseReducer | Derived[Any]],
        namespaces: Iterable[type[Namespace]],
        context_schema: type | None = None,
    ) -> ReducerRegistry:
        """Register *explicit*, then each declaration of *namespaces*.

        *context_schema* is the run context type of the graph. A ``Derived``
        parameter annotated with it is refused, because a ``Derived`` reads
        no run input.

        In *explicit*, a later entry of the same kind replaces an earlier
        one. An explicit entry wins over a declaration of the same kind.
        Raises ``TypeError`` for a reducer and a ``Derived`` of one name,
        and for a reducer that declares no stored type. Raises
        ``ValueError`` for an empty or a reserved name.
        """
        entries: dict[str, BaseReducer | Derived[Any]] = {}
        for entry in explicit:
            _register(entries, entry, replace=True)
        for entry in _discover(namespaces).values():
            _register(entries, entry, replace=False)
        reserved = sorted(entries.keys() & _BASE_FIELDS.keys())
        if reserved:
            kinds = " and ".join(sorted({_kind(entries[n]) for n in reserved}))
            raise ValueError(
                f"{kinds[0].upper()}{kinds[1:]} name(s) {reserved} conflict with "
                f"reserved state fields"
            )
        reducers = {n: e for n, e in entries.items() if isinstance(e, BaseReducer)}
        for reducer in reducers.values():
            # Each reducer declares its stored type. Refuse a missing one
            # here, also when no reader names the reducer.
            reducer.stored_type()
        derived = {n: e for n, e in entries.items() if isinstance(e, Derived)}
        bound = bind_derived(reducers, derived, context_schema)
        readables: dict[str, Readable] = {
            name: bound[name] if isinstance(entry, Derived) else entry
            for name, entry in entries.items()
        }
        return cls(
            types.MappingProxyType(reducers),
            types.MappingProxyType(derived),
            types.MappingProxyType(readables),
        )

    def select(self, include_reducers: bool | list[str]) -> ReducerSelection:
        """The values that *include_reducers* selects.

        ``True`` selects every name in registration order. ``False`` or
        ``[]`` selects none. A list selects its known names in its order.
        An unknown name gets a warning and is skipped.
        """
        if include_reducers is True:
            return ReducerSelection(tuple(self.readables), self.readables)
        if not include_reducers:
            return ReducerSelection((), self.readables)
        unknown = set(include_reducers) - self.readables.keys()
        if unknown:
            warn_user(
                f"Unknown reducer name(s) {unknown} in include_reducers; "
                f"available: {set(self.readables)}",
            )
        names = tuple(n for n in include_reducers if n in self.readables)
        return ReducerSelection(names, self.readables)


@dataclasses.dataclass(frozen=True)
class ReducerSelection:
    """The reducer and ``Derived`` values that one stream shows.

    *names* holds the selected names in order. A stream must follow each
    stored channel in :attr:`channels` to compute the shown values.
    """

    names: tuple[str, ...]
    readables: Mapping[str, Readable] = dataclasses.field(repr=False)

    @functools.cached_property
    def reducers(self) -> dict[str, BaseReducer]:
        """The stored reducers that the shown values read, by name, in order."""
        return {r.name: r for name in self.names for r in self.readables[name].sources}

    @functools.cached_property
    def channels(self) -> tuple[str, ...]:
        """The stored channels that the shown values read, in order."""
        return tuple(self.reducers)

    @functools.cached_property
    def reads_log(self) -> bool:
        """Whether a shown value reads the ``EventLog``. A stream must then
        follow the ``events`` and ``causes`` channels too."""
        return any(self.readables[name]._reads_log for name in self.names)

    def values(self, state: Mapping[str, Any]) -> dict[str, Any]:
        """Each shown value, read or computed from the channels in *state*.

        An unset channel reads as the empty default of its reducer. A value
        that reads the log gets the ``EventLog`` of the ``events`` and
        ``causes`` channels of *state*.
        """
        read = StateRead(state)
        return {name: self.readables[name]._read(read) for name in self.names}

    def changed(self, channels: frozenset[str]) -> frozenset[str]:
        """The shown names whose value can change when one event changes
        *channels*. A value that reads the log changes on each event."""
        return frozenset(
            name
            for name in self.names
            if self.readables[name]._reads_log
            or any(r.name in channels for r in self.readables[name].sources)
        )


@dataclasses.dataclass
class StreamState:
    """The state that a per-event stream follows to show the selected values.

    It holds each stored channel that a shown value reads. When a shown
    value reads the log, it holds the ``events`` and ``causes`` channels
    too. It starts from a checkpoint, or from empty channels, and takes
    each event in the order of the stream.
    """

    selection: ReducerSelection
    state: dict[str, Any]

    @classmethod
    def start(
        cls, selection: ReducerSelection, checkpoint: Mapping[str, Any] | None
    ) -> StreamState:
        """The state of *checkpoint*, or empty channels when it is ``None``."""
        values = checkpoint or {}
        state = {
            name: values.get(name, r.empty) for name, r in selection.reducers.items()
        }
        if selection.reads_log:
            state["events"] = list(values.get("events", []))
            state["causes"] = list(values.get("causes") or [])
        return cls(selection, state)

    def apply(self, event: Event, cause: CauseEntry | None) -> frozenset[str]:
        """Take *event* with its stored *cause*. Give the shown names whose
        value can change."""
        if self.selection.reads_log:
            self.state["events"].append(event)
            self.state["causes"].append(cause)
        return self._fold(event, store=True)

    def changes(self, event: Event) -> frozenset[str]:
        """The shown names whose value *event* can change. The state does not
        take *event*."""
        return self._fold(event, store=False)

    def values(self) -> dict[str, Any]:
        """Each shown value in this state."""
        return self.selection.values(self.state)

    def _fold(self, event: Event, *, store: bool) -> frozenset[str]:
        changed: set[str] = set()
        for name, r in self.selection.reducers.items():
            contribution = r.collect([event])
            if not r.has_contributions(contribution):
                continue
            changed.add(name)
            if store:
                merge = getattr(r, "reducer", None)
                self.state[name] = (
                    merge(self.state[name], contribution)
                    if callable(merge)
                    else contribution
                )
        return self.selection.changed(frozenset(changed))
