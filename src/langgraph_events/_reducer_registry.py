"""The reducers and ``Derived`` values of a graph, and the values a stream shows."""

from __future__ import annotations

import dataclasses
import functools
import types
from typing import TYPE_CHECKING, Any

from langgraph_events._internal import _BASE_FIELDS
from langgraph_events._reducer import BaseReducer, Derived
from langgraph_events._warn import warn_user

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

    from langgraph_events._event import Namespace
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
    ) -> ReducerRegistry:
        """Register *explicit*, then each declaration of *namespaces*.

        In *explicit*, a later entry of the same kind replaces an earlier
        one. An explicit entry wins over a declaration of the same kind.
        Raises ``TypeError`` for a reducer and a ``Derived`` of one name,
        and ``ValueError`` for an empty or a reserved name.
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
        derived = {n: e for n, e in entries.items() if isinstance(e, Derived)}
        readables: dict[str, Readable] = {
            name: entry._bind(reducers, derived)
            if isinstance(entry, Derived)
            else entry
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
    def channels(self) -> tuple[str, ...]:
        """The stored channels that the shown values read, in order."""
        return tuple(
            dict.fromkeys(
                r.name for name in self.names for r in self.readables[name].sources
            )
        )

    def values(self, state: Mapping[str, Any]) -> dict[str, Any]:
        """Each shown value, read or computed from the channels in *state*.

        An unset channel reads as the empty default of its reducer.
        """
        return {name: self.readables[name].read(state) for name in self.names}

    def changed(self, channels: frozenset[str]) -> frozenset[str]:
        """The shown names whose value can change when *channels* change."""
        return frozenset(
            name
            for name in self.names
            if any(r.name in channels for r in self.readables[name].sources)
        )
