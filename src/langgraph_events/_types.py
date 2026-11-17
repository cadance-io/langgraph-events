"""Shared type aliases for langgraph-events."""

import types
from collections.abc import Callable
from typing import Any, Protocol, TypeVar

from langgraph_events._event import Event, Scatter

#: Return type of an @on handler: a single Event, a Scatter, or None.
HandlerReturn = Event | Scatter | None

#: Accepted forms for a reducer's ``event_type`` filter. A single type, a
#: ``A | B`` union, or a tuple of types — all valid second args to
#: ``isinstance``.
EventTypeSpec = type | types.UnionType | tuple[type, ...]

#: Internal alias for LangGraph state dictionaries.
StateDict = dict[str, Any]

#: LangGraph-compatible reducer function (e.g. operator.add, add_messages).
ReducerFn = Callable[[list, list], list]

#: TypeVar for the @on decorator — preserves the decorated function's type.
F = TypeVar("F", bound=Callable[..., Any])


class PauseSignal(Protocol):
    """A ``pause_on=`` or ``stop_on=`` signal: an object with ``is_set()``.

    For ``pause_on=``, the router calls ``is_set()`` between rounds. For
    ``stop_on=``, the async handler node calls it while a handler runs.
    ``asyncio.Event`` and ``threading.Event`` satisfy this protocol.
    """

    def is_set(self) -> bool: ...
