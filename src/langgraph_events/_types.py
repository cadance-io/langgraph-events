"""Shared type aliases for langgraph-events."""

from __future__ import annotations

import types
from collections.abc import Callable
from typing import Any, TypeVar

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
