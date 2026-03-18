"""Protocol definitions for the AG-UI adapter layer."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from ag_ui.core import BaseEvent, RunAgentInput

    from langgraph_events._event import Event

    from ._context import MapperContext


@runtime_checkable
class AGUISerializable(Protocol):
    """Opt-in protocol for events that control their own AG-UI serialization."""

    def agui_dict(self) -> dict[str, Any]:
        """Return a JSON-safe dict representation for AG-UI serialization."""
        ...


@runtime_checkable
class EventMapper(Protocol):
    """Maps a domain Event to zero or more AG-UI events.

    Return semantics:
    - ``None``  — "I don't handle this event, try next mapper"
    - ``[]``    — "I claim this event but suppress output"
    - ``[..]``  — "I claim this event, emit these AG-UI events"
    """

    def map(self, event: Event, ctx: MapperContext) -> list[BaseEvent] | None:
        """Map a domain event to AG-UI events."""
        ...


@runtime_checkable
class SeedFactory(Protocol):
    """Convert AG-UI RunAgentInput into domain seed event(s)."""

    def __call__(self, input_data: RunAgentInput) -> Event | list[Event]:
        """Produce seed event(s) from an AG-UI run request."""
        ...


@runtime_checkable
class ResumeFactory(Protocol):
    """Detect whether an AG-UI request is a resume and produce the domain event."""

    def __call__(self, input_data: RunAgentInput) -> Event | None:
        """Return a domain Event to resume with, or None for a fresh run."""
        ...
