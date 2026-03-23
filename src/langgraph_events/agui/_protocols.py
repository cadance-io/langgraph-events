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
class AGUICustomEvent(AGUISerializable, Protocol):
    """Events that control their AG-UI custom event name.

    Implement ``agui_event_name`` to override the default
    ``type(event).__name__`` used by ``FallbackMapper``.
    """

    @property
    def agui_event_name(self) -> str:
        """The custom event name for AG-UI forwarding."""
        ...


@runtime_checkable
class EventMapper(Protocol):
    """Maps a domain Event to zero or more AG-UI events.

    Return semantics:
    - ``None``  — "I don't handle this event, try next mapper"
    - ``[]``    — "I claim this event but suppress output"
    - ``[..]``  — "I claim this event, emit these AG-UI events"
    """

    def map(self, event: Any, ctx: MapperContext) -> list[BaseEvent] | None:
        """Map a domain event to AG-UI events."""
        ...


@runtime_checkable
class SeedFactory(Protocol):
    """Convert AG-UI RunAgentInput into domain seed event(s).

    Optionally accepts a second ``checkpoint_state`` dict containing
    reducer snapshots, events, messages, and interrupt info from the
    checkpoint.  The adapter detects the arity at runtime, so single-arg
    factories remain fully supported.
    """

    def __call__(
        self,
        input_data: RunAgentInput,
        checkpoint_state: dict[str, Any] | None = ...,
    ) -> Event | list[Event]:
        """Produce seed event(s) from an AG-UI run request."""
        ...


@runtime_checkable
class ResumeFactory(Protocol):
    """Detect whether an AG-UI request is a resume and produce the domain event.

    Optionally accepts a second ``checkpoint_state`` dict.  The adapter
    detects the arity at runtime, so single-arg factories remain fully
    supported.
    """

    def __call__(
        self,
        input_data: RunAgentInput,
        checkpoint_state: dict[str, Any] | None = ...,
    ) -> Event | None:
        """Return a domain Event to resume with, or None for a fresh run."""
        ...
