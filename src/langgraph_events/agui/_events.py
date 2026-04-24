"""Built-in events for the AG-UI adapter."""

from __future__ import annotations

from dataclasses import field
from typing import Any

from langgraph_events import IntegrationEvent


class FrontendStateMutated(IntegrationEvent):
    """Emitted by ``AGUIAdapter`` at the start of a run carrying
    ``RunAgentInput.state``.

    The ``state`` field is the client-owned reducer snapshot with dedicated
    AG-UI keys (``messages``) stripped.  Reducers that mirror client-driven
    channels subscribe to this event like any other::

        from langgraph_events import ScalarReducer, SKIP
        from langgraph_events.agui import FrontendStateMutated

        focus = ScalarReducer(
            name="focus",
            event_type=FrontendStateMutated,
            fn=lambda e: e.state.get("focus", SKIP),
        )

    Handlers may also subscribe via ``@on(FrontendStateMutated)`` to react
    to client-state changes.  The adapter does not echo this event back to
    the client; downstream reducer changes surface through the usual
    ``StateSnapshotEvent`` path.
    """

    state: dict[str, Any] = field(default_factory=dict)
