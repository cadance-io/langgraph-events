"""Built-in events for the AG-UI adapter."""

from __future__ import annotations

from dataclasses import field
from typing import Any

from langgraph_events import Event


class FrontendStateMutated(Event):
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

    **Resume-path semantics.**  On resume (``RunAgentInput`` carrying both
    ``state`` and a resume signal), ``FrontendStateMutated`` is injected as
    a seed alongside ``Resumed`` so reducers see it before the user's
    ``@on(Resumed, …)`` handler emits its domain event.  However,
    ``@on(FrontendStateMutated)`` *handlers* do not fire on resume — the
    LangGraph ``Command(resume=...)`` carries a single value and seeds are
    dispatched out-of-graph.  Use ``@on(Resumed)`` for resume-time side
    effects.
    """

    state: dict[str, Any] = field(default_factory=dict)
