"""Built-in events for the AG-UI adapter."""

from __future__ import annotations

import uuid
from dataclasses import field
from typing import Any, Generic, TypeVar

from langgraph_events import IntegrationEvent
from langgraph_events._event import Interrupted


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

    **Resume-path semantics.**  On resume (``RunAgentInput`` carrying both
    ``state`` and a resume signal), reducers subscribed to
    ``FrontendStateMutated`` *do* fire — the adapter computes their
    contributions and writes them to channels via ``apre_seed`` before the
    resume's domain dispatch, so handlers reading reducer state via
    parameter injection see the updated values.  However,
    ``@on(FrontendStateMutated)`` *handlers* do not fire on resume — the
    LangGraph ``Command(resume=...)`` carries a single value and seeds are
    dispatched out-of-graph.  Use ``@on(Resumed)`` for resume-time side
    effects.
    """

    state: dict[str, Any] = field(default_factory=dict)


PayloadT = TypeVar("PayloadT")


class InterruptedWithPayload(Interrupted, Generic[PayloadT]):
    """Typed-payload variant of ``Interrupted`` for HITL with a discriminated UI.

    HITL projects whose frontend needs an action-discriminated payload
    (entity-review vs environment-select vs walkthrough-choice, …) can
    subclass this base and implement :meth:`interrupt_payload` to return
    the typed payload. Useful as a single anchor that multiple namespace
    modules can import without inventing a project-local shim base::

        from typing import TypedDict

        class ReviewPayload(TypedDict):
            kind: str
            draft: str

        class ReviewInterrupted(InterruptedWithPayload[ReviewPayload]):
            draft: str
            def interrupt_payload(self) -> ReviewPayload:
                return {"kind": "review", "draft": self.draft}

    Pure ``Interrupted`` remains the right choice when no payload is needed —
    this base is opt-in.
    """

    def interrupt_payload(self) -> PayloadT:
        """Return the typed payload for this interrupt.

        Subclasses must override.  The default raises ``NotImplementedError``
        so a forgotten override surfaces as a clear runtime error rather than
        silently returning ``None``.
        """
        raise NotImplementedError(
            f"{type(self).__qualname__} subclasses InterruptedWithPayload but "
            f"does not override interrupt_payload(). Implement it to return "
            f"the typed payload (matching the Generic parameter)."
        )


class FrontendToolCallRequested(Interrupted):
    """Request a frontend-executed tool call and pause the graph.

    Event-native counterpart to LLM-initiated tool calls: a handler returns
    this event, the AG-UI adapter emits ``ToolCallStart``/``ToolCallArgs``/
    ``ToolCallEnd`` for a frontend ``useFrontendTool`` handler to pick up,
    and the graph pauses via the existing ``Interrupted`` machinery.  Resume
    with a domain event (typically ``ToolsExecuted(messages=...)`` built via
    ``detect_new_tool_results`` from the frontend's tool-result message).

    Mirrors the ``ApprovalRequested(Interrupted)`` pattern — tool calls
    become "HITL with typed fields," exactly as the AG-UI spec positions
    them::

        FrontendToolCallRequested(name="confirm", args={"message": "Ship?"})
    """

    name: str
    args: dict[str, Any] = field(default_factory=dict)
    tool_call_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def __post_init__(self) -> None:
        if not self.name or not self.name.strip():
            raise ValueError(
                "FrontendToolCallRequested.name must be a non-empty tool name; "
                "got empty/whitespace. Pass the same `name` your "
                "useFrontendTool({ name: ... }) registration declares."
            )

    def agui_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "args": self.args,
            "tool_call_id": self.tool_call_id,
        }
