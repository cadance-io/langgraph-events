"""AG-UI Frontend Tools (handler-initiated) — langgraph-events demo.

Shows the event-native path: a handler returns ``FrontendToolCallRequested``,
the adapter emits ``ToolCallStart``/``ToolCallArgs``/``ToolCallEnd`` for the
frontend's ``useFrontendTool`` handler to pick up, and the graph pauses via
the existing ``Interrupted`` machinery.  When the frontend returns its tool
message, ``detect_new_tool_results`` surfaces it and the resume factory
continues the graph with ``ToolsExecuted``.

This mirrors the ``ApprovalRequested(Interrupted)`` pattern from
``examples/human_in_the_loop.py`` — tool calls become "HITL with typed
fields," exactly as the AG-UI spec positions them.

Frontend (React + CopilotKit v2) — register a matching tool::

    import { useFrontendTool } from "@copilotkit/react-core";
    import { z } from "zod";

    useFrontendTool({
      name: "confirm",
      description: "Ask the user to confirm an action.",
      parameters: z.object({ prompt: z.string() }),
      render: ({ args }) => <ConfirmDialog prompt={args.prompt} />,
      handler: async ({ prompt }) => {
        const ok = window.confirm(prompt);
        return JSON.stringify({ approved: ok });
      },
    });
"""

from __future__ import annotations

import asyncio
import contextlib
import json
from typing import TYPE_CHECKING

from langgraph.checkpoint.memory import MemorySaver

from langgraph_events import (
    Auditable,
    EventGraph,
    FrontendToolCallRequested,
    MessageEvent,
    message_reducer,
    on,
)
from langgraph_events.agui import AGUIAdapter, detect_new_tool_results

if TYPE_CHECKING:
    from langchain_core.messages import ToolMessage

# ---------------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------------


class ShipCommandReceived(Auditable):
    """A user request to ship something, pending confirmation."""

    release: str = ""


class UserConfirmed(MessageEvent, Auditable):
    """Resume event — the frontend tool message arrived."""

    messages: tuple[ToolMessage, ...] = ()


class ShippedRelease(Auditable):
    release: str = ""
    approved: bool = False


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------


@on(Auditable)
def audit_trail(event: Auditable) -> None:
    print(f"  {event.trail()}")


@on(ShipCommandReceived)
def request_confirmation(event: ShipCommandReceived) -> FrontendToolCallRequested:
    return FrontendToolCallRequested(
        name="confirm",
        args={"prompt": f"Ship release {event.release}?"},
    )


@on(UserConfirmed)
def ship(event: UserConfirmed) -> ShippedRelease:
    # The resume event carries the tool message; parse approval from its
    # content (JSON string returned by the frontend handler).
    approved = False
    for msg in event.messages:
        with contextlib.suppress(json.JSONDecodeError):
            approved = bool(json.loads(msg.content or "{}").get("approved"))
    return ShippedRelease(release="v2026-04-19", approved=approved)


# ---------------------------------------------------------------------------
# Graph + adapter
# ---------------------------------------------------------------------------


graph = EventGraph(
    [request_confirmation, ship, audit_trail],
    reducers=[message_reducer()],
    checkpointer=MemorySaver(),
)


def seed_factory(input_data, checkpoint_state=None):
    del input_data, checkpoint_state
    return ShipCommandReceived(release="v2026-04-19")


def resume_factory(input_data, checkpoint_state=None):
    results = detect_new_tool_results(input_data, checkpoint_state)
    if not results:
        return None
    return UserConfirmed(messages=tuple(results))


adapter = AGUIAdapter(
    graph=graph,
    seed_factory=seed_factory,
    resume_factory=resume_factory,
)


# ---------------------------------------------------------------------------
# Stand-alone smoke driver (no HTTP)
# ---------------------------------------------------------------------------


async def _smoke() -> None:
    from ag_ui.core import RunAgentInput

    inp = RunAgentInput(
        thread_id="confirm-1",
        run_id="run-1",
        state={},
        messages=[],
        tools=[],
        context=[],
        forwarded_props={},
    )
    async for evt in adapter.stream(inp):
        print(evt.type, getattr(evt, "tool_call_name", ""))


if __name__ == "__main__":
    asyncio.run(_smoke())
