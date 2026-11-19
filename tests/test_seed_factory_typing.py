"""Static typing assertions for ``AGUIAdapter(seed_factory=...)``.

These are checked by **mypy**, not pytest. The bodies are typing-only.
Run: ``uv run mypy tests/test_seed_factory_typing.py``

A seed factory takes the run input, and optionally the checkpoint state. A
type checker accepts both forms, a bound method included. Under strict
``warn_unused_ignores`` an unused ignore is an error, so the ignore below
proves that mypy rejects a factory that does not take the run input.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from langgraph_events import EventGraph, IntegrationEvent
from langgraph_events.agui import AGUIAdapter

if TYPE_CHECKING:
    from ag_ui.core import RunAgentInput


class Asked(IntegrationEvent):
    question: str = ""


class Seeds:
    def from_input(self, input_data: RunAgentInput, /) -> Asked:
        return Asked(question=input_data.run_id)


def seed_from_input(input_data: RunAgentInput) -> Asked:
    return Asked(question=input_data.run_id)


def seed_from_checkpoint(
    input_data: RunAgentInput, checkpoint_state: dict[str, Any] | None = None
) -> Asked:
    return Asked(question=input_data.run_id)


async def async_seed_from_input(input_data: RunAgentInput) -> Asked:
    return Asked(question=input_data.run_id)


def seed_from_number(input_data: int) -> Asked:
    return Asked()


def a_factory_of_the_input_only_is_accepted(graph: EventGraph) -> None:
    AGUIAdapter(graph, seed_factory=Seeds().from_input)
    AGUIAdapter(graph, seed_factory=seed_from_input)
    AGUIAdapter(graph, seed_factory=async_seed_from_input)


def a_factory_of_the_checkpoint_state_is_accepted(graph: EventGraph) -> None:
    AGUIAdapter(graph, seed_factory=seed_from_checkpoint)


def a_factory_of_another_input_is_rejected(graph: EventGraph) -> None:
    AGUIAdapter(graph, seed_factory=seed_from_number)  # type: ignore[arg-type]
