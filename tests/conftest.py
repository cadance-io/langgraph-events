"""Shared fixtures and event classes for the test suite."""

import pytest

from langgraph_events import Event, EventGraph, on

# ---------------------------------------------------------------------------
# Reusable event classes (used across multiple test files)
# ---------------------------------------------------------------------------


class Start(Event):
    data: str = ""


class Middle(Event):
    data: str = ""


class End(Event):
    result: str = ""


class MsgIn(Event):
    text: str = ""


class MsgOut(Event):
    text: str = ""


class Done(Event):
    result: str = ""


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def linear_chain():
    """A simple Start -> Middle -> End three-step EventGraph."""

    @on(Start)
    def step1(event: Start) -> Middle:
        return Middle(data=f"processed:{event.data}")

    @on(Middle)
    def step2(event: Middle) -> End:
        return End(result=f"done:{event.data}")

    return EventGraph([step1, step2])
