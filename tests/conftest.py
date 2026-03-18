"""Shared fixtures and event classes for the test suite."""

import pytest

from langgraph_events import Event, EventGraph, on

# ---------------------------------------------------------------------------
# Reusable event classes (used across multiple test files)
# ---------------------------------------------------------------------------


class Started(Event):
    data: str = ""


class Processed(Event):
    data: str = ""


class Ended(Event):
    result: str = ""


class MessageReceived(Event):
    text: str = ""


class MessageSent(Event):
    text: str = ""


class Completed(Event):
    result: str = ""


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def linear_chain():
    """A simple Started -> Processed -> Ended three-step EventGraph."""

    @on(Started)
    def step1(event: Started) -> Processed:
        return Processed(data=f"processed:{event.data}")

    @on(Processed)
    def step2(event: Processed) -> Ended:
        return Ended(result=f"done:{event.data}")

    return EventGraph([step1, step2])
