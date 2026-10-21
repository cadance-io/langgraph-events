"""Static typing assertions for the ``EventLog`` queries.

These are checked by **mypy**, not pytest. The bodies are typing-only.
Run: ``uv run mypy tests/test_event_log_typing.py``

``latest(..., not_followed_by=...)`` accepts one Event class, one event mixin,
or a union of them. Its result keeps the type of the first argument, so a caller
reads the fields of the match without a cast. A type cannot say "event mixin",
so mypy accepts any class there, and the runtime check refuses the rest.
"""

from __future__ import annotations

from typing import assert_type

from langgraph_events import Auditable, EventLog, IntegrationEvent


class Ordered(IntegrationEvent):
    sku: str = ""


class Shipped(IntegrationEvent):
    pass


class Refunded(IntegrationEvent):
    pass


def latest_not_followed_by_a_union_keeps_the_match_type(log: EventLog) -> None:
    assert_type(log.latest(Ordered, not_followed_by=Shipped | Refunded), Ordered | None)


def latest_not_followed_by_one_class_keeps_the_match_type(log: EventLog) -> None:
    assert_type(log.latest(Ordered, not_followed_by=Shipped), Ordered | None)


def latest_not_followed_by_a_mixin_keeps_the_match_type(log: EventLog) -> None:
    assert_type(log.latest(Ordered, not_followed_by=Auditable), Ordered | None)


def latest_not_followed_by_a_union_with_a_mixin_keeps_the_match_type(
    log: EventLog,
) -> None:
    result = log.latest(Ordered, not_followed_by=Shipped | Auditable)
    assert_type(result, Ordered | None)
