"""``EventMixin``: the public base that lets ``@on`` subscribe to a mixin."""

from __future__ import annotations

import pytest

from langgraph_events import (
    Auditable,
    EventGraph,
    EventMixin,
    IntegrationEvent,
    MessageEvent,
    on,
)


class Tracked(EventMixin):
    pass


class Pinged(IntegrationEvent, Tracked):
    pass


class Plain:
    pass


SEEN: list[str] = []


@on(Tracked)
def explicit(event: Tracked) -> None:
    SEEN.append("explicit")


@on
def inferred(event: Tracked) -> None:
    SEEN.append("inferred")


def describe_EventMixin():
    def when_a_handler_subscribes_to_a_mixin():
        def it_receives_every_event_that_carries_it():
            SEEN.clear()
            EventGraph([explicit, inferred]).invoke(Pinged())
            assert sorted(SEEN) == ["explicit", "inferred"]

    def when_the_class_is_neither_an_event_nor_a_mixin():
        def it_refuses_an_explicit_type():
            with pytest.raises(TypeError, match="Event subclasses or mixins"):
                on(Plain)

        def it_refuses_an_annotation():
            def handler(event: Plain) -> None:
                return None

            with pytest.raises(TypeError, match="Event subclass or mixin"):
                on(handler)

    def describe_the_library_mixins():
        def it_counts_message_event_and_auditable_as_mixins():
            assert issubclass(MessageEvent, EventMixin)
            assert issubclass(Auditable, EventMixin)
