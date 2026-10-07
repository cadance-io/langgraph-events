"""``@on(..., handles_command=True)``: one handler records for a command family."""

from __future__ import annotations

import pytest

from langgraph_events import (
    Command,
    CommandPrivacyError,
    DomainEvent,
    EventGraph,
    EventMixin,
    IntegrationEvent,
    Namespace,
    on,
)


class Minted(EventMixin):
    pass


class Notes(Namespace):
    class Jot(Command, Minted):
        text: str = ""

        class Jotted(DomainEvent):
            text: str = ""

    class Erase(Command, Minted):
        class Erased(DomainEvent):
            pass

    class Pin(Command):
        class Pinned(DomainEvent):
            pass


class Poked(IntegrationEvent):
    pass


@on(Minted, handles_command=True)
def record(event: Minted):
    if isinstance(event, Notes.Jot):
        return Notes.Jot.Jotted(text=event.text)
    return Notes.Erase.Erased()


@on(Minted)
def leak(event: Minted):
    return Notes.Jot.Jotted(text="leak")


@on(Minted, handles_command=True)
def cross(event: Minted):
    return Notes.Erase.Erased()


@on(Minted, handles_command=True)
def declared(event: Minted) -> Notes.Jot.Jotted | Notes.Erase.Erased | None:
    return None


@on(Minted, handles_command=True)
def foreign(event: Minted) -> Notes.Pin.Pinned | None:
    return None


def describe_handles_command():
    def when_the_handler_emits_the_outcome_of_the_command_it_received():
        def it_records_each_command_of_the_family():
            graph = EventGraph([record])
            assert graph.invoke(Notes.Jot(text="a")).latest(
                Notes.Jot.Jotted
            ) == Notes.Jot.Jotted(text="a")
            assert graph.invoke(Notes.Erase()).latest(Notes.Erase.Erased) is not None

    def when_the_flag_is_absent():
        def it_raises_command_privacy_error():
            with pytest.raises(CommandPrivacyError, match=r"private to Notes\.Jot"):
                EventGraph([leak]).invoke(Notes.Jot(text="a"))

    def when_the_outcome_belongs_to_another_command():
        def it_raises_command_privacy_error_at_run_time():
            with pytest.raises(CommandPrivacyError, match=r"private to Notes\.Erase"):
                EventGraph([cross]).invoke(Notes.Jot(text="a"))

        def it_raises_at_build_for_a_declared_outcome_outside_the_family():
            with pytest.raises(CommandPrivacyError, match=r"private to Notes\.Pin"):
                EventGraph([foreign])

    def when_the_declared_outcomes_belong_to_the_family():
        def it_builds():
            assert EventGraph([declared]).handler_names == {"declared"}

    def when_the_subscription_is_not_a_command():
        def it_refuses_the_decorator():
            with pytest.raises(TypeError, match="not a Command or an EventMixin"):
                on(Poked, handles_command=True)

        def it_refuses_a_value_that_is_not_a_bool():
            with pytest.raises(TypeError, match="must be a bool"):
                on(Minted, handles_command="yes")  # type: ignore[arg-type]
