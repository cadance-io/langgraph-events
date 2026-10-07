"""Invariant predicates can receive their triggering event."""

from __future__ import annotations

import pytest

from langgraph_events import (
    Command,
    EventGraph,
    IntegrationEvent,
    Invariant,
    Namespace,
    on,
)


class Commands(Namespace):
    class Trigger(Command):
        pass


class Recorded(IntegrationEvent):
    pass


class Rule(Invariant):
    pass


two_argument_sources: list[Commands.Trigger] = []
one_argument_logs = []
post_check_sources: list[Commands.Trigger] = []


def two_argument_predicate(log, source_event):
    two_argument_sources.append(source_event)
    return True


def one_argument_predicate(log):
    one_argument_logs.append(log)
    return True


def type_error_predicate(log, source_event):
    raise TypeError("body failure")


def post_check_predicate(log, source_event):
    post_check_sources.append(source_event)
    return True


@on(Commands.Trigger, invariants={Rule: two_argument_predicate})
def handle_two_arguments(event: Commands.Trigger):
    return None


@on(Commands.Trigger, invariants={Rule: one_argument_predicate})
def handle_one_argument(event: Commands.Trigger):
    return None


@on(Commands.Trigger, invariants={Rule: type_error_predicate})
def handle_type_error(event: Commands.Trigger):
    return None


@on(Commands.Trigger, invariants={Rule: post_check_predicate})
def handle_post_check(event: Commands.Trigger) -> Recorded:
    return Recorded()


def describe_invariant_source_event():
    def when_a_predicate_accepts_two_positional_arguments():
        def it_receives_the_same_command_as_the_handler():
            two_argument_sources.clear()
            command = Commands.Trigger()
            EventGraph([handle_two_arguments]).invoke(command)
            assert len(two_argument_sources) == 1
            assert two_argument_sources[0] is command

    def when_a_predicate_accepts_one_positional_argument():
        def it_receives_the_log_and_permits_the_handler():
            one_argument_logs.clear()
            EventGraph([handle_one_argument]).invoke(Commands.Trigger())
            assert len(one_argument_logs) == 1
            assert one_argument_logs[0].has(Commands.Trigger)

    def when_a_predicate_body_raises_type_error():
        def it_propagates_the_error_unchanged():
            with pytest.raises(TypeError, match="body failure"):
                EventGraph([handle_type_error]).invoke(Commands.Trigger())

    def when_the_handler_emits_an_event():
        def it_receives_the_same_triggering_command_during_the_post_check():
            post_check_sources.clear()
            command = Commands.Trigger()
            EventGraph([handle_post_check]).invoke(command)
            assert len(post_check_sources) == 2
            assert all(source is command for source in post_check_sources)
