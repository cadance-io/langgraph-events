"""Tests for EventLog query container."""

import pickle

import pytest
from conftest import Order

from langgraph_events import (
    Cause,
    Event,
    EventLog,
    FrameworkEvent,
    IntegrationEvent,
    NotRecorded,
    SourceDropped,
    UnknownCause,
)


class Alpha(IntegrationEvent):
    v: int = 0


class Beta(IntegrationEvent):
    v: int = 0


class AlphaChild(Alpha):
    extra: str = ""


def describe_EventLog():

    @pytest.fixture
    def log():
        return EventLog([Alpha(v=1), Beta(v=2), Alpha(v=3)])

    def describe_filter():

        def when_type_present():

            def it_returns_matching_events(log):
                assert log.filter(Alpha) == [Alpha(v=1), Alpha(v=3)]
                assert log.filter(Beta) == [Beta(v=2)]

            def it_returns_all_for_base_Event_type(log):
                assert log.filter(Event) == [Alpha(v=1), Beta(v=2), Alpha(v=3)]

        def when_inheritance():

            def it_includes_child_instances():
                log = EventLog([Alpha(v=1), AlphaChild(v=2, extra="x")])
                result = log.filter(Alpha)
                assert len(result) == 2
                assert isinstance(result[1], AlphaChild)

    def describe_latest():

        def when_match_exists():

            def it_returns_most_recent_match(log):
                assert log.latest(Alpha) == Alpha(v=3)
                assert log.latest(Beta) == Beta(v=2)

        def when_no_match():

            def it_returns_none():
                log = EventLog([Alpha(v=1)])
                assert log.latest(Beta) is None

    def describe_has():

        def it_returns_true_for_present_type():
            log = EventLog([Alpha(v=1)])
            assert log.has(Alpha) is True

        def it_returns_false_for_absent_type():
            log = EventLog([Alpha(v=1)])
            assert log.has(Beta) is False

        def it_returns_true_for_base_Event_type():
            log = EventLog([Alpha(v=1)])
            assert log.has(Event) is True

    def describe_first():

        def when_match_exists():

            def it_returns_first_match(log):
                assert log.first(Alpha) == Alpha(v=1)
                assert log.first(Beta) == Beta(v=2)

        def when_no_match():

            def it_returns_none():
                log = EventLog([Alpha(v=1)])
                assert log.first(Beta) is None

    def describe_count():

        def it_counts_matching_events(log):
            assert log.count(Alpha) == 2
            assert log.count(Beta) == 1

        def it_returns_zero_for_absent_type(log):
            class Gamma(IntegrationEvent):
                pass

            assert log.count(Gamma) == 0

    def describe_after():

        def when_type_present():

            def it_returns_events_after_first_occurrence(log):
                result = log.after(Alpha)
                assert list(result) == [Beta(v=2), Alpha(v=3)]

            def it_supports_chaining(log):
                result = log.after(Alpha).latest(Alpha)
                assert result == Alpha(v=3)

        def when_type_absent():

            def it_returns_empty_log(log):
                class Gamma(IntegrationEvent):
                    pass

                result = log.after(Gamma)
                assert len(result) == 0
                assert isinstance(result, EventLog)

    def describe_before():

        def when_type_present():

            def it_returns_events_before_first_occurrence(log):
                result = log.before(Beta)
                assert list(result) == [Alpha(v=1)]

        def when_type_absent():

            def it_returns_empty_log(log):
                class Gamma(IntegrationEvent):
                    pass

                result = log.before(Gamma)
                assert len(result) == 0
                assert isinstance(result, EventLog)

    def describe_select():

        def it_returns_event_log_of_matching_events(log):
            result = log.select(Alpha)
            assert isinstance(result, EventLog)
            assert list(result) == [Alpha(v=1), Alpha(v=3)]

        def it_filters_after_anchor():
            log = EventLog([Alpha(v=1), Beta(v=2), Alpha(v=3), Beta(v=4)])
            result = log.after(Alpha).select(Beta)
            assert list(result) == [Beta(v=2), Beta(v=4)]

    def describe_events():

        def it_returns_a_tuple_of_all_events(log):
            assert log.events == (Alpha(v=1), Beta(v=2), Alpha(v=3))
            assert isinstance(log.events, tuple)

        def it_matches_iteration_order(log):
            assert list(log.events) == list(log)

    def describe_container_protocol():

        def when_empty():

            def it_reports_zero_length():
                assert len(EventLog([])) == 0

            def it_is_falsy():
                assert not EventLog([])

        def when_nonempty():

            def it_reports_length():
                assert len(EventLog([Alpha(), Beta()])) == 2

            def it_is_truthy():
                assert EventLog([Alpha()])

            def it_iterates_events():
                events = [Alpha(v=1), Beta(v=2)]
                log = EventLog(events)
                assert list(log) == events

            def it_supports_indexing_and_negative_indexing(log):
                assert log[0] == Alpha(v=1)
                assert log[-1] == Alpha(v=3)

            def it_supports_slicing(log):
                assert log[1:3] == [Beta(v=2), Alpha(v=3)]

    def describe_repr():

        def when_small_log():

            def it_includes_EventLog_name():
                log = EventLog([Alpha(v=1)])
                assert "EventLog" in repr(log)

            def it_shows_events():
                log = EventLog([Alpha(v=1), Beta(v=2)])
                r = repr(log)
                assert "Alpha" in r
                assert "Beta" in r

        def when_exactly_5_events():

            def it_uses_full_repr():
                events = [Alpha(v=i) for i in range(5)]
                log = EventLog(events)
                r = repr(log)
                # 5 events → full form with individual event reprs
                assert "v=0" in r
                assert "events" not in r  # no truncated "N events" form

        def when_exactly_6_events():

            def it_uses_truncated_repr():
                events = [Alpha(v=i) for i in range(5)]
                events.append(Beta(v=99))
                log = EventLog(events)
                r = repr(log)
                # 6 events → truncated form
                assert "6 events" in r
                assert "v=0" not in r

        def when_large_log():

            def it_truncates():
                events = [Alpha(v=i) for i in range(10)]
                events.append(Beta(v=99))
                log = EventLog(events)
                r = repr(log)
                assert "11 events" in r
                assert "Alpha" in r
                assert "Beta" in r
                assert "v=0" not in r

    def describe_from_owned():

        def it_normalizes_to_immutable_tuple_storage():
            events = [Alpha(v=1), Beta(v=2)]
            log = EventLog._from_owned(events)
            assert isinstance(log._events, tuple)
            assert log._events == tuple(events)

        def it_produces_functionally_identical_log():
            events = [Alpha(v=1), Beta(v=2), Alpha(v=3)]
            log = EventLog._from_owned(list(events))
            assert log.filter(Alpha) == [Alpha(v=1), Alpha(v=3)]
            assert log.latest(Beta) == Beta(v=2)
            assert len(log) == 3

    def describe_events_caching():

        def it_returns_same_tuple_on_repeated_access():
            log = EventLog([Alpha(v=1), Beta(v=2)])
            first = log.events
            second = log.events
            assert first is second

    def describe_query_independence():

        def it_returns_independent_logs_from_after():
            log = EventLog([Alpha(v=1), Beta(v=2), Alpha(v=3)])
            sub = log.after(Alpha)
            assert list(sub) == [Beta(v=2), Alpha(v=3)]
            assert list(log) == [Alpha(v=1), Beta(v=2), Alpha(v=3)]

        def it_returns_independent_logs_from_select():
            log = EventLog([Alpha(v=1), Beta(v=2), Alpha(v=3)])
            sub = log.select(Alpha)
            assert list(sub) == [Alpha(v=1), Alpha(v=3)]
            assert len(log) == 3


def describe_EventLog_causes_argument():
    def when_a_source_comes_later():
        def it_raises_value_error():
            first = Alpha(v=1)
            later = Beta(v=2)

            with pytest.raises(ValueError, match="not an earlier event"):
                EventLog([first, later], causes=[Cause(later, "h"), None])

    def when_a_source_is_the_event_itself():
        def it_raises_value_error():
            seed = Alpha(v=1)

            with pytest.raises(ValueError, match="not an earlier event"):
                EventLog([seed], causes=[Cause(seed, "h")])

    def when_a_source_is_only_equal_to_an_earlier_event():
        def it_raises_value_error():
            seed = Alpha(v=1)

            with pytest.raises(ValueError, match="same object"):
                EventLog([seed, Beta(v=2)], causes=[None, Cause(Alpha(v=1), "h")])

    def when_via_is_not_a_string():
        def it_raises_type_error():
            with pytest.raises(TypeError, match="via must be a str"):
                Cause(Alpha(v=1), 3)  # type: ignore[arg-type]

    def when_a_source_is_a_copy():
        def it_says_to_pass_the_logged_event():
            seed = Alpha(v=1)

            with pytest.raises(ValueError, match=r"pass events\[j\]"):
                EventLog([seed, Beta(v=2)], causes=[None, Cause(Alpha(v=1), "h")])

    def when_an_entry_is_not_a_cause():
        def it_raises_type_error():
            with pytest.raises(
                TypeError,
                match="must be a Cause, an UnknownCause, a FrameworkEvent or None",
            ):
                EventLog([Alpha(v=1), Beta(v=2)], causes=[None, (0, "h")])

    def when_the_lengths_differ():
        def it_raises_value_error():
            with pytest.raises(ValueError, match="one entry per event"):
                EventLog([Alpha(v=1), Beta(v=2)], causes=[None])


def describe_causes():
    def when_causes_are_given():
        def it_returns_one_entry_per_event():
            seed = Alpha(v=1)

            log = EventLog([seed, Beta(v=2)], causes=[None, Cause(seed, "h")])

            assert log.causes == (None, Cause(source=seed, via="h"))

        def it_rebuilds_the_same_log():
            seed = Alpha(v=1)
            log = EventLog([seed, Beta(v=2)], causes=[None, Cause(seed, "h")])

            rebuilt = EventLog(log.events, causes=log.causes)

            assert rebuilt.causes == log.causes
            assert rebuilt.causes[1].source is seed

    def when_the_log_is_pickled():
        def it_keeps_its_causes():
            seed = Alpha(v=1)
            log = EventLog([seed, Beta(v=2)], causes=[None, Cause(seed, "h")])

            restored = pickle.loads(pickle.dumps(log))  # noqa: S301 - own data

            assert restored.causes == (None, Cause(source=Alpha(v=1), via="h"))
            assert restored.cause(restored[1]).source is restored[0]

    def when_one_event_object_sits_at_two_positions():
        def it_gives_each_position_its_own_cause():
            first, second, shared = Alpha(v=1), Alpha(v=2), Beta(v=9)
            log = EventLog(
                [first, second, shared, shared],
                causes=[None, None, Cause(first, "h"), Cause(second, "h")],
            )

            assert [c.source for c in log.causes[2:]] == [first, second]

    def when_causes_are_omitted():
        def it_is_none():
            assert EventLog([Alpha(v=1)]).causes is None

    def when_the_log_derives_from_a_log_that_has_causes():
        def it_keeps_the_causes_of_its_events():
            seed = Alpha(v=1)
            log = EventLog([seed, Beta(v=2)], causes=[None, Cause(seed, "h")])

            assert log.select(Beta).causes == (Cause(source=seed, via="h"),)
            assert log.before(Beta).causes == (None,)

        def it_cannot_rebuild_a_log_whose_source_is_outside_it():
            seed = Alpha(v=1)
            log = EventLog([seed, Beta(v=2)], causes=[None, Cause(seed, "h")])
            derived = log.select(Beta)

            with pytest.raises(ValueError, match="not an earlier event"):
                EventLog(derived.events, causes=derived.causes)


def _caused_log():
    """seed causes reply and other. reply causes echo."""
    seed = Alpha(v=1)
    reply = Beta(v=2)
    echo = Alpha(v=3)
    other = Beta(v=4)
    log = EventLog(
        [seed, reply, echo, other],
        causes=[
            None,
            Cause(seed, "reply_h"),
            Cause(reply, "echo_h"),
            Cause(seed, "other_h"),
        ],
    )
    return log, seed, reply, echo, other


def _repeated_log():
    """Two equal Alpha(v=1) objects. The second one has a cause."""
    first = Alpha(v=1)
    reply = Beta(v=2)
    again = Alpha(v=1)
    log = EventLog(
        [first, reply, again],
        causes=[None, Cause(first, "h"), Cause(reply, "g")],
    )
    return log, first, reply, again


def describe_cause():
    def when_the_event_has_a_cause():
        def it_returns_the_source_event_and_the_handler():
            log, seed, reply, _echo, _other = _caused_log()

            cause = log.cause(reply)

            assert cause == Cause(source=Alpha(v=1), via="reply_h")
            assert cause.source is seed

    def when_the_event_is_a_seed():
        def it_returns_none():
            log, seed, _reply, _echo, _other = _caused_log()

            assert log.cause(seed) is None

    def when_equal_events_repeat():
        def it_finds_each_instance_by_identity():
            log, first, reply, again = _repeated_log()

            assert log.cause(first) is None
            assert log.cause(again) == Cause(source=reply, via="g")

        def it_refuses_a_copy_that_matches_several_events():
            log, _first, _reply, _again = _repeated_log()

            with pytest.raises(ValueError, match="matches 2 equal events"):
                log.cause(Alpha(v=1))

    def when_a_copy_matches_one_event():
        def it_finds_that_event():
            log, seed, _reply, _echo, _other = _caused_log()

            assert log.cause(Beta(v=2)) == Cause(source=seed, via="reply_h")

    def when_the_event_is_not_in_the_log():
        def it_raises_value_error():
            log, _seed, _reply, _echo, _other = _caused_log()

            with pytest.raises(ValueError, match="not in this log"):
                log.cause(Beta(v=99))

    def when_the_log_records_no_causes():
        def it_raises_value_error():
            seed = Alpha(v=1)

            with pytest.raises(ValueError, match="records no causes"):
                EventLog([seed]).cause(seed)

    def when_the_log_derives_from_a_log():
        def it_answers_like_the_root():
            log, seed, reply, echo, other = _caused_log()

            assert log.after(Beta).cause(echo).source is reply
            assert log.select(Beta).cause(other).source is seed


def describe_effects():
    def when_the_event_caused_events():
        def it_returns_them_in_log_order():
            log, seed, reply, _echo, other = _caused_log()

            assert log.effects(seed) == (reply, other)

    def when_the_event_caused_nothing():
        def it_returns_an_empty_tuple():
            log, _seed, _reply, _echo, other = _caused_log()

            assert log.effects(other) == ()

    def when_the_log_records_no_causes():
        def it_raises_value_error():
            seed = Alpha(v=1)

            with pytest.raises(ValueError, match="records no causes"):
                EventLog([seed]).effects(seed)


def describe_flow():
    def when_the_event_has_a_cause_chain():
        def it_returns_the_chain_from_the_root_seed():
            log, seed, reply, echo, _other = _caused_log()

            assert log.flow(echo) == (seed, reply, echo)

    def when_the_event_is_a_seed():
        def it_returns_the_seed_alone():
            log, seed, _reply, _echo, _other = _caused_log()

            assert log.flow(seed) == (seed,)

    def when_the_log_records_no_causes():
        def it_raises_value_error():
            seed = Alpha(v=1)

            with pytest.raises(ValueError, match="records no causes"):
                EventLog([seed]).flow(seed)


def _every_case_log():
    seed = Alpha(v=1)
    entries = [
        None,
        NotRecorded(),
        SourceDropped(via="h", source_type="Gone"),
        FrameworkEvent(),
        Cause(seed, "h"),
    ]
    events = [seed, Beta(v=2), Beta(v=3), Beta(v=4), Beta(v=5)]
    return EventLog(events, causes=entries), entries


def describe_unknown_causes():
    def when_a_log_holds_every_case():
        def it_keeps_each_case_through_a_rebuild():
            log, entries = _every_case_log()

            rebuilt = EventLog(log.events, causes=log.causes)

            assert rebuilt.causes == tuple(entries)

        def it_answers_each_case_from_cause():
            log, entries = _every_case_log()

            assert [log.cause(e) for e in log] == entries

    def when_an_unknown_case_is_shown():
        def it_states_why_the_cause_is_unknown():
            dropped = SourceDropped(via="h", source_type="Gone")

            assert isinstance(NotRecorded(), UnknownCause)
            assert "before" in NotRecorded().reason
            assert isinstance(dropped, UnknownCause)
            assert "Gone" in dropped.reason
            assert "h" in dropped.reason

    def when_a_chain_reaches_an_unknown_cause():
        def it_stops_the_flow_there():
            old = Alpha(v=1)
            child = Beta(v=2)
            log = EventLog([old, child], causes=[NotRecorded(), Cause(old, "h")])

            assert log.flow(child) == (old, child)

        def it_counts_only_handler_causes_as_effects():
            log, _entries = _every_case_log()

            assert log.effects(log[0]) == (log[4],)


def describe_cause_lookup_errors():
    def when_a_nested_event_is_missing():
        def it_names_the_event_by_qualname():
            seed = Alpha(v=1)
            log = EventLog([seed], causes=[None])

            with pytest.raises(ValueError, match=r"Order\.Place"):
                log.cause(Order.Place(customer_id="c1"))
