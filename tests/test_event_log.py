"""Tests for EventLog query container."""

import pytest

from langgraph_events import Event, EventLog


class Alpha(Event):
    v: int = 0


class Beta(Event):
    v: int = 0


class AlphaChild(Alpha):
    extra: str = ""


def describe_EventLog():

    @pytest.fixture
    def log():
        return EventLog([Alpha(v=1), Beta(v=2), Alpha(v=3)])

    def describe_filter():

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
            class Gamma(Event):
                pass

            assert log.count(Gamma) == 0

    def describe_after():

        def it_returns_events_after_first_occurrence(log):
            result = log.after(Alpha)
            assert list(result) == [Beta(v=2), Alpha(v=3)]

        def it_returns_empty_log_when_type_absent(log):
            class Gamma(Event):
                pass

            result = log.after(Gamma)
            assert len(result) == 0
            assert isinstance(result, EventLog)

        def it_supports_chaining(log):
            result = log.after(Alpha).latest(Alpha)
            assert result == Alpha(v=3)

    def describe_before():

        def it_returns_events_before_first_occurrence(log):
            result = log.before(Beta)
            assert list(result) == [Alpha(v=1)]

        def it_returns_empty_log_when_type_absent(log):
            class Gamma(Event):
                pass

            result = log.before(Gamma)
            assert len(result) == 0
            assert isinstance(result, EventLog)

    def describe_select():

        def it_returns_event_log_of_matching_events(log):
            result = log.select(Alpha)
            assert isinstance(result, EventLog)
            assert list(result) == [Alpha(v=1), Alpha(v=3)]

        def it_supports_chaining_with_after():
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

        def it_reports_length():
            assert len(EventLog([])) == 0
            assert len(EventLog([Alpha(), Beta()])) == 2

        def it_is_falsy_when_empty():
            assert not EventLog([])

        def it_is_truthy_when_nonempty():
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

        def it_includes_EventLog_name():
            log = EventLog([Alpha(v=1)])
            assert "EventLog" in repr(log)

        def it_shows_events_for_small_logs():
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

        def it_truncates_for_large_logs():
            events = [Alpha(v=i) for i in range(10)]
            events.append(Beta(v=99))
            log = EventLog(events)
            r = repr(log)
            assert "11 events" in r
            assert "Alpha" in r
            assert "Beta" in r
            assert "v=0" not in r

    def describe_from_owned():

        def it_shares_the_same_list_object():
            events = [Alpha(v=1), Beta(v=2)]
            log = EventLog._from_owned(events)
            assert log._events is events

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
