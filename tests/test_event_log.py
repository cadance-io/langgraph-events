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
