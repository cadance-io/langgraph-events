"""Tests for EventLog query container."""

from dataclasses import dataclass

from langgraph_events import Event, EventLog


@dataclass(frozen=True)
class Alpha(Event):
    v: int = 0


@dataclass(frozen=True)
class Beta(Event):
    v: int = 0


@dataclass(frozen=True)
class AlphaChild(Alpha):
    extra: str = ""


def test_filter():
    log = EventLog([Alpha(v=1), Beta(v=2), Alpha(v=3)])
    assert log.filter(Alpha) == [Alpha(v=1), Alpha(v=3)]
    assert log.filter(Beta) == [Beta(v=2)]
    assert log.filter(Event) == [Alpha(v=1), Beta(v=2), Alpha(v=3)]


def test_filter_with_inheritance():
    log = EventLog([Alpha(v=1), AlphaChild(v=2, extra="x")])
    # AlphaChild IS an Alpha
    result = log.filter(Alpha)
    assert len(result) == 2
    assert isinstance(result[1], AlphaChild)


def test_latest():
    log = EventLog([Alpha(v=1), Beta(v=2), Alpha(v=3)])
    assert log.latest(Alpha) == Alpha(v=3)
    assert log.latest(Beta) == Beta(v=2)


def test_latest_none():
    log = EventLog([Alpha(v=1)])
    assert log.latest(Beta) is None


def test_has():
    log = EventLog([Alpha(v=1)])
    assert log.has(Alpha) is True
    assert log.has(Beta) is False
    assert log.has(Event) is True


def test_len():
    assert len(EventLog([])) == 0
    assert len(EventLog([Alpha(), Beta()])) == 2


def test_bool():
    assert not EventLog([])
    assert EventLog([Alpha()])


def test_iter():
    events = [Alpha(v=1), Beta(v=2)]
    log = EventLog(events)
    assert list(log) == events


def test_getitem():
    events = [Alpha(v=1), Beta(v=2), Alpha(v=3)]
    log = EventLog(events)
    assert log[0] == Alpha(v=1)
    assert log[-1] == Alpha(v=3)
    assert log[1:3] == [Beta(v=2), Alpha(v=3)]


def test_repr():
    log = EventLog([Alpha(v=1)])
    assert "EventLog" in repr(log)
