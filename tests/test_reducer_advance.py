"""``BaseReducer.advance``: fold events through the channel merge."""

from __future__ import annotations

from typing import Any

from conftest import Noted, keyed_merge

from langgraph_events import (
    RESET,
    SKIP,
    BaseReducer,
    Event,
    FoldReducer,
    IntegrationEvent,
    Reducer,
    ScalarReducer,
)


class Cleared(IntegrationEvent):
    pass


class _LastNote(BaseReducer):
    """A reducer whose channel annotation carries no merge."""

    name = "last"
    event_type = Noted
    namespace = None

    def state_annotation(self) -> Any:
        return str

    @property
    def empty(self) -> Any:
        return ""

    def collect(self, events: list[Event]) -> Any:
        texts = [e.text for e in events if isinstance(e, Noted)]
        return texts[-1] if texts else None

    def has_contributions(self, result: Any) -> bool:
        return result is not None

    def output_type(self) -> Any:
        return str

    def seed(self, events: list[Event]) -> Any:
        return self.collect(events) or self.empty


def describe_empty():
    def it_is_a_property_that_gives_a_fresh_value():
        assert isinstance(BaseReducer.__dict__["empty"], property)
        notes = Reducer("notes", event_type=Noted, fn=lambda e: [e.text])
        assert notes.empty == []
        assert notes.empty is not notes.empty


def describe_advance():
    def when_the_reducer_declares_a_custom_merge():
        def it_folds_through_the_merge_not_through_seed():
            notes = Reducer(
                "notes",
                event_type=Noted,
                fn=lambda e: [[e.key, e.text]],
                reducer=keyed_merge,
            )
            events = [Noted(key="a", text="1"), Noted(key="a", text="2")]
            assert notes.advance(notes.empty, events) == [["a", "2"]]
            assert notes.seed(events) == [["a", "1"], ["a", "2"]]

    def when_the_events_arrive_in_two_batches():
        def it_equals_one_advance_over_both():
            total = FoldReducer(
                "total",
                event_type=Noted,
                default_factory=int,
                fold=lambda s, e: s + 1,
            )
            first, second = [Noted()] * 2, [Noted()] * 3
            stepped = total.advance(total.advance(total.empty, first), second)
            assert stepped == total.advance(total.empty, first + second) == 5

    def when_no_event_contributes():
        def it_returns_the_state_unchanged():
            even = ScalarReducer("even", event_type=Noted, fn=lambda e: SKIP)
            state = object()
            assert even.advance(state, [Noted()]) is state

    def when_a_fold_returns_reset():
        def it_starts_again_from_the_default():
            total = FoldReducer(
                "total",
                event_type=(Noted, Cleared),
                default_factory=int,
                fold=lambda s, e: RESET if isinstance(e, Cleared) else s + 1,
            )
            assert total.advance(7, [Noted(), Cleared(), Noted()]) == 1

    def when_the_annotation_carries_no_merge():
        def it_keeps_the_last_write():
            last = _LastNote()
            assert last.advance("old", [Noted(text="a"), Noted(text="b")]) == "b"
