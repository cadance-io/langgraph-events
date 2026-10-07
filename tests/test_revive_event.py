"""``NamespaceAwareSerde.revive_event``: one stored event, one revival rule."""

from __future__ import annotations

import pytest
from _note_fixture import Note

from langgraph_events import Event, IntegrationEvent
from langgraph_events.serde import NamespaceAwareSerde, UnrevivedIdentity

NOTES = Note.__module__


class Elsewhere(IntegrationEvent):
    text: str = ""


def describe_revive_event():
    def when_the_identity_is_current():
        def it_returns_the_live_event():
            serde = NamespaceAwareSerde(events=[Note])
            revived = serde.revive_event(NOTES, "Note", {"text": "hi", "tag": "t"})
            assert revived == Note(text="hi", tag="t")

        def it_leaves_the_caller_mapping_unchanged():
            fields = {"text": "hi"}
            NamespaceAwareSerde(events=[Note]).revive_event(NOTES, "OldNote", fields)
            assert fields == {"text": "hi"}

    def when_the_identity_is_historic():
        def it_applies_the_rename_and_the_backfill():
            serde = NamespaceAwareSerde(events=[Note])
            revived = serde.revive_event(NOTES, "OldNote", {"text": "hi"})
            assert revived == Note(text="hi", tag="legacy")

    def when_a_resolver_is_given():
        def it_asks_the_resolver_first():
            asked: list[tuple[str, str]] = []

            def resolve(module: str, qualname: str) -> type[Event] | None:
                asked.append((module, qualname))
                return Elsewhere

            revived = NamespaceAwareSerde().revive_event(
                "gone.module", "Runtime.Widget", {"text": "w"}, resolve=resolve
            )
            assert revived == Elsewhere(text="w")
            assert asked == [("gone.module", "Runtime.Widget")]

        def it_falls_back_to_the_import_walk_on_none():
            revived = NamespaceAwareSerde().revive_event(
                __name__, "Elsewhere", {"text": "w"}, resolve=lambda m, q: None
            )
            assert revived == Elsewhere(text="w")

    def when_the_identity_does_not_resolve():
        def it_raises_a_message_naming_the_remedy():
            with pytest.raises(ValueError, match=r"Cannot revive gone\.module\.Gone"):
                NamespaceAwareSerde().revive_event("gone.module", "Gone", {})

        def it_raises_on_a_field_the_class_dropped():
            with pytest.raises(ValueError, match="Cannot revive"):
                NamespaceAwareSerde().revive_event(
                    __name__, "Elsewhere", {"text": "w", "extra": 1}
                )

        def it_degrades_inside_tolerate_unresolved():
            serde = NamespaceAwareSerde()
            with serde.tolerate_unresolved() as missing:
                revived = serde.revive_event("gone.module", "Gone", {"a": 1})
            placeholder = UnrevivedIdentity(module="gone.module", qualname="Gone")
            assert revived == placeholder
            assert missing == [placeholder]
