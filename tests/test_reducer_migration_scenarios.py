"""Exploratory tests for reducer migration scenarios.

The goal is **empirical, not prescriptive**. Each scenario captures a
realistic refactor that touches reducer state and verifies what actually
happens through ``NamespaceAwareSerde``. Tests that pass tell us the v1
migration system already covers that case (sometimes for unobvious
reasons); tests that fail or surface awkward behavior tell us where the
real gaps are.

Read this file as a survey, not a contract. The conclusion lives in
``docs/event-migrations.md``; here we just measure.

Scenarios in this file:

1. **list[Event]** reducer — one event class renamed
2. **ScalarReducer holding an Event instance** — class renamed
3. **dict[str, list[Event]] grouping** — one event class renamed
4. **Plain dataclass channel value** — new field added (with / without default)
5. **dict[str, int] reducer output** — shape changed structurally
6. **Projection function semantics changed** — silent stale data
7. **Replay-from-event-log via `replay_reducer`** — the universal recovery path
8. **Pydantic model + new required field** — silent malformed revival

Scenarios 1-4a exercise the transitive ext-hook coverage we ship for
free. Scenarios 4b, 5, 6, 8 surface silent-fail cases. Scenario 7
demonstrates ``replay_reducer`` — the documented recovery path for
every silent-fail case.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import ormsgpack
import pytest
from pydantic import BaseModel

from langgraph_events.serde import NamespaceAwareSerde
from langgraph_events.serde._jsonplus import EXT_NAMESPACE_AWARE_EVENT, _option
from langgraph_events.serde.migrations import Migration

# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


def _legacy_event_bytes(module: str, qualname: str, kwargs: dict[str, Any]) -> bytes:
    """Raw EXT_NAMESPACE_AWARE_EVENT payload as a prior version would emit."""
    inner = ormsgpack.packb((module, qualname, kwargs), option=_option)
    return ormsgpack.packb(
        ormsgpack.Ext(EXT_NAMESPACE_AWARE_EVENT, inner), option=_option
    )


# Module-level fixture classes — function-local would acquire ``<locals>`` in
# their qualname, which the strict migration validator rejects as a chain
# terminus.


@dataclasses.dataclass(frozen=True)
class OldStats:
    """Old reducer output shape — what a prior release would have written."""

    count: int
    last_seen: str


@dataclasses.dataclass(frozen=True)
class NewStats:
    """Current reducer output shape — adds ``version`` field."""

    count: int
    last_seen: str
    version: int = 1  # New required-with-default field


class OldPydanticState(BaseModel):
    """Old Pydantic-model reducer value."""

    count: int
    name: str


class NewPydanticRequiredField(BaseModel):
    """New Pydantic shape with a REQUIRED new field."""

    count: int
    name: str
    version: int  # required, no default


def describe_reducer_migration_scenarios():
    # -----------------------------------------------------------------------
    # 1. list[Event] reducer with one class renamed
    # -----------------------------------------------------------------------

    def describe_scenario_1_list_of_events_after_a_rename():
        def it_migrates_each_event_via_the_ext_hook():
            # Simulate a reducer that accumulated Event instances of an
            # `EventX` class that has since been renamed `EventY`. The
            # channel value is a Python list of two encoded events.
            from test_serde import Persona

            migrations = [
                Migration.rename(
                    name="rename-legacy-x",
                    old_module="legacy.x",
                    old_qualname="Legacy.X",
                    new_module=Persona.__module__,
                    new_qualname="Persona.Approve.Approved",
                ),
            ]
            serde = NamespaceAwareSerde(migrations=migrations)

            # Build the bytes a checkpoint would hold: a list of two
            # encoded events under the OLD qualname.
            ext1 = ormsgpack.Ext(
                EXT_NAMESPACE_AWARE_EVENT,
                ormsgpack.packb(
                    ("legacy.x", "Legacy.X", {"note": "first"}), option=_option
                ),
            )
            ext2 = ormsgpack.Ext(
                EXT_NAMESPACE_AWARE_EVENT,
                ormsgpack.packb(
                    ("legacy.x", "Legacy.X", {"note": "second"}), option=_option
                ),
            )
            channel_bytes = ormsgpack.packb([ext1, ext2], option=_option)

            revived = serde.loads_typed(("msgpack", channel_bytes))

            assert isinstance(revived, list)
            assert len(revived) == 2
            assert all(isinstance(e, Persona.Approve.Approved) for e in revived)
            assert [e.note for e in revived] == ["first", "second"]

    # -----------------------------------------------------------------------
    # 2. ScalarReducer holding an Event instance, class renamed
    # -----------------------------------------------------------------------

    def describe_scenario_2_scalar_reducer_holding_an_event():
        def it_migrates_the_single_event_via_the_ext_hook():
            # ScalarReducer's channel value is the LAST event instance, not
            # a list. Renaming the event class must still migrate it.
            from test_serde import Persona

            migrations = [
                Migration.rename(
                    name="rename-legacy-x",
                    old_module="legacy.x",
                    old_qualname="Legacy.X",
                    new_module=Persona.__module__,
                    new_qualname="Persona.Approve.Approved",
                ),
            ]
            serde = NamespaceAwareSerde(migrations=migrations)

            channel_bytes = _legacy_event_bytes(
                "legacy.x", "Legacy.X", {"note": "only"}
            )

            revived = serde.loads_typed(("msgpack", channel_bytes))

            assert isinstance(revived, Persona.Approve.Approved)
            assert revived.note == "only"

    # -----------------------------------------------------------------------
    # 3. dict[str, list[Event]] grouping reducer
    # -----------------------------------------------------------------------

    def describe_scenario_3_grouping_dict_of_event_lists():
        def it_migrates_events_inside_a_dict_value():
            # A reducer that groups events by some key — channel value is
            # ``{"persona": [e1, e2], "story": [e3]}``. Each list contains
            # Event instances. After renames, every event in every list
            # should be migrated.
            from test_serde import Persona, Story

            migrations = [
                Migration.rename(
                    name="rename-persona-x",
                    old_module="legacy.persona",
                    old_qualname="Legacy.PersonaX",
                    new_module=Persona.__module__,
                    new_qualname="Persona.Approve.Approved",
                ),
                Migration.rename(
                    name="rename-story-x",
                    old_module="legacy.story",
                    old_qualname="Legacy.StoryX",
                    new_module=Story.__module__,
                    new_qualname="Story.Approve.Approved",
                ),
            ]
            serde = NamespaceAwareSerde(migrations=migrations)

            persona_ext = ormsgpack.Ext(
                EXT_NAMESPACE_AWARE_EVENT,
                ormsgpack.packb(
                    ("legacy.persona", "Legacy.PersonaX", {"note": "p"}),
                    option=_option,
                ),
            )
            story_ext = ormsgpack.Ext(
                EXT_NAMESPACE_AWARE_EVENT,
                ormsgpack.packb(
                    ("legacy.story", "Legacy.StoryX", {"note": "s"}),
                    option=_option,
                ),
            )
            channel_bytes = ormsgpack.packb(
                {"persona": [persona_ext], "story": [story_ext]},
                option=_option,
            )

            revived = serde.loads_typed(("msgpack", channel_bytes))

            assert set(revived.keys()) == {"persona", "story"}
            assert isinstance(revived["persona"][0], Persona.Approve.Approved)
            assert isinstance(revived["story"][0], Story.Approve.Approved)

    # -----------------------------------------------------------------------
    # 4. Plain dataclass channel value, new field added
    # -----------------------------------------------------------------------

    def describe_scenario_4_plain_dataclass_field_addition():
        # Reducer's channel value is a non-Event dataclass (e.g., a stats
        # object computed from events). Adding a new field is a common
        # refactor — old checkpoints don't have the field.

        def when_new_field_has_a_default():
            def it_revives_using_the_dataclass_default():
                # OldStats had count + last_seen; NewStats added version=1.
                # Old payloads (no version in kwargs) should revive at NewStats
                # via the dataclass default — no AddField migration needed.
                serde = NamespaceAwareSerde()
                old_value = OldStats(count=3, last_seen="2026-01-01")
                _, payload = serde.dumps_typed(old_value)

                # Simulate the refactor: monkey-patch the module so the old
                # qualname now resolves to NewStats. This is the cleanest way
                # to simulate "the class shape changed between releases."
                import sys

                mod = sys.modules[__name__]

                original = mod.OldStats
                try:
                    mod.OldStats = NewStats
                    revived = serde.loads_typed(("msgpack", payload))
                finally:
                    mod.OldStats = original

                # We expect: NewStats instance with version at its default.
                # If this fails, plain-dataclass field additions require
                # explicit migration support (not currently provided).
                assert isinstance(revived, NewStats), (
                    f"Plain-dataclass field addition: revived as "
                    f"{type(revived).__name__} — gap if not NewStats."
                )
                assert revived.count == 3
                assert revived.version == 1  # default applied

        def when_new_field_has_no_default():
            def it_silently_revives_as_None():
                # If the new field is REQUIRED (no default), the upstream
                # LangGraph serde does NOT raise. It emits a warning
                # ("Deserializing unregistered type … will be blocked in a
                # future version") and revives the value as ``None`` —
                # the worst possible outcome for a reducer channel value:
                # silent data loss with no exception to catch.
                #
                # A reducer reading the channel sees ``None`` and may
                # silently rebuild from scratch, mis-classify the thread as
                # fresh, or crash on attribute access several hops later.
                @dataclasses.dataclass(frozen=True)
                class StatsRequiredField:
                    count: int
                    last_seen: str
                    version: int  # REQUIRED — no default

                serde = NamespaceAwareSerde()
                old_value = OldStats(count=3, last_seen="2026-01-01")
                _, payload = serde.dumps_typed(old_value)

                import sys

                mod = sys.modules[__name__]

                original = mod.OldStats
                try:
                    mod.OldStats = StatsRequiredField
                    revived = serde.loads_typed(("msgpack", payload))
                    assert revived is None, (
                        f"Expected None (silent data loss); got "
                        f"{type(revived).__name__}: {revived!r}"
                    )
                finally:
                    mod.OldStats = original

    # -----------------------------------------------------------------------
    # 5. dict[str, int] reducer output — shape changed structurally
    # -----------------------------------------------------------------------

    def describe_scenario_5_dict_shape_changed():
        def it_revives_as_the_old_shape_silently():
            # A reducer used to produce ``{"persona": 3, "story": 5}`` and
            # now produces ``{"persona": {"count": 3, "ts": "..."}, ...}``.
            # The checkpoint holds the OLD shape; reviving gives back a
            # plain dict (because msgpack doesn't know the "type"), and the
            # reducer code expecting the new shape will crash at runtime.
            serde = NamespaceAwareSerde()
            old_value = {"persona": 3, "story": 5}
            _, payload = serde.dumps_typed(old_value)

            revived = serde.loads_typed(("msgpack", payload))

            # Demonstrates the gap: plain dict revives as-is, no library
            # transform layer exists to reshape it. A consumer relying on
            # the new shape would see ``revived["persona"]["count"]`` raise
            # ``TypeError: 'int' object is not subscriptable``.
            assert revived == {"persona": 3, "story": 5}
            with pytest.raises(TypeError, match=r"not subscriptable"):
                _ = revived["persona"]["count"]

    # -----------------------------------------------------------------------
    # 6. Projection function semantics changed — silent stale data
    # -----------------------------------------------------------------------

    def describe_scenario_6_projection_semantics_changed():
        def it_returns_stale_old_semantics_silently():
            # ``fn=lambda e: e.amount`` used to mean "dollars". Refactor
            # changed it to ``fn=lambda e: e.amount * 100`` (cents). Old
            # checkpoints carry dollar values. New code reads them, treats
            # them as cents, displays a number 100x too small.
            #
            # No migration transform can fix this without REPLAYING the
            # events through the new fn — and library-side transforms can't
            # know that "5" means "dollars" vs. "cents". This is exactly
            # the case where replay is the only correct answer.
            serde = NamespaceAwareSerde()
            old_value = 500  # dollars, under old fn
            _, payload = serde.dumps_typed(old_value)

            revived = serde.loads_typed(("msgpack", payload))

            # The serde faithfully restored the value. The bug is semantic:
            # new code reads 500 expecting cents, applies presentation logic,
            # shows $5.00 instead of $500.00. No library hook would catch
            # this. Replay-from-event-log would.
            assert revived == 500

    # -----------------------------------------------------------------------
    # 8. Pydantic model gains a required field
    # -----------------------------------------------------------------------

    def describe_scenario_8_pydantic_required_field_addition():
        def it_revives_an_attribute_less_instance_silently():
            # A reducer's value is a Pydantic model — a common LangGraph
            # pattern. Adding a REQUIRED field is the dangerous case: the
            # upstream serde reconstructs the new class but BYPASSES
            # validation, so the resulting instance passes ``isinstance``
            # but is missing the new attribute entirely. The reducer or
            # handler reading the channel only discovers this when it
            # touches ``revived.version`` — and gets ``AttributeError``
            # from somewhere deep in business logic.
            serde = NamespaceAwareSerde()
            old_value = OldPydanticState(count=5, name="foo")
            _, payload = serde.dumps_typed(old_value)

            import sys

            mod = sys.modules[__name__]

            original = mod.OldPydanticState
            try:
                mod.OldPydanticState = NewPydanticRequiredField
                revived = serde.loads_typed(("msgpack", payload))
            finally:
                mod.OldPydanticState = original

            # The revived object IS a NewPydanticRequiredField, but with
            # the new required field unset.
            assert isinstance(revived, NewPydanticRequiredField)
            assert revived.count == 5
            with pytest.raises(AttributeError):
                _ = revived.version

    # -----------------------------------------------------------------------
    # 7. Replay-from-event-log via the public helper
    # -----------------------------------------------------------------------

    def describe_scenario_7_replay_via_public_helper():
        def it_rebuilds_a_reducer_value_from_the_stored_events():
            # The recovery story for every "Silent fail" case above:
            #
            #   1. Read the event log through the migrating serde — events
            #      come back as their CURRENT class (rename migrations apply).
            #   2. Hand the events to ``replay_reducer(reducer, events)`` —
            #      the reducer's current ``fn`` rebuilds the channel value
            #      from scratch, discarding the stale cache.
            #   3. Write the rebuilt value back to the checkpointer.
            #
            # The helper is a thin wrapper around ``BaseReducer.seed`` so
            # the reducer's namespace filter and event_type predicate apply
            # uniformly. Composes with the existing event-rename machinery
            # because ``events`` was already migrated on read.
            from test_serde import Persona

            from langgraph_events import Reducer
            from langgraph_events.serde import replay_reducer

            migrations = [
                Migration.rename(
                    name="rename-x",
                    old_module="legacy.x",
                    old_qualname="Legacy.X",
                    new_module=Persona.__module__,
                    new_qualname="Persona.Approve.Approved",
                ),
            ]
            serde = NamespaceAwareSerde(migrations=migrations)

            # Step 1: simulate a checkpointed event log of pre-rename events.
            ext1 = ormsgpack.Ext(
                EXT_NAMESPACE_AWARE_EVENT,
                ormsgpack.packb(
                    ("legacy.x", "Legacy.X", {"note": "first"}),
                    option=_option,
                ),
            )
            ext2 = ormsgpack.Ext(
                EXT_NAMESPACE_AWARE_EVENT,
                ormsgpack.packb(
                    ("legacy.x", "Legacy.X", {"note": "second"}),
                    option=_option,
                ),
            )
            event_log_bytes = ormsgpack.packb([ext1, ext2], option=_option)

            # Step 2: read the event log; events return as the CURRENT class.
            events = serde.loads_typed(("msgpack", event_log_bytes))

            # Step 3: replay through a reducer whose projection shape changed
            # since the checkpoint was written. The cache is rebuilt from
            # truth (the events themselves), not patched from the stale cache.
            notes_reducer = Reducer(
                name="notes",
                event_type=Persona.Approve.Approved,
                fn=lambda e: [e.note],
            )

            rebuilt = replay_reducer(notes_reducer, events)

            assert rebuilt == ["first", "second"]
