"""Tests for ``langgraph_events.serde.migrations.detect`` — baseline diff
and rename suggestion engine.

The detection tool compares the current ``EventGraph.namespaces()`` topology
against a stored baseline of ``(module, qualname)`` identities, classifying
diffs into confident rename suggestions, ambiguous cases, and pure
removals. Used as the building block for project-level pre-commit hooks —
the library never auto-applies suggestions.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path


def _ghost_entry(fields: list[str] | None = None) -> dict[str, object]:
    """A baseline entry for an identity no graph reaches."""
    entry: dict[str, object] = {"module": "ghost.mod", "qualname": "Ghost.Gone"}
    if fields is not None:
        entry["fields"] = fields
    return entry


def _write_v3(
    target: Path,
    *,
    events: list[dict[str, object]] | None = None,
    retired: list[dict[str, object]] | None = None,
) -> None:
    """Write a v3 baseline file with the given ``events`` and ``retired``."""
    target.write_text(
        json.dumps({"version": 3, "events": events or [], "retired": retired or []})
    )


def describe_detect_changes():
    def when_baseline_matches_current():
        def it_reports_no_diff():
            from langgraph_events.serde.migrations.detect import (
                _diff_identities,
            )

            current = {("cadance.persona", "Persona.Approve.Approved")}
            baseline = {("cadance.persona", "Persona.Approve.Approved")}

            report = _diff_identities(current, baseline)

            assert report.added == ()
            assert report.removed == ()
            assert report.confident_renames == ()
            assert report.ambiguous == ()
            assert report.unmatched_removed == ()

    def when_one_event_was_renamed():
        def with_unique_leaf_name():
            def it_emits_a_confident_rename_suggestion():
                from langgraph_events.serde.migrations.detect import (
                    _diff_identities,
                )

                current = {("cadance.persona", "Persona.Persist.Persisted")}
                baseline = {("cadance.persona", "Persona.Persisted")}

                report = _diff_identities(current, baseline)

                assert len(report.confident_renames) == 1
                suggestion = report.confident_renames[0]
                assert suggestion.old_module == "cadance.persona"
                assert suggestion.old_qualname == "Persona.Persisted"
                assert suggestion.new_module == "cadance.persona"
                assert suggestion.new_qualname == "Persona.Persist.Persisted"
                assert report.ambiguous == ()
                assert report.unmatched_removed == ()

    def when_multiple_additions_share_a_leaf_name():
        def it_marks_the_match_as_ambiguous():
            # Two ``Persisted`` classes added (one under Persona, one under
            # Story) and one removed — the leaf-name heuristic alone can't
            # decide which is the rename. Must surface, never silently pick.
            from langgraph_events.serde.migrations.detect import (
                _diff_identities,
            )

            current = {
                ("cadance.persona", "Persona.Persist.Persisted"),
                ("cadance.story", "Story.Persist.Persisted"),
            }
            baseline = {("cadance.persona", "Persona.Persisted")}

            report = _diff_identities(current, baseline)

            assert report.confident_renames == ()
            assert len(report.ambiguous) == 1
            ambig = report.ambiguous[0]
            assert ambig.removed == ("cadance.persona", "Persona.Persisted")
            assert len(ambig.candidates) == 2

    def when_removal_has_no_matching_addition():
        def it_lands_in_unmatched_removed():
            from langgraph_events.serde.migrations.detect import (
                _diff_identities,
            )

            current = set()
            baseline = {("cadance.persona", "Persona.Deleted")}

            report = _diff_identities(current, baseline)

            assert report.confident_renames == ()
            assert report.ambiguous == ()
            assert report.unmatched_removed == (("cadance.persona", "Persona.Deleted"),)


def describe_detect_changes_version_handling():
    # The baseline file format is versioned so a future change to the
    # snapshot shape (richer identity, hash, etc.) can fail loudly when a
    # project still has an old baseline committed. Without an enforced check,
    # the reader silently treats an unknown version as the current one and
    # surfaces misleading diffs.

    def when_baseline_version_is_unsupported():
        def it_raises_naming_the_version(tmp_path: Path):
            import pytest
            from conftest import Order

            from langgraph_events import EventGraph
            from langgraph_events.serde.migrations.detect import detect_changes

            target = tmp_path / "baseline.json"
            target.write_text(
                json.dumps(
                    {
                        "version": 999,
                        "events": [],
                    }
                )
            )
            graph = EventGraph([Order.Place])

            with pytest.raises(ValueError, match=r"version 999|baseline version"):
                detect_changes(graph, target)


def describe_write_baseline():
    def when_called_against_a_graph():
        def it_writes_a_json_file_listing_every_event_identity(tmp_path: Path):
            # Reuse the conftest Order namespace — it has the full taxonomy
            # we care about: a Namespace with a Command that has outcomes,
            # plus a free-standing DomainEvent. Cover all the paths the
            # baseline writer needs to walk.
            from conftest import Order

            from langgraph_events import EventGraph
            from langgraph_events.serde.migrations.detect import write_baseline

            graph = EventGraph([Order.Place])
            target = tmp_path / "baseline.json"

            write_baseline(graph, target)
            loaded = json.loads(target.read_text())

            assert loaded["version"] == 3
            identities = {(e["module"], e["qualname"]) for e in loaded["events"]}
            # Command outcomes nested inside Place are captured.
            assert (Order.__module__, "Order.Place.Placed") in identities
            assert (Order.__module__, "Order.Place.Rejected") in identities
            # The Command itself is captured.
            assert (Order.__module__, "Order.Place") in identities

        def it_records_the_sorted_field_names_of_every_identity(tmp_path: Path):
            # The revive gate synthesizes a payload from these names, so a
            # field the live class later drops is still exercised.
            from conftest import Order

            from langgraph_events import EventGraph
            from langgraph_events.serde.migrations.detect import write_baseline

            target = tmp_path / "baseline.json"

            write_baseline(EventGraph([Order.Place]), target)

            fields = {
                e["qualname"]: e["fields"]
                for e in json.loads(target.read_text())["events"]
            }
            assert fields["Order.Place.Placed"] == ["order_id"]
            assert fields["Order.Place"] == ["customer_id"]

        def it_accepts_a_str_path(tmp_path: Path):
            # docs/event-migrations.md's workflow prints a bare string —
            # a Path-only signature turns that into
            # AttributeError: 'str' object has no attribute 'exists'
            # with no guidance. Every sibling gate (assert_all_baselined_*)
            # already accepts Path | str; write_baseline matches them.
            from conftest import Order

            from langgraph_events import EventGraph
            from langgraph_events.serde.migrations.detect import write_baseline

            graph = EventGraph([Order.Place])
            target = tmp_path / "baseline.json"

            write_baseline(graph, str(target))

            assert target.exists()
            assert json.loads(target.read_text())["version"] == 3


def describe_write_baseline_cumulative_fields():
    # A field that was ever recorded can sit in a checkpoint. A plain
    # rewrite must keep it, or the next unrelated write blinds the revive
    # gate again. Removing a field from the record is a hand edit.

    def when_the_live_class_dropped_a_recorded_field():
        def it_keeps_the_field_in_the_record(tmp_path: Path):
            from conftest import Order

            from langgraph_events import EventGraph
            from langgraph_events.serde.migrations.detect import write_baseline

            target = tmp_path / "baseline.json"
            target.write_text(
                json.dumps(
                    {
                        "version": 3,
                        "events": [
                            {
                                "module": Order.__module__,
                                "qualname": "Order.Place.Placed",
                                "fields": ["legacy_flag", "order_id"],
                            }
                        ],
                    }
                )
            )

            write_baseline(EventGraph([Order.Place]), target)

            fields = {
                e["qualname"]: e["fields"]
                for e in json.loads(target.read_text())["events"]
            }
            assert fields["Order.Place.Placed"] == ["legacy_flag", "order_id"]

    def when_an_events_entry_has_empty_fields():
        def it_fills_the_list_from_the_live_class(tmp_path: Path):
            # The remedy for a hand-built file without ``fields`` is to add
            # ``"fields": []`` and write. The union fills the list.
            from conftest import Order

            from langgraph_events import EventGraph
            from langgraph_events.serde.migrations.detect import write_baseline

            target = tmp_path / "baseline.json"
            placed = {
                "module": Order.__module__,
                "qualname": "Order.Place.Placed",
                "fields": [],
            }
            _write_v3(target, events=[placed])

            write_baseline(EventGraph([Order.Place]), target)

            fields = {
                e["qualname"]: e["fields"]
                for e in json.loads(target.read_text())["events"]
            }
            assert fields["Order.Place.Placed"] == ["order_id"]


def describe_write_baseline_retirement():
    # A write never erases an identity the old baseline recorded. The
    # identity moves to ``retired``, so the coverage gates keep walking it
    # until a hand edit removes the entry.

    def when_an_existing_identity_is_gone_from_the_graph():
        def with_a_pre_v3_record():
            def it_omits_fields_from_the_retired_entry(tmp_path: Path):
                # A v1 record predates field tracking, so the retired entry
                # carries no ``fields`` key: the one permitted degrade.
                from conftest import Order

                from langgraph_events import EventGraph
                from langgraph_events.serde.migrations.detect import write_baseline

                target = tmp_path / "baseline.json"
                target.write_text(
                    json.dumps({"version": 1, "events": [_ghost_entry()]})
                )

                write_baseline(EventGraph([Order.Place]), target)

                loaded = json.loads(target.read_text())
                assert loaded["retired"] == [_ghost_entry()]
                identities = {(e["module"], e["qualname"]) for e in loaded["events"]}
                assert ("ghost.mod", "Ghost.Gone") not in identities

        def with_a_v3_record():
            def it_keeps_the_recorded_fields_on_the_retired_entry(tmp_path: Path):
                from conftest import Order

                from langgraph_events import EventGraph
                from langgraph_events.serde.migrations.detect import write_baseline

                target = tmp_path / "baseline.json"
                _write_v3(target, events=[_ghost_entry(["x"])])

                write_baseline(EventGraph([Order.Place]), target)

                loaded = json.loads(target.read_text())
                assert loaded["retired"] == [_ghost_entry(["x"])]

    def when_a_later_write_finds_a_retired_entry():
        def it_keeps_the_entry(tmp_path: Path):
            from conftest import Order

            from langgraph_events import EventGraph
            from langgraph_events.serde.migrations.detect import write_baseline

            target = tmp_path / "baseline.json"
            _write_v3(target, retired=[_ghost_entry(["x"])])

            write_baseline(EventGraph([Order.Place]), target)

            assert json.loads(target.read_text())["retired"] == [_ghost_entry(["x"])]

    def when_a_retired_identity_is_live_again():
        def it_moves_the_identity_back_to_events(tmp_path: Path):
            from conftest import Order

            from langgraph_events import EventGraph
            from langgraph_events.serde.migrations.detect import write_baseline

            target = tmp_path / "baseline.json"
            placed = {
                "module": Order.__module__,
                "qualname": "Order.Place.Placed",
                "fields": ["order_id"],
            }
            _write_v3(target, retired=[placed])

            write_baseline(EventGraph([Order.Place]), target)

            loaded = json.loads(target.read_text())
            assert loaded["retired"] == []
            assert placed in loaded["events"]

    def when_no_baseline_exists_yet():
        def it_writes_the_first_baseline(tmp_path: Path):
            from conftest import Order

            from langgraph_events import EventGraph
            from langgraph_events.serde.migrations.detect import write_baseline

            target = tmp_path / "baseline.json"

            write_baseline(EventGraph([Order.Place]), target)

            assert target.exists()

    def when_the_topology_is_unchanged():
        def it_rewrites_idempotently(tmp_path: Path):
            # The seed file carries a retired identity, so the second write
            # must reproduce ``events`` and ``retired`` both.
            from conftest import Order

            from langgraph_events import EventGraph
            from langgraph_events.serde.migrations.detect import write_baseline

            target = tmp_path / "baseline.json"
            _write_v3(target, events=[_ghost_entry(["x"])])
            graph = EventGraph([Order.Place])
            write_baseline(graph, target)
            first = target.read_text()

            write_baseline(graph, target)

            assert target.read_text() == first

    def when_only_new_identities_appear():
        def it_writes_because_additions_never_erase_coverage(tmp_path: Path):
            from conftest import Order

            from langgraph_events import EventGraph
            from langgraph_events.serde.migrations.detect import write_baseline

            target = tmp_path / "baseline.json"
            # Existing baseline is a strict subset of current topology.
            target.write_text(
                json.dumps(
                    {
                        "version": 1,
                        "events": [
                            {"module": Order.__module__, "qualname": "Order.Place"}
                        ],
                    }
                )
            )

            write_baseline(EventGraph([Order.Place]), target)

            identities = {
                (e["module"], e["qualname"])
                for e in json.loads(target.read_text())["events"]
            }
            assert (Order.__module__, "Order.Place.Placed") in identities

    def when_allow_removed_is_set():
        def it_warns_that_the_flag_does_nothing(tmp_path: Path):
            # The flag once erased the dropped identity. A write now retires
            # it whatever the flag says, and the flag will be removed.
            import pytest
            from conftest import Order

            from langgraph_events import EventGraph
            from langgraph_events.serde.migrations.detect import write_baseline

            target = tmp_path / "baseline.json"
            _write_v3(target, events=[_ghost_entry(["x"])])
            graph = EventGraph([Order.Place])
            write_baseline(graph, target)
            plain = target.read_text()
            _write_v3(target, events=[_ghost_entry(["x"])])

            with pytest.warns(DeprecationWarning, match="allow_removed"):
                write_baseline(graph, target, allow_removed=True)

            assert target.read_text() == plain

    def when_the_existing_baseline_has_an_unsupported_version():
        def it_still_raises_the_version_error(tmp_path: Path):
            import pytest
            from conftest import Order

            from langgraph_events import EventGraph
            from langgraph_events.serde.migrations.detect import write_baseline

            target = tmp_path / "baseline.json"
            target.write_text(json.dumps({"version": 999, "events": []}))

            with pytest.raises(ValueError, match="Unsupported baseline version"):
                write_baseline(EventGraph([Order.Place]), target)


def describe_load_baseline():
    # ``_load_baseline`` is the shared parse+version-check extracted from
    # ``detect_changes`` so ``assert_all_baselined_cover`` reuses the exact
    # same error wording on a version bump instead of duplicating it.

    def when_version_matches():
        def it_returns_the_identity_set(tmp_path: Path):
            from langgraph_events.serde.migrations.detect import _load_baseline

            target = tmp_path / "baseline.json"
            target.write_text(
                json.dumps(
                    {
                        "version": 1,
                        "events": [
                            {"module": "cadance.persona", "qualname": "P.Old"},
                            {"module": "cadance.story", "qualname": "S.Old"},
                        ],
                    }
                )
            )

            identities = _load_baseline(target)

            assert identities == {
                ("cadance.persona", "P.Old"),
                ("cadance.story", "S.Old"),
            }

    def when_the_file_predates_v3():
        def it_loads_no_fields_and_no_retired_identities(tmp_path: Path):
            # v1 and v2 recorded neither. The revive gate then synthesizes
            # required placeholders only: the documented degrade.
            from langgraph_events.serde.migrations.detect import (
                _load_baseline_fields,
                _load_baseline_retired,
            )

            target = tmp_path / "baseline.json"
            target.write_text(json.dumps({"version": 2, "events": [_ghost_entry()]}))

            assert _load_baseline_fields(target) == {("ghost.mod", "Ghost.Gone"): None}
            assert _load_baseline_retired(target) == {}

    def when_a_v3_events_entry_lacks_fields():
        def it_rejects_the_file(tmp_path: Path):
            # ``fields`` is mandatory on v3. A hand-built file that omits it
            # would silently degrade the revive gate.
            import pytest

            from langgraph_events.serde.migrations.detect import _load_baseline

            target = tmp_path / "baseline.json"
            _write_v3(target, events=[_ghost_entry()])

            with pytest.raises(ValueError, match=r'"fields": \[\].*write_baseline'):
                _load_baseline(target)

    def when_an_identity_is_in_events_and_retired():
        def it_rejects_the_file(tmp_path: Path):
            # The writer never produces an overlap. One means a hand edit
            # went wrong, and the gates must not guess which list wins.
            import pytest

            from langgraph_events.serde.migrations.detect import _load_baseline

            target = tmp_path / "baseline.json"
            _write_v3(target, events=[_ghost_entry([])], retired=[_ghost_entry([])])

            with pytest.raises(
                ValueError, match=r"Ghost\.Gone.*Remove it from `retired`"
            ):
                _load_baseline(target)

    def when_version_mismatches():
        def it_raises_naming_the_version(tmp_path: Path):
            import pytest

            from langgraph_events.serde.migrations.detect import _load_baseline

            target = tmp_path / "baseline.json"
            target.write_text(json.dumps({"version": 999, "events": []}))

            with pytest.raises(ValueError, match=r"version 999|baseline version"):
                _load_baseline(target)
