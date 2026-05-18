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

            assert loaded["version"] == 1
            identities = {(e["module"], e["qualname"]) for e in loaded["events"]}
            # Command outcomes nested inside Place are captured.
            assert (Order.__module__, "Order.Place.Placed") in identities
            assert (Order.__module__, "Order.Place.Rejected") in identities
            # The Command itself is captured.
            assert (Order.__module__, "Order.Place") in identities


def describe_write_baseline_regression_guard():
    # The prose rule "commit the baseline alongside the migration, never
    # after" is now enforced: silently overwriting away an identity the
    # old baseline recorded would make assert_covers/detect_changes blind
    # to a forgotten migration forever.

    def when_an_existing_identity_is_gone_from_the_graph():
        def it_raises_naming_the_dropped_identity(tmp_path: Path):
            import pytest
            from conftest import Order

            from langgraph_events import EventGraph
            from langgraph_events.serde.migrations.detect import (
                BaselineRegressionError,
                write_baseline,
            )

            target = tmp_path / "baseline.json"
            target.write_text(
                json.dumps(
                    {
                        "version": 1,
                        "events": [{"module": "ghost.mod", "qualname": "Ghost.Gone"}],
                    }
                )
            )
            graph = EventGraph([Order.Place])

            with pytest.raises(BaselineRegressionError) as exc:
                write_baseline(graph, target)

            assert ("ghost.mod", "Ghost.Gone") in exc.value.removed
            assert "Ghost.Gone" in str(exc.value)

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
            from conftest import Order

            from langgraph_events import EventGraph
            from langgraph_events.serde.migrations.detect import write_baseline

            target = tmp_path / "baseline.json"
            graph = EventGraph([Order.Place])
            write_baseline(graph, target)
            first = target.read_text()

            write_baseline(graph, target)  # must not raise

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
        def it_overwrites_the_intentional_delete(tmp_path: Path):
            from conftest import Order

            from langgraph_events import EventGraph
            from langgraph_events.serde.migrations.detect import write_baseline

            target = tmp_path / "baseline.json"
            target.write_text(
                json.dumps(
                    {
                        "version": 1,
                        "events": [{"module": "ghost.mod", "qualname": "Ghost.Gone"}],
                    }
                )
            )

            write_baseline(EventGraph([Order.Place]), target, allow_removed=True)

            identities = {
                (e["module"], e["qualname"])
                for e in json.loads(target.read_text())["events"]
            }
            assert ("ghost.mod", "Ghost.Gone") not in identities

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
    # ``detect_changes`` so ``NamespaceAwareSerde.assert_covers`` reuses the
    # exact same error wording on a version bump instead of duplicating it.

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

    def when_version_mismatches():
        def it_raises_naming_the_version(tmp_path: Path):
            import pytest

            from langgraph_events.serde.migrations.detect import _load_baseline

            target = tmp_path / "baseline.json"
            target.write_text(json.dumps({"version": 999, "events": []}))

            with pytest.raises(ValueError, match=r"version 999|baseline version"):
                _load_baseline(target)
