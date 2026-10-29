"""Tests for ``update_baseline`` and ``hold_migration``."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest
from _migration_fixtures import (
    _CLI,
    _MODULE,
    _TRIMMED,
    Derived,
    Renamed,
    Retiring,
    Shop,
    Trimmed,
    TrimmedWide,
    _add,
    _baselined,
    _deleted_transform,
    _entry,
    _fields,
    _ghost,
    _moved_transform,
    _read,
    _target,
)

from langgraph_events import (
    EventGraph,
)
from langgraph_events.serde.migrations import (
    MigrationTarget,
    TargetError,
    check,
    hold_migration,
    update_baseline,
)

if TYPE_CHECKING:
    from pathlib import Path


def describe_update_baseline():
    def when_no_baseline_exists_yet():
        def it_writes_a_version_4_file(tmp_path: Path):
            target = _target(tmp_path, Renamed)

            update_baseline(target)

            assert _read(target.baseline) == {
                "version": 4,
                "events": [
                    {**_entry(_MODULE, "Renamed.Place", ["note"]), "live": ["note"]},
                    {
                        **_entry(_MODULE, "Renamed.Place.Placed", ["note"]),
                        "live": ["note"],
                    },
                ],
                "retired": [],
                "handlers": [{"name": "Renamed.Place"}],
                "migrations": [{"label": f"rename {_MODULE}:Renamed.Placed"}],
            }

    def when_the_serde_reads_an_event_no_handler_produces():
        def it_records_the_event(tmp_path: Path):
            target = _target(tmp_path, Shop, scope=(Retiring,))

            update_baseline(target)

            assert _fields(target.baseline)["Retiring.RetiredApprovalGate"] == [
                "order_id"
            ]

    def when_a_class_has_an_init_false_field():
        def it_records_the_init_fields_only(tmp_path: Path):
            target = _target(tmp_path, Derived)

            update_baseline(target)

            assert _fields(target.baseline)["Derived.Place.Placed"] == ["note"]

    def when_the_live_class_dropped_a_recorded_field():
        def it_keeps_the_field_in_the_record(tmp_path: Path):
            target = _target(tmp_path, Shop)
            update_baseline(target)
            raw = _read(target.baseline)
            raw["events"][1]["fields"] = ["legacy_flag", "note"]
            target.baseline.write_text(json.dumps(raw))

            update_baseline(target)

            assert _fields(target.baseline)["Shop.Place.Placed"] == [
                "legacy_flag",
                "note",
            ]

    def when_an_events_entry_has_empty_fields():
        def it_fills_the_list_from_the_live_class(tmp_path: Path):
            target = _target(tmp_path, Shop)
            target.baseline.write_text(
                json.dumps(
                    {
                        "version": 3,
                        "events": [_entry(_MODULE, "Shop.Place.Placed", [])],
                    }
                )
            )

            update_baseline(target)

            assert _fields(target.baseline)["Shop.Place.Placed"] == ["note"]

    def when_a_recorded_identity_is_gone_from_the_graph():
        def with_a_record_that_predates_v3():
            def it_omits_fields_from_the_retired_entry(tmp_path: Path):
                target = _target(tmp_path, Shop)
                target.baseline.write_text(
                    json.dumps({"version": 1, "events": [_ghost()]})
                )

                update_baseline(target)

                assert _read(target.baseline)["retired"] == [_ghost()]

        def with_a_v3_record():
            def it_keeps_the_recorded_fields_on_the_retired_entry(tmp_path: Path):
                target = _target(tmp_path, Shop)
                target.baseline.write_text(
                    json.dumps({"version": 3, "events": [_ghost(["x"])]})
                )

                update_baseline(target)

                assert _read(target.baseline)["retired"] == [_ghost(["x"])]

    def when_a_later_update_finds_a_retired_entry():
        def it_keeps_the_entry(tmp_path: Path):
            target = _target(tmp_path, Shop)
            update_baseline(target)
            _add(target.baseline, "retired", _ghost(["x"]))

            update_baseline(target)

            assert _read(target.baseline)["retired"] == [_ghost(["x"])]

    def when_a_retired_identity_is_live_again():
        def it_moves_the_identity_back_to_events(tmp_path: Path):
            target = _target(tmp_path, Shop)
            placed = _entry(_MODULE, "Shop.Place.Placed", ["note"])
            target.baseline.write_text(
                json.dumps({"version": 3, "events": [], "retired": [placed]})
            )

            update_baseline(target)

            assert _read(target.baseline)["retired"] == []
            assert {**placed, "live": ["note"]} in _read(target.baseline)["events"]

    def when_the_code_is_unchanged():
        def it_rewrites_the_same_bytes(tmp_path: Path):
            target = _target(tmp_path, Renamed)
            update_baseline(target)
            _add(target.baseline, "retired", _ghost(["x"]))
            update_baseline(target)
            first = target.baseline.read_text()

            update_baseline(target)

            assert target.baseline.read_text() == first

    def when_the_existing_baseline_has_an_unsupported_version():
        def it_raises_naming_the_version(tmp_path: Path):
            target = _target(tmp_path, Shop)
            target.baseline.write_text(json.dumps({"version": 999, "events": []}))

            with pytest.raises(ValueError, match="Unsupported baseline version 999"):
                update_baseline(target)

    def when_a_v3_events_entry_lacks_fields():
        def it_rejects_the_file(tmp_path: Path):
            target = _target(tmp_path, Shop)
            target.baseline.write_text(json.dumps({"version": 3, "events": [_ghost()]}))

            with pytest.raises(ValueError, match=r'Give each entry "fields": \[\]'):
                update_baseline(target)

    def when_an_identity_is_in_events_and_retired():
        def it_rejects_the_file(tmp_path: Path):
            target = _target(tmp_path, Shop)
            target.baseline.write_text(
                json.dumps(
                    {"version": 3, "events": [_ghost([])], "retired": [_ghost([])]}
                )
            )

            with pytest.raises(ValueError, match=r"Ghost\.Gone.*Remove it from"):
                update_baseline(target)

    def when_a_v4_events_entry_lacks_live_fields():
        def it_rejects_the_file(tmp_path: Path):
            target = _target(tmp_path, Shop)
            target.baseline.write_text(
                json.dumps({"version": 4, "events": [_ghost(["x"])]})
            )

            with pytest.raises(ValueError, match=r"no `live`: Ghost\.Gone"):
                update_baseline(target)

    def when_an_entry_lacks_a_key():
        def it_rejects_the_file(tmp_path: Path):
            target = _target(tmp_path, Shop)
            update_baseline(target)
            raw = _read(target.baseline)
            raw["migrations"] = [{"label": _TRIMMED, "hold": {"against": "x"}}]
            target.baseline.write_text(json.dumps(raw))

            with pytest.raises(ValueError, match="is malformed: KeyError 'reason'"):
                update_baseline(target)

    def when_a_migration_label_is_listed_twice():
        def it_rejects_the_file(tmp_path: Path):
            target = _target(tmp_path, Shop)
            update_baseline(target)
            _add(
                target.baseline, "migrations", {"label": _TRIMMED}, {"label": _TRIMMED}
            )

            with pytest.raises(ValueError, match=r"lists the migration .* twice"):
                update_baseline(target)

    def when_a_v4_migration_label_is_malformed():
        def it_rejects_the_file(tmp_path: Path):
            target = _target(tmp_path, Shop)
            update_baseline(target)
            _add(target.baseline, "migrations", {"label": "nonsense"})

            with pytest.raises(ValueError, match="'nonsense' is not a migration label"):
                update_baseline(target)

    def when_the_code_holds_a_held_migration():
        def it_keeps_the_hold(tmp_path: Path):
            target = _target(tmp_path, Renamed)
            update_baseline(target)
            hold = {"reason": "a subgraph needs it", "against": "lge-m1-0"}
            raw = _read(target.baseline)
            raw["migrations"][0]["hold"] = hold
            target.baseline.write_text(json.dumps(raw))

            update_baseline(target)

            assert _read(target.baseline)["migrations"][0]["hold"] == hold

    def when_a_shipped_migration_left_the_code():
        def with_previous():
            def it_resets_the_fields_the_migration_covered(tmp_path: Path):
                target, previous = _deleted_transform(tmp_path)

                update_baseline(target, previous=previous)

                assert _fields(target.baseline)["Shop.Place.Placed"] == ["note"]
                assert _read(target.baseline)["migrations"] == []

            def when_the_previous_release_wrote_a_field_the_code_dropped():
                def it_keeps_the_field(tmp_path: Path):
                    target, previous = _deleted_transform(tmp_path, TrimmedWide)

                    update_baseline(target, previous=previous)

                    assert _fields(target.baseline)["Shop.Place.Placed"] == [
                        "legacy_b",
                        "note",
                    ]

            def it_drops_the_source_of_a_deleted_rename(tmp_path: Path):
                gone = {"label": f"rename {_MODULE}:Shop.Gone"}
                target = _target(tmp_path, Shop)
                update_baseline(target)
                _add(target.baseline, "migrations", gone)
                _add(target.baseline, "retired", _entry(_MODULE, "Shop.Gone", ["x"]))
                previous = tmp_path / "previous.json"
                previous.write_text(target.baseline.read_text())

                update_baseline(target, previous=previous)

                assert _read(target.baseline)["retired"] == []

            def it_writes_the_same_bytes_twice(tmp_path: Path):
                target, previous = _deleted_transform(tmp_path)
                update_baseline(target, previous=previous)
                first = target.baseline.read_text()

                update_baseline(target, previous=previous)

                assert target.baseline.read_text() == first

        def without_previous():
            def it_refuses_naming_the_previous_option(tmp_path: Path):
                target, _ = _deleted_transform(tmp_path)
                committed = target.baseline.read_text()

                with pytest.raises(ValueError) as raised:
                    update_baseline(target)

                assert (
                    f"Run: {_CLI} baseline update <module:attr of the "
                    f"MigrationTarget> --previous <baseline file of the previous "
                    f"release>" in str(raised.value)
                )
                assert target.baseline.read_text() == committed

            def when_the_key_only_follows_a_rename():
                def it_writes_the_baseline(tmp_path: Path):
                    target = _moved_transform(tmp_path)

                    update_baseline(target)

                    assert check(target).findings == ()

        def with_a_previous_baseline_that_predates_v4():
            def it_keeps_the_history(tmp_path: Path):
                target, _ = _deleted_transform(tmp_path)
                previous = tmp_path / "v3.json"
                previous.write_text(json.dumps({"version": 3, "events": []}))

                update_baseline(target, previous=previous)

                assert _fields(target.baseline)["Shop.Place.Placed"] == [
                    "note",
                    "total",
                ]

    def when_the_graph_has_no_namespace_aware_serde():
        def it_raises_a_target_error(tmp_path: Path):
            target = MigrationTarget(
                graph=lambda saver: EventGraph([Shop.Place], checkpointer=saver),
                baseline=tmp_path / "baseline.json",
            )

            with pytest.raises(TargetError, match="NamespaceAwareSerde"):
                update_baseline(target)


def describe_hold_migration():
    def when_the_previous_release_shipped_the_migration():
        def it_records_the_hold_reason(tmp_path: Path):
            target = _baselined(_target(tmp_path, Trimmed))
            previous = tmp_path / "prev.json"
            previous.write_text(target.baseline.read_text())

            hold_migration(target, _TRIMMED, reason="subgraph", previous=previous)

            (record,) = _read(target.baseline)["migrations"]
            assert record["label"] == _TRIMMED
            assert record["hold"]["reason"] == "subgraph"

    def when_the_code_does_not_hold_the_label():
        def it_refuses(tmp_path: Path):
            target = _baselined(_target(tmp_path, Shop))
            previous = tmp_path / "prev.json"
            previous.write_text(target.baseline.read_text())

            with pytest.raises(ValueError, match="A hold needs a live migration"):
                hold_migration(target, _TRIMMED, reason="x", previous=previous)

    def when_the_previous_release_did_not_ship_it():
        def it_refuses(tmp_path: Path):
            target = _baselined(_target(tmp_path, Trimmed))
            previous = tmp_path / "prev.json"
            previous.write_text(
                json.dumps({"version": 4, "events": [], "migrations": []})
            )

            with pytest.raises(ValueError, match="There is nothing to hold"):
                hold_migration(target, _TRIMMED, reason="x", previous=previous)

    def when_the_previous_baseline_predates_v4():
        def it_refuses(tmp_path: Path):
            target = _baselined(_target(tmp_path, Trimmed))
            previous = tmp_path / "prev.json"
            previous.write_text(json.dumps({"version": 3, "events": []}))

            with pytest.raises(ValueError, match="records no migrations"):
                hold_migration(target, _TRIMMED, reason="x", previous=previous)
